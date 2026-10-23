use crate::encode::CSR_OUT_C0;
use crate::state::{ShaderState, Stage};
use crate::isel::borg_isel;
use crate::opt::classify_uniform;
use std::collections::HashMap;
use crate::{collect_cf_marks, decompose_vertex_offset, packed_layout, CfMark, resolve_vm, BorgInstr, Loc, NO_DST, PUSH_WINDOW_WORDS};
use compiler::bindings::*;
use compiler::nir::AsDef;
use crate::opcodes::Op;
use crate::Mnem;

impl ShaderState {
    /// The texture and sampler descriptor indices; a vertex shader's table is its attributes', so its
    /// textures are borg_core's BORG_VTEX_SLOT copies.
    fn tex_slot(&self, tex: &nir_tex_instr) -> u32 {
        let t = tex.texture_index + if self.stage == Stage::Vertex { 128 + 43 } else { 0 };
        t | tex.sampler_index << 8
    }

    /// Run-time array elements: the texture and sampler offsets added to the control word.
    unsafe fn tex_dyn_slot(&mut self, tex: &nir_tex_instr, mut ctl: u32) -> u32 {
        for (ty, scale) in [(nir_tex_src_texture_offset, 1u32), (nir_tex_src_sampler_offset, 256)] {
            let Some(s) = tex.srcs_as_slice().iter().find(|s| s.src_type == ty) else { continue };
            let (mut v, mut c) = resolve_vm(&self.sel.vec_map, s.src.as_def().index, 0);
            if scale != 1 {
                let (k, m) = (self.konst(scale), self.sel.fresh());
                self.sel.prog.push(BorgInstr { mnem: Op::Imul.into(), dst: m, srcs: vec![v, k], swz: vec![c, 0], imm: 0 });
                (v, c) = (m, 0);
            }
            let sum = self.sel.fresh();
            self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: sum, srcs: vec![ctl, v], swz: vec![0, c], imm: 0 });
            ctl = sum;
        }
        ctl
    }

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
        let ld = srcs.iter().find(|s| s.src_type == nir_tex_src_lod).map(|s| s.src.as_def().index);
        let wz = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        self.sel.loc.insert(wz, Loc::Fixed(30));
        self.frag.per_pixel_fixed.insert(wz);
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
        self.sel.prog.push(BorgInstr { mnem: Op::Texa.into(), dst: NO_DST, srcs: vec![wz, lv, wz], swz: vec![0, lc, 0], imm: 0 });
        let ctl = self.konst(3u32 << 16 | self.tex_slot(tex));
        let ctl = self.tex_dyn_slot(tex, ctl);
        let tex_v = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: Op::Tex.into(), dst: tex_v, srcs: vec![wz, wz, ctl], swz: vec![0, 0, 0], imm: 0 });
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
        if tex.op == nir_texop_txs && self.stage == Stage::Fragment {
            self.tex_size(tex);
            return;
        }
        let is_txl = tex.op == nir_texop_txl && matches!(self.stage, Stage::Fragment | Stage::Compute | Stage::Vertex);
        let is_txb = tex.op == nir_texop_txb && self.stage == Stage::Fragment;
        let is_tg4 = tex.op == nir_texop_tg4 && self.stage == Stage::Fragment;
        // A multisampled image is an array of its sample planes: layer = array layer * 4 + sample.
        let is_ms = tex.op == nir_texop_txf_ms;
        if tex.op != nir_texop_tex && tex.op != nir_texop_txf && !is_txl && !is_txb && !is_tg4 && !is_ms {
            self.refuse(format!("texture op {}", tex.op));
            return;
        }
        let coord = srcs
            .iter()
            .find(|s| s.src_type == nir_tex_src_coord)
            .map(|s| s.src.as_def().index);
        if let Some(cd) = coord {
            let (ud, uc) = resolve_vm(&self.sel.vec_map, cd, 0);
            let is_fetch = tex.op == nir_texop_txf || is_ms;
            // A 1D coordinate has no v: a fetch reuses u, a sample takes the centre of the one row.
            let (vd, vc) = match (is_fetch, tex.coord_components) {
                (true, 1) => (ud, uc),
                (false, 1) => (self.konst(0.5f32.to_bits()), 0),
                _ => resolve_vm(&self.sel.vec_map, cd, 1),
            };
            let is_cube = tex.sampler_dim == GLSL_SAMPLER_DIM_CUBE;
            let has_w = !is_fetch && tex.coord_components >= 3 && !is_cube;
            let has_bias = (is_txb || tex.op == nir_texop_tex) && !is_fetch && srcs.iter().any(|s| s.src_type == nir_tex_src_bias);
            // texelFetch's level and its array layer (or 3D slice) reach the unit through TEXA.
            let lod_src = srcs.iter().find(|s| s.src_type == nir_tex_src_lod).map(|s| s.src.as_def().index);
            let fetch_lod = is_fetch && !is_ms && lod_src.is_some_and(|l| self.scalar_const(l) != Some(0));
            let fetch_w = is_fetch && !is_ms && tex.coord_components == 3;
            let dref = srcs.iter().find(|s| s.src_type == nir_tex_src_comparator && !is_fetch).map(|s| s.src.as_def().index);
            if is_txl || has_w || has_bias || is_ms || fetch_lod || fetch_w || dref.is_some() {
                let ld = srcs.iter().find(|s| s.src_type == nir_tex_src_lod || s.src_type == nir_tex_src_bias).map(|s| s.src.as_def().index).unwrap_or(cd);
                if let Some(&bits) = self.consts.get(&ld).filter(|_| is_txl || has_bias) {
                    if !self.sel.vec_map.contains_key(&ld) {
                        let v = self.konst(bits);
                        self.sel.vec_map.insert(ld, vec![(v, 0u8)]);
                    }
                }
                let wz = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.loc.insert(wz, Loc::Fixed(30));
                self.frag.per_pixel_fixed.insert(wz);
                let (lv, lc) = if is_txl || has_bias {
                    resolve_vm(&self.sel.vec_map, ld, 0)
                } else if let Some(l) = lod_src.filter(|_| fetch_lod) {
                    self.scalar_of(l, 0).unwrap_or_else(|bits| (self.konst(bits), 0))
                } else if is_fetch {
                    (self.konst(0), 0)
                } else {
                    (wz, 0u8)
                };
                let (wv, wc) = if has_w {
                    resolve_vm(&self.sel.vec_map, cd, 2)
                } else if fetch_w {
                    self.scalar_of(cd, 2).unwrap_or_else(|bits| (self.konst(bits), 0))
                } else if let Some(m) = srcs.iter().find(|s| is_ms && s.src_type == nir_tex_src_ms_index) {
                    let m = m.src.as_def().index;
                    let layer = (tex.coord_components == 3).then(|| self.scalar_of(cd, 2));
                    match (layer, self.scalar_const(m)) {
                        (Some(Err(l)), Some(s)) => (self.konst(l * 4 + s), 0),
                        (Some(l), _) => {
                            let l = l.unwrap_or_else(|bits| (self.konst(bits), 0));
                            let s = self.scalar_of(m, 0).unwrap_or_else(|bits| (self.konst(bits), 0));
                            let (four, t, ls) = (self.konst(4), self.sel.fresh(), self.sel.fresh());
                            self.sel.prog.push(BorgInstr { mnem: Op::Imul.into(), dst: t, srcs: vec![l.0, four], swz: vec![l.1, 0], imm: 0 });
                            self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: ls, srcs: vec![t, s.0], swz: vec![0, s.1], imm: 0 });
                            (ls, 0)
                        }
                        (None, _) => self.scalar_of(m, 0).unwrap_or_else(|bits| (self.konst(bits), 0)),
                    }
                } else {
                    (wz, 0u8)
                };
                let (dv, dc) = match dref {
                    Some(d) => self.scalar_of(d, 0).unwrap_or_else(|bits| (self.konst(bits), 0)),
                    None => (wz, 0u8),
                };
                self.sel.prog.push(BorgInstr { mnem: Op::Texa.into(), dst: NO_DST, srcs: vec![wv, lv, dv], swz: vec![wc, lc, dc], imm: 0 });
            }
            // Constant texel offsets: u in bits 26:23, v in 30:27, a 3D image's w in 31 and 20:18.
            let off = match srcs.iter().find(|s| s.src_type == nir_tex_src_offset) {
                None => 0,
                Some(s) => {
                    let d = s.src.as_def();
                    let Some(lc) = d.parent_instr().as_load_const() else {
                        self.refuse("a texel offset that is not constant");
                        return;
                    };
                    (0..d.num_components as usize).fold(0u32, |b, c| {
                        let o = unsafe { lc.values()[c].u32_ } & 15;
                        b | match c { 0 => o << 23, 1 => o << 27, _ => (o & 8) << 28 | (o & 7) << 18 }
                    })
                }
            };
            let off = off | self.tex_slot(tex) | (dref.is_some() as u32) << 20;
            let ctl = if is_tg4 {
                // Operation 2 (gather) of the component in bits 19:18.
                let bits = 2u32 << 16 | (tex.component() & 3) << 18;
                self.konst(bits | off)
            } else if has_bias {
                // LOD mode 1: implicit LOD plus TEXA's bias.
                self.konst(1u32 << 21 | off)
            } else if is_fetch || is_txl {
                self.konst(if is_txl { 2u32 << 21 } else { 1u32 << 16 } | off)
            } else if off != 0 {
                self.konst(off)
            } else {
                match self.zero_ctl {
                Some(v) => v,
                None => {
                    let c = self.sel.next_vreg;
                    let neg = self.sel.next_vreg + 1;
                    let z = self.sel.next_vreg + 2;
                    self.sel.next_vreg += 3;
                    self.sel.loc.insert(c, Loc::Fixed(30));
                    self.frag.per_pixel_fixed.insert(c);
                    self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: neg, srcs: vec![c], swz: vec![0], imm: 0 });
                    self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: z, srcs: vec![neg], swz: vec![0], imm: 0 });
                    self.zero_ctl = Some(z);
                    z
                }
                }
            };
            let ctl = self.tex_dyn_slot(tex, ctl);
            let tex_v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr {
                mnem: Op::Tex.into(), dst: tex_v, srcs: vec![ud, vd, ctl], swz: vec![uc, vc, 0],
                imm: 0,
            });
            self.tex_dsts.insert(tex_v);
            self.sel.vec_map.insert(
                tex.def.index,
                vec![(tex_v, 0), (tex_v, 1), (tex_v, 2), (tex_v, 3)],
            );
        }
    }

    /// A def's constant bits, if it is a scalar constant.
    fn scalar_const(&self, d: u32) -> Option<u32> {
        self.consts.get(&d).copied().filter(|_| !self.sel.vec_map.contains_key(&d))
    }

    /// Component `c` of `d` as (register, component), or the constant's bits.
    fn scalar_of(&self, d: u32, c: u8) -> Result<(u32, u8), u32> {
        let (v, vc) = resolve_vm(&self.sel.vec_map, d, c);
        match self.scalar_const(v) {
            Some(bits) => Err(bits),
            None => Ok((v, vc)),
        }
    }

    pub unsafe fn alu(&mut self, alu: &nir_alu_instr) {
        if self.stage == Stage::Compute && self.fold_const(alu) {
            return;
        }
        if self.stage == Stage::Compute || (matches!(alu.op, nir_op_fadd | nir_op_fmul | nir_op_ffma
            | nir_op_flt | nir_op_fge | nir_op_feq | nir_op_fneu | nir_op_bcsel | nir_op_fmin | nir_op_fmax
            | nir_op_slt | nir_op_sge | nir_op_seq | nir_op_sne
            | nir_op_ieq | nir_op_ine | nir_op_ilt | nir_op_ige | nir_op_ult | nir_op_uge)
            || (self.stage == Stage::Fragment && matches!(alu.op, nir_op_iadd | nir_op_imul | nir_op_ishl
                | nir_op_ishr | nir_op_ushr | nir_op_isub | nir_op_iand | nir_op_ior
                | nir_op_ixor))
            // Vertex bit operations are never address math (gl_VertexIndex & 2, say).
            || (self.stage == Stage::Vertex && matches!(alu.op, nir_op_ishr | nir_op_ushr | nir_op_iand | nir_op_ior
                | nir_op_ixor))
            || (self.stage == Stage::Vertex && self.flags.fetching() && matches!(alu.op, nir_op_iadd | nir_op_imul
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
                let zero = self.sel.pair(Op::Isub, x, x);
                self.sel.prog.push(BorgInstr { mnem: Op::Isub.into(), dst: alu.def.index, srcs: vec![zero.0, x.0], swz: vec![0, x.1], imm: 0 });
            }
            // A condition is 1.0 or 0.0, so the integer 0 or 1 is its truncation.
            nir_op_b2i32 => {
                let s = &alu.srcs_as_slice()[0];
                let (v, c) = resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                self.sel.prog.push(BorgInstr { mnem: Op::F2i.into(), dst: alu.def.index, srcs: vec![v], swz: vec![c], imm: 0 });
            }
            nir_op_flt | nir_op_fge | nir_op_fneu | nir_op_feq | nir_op_slt | nir_op_sge | nir_op_sne | nir_op_seq
                if self.stage == Stage::Compute => self.exact_compare(alu),
            // slt and friends are the same compares with a float result, which a condition is here.
            nir_op_flt | nir_op_fge | nir_op_fneu | nir_op_feq | nir_op_slt | nir_op_sge | nir_op_sne | nir_op_seq => self.float_compare(alu),
            // fmax(0, x) → x · FSTEP(x)  (FSTEP(x)=1 if x>0 else 0).
            nir_op_fmax | nir_op_fmin if self.stage == Stage::Compute => self.exact_minmax(alu),
            nir_op_bcsel if self.stage == Stage::Compute => self.exact_bcsel(alu),
            nir_op_fmax | nir_op_fmin => self.float_minmax(alu),
            nir_op_ieq | nir_op_ine | nir_op_ilt | nir_op_ige | nir_op_ult | nir_op_uge => self.int_compare(alu),
            nir_op_bcsel => self.bcsel(alu),
            // max/min of integers: a + (a < b) * (b - a) for max, b + (a < b) * (a - b) for min.
            nir_op_umax | nir_op_umin | nir_op_imax | nir_op_imin => self.int_minmax(alu),
            nir_op_bit_count | nir_op_ufind_msb | nir_op_ifind_msb | nir_op_find_lsb | nir_op_umul_high
            | nir_op_imul_high => self.bit_op(alu),
            // |x|: clear the sign bit.
            nir_op_fabs => {
                let sl = &alu.srcs_as_slice()[0];
                let (x, xc) = resolve_vm(&self.sel.vec_map, sl.src.as_def().index, sl.swizzle[0]);
                let m = self.konst(0x7fff_ffff);
                self.sel.prog.push(BorgInstr { mnem: Op::Iand.into(), dst: alu.def.index, srcs: vec![x, m], swz: vec![xc, 0], imm: 0 });
            }
            // A condition is 1.0 or 0.0: its inverse is 1.0 - c.
            nir_op_inot if alu.def.bit_size == 1 => {
                let sl = &alu.srcs_as_slice()[0];
                let (x, xc) = resolve_vm(&self.sel.vec_map, sl.src.as_def().index, sl.swizzle[0]);
                let one = self.konst(1.0f32.to_bits());
                let n = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: n, srcs: vec![x], swz: vec![xc], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: alu.def.index, srcs: vec![n, one], swz: vec![0, 0], imm: 0 });
            }
            nir_op_inot => {
                let sl = &alu.srcs_as_slice()[0];
                let x = resolve_vm(&self.sel.vec_map, sl.src.as_def().index, sl.swizzle[0]);
                let m = self.konst(u32::MAX);
                self.sel.prog.push(BorgInstr { mnem: Op::Ixor.into(), dst: alu.def.index, srcs: vec![x.0, m], swz: vec![x.1, 0], imm: 0 });
            }
            nir_op_u2f16 | nir_op_u2f32 => {
                let s = &alu.srcs_as_slice()[0];
                let (x, xc) = resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                let c1 = self.konst(1);
                let k = self.konst(2.0f32.to_bits());
                let v = self.sel.next_vreg;
                self.sel.next_vreg += 5;
                self.sel.prog.push(BorgInstr { mnem: Op::Isrl.into(), dst: v, srcs: vec![x, c1], swz: vec![xc, 0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Iand.into(), dst: v + 2, srcs: vec![x, c1], swz: vec![xc, 0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::I2f.into(), dst: v + 3, srcs: vec![v], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::I2f.into(), dst: v + 4, srcs: vec![v + 2], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fmadd.into(), dst: alu.def.index, srcs: vec![v + 3, k, v + 4], swz: vec![0, 0, 0], imm: 0 });
            }
            _ => {
                if let Some(mnem) = borg_isel(alu.op) {
                    if mnem != Mnem::Mov {
                        let (mut srcs, mut swz, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
                        for s in alu.srcs_as_slice() {
                            let (d, c) =
                                resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                            pairs.push((d, c));
                            // A descriptor's offset (.z) is 0: the driver places each binding at its slot's start.
                            let (d, c) = if c == 2 && self.vert.descriptor_defs.contains(&d) { (self.konst(0), 0) } else { (d, c) };
                            srcs.push(d);
                            swz.push(c);
                        }
                        self.prod.insert(alu.def.index, (alu.op, pairs));
                        self.sel.prog.push(BorgInstr { mnem, dst: alu.def.index, srcs, swz, imm: 0 });
                    }
                }
                else if !self.skipped.contains(&alu.info().name()) {
                    self.skipped.push(alu.info().name());
                }
            }
        }
    }

    /// A float comparison as 0.0 or 1.0, from FSTEP of the difference.
    unsafe fn float_compare(&mut self, alu: &nir_alu_instr) {
        let sl = alu.srcs_as_slice();
        let a = self.operand(sl[0].src.as_def().index, sl[0].swizzle[0]);
        let b = self.operand(sl[1].src.as_def().index, sl[1].swizzle[0]);

        // diff(hi, lo) = hi - lo, as a fresh vreg.

        match alu.op {
            // a < b  ->  FSTEP(b - a)
            nir_op_flt | nir_op_slt => {
                let d = self.sel.fsub(b, a);
                self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: alu.def.index, srcs: vec![d], swz: vec![0], imm: 0 });
            }
            nir_op_fge | nir_op_sge => {
                let d = self.sel.fsub(b, a);
                let st = self.sel.next_vreg; self.sel.next_vreg += 1;
                let nst = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: st, srcs: vec![d], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: nst, srcs: vec![st], swz: vec![0], imm: 0 });
                let one = self.one();
                self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: alu.def.index, srcs: vec![one, nst], swz: vec![0, 0], imm: 0 });
            }
            nir_op_fneu | nir_op_sne => {
                let dab = self.sel.fsub(a, b);
                let dba = self.sel.fsub(b, a);
                let s1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                let s2 = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: s1, srcs: vec![dab], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: s2, srcs: vec![dba], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: alu.def.index, srcs: vec![s1, s2], swz: vec![0, 0], imm: 0 });
            }
            // a == b  ->  1 - (a != b)
            _ => {
                let dab = self.sel.fsub(a, b);
                let dba = self.sel.fsub(b, a);
                let s1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                let s2 = self.sel.next_vreg; self.sel.next_vreg += 1;
                let ne = self.sel.next_vreg; self.sel.next_vreg += 1;
                let nne = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: s1, srcs: vec![dab], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: s2, srcs: vec![dba], swz: vec![0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: ne, srcs: vec![s1, s2], swz: vec![0, 0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: nne, srcs: vec![ne], swz: vec![0], imm: 0 });
                let one = self.one();
                self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: alu.def.index, srcs: vec![one, nne], swz: vec![0, 0], imm: 0 });
            }
        }
    }

    /// Float min and max as b + (a >= b) * (a - b), with FSTEP for the comparison.
    unsafe fn float_minmax(&mut self, alu: &nir_alu_instr) {
        let s = alu.srcs_as_slice();
        let a_zero = s[0].comp_as_uint(0) == Some(0);
        let b_zero = s[1].comp_as_uint(0) == Some(0);
        if alu.op == nir_op_fmax && (a_zero || b_zero) {
            let xs = if a_zero { &s[1] } else { &s[0] };
            let x = resolve_vm(&self.sel.vec_map, xs.src.as_def().index, xs.swizzle[0]);
            let vstep = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: vstep, srcs: vec![x.0], swz: vec![x.1], imm: 0 });
            self.sel.prog.push(BorgInstr { mnem: Op::Fmul.into(), dst: alu.def.index, srcs: vec![x.0, vstep], swz: vec![x.1, 0], imm: 0 });
        } else {
            // max(a, b) = b + (a >= b) * (a - b); min(a, b) = a + (a >= b) * (b - a)
            let a = self.operand(s[0].src.as_def().index, s[0].swizzle[0]);
            let b = self.operand(s[1].src.as_def().index, s[1].swizzle[0]);
            let one = self.one();
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 8;
            self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: v, srcs: vec![a.0], swz: vec![a.1], imm: 0 });          // -a
            self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: v + 1, srcs: vec![b.0, v], swz: vec![b.1, 0], imm: 0 }); // b - a
            self.sel.prog.push(BorgInstr { mnem: Op::Fstep.into(), dst: v + 2, srcs: vec![v + 1], swz: vec![0], imm: 0 });      // a < b
            self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: v + 3, srcs: vec![v + 2], swz: vec![0], imm: 0 });
            self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: v + 4, srcs: vec![one, v + 3], swz: vec![0, 0], imm: 0 }); // a >= b
            if alu.op == nir_op_fmax {
                self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: v + 5, srcs: vec![v + 1], swz: vec![0], imm: 0 });    // a - b
                self.sel.prog.push(BorgInstr { mnem: Op::Fmadd.into(), dst: alu.def.index, srcs: vec![v + 4, v + 5, b.0], swz: vec![0, 0, b.1], imm: 0 });
            } else {
                self.sel.prog.push(BorgInstr { mnem: Op::Fmadd.into(), dst: alu.def.index, srcs: vec![v + 4, v + 1, a.0], swz: vec![0, 0, a.1], imm: 0 });
            }
        }
    }

    /// An integer comparison, from ISLT, ISLTU and ISEQ.
    unsafe fn int_compare(&mut self, alu: &nir_alu_instr) {
        let sl = alu.srcs_as_slice();
        let a = self.operand(sl[0].src.as_def().index, sl[0].swizzle[0]);
        let b = self.operand(sl[1].src.as_def().index, sl[1].swizzle[0]);
        let lt = if matches!(alu.op, nir_op_ult | nir_op_uge) { Op::Isltu } else { Op::Islt };
        let bits = match alu.op {
            nir_op_ieq => self.sel.pair(Op::Iseq, a, b),
            nir_op_ilt | nir_op_ult => self.sel.pair(lt, a, b),
            nir_op_ine => {
                let l = self.sel.pair(Op::Islt, a, b);
                let g = self.sel.pair(Op::Islt, b, a);
                self.sel.pair(Op::Iadd, l, g)
            }
            _ => {
                let g = self.sel.pair(lt, b, a);
                let e = self.sel.pair(Op::Iseq, a, b);
                self.sel.pair(Op::Iadd, g, e)
            }
        };
        self.sel.prog.push(BorgInstr { mnem: Op::I2f.into(), dst: alu.def.index, srcs: vec![bits.0], swz: vec![bits.1], imm: 0 });
    }

    /// A select as c * (a - b) + b; the select of the sRGB curve becomes FSRGB.
    unsafe fn bcsel(&mut self, alu: &nir_alu_instr) {
        let s = alu.srcs_as_slice();
        let then_def = s[1].src.as_def().index;
        let mut x: Option<(u32, u8)> = None;
        if let Some((op, tsrcs)) = self.prod.get(&then_def) {
            if *op == nir_op_fmul && tsrcs.len() == 2 {
                // 12.92 is a NIR constant, or a window word once an arithmetic reader gave it one.
                let is12 = |d: u32| {
                    let bits = self.consts.get(&d).copied().or_else(|| match self.sel.loc.get(&d) {
                        Some(Loc::Uniform(_)) => self.bits_of(d),
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
            self.sel.prog.push(BorgInstr { mnem: Op::Fsrgb.into(), dst: v, srcs: vec![x.0], swz: vec![x.1], imm: 0 });
            self.sel.vec_map.insert(alu.def.index, vec![(v, 0)]);
        } else {
            let n = alu.def.num_components as usize;
            let mut comps = Vec::new();
            for k in 0..n {
                let c = self.operand(s[0].src.as_def().index, s[0].swizzle[k]);
                let t = self.operand(s[1].src.as_def().index, s[1].swizzle[k]);
                let e = self.operand(s[2].src.as_def().index, s[2].swizzle[k]);
                let ne = self.sel.next_vreg;
                let d = self.sel.next_vreg + 1;
                let r = self.sel.next_vreg + 2;
                self.sel.next_vreg += 3;
                self.sel.prog.push(BorgInstr { mnem: Op::Fneg.into(), dst: ne, srcs: vec![e.0], swz: vec![e.1], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fadd.into(), dst: d, srcs: vec![t.0, ne], swz: vec![t.1, 0], imm: 0 });
                self.sel.prog.push(BorgInstr { mnem: Op::Fmadd.into(), dst: r, srcs: vec![c.0, d, e.0], swz: vec![c.1, 0, e.1], imm: 0 });
                comps.push((r, 0u8));
            }
            self.sel.vec_map.insert(alu.def.index, comps);
        }
    }

    /// A value a RISC-V instruction can read: a window word is copied to a register.
    pub(crate) fn rv_reg(&mut self, v: (u32, u8)) -> (u32, u8) {
        let (code, r) = self.window_copy(v);
        self.sel.prog.extend(code);
        r
    }

    /// The bits of a window word in a register: + 0 by integer IADD (FMOV is an FMA and would
    /// change a NaN-looking integer); anything else is returned as it is.
    pub(crate) fn window_copy(&mut self, v: (u32, u8)) -> (Vec<BorgInstr>, (u32, u8)) {
        if !matches!(self.sel.loc.get(&v.0), Some(Loc::Uniform(_))) {
            return (Vec::new(), v);
        }
        // FMOV keeps every other value: the FMA changes only NaNs and -0.
        if self.bits_of(v.0).is_some_and(|b| b != 0x8000_0000 && (b >> 23) & 0xFF != 0xFF) && v.1 == 0 {
            let t = self.sel.fresh();
            return (vec![BorgInstr { mnem: Op::Fmov.into(), dst: t, srcs: vec![v.0], swz: vec![v.1], imm: 0 }], (t, 0));
        }
        let (x0, z, t) = (self.x0(), self.sel.fresh(), self.sel.fresh());
        (vec![BorgInstr { mnem: Op::Addi.into(), dst: z, srcs: vec![x0.0], swz: vec![0], imm: 0 },
              BorgInstr { mnem: Op::Iadd.into(), dst: t, srcs: vec![v.0, z], swz: vec![v.1, 0], imm: 0 }], (t, 0))
    }

    /// x0, the zero register of ISA v2 instructions (only they may read it).
    pub(crate) fn x0(&mut self) -> (u32, u8) {
        let z = self.sel.fresh();
        self.sel.loc.insert(z, Loc::Fixed(0));
        (z, 0)
    }

    /// Integer min and max: Zbb MIN, MINU, MAX, MAXU.
    unsafe fn int_minmax(&mut self, alu: &nir_alu_instr) {
        let sl = alu.srcs_as_slice();
        let a = self.operand(sl[0].src.as_def().index, sl[0].swizzle[0]);
        let b = self.operand(sl[1].src.as_def().index, sl[1].swizzle[0]);
        let (a, b) = (self.rv_reg(a), self.rv_reg(b));
        let op = match alu.op {
            nir_op_umax => Op::Maxu,
            nir_op_umin => Op::Minu,
            nir_op_imax => Op::Max,
            _ => Op::Min,
        };
        self.sel.prog.push(BorgInstr { mnem: op.into(), dst: alu.def.index, srcs: vec![a.0, b.0], swz: vec![a.1, b.1], imm: 0 });
    }

    /// bitCount, findMSB, findLSB and the high halves of products: Zbb and Zmmul.
    unsafe fn bit_op(&mut self, alu: &nir_alu_instr) {
        let sl = alu.srcs_as_slice();
        let x = self.operand(sl[0].src.as_def().index, sl[0].swizzle[0]);
        let x = self.rv_reg(x);
        let op = |s: &mut Self, mnem: Op, srcs: &[(u32, u8)], imm: u32| {
            let d = s.sel.fresh();
            s.sel.prog.push(BorgInstr { mnem: mnem.into(), dst: d, srcs: srcs.iter().map(|v| v.0).collect(), swz: srcs.iter().map(|v| v.1).collect(), imm });
            (d, 0u8)
        };
        let zero = self.x0();
        // findMSB: 31 - CLZ, which is -1 for 0 (CLZ 32); a negative signed value counts its first 0.
        let msb = |s: &mut Self, v: (u32, u8)| {
            let c = op(s, Op::Clz, &[v], 0);
            let k = op(s, Op::Addi, &[zero], 31);
            op(s, Op::Sub, &[k, c], 0)
        };
        let r = match alu.op {
            nir_op_bit_count => op(self, Op::Cpop, &[x], 0),
            nir_op_ufind_msb => msb(self, x),
            nir_op_ifind_msb => {
                let sg = op(self, Op::Srai, &[x], 31);
                let v = op(self, Op::Xor, &[x, sg], 0);
                msb(self, v)
            }
            // findLSB: CTZ, and -1 for 0 (CTZ 32: bit 5 is set only then).
            nir_op_find_lsb => {
                let t = op(self, Op::Ctz, &[x], 0);
                let b5 = op(self, Op::Srli, &[t], 5);
                let m = op(self, Op::Sub, &[zero, b5], 0);
                op(self, Op::Or, &[t, m], 0)
            }
            _ => {
                let y = self.operand(sl[1].src.as_def().index, sl[1].swizzle[0]);
                let y = self.rv_reg(y);
                op(self, if alu.op == nir_op_umul_high { Op::Mulhu } else { Op::Mulh }, &[x, y], 0)
            }
        };
        self.sel.vec_map.insert(alu.def.index, vec![r]);
    }

    pub unsafe fn intrinsic(&mut self, intr: &nir_intrinsic_instr) {
        if self.stage == Stage::Compute {
            self.compute_intrinsic(intr);
            return;
        }
        match intr.intrinsic {
            nir_intrinsic_ddx | nir_intrinsic_ddx_coarse | nir_intrinsic_ddx_fine => self.derivative(intr, Op::Ddx),
            nir_intrinsic_ddy | nir_intrinsic_ddy_coarse | nir_intrinsic_ddy_fine => self.derivative(intr, Op::Ddy),
            nir_intrinsic_load_input | nir_intrinsic_load_point_coord if self.stage != Stage::Vertex => self.frag_input(intr),
            nir_intrinsic_load_push_constant => self.push_constant(intr),
            nir_intrinsic_vulkan_resource_index => {
                self.res_binding.insert(intr.def.index, intr.get_const_index(NIR_INTRINSIC_BINDING));
            }
            nir_intrinsic_load_vulkan_descriptor => {
                self.vert.descriptor_defs.insert(intr.def.index);
                if let Some(&b) = self.res_binding.get(&intr.get_src(0).as_def().index) {
                    self.res_binding.insert(intr.def.index, b);
                }
            }
            nir_intrinsic_load_ssbo | nir_intrinsic_store_ssbo => self.ssbo(intr),
            nir_intrinsic_load_frag_coord if self.stage == Stage::Fragment => self.frag_coord(intr),
            nir_intrinsic_load_vertex_id => {
                self.sel.loc.insert(intr.def.index, Loc::Fixed(30));
                self.vert.vertex_id_def = Some(intr.def.index);
            }
            nir_intrinsic_load_instance_id => {
                self.sel.loc.insert(intr.def.index, Loc::Fixed(31));
            }
            nir_intrinsic_load_input if self.stage == Stage::Vertex && self.flags.fetching() => self.vertex_fetch(intr),
            nir_intrinsic_load_ubo => self.uniform_load(intr),
            nir_intrinsic_load_input if self.stage == Stage::Vertex => self.uniform_load(intr),
            nir_intrinsic_store_output => self.store_output(intr),
            nir_intrinsic_load_output
                if self.stage == Stage::Fragment && intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) >> 14 & 1 != 0 =>
            {
                self.dst_read(intr)
            }
            nir_intrinsic_load_output => {
                let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                match self.outs.stores.get(&loc) {
                    Some(v) => { self.sel.vec_map.insert(intr.def.index, v.clone()); }
                    None => self.refuse(format!("load_output of location {loc} before any store to it")),
                }
            }
            _ => {}
        }
    }

    /// A blend's read of the destination: the colour attachment's current value at this pixel.
    unsafe fn dst_read(&mut self, intr: &nir_intrinsic_instr) {
        let desc = if self.flags.packed() && !self.flags.multi_target() {
            Some(packed_layout(self.flags))
        } else if self.flags.r8_unorm() {
            Some((0, vec![(0usize, 0u32, 8u32)]))
        } else {
            None
        };
        match desc {
            Some((kind, layout)) if layout.iter().all(|&(_, sh, b)| sh + b <= 32) => {
                let w = self.sel.op0(Op::Tld, vec![]);
                let zero = self.sel.op0(Op::Isub, vec![w, w]);
                let mut rgba: [Option<u32>; 4] = [None; 4];
                for &(comp, sh, bits) in layout.iter() {
                    let mut f = w;
                    if sh != 0 {
                        let s = self.konst(sh);
                        f = self.sel.op0(Op::Isrl, vec![f, s]);
                    }
                    if bits < 32 && (kind != 1) && sh + bits < 32 {
                        let m = self.konst((1u32 << bits) - 1);
                        f = self.sel.op0(Op::Iand, vec![f, m]);
                    }
                    let v = match kind {
                        2 if comp != 3 => {
                            // sRGB to linear: a fetch from the driver's 256-entry table (slot 41).
                            let ctl = self.konst((1 << 16) | (128 + 41));
                            let t = self.sel.op0(Op::Tex, vec![f, zero, ctl]);
                            self.tex_dsts.insert(t);
                            self.sel.op0(Op::Ior, vec![t, t])
                        }
                        0 | 2 => {
                            let i = self.sel.op0(Op::I2f, vec![f]);
                            let inv = self.konst((1.0f32 / ((1u64 << bits) - 1) as f32).to_bits());
                            self.sel.op0(Op::Fmul, vec![i, inv])
                        }
                        1 => {
                            // Sign-extend the field from the word itself.
                            let up = self.konst(32 - bits - sh);
                            let a = if 32 - bits - sh != 0 { self.sel.op0(Op::Ishl, vec![w, up]) } else { w };
                            let dn = self.konst(32 - bits);
                            let s = self.sel.op0(Op::Ishr, vec![a, dn]);
                            let i = self.sel.op0(Op::I2f, vec![s]);
                            let inv = self.konst((1.0f32 / ((1u64 << (bits - 1)) - 1) as f32).to_bits());
                            self.sel.op0(Op::Fmul, vec![i, inv])
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
                let (rgba, zero) = if let Some(c) = self.frag.dst_half { c } else {
                    let w0 = self.sel.op0(Op::Tld, vec![]);
                    let w1 = if nch > 2 { self.sel.op0(Mnem::Tld1, vec![]) } else { w0 };
                    let zero = self.sel.op0(Op::Isub, vec![w0, w0]);
                    let mut rgba: [Option<u32>; 4] = [None; 4];
                    for k in 0..nch {
                        let mut h = if k < 2 { w0 } else { w1 };
                        if k % 2 == 1 {
                            let c16 = self.konst(16);
                            h = self.sel.op0(Op::Isrl, vec![h, c16]);
                        }
                        rgba[k] = Some(self.sel.op0(Op::H2f, vec![h]));
                    }
                    self.frag.dst_half = Some((rgba, zero));
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
    }

    /// Derivatives across the quad, one instruction per component.
    unsafe fn derivative(&mut self, intr: &nir_intrinsic_instr, mnem: Op) {
        let src = intr.srcs_as_slice()[0].as_def().index;
        let n = intr.def.num_components as usize;
        let raw: Vec<u32> = (0..n)
            .map(|c| {
                let (sd, sc) = resolve_vm(&self.sel.vec_map, src, c as u8);
                let v = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: mnem.into(), dst: v, srcs: vec![sd], swz: vec![sc], imm: 0 });
                v
            })
            .collect();
        let comps: Vec<(u32, u8)> = raw.into_iter().map(|v| (v, 0u8)).collect();
        self.sel.vec_map.insert(intr.def.index, comps);
    }

    /// A fragment input: a varying interpolated with the barycentrics, or gl_PointCoord.
    unsafe fn frag_input(&mut self, intr: &nir_intrinsic_instr) {
        let is_pc = intr.intrinsic == nir_intrinsic_load_point_coord;
        let loc = if is_pc { 25 } else { intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F };
        let base_index = if loc == 25 { self.flags.pntc_base() } else { 4 * loc.wrapping_sub(VARYING_SLOT_VAR0) + intr.get_const_index(NIR_INTRINSIC_COMPONENT) as u32 };
        let centroid = !is_pc && self.frag.centroid >> loc.wrapping_sub(VARYING_SLOT_VAR0).min(63) & 1 != 0;
        let bary = if centroid {
            self.frag.centroid_bary
        } else {
            [5u8, 6, 7].map(|r| {
                let v = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.loc.insert(v, Loc::Fixed(r));
                self.frag.per_pixel_fixed.insert(v);
                v
            })
        };
        let n = intr.def.num_components as usize;
        // Integer varyings are flat (Vulkan): the provoking vertex's bits, not a blend.
        let dtype = if is_pc { 0x80 } else { intr.get_const_index(NIR_INTRINSIC_DEST_TYPE) };
        let flat = dtype != 0 && dtype & 0x80 == 0;
        let comps: Vec<(u32, u8)> = (0..n)
            .map(|c| {
                let fattr_rd = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.fattr_dsts.insert(fattr_rd);
                self.sel.prog.push(BorgInstr {
                    mnem: Op::Fattr.into(), dst: fattr_rd, srcs: vec![],
                    swz: vec![],
                    imm: base_index + c as u32,
                });
                if flat {
                    let res = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Ior.into(), dst: res, srcs: vec![fattr_rd, fattr_rd], swz: vec![0, 0], imm: 0 });
                    return (res, 0u8);
                }
                // The record holds a0, a1 - a0 and a2 - a0 (sout_varying): v = a0 + l1*d1 + l2*d2.
                let t1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Fmadd.into(), dst: t1, srcs: vec![bary[1], fattr_rd, fattr_rd], swz: vec![0, 1, 0], imm: 0 });
                let res = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Fmadd.into(), dst: res, srcs: vec![bary[2], fattr_rd, t1], swz: vec![0, 2, 0], imm: 0 });
                (res, 0u8)
            })
            .collect();
        self.sel.vec_map.insert(intr.def.index, comps);
    }

    /// Barycentrics inside both the pixel and the triangle: the centre's when the centre is inside
    /// all three edges, else the first covered sample's (`Ek + ak*ox + bk*oy`, normalized).
    pub(crate) fn centroid_barycentrics(&mut self) -> [u32; 3] {
        let fixed = |st: &mut Self, r: u8| {
            let v = st.sel.fresh();
            st.sel.loc.insert(v, Loc::Fixed(r));
            st.frag.per_pixel_fixed.insert(v);
            v
        };
        let e = [0u8, 1, 2].map(|r| fixed(self, r));
        let one = self.konst(1.0f32.to_bits());
        let not = |st: &mut Self, x: u32| {
            let n = st.sel.op0(Op::Fneg, vec![x]);
            st.sel.op0(Op::Fadd, vec![one, n])
        };
        let mut rest = one;
        for &ek in &e {
            let n = self.sel.op0(Op::Fneg, vec![ek]);
            let out = self.sel.op0(Op::Fstep, vec![n]);
            let inn = not(self, out);
            rest = self.sel.op0(Op::Fmul, vec![rest, inn]);
        }
        rest = not(self, rest);
        let mask = self.sel.op0(Op::Smask, vec![]);
        let bit1 = self.konst(1);
        let (mut ox, mut oy) = (None, None);
        for (i, (x, y)) in [(-0.125f32, -0.375f32), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)].into_iter().enumerate() {
            let shifted = if i == 0 { mask } else { let k = self.konst(i as u32); self.sel.op0(Op::Isrl, vec![mask, k]) };
            let b = self.sel.op0(Op::Iand, vec![shifted, bit1]);
            let b = self.sel.op0(Op::I2f, vec![b]);
            let w = self.sel.op0(Op::Fmul, vec![b, rest]);
            if i < 3 {
                let nb = not(self, b);
                rest = self.sel.op0(Op::Fmul, vec![rest, nb]);
            }
            let (kx, ky) = (self.konst(x.to_bits()), self.konst(y.to_bits()));
            ox = Some(match ox { None => self.sel.op0(Op::Fmul, vec![w, kx]), Some(o) => self.sel.op0(Op::Fmadd, vec![w, kx, o]) });
            oy = Some(match oy { None => self.sel.op0(Op::Fmul, vec![w, ky]), Some(o) => self.sel.op0(Op::Fmadd, vec![w, ky, o]) });
        }
        let (ox, oy) = (ox.unwrap(), oy.unwrap());
        let es = [0u8, 1, 2].map(|k| {
            let (a, b) = (self.sel.fresh(), self.sel.fresh());
            self.sel.loc.insert(a, Loc::Uniform(3 * k));
            self.sel.loc.insert(b, Loc::Uniform(3 * k + 1));
            let t = self.sel.op0(Op::Fmadd, vec![a, ox, e[k as usize]]);
            self.sel.op0(Op::Fmadd, vec![b, oy, t])
        });
        let s01 = self.sel.op0(Op::Fadd, vec![es[0], es[1]]);
        let sum = self.sel.op0(Op::Fadd, vec![s01, es[2]]);
        // One Newton step, as the raster ROM's: 1 - l1 - l2 must not dip below 0 inside the triangle.
        let r = self.sel.op0(Op::Frcp, vec![sum]);
        let ns = self.sel.op0(Op::Fneg, vec![sum]);
        let err = self.sel.op0(Op::Fmadd, vec![ns, r, one]);
        let r = self.sel.op0(Op::Fmadd, vec![r, err, r]);
        es.map(|x| self.sel.op0(Op::Fmul, vec![x, r]))
    }

    /// Push constants, from the uniform window or, with a buffer in the shader, the heap window.
    unsafe fn push_constant(&mut self, intr: &nir_intrinsic_instr) {
        let dyn_off = intr.get_src(0).comp_as_uint(0);
        if let Some(dyn_off) = dyn_off {
            let base = intr.base() + dyn_off as i32 + if self.frag.push_window { PUSH_WINDOW_WORDS as i32 * 4 } else { 0 };
            let n = intr.def.num_components as usize;
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let byte_off = base + 4 * (c as i32);
                    if byte_off % 4 != 0 {
                        self.refuse(format!("push constant field at byte {byte_off} is not word-aligned"));
                    }
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
                        mnem: Op::Load.into(), dst, srcs: vec![idx_reg], swz: vec![0],
                        imm: 0,
                    });
                    (dst, 0u8)
                })
                .collect();
            self.sel.vec_map.insert(intr.def.index, comps);
        } else if matches!(self.stage, Stage::Vertex | Stage::Fragment) {
                            let o = resolve_vm(&self.sel.vec_map, intr.get_src(0).as_def().index, 0);
            let two = self.konst(2);
            let w = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: Op::Isrl.into(), dst: w, srcs: vec![o.0, two], swz: vec![o.1, 0], imm: 0 });
            let base_word = (intr.base() / 4) as u32 + if self.frag.push_window { PUSH_WINDOW_WORDS } else { 0 };
            let comps: Vec<(u32, u8)> = (0..intr.def.num_components as u32)
                .map(|c| {
                    let k = self.konst(base_word + c);
                    let addr = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: addr, srcs: vec![w, k], swz: vec![0, 0], imm: 0 });
                    let dst = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Load.into(), dst, srcs: vec![addr], swz: vec![0], imm: 0 });
                    (dst, 0u8)
                })
                .collect();
            self.sel.vec_map.insert(intr.def.index, comps);
        } else {
            self.refuse(format!("push constant at base {} read with a dynamic offset", intr.base()));
        }
    }

    /// A storage buffer load or store.
    unsafe fn ssbo(&mut self, intr: &nir_intrinsic_instr) {
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
        self.sel.loc.insert(f, Loc::Fixed(30));
        let zero = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: Op::Isub.into(), dst: zero, srcs: vec![f, f], swz: vec![0, 0], imm: 0 });
        // The descriptor's own offset component (a dynamic buffer's) is the window base: 0.
        let off_is_descriptor = self.vert.descriptor_defs.contains(&resolve_vm(&self.sel.vec_map, intr.get_src(osrc).as_def().index, 0).0);
        let word = match intr.get_src(osrc).comp_as_uint(0) {
            Some(_) => None,
            None if off_is_descriptor => None,
            None => {
                let o = resolve_vm(&self.sel.vec_map, intr.get_src(osrc).as_def().index, 0);
                let two = self.konst(2);
                let w = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Isrl.into(), dst: w, srcs: vec![o.0, two], swz: vec![o.1, 0], imm: 0 });
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
            self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: addr, srcs: vec![word.unwrap_or(zero), k], swz: vec![0, 0], imm: 0 });
            if store {
                let src = intr.get_src(0).as_def().index;
                let d = resolve_vm(&self.sel.vec_map, src, c as u8);
                let val = match self.consts.get(&d.0) {
                    Some(&bits) if !self.sel.vec_map.contains_key(&d.0) => {
                        let kv = self.konst(bits);
                        let t = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: t, srcs: vec![zero, kv], swz: vec![0, 0], imm: 0 });
                        (t, 0u8)
                    }
                    _ => d,
                };
                self.sel.prog.push(BorgInstr { mnem: Op::Store.into(), dst: NO_DST, srcs: vec![addr, val.0], swz: vec![0, val.1], imm: 0 });
            } else {
                let dst = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Load.into(), dst, srcs: vec![addr], swz: vec![0], imm: 0 });
                comps.push((dst, 0u8));
            }
        }
        if !store {
            self.sel.vec_map.insert(intr.def.index, comps);
        }
    }

    /// gl_FragCoord.
    unsafe fn frag_coord(&mut self, intr: &nir_intrinsic_instr) {
        let comps: Vec<(u32, u8)> = [30u8, 31, 29, 8]
            .iter()
            .map(|&r| {
                let v = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.loc.insert(v, Loc::Fixed(r));
                self.frag.per_pixel_fixed.insert(v);
                (v, 0u8)
            })
            .collect();
        self.frag.coord_w = true;
        self.sel.vec_map.insert(intr.def.index, comps);
    }

    /// A vertex attribute fetched with TEX from its binding.
    unsafe fn vertex_fetch(&mut self, intr: &nir_intrinsic_instr) {
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
        self.sel.loc.insert(idx, Loc::Fixed(idx_reg));
        // The element index i is texel (i & 4095, i >> 12) of a 4096-wide image.
        let ty = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: Op::Isrl.into(), dst: ty, srcs: vec![idx, twelve], swz: vec![0, 0], imm: 0 });
        let hi = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: Op::Ishl.into(), dst: hi, srcs: vec![ty, twelve], swz: vec![0, 0], imm: 0 });
        let tx = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: Op::Isub.into(), dst: tx, srcs: vec![idx, hi], swz: vec![0, 0], imm: 0 });
        let tex_v = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: Op::Tex.into(), dst: tex_v, srcs: vec![tx, ty, ctl], swz: vec![0, 0, 0], imm: 0 });
        self.tex_dsts.insert(tex_v);
        let comp0 = intr.get_const_index(NIR_INTRINSIC_COMPONENT) as u8;
        let mut comps = Vec::with_capacity(n);
        for c in 0..n {
            let m = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: Op::Ior.into(), dst: m, srcs: vec![tex_v, tex_v],
                                  swz: vec![comp0 + c as u8, comp0 + c as u8],
                                  imm: 0, });
            comps.push((m, 0u8));
        }
        self.sel.vec_map.insert(intr.def.index, comps);
    }

    /// A uniform buffer load, or a vertex input pulled from the vertex buffer.
    unsafe fn uniform_load(&mut self, intr: &nir_intrinsic_instr) {
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
                &self.sel.vec_map, &self.prod, &self.consts, &self.vert.descriptor_defs,
                self.vert.vertex_id_def.unwrap_or(u32::MAX), offset_def,
            )) {
            Some(bs) => bs,
            None => {
                self.refuse(format!("load_ubo offset (def {offset_def}) of an unknown address shape"));
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
                (0, _) if self.stage != Stage::Vertex && base_words > 0 => self.konst(base_words as u32),
                (0, _) if self.stage != Stage::Vertex => self.konst(base_words as u32),
                // A single constant word: the address is a window word, no chain.
                (0, _) if n == 1 => self.konst(base_words as u32),
                _ => {
                    let vid = match self.vert.vertex_id_def {
                        Some(v) => v,
                        None => {
                            let v = self.sel.next_vreg; self.sel.next_vreg += 1;
                            self.sel.loc.insert(v, Loc::Fixed(30));
                            v
                        }
                    };
                    let stride_v = self.konst(stride_words as u32);
                    let tmp = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Imul.into(), dst: tmp, srcs: vec![vid, stride_v], swz: vec![0, 0], imm: 0 });
                    let base_v = self.konst(base_words as u32);
                    let a = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: a, srcs: vec![tmp, base_v], swz: vec![0, 0], imm: 0 });
                    a
                }
            };
            let mut cur = addr;
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let dst = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Load.into(), dst, srcs: vec![cur], swz: vec![0], imm: 0 });
                    if c + 1 < n {
                        let one_v = self.konst(1);
                        let next_a = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr {
                            mnem: Op::Iadd.into(), dst: next_a, srcs: vec![cur, one_v], swz: vec![0, 0],
                            imm: 0,
                        });
                        cur = next_a;
                    }
                    (dst, 0u8)
                })
                .collect();
            if stride_words == 0 && n > 1 {
                let one_v = self.konst(1);
                let past = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: past, srcs: vec![cur, one_v], swz: vec![0, 0], imm: 0 });
                self.const_addr_chain = Some((base_words + n as i32, past));
            }
            self.sel.vec_map.insert(intr.def.index, comps);
        }
    }

    /// An output store: position, varyings, colour targets.
    unsafe fn store_output(&mut self, intr: &nir_intrinsic_instr) {
        let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
        let skip = self.stage == Stage::Fragment && self.frag.out_select >= 0 && (4..12).contains(&loc) && loc != 4 + self.frag.out_select as u32;
        let src = intr.get_src(0).as_def().index;
        let ncomp = intr.get_src(0).num_components() as usize;
        // Alpha to coverage takes location 0's alpha, also while drawing another target.
        let a2c_alpha = skip && loc == 4 && self.frag.alpha && ncomp == 4;
        if skip && !a2c_alpha {
            return;
        }
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
        if a2c_alpha {
            self.outs.frag[3] = Some(comps[3]);
            self.outs.roots.push(comps[3].0);
        } else if self.stage == Stage::Vertex && loc == VARYING_SLOT_POS {
            for (c, &v0) in comps.iter().enumerate().take(4) {
                let mut v = v0;
                if matches!(self.sel.loc.get(&v.0), Some(Loc::Uniform(_))) {
                    let out = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Fmov.into(), dst: out, srcs: vec![v.0], swz: vec![0], imm: 0 });
                    v = (out, 0);
                }
                self.outs.pos[c] = Some(v);
                self.outs.roots.push(v.0);
            }
        } else if self.stage == Stage::Fragment && self.flags.multi_target() && (4..8).contains(&loc) {
            self.outs.mrt[(loc - 4) as usize] = Some(comps[0]);
        } else if self.stage == Stage::Fragment {
            let ncomp2 = if self.frag.alpha || self.flags.bytes_packed() { 4 } else { 3 };
            for (c, &v) in comps.iter().enumerate().take(ncomp2) {
                self.outs.frag[c] = Some(v);
                self.outs.roots.push(v.0);
            }
        } else if self.stage == Stage::Vertex && loc == 12 {
            self.outs.psize = comps.first().copied();   // VARYING_SLOT_PSIZ
        } else if loc < VARYING_SLOT_VAR0 {
            // gl_PointSize and the other built-in outputs: triangles ignore them.
        } else {
            let base_index = 4 * loc.wrapping_sub(VARYING_SLOT_VAR0) + intr.get_const_index(NIR_INTRINSIC_COMPONENT) as u32;
            for (c, &(vd0, vc)) in comps.iter().enumerate() {
                let mut vd = vd0;
                // SOUT stores a register; a constant living in the uniform window needs a copy.
                if matches!(self.sel.loc.get(&vd), Some(Loc::Uniform(_))) {
                    // An integer add keeps the bits of an integer constant (a float move changes a NaN or denormal).
                    let zero = self.cp.zero(&mut self.sel);
                    let out = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: Op::Iadd.into(), dst: out, srcs: vec![zero, vd], swz: vec![0, 0], imm: 0 });
                    vd = out;
                }
                let index = base_index + c as u32;
                if index >= 256 {
                    self.refuse(format!("SOUT index {index} does not fit a byte"));
                }
                self.sout_varying(vd, index);
                let _ = vc;
            }
        }
        self.outs.stores.insert(loc, comps);
    }
    /// Varying `index` of the three corners as a0, a1 - a0 and a2 - a0: the corners are lanes 0-2,
    /// so the coarse derivatives are the differences, and a constant interpolates exactly.
    pub(crate) fn sout_varying(&mut self, v: u32, index: u32) {
        let dx = self.sel.op0(Op::Ddx, vec![v]);
        let dy = self.sel.op0(Op::Ddy, vec![v]);
        for (src, corner) in [(v, 0), (dx, 2 << 8), (dy, 3 << 8)] {
            self.sel.prog.push(BorgInstr { mnem: Op::Sout.into(), dst: NO_DST, srcs: vec![src], swz: vec![0], imm: index | corner });
            self.outs.roots.push(src);
        }
    }

    /// A loop runs while EXANY sees an alive lane: `go` masks one trip, a jump clears it.
    fn loop_mark(&mut self, mnem: Mnem, id: usize, (alive, go, any): (u32, u32, u32)) {
        let one = if mnem == Mnem::LoopHead { self.konst(1) } else { 0 };
        let i = |mnem: Mnem, dst: u32, srcs: Vec<u32>, swz: Vec<u8>| BorgInstr { mnem, dst, srcs, swz, imm: 0 };
        let label = BorgInstr { mnem: Mnem::Label, dst: NO_DST, srcs: vec![], swz: vec![], imm: id as u32 };
        let seq = match mnem {
            Mnem::LoopHead => vec![
                i(Op::Ior.into(), alive, vec![one, one], vec![0, 0]),
                label,
                i(Op::Ior.into(), go, vec![alive, alive], vec![0, 0]),
                i(Op::Expush.into(), NO_DST, vec![go], vec![0]),
            ],
            Mnem::LoopEnd => vec![
                i(Op::Expop.into(), NO_DST, vec![], vec![]),
                i(Op::Expush.into(), NO_DST, vec![alive], vec![0]),
                i(Op::Exany.into(), any, vec![], vec![]),
                i(Op::Expop.into(), NO_DST, vec![], vec![]),
                BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![any], swz: vec![0], imm: id as u32 },
            ],
            _ => vec![i(Op::Expush.into(), NO_DST, vec![go], vec![0])],
        };
        self.sel.prog.extend(seq);
    }

    /// Selects Borg instructions for the entry function in virtual registers, one NIR instruction
    /// at a time.
    pub unsafe fn select_entry(&mut self, entry: *mut nir_function_impl) {
        let mut cf_marks: std::collections::HashMap<usize, Vec<CfMark>> =
            std::collections::HashMap::new();
        let mut loops = crate::Loops::default();
        collect_cf_marks((*entry).iter_body(), &mut cf_marks, &mut loops, None);
        if loops.count > 256 {
            loops.err = Some(format!("{} loops; labels are 8 bits", loops.count));
        }
        if let Some(e) = loops.err.take() {
            self.refuse(e);
        }
        // Per loop: lanes still looping, lanes going round this time, and whether any lane is.
        let lv: Vec<(u32, u32, u32)> = (0..loops.count).map(|_| (self.sel.fresh(), self.sel.fresh(), self.sel.fresh())).collect();

        let mut reg_vregs: HashMap<u32, Vec<u32>> = HashMap::new();
        for block in (*entry).iter_blocks() {
            // A chained address is only valid in the lanes of the block that computed it.
            self.const_addr_chain = None;
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
                        imm: 0,
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
                            self.sel.prog.push(BorgInstr { mnem: Op::Ior.into(), dst: d, srcs: vec![zero, zero], swz: vec![0, 0], imm: 0 });
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
                            // Copy the bits: a float move flushes an integer that looks denormal.
                            self.sel.prog.push(BorgInstr { mnem: Op::Ior.into(), dst: reg_vregs[&h][c], srcs: vec![v, v], swz: vec![comp, comp], imm: 0 });
                        }
                        continue;
                    }
                }
                if let Some(alu) = instr.as_alu() {
                    self.alu(alu);
                } else if let Some(intr) = instr.as_intrinsic() {
                    self.intrinsic(intr);
                } else if let Some(tex) = instr.as_tex() {
                    if self.stage == Stage::Compute && tex.op != nir_texop_txl {
                        self.skipped.push("tex");
                        continue;
                    }
                    self.tex(tex);
                } else if let Some(lc) = instr.as_load_const() {
                    self.load_const(lc);
                } else if let Some(u) = instr.as_undef() {
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
            self.sel.prog.push(BorgInstr { mnem: Op::Ior.into(), dst: out, srcs: vec![v, v], swz: vec![comp, comp], imm: 0 });
        } else {
            self.sel.prog.push(BorgInstr { mnem: Op::Fmov.into(), dst: out, srcs: vec![v], swz: vec![comp], imm: 0 });
        }
        self.outs.frag[c] = Some((out, 0));
        self.out_roots.push(out);
    }
    
    }
    /// ISA v2 (B8 step 8): each fragment output also through its CSR, out_c0..out_c3.
    pub fn csr_frag_outputs(&mut self) {
        for c in 0..4 {
            if let Some((v, comp)) = self.outs.frag[c] {
                self.sel.prog.push(BorgInstr { mnem: Op::Csrrw.into(), dst: NO_DST, srcs: vec![v], swz: vec![comp], imm: CSR_OUT_C0 + c as u32 });
            }
        }
    }
    pub fn log_uniformity(&self) {
    let uniform_roots: std::collections::HashSet<u32> = self.sel.loc
        .iter()
        .filter_map(|(&v, u)| match u {
            Loc::Uniform(_) => Some(v),
            Loc::Fixed(_) if !self.frag.per_pixel_fixed.contains(&v) => Some(v),
            _ => None,
        })
        .collect();
    let uniform = classify_uniform(&self.sel.prog, &uniform_roots);
    if crate::config::env().dump_isa {
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

