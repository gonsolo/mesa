use crate::state::Cx;
use crate::{resolve_vm, BorgInstr, Ubo, FRAG_CONST_MAX, NO_DST};
use compiler::bindings::*;
use compiler::nir::AsDef;

impl Cx {
    pub unsafe fn load_const(&mut self, lc: &nir_load_const_instr) {
        let n = lc.def.num_components as usize;
        if n == 1 {
            self.consts.insert(lc.def.index, unsafe { lc.values()[0].u32_ });
        } else if self.draw_mode {
            // Vector constant (lightDir): a word in the existing
            // uniform memory window (u20+, draw_fs_const_offset)
            // rather than a pinned GPR. NOT a legacy-style GPR
            // pin: legacy's r17-19(+23) sit inside a register-
            // pressured draw-mode VERTEX shader's own reachable
            // general-pool range (the self.cp.regs reservation
            // comment has the full story -- a hardware-caught
            // bug, cube.vert's address-chain LOADs silently
            // zeroing cube.frag's lightDir), and r0-4 cannot be
            // the fix either (BorgRasterRom's "REGISTER CLOBBER
            // ABI" destroys them every pixel). A uniform-memory
            // read shares nothing with the GPR file vertex also
            // uses, so it cannot collide with vertex at all --
            // and unlike a real register spill, it costs no
            // extra instructions: the funct3 tag that marks an
            // operand as "read from memory" is already free on
            // every instruction, not a separate LOAD.
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let bits = unsafe { lc.values()[c].u32_ };
                    if self.cp.vs_window {
                        return (self.cp.vs_const(&mut self.sel, bits, false), 0u8);
                    }
                    let idx = self.fw.slot(bits);
                    let v = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    self.sel.ubo.insert(v, Ubo::Uniform(idx as u8));
                    (v, 0u8)
                })
                .collect();
            self.sel.vec_map.insert(lc.def.index, comps);
        } else {
            // Legacy: pin components to constant GPRs (firmware
            // writes them once via MMIO).
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let reg = self.cp.alloc_reg();
                    let bits = unsafe { lc.values()[c].u32_ };
                    self.cp.uniforms.push((reg, bits));
                    let v = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    self.sel.ubo.insert(v, Ubo::Fixed(reg));
                    (v, 0u8)
                })
                .collect();
            self.sel.vec_map.insert(lc.def.index, comps);
        }
    }

    pub unsafe fn tex(&mut self, tex: &nir_tex_instr) {
        // Texture sample → TEX rd, u, v, ctl: R, G, B, A to rd..rd+3.
        // Only a plain implicit-LOD sample of a 2D texture so far;
        // anything else is reported rather than sampled wrongly.
        let srcs = tex.srcs_as_slice();
        let is_txl = tex.op == nir_texop_txl && self.is_fragment && self.draw_mode;
        let is_txb = tex.op == nir_texop_txb && self.is_fragment && self.draw_mode;
        let is_tg4 = tex.op == nir_texop_tg4 && self.is_fragment && self.draw_mode;
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
            // A fetch of a 1D texture or texel buffer takes the integer texel in u;
            // v is unused, so u is passed twice.
            let (vd, vc) = if is_fetch && tex.coord_components == 1 {
                (ud, uc)
            } else {
                resolve_vm(&self.sel.vec_map, cd, 1)
            };
            let is_cube = tex.sampler_dim == GLSL_SAMPLER_DIM_CUBE;
            // A third coordinate (3D, 2D array) goes to TEXA as w; a cube's is the compiler's face math.
            let has_w = !is_fetch && tex.coord_components >= 3 && !is_cube && self.is_fragment && self.draw_mode;
            let has_bias = (is_txb || tex.op == nir_texop_tex) && !is_fetch && srcs.iter().any(|s| s.src_type == nir_tex_src_bias);
            if is_txl || has_w || has_bias {
                // Explicit LOD: TEXA hands this lane's LOD to the TEX after it (w and the
                // depth reference are unused for a 2D sample, so r30 stands in for both).
                let ld = srcs.iter().find(|s| s.src_type == nir_tex_src_lod || s.src_type == nir_tex_src_bias).map(|s| s.src.as_def().index).unwrap_or(cd);
                if let Some(&bits) = self.consts.get(&ld).filter(|_| is_txl || has_bias) {
                    if !self.sel.vec_map.contains_key(&ld) {
                        let u = self.fw.slot(bits);
                        let v = self.sel.next_vreg;
                        self.sel.next_vreg += 1;
                        self.sel.ubo.insert(v, Ubo::Uniform(u as u8));
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
                let u = self.fw.slot(bits);
                let c = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.ubo.insert(c, Ubo::Uniform(u as u8));
                c
            } else if has_bias {
                // LOD mode 1: implicit LOD plus TEXA's bias.
                let u = self.fw.slot(1u32 << 21);
                let c = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.ubo.insert(c, Ubo::Uniform(u as u8));
                c
            } else if is_fetch || is_txl {
                // Control word: operation 1 (fetch) in bits 17:16, texture 0, sampler 0,
                // read from the constant window (docs/B2_texture_unit.md).
                let ctl_bits = if is_txl { 2u32 << 21 } else { 1u32 << 16 };
                let slot = if is_txl { &mut self.lod_ctl_uniform } else { &mut self.fetch_ctl_uniform };
                let u = match *slot {
                    Some(u) => u,
                    None => {
                        assert!(self.fw.count < FRAG_CONST_MAX, "borgc: fragment shader needs more than 11 window constants");
                        let u = 20 + self.fw.count;
                        self.fw.count += 1;
                        self.fw.consts.push((u, ctl_bits));
                        *slot = Some(u);
                        u
                    }
                };
                let c = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.ubo.insert(c, Ubo::Uniform(u as u8));
                c
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
}
