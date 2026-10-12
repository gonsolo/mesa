use crate::state::{Cx, DRAW_VS_CONST_U0, DRAW_VS_CONST_WORDS};
use crate::{packed_layout, BorgInstr, Ubo, NO_DST};

impl Cx {
    pub unsafe fn epilogue_point_quads(&mut self) {
        let ctl_bits = (1u32 << 16) | (128 + 16);
        assert!(self.cp.vs_consts.len() < DRAW_VS_CONST_WORDS,
            "borgc: vertex shader needs more than {} window constants", DRAW_VS_CONST_WORDS);
        let u = DRAW_VS_CONST_U0 + self.cp.vs_consts.len() as u8;
        self.cp.vs_consts.push((u, ctl_bits));
        let ctl = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.ubo.insert(ctl, Ubo::Uniform(u));
        let idx = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.ubo.insert(idx, Ubo::Fixed(30));
        let twelve = self.cp.vs_const(&mut self.sel, 12, false);   // shared with the attribute fetches
        let ty = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "ISRL", dst: ty, srcs: vec![idx, twelve], swz: vec![0, 0] });
        let hi = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "ISHL", dst: hi, srcs: vec![ty, twelve], swz: vec![0, 0] });
        let tx = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "ISUB", dst: tx, srcs: vec![idx, hi], swz: vec![0, 0] });
        let tex_v = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "TEX", dst: tex_v, srcs: vec![tx, ty, ctl], swz: vec![0, 0, 0] });
        self.tex_dsts.insert(tex_v);
        let mut d = [0u32; 4];
        for k in 0..4 {
            let m = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "IOR", dst: m, srcs: vec![tex_v, tex_v], swz: vec![k as u8, k as u8] });
            d[k] = m;
        }
        // gl_PointCoord rides as two more varyings, after the shader's own.
        if self.flags.pntc_varyings() {
            let base = self.sel.prog.iter().filter(|i| i.mnem == "SOUT").map(|i| i.swz[0] as u32 + 1).max().unwrap_or(0);
            assert!(base + 2 <= 255, "borgc: no varying slot left for the point coordinate");
            for k in 0..2u32 {
                self.sel.prog.push(BorgInstr { mnem: "SOUT", dst: NO_DST, srcs: vec![d[2 + k as usize]], swz: vec![(base + k) as u8] });
                self.outs.roots.push(d[2 + k as usize]);
            }
        }
        if let (Some(px), Some(py), Some(pw)) = (self.outs.pos[0], self.outs.pos[1], self.outs.pos[3]) {
            let w = self.sel.reg(pw);
            let size = self.outs.psize.map(|ps| {
                // Clamp to the advertised range [1, 64]: x - c*(x - 64), c = step(x - 64).
                let x = self.sel.reg(ps).0;
                // upper: x -= step(x - 64) * (x - 64)
                let neg_hi = self.cp.vs_win(&mut self.sel, (-64.0f32).to_bits());
                let t = self.sel.op0("FADD", vec![x, neg_hi]);
                let c = self.sel.op0("FSTEP", vec![t]);
                let ct = self.sel.op0("FMUL", vec![c, t]);
                let nct = self.sel.op0("FNEG", vec![ct]);
                let x = self.sel.op0("FADD", vec![x, nct]);
                // lower: x += step(1 - x) * (1 - x); 1.0 = step(dx * dx), as the window has no word left
                let dd = self.sel.op0("FMUL", vec![d[0], d[0]]);
                let one = self.sel.op0("FSTEP", vec![dd]);
                let nx = self.sel.op0("FNEG", vec![x]);
                let t2 = self.sel.op0("FADD", vec![nx, one]);
                let c2 = self.sel.op0("FSTEP", vec![t2]);
                let m2 = self.sel.op0("FMUL", vec![c2, t2]);
                let x = self.sel.op0("FADD", vec![x, m2]);
                (x, 0u8)
            });
            for (k, pc) in [px, py].into_iter().enumerate() {
                let base = self.sel.reg(pc);
                let mut off = d[k];
                if let Some(sz) = size {
                    let o = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: o, srcs: vec![off, sz.0], swz: vec![0, sz.1] });
                    off = o;
                }
                let o = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: o, srcs: vec![off, w.0, base.0],
                                      swz: vec![0, w.1, base.1] });
                self.outs.pos[k] = Some((o, 0));
                self.outs.roots.push(o);
            }
        }
    }
}
