// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::{Op, CSR_UNIFORM0};
use crate::state::{ShaderState, Stage};
use crate::{BorgInstr, Loc, Mnem};
use crate::{HashMap, HashSet};

/// What makes a program one `legal::check` takes.
impl ShaderState {
    /// Between the pressure and the latency schedule: TEX results, and window words where the encoder takes them.
    pub(crate) fn legalize(&mut self) {
        self.copy_tex_results();
        self.v2_operands();
        self.split_window_reads();
    }

    /// An instruction reads one window word; the others reach it through a register.
    fn split_window_reads(&mut self) {
        let is_win = |v: &u32, loc: &HashMap<u32, Loc>| matches!(loc.get(v), Some(Loc::Uniform(_)));
        if self.sel.prog.iter().any(|i| i.srcs.iter().filter(|v| is_win(v, &self.sel.loc)).count() > 1) {
            let f = self.sel.next_vreg;
            let z = self.sel.next_vreg + 1;
            self.sel.next_vreg += 2;
            self.sel.loc.insert(f, Loc::Fixed(30));
            let mut out = vec![BorgInstr { mnem: Op::Isub.into(), dst: z, srcs: vec![f, f], swz: vec![0, 0], imm: 0 }];
            for mut ins in std::mem::take(&mut self.sel.prog) {
                let mut seen = false;
                for k in 0..ins.srcs.len() {
                    if !is_win(&ins.srcs[k], &self.sel.loc) {
                        continue;
                    }
                    if !seen {
                        seen = true;
                        continue;
                    }
                    let t = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    out.push(BorgInstr { mnem: Op::Iadd.into(), dst: t, srcs: vec![ins.srcs[k], z], swz: vec![ins.swz[k], 0], imm: 0 });
                    ins.srcs[k] = t;
                    ins.swz[k] = 0;
                }
                out.push(ins);
            }
            self.sel.prog = out;
        }
    }

    /// Window operands through a register.
    fn v2_operands(&mut self) {
        let x0_forms = |m: Mnem| matches!(m, Mnem::Op(Op::Iadd | Op::Isub | Op::Ishr | Op::Isrl | Op::Imul | Op::Islt
            | Op::Isltu | Op::Fmov | Op::I2f | Op::F2i | Op::FcvtWuS | Op::F2h | Op::H2f));
        let custom = |m: Mnem| matches!(m, Mnem::Op(Op::Frcp | Op::Frsq | Op::Fexp2 | Op::Flog2 | Op::Ddx | Op::Ddy | Op::Expush | Op::Exany));
        let mut out = Vec::with_capacity(self.sel.prog.len());
        for mut i in std::mem::take(&mut self.sel.prog) {
            let (x, c) = (x0_forms(i.mnem), custom(i.mnem));
            if !x && !c {
                out.push(i);
                continue;
            }
            for k in 0..i.srcs.len() {
                let (code, v) = self.window_copy((i.srcs[k], i.swz[k]));
                out.extend(code);
                (i.srcs[k], i.swz[k]) = v;
            }
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// The bits of a window word in a register: + 0 by integer IADD (FMOV is an FMA and would
    /// change a NaN-looking integer); anything else is returned as it is.
    pub(crate) fn window_copy(&mut self, v: (u32, u8)) -> (Vec<BorgInstr>, (u32, u8)) {
        let Some(&Loc::Uniform(u)) = self.sel.loc.get(&v.0) else {
            return (Vec::new(), v);
        };
        let t = self.sel.fresh();
        if self.bits_of(v.0).is_none() {
            // A word the hardware stages (the triangle's record): its CSR.
            return (vec![BorgInstr { mnem: Op::Csrrs.into(), dst: t, srcs: vec![], swz: vec![], imm: CSR_UNIFORM0 + u as u32 }], (t, 0));
        }
        // A constant: an OR with 0 whose window operand the immediates pass replaces.
        let (x0, z) = (self.x0(), self.sel.fresh());
        (vec![BorgInstr { mnem: Op::Addi.into(), dst: z, srcs: vec![x0.0], swz: vec![0], imm: 0 },
              BorgInstr { mnem: Op::Ior.into(), dst: t, srcs: vec![v.0, z], swz: vec![v.1, 0], imm: 0 }], (t, 0))
    }

    /// Every TEX writes the same register block: a result still read after the next TEX is copied
    /// out right after its own.
    fn copy_tex_results(&mut self) {
        let is_tex = |i: &BorgInstr, d: &HashSet<u32>| matches!(i.mnem, Mnem::Op(Op::Tex | Op::Texd)) && d.contains(&i.dst);
        let texs: Vec<usize> = (0..self.sel.prog.len()).filter(|&i| is_tex(&self.sel.prog[i], &self.tex_dsts)).collect();
        let prog = &self.sel.prog;
        // A loop's label and its back jump: a read in the loop also follows a TEX later in it.
        let loops: Vec<(usize, usize)> = (0..prog.len())
            .filter(|&h| prog[h].mnem == Mnem::Label)
            .filter_map(|h| {
                (h..prog.len()).rev().find(|&e| matches!(prog[e].mnem, Mnem::Op(Op::Brnz)) && prog[e].imm == prog[h].imm).map(|e| (h, e))
            })
            .collect();
        for w in texs.windows(2).rev() {
            let (p, next) = (w[0], w[1]);
            let t = self.sel.prog[p].dst;
            let back = loops.iter().filter(|&&(h, e)| p < h && h <= next && next <= e).map(|&(h, _)| h).min();
            let from = back.unwrap_or(next + 1);
            let mut comps: Vec<u8> = self.sel.prog[from..]
                .iter()
                .flat_map(|i| i.srcs.iter().zip(&i.swz).filter(|(s, _)| **s == t).map(|(_, &c)| c))
                .collect();
            comps.sort();
            comps.dedup();
            let copies: Vec<(u8, u32)> = comps.iter().map(|&c| (c, self.sel.fresh())).collect();
            for i in &mut self.sel.prog[from..] {
                for (s, c) in i.srcs.iter_mut().zip(i.swz.iter_mut()) {
                    if let Some(&(_, v)) = copies.iter().find(|(k, _)| *s == t && k == c) {
                        (*s, *c) = (v, 0);
                    }
                }
            }
            for (k, &(c, v)) in copies.iter().enumerate() {
                self.sel.prog.insert(p + 1 + k, BorgInstr { mnem: Op::Ior.into(), dst: v, srcs: vec![t, t], swz: vec![c, c], imm: 0 });
            }
        }
    }

    /// A constant a live instruction reads without a register (integer math the selector left as an
    /// immediate, a vertex shader's `% 3`, say): the register it got later, or one of its own.
    pub(crate) fn place_const_operands(&mut self) {
        let mut live = self.sel.prog.clone();
        crate::opt::dce(&mut live, &self.out_roots);
        let defs: HashSet<u32> = self.sel.prog.iter().map(|i| i.dst).collect();
        let mut need: Vec<u32> = live
            .iter()
            .flat_map(|i| i.srcs.iter().copied())
            .filter(|s| !defs.contains(s) && !self.sel.loc.contains_key(s) && self.consts.contains_key(s))
            .filter(|s| !self.tex_dsts.contains(s) && !self.fattr_dsts.contains(s))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        need.sort();
        let mut front = Vec::new();
        for d in need {
            let (v, c) = match self.sel.vec_map.get(&d) {
                Some(_) => crate::resolve_vm(&self.sel.vec_map, d, 0),
                // A vertex shader's konst(0) is built where selection stands: here, after the uses.
                None if self.consts[&d] == 0 && self.stage == Stage::Vertex => (self.materialize(&mut front, 0), 0),
                None => (self.konst(self.consts[&d]), 0),
            };
            for ins in self.sel.prog.iter_mut() {
                for (s, sc) in ins.srcs.iter_mut().zip(ins.swz.iter_mut()) {
                    if *s == d {
                        (*s, *sc) = (v, c);
                    }
                }
            }
        }
        self.sel.prog.splice(0..0, front);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Flags, NO_DST};

    fn op(m: Mnem, dst: u32, srcs: Vec<u32>, imm: u32) -> BorgInstr {
        BorgInstr { mnem: m, dst, swz: vec![0; srcs.len()], srcs, imm }
    }

    #[test]
    fn tex_result_read_in_a_loop_survives_its_next_tex() {
        let mut c = ShaderState::new(&[17, 18, 19], Stage::Fragment, Flags(0));
        let (u, size, i, t, go) = (c.sel.fresh(), c.sel.fresh(), c.sel.fresh(), c.sel.fresh(), c.sel.fresh());
        c.tex_dsts.extend([size, t]);
        c.sel.prog = vec![
            op(Op::Tex.into(), size, vec![u], 0),
            op(Mnem::Label, NO_DST, vec![], 0),
            op(Op::Mul.into(), i, vec![u, size], 0),
            op(Op::Tex.into(), t, vec![i], 0),
            op(Op::Brnz.into(), NO_DST, vec![go], 0),
        ];
        c.copy_tex_results();
        let p = &c.sel.prog;
        assert!(matches!(p[1].mnem, Mnem::Op(Op::Ior)) && p[1].srcs == [size, size]);
        assert_eq!(p[3].srcs, [u, p[1].dst]);
    }
}
