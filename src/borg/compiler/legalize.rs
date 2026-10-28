// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::{Op, CSR_UNIFORM0};
use crate::state::ShaderState;
use crate::{BorgInstr, Loc, Mnem};
use std::collections::HashMap;

/// What makes a program one `legal::check` takes.
impl ShaderState {
    /// An instruction reads one window word; the others reach it through a register.
    pub(crate) fn split_window_reads(&mut self) {
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
    pub(crate) fn v2_operands(&mut self) {
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
}
