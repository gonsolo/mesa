// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::Op;
use crate::state::ShaderState;
use crate::{BorgInstr, Loc};
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
}
