// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ir::*;

struct TrivialRegAlloc {}

impl TrivialRegAlloc {

    pub fn new() -> TrivialRegAlloc {
        TrivialRegAlloc {}
    }

    pub fn alloc_reg(&mut self, file: RegFile) -> Dst {
        let idx = match file {
            RegFile::GPR => {
                let idx = 0;
                idx
            }
        };
        let r = RegRef::new(file, idx, 1);
        Dst::Reg(r)
    }

    pub fn rewrite_ref(&mut self, dst: &Dst ) -> Dst {
        if let Dst::SSA(ssa) = dst {
            let reg = self.alloc_reg(ssa.file().unwrap());
            reg
        } else {
            *dst
        }
    }

    pub fn do_alloc(&mut self, s: &mut Shader) {
        for f in &mut s.functions {
            for b in &mut f.blocks {
                for instr in &mut b.instrs {
                    for dst in instr.dsts_mut() {
                        let new_dst = self.rewrite_ref(&dst);
                        *dst = new_dst;
                    }
                }
            }
        }
    }
}


impl Shader {

    pub fn assign_regs(&mut self) {
        TrivialRegAlloc::new().do_alloc(self);
    }
}

