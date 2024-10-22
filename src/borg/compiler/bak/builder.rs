// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

use crate::ir::*;

pub trait Builder {
    fn push_instr(&mut self, instr: Box<Instr>) -> &mut Instr;

    fn copy_to(&mut self, dst: Dst, src: Src) {
        self.push_op(OpCopy { dst: dst, src: src });
    }

    fn push_op(&mut self, op: impl Into<Op>) -> &mut Instr {
        self.push_instr(Instr::new_boxed(op))
    }
}

pub trait SSABuilder: Builder {

    fn alloc_ssa(&mut self, file: RegFile, comps: u8) -> SSARef;

    fn copy(&mut self, src: Src) -> SSARef {
        let dst = if src.is_predicate() {
            self.alloc_ssa(RegFile::Pred, 1)
        } else {
            self.alloc_ssa(RegFile::GPR, 1)
        };
        self.copy_to(dst.into(), src);
        dst
    }
}

pub struct InstrBuilder {
    instrs: MappedInstrs,
}

impl InstrBuilder {
    pub fn new() -> Self {
        Self {
            instrs: MappedInstrs::None,
        }
    }
}


impl Builder for InstrBuilder {
    fn push_instr(&mut self, instr: Box<Instr>) -> &mut Instr {
        self.instrs.push(instr);
        self.instrs.last_mut().unwrap().as_mut()
    }
}

pub struct SSAInstrBuilder<'a> {
    b: InstrBuilder,
    alloc: &'a mut SSAValueAllocator,
}

impl<'a> SSAInstrBuilder<'a> {
    pub fn new(
        alloc: &'a mut SSAValueAllocator,
    ) ->Self {
        Self {
            b: InstrBuilder::new(),
            alloc: alloc
        }
    }
}


impl<'a> Builder for SSAInstrBuilder<'a> {

    fn push_instr(&mut self, instr: Box<Instr>) -> &mut Instr {
        self.b.push_instr(instr)
    }

}

impl<'a> SSABuilder for SSAInstrBuilder<'a> {
    fn alloc_ssa(&mut self, file: RegFile, comps: u8) -> SSARef {
        self.alloc.alloc_vec(file, comps)
    }
}

