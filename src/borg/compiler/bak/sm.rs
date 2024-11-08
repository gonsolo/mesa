// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ir::*;
use std::ops::Range;

struct SMEncoder {
    #[allow(dead_code)]
    ip: usize,
    inst: u32
}

impl SMEncoder {

    fn set_reg(&mut self, range: Range<usize>, reg: RegRef) {
        // RISCV ABI 1.1
        // a0 is x10, the 10th register, or 01010 in binary
        let a0 = 0b01010;
        let register = a0 + reg.base_idx();
        self.set_field32(range, register);
    }

    fn set_src(&mut self, src: Src) {
        match src.src_ref {
            SrcRef::Zero => {
                self.set_field32(20..32, 0);
            },
            _ => panic!("Unimplemented in set_src!")
        }
    }

    fn set_dst(&mut self, dst: Dst) {
        match dst {
            Dst::Reg(reg) => self.set_reg(7..12, reg),
            _ => panic!("Not a register"),
        }
    }

    fn set_field32(&mut self, range: Range<usize>, val: u32) {
        let mask = u32::MAX >> (32 - range.len());
        self.inst= (self.inst & !(mask << range.start)) | (val << range.start);
    }

    fn set_field8(&mut self, range: Range<usize>, val: u8) {
        let val32 = val as u32;
        self.set_field32(range, val32);
    }

    fn set_opcode(&mut self, opcode: u8) {
        self.set_field8(0..7, opcode);
    }
}

trait SMOp {
    fn encode(&self, e: &mut SMEncoder);
}

impl SMOp for OpMov {

    fn encode(&self, e: &mut SMEncoder) {

        //let lui:  u8 = 0b0110111;
        let addi: u8 = 0b0010011;
        println!("pre  set_opcode: {:#034b}", e.inst);
        e.set_opcode(addi);
        println!("post set_opcode: {:#034b}", e.inst);

        e.set_dst(self.dst);
        println!("post set_dst:    {:#034b}", e.inst);

        e.set_src(self.src);
        println!("post set_src:    {:#034b}", e.inst);
    }
}

macro_rules! as_sm_op_match {
    ($op: expr) => {
        match $op {
            Op::Mov(op) => op,
            _ => panic!("Unsupported op: {}", $op),
        }
    }
}

fn as_sm_op(op: &Op) -> &dyn SMOp {
    as_sm_op_match!(op)
}

pub struct ShaderModel {}

impl ShaderModel {

    pub fn encode_shader(&self, shader: &Shader) -> Vec<u32> {
        println!("Shader::encode");

        assert!(shader.functions.len() == 1);
        let func = &shader.functions[0];
        let mut encoded = Vec::new();

        for b in &func.blocks {
            for instr in &b.instrs {
                println!("Encoding instr {}", instr);
                let mut e = SMEncoder {
                    ip: encoded.len(),
                    inst: 0_u32,
                };
                as_sm_op(&instr.op).encode(&mut e);
                encoded.push(e.inst);
            }
        }
        println!("encoded: {:?}", encoded);
        encoded
    }
}
