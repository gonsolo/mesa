// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ir::*;

struct SMEncoder {
    #[allow(dead_code)]
    ip: usize,
    inst: [u32; 4]
}

impl SMEncoder {
    fn set_opcode(&mut self, opcode: u16) {
        // TODO
    }
}

trait SMOp {
    fn encode(&self, e: &mut SMEncoder);
}

impl SMOp for OpMov {

    fn encode(&self, e: &mut SMEncoder) {
        e.set_opcode(0x666)
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
                    inst: [0_u32; 4],
                };
                as_sm_op(&instr.op).encode(&mut e);
                encoded.extend_from_slice(&e.inst[..]);
            }
        }
        encoded
    }
}
