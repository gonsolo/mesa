// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::builder::*;
use crate::ir::*;

struct LowerCopy {}

impl LowerCopy {

    fn run(&self, s: &mut Shader) {
        s.map_instrs(|instr: Box<Instr>, _| -> MappedInstrs {
            match instr.op {
                Op::Copy(copy) => {
                    let mut b = InstrBuilder::new() ;
                    let dst_reg = copy.dst.as_reg().unwrap();
                    match dst_reg.file() {
                        RegFile::GPR => match copy.src.src_ref {
                            SrcRef::Zero => {
                                b.push_op(OpMov {
                                    dst: copy.dst,
                                    src: copy.src,
                                });

                            },
                            _ => panic!("Bla")
                        }
                    }
                    b.as_mapped_instrs()
                },
                _ => MappedInstrs::One(instr)
            }
        })
    }
}

impl Shader {

    pub fn lower_copy(&mut self) {
        let pass = LowerCopy {};
        pass.run(self)
    }
}

