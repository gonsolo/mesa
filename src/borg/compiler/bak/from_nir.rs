// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(non_upper_case_globals)]

#![allow(dead_code)]

use crate::builder::*;
use crate::ir::*;

use bak_bindings::*;

use compiler::bindings::*;
use compiler::cfg::CFGBuilder;
use compiler::nir::*;

use std::collections::HashMap;

struct PhiAllocMap<'a> {
    alloc: &'a mut PhiAllocator,
    map: HashMap<(u32, u8), u32>,
}

impl<'a> PhiAllocMap<'a> {
    fn new(alloc: &'a mut PhiAllocator) -> PhiAllocMap<'a> {
        PhiAllocMap {
            alloc: alloc,
            map: HashMap::new(),
        }
    }
}

struct ShaderFromNir<'a> {
    nir: &'a nir_shader,
    cfg: CFGBuilder<u32, BasicBlock>,
    label_alloc: LabelAllocator,
    block_label: HashMap<u32, Label>,
    ssa_map: HashMap<u32, Vec<SSAValue>>,
}

impl<'a> ShaderFromNir<'a> {

    fn new(nir: &'a nir_shader) -> Self {
        Self {
            nir: nir,
            cfg: CFGBuilder::new(),
            label_alloc: LabelAllocator::new(),
            block_label: HashMap::new(),
            ssa_map: HashMap::new(),
        }
    }

    fn get_block_label(&mut self, block: &nir_block) -> Label {
        *self
            .block_label
            .entry(block.index)
            .or_insert_with(|| self.label_alloc.alloc())
    }

    fn parse_load_const(
        &mut self,
        b: &mut impl SSABuilder,
        load_const: &nir_load_const_instr,
    ) {
        println!("  load const instruction");
        let values = &load_const.values();

        let mut dst = Vec::new();
        match load_const.def.bit_size {
            32 => {
                println!("  bit size 32, num components: {}", load_const.def.num_components);
                for c in 0..load_const.def.num_components {
                    let imm_u32 = unsafe { values[usize::from(c)].u32_ };
                    println!("  imm_u32: {}", imm_u32);
                    dst.push(b.copy(imm_u32.into())[0]);
                }
            }
            _ => panic!("Unknown bit size: {}", load_const.def.bit_size),
        }
        self.set_ssa(&load_const.def, dst);

        println!("{:?}", self.ssa_map);
    }

    fn parse_block(
        &mut self,
        ssa_alloc: &mut SSAValueAllocator,
        _phi_map: &mut PhiAllocMap,
        nb: &nir_block,
    ) {
        println!("ShaderFromNir::parse_block TODO");

        let mut b = SSAInstrBuilder::new(ssa_alloc);

        for ni in nb.iter_instr_list() {
            unsafe {
                bak_print_instr(ni);
            }
            match ni.type_ {
                nir_instr_type_alu => {
                    println!("  alu instruction");
                },
                nir_instr_type_deref => {
                    unsafe {
                        bak_print_defer_instr(ni);
                    }
                    println!("  deref instruction");
                },
                nir_instr_type_jump => {
                    println!("  jump instruction");
                },
                nir_instr_type_tex => {
                    println!("  tex instruction");
                },
                nir_instr_type_intrinsic => {
                    println!("  intrinsic instruction");
                },
                nir_instr_type_load_const => {
                    self.parse_load_const(&mut b, ni.as_load_const().unwrap());
                },
                nir_instr_type_undef => {
                    println!("  type undef instruction");
                },
                nir_instr_type_phi => {
                    println!("  phi instruction");
                },
                _ => {
                    println!("  {} instruction", ni.type_);
                    panic!("Unsupported instruction type")
                }
              }
        }

        let bb = BasicBlock {
            label: self.get_block_label(nb),
            instrs: b.as_vec(),
        };
        self.cfg.add_node(nb.index, bb);
    }

    fn parse_cf_list(
        &mut self,
        ssa_alloc: &mut SSAValueAllocator,
        phi_map: &mut PhiAllocMap,
        list: ExecListIter<nir_cf_node>,
    ) {
        println!("ShaderFromNir::parse_cf_list");
        for node in list {
            match node.type_ {
                nir_cf_node_block => {
                    println!("node block");
                    let nb = node.as_block().unwrap();
                    self.parse_block(ssa_alloc, phi_map, nb);
                }
                nir_cf_node_if => {
                    println!("node if");
                    //let ni = node.as_if().unwrap();
                    //self.parse_if(ssa_alloc, phi_map, ni);
                }
                nir_cf_node_loop => {
                    println!("node loop");
                    //let nl = node.as_loop().unwrap();
                    //self.parse_loop(ssa_alloc, phi_map, nl);
                }
                _ => panic!("Invalid inner CF node type"),
            }
        }
    }

    pub fn parse_function_impl(&mut self, nfi: &nir_function_impl) -> Function {
        println!("ShaderFromNir::parse_function_impl!");

        let mut ssa_alloc = SSAValueAllocator::new();
        let _end_nb = nfi.end_block();
        //self.end_block_id = end_nb.index;

        let mut phi_alloc = PhiAllocator::new();
        let mut phi_map = PhiAllocMap::new(&mut phi_alloc);

        self.parse_cf_list(&mut ssa_alloc, &mut phi_map, nfi.iter_body());

        let f = Function {
            // TODO
        };
        f
    }


    pub fn parse_shader(mut self) -> Shader {
        println!("ShaderFromNir::parse_shader!");
        let mut functions = Vec::new();
        for nf in self.nir.iter_functions() {
            if let Some(nfi) = nf.get_impl() {
                let f = self.parse_function_impl(nfi);
                functions.push(f);
            }
        }

        Shader {
            functions: functions,
        }
    }

    fn set_ssa(&mut self, def: &nir_def, vec: Vec<SSAValue>) {
        if def.bit_size == 1 {
            //for s in &vec {
                // TODO assert!(s.is_predicate());
            //}
        } else {
            //for s in &vec {
                // TODO assert!(!s.is_predicate());
            //}
            let bits =
                usize::from(def.bit_size) * usize::from(def.num_components);
            assert!(vec.len() == bits.div_ceil(32));
        }
        self.ssa_map
            .entry(def.index)
            .and_modify(|_| panic!("Cannot set an SSA def twice"))
            .or_insert(vec);
    }
}

pub fn bak_shader_from_nir(ns: &nir_shader) -> Shader {
    println!("bak_shader_from_nir");
    ShaderFromNir::new(ns).parse_shader()
}
