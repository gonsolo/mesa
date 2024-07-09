// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::nir::*;

use bak_bindings::*;

struct ShaderFromNir<'a> {
    nir: &'a nir_shader,
}

impl<'a> ShaderFromNir<'a> {

    fn new(nir: &'a nir_shader) -> Self {
        Self {
            nir: nir,
        }
    }

    fn parse_cf_list(
        &mut self,
        list: ExecListIter<nir_cf_node>,
    ) {
        println!("ShaderFromNir::parse_cf_list");
        for node in list {
            match node.type_ {
                nir_cf_node_block => {
                    println!("node block");
                    //let nb = node.as_block().unwrap();
                    //self.parse_block(ssa_alloc, phi_map, nb);
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

        self.parse_cf_list(nfi.iter_body());

        let mut f = Function {
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
}

pub fn bak_shader_from_nir(ns: &nir_shader) -> Shader {
    println!("bak_shader_from_nir");
    ShaderFromNir::new(ns).parse_shader()
}

