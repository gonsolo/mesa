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

    pub fn parse_function_impl(&mut self, nfi: &nir_function_impl) -> Function {

        println!("ShaderFromNir::parse_function_impl!");
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

