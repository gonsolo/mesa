// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ir::*;

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

    pub fn parse_shader(mut self) -> Shader {
        let mut functions = Vec::new();
        // TODO 
        Shader {
            functions: functions,
        }
    }

}

pub fn bak_shader_from_nir(ns: &nir_shader) -> Shader {
    ShaderFromNir::new(ns).parse_shader()
}

