// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::from_nir::*;

extern crate bak_bindings;
use bak_bindings::*;

#[repr(C)]
struct ShaderBin {
    code: Vec<u32>
}

impl ShaderBin {
    pub fn new(code: Vec<u32>) -> ShaderBin {

        // TODO

        ShaderBin {
            code: code
        }
    }
}

#[no_mangle]
pub extern "C" fn bak_compile_shader(
    nir: *mut nir_shader
) -> *mut bak_shader_bin {

    let nir = unsafe { &*nir };

    let s = bak_shader_from_nir(nir);
    let code = s.encode();
    // TODO

    let bin = Box::new(ShaderBin::new(code));
    Box::into_raw(bin) as *mut bak_shader_bin
}

#[no_mangle]
pub extern "C" fn bak_nir_options(
    bak: *const bak_compiler,
) -> *const nir_shader_compiler_options {
    assert!(!bak.is_null());
    let bak = unsafe { &*bak };
    &bak.nir_options
}

fn nir_options() -> nir_shader_compiler_options {
    let op: nir_shader_compiler_options = unsafe { std::mem::zeroed() };

    // TODO

    op
}   

#[no_mangle]
pub extern "C" fn bak_compiler_create() -> *mut bak_compiler {

    let bak = Box::new(bak_compiler {
        nir_options: nir_options(),
    });

    Box::into_raw(bak)
}

