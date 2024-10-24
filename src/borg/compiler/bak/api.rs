// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT


use crate::from_nir::*;
use crate::sm::*;

use compiler::bindings::*;
use bak_bindings::*;

use std::os::raw::c_void;

#[repr(C)]
struct ShaderBin {
    bin: bak_shader_bin,
    code: Vec<u32>
}

impl ShaderBin {
    pub fn new(code: Vec<u32>) -> ShaderBin {
        let bin = bak_shader_bin {
            code_size: (code.len() * 4).try_into().unwrap(),
            code: code.as_ptr() as *const c_void
        };

        ShaderBin {
            bin: bin,
            code: code
        }
    }
}

#[no_mangle]
pub extern "C" fn bak_compile_shader(
    nir: *mut nir_shader,
    bak: *const bak_compiler,
) -> *mut bak_shader_bin {

    println!("bak_compile_shader");

    unsafe { bak_postprocess_nir(nir, bak) };
    let nir = unsafe { &*nir };

    let sm = ShaderModel {};
    let s = bak_shader_from_nir(nir);

    eprintln!("BAK IR:\n{}", &s);

    // TODO: passes

    let code = sm.encode_shader(&s);

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

    println!("bak_compiler_create");

    let bak = Box::new(bak_compiler {
        nir_options: nir_options(),
    });

    Box::into_raw(bak)
}

