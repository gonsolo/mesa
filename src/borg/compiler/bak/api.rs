// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use bak_bindings::*;

#[repr(C)]
struct ShaderBin {
}

impl ShaderBin {
    pub fn new() -> ShaderBin {

        // TODO

        ShaderBin {}
    }
}

#[no_mangle]
pub extern "C" fn bak_compile_shader(
    _nir: *mut nir_shader
) -> *mut bak_shader_bin {

    // TODO

    let bin = Box::new(ShaderBin::new());
    Box::into_raw(bin) as *mut bak_shader_bin
}
