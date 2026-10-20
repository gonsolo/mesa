// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! What a shader's NIR uses, for the pipeline: uniform buffers, push constants, the sampled
//! binding, point coordinates.

use compiler::bindings::*;
use compiler::nir::*;

fn instrs(nir: &nir_shader) -> impl Iterator<Item = &nir_instr> {
    nir.iter_functions()
        .filter_map(|f| f.get_impl())
        .flat_map(|i| i.iter_blocks())
        .flat_map(|b| b.iter_instr_list())
}

fn intrinsics(nir: &nir_shader) -> impl Iterator<Item = &nir_intrinsic_instr> {
    instrs(nir).filter_map(|i| i.as_intrinsic())
}

/// The uniform-buffer bindings the shader loads from, as a bit mask: each load_ubo's resource
/// index, found within 8 steps back through ALU ops and descriptor loads.
pub fn ubo_mask(nir: &nir_shader) -> u32 {
    let mut mask = 0;
    for ld in intrinsics(nir).filter(|i| i.intrinsic == nir_intrinsic_load_ubo) {
        let mut d = Some(ld.get_src(0).as_def());
        for _ in 0..8 {
            let Some(def) = d else { break };
            let di = def.parent_instr();
            if let Some(alu) = di.as_alu() {
                d = Some(alu.get_src(0).src.as_def());
            } else if let Some(r) = di.as_intrinsic() {
                if r.intrinsic == nir_intrinsic_vulkan_resource_index {
                    mask |= 1u32.wrapping_shl(r.get_const_index(NIR_INTRINSIC_BINDING));
                    break;
                }
                d = (r.intrinsic == nir_intrinsic_load_vulkan_descriptor).then(|| r.get_src(0).as_def());
            } else {
                break;
            }
        }
    }
    mask
}

pub fn uses_push_constants(nir: &nir_shader) -> bool {
    intrinsics(nir).any(|i| i.intrinsic == nir_intrinsic_load_push_constant)
}

pub fn reads_point_coord(nir: &nir_shader) -> bool {
    intrinsics(nir).any(|i| i.intrinsic == nir_intrinsic_load_point_coord)
}

/// The binding of the first texture the shader samples through a deref; -1 without one.
pub fn sampled_binding(nir: &nir_shader) -> i32 {
    for tex in instrs(nir).filter_map(|i| i.as_tex()) {
        let di = unsafe { nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) };
        if di < 0 {
            continue;
        }
        let deref = unsafe { nir_src_as_deref(tex.get_src(di as usize).src) };
        if let Some(var) = unsafe { nir_deref_instr_get_variable(deref).as_ref() } {
            return var.data.binding as i32;
        }
    }
    -1
}
