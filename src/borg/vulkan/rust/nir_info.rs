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
pub fn ubo_mask(nir: &nir_shader) -> u128 {
    mask_of(nir, |_| true)
}

/// The uniform buffers read at a run-time offset: through a memory window, not the uniform pages.
pub fn ubo_window_mask(nir: &nir_shader) -> u128 {
    mask_of(nir, |ld| ld.get_src(1).comp_as_uint(0).is_none())
}

fn mask_of(nir: &nir_shader, want: impl Fn(&nir_intrinsic_instr) -> bool) -> u128 {
    let mut mask = 0;
    for ld in intrinsics(nir).filter(|i| i.intrinsic == nir_intrinsic_load_ubo && want(i)) {
        let mut d = Some(ld.get_src(0).as_def());
        for _ in 0..8 {
            let Some(def) = d else { break };
            let di = def.parent_instr();
            if let Some(alu) = di.as_alu() {
                d = Some(alu.get_src(0).src.as_def());
            } else if let Some(r) = di.as_intrinsic() {
                if r.intrinsic == nir_intrinsic_vulkan_resource_index {
                    mask |= 1u128.wrapping_shl(r.get_const_index(NIR_INTRINSIC_BINDING));
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

/// A variable's varying slots, as a mask from VARYING_SLOT_VAR0.
fn slots(var: &nir_variable) -> u64 {
    let first = var.data.location as u32 - VARYING_SLOT_VAR0;
    let n = unsafe { glsl_count_attribute_slots(var.type_, false) };
    (0..n).fold(0, |m, k| m | 1u64.checked_shl(first + k).unwrap_or(0))
}

/// The varying slots a fragment shader reads flat.
pub fn flat_inputs(nir: &nir_shader) -> u64 {
    nir.iter_variables()
        .filter(|v| v.data.mode() == nir_var_shader_in && v.data.location >= VARYING_SLOT_VAR0 as i32 && v.data.interpolation() == INTERP_MODE_FLAT)
        .fold(0, |m, v| m | slots(v))
}

/// A vertex shader's outputs flat exactly where the fragment shader reads them flat.
pub fn link_flat(nir: *mut nir_shader, flat: u64) {
    for var in unsafe { &*nir }.iter_variables() {
        if var.data.mode() == nir_var_shader_out && var.data.location >= VARYING_SLOT_VAR0 as i32 {
            let s = slots(var);
            let mode = if s != 0 && flat & s == s { INTERP_MODE_FLAT } else { INTERP_MODE_NONE };
            unsafe { (*std::ptr::from_ref(var).cast_mut()).data.set_interpolation(mode) };
        }
    }
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
