// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! NIR passes the driver runs before borgc.

use crate::compiler::{compile_layout, set_base, set_map};
use crate::device::Scratch;
use compiler::bindings::*;

/// Descriptor (set, binding, array element) to the flat slot borgc addresses: set 0, binding
/// base of the set plus the element's slot.
pub fn remap_array_bindings(sc: &Scratch, nir: &nir_shader) {
    let Some(pl) = compile_layout(sc) else {
        return;
    };
    let base = set_base(sc);
    let intrs = nir.iter_functions().filter_map(|f| f.get_impl()).flat_map(|i| i.iter_blocks()).flat_map(|b| b.iter_instr_list());
    for intr in intrs.filter_map(|i| i.as_intrinsic()) {
        if intr.intrinsic != nir_intrinsic_vulkan_resource_index {
            continue;
        }
        let (set, binding) = (intr.get_const_index(NIR_INTRINSIC_DESC_SET), intr.get_const_index(NIR_INTRINSIC_BINDING));
        if set >= pl.set_count || set >= 4 {
            continue;
        }
        let Some(m) = set_map(pl, set as usize) else {
            continue;
        };
        let src = *intr.get_src(0);
        let is_const = unsafe { nir_src_is_const(src) };
        let elem = if is_const { (unsafe { nir_src_as_uint(src) }) as u32 } else { 0 };
        let slot = (0..m.n as usize)
            .find(|&j| m.num[j] as u32 == binding)
            .map_or(binding, |j| m.slot[j] as u32 + elem.min((m.count[j] as u32).wrapping_sub(1)));
        let intr = intr as *const nir_intrinsic_instr as *mut nir_intrinsic_instr;
        unsafe {
            nir_intrinsic_set_binding(intr, base[set as usize] as u32 + slot);
            nir_intrinsic_set_desc_set(intr, 0);
        }
    }
}

/// Lines as quads: the vertex shader runs a second time for the line's other end, in a copy of
/// itself, and moves its position along the minor axis by `line_step`, the half width in clip
/// units. Inputs of per-vertex attributes come from BORGVK_LINE_OTHER slots in the copy.
pub fn lower_lines(nir: *mut nir_shader, instance_mask: u32) {
    use crate::ffi::{
        nir_after_impl, nir_bcsel, nir_before_impl, nir_build_call, nir_builder_at, nir_channel, nir_channels,
        nir_fabs, nir_fadd, nir_fdiv, nir_fge, nir_fmul, nir_fsub, nir_function_set_impl, nir_imm_float,
        nir_load_var, nir_progress, nir_store_var, nir_trim_vector, nir_vec4, BORGVK_LINE_OTHER, BORGVK_LINE_SLOT,
    };
    unsafe {
        let pos = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_POS);
        if pos.is_null() {
            return;
        }
        // One function, so the copy holds every use of the inputs and of gl_Position.
        nir_lower_variable_initializers(nir, nir_var_function_temp);
        nir_lower_returns(nir);
        nir_inline_functions(nir);
        nir_remove_non_entrypoints(nir);
        let imp = nir_shader_get_entrypoint(nir);
        let vec4 = glsl_vec4_type();
        let own = nir_variable_create(nir, nir_var_shader_temp, vec4, c"line_own".as_ptr());
        let other = nir_variable_create(nir, nir_var_shader_temp, vec4, c"line_other".as_ptr());
        let line = nir_variable_create(nir, nir_var_shader_in, vec4, c"line_step".as_ptr());
        (*line).data.location = (VERT_ATTRIB_GENERIC0 + BORGVK_LINE_SLOT) as i32;

        let f = nir_function_create(nir, c"line_other_end".as_ptr());
        let copy = nir_function_impl_clone(nir, imp);
        nir_function_set_impl(f, copy);

        let mut remap = std::collections::HashMap::new();
        for d in derefs(copy) {
            let v = (*d).__bindgen_anon_1.var;
            if v == pos {
                (*d).__bindgen_anon_1.var = other;
                continue;
            }
            let mode = (*v).data.mode();
            let (is_in, is_out) = (mode == nir_var_shader_in, mode == nir_var_shader_out);
            let loc = (*v).data.location - VERT_ATTRIB_GENERIC0 as i32;
            let per_vertex = is_in && (0..16).contains(&loc) && instance_mask >> loc & 1 == 0;
            if !is_out && !per_vertex {
                continue;
            }
            (*d).__bindgen_anon_1.var = *remap.entry(v).or_insert_with(|| {
                let n = nir_variable_create(nir, if is_in { nir_var_shader_in } else { nir_var_shader_temp }, (*v).type_, c"line_end".as_ptr());
                if is_in {
                    (*n).data = (*v).data;
                    (*n).data.location = (*v).data.location + BORGVK_LINE_OTHER as i32;
                }
                n
            });
        }
        for d in derefs(imp) {
            if (*d).__bindgen_anon_1.var == pos {
                (*d).__bindgen_anon_1.var = own;
            }
        }

        let mut b = nir_builder_at(nir_before_impl(imp));
        nir_build_call(&mut b, f, 0, std::ptr::null_mut());
        b.cursor = nir_after_impl(imp);
        let (a, o, l) = (nir_load_var(&mut b, own), nir_load_var(&mut b, other), nir_load_var(&mut b, line));
        // The builder calls in the order GCC emitted the C version's nested arguments: right to
        // left, so the instructions come out in the same order.
        let aw = nir_channel(&mut b, a, 3);
        let ta = nir_trim_vector(&mut b, a, 2);
        let na = nir_fdiv(&mut b, ta, aw);
        let ow = nir_channel(&mut b, o, 3);
        let to = nir_trim_vector(&mut b, o, 2);
        let nb = nir_fdiv(&mut b, to, ow);
        let step = nir_channels(&mut b, l, 0xc);
        let diff = nir_fsub(&mut b, nb, na);
        let d = nir_fmul(&mut b, diff, step);
        let dy = nir_channel(&mut b, d, 1);
        let ady = nir_fabs(&mut b, dy);
        let dx = nir_channel(&mut b, d, 0);
        let adx = nir_fabs(&mut b, dx);
        let xmajor = nir_fge(&mut b, adx, ady);
        let zero = nir_imm_float(&mut b, 0.0);
        let lx = nir_channel(&mut b, l, 0);
        let sx = nir_bcsel(&mut b, xmajor, zero, lx);
        let ly = nir_channel(&mut b, l, 1);
        let sy = nir_bcsel(&mut b, xmajor, ly, zero);
        let az = nir_channel(&mut b, a, 2);
        let my = nir_fmul(&mut b, sy, aw);
        let ay = nir_channel(&mut b, a, 1);
        let py = nir_fadd(&mut b, ay, my);
        let mx = nir_fmul(&mut b, sx, aw);
        let ax = nir_channel(&mut b, a, 0);
        let px = nir_fadd(&mut b, ax, mx);
        let out = nir_vec4(&mut b, px, py, az, aw);
        nir_store_var(&mut b, pos, out, 0xf);
        nir_progress(true, imp, nir_metadata_none);
        nir_fixup_deref_modes(nir);
        nir_inline_functions(nir);
        nir_remove_non_entrypoints(nir);
    }
}

/// The variable derefs of a function, as pointers to change.
unsafe fn derefs(imp: *mut nir_function_impl) -> Vec<*mut nir_deref_instr> {
    let imp = unsafe { &*imp };
    imp.iter_blocks()
        .flat_map(|b| b.iter_instr_list())
        .filter(|i| i.type_ == nir_instr_type_deref)
        .map(|i| unsafe { nir_instr_as_deref(i as *const nir_instr as *mut nir_instr) })
        .filter(|&d| unsafe { (*d).deref_type } == nir_deref_type_var)
        .collect()
}
