// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! NIR passes the driver runs before borgc.

use crate::compiler::{compile_layout, set_base, set_map};
use crate::ffi::borgvk_device;
use compiler::bindings::*;

/// Descriptor (set, binding, array element) to the flat slot borgc addresses: set 0, binding
/// base of the set plus the element's slot.
pub fn remap_array_bindings(dev: &borgvk_device, nir: &nir_shader) {
    let Some(pl) = compile_layout(dev) else {
        return;
    };
    let base = set_base(dev);
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

/// # Safety
/// `device` and `nir` are valid.
#[no_mangle]
pub unsafe extern "C" fn borgvk_nir_remap_array_bindings(device: *const borgvk_device, nir: *mut nir_shader) {
    remap_array_bindings(unsafe { &*device }, unsafe { &*nir });
}
