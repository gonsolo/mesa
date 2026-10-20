// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Shaders for pipelines: from the shader cache, or through NIR and borgc.

use crate::descriptor_set::DescriptorSetLayout;
use crate::ffi::*;
use crate::device::Device;
#[cfg(not(borgc))]
use crate::pipeline::Pipeline;
#[cfg(borgc)]
pub use crate::compile::{borgvk_compile_compute_stage, borgvk_compile_empty_frag, borgvk_compile_stage};

/// The number of varyings a vertex shader blob writes, from its header; 0 without the flag.
pub fn num_varyings(b: &[u8]) -> u32 {
    if b.len() < 6 || b[5] & 1 == 0 {
        return 0;
    }
    let at = 6 + 4 * b[0] as usize + b[3] as usize + 5 * b[4] as usize;
    b.get(at).map_or(0, |&n| n as u32)
}

/// The pipeline layout being compiled for, from its handle.
#[cfg(borgc)]
pub fn compile_layout(dev: &Device) -> Option<&vk_pipeline_layout> {
    unsafe { (dev.compile_layout as *const vk_pipeline_layout).as_ref() }
}

/// The map of set `s` of a layout: its bindings' flat slots.
#[cfg(borgc)]
pub fn set_map(pl: &vk_pipeline_layout, s: usize) -> Option<&borgvk_desc_map> {
    let l = pl.set_layouts[s];
    if l.is_null() {
        return None;
    }
    // container_of
    let l = unsafe { (l as *const u8).sub(std::mem::offset_of!(DescriptorSetLayout, vk)) };
    Some(unsafe { &(*l.cast::<DescriptorSetLayout>()).map })
}

/// The first flat descriptor slot of each of the first four sets.
#[cfg(borgc)]
pub fn set_base(dev: &Device) -> [u8; 4] {
    let mut base = [0u8; 4];
    let Some(pl) = compile_layout(dev) else {
        return base;
    };
    let mut at = 0u32;
    for (s, b) in base.iter_mut().enumerate().take(pl.set_count as usize) {
        *b = at.min(255) as u8;
        if let Some(m) = set_map(pl, s) {
            at += m.total as u32;
        }
    }
    base
}

/// Without the compiler the shaders come from the cache, which numbers slots per set.
#[cfg(not(borgc))]
pub fn set_base(_dev: &Device) -> [u8; 4] {
    [0; 4]
}

/// # Safety
/// `device` and `stage_info` are valid.
#[cfg(not(borgc))]
pub unsafe fn borgvk_compile_stage(device: *mut Device, vfetch: u32, stage_info: *const VkPipelineShaderStageCreateInfo) {
    let (dev, si) = unsafe { (&mut *device, &*stage_info) };
    let slot = match si.stage {
        VK_SHADER_STAGE_VERTEX_BIT => BORGVK_STAGE_VERT,
        VK_SHADER_STAGE_FRAGMENT_BIT => BORGVK_STAGE_FRAG,
        _ => return,
    } as usize;
    if let Some(pntc) = crate::shader_cache::get(si, vfetch, &mut dev.shader_blob[slot]) {
        if slot == BORGVK_STAGE_FRAG as usize {
            dev.frag_reads_pntc = pntc;
        }
        return;
    }
    dev.shader_blob[slot].len = 0;
    crate::util::log(crate::util::MESA_LOG_WARN, "borgvk: shader not in the cache and no compiler in this build".into());
}

/// # Safety
/// `pipeline` is valid.
#[cfg(not(borgc))]
pub unsafe fn borgvk_compile_compute_stage(
    _device: *mut Device,
    _stage_info: *const VkPipelineShaderStageCreateInfo,
    pipeline: *mut Pipeline,
) {
    unsafe { (*pipeline).cs_ok = false };
}

/// # Safety
/// `device` is valid.
#[cfg(not(borgc))]
pub unsafe fn borgvk_compile_empty_frag(device: *mut Device, _opt: u32) {
    unsafe { (*device).shader_blob[BORGVK_STAGE_FRAG as usize].len = 0 };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varyings_follow_the_header_tables() {
        let mut b = vec![0u8; 32];
        b[0] = 1; // one word of something
        b[3] = 2;
        b[4] = 1;
        b[5] = 1;
        b[6 + 4 + 2 + 5] = 7;
        assert_eq!(num_varyings(&b), 7);
        b[5] = 0;
        assert_eq!(num_varyings(&b), 0);
        assert_eq!(num_varyings(&b[..4]), 0);
    }
}
