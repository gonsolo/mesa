// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Pipeline stages through SPIR-V to NIR, Borg's NIR passes and borgc.

use crate::compiler::set_base;
use crate::ffi::*;
use crate::device::{Device, Scratch};
use crate::pipeline::Pipeline;
use crate::nir_info::*;
use crate::nir_passes::{lower_lines, remap_array_bindings};
use crate::shader_cache;
use crate::util::{log, MESA_LOG_INFO, MESA_LOG_WARN};
use borgc_rs::{borgc_compile_compute, borgc_compile_nir, BorgcInfo};
use std::ptr::{null, null_mut};

const VERT: usize = BORGVK_STAGE_VERT as usize;
const FRAG: usize = BORGVK_STAGE_FRAG as usize;
const BLOB_MAX: u32 = BORGVK_SHADER_BLOB_MAX;

fn to_nir(dev: &Device, si: &VkPipelineShaderStageCreateInfo) -> Result<*mut nir_shader, VkResult> {
    let mut nir = null_mut();
    let r = unsafe {
        vk_pipeline_shader_stage_to_nir(dev.vk_ptr(), 0, si, &borg_spirv_options, &borg_nir_options, null_mut(), &mut nir)
    };
    if r != VK_SUCCESS || nir.is_null() { Err(r) } else { Ok(nir) }
}

fn empty_info() -> BorgcInfo {
    BorgcInfo { pool: [0; 64], npool: 0, spill_slots: 0, push_window: 0 }
}

/// A pipeline without a fragment stage (depth only): a shader that writes nothing.
pub fn compile_empty_frag(sc: &mut Scratch, opt: u32) {
    let b = &mut sc.shader_blob[FRAG];
    let nb = unsafe { nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &borg_nir_options, c"borgvk_empty_fs".as_ptr()) };
    let mut len = 0;
    unsafe {
        borg_lower_nir_for_borgc(nb.shader);
        borgc_compile_nir(nb.shader, b.data.as_mut_ptr(), BLOB_MAX, &mut len, opt, -1, null_mut());
    }
    b.len = if len <= BLOB_MAX { len } else { 0 };
    b.npool = 0;
    sc.frag_reads_pntc = false;
    sc.frag_tex_binding = -1;
    sc.stage_ubo[1] = 0;
    sc.stage_push = [false; 2];
    sc.stage_pwin[1] = false;
    sc.stage_spill[1] = false;
    unsafe { ralloc_free(nb.shader.cast()) };
}

/// The blob from the shader cache, which knows neither the blend state nor a constant pool.
fn from_cache(sc: &mut Scratch, si: &VkPipelineShaderStageCreateInfo, vfetch: u32, slot: usize) -> bool {
    let Some(pntc) = shader_cache::get(si, vfetch, &mut sc.shader_blob[slot]) else {
        return false;
    };
    let fs = slot == FRAG;
    if fs {
        sc.frag_reads_pntc = pntc;
        sc.frag_tex_binding = -1;
    }
    sc.stage_ubo[fs as usize] = 0;
    sc.stage_push[fs as usize] = false;
    sc.stage_pwin[fs as usize] = false;
    sc.stage_spill[fs as usize] = false;
    sc.shader_blob[slot].npool = 0;
    log(MESA_LOG_INFO, format!("borgvk: shader blob from the cache ({} bytes)", sc.shader_blob[slot].len));
    true
}

pub fn compile_stage(dev: &Device, sc: &mut Scratch, vfetch: u32, si: &VkPipelineShaderStageCreateInfo) {
    let (vs, fs) = (si.stage == VK_SHADER_STAGE_VERTEX_BIT, si.stage == VK_SHADER_STAGE_FRAGMENT_BIT);
    let slot = if vs { Some(VERT) } else if fs { Some(FRAG) } else { None };
    let blend = if fs { sc.blend_lower } else { null() };
    if let Some(slot) = slot.filter(|_| blend.is_null()) {
        if from_cache(sc, si, vfetch, slot) {
            return;
        }
    }
    let nir = match to_nir(dev, si) {
        Ok(nir) => nir,
        Err(r) => {
            log(MESA_LOG_WARN, format!("borgvk: SPIR-V→NIR failed for stage 0x{:x} ({r})", si.stage));
            return;
        }
    };
    remap_array_bindings(sc, unsafe { &*nir });
    let lines = BORGC_VS_FETCH | BORGC_VS_POINT_QUADS | BORGC_VS_LINES_BIT;
    if vs && vfetch & lines == BORGC_VS_FETCH | BORGC_VS_LINES_BIT {
        lower_lines(nir, vfetch & 0xFFFF);
    }
    let retry = if vs { unsafe { nir_shader_clone(null_mut(), nir) } } else { null_mut() };
    unsafe { borg_lower_nir_for_borgc_blend(nir, blend) };
    let n = unsafe { &*nir };
    sc.frag_reads_pntc = fs && reads_point_coord(n);
    if fs {
        sc.frag_tex_binding = sampled_binding(n);
    }
    if vs || fs {
        sc.stage_ubo[fs as usize] = ubo_mask(n);
        sc.stage_push[fs as usize] = uses_push_constants(n);
    }
    match slot {
        Some(slot) => compile_blob(sc, si, vfetch, slot, nir, retry, blend),
        None => unsafe {
            borgc_compile_nir(nir, null_mut(), 0, null_mut(), 0, -1, null_mut()); // diagnostics only
        },
    }
    unsafe {
        ralloc_free(retry.cast());
        ralloc_free(nir.cast());
    }
}

/// borgc's blob for `slot`; a vertex shader that does not fit is retried with rematerialized
/// values from `retry`, its unlowered clone.
fn compile_blob(
    sc: &mut Scratch,
    si: &VkPipelineShaderStageCreateInfo,
    vfetch: u32,
    slot: usize,
    nir: *mut nir_shader,
    retry: *mut nir_shader,
    blend: *const borg_blend,
) {
    let fs = slot == FRAG;
    let out_select = if fs { sc.frag_select as i32 - 1 } else { -1 };
    let b = &mut sc.shader_blob[slot];
    let (mut len, mut info) = (0, empty_info());
    unsafe { borgc_compile_nir(nir, b.data.as_mut_ptr(), BLOB_MAX, &mut len, vfetch, out_select, &mut info) };
    if len == 0 && !retry.is_null() {
        unsafe {
            borg_lower_set_vertex_remat(true);
            borg_lower_nir_for_borgc_blend(retry, blend);
            borg_lower_set_vertex_remat(false);
            borgc_compile_nir(retry, b.data.as_mut_ptr(), BLOB_MAX, &mut len, vfetch, out_select, &mut info);
        }
    }
    let kind = if fs { "fragment" } else { "vertex" };
    if len > 0 && len <= BLOB_MAX {
        b.len = len;
        b.npool = info.npool.min(BORGVK_POOL_MAX);
        let n = b.npool as usize;
        b.pool[..n].copy_from_slice(&info.pool[..n]);
        if blend.is_null() && n == 0 {
            shader_cache::record(si, vfetch, b, sc.frag_reads_pntc);
        }
        log(MESA_LOG_INFO, format!("borgvk: captured {kind} shader blob ({len} bytes) for upload"));
    } else {
        b.len = 0;
        log(MESA_LOG_WARN, format!("borgvk: {kind} shader blob not captured (size {len}, max {BLOB_MAX})"));
    }
    sc.stage_pwin[fs as usize] = info.push_window != 0;
    sc.stage_spill[fs as usize] = info.spill_slots > 0;
}

/// Compute pipeline: SPIR-V to NIR to borgc's compute backend.
pub fn compile_compute_stage(dev: &Device, sc: &mut Scratch, si: &VkPipelineShaderStageCreateInfo, p: &mut Pipeline) {
    let nir = match to_nir(dev, si) {
        Ok(nir) => nir,
        Err(r) => {
            log(MESA_LOG_WARN, format!("borgvk: SPIR-V->NIR failed for compute stage ({r})"));
            return;
        }
    };
    remap_array_bindings(sc, unsafe { &*nir });
    let rc = unsafe {
        borg_lower_nir_for_borgc(nir);
        borgc_compile_compute(
            nir,
            p.cs_words.as_mut_ptr(), BORGVK_CS_WORDS_MAX, &mut p.cs_nwords,
            p.cs_regs.as_mut_ptr(), 32, &mut p.cs_nregs,
            p.cs_local.as_mut_ptr(), &mut p.cs_flags,
            p.cs_pool.as_mut_ptr(), BORGVK_CS_POOL_MAX, &mut p.cs_npool,
        )
    };
    p.cs_ok = rc == 0;
    p.set_base = set_base(sc);
    let what = if rc == 0 { "compiled" } else { "REFUSED" };
    log(MESA_LOG_INFO, format!("borgvk: compute shader {what} ({} words, {} presets)", p.cs_nwords, p.cs_nregs));
    unsafe { ralloc_free(nir.cast()) };
}
