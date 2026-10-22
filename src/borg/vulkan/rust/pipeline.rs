// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Graphics and compute pipelines: the stages compiled with the fetch, target and blend options
//! of the pipeline, and the state the draw and the hardware registers take from it.

use crate::compiler::{compile_compute_stage, compile_empty_frag, compile_stage, num_varyings, set_base};
use crate::entrypoint::{entrypoint, handle, slice, vk_error, Handle, VkError};
use crate::ffi::*;
use crate::vk::*;
use crate::fmt;
use crate::device::{Device, Scratch};
use crate::object::{allocate, Owned, Zeroed};
use crate::format::{bytes_packed, half1, half2, packed_info};
use crate::wire::{blend_pack, state_pack};
use std::ptr::{null, null_mut};

handle!(VkPipeline, Pipeline, VK_OBJECT_TYPE_PIPELINE);

#[repr(C)]
pub struct Pipeline {
    pub base: vk_object_base,
    pub topology: VkPrimitiveTopology,
    pub binding_stride: [u32; VB],
    pub attr_count: u32,
    pub attrs: [VertexAttr; BORGVK_MAX_VERTEX_ATTRS as usize],
    pub binding_instance: [bool; VB], // per-instance rate
    pub restart: bool,                // primitiveRestartEnable
    pub color_count: u32,             // colour attachments of the pipeline's render pass
    pub mrt: bool,
    pub ubo_mask: u32,       // the uniform buffer bindings the draw loads from
    pub push_used: bool,     // a shader loads push constants
    pub emul_n: u8,          // wide colour targets: this and emul[] draw one each
    pub emul: [VkPipeline; 3],
    pub push_window: bool,   // a stage reads push constants from the heap window
    pub spills: bool,        // a stage spills to the region's scratch slot
    pub push_stage: u8,      // bit per stage: loads push constants, no UBO
    pub set_base: [u8; 4],
    pub tex_binding: i32,    // binding of the texture the draw samples (fragment, else vertex shader)
    pub tex_pairs: [[u8; 2]; BORGVK_TEX_PAIRS], // texture and sampler slots of both stages
    pub tex_npairs: u8,
    pub vfetch: u32,
    pub blob: [ShaderBlob; BORGVK_SHADER_STAGE_COUNT as usize],
    pub blend_cfg: u32,
    pub blend_const: u32,
    pub state_reg: [u32; 5],
    pub vp: Vp,       // static viewport/scissor, if the pipeline has them
    pub vp_dyn: bool,        // taken from the command buffer
    pub sc_dyn: bool,
    pub srgb_table: bool,    // the fragment shader decodes the destination
    pub state_valid: bool,
    pub cs_ok: bool,
    pub cs_tex: i32,         // slot of the texture the compute shader filters, or -1
    pub cs_nwords: u32,
    pub cs_nregs: u32,
    pub cs_words: [u32; BORGVK_CS_WORDS_MAX as usize],
    pub cs_regs: [u32; 64],
    pub cs_local: [u32; 3],
    pub cs_flags: u32,
    pub cs_pool: [u32; BORGVK_CS_POOL_MAX as usize],
    pub cs_npool: u32,
}

const VB: usize = BORGVK_MAX_VERTEX_BINDINGS as usize;

// Numbers, handles, the shader blobs and the runtime's object base.
unsafe impl Zeroed for Pipeline {}

const VERT: usize = BORGVK_STAGE_VERT as usize;

fn create(dev: &Device, alloc: *const VkAllocationCallbacks) -> Result<&'static mut Pipeline, VkError> {
    let p = unsafe { allocate::<Pipeline>(|n| vk_object_zalloc(dev.vk_ptr(), alloc, n, VK_OBJECT_TYPE_PIPELINE)) };
    p.ok_or_else(|| vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY))
}

/// Vertex fetch options: fetched inputs, instance-rate locations, points as quads, lines.
fn vertex_fetch(vi: Option<&VkPipelineVertexInputStateCreateInfo>, points: bool, lines: bool) -> u32 {
    let mut vfetch = 0;
    if let Some(vi) = vi.filter(|vi| vi.vertexAttributeDescriptionCount > 0) {
        vfetch = BORGC_VS_FETCH;
        let (attrs, bindings) = (vi.attributes(), vi.bindings());
        for vb in bindings.iter().filter(|vb| vb.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE) {
            for va in attrs.iter().filter(|va| va.binding == vb.binding && va.location < 16) {
                vfetch |= 1 << va.location;
            }
        }
    }
    if points {
        vfetch |= BORGC_VS_FETCH | BORGC_VS_POINT_QUADS;
    }
    if lines {
        vfetch |= BORGC_VS_FETCH | BORGC_VS_LINES_BIT;
    }
    vfetch
}

/// Two to four R8 targets drawn by one fragment shader, and their class bits.
fn mrt(formats: &[VkFormat]) -> (bool, u32) {
    let (mut mrt, mut opt) = ((2..=4).contains(&formats.len()), 0);
    for (k, &f) in formats.iter().enumerate() {
        if !mrt {
            break;
        }
        let cls = match f {
            VK_FORMAT_R8_UNORM => 1,
            VK_FORMAT_R8_SNORM => 2,
            VK_FORMAT_R8_UINT | VK_FORMAT_R8_SINT => 3,
            _ => 0,
        };
        mrt = cls != 0;
        opt |= cls << (BORGC_FS_TARGET_CLASS_SHIFT + 3 * k as u32);
    }
    (mrt, opt)
}

/// Fragment options from the colour targets.
fn frag_options(formats: &[VkFormat], mrt: bool, mrt_opt: u32) -> u32 {
    if mrt {
        return BORGC_FS_MULTI_TARGET | (formats.len() as u32 - 1) << BORGC_FS_TARGETS_SHIFT | mrt_opt;
    }
    let Some(&f) = formats.first() else {
        return 0;
    };
    let packed = packed_info(f).map_or(0, |(opt, _)| opt);
    if bytes_packed(f) {
        BORGC_FS_BYTES_PACKED
    } else if f == VK_FORMAT_R8_UNORM {
        BORGC_FS_R8_UNORM
    } else if packed != 0 {
        packed
    } else if half1(f) {
        BORGC_FS_HALF | BORGC_FS_HALF1
    } else if half2(f) {
        BORGC_FS_HALF
    } else if fmt::vk_has_alpha(f) {
        BORGC_FS_ALPHA_OUTPUT
    } else {
        0
    }
}

fn blend_rt(format: pipe_format, mask: u32, rgb: borgvk_blend_channel, alpha: borgvk_blend_channel) -> borgvk_blend_rt {
    let mut rt = borgvk_blend_rt { format, rgb, alpha, ..Default::default() };
    rt.set_colormask(mask);
    rt
}

fn channel(func: pipe_blend_func, src_factor: pipe_blendfactor, dst_factor: pipe_blendfactor) -> borgvk_blend_channel {
    borgvk_blend_channel { func, src_factor, dst_factor }
}

/// The blend state the fragment shader lowers, for one target that is neither RGBA8 nor
/// BGRA8 (those blend in hardware); None when it writes every channel unblended.
fn shader_blend(cb: Option<&VkPipelineColorBlendStateCreateInfo>, f0: VkFormat, mrt: bool) -> Option<borg_blend> {
    let cb = cb?;
    let a = cb.attachments().first()?;
    if mrt || f0 == VK_FORMAT_UNDEFINED || f0 == VK_FORMAT_R8G8B8A8_UNORM || f0 == VK_FORMAT_B8G8R8A8_UNORM {
        return None;
    }
    let enable = a.blendEnable != 0 && !fmt::vk_is_int(f0);
    if !enable && a.colorWriteMask & 0xF == 0xF {
        return None;
    }
    let pass = channel(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
    let ch = |op, src, dst| unsafe { channel(vk_blend_op_to_pipe(op), vk_blend_factor_to_pipe(src), vk_blend_factor_to_pipe(dst)) };
    let mut b = borg_blend::default();
    b.opts.rt = [blend_rt(PIPE_FORMAT_NONE, 0xF, pass, pass); 8];
    b.opts.rt[0] = blend_rt(
        fmt::pipe(f0),
        a.colorWriteMask & 0xF,
        if enable { ch(a.colorBlendOp, a.srcColorBlendFactor, a.dstColorBlendFactor) } else { pass },
        if enable { ch(a.alphaBlendOp, a.srcAlphaBlendFactor, a.dstAlphaBlendFactor) } else { pass },
    );
    b.konst = cb.blendConstants;
    Some(b)
}

fn compile_stages(dev: &Device, sc: &mut Scratch, stages: &[VkPipelineShaderStageCreateInfo], vfetch: u32, frag_opt: u32, points: bool) {
    let opt = |s: &VkPipelineShaderStageCreateInfo| if s.stage == VK_SHADER_STAGE_FRAGMENT_BIT { frag_opt } else { vfetch };
    for s in stages {
        compile_stage(dev, sc, opt(s), s);
    }
    let fs = stages.iter().find(|s| s.stage == VK_SHADER_STAGE_FRAGMENT_BIT);
    if fs.is_none() {
        compile_empty_frag(sc, frag_opt);
    }
    let vs = stages.iter().filter(|s| s.stage == VK_SHADER_STAGE_VERTEX_BIT).last();
    let fs = stages.iter().filter(|s| s.stage == VK_SHADER_STAGE_FRAGMENT_BIT).last();
    // Points: gl_PointCoord as varyings past the vertex shader's own.
    if let (true, Some(vs), Some(fs)) = (points, vs, fs) {
        let b = &sc.shader_blob[VERT];
        let base = num_varyings(&b.data[..(b.len as usize).min(b.data.len())]);
        compile_stage(dev, sc, frag_opt | base << BORGC_FS_PNTC_BASE_SHIFT, fs);
        if sc.frag_reads_pntc && base + 2 <= 255 {
            compile_stage(dev, sc, vfetch | BORGC_VS_PNTC_VARYINGS, vs);
        } else {
            compile_stage(dev, sc, frag_opt, fs);
        }
    }
}

fn build_graphics(dev: &Device, sc: &mut Scratch, alloc: *const VkAllocationCallbacks, ci: &VkGraphicsPipelineCreateInfo) -> Result<VkPipeline, VkError> {
    sc.compile_layout = ci.layout;
    let vi = unsafe { ci.pVertexInputState.as_ref() };
    let ia = unsafe { ci.pInputAssemblyState.as_ref() };
    let rs = unsafe { ci.pRasterizationState.as_ref() };
    let topology = ia.map_or(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, |ia| ia.topology);
    let points = ia.is_some_and(|ia| ia.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    let lines = matches!(topology, VK_PRIMITIVE_TOPOLOGY_LINE_LIST | VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
    let vfetch = vertex_fetch(vi, points, lines);
    for b in &mut sc.shader_blob {
        b.len = 0;
    }
    sc.tex_pairs.clear();
    let ri = unsafe { vk_get_pipeline_rendering_create_info(ci).as_ref() };
    let formats = ri.map_or(&[][..], |ri| unsafe { slice(ri.pColorAttachmentFormats, ri.colorAttachmentCount) });
    let (mrt, mrt_opt) = mrt(formats);
    let frag_opt = frag_options(formats, mrt, mrt_opt);
    // Alpha to coverage takes location 0's alpha even without a target that keeps it.
    let a2c = unsafe { ci.pMultisampleState.as_ref() }.is_some_and(|ms| ms.alphaToCoverageEnable != 0);
    let frag_opt = if a2c && !mrt { frag_opt | BORGC_FS_ALPHA_OUTPUT } else { frag_opt };
    let discard = rs.is_some_and(|r| r.rasterizerDiscardEnable != 0);
    let cb = if discard { None } else { unsafe { ci.pColorBlendState.as_ref() } };
    let blend = shader_blend(cb, formats.first().copied().unwrap_or(VK_FORMAT_UNDEFINED), mrt);
    sc.blend_lower = blend.as_ref().map_or(null(), |b| b);
    let stages = unsafe { slice(ci.pStages, ci.stageCount) };
    compile_stages(dev, sc, stages, vfetch, frag_opt, points);
    sc.blend_lower = null();

    let pl = create(dev, alloc)?;
    pl.blob = sc.shader_blob;
    pl.vfetch = vfetch;
    // One texture per draw: the fragment shader's, else the vertex shader's.
    pl.tex_binding = if sc.frag_tex_binding >= 0 { sc.frag_tex_binding } else { sc.vert_tex_binding };
    pl.tex_npairs = sc.tex_pairs.len() as u8;
    pl.tex_pairs[..sc.tex_pairs.len()].copy_from_slice(&sc.tex_pairs);
    pl.set_base = set_base(sc);
    pl.ubo_mask = sc.stage_ubo[0] | sc.stage_ubo[1];
    pl.push_used = sc.stage_push[0] || sc.stage_push[1];
    pl.push_window = sc.stage_pwin[0] || sc.stage_pwin[1];
    pl.spills = sc.stage_spill[0] || sc.stage_spill[1];
    let push_only = |s: usize| (sc.stage_push[s] && sc.stage_ubo[s] == 0 && !sc.stage_pwin[s]) as u8;
    pl.push_stage = push_only(0) | push_only(1) << 1;
    pl.color_count = formats.len() as u32;
    pl.mrt = mrt;
    pl.topology = topology;
    pl.restart = ia.is_some_and(|ia| ia.primitiveRestartEnable != 0);
    if let Some(vi) = vi {
        for vb in vi.bindings() {
            if vb.binding < BORGVK_MAX_VERTEX_BINDINGS {
                pl.binding_stride[vb.binding as usize] = vb.stride;
                pl.binding_instance[vb.binding as usize] = vb.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
            }
        }
        for va in vi.attributes() {
            if pl.attr_count >= BORGVK_MAX_VERTEX_ATTRS {
                break;
            }
            pl.attrs[pl.attr_count as usize] = VertexAttr { location: va.location, binding: va.binding, offset: va.offset, format: va.format };
            pl.attr_count += 1;
        }
    }
    let mut c = *ci;
    if discard {
        (c.pViewportState, c.pMultisampleState, c.pDepthStencilState, c.pColorBlendState) = (null(), null(), null(), null());
    }
    (sc.blend_cfg, sc.blend_const) = blend_pack(unsafe { c.pColorBlendState.as_ref() });
    sc.state_reg = state_pack(&c);
    if points || lines {
        sc.state_reg[4] &= !3; // points and lines are not culled
    }
    sc.state_valid = true;
    let rt0 = blend.map_or(PIPE_FORMAT_NONE, |b| b.opts.rt[0].format);
    pl.srgb_table = rt0 != PIPE_FORMAT_NONE && fmt::is_srgb(rt0);
    (pl.blend_cfg, pl.blend_const, pl.state_reg) = (sc.blend_cfg, sc.blend_const, sc.state_reg);
    pl.state_valid = true;
    viewport(pl, &c, discard);
    Ok(VkPipeline::from_object(pl))
}

/// The static viewport and scissor; dynamic ones come from the command buffer.
fn viewport(pl: &mut Pipeline, c: &VkGraphicsPipelineCreateInfo, discard: bool) {
    (pl.vp_dyn, pl.sc_dyn) = (false, false);
    if let Some(d) = c.dynamic_state() {
        for &ds in d.dynamic_states() {
            pl.vp_dyn |= matches!(ds, VK_DYNAMIC_STATE_VIEWPORT | VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT);
            pl.sc_dyn |= matches!(ds, VK_DYNAMIC_STATE_SCISSOR | VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT);
        }
    }
    pl.vp = Vp::default();
    let Some(vs) = unsafe { c.pViewportState.as_ref() }.filter(|_| !discard) else {
        return;
    };
    if !vs.pViewports.is_null() && vs.viewportCount != 0 && !pl.vp_dyn {
        let v = unsafe { &*vs.pViewports };
        pl.vp.v = [v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth];
        pl.vp.valid = true;
    }
    if !vs.pScissors.is_null() && vs.scissorCount != 0 && !pl.sc_dyn {
        let r = unsafe { &*vs.pScissors };
        let (x, y) = (r.offset.x, r.offset.y);
        let (x1, y1) = (x.wrapping_add(r.extent.width as i32), y.wrapping_add(r.extent.height as i32));
        pl.vp.sc = [x.max(0) as u32, y.max(0) as u32, x1.max(0) as u32, y1.max(0) as u32];
        pl.vp.valid = true;
    }
}

/// Several wide colour targets: one pipeline per target, drawn one after the other.
fn build_emulated(
    dev: &Device,
    sc: &mut Scratch,
    alloc: *const VkAllocationCallbacks,
    ci: &VkGraphicsPipelineCreateInfo,
    ri: &VkPipelineRenderingCreateInfo,
) -> Result<VkPipeline, VkError> {
    let mut sub: [VkPipeline; 4] = [null_mut(); 4];
    let mut sub = Owned::new(&mut sub, |s| {
        for p in s.iter().filter_map(|p| unsafe { p.get() }) {
            destroy(dev, p, alloc);
        }
    });
    let mut result = Ok(());
    for k in 0..ri.colorAttachmentCount as usize {
        let mut rk = *ri;
        rk.colorAttachmentCount = 1;
        rk.pColorAttachmentFormats = unsafe { ri.pColorAttachmentFormats.add(k) };
        rk.pNext = ci.pNext;
        let mut ck = *ci;
        ck.pNext = (&rk as *const VkPipelineRenderingCreateInfo).cast();
        ck.renderPass = null_mut();
        ck.subpass = 0;
        let mut cbk;
        if let Some(cb) = unsafe { ci.pColorBlendState.as_ref() }.filter(|cb| cb.attachmentCount as usize > k) {
            cbk = *cb;
            cbk.attachmentCount = 1;
            cbk.pAttachments = unsafe { cb.pAttachments.add(k) };
            ck.pColorBlendState = &cbk;
        }
        sc.frag_select = k as u32 + 1;
        match build_graphics(dev, sc, alloc, &ck) {
            Ok(p) => sub[k] = p,
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }
    sc.frag_select = 0;
    result?;
    let sub = sub.keep();
    let first = unsafe { sub[0].get() }.unwrap();
    first.emul_n = ri.colorAttachmentCount as u8;
    first.emul.copy_from_slice(&sub[1..]);
    Ok(sub[0])
}

fn mrt_formats(formats: &[VkFormat]) -> bool {
    formats.iter().all(|f| matches!(*f, VK_FORMAT_R8_UNORM | VK_FORMAT_R8_SNORM | VK_FORMAT_R8_UINT | VK_FORMAT_R8_SINT))
}

entrypoint! {
    fn borgvk_CreateGraphicsPipelines(device: &Device, _cache: VkPipelineCache, _count: u32, infos: &[VkGraphicsPipelineCreateInfo], alloc: *const VkAllocationCallbacks, out: &mut [VkPipeline]) -> Result<(), VkError> {
        let mut sc = device.scratch.lock().unwrap();
        let mut result = Ok(());
        for (i, ci) in infos.iter().enumerate() {
            let ri = unsafe { vk_get_pipeline_rendering_create_info(ci).as_ref() };
            let formats = ri.map_or(&[][..], |ri| unsafe { slice(ri.pColorAttachmentFormats, ri.colorAttachmentCount) });
            let emulate = (2..=4).contains(&formats.len()) && !mrt_formats(formats);
            let r = match ri {
                Some(ri) if emulate => build_emulated(device, &mut sc, alloc, ci, ri),
                _ => build_graphics(device, &mut sc, alloc, ci),
            };
            match r {
                Ok(p) => out[i] = p,
                Err(e) => {
                    out[i..].fill(null_mut());
                    result = Err(e);
                    break;
                }
            }
        }
        result
    }
}

entrypoint! {
    fn borgvk_CreateComputePipelines(device: &Device, _cache: VkPipelineCache, _count: u32, infos: &[VkComputePipelineCreateInfo], alloc: *const VkAllocationCallbacks, out: &mut [VkPipeline]) -> Result<(), VkError> {
        let mut sc = device.scratch.lock().unwrap();
        for (i, ci) in infos.iter().enumerate() {
            let pl = create(device, alloc).inspect_err(|_| out[i..].fill(null_mut()))?;
            let mut pl = Owned::new(pl, |p| destroy(device, p, alloc));
            sc.compile_layout = ci.layout;
            compile_compute_stage(device, &mut sc, &ci.stage, &mut pl);
            // A refused shader fails here, not as a dispatch that silently does nothing.
            if !pl.cs_ok {
                out[i..].fill(null_mut());
                return Err(VkError(VK_ERROR_UNKNOWN));
            }
            out[i] = VkPipeline::from_object(pl.keep());
        }
        Ok(())
    }
}

/// Frees a pipeline and the pipelines of its other colour targets.
fn destroy(dev: &Device, pl: &mut Pipeline, alloc: *const VkAllocationCallbacks) {
    for p in pl.emul.into_iter().filter_map(|p| unsafe { p.get() }) {
        destroy(dev, p, alloc);
    }
    unsafe { vk_object_free(dev.vk_ptr(), alloc, std::ptr::from_mut(pl).cast()) };
}

entrypoint! {
    fn borgvk_DestroyPipeline(device: &Device, pipeline: Option<&mut Pipeline>, alloc: *const VkAllocationCallbacks) {
        if let Some(pl) = pipeline {
            destroy(device, pl, alloc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r8_targets_draw_together() {
        let (m, opt) = mrt(&[VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UINT]);
        assert!(m);
        assert_eq!(opt, 1 << BORGC_FS_TARGET_CLASS_SHIFT | 3 << (BORGC_FS_TARGET_CLASS_SHIFT + 3));
        assert!(!mrt(&[VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8B8A8_UNORM]).0);
        assert!(!mrt(&[VK_FORMAT_R8_UNORM]).0);
        assert_eq!(frag_options(&[VK_FORMAT_R8_UNORM; 3], true, 0), BORGC_FS_MULTI_TARGET | 2 << BORGC_FS_TARGETS_SHIFT);
    }

    #[test]
    fn instance_rate_locations_are_fetch_bits() {
        let b = [VkVertexInputBindingDescription { binding: 1, inputRate: VK_VERTEX_INPUT_RATE_INSTANCE, ..Default::default() }];
        let a = [
            VkVertexInputAttributeDescription { location: 0, binding: 0, ..Default::default() },
            VkVertexInputAttributeDescription { location: 3, binding: 1, ..Default::default() },
        ];
        let vi = VkPipelineVertexInputStateCreateInfo {
            vertexBindingDescriptionCount: 1,
            pVertexBindingDescriptions: b.as_ptr(),
            vertexAttributeDescriptionCount: 2,
            pVertexAttributeDescriptions: a.as_ptr(),
            ..Default::default()
        };
        assert_eq!(vertex_fetch(Some(&vi), false, false), BORGC_VS_FETCH | 1 << 3);
        assert_eq!(vertex_fetch(None, true, false), BORGC_VS_FETCH | BORGC_VS_POINT_QUADS);
    }
}
