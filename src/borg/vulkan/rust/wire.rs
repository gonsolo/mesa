// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The commands a recorded draw sends the list builder, and the state words they carry.

use crate::ffi::*;
use crate::list::Cmd;
use crate::vk::*;
use crate::transport::capturing;
use crate::util::{log, MESA_LOG_INFO, MESA_LOG_WARN};

const SHADER_MAX: usize = BORGVK_SHADER_BLOB_MAX as usize;
const PUSH_MAX_WORDS: u32 = 32;
const TEXG_CHUNK: usize = BORGVK_TEXG_CHUNK as usize;

fn emit(c: Cmd) {
    crate::transport::emit(c);
}

pub fn send_shader(stage: u8, blob: &[u8]) {
    let len = blob.len();
    if len == 0 || len > SHADER_MAX {
        log(MESA_LOG_WARN, format!("borgvk: refusing to send shader (stage {stage}, len {len})"));
        return;
    }
    emit(Cmd::Shader(stage, blob.to_vec()));
    let (what, kind) = (if capturing() { "captured" } else { "uploaded" }, if stage == 0 { "vertex" } else { "fragment" });
    log(MESA_LOG_INFO, format!("borgvk: {what} {kind} shader ({len} bytes)"));
}

pub fn send_push_constants(offset: u32, values: &[u8]) {
    let size = values.len() as u32;
    if size == 0 || offset & 3 != 0 || size & 3 != 0 {
        log(MESA_LOG_WARN, format!("borgvk: bad push-constant range (offset {offset}, size {size})"));
        return;
    }
    let (off_words, n_words) = (offset / 4, size / 4);
    if off_words >= PUSH_MAX_WORDS || n_words > PUSH_MAX_WORDS - off_words {
        log(
            MESA_LOG_WARN,
            format!("borgvk: push-constant range {offset}..{} B exceeds the {} B limit", offset.wrapping_add(size), PUSH_MAX_WORDS * 4),
        );
        return;
    }
    emit(Cmd::Push(off_words, values.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()));
    let what = if capturing() { "captured" } else { "sent" };
    log(MESA_LOG_INFO, format!("borgvk: {what} {size} B of push constants at offset {offset}"));
}


/// Depth bias constant and slope factors of the next draw.
pub fn send_depth_bias(b: [f32; 2]) {
    emit(Cmd::Bias(b[0].to_bits(), b[1].to_bits()));
}

pub fn send_blend(cfg: u32, konst: u32) {
    emit(Cmd::Blend(cfg, konst));
}

fn unorm8(f: f32) -> u32 {
    if !(f > 0.0) {
        0
    } else if f >= 1.0 {
        255
    } else {
        (f * 255.0 + 0.5) as u32
    }
}

/// BLEND_CFG and BLEND_CONST of the first attachment; blending off and all channels written
/// without one.
pub fn blend_pack(cb: Option<&VkPipelineColorBlendStateCreateInfo>) -> (u32, u32) {
    let Some((cb, a)) = cb.and_then(|cb| Some((cb, cb.attachments().first()?))) else {
        return (0xF << 27, 0);
    };
    let cfg = (a.blendEnable != 0) as u32
        | (a.srcColorBlendFactor as u32 & 0x1F) << 1
        | (a.dstColorBlendFactor as u32 & 0x1F) << 6
        | (a.colorBlendOp as u32 & 7) << 11
        | (a.srcAlphaBlendFactor as u32 & 0x1F) << 14
        | (a.dstAlphaBlendFactor as u32 & 0x1F) << 19
        | (a.alphaBlendOp as u32 & 7) << 24
        | (a.colorWriteMask & 0xF) << 27;
    (cfg, blend_const_pack(cb.blendConstants))
}

/// BLEND_CONST: the four constants as UNORM8.
pub fn blend_const_pack(k: [f32; 4]) -> u32 {
    let k = k.map(unorm8);
    k[0] | k[1] << 8 | k[2] << 16 | k[3] << 24
}

pub fn send_state(reg: &[u32; 5]) {
    emit(Cmd::State(*reg));
}

pub fn send_viewport(vp: &Vp) {
    emit(Cmd::Viewport(vp.v, vp.sc));
}

pub fn send_target(flush_format: u8, clear: &[f32; 4]) {
    emit(Cmd::Target(flush_format, *clear));
}

pub fn send_mem(off: u32, data: &[u8]) {
    emit(Cmd::Mem(off, data.to_vec()));
}

pub fn send_vattr(slot: u32, fmt: u32, base: u32, count: u32, stride: u32, swz: u32) {
    emit(Cmd::Vattr([slot, fmt, base, count, stride, swz]));
}

/// A draw: topology, index type, the restart, load and push-stage bits, and its words.
#[derive(Clone, PartialEq, Debug)]
pub struct Draw {
    pub topology: u32,
    pub index_type: u32,
    pub restart: u32,
    pub words: [u32; 8],
}

pub fn send_draw(d: &Draw) {
    emit(Cmd::Draw(d.clone()));
}

pub fn send_att(count: u8, fmt: &[u8; 3]) {
    emit(Cmd::Att(count, *fmt));
}

pub fn send_pass(flush_format: u8, flags: u8) {
    emit(Cmd::Pass(flush_format, flags));
}

/// The pass's attachments in device memory (colour, depth, stencil, 1-3) and their pitch in
/// tiles; zeros: the simulator's fixed planes.
pub fn send_place(at: &[u32; 6], pitch: u32) {
    emit(Cmd::Place(*at, pitch));
}

fn stencil_face_pack(f: &VkStencilOpState) -> u32 {
    (f.compareMask & 0xFF) | (f.writeMask & 0xFF) << 8 | (f.reference & 0xFF) << 16
}

/// STENCIL_CFG, STENCIL_FRONT, STENCIL_BACK, DEPTH_CFG and CULL_CFG of a pipeline. Without
/// state: no stencil, depth LESS with writes, back faces culled.
pub fn state_pack(ci: &VkGraphicsPipelineCreateInfo) -> [u32; 5] {
    let mut reg = [0, 0, 0, 1 | 1 << 3, 2];
    let rs = unsafe { ci.pRasterizationState.as_ref() };
    if let Some(ds) = unsafe { ci.pDepthStencilState.as_ref() } {
        let test = ds.depthTestEnable != 0;
        let cmp = if test { ds.depthCompareOp as u32 & 7 } else { 7 };
        reg[3] = cmp | if test && ds.depthWriteEnable != 0 { 1 << 3 } else { 0 };
        if ds.stencilTestEnable != 0 {
            let (f, b) = (&ds.front, &ds.back);
            let ops = |s: &VkStencilOpState| {
                (s.compareOp as u32 & 7) | (s.failOp as u32 & 7) << 3 | (s.passOp as u32 & 7) << 6 | (s.depthFailOp as u32 & 7) << 9
            };
            reg[0] = 1 | ops(f) << 1 | ops(b) << 13;
            reg[1] = stencil_face_pack(f);
            reg[2] = stencil_face_pack(b);
        }
    }
    if let Some(rs) = rs {
        reg[4] = (rs.cullMode & 3) | if rs.frontFace == VK_FRONT_FACE_CLOCKWISE { 1 << 2 } else { 0 };
    }
    if let Some(ms) = unsafe { ci.pMultisampleState.as_ref() } {
        let discard = rs.is_some_and(|rs| rs.rasterizerDiscardEnable != 0);
        if ms.rasterizationSamples == VK_SAMPLE_COUNT_4_BIT && !discard {
            let mask = if ms.pSampleMask.is_null() { 0xF } else { (unsafe { *ms.pSampleMask }) & 0xF };
            reg[4] |= 1 << 31 | mask << 8 | if ms.alphaToCoverageEnable != 0 { 1 << 12 } else { 0 };
        }
    }
    reg
}

pub fn send_texture_chunk(off: u32, data: &[u8], desc_w123: &[u32; 3], sampler: &[u32; 4]) {
    emit(Cmd::TexChunk(off, data[..data.len().min(TEXG_CHUNK)].to_vec(), *desc_w123, *sampler));
}

/// The texture and sampler slot and the texel byte base of the texture packets that follow.
pub fn send_texture_slot(tex: u8, sampler: u8, base: u32) {
    emit(Cmd::TexSlot(tex, sampler, base));
}

/// A texture the sampler reads in device memory: its descriptor and the sampler's, into the slots
/// of the last [`send_texture_slot`].
pub fn send_texture_desc(desc: &[u32; 16], sampler: &[u32; 4]) {
    emit(Cmd::TexDesc(*desc, *sampler));
}

pub fn send_texture_levels(offs: &[u32; 12]) {
    emit(Cmd::TexLevels(*offs));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `f` under a capture and returns what it sent.
    fn captured(f: impl FnOnce()) -> Vec<Cmd> {
        crate::transport::borgvk_transport_capture_begin();
        f();
        crate::transport::capture_end()
    }

    #[test]
    fn shaders_and_texture_chunks() {
        assert_eq!(captured(|| send_shader(1, &[7, 8, 9])), [Cmd::Shader(1, vec![7, 8, 9])]);
        assert!(captured(|| send_shader(0, &[])).is_empty(), "an empty shader is refused");
        let c = captured(|| send_texture_chunk(0x1234, &[5; 300], &[1, 2, 3], &[4, 5, 6, 7]));
        assert_eq!(c, [Cmd::TexChunk(0x1234, vec![5; TEXG_CHUNK], [1, 2, 3], [4, 5, 6, 7])]);
    }

    #[test]
    fn blend_pack_defaults_and_alpha_blending() {
        assert_eq!(blend_pack(None), (0xF << 27, 0));
        let a = VkPipelineColorBlendAttachmentState {
            blendEnable: 1,
            srcColorBlendFactor: VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            srcAlphaBlendFactor: VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor: VK_BLEND_FACTOR_ZERO,
            colorWriteMask: 0xF,
            ..Default::default()
        };
        let cb = VkPipelineColorBlendStateCreateInfo {
            attachmentCount: 1,
            pAttachments: &a,
            blendConstants: [1.0, 0.5, 0.0, -1.0],
            ..Default::default()
        };
        let cfg = 1 | 6 << 1 | 7 << 6 | 1 << 14 | 0xF << 27;
        assert_eq!(blend_pack(Some(&cb)), (cfg, 255 | 128 << 8));
    }

    #[test]
    fn state_pack_defaults_depth_stencil_and_msaa() {
        assert_eq!(state_pack(&Default::default()), [0, 0, 0, 1 | 1 << 3, 2]);
        let face = VkStencilOpState { compareOp: 7, failOp: 1, passOp: 2, depthFailOp: 3, compareMask: 0x1FF, writeMask: 0xAB, reference: 0xCD };
        let ds = VkPipelineDepthStencilStateCreateInfo {
            depthTestEnable: 1,
            depthCompareOp: VK_COMPARE_OP_LESS_OR_EQUAL,
            stencilTestEnable: 1,
            front: face,
            back: face,
            ..Default::default()
        };
        let rs = VkPipelineRasterizationStateCreateInfo { cullMode: 1, frontFace: VK_FRONT_FACE_CLOCKWISE, ..Default::default() };
        let ms = VkPipelineMultisampleStateCreateInfo { rasterizationSamples: VK_SAMPLE_COUNT_4_BIT, ..Default::default() };
        let ci = VkGraphicsPipelineCreateInfo {
            pDepthStencilState: &ds,
            pRasterizationState: &rs,
            pMultisampleState: &ms,
            ..Default::default()
        };
        let ops = 7 | 1 << 3 | 2 << 6 | 3 << 9;
        assert_eq!(state_pack(&ci), [1 | ops << 1 | ops << 13, 0xFF | 0xAB << 8 | 0xCD << 16, 0xFF | 0xAB << 8 | 0xCD << 16, 3, 1 | 1 << 2 | 1 << 31 | 0xF << 8]);
    }

    fn push(offset: u32, words: &[u32], size: u32) -> Vec<Cmd> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).chain(std::iter::repeat(0)).take(size as usize).collect();
        captured(|| send_push_constants(offset, &bytes))
    }

    #[test]
    fn push_constants() {
        let words = [0xB2B20000u32, 0xB2B20001, 0xB2B20002, 0xB2B20003];
        assert_eq!(push(8 * 4, &words, 16), [Cmd::Push(8, words.to_vec())]);
        let v = [1u32, 2];
        for (offset, size) in [(2, 8), (0, 6), (0, 33 * 4), (32 * 4, 4), (0, 0)] {
            assert!(push(offset, &v, size).is_empty(), "{offset} {size}");
        }
    }
}
