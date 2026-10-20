// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The wire packets borgvk sends to the firmware, the simulator or the DRM driver. Each packet is
//! a marker byte, a payload and an XOR checksum over everything between them.

use crate::ffi::*;
use crate::transport::capturing;
use crate::util::{log, MESA_LOG_INFO, MESA_LOG_WARN};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const MARKER_MVP: u8 = 0xAD;
const MARKER_GEOM: u8 = 0xAE;
const MARKER_TEX: u8 = 0xAF;
const MARKER_SHADER: u8 = 0xB0;
const MARKER_SHADER_BIG: u8 = 0xC1;
const MARKER_PUSH: u8 = 0xB2;
const MARKER_BLEND: u8 = 0xB3;
const MARKER_STATE: u8 = 0xB4;
const MARKER_TEXG: u8 = 0xB5;
const MARKER_TARGET: u8 = 0xB6;
const MARKER_MEM: u8 = 0xB8;
const MARKER_VATTR: u8 = 0xB9;
const MARKER_DRAW: u8 = 0xBA;
const MARKER_PASS: u8 = 0xBB;
const MARKER_ATT: u8 = 0xBF;
const MARKER_TEXL: u8 = 0xC0;
const MARKER_VP: u8 = 0xC2;

const GEOM_MAX_VERTS: usize = BORGVK_GEOM_MAX_VERTS as usize;
const GEOM_MAX_TRIS: usize = BORGVK_GEOM_MAX_TRIS as usize;
const GEOM_PKT_LEN: usize = 1 + 2 + GEOM_MAX_VERTS * 12 + GEOM_MAX_TRIS * 3 + GEOM_MAX_TRIS * 24 + 1;
const TEX_DIM: usize = BORGVK_TEX_DIM as usize;
const SHADER_SMALL: usize = 512;
const SHADER_MAX: usize = BORGVK_SHADER_BLOB_MAX as usize;
const PUSH_MAX_WORDS: u32 = 32;
const TEXG_CHUNK: usize = BORGVK_TEXG_CHUNK as usize;

/// Fills in the checksum and hands the packet to the transport.
fn emit(pkt: &mut [u8]) {
    let n = pkt.len();
    pkt[n - 1] = pkt[1..n - 1].iter().fold(0, |c, b| c ^ b);
    crate::transport::emit(pkt);
}

fn put_u32(d: &mut [u8], v: u32) {
    d[..4].copy_from_slice(&v.to_le_bytes());
}

fn put_f32(d: &mut [u8], f: f32) {
    put_u32(d, f.to_bits());
}

pub fn send_geom(verts: &[f32], idx: &[u8], uv: &[f32]) {
    let (nverts, ntris) = (verts.len() / 3, idx.len() / 3);
    if !(1..=GEOM_MAX_VERTS).contains(&nverts) || !(1..=GEOM_MAX_TRIS).contains(&ntris) {
        return;
    }
    let mut pkt = [0u8; GEOM_PKT_LEN];
    pkt[0] = MARKER_GEOM;
    pkt[1] = nverts as u8;
    pkt[2] = ntris as u8;
    let (vbase, ibase) = (3, 3 + GEOM_MAX_VERTS * 12);
    let ubase = ibase + GEOM_MAX_TRIS * 3;
    for (i, &v) in verts.iter().enumerate() {
        put_f32(&mut pkt[vbase + i * 4..], v);
    }
    pkt[ibase..ibase + idx.len()].copy_from_slice(idx);
    for (i, &u) in uv[..ntris * 6].iter().enumerate() {
        put_f32(&mut pkt[ubase + i * 4..], u);
    }
    emit(&mut pkt);
}

pub fn send_tex_row(y: u8, rgba: &[u8], sampler: &[u32; 4]) {
    let mut pkt = [0u8; 1 + 1 + 16 + TEX_DIM * 4 + 1];
    pkt[0] = MARKER_TEX;
    pkt[1] = y;
    for (w, &s) in sampler.iter().enumerate() {
        put_u32(&mut pkt[2 + 4 * w..], s);
    }
    pkt[18..18 + TEX_DIM * 4].copy_from_slice(&rgba[..TEX_DIM * 4]);
    emit(&mut pkt);
}

pub fn send_shader(stage: u8, blob: &[u8]) {
    let len = blob.len();
    if len == 0 || len > SHADER_MAX {
        log(MESA_LOG_WARN, format!("borgvk: refusing to send shader (stage {stage}, len {len})"));
        return;
    }
    let big = len > SHADER_SMALL;
    let mut pkt = vec![0u8; 1 + 1 + 2 + if big { SHADER_MAX } else { SHADER_SMALL } + 1];
    pkt[0] = if big { MARKER_SHADER_BIG } else { MARKER_SHADER };
    pkt[1] = stage;
    pkt[2..4].copy_from_slice(&(len as u16).to_le_bytes());
    pkt[4..4 + len].copy_from_slice(blob);
    emit(&mut pkt);
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
    let mut pkt = [0u8; 1 + 1 + 1 + PUSH_MAX_WORDS as usize * 4 + 1];
    pkt[0] = MARKER_PUSH;
    pkt[1] = off_words as u8;
    pkt[2] = n_words as u8;
    pkt[3..3 + values.len()].copy_from_slice(values);
    emit(&mut pkt);
    let what = if capturing() { "captured" } else { "sent" };
    log(MESA_LOG_INFO, format!("borgvk: {what} {size} B of push constants at offset {offset}"));
}

pub fn send_blend(cfg: u32, konst: u32) {
    let mut pkt = [0u8; 10];
    pkt[0] = MARKER_BLEND;
    put_u32(&mut pkt[1..], cfg);
    put_u32(&mut pkt[5..], konst);
    emit(&mut pkt);
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
    let Some(cb) = cb.filter(|cb| cb.attachmentCount >= 1 && !cb.pAttachments.is_null()) else {
        return (0xF << 27, 0);
    };
    let a = unsafe { &*cb.pAttachments };
    let cfg = (a.blendEnable != 0) as u32
        | (a.srcColorBlendFactor as u32 & 0x1F) << 1
        | (a.dstColorBlendFactor as u32 & 0x1F) << 6
        | (a.colorBlendOp as u32 & 7) << 11
        | (a.srcAlphaBlendFactor as u32 & 0x1F) << 14
        | (a.dstAlphaBlendFactor as u32 & 0x1F) << 19
        | (a.alphaBlendOp as u32 & 7) << 24
        | (a.colorWriteMask & 0xF) << 27;
    let k = cb.blendConstants.map(unorm8);
    (cfg, k[0] | k[1] << 8 | k[2] << 16 | k[3] << 24)
}

static PACING: Mutex<Option<Instant>> = Mutex::new(None);

/// Holds frames to the period `BORGVK_FRAME_MS` asks for; a late frame resyncs instead of
/// building up debt.
fn pace_frame() {
    // atol: the leading digits count.
    let ms: u64 = std::env::var("BORGVK_FRAME_MS")
        .map(|v| v.trim_start().chars().take_while(char::is_ascii_digit).collect::<String>())
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(0);
    if ms == 0 {
        return;
    }
    let period = Duration::from_millis(ms);
    let mut next = PACING.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let target = match *next {
        Some(t) if t > now => {
            std::thread::sleep(t - now);
            t
        }
        _ => now,
    };
    *next = Some(target + period);
}

pub fn send_mvp(mvp: &[f32; 16]) {
    let mut pkt = [0u8; 66];
    pkt[0] = MARKER_MVP;
    for (i, &v) in mvp.iter().enumerate() {
        pkt[1 + 4 * i..5 + 4 * i].copy_from_slice(&v.to_ne_bytes());
    }
    emit(&mut pkt);
    if !capturing() {
        pace_frame();
    }
}

pub fn send_state(reg: &[u32; 5]) {
    let mut pkt = [0u8; 22];
    pkt[0] = MARKER_STATE;
    for (r, &v) in reg.iter().enumerate() {
        put_u32(&mut pkt[1 + 4 * r..], v);
    }
    emit(&mut pkt);
}

pub fn send_viewport(vp: &borgvk_vp) {
    let mut pkt = [0u8; 42];
    pkt[0] = MARKER_VP;
    for (k, &v) in vp.v.iter().enumerate() {
        pkt[1 + 4 * k..5 + 4 * k].copy_from_slice(&v.to_ne_bytes());
    }
    for (k, &s) in vp.sc.iter().enumerate() {
        put_u32(&mut pkt[25 + 4 * k..], s);
    }
    emit(&mut pkt);
}

pub fn send_target(flush_format: u8, clear: &[f32; 4]) {
    let mut pkt = [0u8; 19];
    pkt[0] = MARKER_TARGET;
    pkt[1] = flush_format;
    for (i, &c) in clear.iter().enumerate() {
        put_f32(&mut pkt[2 + 4 * i..], c);
    }
    emit(&mut pkt);
}

pub fn send_mem(off: u32, data: &[u8]) {
    let mut pkt = [0u8; 1 + 4 + 2 + 256 + 1];
    pkt[0] = MARKER_MEM;
    put_u32(&mut pkt[1..], off);
    pkt[5..7].copy_from_slice(&(data.len() as u16).to_le_bytes());
    pkt[7..7 + data.len()].copy_from_slice(data);
    emit(&mut pkt);
}

pub fn send_vattr(slot: u32, fmt: u32, base: u32, count: u32, stride: u32, swz: u32) {
    let mut pkt = [0u8; 20];
    pkt[0] = MARKER_VATTR;
    pkt[1] = slot as u8;
    pkt[2] = fmt as u8;
    for (i, v) in [base, count, stride, swz].into_iter().enumerate() {
        put_u32(&mut pkt[3 + 4 * i..], v);
    }
    emit(&mut pkt);
}

/// The fields of a 0xBA draw packet, in wire order after the three byte fields.
pub struct Draw {
    pub topology: u32,
    pub index_type: u32,
    pub restart: u32,
    pub words: [u32; 7],
}

pub fn send_draw(d: &Draw) {
    let mut pkt = [0u8; 1 + 3 + 7 * 4 + 1];
    pkt[0] = MARKER_DRAW;
    pkt[1..4].copy_from_slice(&[d.topology as u8, d.index_type as u8, d.restart as u8]);
    for (i, &v) in d.words.iter().enumerate() {
        put_u32(&mut pkt[4 + 4 * i..], v);
    }
    emit(&mut pkt);
}

pub fn send_att(count: u8, fmt: &[u8; 3]) {
    emit(&mut [MARKER_ATT, count, fmt[0], fmt[1], fmt[2], 0]);
}

pub fn send_pass(flush_format: u8, flags: u8) {
    emit(&mut [MARKER_PASS, flush_format, flags, 0]);
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
    let data = &data[..data.len().min(TEXG_CHUNK)];
    let mut pkt = [0u8; 1 + 4 + 2 + 12 + 16 + TEXG_CHUNK + 1];
    pkt[0] = MARKER_TEXG;
    put_u32(&mut pkt[1..], off);
    pkt[5..7].copy_from_slice(&(data.len() as u16).to_le_bytes());
    for (w, &v) in desc_w123.iter().chain(sampler).enumerate() {
        put_u32(&mut pkt[7 + 4 * w..], v);
    }
    pkt[35..35 + data.len()].copy_from_slice(data);
    emit(&mut pkt);
}

pub fn send_texture_levels(offs: &[u32; 12]) {
    let mut pkt = [0u8; 1 + 48 + 1];
    pkt[0] = MARKER_TEXL;
    for (w, &v) in offs.iter().enumerate() {
        put_u32(&mut pkt[1 + 4 * w..], v);
    }
    emit(&mut pkt);
}

// The C interface; pointers come from the C callers and are valid for the sizes stated.

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_geom(verts: *const f32, nverts: i32, idx: *const u8, uv: *const f32, ntris: i32) {
    if !(1..=GEOM_MAX_VERTS as i32).contains(&nverts) || !(1..=GEOM_MAX_TRIS as i32).contains(&ntris) {
        return;
    }
    let (nv, nt) = (nverts as usize, ntris as usize);
    unsafe {
        send_geom(
            std::slice::from_raw_parts(verts, nv * 3),
            std::slice::from_raw_parts(idx, nt * 3),
            std::slice::from_raw_parts(uv, nt * 6),
        )
    };
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_tex_row(y: i32, rgba: *const u8, sampler: *const u32) {
    unsafe { send_tex_row(y as u8, std::slice::from_raw_parts(rgba, TEX_DIM * 4), &*(sampler as *const [u32; 4])) };
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_shader(stage: u8, blob: *const u8, len: u32) {
    if blob.is_null() || len == 0 || len as usize > SHADER_MAX {
        log(MESA_LOG_WARN, format!("borgvk: refusing to send shader (stage {stage}, len {len})"));
        return;
    }
    send_shader(stage, unsafe { std::slice::from_raw_parts(blob, len as usize) });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_push_constants(offset: u32, size: u32, values: *const std::ffi::c_void) {
    if values.is_null() {
        log(MESA_LOG_WARN, format!("borgvk: bad push-constant range (offset {offset}, size {size})"));
        return;
    }
    send_push_constants(offset, unsafe { std::slice::from_raw_parts(values as *const u8, size as usize) });
}

#[no_mangle]
pub extern "C" fn borgvk_serial_send_blend(blend_cfg: u32, blend_const: u32) {
    send_blend(blend_cfg, blend_const);
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_blend_pack(cb: *const VkPipelineColorBlendStateCreateInfo, cfg: *mut u32, konst: *mut u32) {
    let (c, k) = blend_pack(unsafe { cb.as_ref() });
    unsafe {
        *cfg = c;
        *konst = k;
    }
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_mvp(mvp: *const f32) {
    send_mvp(unsafe { &*(mvp as *const [f32; 16]) });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_state(reg: *const u32) {
    send_state(unsafe { &*(reg as *const [u32; 5]) });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_viewport(vp: *const borgvk_vp) {
    send_viewport(unsafe { &*vp });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_target(flush_format: u8, clear: *const f32) {
    send_target(flush_format, unsafe { &*(clear as *const [f32; 4]) });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_pass(flush_format: u8, flags: u8) {
    send_pass(flush_format, flags);
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_att(count: u8, fmt: *const u8) {
    send_att(count, unsafe { &*(fmt as *const [u8; 3]) });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_mem(off: u32, data: *const u8, n: u32) {
    send_mem(off, unsafe { std::slice::from_raw_parts(data, n as usize) });
}

#[no_mangle]
pub extern "C" fn borgvk_serial_send_vattr(slot: u32, fmt: u32, base: u32, count: u32, stride: u32, swz: u32) {
    send_vattr(slot, fmt, base, count, stride, swz);
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn borgvk_serial_send_draw(
    topology: u32,
    index_type: u32,
    restart: u32,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
    vertex_offset: i32,
    index_base: u32,
    ubo_base: u32,
) {
    let words = [vertex_count, instance_count, first_vertex, first_instance, vertex_offset as u32, index_base, ubo_base];
    send_draw(&Draw { topology, index_type, restart, words });
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_state_pack(ci: *const VkGraphicsPipelineCreateInfo, reg: *mut u32) {
    unsafe { *(reg as *mut [u32; 5]) = state_pack(&*ci) };
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_texture_chunk(
    off: u32,
    data: *const u8,
    n: u32,
    desc_w123: *const u32,
    sampler: *const u32,
) {
    let n = (n as usize).min(TEXG_CHUNK);
    unsafe { send_texture_chunk(off, std::slice::from_raw_parts(data, n), &*(desc_w123 as *const [u32; 3]), &*(sampler as *const [u32; 4])) };
}

#[no_mangle]
pub unsafe extern "C" fn borgvk_serial_send_texture_levels(offs: *const u32) {
    send_texture_levels(unsafe { &*(offs as *const [u32; 12]) });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `f` under a capture and returns what it sent; empty if nothing.
    fn captured(f: impl FnOnce()) -> Vec<u8> {
        crate::transport::borgvk_transport_capture_begin();
        f();
        let mut len = 0;
        let buf = unsafe { crate::transport::borgvk_transport_capture_end(&mut len) };
        if buf.is_null() {
            return Vec::new();
        }
        let v = unsafe { std::slice::from_raw_parts(buf, len) }.to_vec();
        unsafe { free(buf.cast()) };
        v
    }

    /// Runs `f` and returns the one packet it sent, after checking its checksum.
    fn sent(f: impl FnOnce()) -> Vec<u8> {
        let p = captured(f);
        assert!(!p.is_empty());
        assert_eq!(p[1..].iter().fold(0, |c, b| c ^ b), 0, "checksum");
        p
    }

    fn u32_at(p: &[u8], i: usize) -> u32 {
        u32::from_le_bytes(p[i..i + 4].try_into().unwrap())
    }

    #[test]
    fn viewport() {
        let vp = borgvk_vp { valid: true, v: [1.0, 2.0, 3.0, 4.0, 0.0, 1.0], sc: [5, 6, 7, 8] };
        let p = sent(|| send_viewport(&vp));
        assert_eq!((p.len(), p[0]), (42, MARKER_VP));
        assert_eq!(f32::from_le_bytes(p[9..13].try_into().unwrap()), 3.0);
        assert_eq!(u32_at(&p, 37), 8);
    }

    #[test]
    fn shaders_small_and_big() {
        let p = sent(|| send_shader(1, &[7, 8, 9]));
        assert_eq!((p.len(), p[0], p[1], &p[2..7]), (517, MARKER_SHADER, 1, &[3, 0, 7, 8, 9][..]));
        let p = sent(|| send_shader(0, &[1; 600]));
        assert_eq!((p.len(), p[0], u16::from_le_bytes([p[2], p[3]])), (1 + 1 + 2 + SHADER_MAX + 1, MARKER_SHADER_BIG, 600));
        assert!(captured(|| send_shader(0, &[])).is_empty(), "an empty shader is refused");
    }

    #[test]
    fn mvp() {
        let mut m = [0.0f32; 16];
        m[15] = 1.0;
        let p = sent(|| send_mvp(&m));
        assert_eq!((p.len(), p[0]), (66, MARKER_MVP));
        assert_eq!(f32::from_ne_bytes(p[61..65].try_into().unwrap()), 1.0);
    }

    #[test]
    fn texture_chunk_is_capped() {
        let p = sent(|| send_texture_chunk(0x1234, &[5; 300], &[1, 2, 3], &[4, 5, 6, 7]));
        assert_eq!((p.len(), p[0], u32_at(&p, 1)), (1 + 4 + 2 + 12 + 16 + TEXG_CHUNK + 1, MARKER_TEXG, 0x1234));
        assert_eq!(u16::from_le_bytes([p[5], p[6]]), TEXG_CHUNK as u16);
        assert_eq!((u32_at(&p, 7), u32_at(&p, 31)), (1, 7));
    }

    #[test]
    fn texture_levels() {
        let offs: [u32; 12] = std::array::from_fn(|i| i as u32 * 16);
        let p = sent(|| send_texture_levels(&offs));
        assert_eq!((p.len(), p[0], u32_at(&p, 1 + 4 * 11)), (50, MARKER_TEXL, 176));
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

    fn push(offset: u32, size: u32, values: *const u32) -> Vec<u8> {
        captured(|| unsafe { borgvk_serial_send_push_constants(offset, size, values.cast()) })
    }

    #[test]
    fn push_constants() {
        let words = [0xB2B20000u32, 0xB2B20001, 0xB2B20002, 0xB2B20003];
        let p = push(8 * 4, 16, words.as_ptr());
        assert_eq!(p.len(), 132);
        assert_eq!(p[..3], [0xB2, 8, 4]); // 0xB1 is the serial-reload trigger
        for (i, &w) in words.iter().enumerate() {
            assert_eq!(u32_at(&p, 3 + 4 * i), w);
        }
        assert!(p[3 + 16..131].iter().all(|&b| b == 0));
        assert_eq!(p[1..].iter().fold(0, |c, b| c ^ b), 0, "checksum");

        let full: Vec<u32> = (0..32).map(|i| 0xA5A50000 + i).collect();
        let p = push(0, 128, full.as_ptr());
        assert_eq!((p.len(), p[1], p[2]), (132, 0, 32));

        let v = [1u32, 2];
        for (offset, size) in [(2, 8), (0, 6), (0, 33 * 4), (32 * 4, 4), (0, 0)] {
            assert!(push(offset, size, v.as_ptr()).is_empty(), "{offset} {size}");
        }
        assert!(push(0, 4, std::ptr::null()).is_empty());
    }

    #[test]
    fn geometry() {
        let verts = [-1.0f32, 0.5, 0.1, 1.0, -0.25, 3.0];
        let (idx, uv) = ([0u8, 1, 1], [0.0f32, 1.0, 0.75, 0.5, 0.1, 0.9]);
        let p = sent(|| unsafe { borgvk_serial_send_geom(verts.as_ptr(), 2, idx.as_ptr(), uv.as_ptr(), 1) });
        assert_eq!(p.len(), GEOM_PKT_LEN);
        assert_eq!(p[..3], [0xAE, 2, 1]);
        let (vb, ib) = (3, 3 + GEOM_MAX_VERTS * 12);
        let ub = ib + GEOM_MAX_TRIS * 3;
        assert_eq!([u32_at(&p, vb), u32_at(&p, vb + 4), u32_at(&p, vb + 8), u32_at(&p, vb + 20)], [0xBF800000, 0x3F000000, 0x3DCCCCCD, 0x40400000]);
        assert_eq!(p[ib..ib + 3], [0, 1, 1]);
        assert_eq!([u32_at(&p, ub + 8), u32_at(&p, ub + 12)], [0x3F400000, 0x3F000000]);
        assert!(p[vb + 24..ib].iter().chain(&p[ub + 24..GEOM_PKT_LEN - 1]).all(|&b| b == 0));
    }

    #[test]
    fn texture_row() {
        let rgba: Vec<u8> = (0..TEX_DIM * 4).map(|i| (i * 7) as u8).collect();
        let sampler = [0x000A0493u32, 0x3F800000, 0, 0x447A0000];
        let p = sent(|| unsafe { borgvk_serial_send_tex_row(5, rgba.as_ptr(), sampler.as_ptr()) });
        assert_eq!(p.len(), 1 + 1 + 16 + TEX_DIM * 4 + 1);
        assert_eq!(p[..2], [0xAF, 5]);
        assert_eq!((0..4).map(|w| u32_at(&p, 2 + 4 * w)).collect::<Vec<_>>(), sampler);
        assert_eq!(p[18..18 + TEX_DIM * 4], rgba[..]);
    }

    #[test]
    fn fixed_packets() {
        let p = sent(|| send_blend(0x11223344, 0x55667788));
        assert_eq!((p.len(), u32_at(&p, 1), u32_at(&p, 5)), (10, 0x11223344, 0x55667788));
        let p = sent(|| send_state(&[1, 2, 3, 4, 0xdeadbeef]));
        assert_eq!((p.len(), u32_at(&p, 17)), (22, 0xdeadbeef));
        let p = sent(|| send_target(2, &[0.0, 0.5, 1.0, 0.25]));
        assert_eq!((p.len(), p[1]), (19, 2));
        let p = sent(|| send_mem(0x1000, &[1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(p.len(), 1 + 4 + 2 + 256 + 1);
        assert_eq!((p[0], u32_at(&p, 1), p[5], p[6], p[7], p[14]), (0xB8, 0x1000, 8, 0, 1, 8));
        let p = sent(|| send_vattr(3, 41, 0x100, 12, 16, 0x40000000));
        assert_eq!((p.len(), p[0], p[1], p[2]), (20, 0xB9, 3, 41));
        assert_eq!([u32_at(&p, 3), u32_at(&p, 7), u32_at(&p, 11), u32_at(&p, 15)], [0x100, 12, 16, 0x40000000]);
        let p = sent(|| borgvk_serial_send_draw(1, 2, 1, 30, 4, 5, 6, -7, 0x200, 0x300));
        assert_eq!((p.len(), p[0], p[1], p[2], p[3]), (33, 0xBA, 1, 2, 1));
        assert_eq!((4..11).map(|i| u32_at(&p, 4 * i - 12)).collect::<Vec<_>>(), [30, 4, 5, 6, -7i32 as u32, 0x200, 0x300]);
        let p = sent(|| send_att(3, &[5, 6, 7]));
        assert_eq!((p.len(), p[0], p[1], p[4]), (6, 0xBF, 3, 7));
        let p = sent(|| send_pass(4, 9));
        assert_eq!(p[..3], [0xBB, 4, 9]);
        assert_eq!(p.len(), 4);
    }
}
