// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The cube path: Vulkan-Tools' cube.c draws from a uniform buffer holding the MVP and every
//! vertex; its frames go to the Borg as one geometry and texture upload, then an MVP per frame.

use crate::corpus::{borgvk_corpus_hash, borgvk_corpus_image, borgvk_corpus_log, borgvk_corpus_on};
use crate::drm::{ioctl, SETUP, SUBMIT};
use crate::cmd_buffer::CommandBuffer;
use crate::descriptor_set::DescriptorSet;
use crate::ffi::*;
use crate::image::Image;
use crate::device::Device;
use crate::serial::{borgvk_serial_send_geom, borgvk_serial_send_mvp};
use crate::sim_process::borgvk_sim_run_stream;
use crate::state_upload::{borgvk_find_set, borgvk_sampler_desc, send_blend_state, upload_shaders};
use crate::texture_upload::borgvk_send_texture_row;
use crate::transport::{borgvk_transport_capture_begin, borgvk_transport_capture_end};
use crate::util::{log, MESA_LOG_INFO, MESA_LOG_WARN};
use std::sync::atomic::{AtomicBool, Ordering};

const MVP_FLOATS: usize = 16;
const NUM_VERTS: usize = 36; // 12 triangles, expanded
const POS_FLOAT0: usize = MVP_FLOATS; // position[36][4]
const ATTR_FLOAT0: usize = MVP_FLOATS + NUM_VERTS * 4; // attr[36][4]
const MIN_FLOATS: usize = ATTR_FLOAT0 + NUM_VERTS * 4; // 304 = 1216 B

/// The first colour attachment of the submit's command buffers.
fn color_attachment(s: &vk_queue_submit) -> *mut Image {
    for &cb in unsafe { crate::entrypoint::slice(s.command_buffers, s.command_buffer_count) } {
        let v = unsafe { (*cb.cast::<CommandBuffer>()).color_views[0] };
        if let Some(v) = unsafe { v.as_ref() }.filter(|v| !v.image.is_null()) {
            return v.image.cast();
        }
    }
    std::ptr::null_mut()
}

/// Positions deduplicated into at most BORGVK_GEOM_MAX_VERTS, with an index and a UV per vertex.
fn send_geometry(ubo: &[f32]) {
    let (pos, att) = (&ubo[POS_FLOAT0..], &ubo[ATTR_FLOAT0..]);
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let (mut idx, mut uv) = ([0u8; NUM_VERTS], [0f32; NUM_VERTS * 2]);
    for i in 0..NUM_VERTS {
        let p = [pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]];
        let u = match verts.iter().position(|v| *v == p) {
            Some(u) => u,
            None if verts.len() >= BORGVK_GEOM_MAX_VERTS as usize => return, // the cube fits
            None => {
                verts.push(p);
                verts.len() - 1
            }
        };
        idx[i] = u as u8;
        uv[i * 2..i * 2 + 2].copy_from_slice(&att[i * 4..i * 4 + 2]);
    }
    unsafe { borgvk_serial_send_geom(verts.as_ptr().cast(), verts.len() as i32, idx.as_ptr(), uv.as_ptr(), (NUM_VERTS / 3) as i32) };
}

fn sentinel_path() -> String {
    let dev = std::env::var("BORGVK_SERIAL").ok().filter(|d| !d.is_empty()).unwrap_or_else(|| "/dev/ttyUSB0".into());
    format!("/tmp/borgvk_{}_setup", dev.rsplit('/').next().unwrap_or(&dev))
}

fn setup_already_done() -> bool {
    std::env::var_os("BORGVK_FORCE_UPLOAD").is_none() && std::path::Path::new(&sentinel_path()).exists()
}

/// The set's uniform buffer from its offset on, as floats.
fn ubo(set: &DescriptorSet) -> Option<&[f32]> {
    let b = unsafe { set.buffers[0].as_ref() }?;
    let mem = unsafe { b.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    let off = b.offset + set.offsets[0];
    if off >= b.vk.size {
        return None;
    }
    let n = ((b.vk.size - off) / 4) as usize;
    Some(unsafe { std::slice::from_raw_parts(mem.map.byte_add(off as usize).cast(), n) })
}

fn texture(set: &DescriptorSet) -> Option<*mut Image> {
    let t = set.images[1];
    let ti = unsafe { t.as_ref() }?;
    let mapped = unsafe { ti.mem.as_ref() }.is_some_and(|m| !m.map.is_null());
    (mapped && ti.vk.extent.width != 0 && ti.vk.extent.height != 0).then_some(t)
}

fn send_texture(set: &DescriptorSet, tex: *mut Image) {
    for row in 0..BORGVK_TEX_DIM as i32 {
        unsafe { borgvk_send_texture_row(tex, set.samplers[1], row) };
    }
}

/// True when the set is cube.c's UBO-driven frame, not a CTS draw from vertex buffers.
///
/// # Safety
/// `set` is valid.
pub unsafe fn borgvk_set_is_cube(set: *mut DescriptorSet) -> bool {
    let set = unsafe { &*set };
    let Some(b) = (unsafe { set.buffers[0].as_ref() }) else { return false };
    if unsafe { b.mem.as_ref() }.is_none_or(|m| m.map.is_null()) {
        return false;
    }
    let off = b.offset + set.offsets[0];
    if off >= b.vk.size {
        return false;
    }
    let mut avail = b.vk.size - off;
    if set.ranges[0] != VK_WHOLE_SIZE as u64 && set.ranges[0] < avail {
        avail = set.ranges[0];
    }
    avail / 4 >= MIN_FLOATS as u64
}

/// # Safety
/// The runtime's submit.
pub unsafe fn borgvk_submit_is_cube(submit: *mut vk_queue_submit) -> bool {
    let set = unsafe { borgvk_find_set(submit) };
    !set.is_null() && unsafe { borgvk_set_is_cube(set) }
}

/// One frame's whole wire stream rendered by the simulator or the device into the colour
/// attachment.
///
/// # Safety
/// A valid device and the runtime's submit.
pub unsafe fn borgvk_submit_sim_cube(device: *mut Device, submit: *mut vk_queue_submit) -> VkResult {
    let sim = std::env::var_os("BORGVK_SIM").is_some() && std::env::var_os("BORGVK_SIM_FW").is_some();
    if !crate::hw::enabled() && !sim {
        return VK_SUCCESS;
    }
    let Some(set) = (unsafe { borgvk_find_set(submit).as_ref() }) else { return VK_SUCCESS };
    let Some(ubo) = ubo(set).filter(|u| u.len() >= MIN_FLOATS) else { return VK_SUCCESS };
    let d = unsafe { &*device };
    let sc = d.scratch.lock().unwrap();
    borgvk_transport_capture_begin();
    unsafe {
        upload_shaders(d, &sc);
        send_geometry(ubo);
        send_blend_state(d, &sc);
        if let Some(tex) = texture(set) {
            send_texture(set, tex);
        }
        borgvk_serial_send_mvp(ubo.as_ptr());
    }
    let mut n = 0usize;
    let bytes = unsafe { borgvk_transport_capture_end(&mut n) };
    if bytes.is_null() || n == 0 {
        unsafe { free(bytes.cast()) };
        return VK_SUCCESS;
    }
    if let Some(dump) = std::env::var_os("BORGVK_SIM_DUMP").filter(|d| !d.is_empty()) {
        let _ = std::fs::write(&dump, unsafe { std::slice::from_raw_parts(bytes, n) });
        log(MESA_LOG_INFO, format!("borgvk: dumped {n}-byte sim stream to {}", dump.to_string_lossy()));
        unsafe { free(bytes.cast()) };
        return VK_SUCCESS;
    }
    let color = color_attachment(unsafe { &*submit });
    let rec = borgvk_corpus_on();
    let key = if rec { unsafe { borgvk_corpus_hash(BORGVK_CORPUS_SEED as u64, bytes.cast(), n) } } else { 0 };
    let res = unsafe { borgvk_sim_run_stream(bytes, n, color, 0) };
    if rec {
        borgvk_corpus_log(b'C' as _, key, unsafe { borgvk_corpus_image(BORGVK_CORPUS_SEED as u64, color) });
    }
    res
}

static SETUP_DONE: AtomicBool = AtomicBool::new(false);

/// The board: the upload once (DRM SETUP, or the serial burst unless its sentinel file exists),
/// then the blend state and an MVP per frame.
///
/// # Safety
/// A valid device and the runtime's submit.
pub unsafe fn borgvk_submit_cube_live(device: *mut Device, submit: *mut vk_queue_submit) -> VkResult {
    let Some(set) = (unsafe { borgvk_find_set(submit).as_ref() }) else { return VK_SUCCESS };
    let Some(ubo) = ubo(set) else { return VK_SUCCESS };
    let d = unsafe { &*device };
    let sc = d.scratch.lock().unwrap();
    let ubuf = unsafe { &*set.buffers[0] };
    let tex = texture(set);
    let done = || SETUP_DONE.load(Ordering::Relaxed);
    if d.drm_fd >= 0 {
        let tex_mem = unsafe { set.images[1].as_ref().and_then(|t| t.mem.as_ref()) };
        let ubo_gem = unsafe { (*ubuf.mem).gem_handle };
        if !done() && ubo_gem != 0 && tex_mem.is_some_and(|m| m.gem_handle != 0) {
            let t = unsafe { &*set.images[1] };
            upload_shaders(d, &sc); // borgc-compiled shaders, before geometry and texture
            let mut s = drm_borg_setup {
                ubo_handle: ubo_gem,
                tex_handle: tex_mem.unwrap().gem_handle,
                tex_offset: t.offset,
                tex_width: t.vk.extent.width,
                tex_height: t.vk.extent.height,
                sampler: unsafe { *borgvk_sampler_desc(set.samplers[1]).cast::<[u32; 4]>() },
            };
            if ioctl(d.drm_fd, SETUP, &mut s) {
                SETUP_DONE.store(true, Ordering::Relaxed);
            } else {
                log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_SETUP failed".into());
            }
        }
        if done() {
            send_blend_state(d, &sc);
            ioctl(d.drm_fd, SUBMIT, &mut drm_borg_submit { ubo_handle: ubo_gem, pad: 0 });
        }
        return VK_SUCCESS;
    }
    if !done() {
        if setup_already_done() {
            log(MESA_LOG_INFO, "borgvk: skipping upload (sentinel present); set BORGVK_FORCE_UPLOAD=1 to re-upload".into());
            SETUP_DONE.store(true, Ordering::Relaxed);
        } else if let (true, Some(tex)) = (ubo.len() >= MIN_FLOATS, tex) {
            log(MESA_LOG_INFO, "borgvk: uploading shaders + geometry + texture burst...".into());
            upload_shaders(d, &sc); // borgc-compiled shaders, before geometry and texture
            send_geometry(ubo);
            send_texture(set, tex);
            let _ = std::fs::File::create(sentinel_path());
            log(MESA_LOG_INFO, "borgvk: upload complete".into());
            SETUP_DONE.store(true, Ordering::Relaxed);
        }
    }
    if done() {
        send_blend_state(d, &sc);
        unsafe { borgvk_serial_send_mvp(ubo.as_ptr()) };
    }
    VK_SUCCESS
}

#[cfg(test)]
mod tests {
    #[test]
    fn sentinel_names_the_serial_device() {
        std::env::set_var("BORGVK_SERIAL", "/dev/ttyACM1");
        assert_eq!(super::sentinel_path(), "/tmp/borgvk_ttyACM1_setup");
    }
}
