// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! State and shaders to the Borg: over the DRM device when there is one, else as wire packets.

use crate::drm::{ioctl, BLEND, PUSH, SHADER, STATE};
use crate::entrypoint::slice;
use crate::descriptor_set::DescriptorSet;
use crate::ffi::*;
use crate::device::Device;
use crate::serial::*;
use crate::util::{log, MESA_LOG_WARN};

/// The first descriptor set a submit binds.
///
/// # Safety
/// The runtime's submit.
pub unsafe fn borgvk_find_set(submit: *mut vk_queue_submit) -> *mut DescriptorSet {
    let s = unsafe { &*submit };
    for &cb in unsafe { slice(s.command_buffers, s.command_buffer_count) } {
        let head: *mut list_head = unsafe { &mut (*cb).cmd_queue.cmds };
        let mut p = unsafe { (*head).next };
        while p != head {
            let e = unsafe { &*p.byte_sub(std::mem::offset_of!(vk_cmd_queue_entry, cmd_link)).cast::<vk_cmd_queue_entry>() };
            p = unsafe { (*p).next };
            if e.type_ != VK_CMD_BIND_DESCRIPTOR_SETS {
                continue;
            }
            let b = unsafe { &e.u.bind_descriptor_sets };
            if b.descriptor_set_count != 0 && !b.descriptor_sets.is_null() {
                let set = unsafe { *b.descriptor_sets }.cast::<DescriptorSet>();
                if !set.is_null() {
                    return set;
                }
            }
        }
    }
    std::ptr::null_mut()
}

/// Clamp to edge in U, V and W, nearest.
static DEFAULT_SAMPLER: [u32; 4] = [2 << 3 | 2 << 6 | 2 << 9, 0, 0, 0];

/// # Safety
/// `sampler` is null or valid.
pub unsafe fn borgvk_sampler_desc(sampler: *const borgvk_sampler) -> *const u32 {
    unsafe { sampler.as_ref() }.map_or(DEFAULT_SAMPLER.as_ptr(), |s| s.desc.as_ptr())
}

fn push_range(d: &Device) -> Option<(u32, &[u32])> {
    (d.push_lo < d.push_hi).then(|| (d.push_lo, &d.push_words[d.push_lo as usize..d.push_hi as usize]))
}

/// # Safety
/// `device` is valid.
pub unsafe fn borgvk_send_blend_state_serial(device: *const Device) {
    let d = unsafe { &*device };
    unsafe {
        if d.state_valid {
            borgvk_serial_send_state(d.state_reg.as_ptr());
        }
        if let Some((lo, words)) = push_range(d) {
            borgvk_serial_send_push_constants(lo * 4, words.len() as u32 * 4, words.as_ptr().cast());
        }
        if d.vp.valid {
            borgvk_serial_send_viewport(&d.vp);
        }
    }
    if d.blend_cfg != 0 {
        borgvk_serial_send_blend(d.blend_cfg, d.blend_const);
    }
}

fn send_blend_state_drm(d: &Device) {
    let fd = d.drm_fd;
    if d.state_valid && !ioctl(fd, STATE, &mut drm_borg_state { reg: d.state_reg }) {
        log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_STATE failed".into());
    }
    if let Some((lo, words)) = push_range(d) {
        let mut p = drm_borg_push { off_words: lo, n_words: words.len() as u32, ..Default::default() };
        p.words[..words.len()].copy_from_slice(words);
        if !ioctl(fd, PUSH, &mut p) {
            log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_PUSH failed".into());
        }
    }
    if d.blend_cfg != 0 && !ioctl(fd, BLEND, &mut drm_borg_blend { cfg: d.blend_cfg, constant: d.blend_const }) {
        log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_BLEND failed".into());
    }
}

/// # Safety
/// `device` is valid.
pub unsafe fn borgvk_send_blend_state(device: *const Device) {
    let d = unsafe { &*device };
    if d.drm_fd >= 0 {
        send_blend_state_drm(d);
    } else {
        unsafe { borgvk_send_blend_state_serial(d) };
    }
}

/// # Safety
/// `device` is valid.
pub unsafe fn borgvk_upload_shaders_serial(device: *mut Device) {
    for (st, b) in unsafe { &*device }.shader_blob.iter().enumerate().filter(|(_, b)| b.len != 0) {
        unsafe { borgvk_serial_send_shader(st as u8, b.data.as_ptr(), b.len) };
    }
}

/// # Safety
/// `device` is valid.
pub unsafe fn borgvk_upload_shaders(device: *mut Device) {
    let d = unsafe { &*device };
    if d.drm_fd < 0 {
        unsafe { borgvk_upload_shaders_serial(device) };
        return;
    }
    for (st, b) in d.shader_blob.iter().enumerate().filter(|(_, b)| b.len != 0) {
        let mut s = drm_borg_shader { stage: st as u32, len: b.len, ..unsafe { std::mem::zeroed() } };
        let n = b.len as usize;
        // The DRM device takes BORG_SHADER_MAX bytes; C copied longer blobs past the end.
        if n > s.data.len() || !{
            s.data[..n].copy_from_slice(&b.data[..n]);
            ioctl(d.drm_fd, SHADER, &mut s)
        } {
            log(MESA_LOG_WARN, format!("borgvk: DRM_IOCTL_BORG_SHADER failed (stage {st})"));
        }
    }
}
