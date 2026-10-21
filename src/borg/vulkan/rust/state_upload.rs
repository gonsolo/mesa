// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! State and shaders to the Borg: over the DRM device when there is one, else as wire packets.

use crate::drm::{ioctl, BLEND, PUSH, SHADER, STATE};
use crate::entrypoint::slice;
use crate::descriptor_set::DescriptorSet;
use crate::ffi::*;
use crate::sampler::Sampler;
use crate::device::{Device, Scratch};
use crate::wire::*;
use crate::util::{log, MESA_LOG_WARN};

/// The first descriptor set a submit binds; its command buffers and their recorded commands are
/// the runtime's, valid for the submit.
pub fn find_set(s: &vk_queue_submit) -> Option<&DescriptorSet> {
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
                if let Some(set) = unsafe { (*b.descriptor_sets).cast::<DescriptorSet>().as_ref() } {
                    return Some(set);
                }
            }
        }
    }
    None
}

/// Clamp to edge in U, V and W, nearest.
static DEFAULT_SAMPLER: [u32; 4] = [2 << 3 | 2 << 6 | 2 << 9, 0, 0, 0];

/// The sampler's descriptor, the default without one.
pub fn sampler_desc(sampler: Option<&Sampler>) -> &[u32; 4] {
    sampler.map_or(&DEFAULT_SAMPLER, |s| &s.desc)
}

fn push_range(sc: &Scratch) -> Option<(u32, &[u32])> {
    (sc.push_lo < sc.push_hi).then(|| (sc.push_lo, &sc.push_words[sc.push_lo as usize..sc.push_hi as usize]))
}

pub fn send_blend_state_serial(sc: &Scratch) {
    if sc.state_valid {
        send_state(&sc.state_reg);
    }
    if let Some((lo, words)) = push_range(sc) {
        send_push_constants(lo * 4, &words.iter().flat_map(|w| w.to_ne_bytes()).collect::<Vec<u8>>());
    }
    if sc.vp.valid {
        send_viewport(&sc.vp);
    }
    if sc.blend_cfg != 0 {
        send_blend(sc.blend_cfg, sc.blend_const);
    }
}

fn send_blend_state_drm(fd: i32, sc: &Scratch) {
    if sc.state_valid && !ioctl(fd, STATE, &mut drm_borg_state { reg: sc.state_reg }) {
        log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_STATE failed".into());
    }
    if let Some((lo, words)) = push_range(sc) {
        let mut p = drm_borg_push { off_words: lo, n_words: words.len() as u32, ..Default::default() };
        p.words[..words.len()].copy_from_slice(words);
        if !ioctl(fd, PUSH, &mut p) {
            log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_PUSH failed".into());
        }
    }
    if sc.blend_cfg != 0 && !ioctl(fd, BLEND, &mut drm_borg_blend { cfg: sc.blend_cfg, constant: sc.blend_const }) {
        log(MESA_LOG_WARN, "borgvk: DRM_IOCTL_BORG_BLEND failed".into());
    }
}

pub fn send_blend_state(dev: &Device, sc: &Scratch) {
    if dev.drm_fd >= 0 {
        send_blend_state_drm(dev.drm_fd, sc);
    } else {
        send_blend_state_serial(sc);
    }
}

pub fn upload_shaders_serial(sc: &Scratch) {
    for (st, b) in sc.shader_blob.iter().enumerate().filter(|(_, b)| b.len != 0) {
        send_shader(st as u8, &b.data[..b.len as usize]);
    }
}

pub fn upload_shaders(dev: &Device, sc: &Scratch) {
    if dev.drm_fd < 0 {
        upload_shaders_serial(sc);
        return;
    }
    for (st, b) in sc.shader_blob.iter().enumerate().filter(|(_, b)| b.len != 0) {
        let mut s = drm_borg_shader { stage: st as u32, len: b.len, ..unsafe { std::mem::zeroed() } };
        let n = b.len as usize;
        // The DRM device takes BORG_SHADER_MAX bytes; C copied longer blobs past the end.
        if n > s.data.len() || !{
            s.data[..n].copy_from_slice(&b.data[..n]);
            ioctl(dev.drm_fd, SHADER, &mut s)
        } {
            log(MESA_LOG_WARN, format!("borgvk: DRM_IOCTL_BORG_SHADER failed (stage {st})"));
        }
    }
}
