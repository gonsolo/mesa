// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! A draw's state and shaders as list builder commands.

use crate::device::Scratch;
use crate::wire::*;

/// Clamp to edge in U, V and W, nearest.
pub static DEFAULT_SAMPLER: [u32; 4] = [2 << 3 | 2 << 6 | 2 << 9, 0, 0, 0];

pub fn send_draw_state(sc: &Scratch, push: Option<(u32, &[u32])>) {
    if sc.state_valid {
        send_state(&sc.state_reg);
    }
    if let Some((lo, words)) = push {
        send_push_constants(lo * 4, &words.iter().flat_map(|w| w.to_ne_bytes()).collect::<Vec<u8>>());
    }
    if sc.vp.valid {
        send_viewport(&sc.vp);
    }
    if sc.bias != [0.0; 2] {
        send_depth_bias(sc.bias);
    }
    if sc.blend_cfg != 0 {
        send_blend(sc.blend_cfg, sc.blend_const);
    }
}

pub fn send_shaders(sc: &Scratch) {
    for (st, b) in sc.shader_blob.iter().enumerate().filter(|(_, b)| b.len != 0) {
        send_shader(st as u8, &b.data[..b.len as usize]);
    }
}
