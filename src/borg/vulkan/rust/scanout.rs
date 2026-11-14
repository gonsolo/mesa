// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
//! The board's HDMI scan-out (MigScanout): on real hardware, the frame WSI copied out for a
//! present goes to one of two front buffers, and the descriptor at BORG_SCANOUT_SPI points at it.

use crate::entrypoint::entrypoint;
use crate::ffi::*;
use crate::image::Image;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

struct Frame {
    w: u32,
    h: u32,
    pitch: u32,
    bgra: bool,
    bytes: Vec<u8>,
}

static LAST: Mutex<Option<Frame>> = Mutex::new(None);
static FRONT: AtomicU32 = AtomicU32::new(0);
/// KMS shows the frames (VK_KHR_display): the board's own scan-out stays off.
pub static KMS: AtomicBool = AtomicBool::new(false);

/// A whole BGRA8 or RGBA8 image copied to a buffer: what a present shows, if one follows.
pub fn copied(img: &Image, row_length: u32, extent: [u32; 3], bytes: impl FnOnce(usize) -> Option<Vec<u8>>) {
    let bgra = matches!(img.vk.format, VK_FORMAT_B8G8R8A8_UNORM | VK_FORMAT_B8G8R8A8_SRGB);
    let rgba = matches!(img.vk.format, VK_FORMAT_R8G8B8A8_UNORM | VK_FORMAT_R8G8B8A8_SRGB);
    let e = img.vk.extent;
    if !(bgra || rgba) || extent != [e.width, e.height, 1] || KMS.load(Ordering::Relaxed) {
        return;
    }
    let pitch = row_length.max(e.width) * 4;
    if let Some(bytes) = bytes((pitch * e.height) as usize) {
        *LAST.lock().unwrap_or_else(|p| p.into_inner()) = Some(Frame { w: e.width, h: e.height, pitch, bgra, bytes });
    }
}

/// After a present: the last copied frame, centred and cut to 640x480, to the next front buffer.
pub fn present() {
    let Some(f) = LAST.lock().unwrap_or_else(|p| p.into_inner()).take() else { return };
    let (w, h) = (f.w.min(640), f.h.min(480));
    let (x0, y0) = ((f.w - w) / 2, (f.h - h) / 2);
    let pitch = (w * 4).next_multiple_of(16);
    let mut out = vec![0u8; (pitch * h) as usize];
    for y in 0..h {
        let s = ((y0 + y) * f.pitch + x0 * 4) as usize;
        out[(y * pitch) as usize..][..(w * 4) as usize].copy_from_slice(&f.bytes[s..s + (w * 4) as usize]);
    }
    let k = FRONT.fetch_xor(1, Ordering::Relaxed);
    let base = BORG_SCANOUT_FB + k * BORG_SCANOUT_FB_BYTES;
    let desc = [base, pitch, w | h << 16, if f.bgra { 1 } else { 2 }];
    crate::dev::with(|d| {
        d.mem_write(base, &out);
        d.mem_write(BORG_SCANOUT_SPI, &desc.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    });
}

#[allow(unsafe_code)]
unsafe extern "C" {
    fn wsi_QueuePresentKHR(queue: VkQueue, info: *const std::ffi::c_void) -> VkResult;
}

entrypoint! {
    unsafe fn borgvk_QueuePresentKHR(queue: VkQueue, info: *const std::ffi::c_void) -> VkResult {
        let r = wsi_QueuePresentKHR(queue, info);
        present();
        r
    }
}
