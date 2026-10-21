// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The Borg device (`BORGVK_HW`): renders and compute jobs run by `borg_core` and `borg_hw` in
//! this process, over the DRM render node.

use crate::ffi::*;
use crate::util::{log, MESA_LOG_WARN};

fn env_set(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| !v.is_empty())
}

/// Whether passes and the cube path render on the device.
pub fn enabled() -> bool {
    env_set("BORGVK_HW")
}

/// Whether the simulator or the device can render a pass of this size.
pub fn serves_extent(w: u32, h: u32, has_depth_stencil: bool) -> bool {
    if !env_set("BORGVK_SIM_DIRECT") && !enabled() {
        return false;
    }
    let max = if has_depth_stencil { 1024 } else { 4096 };
    (1..=max).contains(&w) && (1..=max).contains(&h)
}

/// Pixel (x, y) of a flushed 4x4-tiled frame: RGB for R5G6B5 (0), RGBA8 (1) and BGRA8 (2), the
/// raw word for RAW32 (3).
fn pixel(fmt: i32, words: &[u32], w: u32, x: u32, y: u32) -> ([u8; 4], usize) {
    let (tile, ti) = borg_layout::tile::index(w, x, y);
    match fmt {
        3 => (words[tile * 16 + ti].to_ne_bytes(), 4),
        0 => {
            let word = words[tile * 8 + (ti >> 1)];
            let px = if ti & 1 != 0 { word >> 16 } else { word & 0xFFFF };
            let (r, g, b) = ((((px >> 11) & 0x1F) << 3) as u8, (((px >> 5) & 0x3F) << 2) as u8, ((px & 0x1F) << 3) as u8);
            ([r | r >> 5, g | g >> 6, b | b >> 5, 0], 3)
        }
        _ => {
            let [b0, b1, b2, _] = words[tile * 16 + ti].to_le_bytes();
            let (r, b) = if fmt == 1 { (b0, b2) } else { (b2, b0) };
            ([r, b1, b, 0], 3)
        }
    }
}

/// Renders the stream on the device into `out`: `w * h` pixels of 3 bytes, 4 for RAW32.
pub fn render(b: &[u8], w: u32, h: u32, out: *mut u8) -> usize {
    if unsafe { borg_hw_open() } != 0 {
        log(MESA_LOG_WARN, "borgvk: cannot open the Borg device".into());
        return 0;
    }
    unsafe {
        borg_core_set_flush_format(0);
        borg_core_init(w as i32, h as i32);
    }
    let (mut draws, mut target_seen, mut i) = (0, false, 0);
    while i < b.len() {
        if b[i] == 0xB1 {
            i += 1; // serial-reload trigger: not a draw packet
            continue;
        }
        let len = unsafe { borg_core_pkt_len(b[i]) } as usize;
        if len == 0 || i + len > b.len() {
            log(MESA_LOG_WARN, format!("borgvk: bad packet 0x{:02x} at byte {i}", b[i]));
            break;
        }
        let kind = unsafe { borg_core_packet(b[i..].as_ptr()) };
        if kind == BC_BAD as i32 {
            log(MESA_LOG_WARN, format!("borgvk: rejected packet 0x{:02x} at byte {i}", b[i]));
            break;
        }
        target_seen |= kind == BC_TARGET as i32;
        if kind == BC_DRAW as i32 || (kind == BC_MVP as i32 && unsafe { borg_core_ready() } != 0) {
            unsafe {
                if !target_seen {
                    borg_core_set_clear(0x3266, 0x3266, 0x3266);
                }
                if kind == BC_DRAW as i32 {
                    borg_core_list_draw();
                } else {
                    borg_core_draw(std::ptr::null(), 0);
                }
            }
            draws += 1;
        }
        i += len;
    }
    unsafe {
        borg_core_list_flush();
        borg_hw_flush();
    }
    if draws == 0 {
        return 0;
    }
    let fmt = unsafe { borg_core_flush_format() };
    let mut words = vec![0u32; (w * h * if fmt != 0 { 4 } else { 2 }) as usize / 4];
    unsafe { borg_hw_mem_read(DRAM_OUT_BASE_SPI, words.as_mut_ptr().cast(), (words.len() * 4) as u32) };
    let mut n = 0;
    for y in 0..h {
        for x in 0..w {
            let (px, bpp) = pixel(fmt, &words, w, x, y);
            let o = (y * w + x) as usize * bpp;
            unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), out.add(o), bpp) };
            n = bpp;
        }
    }
    (w * h) as usize * n
}

/// A compute job as borgvk builds it (borgvk_cmd_dispatch.c): 12 header words, the program,
/// register presets and buffers.
struct Job<'a> {
    b: &'a [u8],
    p: usize,
}

impl Job<'_> {
    fn word(&self, at: usize) -> u32 {
        u32::from_ne_bytes(self.b[at..at + 4].try_into().unwrap())
    }

    fn take(&mut self) -> u32 {
        self.p += 4;
        self.word(self.p - 4)
    }
}

/// The register/memory program of direct_sim --compute (simulation/direct/main.cpp), on the
/// device, its output written to `path`.
pub fn compute(job: &[u8], path: &std::path::Path) -> bool {
    const GPR: u32 = 0;
    const IMEM: u32 = 128;
    const CONTROL: u32 = 424;
    const LS_BASE: u32 = 724;
    const C_CTRL: u32 = 752;
    const C_PC: u32 = 756;
    const C_XY: u32 = 760;
    const C_Z: u32 = 764;
    const C_LOCAL: u32 = 768;
    const CODE_BASE: u32 = 0x318;
    const IMEM_WORDS: u32 = 72;
    const CODE_ADDR: u32 = 0xF0000;
    if job.len() < 48 || unsafe { borg_hw_open() } != 0 {
        return false;
    }
    let mut j = Job { b: job, p: 0 };
    let h: [u32; 12] = std::array::from_fn(|_| j.take());
    let [nprog, gx, gy, gz, lx, ly, lz, nregs, nbufs, bx, by, bz] = h;
    let prog: Vec<u32> = (0..nprog).map(|_| j.take()).collect();
    let reg = |r: u32, v: u32| unsafe { bh_reg_write(r, v) };
    reg(CONTROL, 2);
    reg(LS_BASE, 0);
    if nprog > IMEM_WORDS {
        let mut code = prog.clone();
        code.push(0);
        unsafe { borg_hw_mem_write(CODE_ADDR, code.as_ptr().cast(), (code.len() * 4) as u32) };
        reg(CODE_BASE, CODE_ADDR);
    }
    for (i, &w) in prog.iter().take(IMEM_WORDS as usize).enumerate() {
        reg(IMEM + 4 * i as u32, w);
    }
    if nprog < IMEM_WORDS {
        reg(IMEM + 4 * nprog, 0);
    }
    for _ in 0..nregs {
        let (r, v) = (j.take(), j.take());
        reg(GPR + 4 * r, v);
    }
    if bx | by | bz != 0 {
        reg(GPR + 4 * 12, bx);
        reg(GPR + 4 * 13, by);
        reg(GPR + 4 * 14, bz);
    }
    if nbufs > BORGVK_CS_MAX_BUFS {
        return false;
    }
    let mut bufs = Vec::new();
    for _ in 0..nbufs {
        let (addr, size) = (j.take(), j.take());
        let padded = (size + 3) & !3;
        unsafe { borg_hw_mem_write(addr, job[j.p..].as_ptr().cast(), padded) };
        j.p += padded as usize;
        bufs.push((addr, size));
    }
    reg(C_PC, 0);
    reg(C_XY, gx | gy << 16);
    reg(C_Z, gz);
    reg(C_LOCAL, lx | ly << 8 | lx.wrapping_mul(ly).wrapping_mul(lz) << 16);
    reg(C_CTRL, 1);
    let mut st = 0;
    for _ in 0..2_000_000 {
        st = unsafe { bh_reg_read(C_CTRL) };
        if st & 1 != 0 {
            break;
        }
    }
    if st & 1 == 0 || st & 8 != 0 {
        return false;
    }
    let mut out = Vec::new();
    for (addr, size) in bufs {
        let mut tmp = vec![0u8; ((size + 3) & !3) as usize];
        unsafe { borg_hw_mem_read(addr, tmp.as_mut_ptr().cast(), tmp.len() as u32) };
        out.extend_from_slice(&tmp[..size as usize]);
    }
    std::fs::write(path, out).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb565_pixels_expand_to_eight_bits() {
        // Tile 0 of a 4-wide frame: pixel 0 white in the low half, pixel 1 red in the high half.
        let mut words = vec![0u32; 8];
        words[0] = 0xF800 << 16 | 0xFFFF;
        assert_eq!(pixel(0, &words, 4, 0, 0), ([255, 255, 255, 0], 3));
        assert_eq!(pixel(0, &words, 4, 1, 0), ([255, 0, 0, 0], 3));
    }

    #[test]
    fn rgba_and_bgra_pixels() {
        let mut words = vec![0u32; 32];
        words[16 + 5] = 0x0044_3322; // tile 1, pixel (1, 1) of a frame 8 wide
        assert_eq!(pixel(1, &words, 8, 5, 1).0[..3], [0x22, 0x33, 0x44]);
        assert_eq!(pixel(2, &words, 8, 5, 1).0[..3], [0x44, 0x33, 0x22]);
        assert_eq!(pixel(3, &words, 8, 5, 1), (0x0044_3322u32.to_ne_bytes(), 4));
    }

    #[test]
    fn serves_needs_a_backend() {
        std::env::remove_var("BORGVK_SIM_DIRECT");
        std::env::remove_var("BORGVK_HW");
        assert!(!serves_extent(64, 64, false));
    }
}
