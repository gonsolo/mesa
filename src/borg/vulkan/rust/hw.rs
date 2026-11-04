// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The Borg device (`BORGVK_HW`, `dev`): whether passes render on it, and compute jobs.

use crate::ffi::*;
use crate::list::Gpu;

fn env_set(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| !v.is_empty())
}

/// Whether passes render on the device.
pub fn enabled() -> bool {
    env_set("BORGVK_HW")
}

/// Whether the simulator or the device can render a pass of this size.
pub fn serves_extent(w: u32, h: u32, has_depth_stencil: bool) -> bool {
    if !env_set("BORGVK_SIM_DIRECT") && !enabled() {
        return false;
    }
    // Depth and stencil take memory by area: a wide, low pass fits.
    (1..=4096).contains(&w) && (1..=4096).contains(&h) && (!has_depth_stencil || w * h <= 1 << 20)
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

/// A compute job on the device: the buffers' bytes after it, in job order.
pub fn compute(job: &[u8]) -> Option<Vec<u8>> {
    crate::dev::with(|d| run_job(d, job)).flatten()
}

/// The job as register and memory accesses: program, presets and buffers in, one dispatch, the
/// buffers out.
pub fn run_job(d: &mut crate::dev::Dev, job: &[u8]) -> Option<Vec<u8>> {
    if job.len() < 48 {
        return None;
    }
    const GPR: u32 = 0;
    const IMEM: u32 = 128;
    const CONTROL: u32 = 424;
    const C_CTRL: u32 = 752;
    const C_PC: u32 = 756;
    const C_XY: u32 = 760;
    const C_Z: u32 = 764;
    const C_LOCAL: u32 = 768;
    const CODE_BASE: u32 = 0x318;
    const IMEM_WORDS: u32 = 72;
    const CODE_ADDR: u32 = 0xF0000;
    let mut j = Job { b: job, p: 0 };
    let h: [u32; 12] = std::array::from_fn(|_| j.take());
    let [nprog, gx, gy, gz, lx, ly, lz, nregs, nbufs, bx, by, bz] = h;
    let prog: Vec<u32> = (0..nprog).map(|_| j.take()).collect();
    d.reg(CONTROL, 2);
    if nprog > IMEM_WORDS {
        let mut code = prog.clone();
        code.push(0);
        d.mem_write(CODE_ADDR, unsafe { std::slice::from_raw_parts(code.as_ptr().cast(), code.len() * 4) });
        d.reg(CODE_BASE, CODE_ADDR);
    }
    for (i, &w) in prog.iter().take(IMEM_WORDS as usize).enumerate() {
        d.reg(IMEM + 4 * i as u32, w);
    }
    if nprog < IMEM_WORDS {
        d.reg(IMEM + 4 * nprog, 0);
    }
    for _ in 0..nregs {
        let (r, v) = (j.take(), j.take());
        d.reg(GPR + 4 * r, v);
    }
    if bx | by | bz != 0 {
        d.reg(GPR + 4 * 12, bx);
        d.reg(GPR + 4 * 13, by);
        d.reg(GPR + 4 * 14, bz);
    }
    if nbufs > BORGVK_CS_MAX_BUFS {
        return None;
    }
    let mut bufs = Vec::new();
    for _ in 0..nbufs {
        let (addr, size) = (j.take(), j.take());
        let padded = (size + 3) & !3;
        d.mem_write(addr, &job[j.p..j.p + padded as usize]);
        j.p += padded as usize;
        bufs.push((addr, size));
    }
    d.reg(C_PC, 0);
    d.reg(C_XY, gx | gy << 16);
    d.reg(C_Z, gz);
    d.reg(C_LOCAL, lx | ly << 8 | lx.wrapping_mul(ly).wrapping_mul(lz) << 16);
    d.reg(C_CTRL, 1);
    let st = d.wait(C_CTRL, 1, 1);
    if st & 1 == 0 || st & 8 != 0 || !d.ok {
        return None;
    }
    let mut out = Vec::new();
    for (addr, size) in bufs {
        let mut tmp = vec![0u8; ((size + 3) & !3) as usize];
        d.mem_read(addr, &mut tmp);
        out.extend_from_slice(&tmp[..size as usize]);
    }
    d.ok.then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serves_needs_a_backend() {
        std::env::remove_var("BORGVK_SIM_DIRECT");
        std::env::remove_var("BORGVK_HW");
        assert!(!serves_extent(64, 64, false));
    }
}
