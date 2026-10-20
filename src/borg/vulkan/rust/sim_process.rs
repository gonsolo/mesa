// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The cube path: one render of a whole recorded stream, by `hw::render` on the device or
//! by simulator processes that write the frame to stdout. Wide frames are split into windows of
//! 128 pixels, dealt to several processes.

use crate::ffi::*;
use crate::image::Image;
use crate::format::raw32;
use crate::util::{log, mkstemp, MESA_LOG_INFO};
use std::ffi::{OsStr, OsString};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::FromRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::process::{Child, Command, Stdio};

const WIN: u32 = 128;

fn env_set(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|v| !v.is_empty())
}

/// The frame into the image: cropped when it is up to 3 pixels larger, else scaled.
fn store_rgb(img: &Image, rgb: Option<&[u8]>, raw32: bool, sw: u32, sh: u32) {
    let (width, height) = (img.vk.extent.width, img.vk.extent.height);
    let Some(rgb) = rgb else {
        return;
    };
    if let Some(ppm) = env_set("BORGVK_DUMP_PPM").filter(|_| !raw32) {
        // the rendered frame as it came from the Borg
        if let Ok(mut f) = std::fs::File::create(ppm) {
            let _ = write!(f, "P6\n{sw} {sh}\n255\n").and_then(|_| f.write_all(rgb));
        }
    }
    let samples = img.vk.samples.max(1);
    let mapped = img.host().is_some();
    if !mapped || width as u64 * height as u64 * 4 * samples as u64 > img.size {
        return;
    }
    let base = unsafe { ((*img.mem).map as *mut u8).add(img.offset as usize) };
    if raw32 {
        unsafe { std::ptr::copy_nonoverlapping(rgb.as_ptr(), base, rgb.len()) };
        return;
    }
    let crop = sw >= width && sh >= height && sw - width < 4 && sh - height < 4;
    let pick = |i: u32, n: u32, s: u32| if crop { i } else { ((2 * i as u64 + 1) * s as u64 / (2 * n as u64)) as u32 };
    let plane = width as usize * height as usize * 4;
    for smp in 0..samples as usize {
        let dst = unsafe { std::slice::from_raw_parts_mut(base.add(smp * plane), plane) };
        for y in 0..height {
            let sy = pick(y, height, sh);
            for x in 0..width {
                let sx = pick(x, width, sw);
                let s = (sy as usize * sw as usize + sx as usize) * 3;
                let d = (y as usize * width as usize + x as usize) * 4;
                dst[d..d + 3].copy_from_slice(&rgb[s..s + 3]);
                dst[d + 3] = 255;
            }
        }
    }
}

fn choose_dim(img: &Image, sim_dim: u32, raw32: bool) -> u32 {
    let e = img.vk.extent;
    let mut sdim = if raw32 { e.width } else if sim_dim != 0 { sim_dim } else { 128 };
    if env_set("BORGVK_SIM_DIRECT").is_some() || crate::hw::enabled() {
        let d = e.width;
        if (4..=256).contains(&d) && d.is_power_of_two() && e.height == d {
            sdim = d;
        }
    }
    if let Some(d) = env_set("BORGVK_SIM_DIM").filter(|_| sim_dim == 0) {
        let d = crate::sim::atoi(&d.to_string_lossy());
        if d > 0 {
            sdim = d as u32;
        }
    }
    sdim
}

fn keep_stream(bytes: &[u8]) {
    if let Some(keep) = env_set("BORGVK_SIM_KEEP") {
        if let Ok(mut f) = OpenOptions::new().write(true).create(true).truncate(true).mode(0o644).open(keep) {
            let _ = f.write_all(bytes);
        }
    }
}

/// The stream in a temporary file for the simulators to read; removed on drop.
struct UartFile(std::path::PathBuf);

impl UartFile {
    fn new(bytes: &[u8]) -> Option<Self> {
        let mut template = *b"/tmp/borgvk_uart_XXXXXX\0";
        let fd = unsafe { mkstemp(template.as_mut_ptr().cast()) };
        if fd < 0 {
            return None;
        }
        let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
        let _ = f.write_all(bytes);
        let path = std::str::from_utf8(&template[..template.len() - 1]).ok()?.into();
        Some(UartFile(path))
    }
}

impl Drop for UartFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// How many processes: up to one per CPU and window, at most `BORGVK_SIM_PARTS` and 64.
fn parts(nwx: u32, direct: bool) -> u32 {
    if !direct || nwx <= 1 {
        return 1;
    }
    let mut n = (unsafe { sysconf(_SC_NPROCESSORS_ONLN as i32) }).max(1) as u32;
    if let Some(p) = std::env::var_os("BORGVK_SIM_PARTS") {
        let p = crate::sim::atoi(&p.to_string_lossy());
        if p > 0 {
            n = n.min(p as u32);
        }
    }
    n.min(nwx * nwx).min(64)
}

/// Starts process `k` of `n`: the direct simulator, or the firmware simulator when that fails.
fn spawn(k: u32, n: u32, nwx: u32, uart: &OsStr, dim: &str) -> Option<Child> {
    let direct = env_set("BORGVK_SIM_DIRECT");
    let windowed = |c: &mut Command| {
        if direct.is_some() && nwx > 1 {
            c.env("BORG_PART", format!("{k}/{n}")).env("BORG_WINDOW_TILES", "32");
        }
        c.stdout(Stdio::piped());
    };
    if let Some(d) = &direct {
        let mut c = Command::new(d);
        c.arg(uart).arg(dim).arg(dim);
        windowed(&mut c);
        if let Ok(child) = c.spawn() {
            return Some(child);
        }
    }
    let mut c = Command::new(std::env::var_os("BORGVK_SIM")?);
    c.arg("--cts-uart").arg(uart);
    // As execlp: without firmware the argument list ends there.
    if let Some(fw) = std::env::var_os("BORGVK_SIM_FW") {
        c.arg(fw).arg(dim).arg(dim);
    }
    windowed(&mut c);
    c.spawn().ok()
}

fn read_frame(child: &mut Child, out: &mut [u8]) -> usize {
    let mut got = 0;
    if let Some(stdout) = child.stdout.as_mut() {
        while got < out.len() {
            match stdout.read(&mut out[got..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
    }
    child.stdout = None;
    let _ = child.wait();
    got
}

fn run_processes(bytes: &[u8], img: &Image, sdim: u32, raw32: bool) {
    let Some(uart) = UartFile::new(bytes) else {
        return;
    };
    let nwx = if sdim > WIN { sdim / WIN } else { 1 };
    let n = parts(nwx, env_set("BORGVK_SIM_DIRECT").is_some());
    let bpp = if raw32 { 4 } else { 3 };
    let expected = sdim as usize * sdim as usize * bpp;
    let dim = sdim.to_string();
    let mut children: Vec<Option<Child>> = (0..n).map(|k| spawn(k, n, nwx, uart.0.as_os_str(), &dim)).collect();
    let mut rgb = vec![0u8; expected];
    let got = if n == 1 {
        children[0].as_mut().map_or(0, |c| read_frame(c, &mut rgb))
    } else {
        let mut piece = vec![0u8; expected];
        let mut ok = 0;
        for (k, child) in children.iter_mut().enumerate() {
            let Some(child) = child else {
                continue;
            };
            if read_frame(child, &mut piece) != expected {
                continue;
            }
            for y in 0..sdim {
                for x in 0..sdim {
                    if ((y / WIN) * nwx + x / WIN) % n == k as u32 {
                        let i = (y as usize * sdim as usize + x as usize) * bpp;
                        rgb[i..i + bpp].copy_from_slice(&piece[i..i + bpp]);
                    }
                }
            }
            ok += 1;
        }
        if ok == n { expected } else { 0 }
    };
    drop(uart);
    if std::env::var_os("BORGVK_DEBUG").is_some() {
        log(MESA_LOG_INFO, format!("borgvk: sim returned {got} of {expected} bytes"));
    }
    store_rgb(img, (got == expected).then_some(&rgb[..]), raw32, sdim, sdim);
}

fn run_stream(bytes: &[u8], img: &Image, sim_dim: u32) {
    let e = img.vk.extent;
    let raw32 = raw32(img.vk.format) && e.width == e.height;
    let sdim = choose_dim(img, sim_dim, raw32);
    keep_stream(bytes);
    if crate::hw::enabled() {
        let (mut sw, mut sh) = (sdim, sdim);
        let (rw, rh) = ((e.width + 3) & !3, (e.height + 3) & !3);
        if !raw32 && rw <= 1024 && rh <= 1024 {
            (sw, sh) = (rw, rh);
        }
        let mut rgb = vec![0u8; sw as usize * sh as usize * if raw32 { 4 } else { 3 }];
        let got = crate::hw::render(bytes, sw, sh, rgb.as_mut_ptr());
        store_rgb(img, (got == rgb.len()).then_some(&rgb[..]), raw32, sw, sh);
        return;
    }
    run_processes(bytes, img, sdim, raw32);
}

/// Renders `stream` into `color_img`.
pub fn sim_run_stream(stream: &[u8], color_img: Option<&Image>, sim_dim: u32) -> VkResult {
    if let Some(img) = color_img {
        let mapped = img.host().is_some();
        if mapped && img.vk.extent.width != 0 && img.vk.extent.height != 0 {
            run_stream(stream, img, sim_dim);
        }
    }
    VK_SUCCESS
}

#[cfg(test)]
mod tests {
    use crate::memory::DeviceMemory;
    use super::*;

    #[test]
    fn one_part_without_windows() {
        assert_eq!(parts(1, true), 1);
        assert_eq!(parts(4, false), 1);
    }

    /// With BORGVK_TEST_DIRECT_SIM and BORGVK_TEST_CAPTURE set: the frame through run_processes
    /// equals direct_sim's own, in one process and in several.
    #[test]
    fn processes_render_like_direct_sim() {
        let (Some(sim), Some(cap)) = (std::env::var_os("BORGVK_TEST_DIRECT_SIM"), std::env::var_os("BORGVK_TEST_CAPTURE")) else {
            return;
        };
        let bytes = std::fs::read(&cap).unwrap();
        std::env::set_var("BORGVK_SIM_DIRECT", &sim);
        for dim in [128u32, 256] {
            let out = Command::new(&sim).arg(&cap).arg(dim.to_string()).arg(dim.to_string()).output().unwrap();
            assert_eq!(out.stdout.len(), (dim * dim * 3) as usize);
            let mut pixels = vec![0u8; (dim * dim * 4) as usize];
            let mut mem = DeviceMemory { map: pixels.as_mut_ptr().cast(), ..Default::default() };
            let mut img = Image { size: pixels.len() as u64, mem: &mut mem, ..Default::default() };
            img.vk.extent = VkExtent3D { width: dim, height: dim, depth: 1 };
            img.vk.samples = 1;
            run_processes(&bytes, &img, dim, false);
            for (d, s) in pixels.chunks(4).zip(out.stdout.chunks(3)) {
                assert_eq!((&d[..3], d[3]), (s, 255));
            }
        }
    }
}
