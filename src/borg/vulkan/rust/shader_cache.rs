// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Compiled shader blobs by a hash of their SPIR-V, entry point, stage and fetch options:
//! read from BORGVK_SHADER_CACHE, appended to BORGVK_SHADER_CACHE_RECORD. Builds without the
//! compiler take their shaders from here.

use crate::entrypoint::chain;
use crate::ffi::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::OnceLock;

const MAX: usize = BORGVK_SHADER_BLOB_MAX as usize;

struct Entry {
    key: u64,
    flags: u8,
    data: Vec<u8>,
}

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn env_path(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|p| !p.is_empty())
}

/// The SPIR-V of a stage: its module's, or one chained in place of a module.
fn code(si: &VkPipelineShaderStageCreateInfo) -> Option<&[u8]> {
    let (p, n) = if !si.module.is_null() {
        let m = unsafe { &*si.module.cast::<vk_shader_module>() };
        (m.data.as_ptr().cast::<u8>(), m.size as usize)
    } else {
        let ci = unsafe { chain(si.pNext as *mut _) }.find(|&s| unsafe { (*s).sType } == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)?;
        let ci = unsafe { &*ci.cast::<VkShaderModuleCreateInfo>() };
        (ci.pCode.cast::<u8>(), ci.codeSize)
    };
    (!p.is_null() && n != 0).then(|| unsafe { std::slice::from_raw_parts(p, n) })
}

fn key(si: &VkPipelineShaderStageCreateInfo, vfetch: u32) -> Option<u64> {
    let name = unsafe { std::ffi::CStr::from_ptr(si.pName) }.to_bytes();
    let h = fnv(0xcbf29ce484222325, code(si)?);
    let h = fnv(fnv(h, name), &si.stage.to_ne_bytes());
    Some(fnv(h, &vfetch.to_ne_bytes()))
}

fn read_entries(mut f: impl Read) -> Vec<Entry> {
    let mut out = Vec::new();
    loop {
        let mut head = [0u8; 11];
        if f.read_exact(&mut head).is_err() {
            return out;
        }
        let key = u64::from_ne_bytes(head[..8].try_into().unwrap());
        let len = u16::from_ne_bytes([head[9], head[10]]) as usize;
        let mut data = vec![0; len];
        if len > MAX || f.read_exact(&mut data).is_err() {
            return out;
        }
        out.push(Entry { key, flags: head[8], data });
    }
}

fn entries() -> &'static [Entry] {
    static CACHE: OnceLock<Vec<Entry>> = OnceLock::new();
    CACHE.get_or_init(|| env_path("BORGVK_SHADER_CACHE").and_then(|p| File::open(p).ok()).map_or_else(Vec::new, read_entries))
}

/// The cached blob of a stage and whether it reads gl_PointCoord.
pub fn get(si: &VkPipelineShaderStageCreateInfo, vfetch: u32, blob: &mut borgvk_shader_blob) -> Option<bool> {
    let cache = entries();
    if cache.is_empty() {
        return None;
    }
    let key = key(si, vfetch)?;
    let e = cache.iter().find(|e| e.key == key)?;
    blob.data[..e.data.len()].copy_from_slice(&e.data);
    blob.len = e.data.len() as u32;
    Some(e.flags & 1 != 0)
}

pub fn record(si: &VkPipelineShaderStageCreateInfo, vfetch: u32, blob: &borgvk_shader_blob, reads_pntc: bool) {
    let Some(path) = env_path("BORGVK_SHADER_CACHE_RECORD") else {
        return;
    };
    let Some(key) = key(si, vfetch).filter(|_| blob.len != 0) else {
        return;
    };
    let Ok(mut f) = OpenOptions::new().append(true).create(true).open(path) else {
        return;
    };
    let len = blob.len as u16;
    let mut rec = key.to_ne_bytes().to_vec();
    rec.push(reads_pntc as u8);
    rec.extend_from_slice(&len.to_ne_bytes());
    rec.extend_from_slice(&blob.data[..len as usize]);
    let _ = f.write_all(&rec);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_of_known_strings() {
        assert_eq!(fnv(0xcbf29ce484222325, b""), 0xcbf29ce484222325);
        assert_eq!(fnv(0xcbf29ce484222325, b"a"), 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn entries_read_until_a_short_one() {
        let mut f = Vec::new();
        for (k, flags, d) in [(7u64, 1u8, &b"abc"[..]), (9, 0, &b"zz"[..])] {
            f.extend_from_slice(&k.to_ne_bytes());
            f.push(flags);
            f.extend_from_slice(&(d.len() as u16).to_ne_bytes());
            f.extend_from_slice(d);
        }
        f.extend_from_slice(&[1, 2, 3]);
        let e = read_entries(&f[..]);
        assert_eq!(e.len(), 2);
        assert_eq!((e[0].key, e[0].flags, &e[0].data[..]), (7, 1, &b"abc"[..]));
        assert_eq!((e[1].key, &e[1].data[..]), (9, &b"zz"[..]));
    }

    #[test]
    fn reads_the_boards_cache_file() {
        let f = include_bytes!("../../../../../software/vkcube/cube.shaders");
        let e = read_entries(&f[..]);
        assert_eq!(e.len(), 2);
        assert_eq!(e.iter().map(|e| 11 + e.data.len()).sum::<usize>(), f.len());
    }
}
