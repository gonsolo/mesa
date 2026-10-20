// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The driver corpus (scripts/borgvk_corpus): with BORGVK_CORPUS_DIR set, every recorded
//! output as a line "kind key content" in <dir>/<pid>.log.

use crate::ffi::*;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::OnceLock;

fn file() -> Option<&'static File> {
    static FILE: OnceLock<Option<File>> = OnceLock::new();
    FILE.get_or_init(|| {
        let dir = std::env::var_os("BORGVK_CORPUS_DIR")?;
        let mut path = std::path::PathBuf::from(dir);
        path.push(format!("{}.log", std::process::id()));
        OpenOptions::new().create(true).append(true).mode(0o644).open(path).ok()
    })
    .as_ref()
}

pub fn hash(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

#[no_mangle]
pub extern "C" fn borgvk_corpus_on() -> bool {
    file().is_some()
}

/// # Safety
/// `p` holds `n` bytes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_corpus_hash(h: u64, p: *const std::ffi::c_void, n: usize) -> u64 {
    if n == 0 { h } else { hash(h, unsafe { std::slice::from_raw_parts(p.cast(), n) }) }
}

/// # Safety
/// `img` is null or a valid image.
#[no_mangle]
pub unsafe extern "C" fn borgvk_corpus_image(h: u64, img: *const borgvk_image) -> u64 {
    let Some(img) = (unsafe { img.as_ref() }) else { return h };
    match unsafe { img.mem.as_ref() }.filter(|m| !m.map.is_null()) {
        Some(m) => unsafe { borgvk_corpus_hash(h, m.map.byte_add(img.offset as usize), img.size as usize) },
        None => h,
    }
}

#[no_mangle]
pub extern "C" fn borgvk_corpus_log(kind: std::ffi::c_char, key: u64, content: u64) {
    if let Some(mut f) = file() {
        let _ = f.write_all(format!("{} {key:016x} {content:016x}\n", kind as u8 as char).as_bytes());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hash_is_fnv1a() {
        assert_eq!(super::hash(0xcbf29ce484222325, b"a"), 0xaf63dc4c8601ec8c);
    }
}
