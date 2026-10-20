// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Shaders for pipelines: from the shader cache, or through NIR and borgc.

use crate::ffi::*;

/// The number of varyings a vertex shader blob writes, from its header; 0 without the flag.
pub fn num_varyings(b: &[u8]) -> u32 {
    if b.len() < 6 || b[5] & 1 == 0 {
        return 0;
    }
    let at = 6 + 4 * b[0] as usize + b[3] as usize + 5 * b[4] as usize;
    b.get(at).map_or(0, |&n| n as u32)
}

/// # Safety
/// `b` is a valid blob.
#[no_mangle]
pub unsafe extern "C" fn borgvk_blob_num_varyings(b: *const borgvk_shader_blob) -> u32 {
    let b = unsafe { &*b };
    num_varyings(&b.data[..(b.len as usize).min(b.data.len())])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varyings_follow_the_header_tables() {
        let mut b = vec![0u8; 32];
        b[0] = 1; // one word of something
        b[3] = 2;
        b[4] = 1;
        b[5] = 1;
        b[6 + 4 + 2 + 5] = 7;
        assert_eq!(num_varyings(&b), 7);
        b[5] = 0;
        assert_eq!(num_varyings(&b), 0);
        assert_eq!(num_varyings(&b[..4]), 0);
    }
}
