// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The driver's limits and plain data, shared by the modules through `ffi`.

use crate::ffi::{pipe_format, VkFormat, PIPE_FORMAT_NONE};

pub const BORGVK_STAGE_VERT: u32 = 0;
pub const BORGVK_STAGE_FRAG: u32 = 1;
pub const BORGVK_SHADER_STAGE_COUNT: u32 = 2;
pub const BORGVK_SHADER_BLOB_MAX: u32 = 32768;
pub const BORGVK_POOL_MAX: u32 = 64;
pub const BORGVK_POOL_SLOT: u32 = 40;
pub const BORGVK_VPOOL_SLOT: u32 = 42;
// borgc constant_load: a stage's constant data, rows of four words.
pub const BORGVK_CDATA_SLOT: u32 = 41;
pub const BORGVK_VCDATA_SLOT: u32 = 23;
pub const BORGVK_CDATA_MAX: u32 = 4096;
pub const BORGVK_LINE_SLOT: u32 = 16;
pub const BORGVK_LINE_OTHER: u32 = 24;
pub const BORGVK_MAX_COLOR_ATTACHMENTS: u32 = 8;
pub const BORGVK_MAX_VERTEX_BINDINGS: u32 = 16;
pub const BORGVK_MAX_VERTEX_ATTRS: u32 = 16;
pub const BORGVK_MAX_BINDINGS: u32 = 64;
pub const BORGVK_MAX_WRITEBACKS: u32 = 64;
pub const BORGVK_SSBO_SLOT_BYTES: u32 = 65536;
pub const BORGVK_CS_MAX_BUFS: u32 = 72; // 32 binding slots, the push constants, the constant pool and the root table
pub const BORGVK_CS_POOL_MAX: u32 = 4096;
pub const BORGVK_CS_WORDS_MAX: u32 = 16384;
pub const BORGVK_CORPUS_SEED: u64 = 0xcbf2_9ce4_8422_2325;
pub const BORGVK_TEXG_CHUNK: u32 = 256;
pub const BORGVK_TEX_PAIRS: usize = 32;

#[derive(Clone, Copy)]
pub struct ShaderBlob {
    pub data: [u8; BORGVK_SHADER_BLOB_MAX as usize],
    pub len: u32, // 0: not compiled, or did not fit
    pub pool: [u32; BORGVK_POOL_MAX as usize],
    pub npool: u32,
    pub cdata: [u32; BORGVK_CDATA_MAX as usize],
    pub ncdata: u32,
}

impl Default for ShaderBlob {
    fn default() -> Self {
        Self { data: [0; BORGVK_SHADER_BLOB_MAX as usize], len: 0, pool: [0; BORGVK_POOL_MAX as usize], npool: 0, cdata: [0; BORGVK_CDATA_MAX as usize], ncdata: 0 }
    }
}

/// A viewport and scissor as the list builder takes them.
#[derive(Clone, Copy, Default)]
pub struct Vp {
    pub valid: bool,
    pub v: [f32; 6],  // x, y, width, height, minDepth, maxDepth
    pub sc: [u32; 4], // scissor x0, y0, x1, y1
}

/// A buffer range the GPU may write, copied back to host memory after the pass.
#[derive(Clone, Copy)]
pub struct Writeback {
    pub off: u32,
    pub size: u32,
    pub host: *mut u8,
    pub pf: pipe_format, // not NONE: the window holds RGBA32 per texel, the host this format
}

impl Writeback {
    pub fn host_size(&self) -> usize {
        if self.pf == PIPE_FORMAT_NONE { self.size as usize } else { self.size as usize / 16 * crate::fmt::blocksize(self.pf) as usize }
    }
}

/// Bindings to the flat slots borgc addresses.
#[derive(Clone, Copy)]
pub struct DescMap {
    pub n: u8,
    pub num: [u32; BORGVK_MAX_BINDINGS as usize],
    pub slot: [u8; BORGVK_MAX_BINDINGS as usize],
    pub count: [u8; BORGVK_MAX_BINDINGS as usize],
    pub total: u8,
    pub arrays: bool,
}

impl Default for DescMap {
    fn default() -> Self {
        const N: usize = BORGVK_MAX_BINDINGS as usize;
        Self { n: 0, num: [0; N], slot: [0; N], count: [0; N], total: 0, arrays: false }
    }
}

#[derive(Clone, Copy)]
pub struct VertexAttr {
    pub location: u32,
    pub binding: u32,
    pub offset: u32,
    pub format: VkFormat,
}

/// How a texel converts to the texture unit's format.
#[derive(Clone, Copy, Default)]
pub struct TexConv {
    pub bpp: u32,
    pub obpp: u32,
    pub cs: u32,
    pub sch: u32,
    pub alpha: u32,
    pub pack16: u32,
    pub zstride: u32,
    pub soff: u32,
    pub swap_rb: bool,
}
