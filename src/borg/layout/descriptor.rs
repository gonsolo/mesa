// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The texture unit's descriptors (docs/B2_texture_unit.md): words in memory the sampler reads.

use crate::tex_format::TexFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexType {
    D1 = 0,
    D2 = 1,
    D3 = 2,
    Cube = 3,
}

/// Words 1-3 of a texture descriptor. Word 0, the base address, and words 4-15, the level
/// offsets, are written where the texture is placed.
#[derive(Clone, Copy, Debug)]
pub struct TexDesc {
    pub width: u32,
    pub height: u32,
    pub ty: TexType,
    pub linear: bool,
    /// Depth for 3D, else array layers (6 per cube).
    pub depth_or_layers: u32,
    pub levels: u32,
    pub format: TexFormat,
    /// `VkComponentSwizzle` of R, G, B and A.
    pub swizzle: [u32; 4],
    /// Tiled: bytes per array layer; linear: bytes per row.
    pub pitch: u32,
}

impl TexDesc {
    pub fn pack(&self) -> [u32; 3] {
        let sw = |i: usize| (self.swizzle[i] & 7) << (20 + 3 * i);
        [
            (self.width - 1) | (self.height - 1) << 16 | (self.ty as u32) << 28 | (self.linear as u32) << 30,
            (self.depth_or_layers - 1) | (self.levels - 1) << 10 | (self.format as u32) << 14 | sw(0) | sw(1) | sw(2) | sw(3),
            self.pitch,
        ]
    }
}

/// A sampler descriptor; modes and ops are the Vulkan enum values.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplerDesc {
    pub mag_linear: bool,
    pub min_linear: bool,
    pub mip_linear: bool,
    pub address: [u32; 3],
    pub border: u32,
    pub compare: bool,
    pub compare_op: u32,
    pub unnormalized: bool,
    pub lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
}

impl SamplerDesc {
    pub fn pack(&self) -> [u32; 4] {
        let a = |i: usize| (self.address[i] & 7) << (3 + 3 * i);
        [
            self.mag_linear as u32
                | (self.min_linear as u32) << 1
                | (self.mip_linear as u32) << 2
                | a(0)
                | a(1)
                | a(2)
                | (self.border & 7) << 12
                | (self.compare as u32) << 15
                | (self.compare_op & 7) << 16
                | (self.unnormalized as u32) << 19,
            self.lod_bias.to_bits(),
            self.min_lod.to_bits(),
            self.max_lod.to_bits(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_words_follow_the_table() {
        let d = TexDesc {
            width: 256,
            height: 64,
            ty: TexType::Cube,
            linear: false,
            depth_or_layers: 12,
            levels: 9,
            format: TexFormat::R8G8B8A8_SRGB,
            swizzle: [6, 5, 4, 3],
            pitch: 0x1234,
        };
        let [w1, w2, w3] = d.pack();
        assert_eq!((w1 & 0xffff, w1 >> 16 & 0xfff, w1 >> 28 & 3, w1 >> 30 & 1), (255, 63, 3, 0));
        assert_eq!((w2 & 0x3ff, w2 >> 10 & 0xf, w2 >> 14 & 0x3f), (11, 8, 15));
        assert_eq!((w2 >> 20 & 7, w2 >> 23 & 7, w2 >> 26 & 7, w2 >> 29), (6, 5, 4, 3));
        assert_eq!(w3, 0x1234);
    }

    #[test]
    fn texel_buffers_use_the_whole_width_field() {
        let d = TexDesc {
            width: 65536,
            height: 1,
            ty: TexType::D1,
            linear: true,
            depth_or_layers: 1,
            levels: 1,
            format: TexFormat::R32_UINT,
            swizzle: [0; 4],
            pitch: 0,
        };
        assert_eq!(d.pack()[0], 0xffff | 1 << 30);
    }

    #[test]
    fn sampler_words_follow_the_table() {
        let s = SamplerDesc {
            min_linear: true,
            address: [4, 2, 1],
            border: 5,
            compare: true,
            compare_op: 7,
            max_lod: 1000.0,
            ..Default::default()
        };
        let [w0, w1, w2, w3] = s.pack();
        assert_eq!(w0 & 7, 2);
        assert_eq!((w0 >> 3 & 7, w0 >> 6 & 7, w0 >> 9 & 7, w0 >> 12 & 7), (4, 2, 1, 5));
        assert_eq!((w0 >> 15 & 1, w0 >> 16 & 7, w0 >> 19), (1, 7, 0));
        assert_eq!((w1, w2, f32::from_bits(w3)), (0, 0, 1000.0));
    }
}
