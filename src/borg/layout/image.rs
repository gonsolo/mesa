// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Linear images: per array layer every sample plane, per plane every mip level, tightly packed
//! in blocks of the format.

/// Block size and dimensions of a format: bytes per block, texels per block in x, y and z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocks {
    pub bs: u32,
    pub bw: u32,
    pub bh: u32,
    pub bd: u32,
}

/// What an image's layout depends on besides the format.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub extent: [u32; 3],
    pub levels: u32,
    pub layers: u32,
    pub samples: u32,
}

/// Bytes of mip `level` and its width in blocks.
pub fn mip_level_size(extent: [u32; 3], level: u32, b: Blocks) -> (u64, u32) {
    let dim = |e: u32, block: u32| e.checked_shr(level).unwrap_or(0).max(1).div_ceil(block);
    let w = dim(extent[0], b.bw);
    let (h, d) = (dim(extent[1], b.bh), dim(extent[2], b.bd));
    (w as u64 * h as u64 * d as u64 * b.bs as u64, w)
}

impl Shape {
    /// Bytes of all mip levels of one sample plane.
    pub fn sample_plane_size(&self, b: Blocks) -> u64 {
        (0..self.levels).map(|l| mip_level_size(self.extent, l, b).0).sum()
    }

    /// Bytes of one array layer: every sample plane.
    pub fn layer_size(&self, b: Blocks) -> u64 {
        self.sample_plane_size(b) * self.samples.max(1) as u64
    }

    /// Offset of `level` in `layer`, and that level's row stride in blocks.
    pub fn level_offset(&self, level: u32, layer: u32, b: Blocks) -> (u64, u32) {
        let before: u64 = (0..level).map(|l| mip_level_size(self.extent, l, b).0).sum();
        let stride = mip_level_size(self.extent, level, b).1;
        (layer as u64 * self.layer_size(b) + before, stride)
    }

    /// Bytes the image needs in memory, a multiple of 256.
    pub fn size(&self, b: Blocks) -> u64 {
        let levels = Shape { levels: self.levels.max(1), ..*self };
        (levels.layer_size(b) * self.layers.max(1) as u64).next_multiple_of(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA8: Blocks = Blocks { bs: 4, bw: 1, bh: 1, bd: 1 };

    #[test]
    fn mip_sizes() {
        assert_eq!(mip_level_size([16, 16, 1], 0, RGBA8), (1024, 16));
        assert_eq!(mip_level_size([16, 16, 1], 4, RGBA8).0, 4);
        assert_eq!(mip_level_size([16, 16, 1], 9, RGBA8).0, 4);
    }

    #[test]
    fn block_compressed_size_rounds_up() {
        assert_eq!(mip_level_size([10, 10, 1], 0, Blocks { bs: 8, bw: 4, bh: 4, bd: 1 }), (72, 3));
    }

    #[test]
    fn layers_hold_every_sample_plane() {
        let s = Shape { extent: [4, 4, 1], levels: 3, layers: 2, samples: 4 };
        assert_eq!(s.sample_plane_size(RGBA8), 64 + 16 + 4);
        assert_eq!(s.layer_size(RGBA8), 4 * 84);
        assert_eq!(s.level_offset(2, 1, RGBA8), (4 * 84 + 80, 1));
        assert_eq!(s.size(RGBA8), (2 * 4 * 84u64).next_multiple_of(256));
        assert_eq!(Shape { levels: 0, layers: 0, samples: 0, ..s }.size(RGBA8), 256);
    }
}
