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
        let (size, stride) = mip_level_size(self.extent, level, b);
        // A view's layer of a 3D image is a slice of that level.
        let step = if self.extent[2] > 1 { size / (self.extent[2] >> level).max(1) as u64 } else { self.layer_size(b) };
        (layer as u64 * step + before, stride)
    }

    /// Bytes the image needs in memory, a multiple of 256.
    pub fn size(&self, b: Blocks) -> u64 {
        let levels = Shape { levels: self.levels.max(1), ..*self };
        (levels.layer_size(b) * self.layers.max(1) as u64).next_multiple_of(256)
    }
}

/// Optimal images on the GPU (B2, "Layouts"): per plane every array layer, per layer every mip
/// level, per level 4x4 tiles of texels, each tile's samples after each other as the flusher
/// stores them. Depth and stencil are two planes; `planes[1].bs == 0` is one plane. `plane1`
/// places the second plane when the image is a level or layer of a larger one (0: after plane 0).
#[derive(Clone, Copy, Debug)]
pub struct Tiled {
    pub shape: Shape,
    pub planes: [Blocks; 2],
    pub plane1: u64,
}

impl Tiled {
    fn dims(&self, level: u32) -> [u32; 3] {
        self.shape.extent.map(|e| e.checked_shr(level).unwrap_or(0).max(1))
    }

    /// Bytes of one tile of plane `p`: 16 texels of every sample.
    pub fn tile_bytes(&self, p: usize) -> u64 {
        16 * self.planes[p].bs as u64 * self.shape.samples.max(1) as u64
    }

    fn level_bytes(&self, p: usize, level: u32) -> u64 {
        let [w, h, d] = self.dims(level);
        w.div_ceil(4) as u64 * h.div_ceil(4) as u64 * d as u64 * self.tile_bytes(p)
    }

    /// Offset of `level` within a layer of plane `p`.
    pub fn level_offset(&self, p: usize, level: u32) -> u64 {
        (0..level).map(|l| self.level_bytes(p, l)).sum()
    }

    /// Bytes of one array layer of plane `p`: all its levels.
    pub fn layer_size(&self, p: usize) -> u64 {
        self.level_offset(p, self.shape.levels.max(1))
    }

    fn plane_offset(&self, p: usize) -> u64 {
        match (p, self.plane1) {
            (0, _) => 0,
            (_, 0) => (self.layer_size(0) * self.shape.layers.max(1) as u64).next_multiple_of(256),
            (_, o) => o,
        }
    }

    /// Where `level` of `layer` in plane `p` starts.
    pub fn level_base(&self, p: usize, level: u32, layer: u32) -> u64 {
        self.plane_offset(p) + layer as u64 * self.layer_size(p) + self.level_offset(p, level)
    }

    /// Offset of texel `xyz` of sample `s` from its level's base; a 16-byte texel's second half
    /// is 128 bytes further.
    pub fn in_level(&self, p: usize, level: u32, [x, y, z]: [u32; 3], s: u32) -> u64 {
        let [w, h, _] = self.dims(level);
        let tpr = w.div_ceil(4) as u64;
        let tile = (z as u64 * h.div_ceil(4) as u64 + (y >> 2) as u64) * tpr + (x >> 2) as u64;
        let i = ((x & 3) | (y & 3) << 2) as u64;
        let bs = self.planes[p].bs as u64;
        let within = if bs == 16 { i * 8 } else { i * bs };
        (tile * self.shape.samples.max(1) as u64 + s as u64) * 16 * bs + within
    }

    pub fn texel(&self, p: usize, level: u32, layer: u32, xyz: [u32; 3], s: u32) -> u64 {
        self.level_base(p, level, layer) + self.in_level(p, level, xyz, s)
    }

    /// Bytes the image needs in memory, a multiple of 256.
    pub fn size(&self) -> u64 {
        let layers = self.shape.layers.max(1) as u64;
        let two = self.planes[1].bs != 0;
        (if two { self.plane_offset(1) + self.layer_size(1) * layers } else { self.layer_size(0) * layers }).next_multiple_of(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA8: Blocks = Blocks { bs: 4, bw: 1, bh: 1, bd: 1 };

    #[test]
    fn slice_of_a_3d_level() {
        let s = Shape { extent: [4, 4, 4], levels: 2, layers: 1, samples: 1 };
        assert_eq!(s.level_offset(0, 3, RGBA8).0, 3 * 64);
        assert_eq!(s.level_offset(1, 1, RGBA8).0, 256 + 16);
    }

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

    #[test]
    fn tiled_texels() {
        let t = Tiled { shape: Shape { extent: [6, 5, 1], levels: 2, layers: 2, samples: 1 }, planes: [RGBA8, Blocks { bs: 0, ..RGBA8 }], plane1: 0 };
        assert_eq!(t.texel(0, 0, 0, [1, 2, 0], 0), (2 * 4 + 1) * 4);
        assert_eq!(t.texel(0, 0, 0, [5, 4, 0], 0), (3 * 16 + 1) * 4);
        assert_eq!(t.layer_size(0), 4 * 64 + 64);
        assert_eq!(t.texel(0, 1, 1, [2, 1, 0], 0), 320 + 256 + (4 + 2) * 4);
        assert_eq!(t.size(), 768);
    }

    #[test]
    fn tiled_samples_planes_and_halves() {
        let ms = Tiled { shape: Shape { extent: [8, 4, 1], levels: 1, layers: 1, samples: 4 }, planes: [RGBA8, Blocks { bs: 1, ..RGBA8 }], plane1: 0 };
        assert_eq!(ms.texel(0, 0, 0, [4, 0, 0], 2), (4 + 2) * 64);
        assert_eq!(ms.texel(1, 0, 0, [0, 0, 0], 0), 512);
        assert_eq!(ms.size(), 512 + 128 + 128);
        let wide = Tiled { shape: Shape { extent: [4, 4, 2], levels: 1, layers: 1, samples: 1 }, planes: [Blocks { bs: 16, ..RGBA8 }; 2], plane1: 0 };
        assert_eq!(wide.texel(0, 0, 0, [1, 0, 1], 0), 256 + 8);
    }
}
