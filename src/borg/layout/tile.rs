// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Render targets and tiled textures are stored in 4x4 tiles, row-major within a tile and in
//! rows of tiles across the surface.

/// Tile of pixel (x, y) in a surface `w` pixels wide, and the pixel's index within the tile.
pub fn index(w: u32, x: u32, y: u32) -> (usize, usize) {
    (((y >> 2) * (w >> 2) + (x >> 2)) as usize, ((x & 3) | (y & 3) << 2) as usize)
}

/// Byte offset of pixel (x, y) with tiles of `tile_bytes` and pixels of `px_bytes`.
pub fn offset(w: u32, x: u32, y: u32, tile_bytes: u32, px_bytes: u32) -> u32 {
    let (t, i) = index(w, x, y);
    t as u32 * tile_bytes + i as u32 * px_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_run_across_then_down() {
        assert_eq!(index(8, 0, 0), (0, 0));
        assert_eq!(index(8, 3, 0), (0, 3));
        assert_eq!(index(8, 0, 1), (0, 4));
        assert_eq!(index(8, 4, 0), (1, 0));
        assert_eq!(index(8, 5, 5), (3, 5));
        assert_eq!(offset(8, 5, 5, 64, 4), 3 * 64 + 5 * 4);
    }
}
