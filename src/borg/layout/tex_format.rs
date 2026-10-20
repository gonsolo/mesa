// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The texel formats the sampler decodes (docs/B2_texture_unit.md): their code in bits 19:14
//! of the texture descriptor's second word and their size. The table of
//! hardware/borg/src/TexFormat.scala; a test holds the codes to the header generated from it.

macro_rules! formats {
    ($($name:ident = $code:literal, $bytes:literal;)*) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub enum TexFormat {
            $($name = $code),*
        }

        impl TexFormat {
            pub const ALL: &[TexFormat] = &[$(TexFormat::$name),*];

            /// Bytes of one texel.
            pub fn bytes(self) -> u32 {
                match self {
                    $(TexFormat::$name => $bytes),*
                }
            }

            pub fn name(self) -> &'static str {
                match self {
                    $(TexFormat::$name => stringify!($name)),*
                }
            }
        }
    };
}

formats! {
    R8_UNORM = 1, 1; R8_SNORM = 2, 1; R8_UINT = 3, 1; R8_SINT = 4, 1; R8_SRGB = 5, 1;
    R8G8_UNORM = 6, 2; R8G8_SNORM = 7, 2; R8G8_UINT = 8, 2; R8G8_SINT = 9, 2; R8G8_SRGB = 10, 2;
    R8G8B8A8_UNORM = 11, 4; R8G8B8A8_SNORM = 12, 4; R8G8B8A8_UINT = 13, 4; R8G8B8A8_SINT = 14, 4;
    R8G8B8A8_SRGB = 15, 4;
    B8G8R8A8_UNORM = 16, 4; B8G8R8A8_SRGB = 17, 4;
    R16_UNORM = 18, 2; R16_SNORM = 19, 2; R16_UINT = 20, 2; R16_SINT = 21, 2; R16_SFLOAT = 22, 2;
    R16G16_UNORM = 23, 4; R16G16_SNORM = 24, 4; R16G16_UINT = 25, 4; R16G16_SINT = 26, 4;
    R16G16_SFLOAT = 27, 4;
    R16G16B16A16_UNORM = 28, 8; R16G16B16A16_SNORM = 29, 8; R16G16B16A16_UINT = 30, 8;
    R16G16B16A16_SINT = 31, 8; R16G16B16A16_SFLOAT = 32, 8;
    R32_UINT = 33, 4; R32_SINT = 34, 4; R32_SFLOAT = 35, 4;
    R32G32_UINT = 36, 8; R32G32_SINT = 37, 8; R32G32_SFLOAT = 38, 8;
    R32G32B32A32_UINT = 39, 16; R32G32B32A32_SINT = 40, 16; R32G32B32A32_SFLOAT = 41, 16;
    A2B10G10R10_UNORM_PACK32 = 42, 4; A2B10G10R10_UINT_PACK32 = 43, 4;
    R5G6B5_UNORM_PACK16 = 44, 2; A1R5G5B5_UNORM_PACK16 = 45, 2; B4G4R4A4_UNORM_PACK16 = 46, 2;
    B10G11R11_UFLOAT_PACK32 = 47, 4; E5B9G9R9_UFLOAT_PACK32 = 48, 4;
    D16_UNORM = 49, 2; X8_D24_UNORM_PACK32 = 50, 4; D32_SFLOAT = 51, 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_the_hardware_header() {
        let h = include_str!("../../../../software/borg/borg_tex_format.h");
        let defs: Vec<(&str, u8)> = h
            .lines()
            .filter_map(|l| l.strip_prefix("#define BORG_TEX_FORMAT_"))
            .map(|l| {
                let (n, c) = l.split_once(' ').unwrap();
                (n, c.trim().parse().unwrap())
            })
            .collect();
        let ours: Vec<(&str, u8)> = TexFormat::ALL.iter().map(|&f| (f.name(), f as u8)).collect();
        assert_eq!(ours, defs);
    }

    #[test]
    fn sizes_follow_the_names() {
        for &f in TexFormat::ALL {
            let n = f.name();
            let bits: u32 = if n.ends_with("PACK16") {
                16
            } else if n.ends_with("PACK32") || n.starts_with("X8_D24") {
                32
            } else {
                n.split('_').next().unwrap().split(|c: char| c.is_ascii_alphabetic()).filter_map(|d| d.parse::<u32>().ok()).sum()
            };
            assert_eq!(f.bytes() * 8, bits, "{n}");
        }
    }
}
