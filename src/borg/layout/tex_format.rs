// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The texel formats the sampler decodes (docs/B2_texture_unit.md): their code in bits 19:14
//! of the texture descriptor's second word and their size, generated from
//! hardware/borg/src/TexFormat.scala.

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

include!("../../../../software/borg/borg_tex_format.rs");

#[cfg(test)]
mod tests {
    use super::*;

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
