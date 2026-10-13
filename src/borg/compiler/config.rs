#[derive(Clone, Copy)]
pub struct Flags(pub u32);

pub mod bit {
    pub const VS_FETCH: u32 = 0x8000_0000;
    pub const VS_POINT_QUADS: u32 = 0x4000_0000;
    pub const VS_PNTC_VARYINGS: u32 = 0x2000_0000;
    pub const FS_BYTES_PACKED: u32 = 0x1;
    pub const FS_R8_UNORM: u32 = 0x2;
    pub const FS_ALPHA_OUTPUT: u32 = 0x4;
    pub const FS_HALF: u32 = 0x8;
    pub const FS_PACKED: u32 = 0x10;
    pub const FS_HALF1: u32 = 0x20;
    pub const FS_HALF3: u32 = 0x40;
    pub const FS_HALF4: u32 = 0x80;
    pub const FS_MULTI_TARGET: u32 = 0x100;
    pub const FS_TARGETS_SHIFT: u32 = 9;
    pub const FS_TARGET_CLASS_SHIFT: u32 = 12;
    pub const FS_PACKED_KIND_SHIFT: u32 = 16;
    pub const FS_PACKED_SIZE_SHIFT: u32 = 19;
    pub const FS_PACKED_N_SHIFT: u32 = 21;
    pub const FS_PACKED_SWAP_SHIFT: u32 = 23;
    pub const FS_PNTC_BASE_SHIFT: u32 = 24;
}

impl Flags {
    // Vertex stage.
    pub fn fetching(self) -> bool { self.0 & bit::VS_FETCH != 0 }
    pub fn point_quads(self) -> bool { self.0 & bit::VS_POINT_QUADS != 0 }
    pub fn pntc_varyings(self) -> bool { self.0 & bit::VS_PNTC_VARYINGS != 0 }
    pub fn instanced(self, loc: u32) -> bool { self.0 >> loc & 1 != 0 }
    // Fragment stage.
    pub fn bytes_packed(self) -> bool { self.0 & bit::FS_BYTES_PACKED != 0 }
    pub fn r8_unorm(self) -> bool { self.0 & bit::FS_R8_UNORM != 0 }
    pub fn alpha_output(self) -> bool { self.0 & bit::FS_ALPHA_OUTPUT != 0 }
    pub fn half(self) -> bool { self.0 & bit::FS_HALF != 0 }
    pub fn packed(self) -> bool { self.0 & bit::FS_PACKED != 0 }
    pub fn multi_target(self) -> bool { self.0 & bit::FS_MULTI_TARGET != 0 }
    pub fn target_count(self) -> usize { ((self.0 >> bit::FS_TARGETS_SHIFT) & 7) as usize + 1 }
    pub fn target_class(self, k: usize) -> u32 { (self.0 >> (bit::FS_TARGET_CLASS_SHIFT + 3 * k as u32)) & 7 }
    pub fn half_channels(self) -> usize {
        if self.0 & bit::FS_HALF1 != 0 { 1 } else if self.0 & bit::FS_HALF3 != 0 { 3 } else if self.0 & bit::FS_HALF4 != 0 { 4 } else { 2 }
    }
    pub fn pntc_base(self) -> u32 { self.0 >> bit::FS_PNTC_BASE_SHIFT }
}

#[cfg(test)]
mod tests {
    use super::{bit, Flags};

    #[test]
    fn vertex_bits() {
        let f = Flags(0xC000_0005);
        assert!(f.fetching() && f.point_quads() && !f.pntc_varyings());
        assert!(f.instanced(0) && !f.instanced(1) && f.instanced(2));
        assert!(Flags(0x2000_0000).pntc_varyings());
    }

    #[test]
    fn fragment_bits() {
        assert!(Flags(1).bytes_packed() && Flags(2).r8_unorm() && Flags(4).alpha_output());
        assert!(Flags(8).half() && Flags(0x10).packed() && Flags(0x100).multi_target());
        assert!(!Flags(0).half() && !Flags(0).multi_target());
    }

    #[test]
    fn half_channels() {
        let n = |w| Flags(w).half_channels();
        assert_eq!((n(0), n(0x20), n(0x40), n(0x80)), (2, 1, 3, 4));
    }

    #[test]
    fn targets() {
        let f = Flags(0x100 | 2 << 9 | 1 << 12 | 3 << 15 | 2 << 18);
        assert_eq!(f.target_count(), 3);
        assert_eq!((f.target_class(0), f.target_class(1), f.target_class(2)), (1, 3, 2));
    }

    #[test]
    fn pntc_base_is_the_top_byte() {
        assert_eq!(Flags(7 << 24 | 0x2000_0000).pntc_base(), 0x27);
    }

    #[test]
    fn header_matches() {
        let h = include_str!("borgc_flags.h");
        let value = |name: &str| -> u32 {
            let line = h.lines().find(|l| l.split_whitespace().nth(1) == Some(name)).unwrap();
            let v = line.split_whitespace().nth(2).unwrap().trim_end_matches('u');
            match v.strip_prefix("0x") { Some(x) => u32::from_str_radix(x, 16).unwrap(), None => v.parse().unwrap() }
        };
        for (name, v) in [
            ("BORGC_VS_FETCH", bit::VS_FETCH), ("BORGC_VS_POINT_QUADS", bit::VS_POINT_QUADS),
            ("BORGC_VS_PNTC_VARYINGS", bit::VS_PNTC_VARYINGS), ("BORGC_FS_BYTES_PACKED", bit::FS_BYTES_PACKED),
            ("BORGC_FS_R8_UNORM", bit::FS_R8_UNORM), ("BORGC_FS_ALPHA_OUTPUT", bit::FS_ALPHA_OUTPUT),
            ("BORGC_FS_HALF", bit::FS_HALF), ("BORGC_FS_PACKED", bit::FS_PACKED), ("BORGC_FS_HALF1", bit::FS_HALF1),
            ("BORGC_FS_HALF3", bit::FS_HALF3), ("BORGC_FS_HALF4", bit::FS_HALF4),
            ("BORGC_FS_MULTI_TARGET", bit::FS_MULTI_TARGET), ("BORGC_FS_TARGETS_SHIFT", bit::FS_TARGETS_SHIFT),
            ("BORGC_FS_TARGET_CLASS_SHIFT", bit::FS_TARGET_CLASS_SHIFT),
            ("BORGC_FS_PACKED_KIND_SHIFT", bit::FS_PACKED_KIND_SHIFT),
            ("BORGC_FS_PACKED_SIZE_SHIFT", bit::FS_PACKED_SIZE_SHIFT),
            ("BORGC_FS_PACKED_N_SHIFT", bit::FS_PACKED_N_SHIFT),
            ("BORGC_FS_PACKED_SWAP_SHIFT", bit::FS_PACKED_SWAP_SHIFT),
            ("BORGC_FS_PNTC_BASE_SHIFT", bit::FS_PNTC_BASE_SHIFT),
        ] {
            assert_eq!(value(name), v, "{name}");
        }
    }
}
