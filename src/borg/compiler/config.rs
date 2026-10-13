use borgc_flags::*;

#[derive(Clone, Copy)]
pub struct Flags(pub u32);

impl Flags {
    // Vertex stage.
    pub fn fetching(self) -> bool { self.0 & BORGC_VS_FETCH != 0 }
    pub fn point_quads(self) -> bool { self.0 & BORGC_VS_POINT_QUADS != 0 }
    pub fn pntc_varyings(self) -> bool { self.0 & BORGC_VS_PNTC_VARYINGS != 0 }
    pub fn instanced(self, loc: u32) -> bool { self.0 >> loc & 1 != 0 }
    // Fragment stage.
    pub fn bytes_packed(self) -> bool { self.0 & BORGC_FS_BYTES_PACKED != 0 }
    pub fn r8_unorm(self) -> bool { self.0 & BORGC_FS_R8_UNORM != 0 }
    pub fn alpha_output(self) -> bool { self.0 & BORGC_FS_ALPHA_OUTPUT != 0 }
    pub fn half(self) -> bool { self.0 & BORGC_FS_HALF != 0 }
    pub fn packed(self) -> bool { self.0 & BORGC_FS_PACKED != 0 }
    pub fn multi_target(self) -> bool { self.0 & BORGC_FS_MULTI_TARGET != 0 }
    pub fn target_count(self) -> usize { ((self.0 >> BORGC_FS_TARGETS_SHIFT) & 7) as usize + 1 }
    pub fn target_class(self, k: usize) -> u32 { (self.0 >> (BORGC_FS_TARGET_CLASS_SHIFT + 3 * k as u32)) & 7 }
    pub fn half_channels(self) -> usize {
        if self.0 & BORGC_FS_HALF1 != 0 { 1 } else if self.0 & BORGC_FS_HALF3 != 0 { 3 } else if self.0 & BORGC_FS_HALF4 != 0 { 4 } else { 2 }
    }
    pub fn pntc_base(self) -> u32 { self.0 >> BORGC_FS_PNTC_BASE_SHIFT }
}

#[cfg(test)]
mod tests {
    use super::Flags;

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
}
