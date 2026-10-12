#[derive(Clone, Copy)]
pub struct Flags(pub u32);

impl Flags {
    // Vertex stage.
    pub fn fetching(self) -> bool { self.0 & 0x8000_0000 != 0 }
    pub fn point_quads(self) -> bool { self.0 & 0x4000_0000 != 0 }
    pub fn pntc_varyings(self) -> bool { self.0 & 0x2000_0000 != 0 }
    pub fn instanced(self, loc: u32) -> bool { self.0 >> loc & 1 != 0 }
    // Fragment stage.
    pub fn bytes_packed(self) -> bool { self.0 & 1 != 0 }
    pub fn r8_unorm(self) -> bool { self.0 & 2 != 0 }
    pub fn alpha_output(self) -> bool { self.0 & 4 != 0 }
    pub fn half(self) -> bool { self.0 & 8 != 0 }
    pub fn packed(self) -> bool { self.0 & 0x10 != 0 }
    pub fn multi_target(self) -> bool { self.0 & 0x100 != 0 }
    pub fn target_count(self) -> usize { ((self.0 >> 9) & 7) as usize + 1 }
    pub fn target_class(self, k: usize) -> u32 { (self.0 >> (12 + 3 * k)) & 7 }
    pub fn half_channels(self) -> usize {
        if self.0 & 0x20 != 0 { 1 } else if self.0 & 0x40 != 0 { 3 } else if self.0 & 0x80 != 0 { 4 } else { 2 }
    }
    pub fn pntc_base(self) -> u32 { self.0 >> 24 }
}
