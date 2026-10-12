use std::collections::HashMap;
#[derive(Default)]
pub struct DrawOutputs {
    pub stores: HashMap<u32, Vec<(u32, u8)>>,
    pub pos: [Option<(u32, u8)>; 4],
    pub psize: Option<(u32, u8)>,
    pub mrt: [Option<(u32, u8)>; 4],
    pub frag: [Option<(u32, u8)>; 4],
    pub roots: Vec<u32>,
}
pub struct Selector {
    pub vec_map: HashMap<u32, Vec<(u32, u8)>>,
    pub prog: Vec<crate::BorgInstr>,
    pub next_vreg: u32,
    pub ubo: HashMap<u32, crate::Ubo>,
}

impl Selector {
    pub fn new() -> Self {
        Selector { vec_map: HashMap::new(), prog: Vec::new(), next_vreg: 1_000_000, ubo: HashMap::new() }
    }
}

impl Selector {
    pub fn fresh(&mut self) -> u32 {
        let v = self.next_vreg;
        self.next_vreg += 1;
        v
    }
    pub fn op(&mut self, mnem: &'static str, srcs: Vec<u32>, swz: Vec<u8>) -> u32 {
        let dst = self.fresh();
        self.prog.push(crate::BorgInstr { mnem, dst, srcs, swz });
        dst
    }
    pub fn op0(&mut self, mnem: &'static str, srcs: Vec<u32>) -> u32 {
        let swz = vec![0; srcs.len()];
        self.op(mnem, srcs, swz)
    }
}
#[derive(Default)]
pub struct FragWindow {
    pub by_bits: HashMap<u32, u32>,
    pub count: u32,
    pub consts: Vec<(u32, u32)>,
}

impl FragWindow {
    pub fn slot(&mut self, bits: u32) -> u32 {
        if let Some(&u) = self.by_bits.get(&bits) {
            return u;
        }
        assert!(self.count < crate::FRAG_CONST_MAX, "borgc: fragment shader needs more than 11 window constants");
        let u = 20 + self.count;
        self.count += 1;
        self.consts.push((u, bits));
        self.by_bits.insert(bits, u);
        u
    }
}

impl FragWindow {
    pub fn win(&mut self, sel: &mut Selector, bits: u32) -> u32 {
        let u = self.slot(bits);
        let v = sel.fresh();
        sel.ubo.insert(v, crate::Ubo::Uniform(u as u8));
        v
    }
}
