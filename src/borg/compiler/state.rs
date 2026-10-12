use std::collections::HashMap;

/// Where the draw front end finds each shader output: (scalar def, component) per slot.
#[derive(Default)]
pub struct DrawOutputs {
    pub stores: HashMap<u32, Vec<(u32, u8)>>,
    pub pos: [Option<(u32, u8)>; 4],
    pub psize: Option<(u32, u8)>,
    /// Fragment outputs of locations 0..3 (multiple attachments).
    pub mrt: [Option<(u32, u8)>; 4],
    pub frag: [Option<(u32, u8)>; 4],
    pub roots: Vec<u32>,
}

/// What the selection walk builds: the instruction list and the maps that resolve operands.
pub struct Selector {
    /// mov/vecN results: per component, the scalar def and component it really reads.
    pub vec_map: HashMap<u32, Vec<(u32, u8)>>,
    pub prog: Vec<crate::BorgInstr>,
    /// Synthetic results use vregs above any NIR ssa index.
    pub next_vreg: u32,
    /// Operand classification (uniform, fixed register, ...).
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

    /// Emit one instruction into a fresh vreg.
    pub fn op(&mut self, mnem: &'static str, srcs: Vec<u32>, swz: Vec<u8>) -> u32 {
        let dst = self.fresh();
        self.prog.push(crate::BorgInstr { mnem, dst, srcs, swz });
        dst
    }

    /// `op` reading component 0 of every source.
    pub fn op0(&mut self, mnem: &'static str, srcs: Vec<u32>) -> u32 {
        let swz = vec![0; srcs.len()];
        self.op(mnem, srcs, swz)
    }
}

/// Fragment window constants: draw-mode uniforms u20.. holding values by bit pattern.
#[derive(Default)]
pub struct FragWindow {
    pub by_bits: HashMap<u32, u32>,
    pub count: u32,
    /// (u-index, bits)
    pub consts: Vec<(u32, u32)>,
}

impl FragWindow {
    /// The window word holding `bits`, allocating the next one if it is new.
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
    /// A fresh vreg reading the window word that holds `bits`.
    pub fn win(&mut self, sel: &mut Selector, bits: u32) -> u32 {
        let u = self.slot(bits);
        let v = sel.fresh();
        sel.ubo.insert(v, crate::Ubo::Uniform(u as u8));
        v
    }
}
