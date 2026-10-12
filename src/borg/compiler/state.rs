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
