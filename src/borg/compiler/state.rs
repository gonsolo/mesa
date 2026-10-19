use crate::config::Flags;
use crate::consts::{ConstPool, FragWindow};
use compiler::bindings::*;
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

impl Selector {
    pub fn opc(&mut self, mnem: &'static str, s: &[(u32, u8)]) -> u32 {
        let srcs = s.iter().map(|x| x.0).collect();
        let swz = s.iter().map(|x| x.1).collect();
        self.op(mnem, srcs, swz)
    }

    pub fn pair(&mut self, mnem: &'static str, x: (u32, u8), y: (u32, u8)) -> (u32, u8) {
        (self.opc(mnem, &[x, y]), 0)
    }

    pub fn fsub(&mut self, hi: (u32, u8), lo: (u32, u8)) -> u32 {
        let neg = self.opc("FNEG", &[lo]);
        self.opc("FADD", &[hi, (neg, 0)])
    }

    pub fn reg(&mut self, v: (u32, u8)) -> (u32, u8) {
        if matches!(self.ubo.get(&v.0), Some(crate::Ubo::Uniform(_))) {
            (self.opc("FMOV", &[(v.0, 0)]), 0)
        } else {
            v
        }
    }
}

/// Compile state of one shader: selector, constant pool, window and output bookkeeping.
pub struct ShaderState {
    pub sel: Selector,
    pub cp: ConstPool,
    pub fw: FragWindow,
    pub outs: DrawOutputs,
    pub consts: HashMap<u32, u32>,
    pub out_roots: Vec<u32>,
    pub is_vertex: bool,
    pub is_fragment: bool,
    pub flags: Flags,
    pub frag_alpha: bool,
    pub frag_z: Option<u32>,
    pub pos_out: [Option<u32>; 4],
    pub frag_out: [Option<u32>; 4],
    pub skipped: Vec<&'static str>,
    pub weights: Option<[u32; 3]>,
    pub per_pixel_fixed: std::collections::HashSet<u32>,
    pub spill_limit: Option<usize>,
    pub alloc_failed: bool,
    pub spill_slots: u32,
    pub pool: Vec<u32>,
    pub spill_lane: Option<u32>,
    pub no_spill: std::collections::HashSet<u32>,
    pub frag_coord_w: bool,
    pub res_binding: HashMap<u32, u32>,
    pub prod: HashMap<u32, (nir_op, Vec<(u32, u8)>)>,
    pub tex_dsts: std::collections::HashSet<u32>,
    pub fattr_dsts: std::collections::HashSet<u32>,
    pub zero_ctl: Option<u32>,
    pub tex_derefs: std::collections::HashSet<u32>,
    pub one_vreg: Option<u32>,
    pub push_const_reg: HashMap<u32, u32>,
    pub push_window: bool,
    pub out_select: i32,
    pub vertex_id_def: Option<u32>,
    pub descriptor_defs: std::collections::HashSet<u32>,
    pub const_addr_chain: Option<(i32, u32)>,
    pub dst_half: Option<([Option<u32>; 4], u32)>,
    pub is_compute: bool,
    pub cs: crate::compute::ComputeState,
}

impl ShaderState {
    pub fn new(regs: &'static [u8], is_vertex: bool, is_fragment: bool, flags: Flags) -> Self {
        ShaderState {
            sel: Selector::new(),
            cp: ConstPool::new(regs),
            fw: FragWindow::default(),
            outs: DrawOutputs::default(),
            consts: HashMap::new(),
            out_roots: Vec::new(),
            is_vertex,
            is_fragment,
            flags,
            frag_alpha: std::env::var("BORGC_FRAG_ALPHA").is_ok() || (is_fragment && flags.alpha_output()),
            frag_z: None,
            pos_out: [None; 4],
            frag_out: [None; 4],
            skipped: Default::default(),
            weights: Default::default(),
            per_pixel_fixed: Default::default(),
            spill_limit: None,
            alloc_failed: false,
            spill_slots: 0,
            pool: Vec::new(),
            spill_lane: None,
            no_spill: Default::default(),
            frag_coord_w: Default::default(),
            res_binding: Default::default(),
            prod: Default::default(),
            tex_dsts: Default::default(),
            fattr_dsts: Default::default(),
            zero_ctl: Default::default(),
            tex_derefs: Default::default(),
            one_vreg: Default::default(),
            push_const_reg: Default::default(),
            push_window: false,
            out_select: -1,
            vertex_id_def: Default::default(),
            descriptor_defs: Default::default(),
            const_addr_chain: Default::default(),
            dst_half: None,
            is_compute: false,
            cs: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_hands_out_vregs_from_a_million() {
        let mut s = Selector::new();
        assert_eq!((s.fresh(), s.fresh()), (1_000_000, 1_000_001));
    }

    #[test]
    fn op0_reads_component_zero_of_every_source() {
        let mut s = Selector::new();
        let d = s.op0("FADD", vec![3, 4]);
        assert_eq!((s.prog[0].dst, s.prog[0].swz.clone()), (d, vec![0, 0]));
    }

    #[test]
    fn fsub_negates_then_adds() {
        let mut s = Selector::new();
        let d = s.fsub((7, 2), (8, 1));
        let mnems: Vec<_> = s.prog.iter().map(|i| i.mnem).collect();
        assert_eq!(mnems, ["FNEG", "FADD"]);
        assert_eq!((s.prog[1].dst, s.prog[1].srcs[0], s.prog[1].swz[0]), (d, 7, 2));
    }

    #[test]
    fn reg_copies_only_uniform_operands() {
        let mut s = Selector::new();
        s.ubo.insert(5, crate::Ubo::Uniform(20));
        assert_eq!(s.reg((6, 1)), (6, 1));
        let (v, c) = s.reg((5, 0));
        assert!(v != 5 && c == 0 && s.prog[0].mnem == "FMOV");
    }
}
