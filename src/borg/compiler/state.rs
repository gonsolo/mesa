use crate::config::Flags;
use crate::opcodes::Op;
use crate::Mnem;
use crate::consts::{ConstPool, Window};
use compiler::bindings::*;
use std::collections::{HashMap, HashSet};
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
    pub fn op(&mut self, mnem: impl Into<Mnem>, srcs: Vec<u32>, swz: Vec<u8>) -> u32 {
        let dst = self.fresh();
        self.prog.push(crate::BorgInstr { mnem: mnem.into(), dst, srcs, swz });
        dst
    }
    pub fn op0(&mut self, mnem: impl Into<Mnem>, srcs: Vec<u32>) -> u32 {
        let swz = vec![0; srcs.len()];
        self.op(mnem, srcs, swz)
    }
}

impl Selector {
    pub fn opc(&mut self, mnem: impl Into<Mnem>, s: &[(u32, u8)]) -> u32 {
        let srcs = s.iter().map(|x| x.0).collect();
        let swz = s.iter().map(|x| x.1).collect();
        self.op(mnem, srcs, swz)
    }

    pub fn pair(&mut self, mnem: impl Into<Mnem>, x: (u32, u8), y: (u32, u8)) -> (u32, u8) {
        (self.opc(mnem, &[x, y]), 0)
    }

    pub fn fsub(&mut self, hi: (u32, u8), lo: (u32, u8)) -> u32 {
        let neg = self.opc(Op::Fneg, &[lo]);
        self.opc(Op::Fadd, &[hi, (neg, 0)])
    }

    pub fn reg(&mut self, v: (u32, u8)) -> (u32, u8) {
        if matches!(self.ubo.get(&v.0), Some(crate::Ubo::Uniform(_))) {
            (self.opc(Op::Fmov, &[(v.0, 0)]), 0)
        } else {
            v
        }
    }
}

/// The shader stage; Other is compiled for its diagnostics only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
    Other,
}

impl Stage {
    pub fn of(stage: mesa_shader_stage) -> Self {
        match stage {
            MESA_SHADER_VERTEX => Stage::Vertex,
            MESA_SHADER_FRAGMENT => Stage::Fragment,
            MESA_SHADER_COMPUTE => Stage::Compute,
            _ => Stage::Other,
        }
    }
}

/// Compile state of one shader: selector, constant pool, window and output bookkeeping.
pub struct ShaderState {
    pub sel: Selector,
    pub cp: ConstPool,
    pub win: Window,
    pub outs: DrawOutputs,
    pub consts: HashMap<u32, u32>,
    pub out_roots: Vec<u32>,
    pub stage: Stage,
    pub flags: Flags,
    pub skipped: Vec<&'static str>,
    pub pool: Vec<u32>,
    pub res_binding: HashMap<u32, u32>,
    pub prod: HashMap<u32, (nir_op, Vec<(u32, u8)>)>,
    pub tex_dsts: HashSet<u32>,
    pub fattr_dsts: HashSet<u32>,
    pub zero_ctl: Option<u32>,
    pub tex_derefs: HashSet<u32>,
    pub one_vreg: Option<u32>,
    pub push_const_reg: HashMap<u32, u32>,
    pub const_addr_chain: Option<(i32, u32)>,
    /// Why the shader is refused: the first reason, reported when the phase ends.
    pub refused: Option<String>,
    pub frag: Frag,
    pub vert: Vert,
    pub spill: Spill,
    pub cs: crate::compute::ComputeState,
}

/// What only a fragment shader has.
pub struct Frag {
    pub alpha: bool,
    pub out: [Option<u32>; 4],
    pub per_pixel_fixed: HashSet<u32>,
    pub coord_w: bool,
    pub push_window: bool,
    /// The one colour output kept for a target drawn on its own, -1 for all.
    pub out_select: i32,
    pub dst_half: Option<([Option<u32>; 4], u32)>,
}

/// What only a vertex shader has.
#[derive(Default)]
pub struct Vert {
    pub pos_out: [Option<u32>; 4],
    pub vertex_id_def: Option<u32>,
    pub descriptor_defs: HashSet<u32>,
}

/// Spilling: the live-value limit of this attempt, its outcome and the scratch slots used.
#[derive(Default)]
pub struct Spill {
    pub limit: Option<usize>,
    pub alloc_failed: bool,
    pub slots: u32,
    pub lane: Option<u32>,
    /// Values never spilled: the spill code's own.
    pub exempt: HashSet<u32>,
}

impl ShaderState {
    pub fn refuse(&mut self, why: impl Into<String>) {
        self.refused.get_or_insert_with(|| why.into());
    }

    /// Ends a phase: the reason the shader is refused, if it is.
    pub fn refusal(&mut self) -> Result<(), String> {
        self.refused.take().map_or(Ok(()), Err)
    }
}

impl ShaderState {
    pub fn new(regs: &'static [u8], stage: Stage, flags: Flags) -> Self {
        ShaderState {
            sel: Selector::new(),
            cp: ConstPool::new(regs),
            win: if stage == Stage::Vertex { Window::vertex() } else { Window::fragment() },
            outs: DrawOutputs::default(),
            consts: HashMap::new(),
            out_roots: Vec::new(),
            stage,
            flags,
            skipped: Default::default(),
            pool: Vec::new(),
            res_binding: Default::default(),
            prod: Default::default(),
            tex_dsts: Default::default(),
            fattr_dsts: Default::default(),
            zero_ctl: Default::default(),
            tex_derefs: Default::default(),
            one_vreg: Default::default(),
            push_const_reg: Default::default(),
            const_addr_chain: Default::default(),
            refused: None,
            frag: Frag {
                alpha: std::env::var("BORGC_FRAG_ALPHA").is_ok() || (stage == Stage::Fragment && flags.alpha_output()),
                out: [None; 4],
                per_pixel_fixed: Default::default(),
                coord_w: false,
                push_window: false,
                out_select: -1,
                dst_half: None,
            },
            vert: Default::default(),
            spill: Default::default(),
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
        let d = s.op0(Op::Fadd, vec![3, 4]);
        assert_eq!((s.prog[0].dst, s.prog[0].swz.clone()), (d, vec![0, 0]));
    }

    #[test]
    fn fsub_negates_then_adds() {
        let mut s = Selector::new();
        let d = s.fsub((7, 2), (8, 1));
        let mnems: Vec<_> = s.prog.iter().map(|i| i.mnem).collect();
        assert_eq!(mnems, [Op::Fneg, Op::Fadd]);
        assert_eq!((s.prog[1].dst, s.prog[1].srcs[0], s.prog[1].swz[0]), (d, 7, 2));
    }

    #[test]
    fn reg_copies_only_uniform_operands() {
        let mut s = Selector::new();
        s.ubo.insert(5, crate::Ubo::Uniform(20));
        assert_eq!(s.reg((6, 1)), (6, 1));
        let (v, c) = s.reg((5, 0));
        assert!(v != 5 && c == 0 && s.prog[0].mnem == Op::Fmov);
    }
}
