use crate::config::Flags;
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

pub const DRAW_VS_CONST_U0: u8 = 25;
pub const DRAW_VS_CONST_WORDS: usize = 6;

pub struct ConstPool {
    pub regs: &'static [u8],
    pub count: usize,
    pub int_reg: HashMap<i32, u8>,
    pub uniforms: Vec<(u8, u32)>,
    pub vs_consts: Vec<(u8, u32)>,
    pub vs_by_bits: HashMap<u32, u32>,
    pub vs_zero: Option<u32>,
    pub vs_window: bool,
}

impl ConstPool {
    pub fn new(regs: &'static [u8], vs_window: bool) -> Self {
        ConstPool {
            regs,
            count: 0,
            int_reg: HashMap::new(),
            uniforms: Vec::new(),
            vs_consts: Vec::new(),
            vs_by_bits: HashMap::new(),
            vs_zero: None,
            vs_window,
        }
    }

    pub fn alloc_reg(&mut self) -> u8 {
        assert!(self.count < self.regs.len(),
            "borgc: shader needs more than {} constant registers ({:?})", self.regs.len(), self.regs);
        let reg = self.regs[self.count];
        self.count += 1;
        reg
    }

    pub fn pin_int(&mut self, v: i32) -> u8 {
        if let Some(&r) = self.int_reg.get(&v) {
            return r;
        }
        let r = self.alloc_reg();
        self.uniforms.push((r, v as u32));
        self.int_reg.insert(v, r);
        r
    }

    pub fn int_operand(&mut self, sel: &mut Selector, v: i32) -> u32 {
        let kind = if self.vs_window {
            let idx = match self.int_reg.get(&v) {
                Some(&u) => u,
                None => {
                    assert!(self.vs_consts.len() < DRAW_VS_CONST_WORDS,
                        "borgc: vertex shader needs more than {} window constants", DRAW_VS_CONST_WORDS);
                    let u = DRAW_VS_CONST_U0 + self.vs_consts.len() as u8;
                    self.vs_consts.push((u, v as u32));
                    self.int_reg.insert(v, u);
                    u
                }
            };
            crate::Ubo::Uniform(idx)
        } else {
            crate::Ubo::Fixed(self.pin_int(v))
        };
        let r = sel.fresh();
        sel.ubo.insert(r, kind);
        r
    }

    pub fn vs_win(&mut self, sel: &mut Selector, bits: u32) -> u32 {
        let u = DRAW_VS_CONST_U0 + self.vs_consts.len() as u8;
        assert!(self.vs_consts.len() < DRAW_VS_CONST_WORDS,
            "borgc: vertex shader needs more than {} window constants", DRAW_VS_CONST_WORDS);
        self.vs_consts.push((u, bits));
        let v = sel.fresh();
        sel.ubo.insert(v, crate::Ubo::Uniform(u));
        v
    }

    pub fn vs_const(&mut self, sel: &mut Selector, bits: u32, derive: bool) -> u32 {
        if let Some(&v) = self.vs_by_bits.get(&bits) {
            return v;
        }
        let mut return_zero = false;
        let v = sel.fresh();
        let derivable = bits == 0
            || self.vs_by_bits.keys().any(|&a| a >= 1 && a < bits && self.vs_by_bits.get(&(bits - a)).is_some());
        if self.vs_consts.len() < DRAW_VS_CONST_WORDS && !(derive && derivable) && bits != 0 {
            let u = DRAW_VS_CONST_U0 as u32 + self.vs_consts.len() as u32;
            self.vs_consts.push((u as u8, bits));
            sel.ubo.insert(v, crate::Ubo::Uniform(u as u8));
        } else {
            if self.vs_zero.is_none() {
                let f = sel.fresh();
                sel.ubo.insert(f, crate::Ubo::Fixed(30));
                let z = sel.fresh();
                sel.prog.push(crate::BorgInstr { mnem: "ISUB", dst: z, srcs: vec![f, f], swz: vec![0, 0] });
                self.vs_zero = Some(z);
            }
            let z = self.vs_zero.unwrap();
            let mut small: Vec<(u32, u32)> =
                self.vs_by_bits.iter().filter(|(&b, _)| b >= 1 && b < bits).map(|(&b, &r)| (b, r)).collect();
            small.sort();
            if bits == 0 {
                self.vs_by_bits.insert(0, z);
                return_zero = true;
            } else {
                let pair = small.iter().find_map(|&(a, ra)| {
                    small.iter().find(|&&(b, _)| a + b == bits).map(|&(_, rb)| (ra, rb))
                });
                let (ra, rb) = pair.unwrap_or_else(|| {
                    panic!("borgc: vertex shader needs more than {} window constants", DRAW_VS_CONST_WORDS)
                });
                let t = sel.fresh();
                sel.prog.push(crate::BorgInstr { mnem: "IADD", dst: t, srcs: vec![z, ra], swz: vec![0, 0] });
                sel.prog.push(crate::BorgInstr { mnem: "IADD", dst: v, srcs: vec![t, rb], swz: vec![0, 0] });
            }
        }
        if return_zero {
            self.vs_zero.unwrap()
        } else {
            self.vs_by_bits.insert(bits, v);
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
    pub frag_coord_w: bool,
    pub fetch_ctl_uniform: Option<u32>,
    pub lod_ctl_uniform: Option<u32>,
    pub res_binding: HashMap<u32, u32>,
    pub prod: HashMap<u32, (nir_op, Vec<(u32, u8)>)>,
    pub tex_dsts: std::collections::HashSet<u32>,
    pub fattr_dsts: std::collections::HashSet<u32>,
    pub zero_ctl: Option<u32>,
    pub tex_derefs: std::collections::HashSet<u32>,
    pub one_vreg: Option<u32>,
    pub push_const_reg: HashMap<u32, u32>,
    pub vertex_id_def: Option<u32>,
    pub descriptor_defs: std::collections::HashSet<u32>,
    pub const_addr_chain: Option<(i32, u32)>,
}

impl ShaderState {
    pub fn new(regs: &'static [u8], is_vertex: bool, is_fragment: bool, flags: Flags) -> Self {
        ShaderState {
            sel: Selector::new(),
            cp: ConstPool::new(regs, is_vertex),
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
            frag_coord_w: Default::default(),
            fetch_ctl_uniform: Default::default(),
            lod_ctl_uniform: Default::default(),
            res_binding: Default::default(),
            prod: Default::default(),
            tex_dsts: Default::default(),
            fattr_dsts: Default::default(),
            zero_ctl: Default::default(),
            tex_derefs: Default::default(),
            one_vreg: Default::default(),
            push_const_reg: Default::default(),
            vertex_id_def: Default::default(),
            descriptor_defs: Default::default(),
            const_addr_chain: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cx(vertex: bool) -> ShaderState {
        ShaderState::new(&[17, 18, 19], vertex, !vertex, Flags(0))
    }

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

    #[test]
    fn const_regs_run_out() {
        let mut c = cx(false);
        assert_eq!((c.cp.alloc_reg(), c.cp.alloc_reg(), c.cp.alloc_reg()), (17, 18, 19));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.cp.alloc_reg())).is_err());
    }

    #[test]
    fn pinned_integers_share_a_register() {
        let mut c = cx(false);
        let (a, b) = (c.cp.pin_int(7), c.cp.pin_int(9));
        assert_eq!((c.cp.pin_int(7), a == b), (a, false));
        assert_eq!(c.cp.uniforms, [(17, 7), (18, 9)]);
    }

    #[test]
    fn vertex_window_words_start_at_u25_and_dedupe() {
        let mut c = cx(true);
        let a = c.cp.vs_const(&mut c.sel, 0x3f80_0000, false);
        assert_eq!(c.cp.vs_const(&mut c.sel, 0x3f80_0000, false), a);
        c.cp.vs_const(&mut c.sel, 0x4000_0000, false);
        assert_eq!(c.cp.vs_consts, [(25, 0x3f80_0000), (26, 0x4000_0000)]);
        assert!(matches!(c.sel.ubo[&a], crate::Ubo::Uniform(25)));
    }

    #[test]
    fn full_vertex_window_derives_constants_from_sums() {
        let mut c = cx(true);
        for b in 1..=DRAW_VS_CONST_WORDS as u32 {
            c.cp.vs_const(&mut c.sel, b, false);
        }
        let before = c.cp.vs_consts.len();
        c.cp.vs_const(&mut c.sel, 3 + 4, false);
        assert_eq!(c.cp.vs_consts.len(), before);
        assert!(c.sel.prog.iter().any(|i| i.mnem == "IADD"));
    }

    #[test]
    fn fragment_window_numbers_from_u20_and_dedupes() {
        let mut w = FragWindow::default();
        assert_eq!((w.slot(5), w.slot(6), w.slot(5)), (20, 21, 20));
        assert_eq!(w.consts, [(20, 5), (21, 6)]);
    }

    #[test]
    fn fragment_window_overflow_panics() {
        let mut w = FragWindow::default();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for b in 0..=crate::FRAG_CONST_MAX {
                w.slot(b);
            }
        }));
        assert!(r.is_err());
    }

    #[test]
    fn win_makes_a_uniform_vreg() {
        let (mut w, mut s) = (FragWindow::default(), Selector::new());
        let v = w.win(&mut s, 42);
        assert!(matches!(s.ubo[&v], crate::Ubo::Uniform(20)));
    }
}
