// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::encode::encode;
use crate::regalloc::regalloc;
use crate::{BorgInstr, NO_DST};
use crate::{collect_cf_marks, CfMark};
use crate::isel::resolve_vm;
use crate::state::{Kv, ShaderState};
use crate::Ubo;
use compiler::bindings::*;
use compiler::nir::AsDef;
use std::collections::HashMap;

pub(crate) const SLOT_BASE_WORDS: u32 = 0x40000;
pub(crate) const SLOT_WORDS: u32 = 0x80000;
const LANES: u32 = 4; // BorgConfig.Simt: fragLanes
const MAX_WORDS: usize = 4096; // the DRAM code region below the first slot holds 16k
const PHYS: u32 = 0x4000_0000;
const FIRST_VREG: u32 = 1000;
pub(crate) const WG_BASE_REG: u32 = 12;

struct CI {
    mnem: &'static str,
    dst: u32, // NO_DST, or a virtual id
    srcs: [u32; 2],
    f3: u32,
}

#[derive(Clone, Copy)]
enum V {
    R(u32), // virtual id, or PHYS + r
    C(u32),
    /// Component 0/1 of a buffer descriptor: which binding it names.
    D(u32, u32),
}

pub(crate) struct Out {
    pub words: Vec<u32>,
    pub regs: Vec<(u8, u32)>,
    pub local: [u32; 3],
    /// bit 0: uses atomics, so the grid must not be split across runs.
    /// bit 1: reads NumWorkGroups; bits 8-12 name the preset register of x (y, z below it).
    pub flags: u32,
}

struct Em {
    words: Vec<u32>, // raw words that need no register (EXELSE, EXPOP)
    prog: Vec<CI>,
    next: u32,
    next_const: u8,
    regs: Vec<(u8, u32)>,
    cmap: HashMap<u32, u8>,
    vals: HashMap<(u32, u8), V>,
    /// NIR registers (decl_reg): one virtual register per component.
    nregs: HashMap<u32, Vec<u32>>,
    err: Option<String>,
    base_used: bool,
    atomics: bool,
    nwg_reg: Option<u8>,
}

impl Em {
    fn fail(&mut self, m: String) {
        if self.err.is_none() {
            eprintln!("borgc[compute]: {m}");
            self.err = Some(m);
        }
    }
    fn fresh(&mut self) -> u32 {
        let r = self.next;
        self.next += 1;
        r
    }
    fn creg(&mut self, c: u32) -> u32 {
        if let Some(&r) = self.cmap.get(&c) {
            return PHYS + r as u32;
        }
        if self.next_const < 15 {
            self.fail("too many distinct constants".into());
            return PHYS + 15;
        }
        let r = self.next_const;
        self.next_const -= 1;
        self.cmap.insert(c, r);
        self.regs.push((r, c));
        PHYS + r as u32
    }
    fn reg(&mut self, v: V) -> u32 {
        match v {
            V::R(r) => r,
            V::C(c) => self.creg(c),
            V::D(..) => {
                self.fail("buffer descriptor used as a value".into());
                PHYS + 1
            }
        }
    }
    fn op(&mut self, mnem: &'static str, rd: u32, rs1: u32, rs2: u32, f3: u32) {
        self.prog.push(CI { mnem, dst: rd, srcs: [rs1, rs2], f3 });
    }
    /// An instruction with no destination (STORE, EXPUSH).
    fn op_nd(&mut self, mnem: &'static str, rs1: u32, rs2: u32) {
        self.prog.push(CI { mnem, dst: NO_DST, srcs: [rs1, rs2], f3: 0 });
    }
    fn raw(&mut self, word: u32) {
        self.prog.push(CI { mnem: "RAW", dst: NO_DST, srcs: [word, 0], f3: 0 });
    }
    fn bin(&mut self, mnem: &'static str, a: V, b: V) -> V {
        let (ra, rb) = (self.reg(a), self.reg(b));
        let rd = self.fresh();
        self.op(mnem, rd, ra, rb, 0);
        V::R(rd)
    }
    fn add(&mut self, a: V, b: V) -> V {
        match (a, b) {
            (V::C(x), V::C(y)) => V::C(x.wrapping_add(y)),
            (V::C(0), o) | (o, V::C(0)) => o,
            _ => self.bin("IADD", a, b),
        }
    }
    fn mul(&mut self, a: V, b: V) -> V {
        match (a, b) {
            (V::C(x), V::C(y)) => V::C(x.wrapping_mul(y)),
            _ => self.bin("IMUL", a, b),
        }
    }
    /// Logical right shift by a constant amount.
    fn shr(&mut self, a: V, n: u32) -> V {
        match a {
            V::C(x) => V::C(x >> n),
            _ => self.bin("ISRL", a, V::C(n)),
        }
    }
    fn get(&mut self, d: u32, c: u8) -> V {
        match self.vals.get(&(d, c)) {
            Some(&v) => v,
            None => {
                self.fail(format!("value %{d}.{c} has no producer"));
                V::C(0)
            }
        }
    }
    fn src(&mut self, s: &nir_src, c: u8) -> V {
        self.get(s.as_def().index, c)
    }
    fn word_n(&mut self, a0: V, c: u32) -> V {
        if c == 0 {
            return a0;
        }
        if let V::C(x) = a0 {
            let off = x.wrapping_sub(SLOT_BASE_WORDS) & (SLOT_WORDS - 1);
            if off + c >= SLOT_WORDS {
                return V::C(x.wrapping_add(c)); // would leave the window: keep the exact constant
            }
            let r0 = self.reg(a0);
            return self.bin("IADD", V::R(r0), V::C(c));
        }
        a0
    }
    /// Word index of the byte offset `off` in the buffer named by `idx`'s first component.
    fn word_addr(&mut self, idx: &nir_src, off: &nir_src, extra_words: u32) -> V {
        let base = match self.src(idx, 0) {
            V::D(set, binding) => SLOT_BASE_WORDS + (set * 16 + binding) * SLOT_WORDS,
            _ => {
                self.fail("buffer index is not a known binding".into());
                0
            }
        };
        let o = self.src(off, 0);
        let w = self.shr(o, 2);
        let w = self.add(w, V::C(extra_words));
        let w = match w {
            V::C(x) => V::C(x & (SLOT_WORDS - 1)),
            _ => self.bin("IAND", w, V::C(SLOT_WORDS - 1)),
        };
        self.add(V::C(base), w)
    }
}

pub(crate) unsafe fn compile(nir: *mut nir_shader) -> Result<Out, String> {
    let mut em = Em { words: vec![], prog: vec![], next: FIRST_VREG, next_const: 29, regs: vec![], cmap: HashMap::new(), vals: HashMap::new(), nregs: HashMap::new(), err: None, base_used: false, atomics: false, nwg_reg: None };
    let entry = nir_shader_get_entrypoint(nir);
    let mut marks: HashMap<usize, Vec<CfMark>> = HashMap::new();
    let mut unsupported: Vec<&'static str> = Vec::new();
    collect_cf_marks((*entry).iter_body(), &mut marks, &mut unsupported);
    if !unsupported.is_empty() {
        return Err(format!("unsupported control flow: {}", unsupported.join(", ")));
    }
    for block in (*entry).iter_blocks() {
        for m in marks.get(&(block as *const nir_block as usize)).map(|v| v.as_slice()).unwrap_or(&[]) {
            match m.mnem {
                "EXPUSH" => {
                    let c = em.get(m.cond.unwrap(), 0);
                    let rc = em.reg(c);
                    em.op_nd("EXPUSH", rc, 0);
                }
                "EXELSE" => em.raw(0x5800_0000),
                _ => em.raw(0x5C00_0000),
            }
        }
        for instr in block.iter_instr_list() {
            if let Some(lc) = instr.as_load_const() {
                for c in 0..lc.def.num_components as usize {
                    em.vals.insert((lc.def.index, c as u8), V::C(lc.values()[c].u32_));
                }
            } else if let Some(u) = instr.as_undef() {
                // An undefined value (an unused component of a widened coordinate): any value.
                for c in 0..u.def.num_components as usize {
                    em.vals.insert((u.def.index, c as u8), V::C(0));
                }
            } else if let Some(alu) = instr.as_alu() {
                let n = alu.def.num_components as usize;
                let nsrc = alu.info().num_inputs as usize;
                let op = alu.op;
                for k in 0..n {
                    let s: Vec<V> = (0..nsrc)
                        .map(|i| {
                            let a = alu.get_src(i);
                            // vecN takes component 0 of each source; everything else follows k.
                            let c = if op == nir_op_vec2 || op == nir_op_vec3 || op == nir_op_vec4 { a.swizzle[0] } else { a.swizzle[k] };
                            em.get(a.src.as_def().index, c)
                        })
                        .collect();
                    let v = match op {
                        nir_op_mov | nir_op_i2i32 | nir_op_u2u32 => s[0],
                        nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => {
                            // vecN: component k comes from source k.
                            let a = alu.get_src(k);
                            em.get(a.src.as_def().index, a.swizzle[0])
                        }
                        nir_op_iadd => em.add(s[0], s[1]),
                        nir_op_imul => em.mul(s[0], s[1]),
                        nir_op_ishl => match (s[0], s[1]) {
                            (V::C(x), V::C(y)) => V::C(x << (y & 31)),
                            _ => em.bin("ISHL", s[0], s[1]),
                        },
                        nir_op_ushr => match (s[0], s[1]) {
                            (V::C(x), V::C(y)) => V::C(x >> (y & 31)),
                            _ => em.bin("ISRL", s[0], s[1]),
                        },
                        // Booleans are 32-bit: true is all ones. The compares yield 0/1, so negate.
                        nir_op_ieq32 => { let t = em.bin("ISEQ", s[0], s[1]); em.bin("ISUB", V::C(0), t) }
                        nir_op_ine32 => { let t = em.bin("ISEQ", s[0], s[1]); em.bin("IADD", t, V::C(u32::MAX)) }
                        nir_op_ilt32 => { let t = em.bin("ISLT", s[0], s[1]); em.bin("ISUB", V::C(0), t) }
                        nir_op_ige32 => { let t = em.bin("ISLT", s[0], s[1]); em.bin("IADD", t, V::C(u32::MAX)) }
                        nir_op_ult32 => { let t = em.bin("ISLTU", s[0], s[1]); em.bin("ISUB", V::C(0), t) }
                        nir_op_uge32 => { let t = em.bin("ISLTU", s[0], s[1]); em.bin("IADD", t, V::C(u32::MAX)) }
                        nir_op_b2i32 => em.bin("ISUB", V::C(0), s[0]),
                        nir_op_inot => em.bin("IXOR", s[0], V::C(u32::MAX)),
                        nir_op_iand => em.bin("IAND", s[0], s[1]),
                        nir_op_ior => em.bin("IOR", s[0], s[1]),
                        nir_op_fadd => em.bin("FADD", s[0], s[1]),
                        nir_op_fmul => em.bin("FMUL", s[0], s[1]),
                        nir_op_i2f32 | nir_op_u2f32 | nir_op_f2i32 | nir_op_f2u32 => {
                            let m = if matches!(alu.op, nir_op_i2f32 | nir_op_u2f32) { "I2F" } else { "F2I" };
                            let (ra, rd) = (em.reg(s[0]), em.fresh());
                            em.op(m, rd, ra, 0, 0);
                            V::R(rd)
                        }
                        nir_op_fneg => {
                            let (ra, rd) = (em.reg(s[0]), em.fresh());
                            em.op("FNEG", rd, ra, 0, 0);
                            V::R(rd)
                        }
                        nir_op_fsub => {
                            let (rb, nb) = (em.reg(s[1]), em.fresh());
                            em.op("FNEG", nb, rb, 0, 0);
                            em.bin("FADD", s[0], V::R(nb))
                        }
                        _ => {
                            em.fail(format!("unsupported ALU op {}", alu.info().name()));
                            V::C(0)
                        }
                    };
                    em.vals.insert((alu.def.index, k as u8), v);
                }
            } else if let Some(i) = instr.as_intrinsic() {
                let d = i.def.index;
                match i.intrinsic {
                    nir_intrinsic_decl_reg => {
                        let n = i.get_const_index(NIR_INTRINSIC_NUM_COMPONENTS);
                        let v: Vec<u32> = (0..n).map(|_| em.fresh()).collect();
                        em.nregs.insert(d, v);
                    }
                    nir_intrinsic_load_reg => {
                        let h = i.get_src(0).as_def().index;
                        for c in 0..i.def.num_components as usize {
                            let r = em.nregs.get(&h).map(|v| v[c]).unwrap_or(PHYS + 1);
                            em.vals.insert((d, c as u8), V::R(r));
                        }
                    }
                    nir_intrinsic_store_reg => {
                        let h = i.get_src(1).as_def().index;
                        let zero = em.creg(0);
                        for c in 0..i.get_src(0).num_components() as usize {
                            if i.write_mask() & (1 << c) == 0 {
                                continue;
                            }
                            let v = em.src(i.get_src(0), c as u8);
                            let rs = em.reg(v);
                            let r = em.nregs.get(&h).map(|v| v[c]).unwrap_or(PHYS + 1);
                            // A copy: IADD r, value, 0.
                            em.op("IADD", r, rs, zero, 0);
                        }
                    }
                    nir_intrinsic_load_base_workgroup_id => {
                        for c in 0..3 {
                            em.vals.insert((d, c), V::C(0));
                        }
                    }
                    nir_intrinsic_load_num_workgroups => {
                        // Presets the driver fills with the dispatch's group counts.
                        if em.nwg_reg.is_none() {
                            if em.next_const < 17 {
                                em.fail("too many distinct constants".into());
                            } else {
                                em.nwg_reg = Some(em.next_const);
                                for _ in 0..3 {
                                    em.regs.push((em.next_const, 0));
                                    em.next_const -= 1;
                                }
                            }
                        }
                        let x = em.nwg_reg.unwrap_or(15) as u32;
                        for c in 0..3u8 {
                            em.vals.insert((d, c), V::R(PHYS + x - c as u32));
                        }
                    }
                    nir_intrinsic_load_workgroup_id => {
                        em.base_used = true;
                        for c in 0..3u8 {
                            let rd = em.fresh();
                            em.op("IADD", rd, PHYS + 29 + c as u32, PHYS + WG_BASE_REG + c as u32, 1);
                            em.vals.insert((d, c), V::R(rd));
                        }
                    }
                    nir_intrinsic_load_base_global_invocation_id => {
                        for c in 0..3 {
                            em.vals.insert((d, c), V::C(0));
                        }
                    }
                    nir_intrinsic_load_global_invocation_id => {
                        // workgroup_id * local_size + local_id (r31 = x | y << 10 | z << 20).
                        em.base_used = true;
                        let ws = (*nir).info.workgroup_size;
                        for c in 0..3u8 {
                            let wg = em.fresh();
                            em.op("IADD", wg, PHYS + 29 + c as u32, PHYS + WG_BASE_REG + c as u32, 1);
                            let size = ws[c as usize] as u32;
                            let mut v = if size == 1 { V::R(wg) } else { em.mul(V::R(wg), V::C(size)) };
                            if size > 1 {
                                let sh = em.bin("ISRL", V::R(PHYS + 31), V::C(10 * c as u32));
                                let lid = em.bin("IAND", sh, V::C(1023));
                                v = em.add(v, lid);
                            }
                            em.vals.insert((d, c), v);
                        }
                    }
                    nir_intrinsic_load_local_invocation_index => {
                        em.vals.insert((d, 0), V::R(PHYS + 30));
                    }
                    nir_intrinsic_vulkan_resource_index => {
                        let set = i.get_const_index(NIR_INTRINSIC_DESC_SET);
                        let binding = i.get_const_index(NIR_INTRINSIC_BINDING);
                        em.vals.insert((d, 0), V::D(set, binding));
                        em.vals.insert((d, 1), V::D(set, binding));
                        em.vals.insert((d, 2), V::C(0));
                    }
                    nir_intrinsic_load_vulkan_descriptor => {
                        for c in 0..3u8 {
                            let v = em.src(i.get_src(0), c);
                            em.vals.insert((d, c), v);
                        }
                    }
                    nir_intrinsic_load_push_constant => {
                        let o = em.src(i.get_src(0), 0);
                        let w = em.shr(o, 2);
                        let base_w = (i.base() as u32) / 4;
                        let pbase = SLOT_BASE_WORDS + 32 * SLOT_WORDS;
                        for c in 0..i.def.num_components as u32 {
                            let a = match w {
                                V::C(x) => {
                                    let mut r = ((x + base_w) & (SLOT_WORDS - 1)) + c;
                                    let rb = em.creg(pbase);
                                    let mut a = V::R(rb);
                                    while r >= 4 {
                                        a = em.bin("IADD", a, V::C(4));
                                        r -= 4;
                                    }
                                    if r > 0 {
                                        a = em.bin("IADD", a, V::C(r));
                                    }
                                    a
                                }
                                _ => {
                                    let w2 = em.add(w, V::C(base_w + c));
                                    let w2 = em.bin("IAND", w2, V::C(SLOT_WORDS - 1));
                                    em.add(V::C(pbase), w2)
                                }
                            };
                            let ra = em.reg(a);
                            let rd = em.fresh();
                            em.op("LOAD", rd, ra, 0, 0);
                            em.vals.insert((d, c as u8), V::R(rd));
                        }
                    }
                    nir_intrinsic_load_ubo | nir_intrinsic_load_ssbo => {
                        let a0 = em.word_addr(i.get_src(0), i.get_src(1), 0);
                        for c in 0..i.def.num_components as u32 {
                            let a = match a0 {
                                V::C(_) => em.word_n(a0, c),
                                _ if c == 0 => a0,
                                _ => em.word_addr(i.get_src(0), i.get_src(1), c),
                            };
                            let ra = em.reg(a);
                            let rd = em.fresh();
                            em.op("LOAD", rd, ra, 0, 0);
                            em.vals.insert((d, c as u8), V::R(rd));
                        }
                    }
                    nir_intrinsic_store_ssbo => {
                        let mask = i.write_mask();
                        let a0 = em.word_addr(i.get_src(1), i.get_src(2), 0);
                        for c in 0..i.get_src(0).num_components() as u32 {
                            if mask & (1 << c) == 0 {
                                continue;
                            }
                            let a = match a0 {
                                V::C(_) => em.word_n(a0, c),
                                _ if c == 0 => a0,
                                _ => em.word_addr(i.get_src(1), i.get_src(2), c),
                            };
                            let ra = em.reg(a);
                            let v = em.src(i.get_src(0), c as u8);
                            let rv = em.reg(v);
                            em.op_nd("STORE", ra, rv);
                        }
                    }
                    nir_intrinsic_barrier => {
                        let ws = (*nir).info.workgroup_size;
                        let control = i.get_const_index(NIR_INTRINSIC_EXECUTION_SCOPE) != 0;
                        if control && ws[0] as u32 * ws[1] as u32 * ws[2] as u32 != 1 {
                            em.fail("barrier() with more than one invocation per workgroup".into());
                        }
                    }
                    nir_intrinsic_load_shared | nir_intrinsic_store_shared => {
                        let store = i.intrinsic == nir_intrinsic_store_shared;
                        let o = em.src(i.get_src(if store { 1 } else { 0 }), 0);
                        let w = em.shr(o, 2);
                        let base_w = (i.base() as u32) / 4;
                        let n = if store { i.get_src(0).num_components() as u32 } else { i.def.num_components as u32 };
                        for c in 0..n {
                            if store && i.write_mask() & (1 << c) == 0 {
                                continue;
                            }
                            let w2 = em.add(w, V::C(base_w + c));
                            let w2 = match w2 {
                                V::C(x) => V::C(x & (SLOT_WORDS - 1)),
                                _ => em.bin("IAND", w2, V::C(SLOT_WORDS - 1)),
                            };
                            let a = em.add(V::C(SLOT_BASE_WORDS + 33 * SLOT_WORDS), w2);
                            let ra = em.reg(a);
                            if store {
                                let v = em.src(i.get_src(0), c as u8);
                                let rv = em.reg(v);
                                em.op_nd("STORE", ra, rv);
                            } else {
                                let rd = em.fresh();
                                em.op("LOAD", rd, ra, 0, 0);
                                em.vals.insert((d, c as u8), V::R(rd));
                            }
                        }
                    }
                    nir_intrinsic_ssbo_atomic => {
                        em.atomics = true;
                        if i.atomic_op() != nir_atomic_op_iadd {
                            em.fail("only atomicAdd is supported".into());
                            continue;
                        }
                        let a = em.word_addr(i.get_src(0), i.get_src(1), 0);
                        let ra = em.reg(a);
                        let val = em.src(i.get_src(2), 0);
                        let rv = em.reg(val);
                        let old = em.fresh();
                        let sum = em.fresh();
                        let t = em.fresh();
                        for lane in 0..LANES {
                            let rl = em.creg(lane);
                            em.op("ISEQ", t, PHYS + 30, rl, 0);
                            em.op_nd("EXPUSH", t, 0);
                            em.op("LOAD", old, ra, 0, 0);
                            em.op("IADD", sum, old, rv, 0);
                            em.op_nd("STORE", ra, sum);
                            em.raw(0x5C00_0000); // EXPOP
                        }
                        em.vals.insert((d, 0), V::R(old));
                    }
                    _ => em.fail(format!("unsupported intrinsic {}", i.info().name())),
                }
            }
        }
    }
    if let Some(e) = em.err {
        return Err(e);
    }
    // Register allocation (borgc's linear scan), then encode.
    let bp: Vec<BorgInstr> = em
        .prog
        .iter()
        .map(|i| BorgInstr {
            mnem: i.mnem,
            dst: i.dst,
            srcs: if i.mnem == "RAW" { vec![] } else { i.srcs.iter().copied().filter(|&s| s < PHYS).collect() },
            swz: vec![],
        })
        .collect();
    if em.base_used {
        for c in 0..3u32 {
            em.regs.push(((WG_BASE_REG + c) as u8, 0));
        }
    }
    let nwg: Vec<u8> = em.nwg_reg.map_or(vec![], |r| vec![r, r - 1, r - 2]);
    let reserved: Vec<u8> = em.cmap.values().copied().chain(nwg).chain([30u8, 31, 12, 13, 14]).collect();
    let alloc = regalloc(&bp, &HashMap::new(), &reserved, &[], &HashMap::new());
    let phys = |x: u32| -> u8 { if x >= PHYS { (x - PHYS) as u8 } else { *alloc.get(&x).unwrap_or(&0) } };
    for i in &em.prog {
        if i.mnem == "RAW" {
            em.words.push(i.srcs[0]);
            continue;
        }
        let rd = if i.dst == NO_DST { 0 } else { phys(i.dst) };
        match encode(i.mnem, rd, phys(i.srcs[0]), phys(i.srcs[1]), 0, i.f3) {
            Some(w) => em.words.push(w),
            None => return Err(format!("cannot encode {}", i.mnem)),
        }
    }
    if em.words.len() > MAX_WORDS {
        return Err(format!("{} instructions exceed the {MAX_WORDS}-word program limit", em.words.len()));
    }
    let ws = (*nir).info.workgroup_size;
    Ok(Out { words: em.words, regs: em.regs, local: [ws[0] as u32, ws[1] as u32, ws[2] as u32], flags: em.atomics as u32 | em.nwg_reg.map_or(0, |r| 2 | (r as u32) << 8) })
}

// ---------------------------------------------------------------------------------------------
// Compute as a stage of the graphics selector (BORGC_UNIFIED_CS=1): the ALU ops are select.rs's,
// what follows is only what a compute shader has and a draw has not.

/// Compute-only selection state.
#[derive(Default)]
pub struct ComputeState {
    desc: HashMap<u32, (u32, u32)>, // descriptor marker vreg -> (set, binding)
    base_used: bool,
    atomics: bool,
    nwg: Option<u8>,
    presets: Vec<(u8, u32)>, // registers the driver or the grid split fills
    ws: [u32; 3],
    kreg: HashMap<u32, u32>, // constant bits -> placeholder vreg, placed by pool_consts
    kval: HashMap<u32, u32>,
    pub pool: Vec<u32>,
    pub err: Option<String>,
}

/// A value while selecting: a known constant or a register component.
#[derive(Clone, Copy)]
enum Val {
    K(u32),
    R(u32, u8),
}

/// Word index of the constant pool, the compute job's slot 34.
pub(crate) const POOL_BASE_WORDS: u32 = SLOT_BASE_WORDS + 34 * SLOT_WORDS;
const POOL_MAX: usize = 4096;

impl ShaderState {
    /// A constant's placeholder register; pool_consts makes it a preset or a pool load.
    pub(crate) fn cs_const(&mut self, bits: u32) -> u32 {
        if let Some(&v) = self.cs.kreg.get(&bits) {
            return v;
        }
        let v = self.sel.fresh();
        self.cs.kreg.insert(bits, v);
        self.cs.kval.insert(v, bits);
        v
    }

    /// The `cap` most used constants become preset registers; the rest are loaded from the pool
    /// near each use (a load is reused for 16 instructions), each load a word of its own, so the
    /// pointer only steps forward. A load inside an `if` moves before its outermost EXPUSH: a
    /// masked step would leave the inactive lanes' pointer behind.
    pub(crate) fn pool_consts(&mut self, cap: usize) -> Result<(), String> {
        let mut uses: HashMap<u32, usize> = HashMap::new();
        for i in &self.sel.prog {
            for s in &i.srcs {
                if self.cs.kval.contains_key(s) {
                    *uses.entry(*s).or_insert(0) += 1;
                }
            }
        }
        let mut ks: Vec<(u32, u32)> = uses.keys().map(|&v| (v, self.cs.kval[&v])).collect();
        ks.sort_by_key(|&(v, b)| (std::cmp::Reverse(uses[&v]), b));
        let free = self.cp.regs.len() - self.cp.count;
        let pooled_n = if ks.len() <= free.min(cap) { 0 } else { ks.len() - free.saturating_sub(2).min(cap) };
        let (keep, pooled) = ks.split_at(ks.len() - pooled_n);
        for &(v, bits) in keep {
            let r = self.cp.alloc_reg();
            self.cp.uniforms.push((r, bits));
            self.sel.ubo.insert(v, Ubo::Fixed(r));
        }
        if pooled.is_empty() {
            return Ok(());
        }
        if free < 2 {
            return Err(format!("{} constants do not fit {free} registers", ks.len()));
        }
        let rb = self.cp.alloc_reg();
        self.cp.uniforms.push((rb, POOL_BASE_WORDS));
        let base = self.fixed(rb);
        let r1 = self.cp.alloc_reg();
        self.cp.uniforms.push((r1, 1));
        let one = self.fixed(r1);
        let pooled: HashMap<u32, u32> = pooled.iter().copied().collect();
        const GAP: usize = 16;
        let mut loaded: HashMap<u32, (u32, usize)> = HashMap::new();
        let mut ptr: Option<u32> = None;
        let (mut depth, mut anchor) = (0usize, 0usize);
        let mut out: Vec<BorgInstr> = Vec::with_capacity(self.sel.prog.len());
        for (n, mut ins) in std::mem::take(&mut self.sel.prog).into_iter().enumerate() {
            for s in ins.srcs.iter_mut() {
                let Some(&bits) = pooled.get(s) else { continue };
                if loaded.get(s).map_or(true, |&(_, last)| n - last > GAP) {
                    let at = if depth == 0 { out.len() } else { anchor };
                    let p = match ptr {
                        None => base,
                        Some(prev) => {
                            let p = self.sel.fresh();
                            out.insert(at, BorgInstr { mnem: "IADD", dst: p, srcs: vec![prev, one], swz: vec![0, 0] });
                            anchor += (depth > 0) as usize;
                            p
                        }
                    };
                    let at = if depth == 0 { out.len() } else { anchor };
                    let v = self.sel.fresh();
                    out.insert(at, BorgInstr { mnem: "LOAD", dst: v, srcs: vec![p], swz: vec![0] });
                    anchor += (depth > 0) as usize;
                    ptr = Some(p);
                    self.cs.pool.push(bits);
                    loaded.insert(*s, (v, n));
                }
                let e = loaded.get_mut(s).unwrap();
                e.1 = n;
                *s = e.0;
            }
            match ins.mnem {
                "EXPUSH" => {
                    if depth == 0 {
                        anchor = out.len();
                    }
                    depth += 1;
                }
                "EXPOP" => depth = depth.saturating_sub(1),
                _ => {}
            }
            out.push(ins);
        }
        self.sel.prog = out;
        if self.cs.pool.len() > POOL_MAX {
            return Err(format!("a constant pool of {} words ({POOL_MAX} fit)", self.cs.pool.len()));
        }
        eprintln!("borgc[compute]: constant pool of {} word(s)", self.cs.pool.len());
        Ok(())
    }

    /// A source component as a register operand; a scalar constant gets a preset register.
    pub(crate) fn operand(&mut self, d: u32, c: u8) -> (u32, u8) {
        if self.is_compute && !self.sel.vec_map.contains_key(&d) {
            if let Some(&bits) = self.consts.get(&d) {
                return (self.konst(bits, Kv::Int), 0);
            }
        }
        resolve_vm(&self.sel.vec_map, d, c)
    }

    fn known(&self, d: u32, c: u8) -> Option<u32> {
        let (v, _) = resolve_vm(&self.sel.vec_map, d, c);
        if let Some(&b) = self.cs.kval.get(&v) {
            return Some(b);
        }
        if !self.sel.vec_map.contains_key(&v) {
            if let Some(&b) = self.consts.get(&v) {
                return Some(b);
            }
        }
        match self.sel.ubo.get(&v) {
            Some(Ubo::Fixed(r)) => self.cp.uniforms.iter().find(|(x, _)| x == r).map(|&(_, b)| b),
            _ => None,
        }
    }

    fn val(&mut self, s: &nir_src, c: u8) -> Val {
        let d = s.as_def().index;
        match self.known(d, c) {
            Some(b) => Val::K(b),
            None => {
                let (v, k) = self.operand(d, c);
                Val::R(v, k)
            }
        }
    }

    fn vreg(&mut self, v: Val) -> (u32, u8) {
        match v {
            Val::K(b) => (self.konst(b, Kv::Int), 0),
            Val::R(r, c) => (r, c),
        }
    }

    fn fixed(&mut self, r: u8) -> u32 {
        let v = self.sel.fresh();
        self.sel.ubo.insert(v, Ubo::Fixed(r));
        v
    }

    fn bin(&mut self, mnem: &'static str, a: Val, b: Val) -> Val {
        let (a, b) = (self.vreg(a), self.vreg(b));
        Val::R(self.sel.opc(mnem, &[a, b]), 0)
    }

    fn add(&mut self, a: Val, b: Val) -> Val {
        match (a, b) {
            (Val::K(x), Val::K(y)) => Val::K(x.wrapping_add(y)),
            (Val::K(0), o) | (o, Val::K(0)) => o,
            _ => self.bin("IADD", a, b),
        }
    }

    fn mul(&mut self, a: Val, b: Val) -> Val {
        match (a, b) {
            (Val::K(x), Val::K(y)) => Val::K(x.wrapping_mul(y)),
            _ => self.bin("IMUL", a, b),
        }
    }

    fn shr(&mut self, a: Val, n: u32) -> Val {
        match a {
            Val::K(x) => Val::K(x >> n),
            _ => self.bin("ISRL", a, Val::K(n)),
        }
    }

    fn mask(&mut self, a: Val) -> Val {
        match a {
            Val::K(x) => Val::K(x & (SLOT_WORDS - 1)),
            _ => self.bin("IAND", a, Val::K(SLOT_WORDS - 1)),
        }
    }

    fn set(&mut self, d: u32, vals: &[Val]) {
        if let [Val::K(b)] = vals {
            self.consts.insert(d, *b);
            return;
        }
        let comps: Vec<(u32, u8)> = vals.iter().map(|&v| self.vreg(v)).collect();
        self.sel.vec_map.insert(d, comps);
    }

    fn fail(&mut self, m: String) {
        if self.cs.err.is_none() {
            eprintln!("borgc[compute]: {m}");
            self.cs.err = Some(m);
        }
    }

    /// An integer op on constants: folded, so an address stays a constant.
    pub(crate) fn fold_const(&mut self, alu: &nir_alu_instr) -> bool {
        let s = alu.srcs_as_slice();
        if alu.def.num_components != 1 || s.len() != 2 {
            return false;
        }
        let (Some(a), Some(b)) = (self.known(s[0].src.as_def().index, s[0].swizzle[0]), self.known(s[1].src.as_def().index, s[1].swizzle[0])) else {
            return false;
        };
        let r = match alu.op {
            nir_op_iadd => a.wrapping_add(b),
            nir_op_isub => a.wrapping_sub(b),
            nir_op_imul => a.wrapping_mul(b),
            nir_op_ishl => a << (b & 31),
            nir_op_ushr => a >> (b & 31),
            nir_op_iand => a & b,
            nir_op_ior => a | b,
            nir_op_ixor => a ^ b,
            _ => return false,
        };
        self.consts.insert(alu.def.index, r);
        true
    }

    /// `t` where the condition `c` (1.0 or 0.0) holds, else `e`, bit for bit: an arithmetic blend
    /// turns an infinity into NaN and changes integer bits.
    fn select(&mut self, c: (u32, u8), t: (u32, u8), e: (u32, u8)) -> u32 {
        let zero = self.vreg(Val::K(0));
        let ci = self.sel.opc("F2I", &[c]);
        let mask = self.sel.opc("ISUB", &[zero, (ci, 0)]);
        let x = self.sel.opc("IXOR", &[t, e]);
        let xm = self.sel.opc("IAND", &[(x, 0), (mask, 0)]);
        self.sel.opc("IXOR", &[e, (xm, 0)])
    }

    pub(crate) unsafe fn exact_minmax(&mut self, alu: &nir_alu_instr) {
        let s = alu.srcs_as_slice();
        let a = resolve_vm(&self.sel.vec_map, s[0].src.as_def().index, s[0].swizzle[0]);
        let b = resolve_vm(&self.sel.vec_map, s[1].src.as_def().index, s[1].swizzle[0]);
        let d = self.sel.fsub(b, a);
        let lt = self.sel.opc("FSTEP", &[(d, 0)]);
        let v = if alu.op == nir_op_fmin { self.select((lt, 0), a, b) } else { self.select((lt, 0), b, a) };
        self.sel.vec_map.insert(alu.def.index, vec![(v, 0)]);
    }

    pub(crate) unsafe fn exact_bcsel(&mut self, alu: &nir_alu_instr) {
        let s = alu.srcs_as_slice();
        let comps = (0..alu.def.num_components as usize)
            .map(|k| {
                let c = resolve_vm(&self.sel.vec_map, s[0].src.as_def().index, s[0].swizzle[k]);
                let t = resolve_vm(&self.sel.vec_map, s[1].src.as_def().index, s[1].swizzle[k]);
                let e = resolve_vm(&self.sel.vec_map, s[2].src.as_def().index, s[2].swizzle[k]);
                (self.select(c, t, e), 0u8)
            })
            .collect();
        self.sel.vec_map.insert(alu.def.index, comps);
    }

    pub(crate) unsafe fn undef(&mut self, u: &nir_undef_instr) {
        let zeros = vec![Val::K(0); u.def.num_components as usize];
        self.set(u.def.index, &zeros);
    }

    /// Word index of the byte offset `off` in the buffer named by `idx`'s first component.
    fn word_addr(&mut self, idx: &nir_src, off: &nir_src, extra: u32) -> Val {
        let m = resolve_vm(&self.sel.vec_map, idx.as_def().index, 0).0;
        let Some(&(set, binding)) = self.cs.desc.get(&m) else {
            self.fail("buffer index is not a known binding".into());
            return Val::K(0);
        };
        let base = SLOT_BASE_WORDS + (set * 16 + binding) * SLOT_WORDS;
        let o = self.val(off, 0);
        let w = self.shr(o, 2);
        let w = self.add(w, Val::K(extra));
        let w = self.mask(w);
        self.add(Val::K(base), w)
    }

    /// Component `c` of a buffer access starting at `a0`.
    fn word_n(&mut self, a0: Val, idx: &nir_src, off: &nir_src, c: u32) -> Val {
        match a0 {
            _ if c == 0 => a0,
            Val::K(x) if x.wrapping_sub(SLOT_BASE_WORDS) % SLOT_WORDS + c >= SLOT_WORDS => Val::K(x.wrapping_add(c)),
            Val::K(_) => {
                let r = self.vreg(a0);
                self.bin("IADD", Val::R(r.0, r.1), Val::K(c))
            }
            _ => self.word_addr(idx, off, c),
        }
    }

    fn load(&mut self, a: Val) -> Val {
        let a = self.vreg(a);
        Val::R(self.sel.opc("LOAD", &[a]), 0)
    }

    fn store(&mut self, a: Val, v: Val) {
        let (a, v) = (self.vreg(a), self.vreg(v));
        self.sel.prog.push(BorgInstr { mnem: "STORE", dst: NO_DST, srcs: vec![a.0, v.0], swz: vec![a.1, v.1] });
    }

    fn workgroup_id(&mut self, c: u8) -> Val {
        self.cs.base_used = true;
        let u = self.sel.fresh();
        self.sel.ubo.insert(u, Ubo::Uniform(29 + c));
        let b = self.fixed(WG_BASE_REG as u8 + c);
        Val::R(self.sel.opc("IADD", &[(u, 0), (b, 0)]), 0)
    }

    pub(crate) unsafe fn compute_intrinsic(&mut self, i: &nir_intrinsic_instr) {
        let d = i.def.index;
        match i.intrinsic {
            nir_intrinsic_load_base_workgroup_id | nir_intrinsic_load_base_global_invocation_id => {
                self.set(d, &[Val::K(0); 3]);
            }
            nir_intrinsic_load_workgroup_id => {
                let v: Vec<Val> = (0..3).map(|c| self.workgroup_id(c)).collect();
                self.set(d, &v);
            }
            nir_intrinsic_load_global_invocation_id => {
                // workgroup_id * local_size + local_id (r31 = x | y << 10 | z << 20).
                let ws = self.cs.ws;
                let mut v = vec![];
                for c in 0..3u8 {
                    let wg = self.workgroup_id(c);
                    let size = ws[c as usize];
                    let mut x = if size == 1 { wg } else { self.mul(wg, Val::K(size)) };
                    if size > 1 {
                        let r31 = self.fixed(31);
                        let sh = self.bin("ISRL", Val::R(r31, 0), Val::K(10 * c as u32));
                        let lid = self.bin("IAND", sh, Val::K(1023));
                        x = self.add(x, lid);
                    }
                    v.push(x);
                }
                self.set(d, &v);
            }
            nir_intrinsic_load_local_invocation_id => {
                // r31 = x | y << 10 | z << 20.
                let r31 = self.fixed(31);
                let mut v = vec![];
                for c in 0..3u32 {
                    let sh = self.shr(Val::R(r31, 0), 10 * c);
                    v.push(if c == 2 { sh } else { self.bin("IAND", sh, Val::K(1023)) });
                }
                self.set(d, &v);
            }
            nir_intrinsic_load_local_invocation_index => {
                let r30 = self.fixed(30);
                self.set(d, &[Val::R(r30, 0)]);
            }
            nir_intrinsic_load_num_workgroups => {
                // Presets the driver fills with the dispatch's group counts.
                let x = match self.cs.nwg {
                    Some(x) => x,
                    None => {
                        let x = self.cp.alloc_reg();
                        let (y, z) = (self.cp.alloc_reg(), self.cp.alloc_reg());
                        self.cs.presets.extend([(x, 0), (y, 0), (z, 0)]);
                        self.cs.nwg = Some(x);
                        x
                    }
                };
                let v: Vec<Val> = (0..3).map(|c| Val::R(self.fixed(x - c), 0)).collect();
                self.set(d, &v);
            }
            nir_intrinsic_vulkan_resource_index => {
                let m = self.sel.fresh();
                self.cs.desc.insert(m, (i.get_const_index(NIR_INTRINSIC_DESC_SET), i.get_const_index(NIR_INTRINSIC_BINDING)));
                let zero = self.vreg(Val::K(0));
                self.sel.vec_map.insert(d, vec![(m, 0), (m, 1), zero]);
            }
            nir_intrinsic_load_vulkan_descriptor => {
                let s = i.get_src(0).as_def().index;
                let comps = (0..3).map(|c| resolve_vm(&self.sel.vec_map, s, c)).collect();
                self.sel.vec_map.insert(d, comps);
            }
            nir_intrinsic_load_push_constant => {
                let o = self.val(i.get_src(0), 0);
                let w = self.shr(o, 2);
                let base_w = (i.base() as u32) / 4;
                let pbase = SLOT_BASE_WORDS + 32 * SLOT_WORDS;
                let mut v = vec![];
                for c in 0..i.def.num_components as u32 {
                    let a = match w {
                        Val::K(x) => {
                            // Small steps from one base keep the distinct constants few.
                            let mut r = ((x + base_w) & (SLOT_WORDS - 1)) + c;
                            let rb = self.konst(pbase, Kv::Int);
                            let mut a = Val::R(rb, 0);
                            while r >= 4 {
                                a = self.bin("IADD", a, Val::K(4));
                                r -= 4;
                            }
                            if r > 0 {
                                a = self.bin("IADD", a, Val::K(r));
                            }
                            a
                        }
                        _ => {
                            let w2 = self.add(w, Val::K(base_w + c));
                            let w2 = self.bin("IAND", w2, Val::K(SLOT_WORDS - 1));
                            self.add(Val::K(pbase), w2)
                        }
                    };
                    v.push(self.load(a));
                }
                self.set(d, &v);
            }
            nir_intrinsic_load_ubo | nir_intrinsic_load_ssbo => {
                let a0 = self.word_addr(i.get_src(0), i.get_src(1), 0);
                let mut v = vec![];
                for c in 0..i.def.num_components as u32 {
                    let a = self.word_n(a0, i.get_src(0), i.get_src(1), c);
                    v.push(self.load(a));
                }
                self.set(d, &v);
            }
            nir_intrinsic_store_ssbo => {
                let a0 = self.word_addr(i.get_src(1), i.get_src(2), 0);
                for c in 0..i.get_src(0).num_components() as u32 {
                    if i.write_mask() & (1 << c) == 0 {
                        continue;
                    }
                    let a = self.word_n(a0, i.get_src(1), i.get_src(2), c);
                    let v = self.val(i.get_src(0), c as u8);
                    self.store(a, v);
                }
            }
            nir_intrinsic_barrier => {
                let ws = self.cs.ws;
                let control = i.get_const_index(NIR_INTRINSIC_EXECUTION_SCOPE) != 0;
                if control && ws[0] * ws[1] * ws[2] != 1 {
                    self.fail("barrier() with more than one invocation per workgroup".into());
                }
            }
            nir_intrinsic_load_shared | nir_intrinsic_store_shared => {
                let store = i.intrinsic == nir_intrinsic_store_shared;
                let o = self.val(i.get_src(if store { 1 } else { 0 }), 0);
                let w = self.shr(o, 2);
                let base_w = (i.base() as u32) / 4;
                let n = if store { i.get_src(0).num_components() as u32 } else { i.def.num_components as u32 };
                let mut v = vec![];
                for c in 0..n {
                    if store && i.write_mask() & (1 << c) == 0 {
                        continue;
                    }
                    let w2 = self.add(w, Val::K(base_w + c));
                    let w2 = self.mask(w2);
                    let a = self.add(Val::K(SLOT_BASE_WORDS + 33 * SLOT_WORDS), w2);
                    if store {
                        let x = self.val(i.get_src(0), c as u8);
                        self.store(a, x);
                    } else {
                        v.push(self.load(a));
                    }
                }
                if !store {
                    self.set(d, &v);
                }
            }
            nir_intrinsic_ssbo_atomic => {
                self.cs.atomics = true;
                if i.atomic_op() != nir_atomic_op_iadd {
                    self.fail("only atomicAdd is supported".into());
                    return;
                }
                let a = self.word_addr(i.get_src(0), i.get_src(1), 0);
                let a = self.vreg(a);
                let x = self.val(i.get_src(2), 0);
                let x = self.vreg(x);
                // One lane at a time; the same registers in every lane's turn.
                let (old, sum, t) = (self.sel.fresh(), self.sel.fresh(), self.sel.fresh());
                let r30 = self.fixed(30);
                for lane in 0..LANES {
                    let l = self.konst(lane, Kv::Int);
                    self.sel.prog.push(BorgInstr { mnem: "ISEQ", dst: t, srcs: vec![r30, l], swz: vec![0, 0] });
                    self.sel.prog.push(BorgInstr { mnem: "EXPUSH", dst: NO_DST, srcs: vec![t], swz: vec![0] });
                    self.sel.prog.push(BorgInstr { mnem: "LOAD", dst: old, srcs: vec![a.0], swz: vec![a.1] });
                    self.sel.prog.push(BorgInstr { mnem: "IADD", dst: sum, srcs: vec![old, x.0], swz: vec![0, x.1] });
                    self.sel.prog.push(BorgInstr { mnem: "STORE", dst: NO_DST, srcs: vec![a.0, sum], swz: vec![a.1, 0] });
                    self.sel.prog.push(BorgInstr { mnem: "EXPOP", dst: NO_DST, srcs: vec![], swz: vec![] });
                }
                self.set(d, &[Val::R(old, 0)]);
            }
            _ => self.fail(format!("unsupported intrinsic {}", i.info().name())),
        }
    }
}

const OUT_OF_REGISTERS: &str = "out of registers";

/// Fewer constants in registers leave more for values: retried like the graphics spill limits.
pub(crate) unsafe fn compile_unified(nir: *mut nir_shader) -> Result<Out, String> {
    let mut r = Err(String::new());
    for cap in [usize::MAX, 8, 4, 0] {
        r = compile_unified_with(nir, cap);
        if !matches!(&r, Err(e) if e == OUT_OF_REGISTERS) {
            break;
        }
    }
    r
}

unsafe fn compile_unified_with(nir: *mut nir_shader, cap: usize) -> Result<Out, String> {
    let (mut cx, entry, _) = ShaderState::start(nir, crate::config::Flags(0));
    let ws = (*nir).info.workgroup_size;
    cx.cs.ws = [ws[0] as u32, ws[1] as u32, ws[2] as u32];
    cx.select_entry(entry);
    if let Some(e) = cx.cs.err.take() {
        return Err(e);
    }
    if !cx.check_selected() {
        return Err(format!("unsupported: {}", cx.skipped.join(", ")));
    }
    cx.optimise();
    cx.pool_consts(cap)?;
    crate::set_cs_pool(std::mem::take(&mut cx.cs.pool));
    let mut reserved: Vec<u8> = cx.cp.regs[..cx.cp.count].to_vec();
    reserved.extend([30u8, 31, 12, 13, 14]);
    let alloc = crate::fixed::allocate(&cx, &HashMap::new(), &mut reserved).ok_or(OUT_OF_REGISTERS)?;
    let enc = crate::emit::encode_program(&cx, &alloc);
    if enc.pending > 0 || enc.n_preload > 0 {
        return Err(format!("{} operand(s) without a register", enc.pending));
    }
    if enc.words.len() > MAX_WORDS {
        return Err(format!("{} instructions exceed the {MAX_WORDS}-word program limit", enc.words.len()));
    }
    let mut regs = cx.cp.uniforms.clone();
    regs.extend(cx.cs.presets.iter().copied());
    if cx.cs.base_used {
        regs.extend((0..3).map(|c| ((WG_BASE_REG + c) as u8, 0)));
    }
    let flags = cx.cs.atomics as u32 | cx.cs.nwg.map_or(0, |r| 2 | (r as u32) << 8);
    Ok(Out { words: enc.words, regs, local: cx.cs.ws, flags })
}

#[no_mangle]
pub unsafe extern "C" fn borgc_compile_compute(
    nir: *mut nir_shader,
    words: *mut u32, word_cap: u32, nwords: *mut u32,
    regs: *mut u32, reg_cap: u32, nregs: *mut u32,
    local: *mut u32, flags: *mut u32,
) -> u32 {
    let key = if !nir.is_null() && std::env::var("BORGC_CORPUS_DIR").is_ok() {
        (*nir).to_string().ok().map(|s| crate::corpus::key(&s, 0))
    } else {
        None
    };
    let ret = compile_compute_inner(nir, words, word_cap, nwords, regs, reg_cap, nregs, local, flags);
    if let Some(key) = key {
        // flags, local size, program words, then the (register, value) presets.
        let mut blob: Vec<u32> = vec![];
        if ret == 0 {
            blob.push(*flags);
            blob.extend((0..3).map(|k| *local.add(k)));
            blob.extend(std::slice::from_raw_parts(words, *nwords as usize));
            blob.extend(std::slice::from_raw_parts(regs, 2 * *nregs as usize));
        }
        let bytes: Vec<u8> = blob.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::corpus::record(key, ret, &bytes);
    }
    ret
}

unsafe fn compile_compute_inner(
    nir: *mut nir_shader,
    words: *mut u32, word_cap: u32, nwords: *mut u32,
    regs: *mut u32, reg_cap: u32, nregs: *mut u32,
    local: *mut u32, flags: *mut u32,
) -> u32 {
    // An assert in the backend must refuse the shader, not abort the process.
    crate::set_cs_pool(Vec::new());
    let unified = std::env::var("BORGC_UNIFIED_CS").is_ok();
    let compiled = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| if unified { compile_unified(nir) } else { compile(nir) })) {
        Ok(r) => r,
        Err(_) => return 1,
    };
    match compiled {
        Ok(o) if o.words.len() as u32 <= word_cap && o.regs.len() as u32 <= reg_cap => {
            for (i, w) in o.words.iter().enumerate() {
                *words.add(i) = *w;
            }
            for (i, (r, v)) in o.regs.iter().enumerate() {
                *regs.add(2 * i) = *r as u32;
                *regs.add(2 * i + 1) = *v;
            }
            *nwords = o.words.len() as u32;
            *nregs = o.regs.len() as u32;
            for k in 0..3 {
                *local.add(k) = o.local[k];
            }
            *flags = o.flags;
            0
        }
        Ok(o) => {
            eprintln!("borgc[compute]: {} words / {} preset registers exceed the driver's {word_cap} / {reg_cap}", o.words.len(), o.regs.len());
            2
        }
        Err(e) => {
            eprintln!("borgc[compute]: refused: {e}");
            1
        }
    }
}
