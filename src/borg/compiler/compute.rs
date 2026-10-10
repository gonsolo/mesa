// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
//
// Compute shaders (docs/B0_compiler_contract.md, "Compute shaders").
//
// A separate, deliberately small backend: compute shares nothing with the draw front end's
// register conventions, so it does its own selection, then hands the program to borgc's
// linear-scan `regalloc` (values are virtual registers until the end).
//
// Conventions implemented here:
//   * r30 = LocalInvocationIndex, WorkgroupID in the uniforms u29/u30/u31.
//   * Constants have no immediate form: each distinct constant gets a register the driver
//     presets over MMIO (returned as (register, value) pairs).
//   * A buffer binding (set, binding) lives in a fixed window of GPU memory the driver fills
//     and reads back: word index SLOT_BASE_WORDS + (set * 8 + binding) * SLOT_WORDS.  LS_BASE
//     stays 0, so LOAD/STORE take full word indices.
//   * robustBufferAccess: every buffer word index is masked to the binding's window (word_addr);
//     the host zero-fills the window past the buffer and copies back only the buffer's bytes.
//   * atomics are per-lane critical sections (EXPUSH(LocalInvocationIndex == i)).

use crate::encode::encode;
use crate::regalloc::regalloc;
use crate::{BorgInstr, NO_DST};
use crate::{collect_cf_marks, CfMark};
use compiler::bindings::*;
use compiler::nir::AsDef;
use std::collections::HashMap;

pub(crate) const SLOT_BASE_WORDS: u32 = 0x40000;
pub(crate) const SLOT_WORDS: u32 = 0x80000;
const LANES: u32 = 4; // BorgConfig.Simt: fragLanes
const MAX_WORDS: usize = 512;
/// Operand ids at or above this are physical registers (r30, preset constants); below, virtual.
const PHYS: u32 = 0x4000_0000;
const FIRST_VREG: u32 = 1000;
/// First of three registers (x, y, z) holding the dispatch's workgroup origin: 0 normally, the
/// slice start when the driver splits a grid over several simulator runs.
pub(crate) const WG_BASE_REG: u32 = 12;

/// One selected instruction. `srcs` are virtual or physical ids; `f3` is the funct3 operand slot.
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
    pub flags: u32,
}

struct Em {
    words: Vec<u32>, // raw words that need no register (EXELSE, EXPOP)
    prog: Vec<CI>,   // everything else; EXELSE/EXPOP are CI with mnem "RAW" and srcs[0] the word
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
    /// A register the driver presets to `c` before the dispatch (physical, taken from r29 down).
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
    /// Address of word `c` of a multi-word access whose first word is `a0`.  A constant address
    /// would take one preset register per word (there are only 18), so word c > 0 is the first
    /// word's register plus the shared constant c, as long as that stays inside the window.
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
            V::D(set, binding) => SLOT_BASE_WORDS + (set * 8 + binding) * SLOT_WORDS,
            _ => {
                self.fail("buffer index is not a known binding".into());
                0
            }
        };
        let o = self.src(off, 0);
        let w = self.shr(o, 2);
        // robustBufferAccess: the access stays inside the binding's window. The host zero-fills
        // the window past the buffer and copies back only the buffer's bytes, so an out-of-range
        // load reads zero and an out-of-range store is discarded; neither reaches another buffer.
        let w = self.add(w, V::C(extra_words));
        let w = match w {
            V::C(x) => V::C(x & (SLOT_WORDS - 1)),
            _ => self.bin("IAND", w, V::C(SLOT_WORDS - 1)),
        };
        self.add(V::C(base), w)
    }
}

pub(crate) unsafe fn compile(nir: *mut nir_shader) -> Result<Out, String> {
    let mut em = Em { words: vec![], prog: vec![], next: FIRST_VREG, next_const: 29, regs: vec![], cmap: HashMap::new(), vals: HashMap::new(), nregs: HashMap::new(), err: None, base_used: false, atomics: false };
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
                    nir_intrinsic_load_workgroup_id => {
                        em.base_used = true;
                        for c in 0..3u8 {
                            let rd = em.fresh();
                            // IADD rd, u(29+c), r(12+c) with funct3 = 1: rs1 is the uniform operand;
                            // r12-r14 hold the grid origin (0 unless the driver split the dispatch).
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
                        // Push constants are one more window (slot 8, set 1 binding 0) that the
                        // host fills at dispatch.
                        let o = em.src(i.get_src(0), 0);
                        let w = em.shr(o, 2);
                        let base_w = (i.base() as u32) / 4;
                        for c in 0..i.def.num_components as u32 {
                            let w2 = em.add(w, V::C(base_w + c));
                            let w2 = match w2 {
                                V::C(x) => V::C(x & (SLOT_WORDS - 1)),
                                _ => em.bin("IAND", w2, V::C(SLOT_WORDS - 1)),
                            };
                            let a = em.add(V::C(SLOT_BASE_WORDS + 8 * SLOT_WORDS), w2);
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
                        // A memory barrier is nothing here: a store is visible at once. A control
                        // barrier is nothing with one invocation per workgroup.
                        let ws = (*nir).info.workgroup_size;
                        let control = i.get_const_index(NIR_INTRINSIC_EXECUTION_SCOPE) != 0;
                        if control && ws[0] as u32 * ws[1] as u32 * ws[2] as u32 != 1 {
                            em.fail("barrier() with more than one invocation per workgroup".into());
                        }
                    }
                    nir_intrinsic_load_shared | nir_intrinsic_store_shared => {
                        // Workgroup-shared variables are one more window (slot 9); workgroups run
                        // one after the other, so they can all use the same one.
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
                            let a = em.add(V::C(SLOT_BASE_WORDS + 9 * SLOT_WORDS), w2);
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
    let reserved: Vec<u8> = em.cmap.values().copied().chain([30u8, 31, 12, 13, 14]).collect();
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
    Ok(Out { words: em.words, regs: em.regs, local: [ws[0] as u32, ws[1] as u32, ws[2] as u32], flags: em.atomics as u32 })
}

/// Compile a compute shader.  `words` receives the program, `regs` (register, value) pairs the
/// driver must preset, `local` the LocalSize.  Returns 0 on success.
///
/// # Safety
/// `nir` is a valid compute `nir_shader`; the buffers have the stated capacities.
#[no_mangle]
pub unsafe extern "C" fn borgc_compile_compute(
    nir: *mut nir_shader,
    words: *mut u32, word_cap: u32, nwords: *mut u32,
    regs: *mut u32, reg_cap: u32, nregs: *mut u32,
    local: *mut u32, flags: *mut u32,
) -> u32 {
    // An assert in the backend must refuse the shader, not abort the process.
    let compiled = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile(nir))) {
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
