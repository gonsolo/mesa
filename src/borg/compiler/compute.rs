// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::{BorgInstr, NO_DST};
use crate::isel::resolve_vm;
use crate::consts::Kv;
use crate::state::ShaderState;
use crate::Ubo;
use compiler::bindings::*;
use compiler::nir::AsDef;
use std::collections::HashMap;

pub(crate) const SLOT_BASE_WORDS: u32 = 0x40000;
pub(crate) const SLOT_WORDS: u32 = 0x80000;
const LANES: u32 = 4; // BorgConfig.Simt: fragLanes
const MAX_WORDS: usize = 4096; // the DRAM code region below the first slot holds 16k
pub(crate) const WG_BASE_REG: u32 = 12;

pub(crate) struct Out {
    pub words: Vec<u32>,
    pub regs: Vec<(u8, u32)>,
    pub local: [u32; 3],
    /// bit 0: uses atomics, so the grid must not be split across runs.
    /// bit 1: reads NumWorkGroups; bits 8-12 name the preset register of x (y, z below it).
    /// bit 2: packed: four one-invocation workgroups along x share a quad, `local` is (4, 1, 1).
    pub flags: u32,
}

// ---------------------------------------------------------------------------------------------
// Compute as a stage of the graphics selector: the ALU ops are select.rs's,
// what follows is only what a compute shader has and a draw has not.

/// Compute-only selection state.
#[derive(Default)]
pub struct ComputeState {
    desc: HashMap<u32, (u32, u32)>, // descriptor marker vreg -> (set, binding)
    base_used: bool,
    atomics: bool,
    packed: bool,
    nwg: Option<u8>,
    presets: Vec<(u8, u32)>, // registers the driver or the grid split fills
    ws: [u32; 3],
    kreg: HashMap<u32, u32>, // constant bits -> placeholder vreg, placed by pool_consts
    pub(crate) kval: HashMap<u32, u32>,
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
/// Spill slots follow the pool in slot 34, one word per invocation of the workgroup.
pub(crate) const SPILL_BASE_WORDS: u32 = POOL_BASE_WORDS + POOL_MAX as u32;

impl ShaderState {
    pub(crate) fn invocations(&self) -> u32 {
        self.cs.ws.iter().product()
    }

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

    /// Moves each load of a constant address down to its first use, past stores to other constant
    /// addresses: a whole-struct copy loads every word before it stores any.
    fn sink_loads(&mut self) {
        let (prog, kval) = (&mut self.sel.prog, &self.cs.kval);
        for i in (0..prog.len()).rev() {
            if prog[i].mnem != "LOAD" {
                continue;
            }
            let Some(&a) = kval.get(&prog[i].srcs[0]) else { continue };
            let d = prog[i].dst;
            let mut j = i + 1;
            while j < prog.len() {
                let n = &prog[j];
                let stop = n.dst == d || n.srcs.contains(&d) || match n.mnem {
                    "STORE" => kval.get(&n.srcs[0]).map_or(true, |&b| b == a),
                    "EXPUSH" | "EXELSE" | "EXPOP" | "BRZ" | "BRNZ" | "BARRIER" | "LABEL" => true,
                    _ => false,
                };
                if stop {
                    break;
                }
                j += 1;
            }
            let ins = prog.remove(i);
            prog.insert(j - 1, ins);
        }
    }

    /// The `cap` most used constants become preset registers; the rest are loaded from the pool
    /// near each use (a load is reused for 16 instructions), each load a word of its own, so the
    /// pointer only steps forward. A load inside an `if` moves before its outermost EXPUSH: a
    /// masked step would leave the inactive lanes' pointer behind. With `hoist` off a load stays
    /// in place: after a mask change the pointer restarts from the one valid at the region's entry
    /// and steps over the words the branch used.
    pub(crate) fn pool_consts(&mut self, cap: usize, hoist: bool) -> Result<(), String> {
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
        let free = self.cp.free_regs();
        // The pool base, one, and a pointer to restart from after each barrier, loop head and loop.
        let ptrs = 2 + self.sel.prog.iter().filter(|i| matches!(i.mnem, "BARRIER" | "LABEL" | "BRNZ")).count();
        let pooled_n = if ks.len() <= free.min(cap) { 0 } else { ks.len() - free.saturating_sub(ptrs).min(cap) };
        let (keep, pooled) = ks.split_at(ks.len() - pooled_n);
        for &(v, bits) in keep {
            let r = self.cp.preset(bits);
            self.sel.ubo.insert(v, Ubo::Fixed(r));
        }
        if pooled.is_empty() {
            return Ok(());
        }
        if free < ptrs {
            return Err(format!("{} constants do not fit {free} registers", ks.len()));
        }
        let rb = self.cp.preset(POOL_BASE_WORDS);
        let base = self.fixed(rb);
        let r1 = self.cp.preset(1);
        let one = self.fixed(r1);
        let pooled: HashMap<u32, u32> = pooled.iter().copied().collect();
        const GAP: usize = 16;
        let mut loaded: HashMap<u32, (u32, usize)> = HashMap::new();
        // A pointer valid in the active lanes and the pool word it points at.
        let mut ptr = (base, 0usize);
        let mut frames: Vec<((u32, usize), HashMap<u32, (u32, usize)>)> = vec![];
        let (mut depth, mut anchor) = (0usize, 0usize);
        let mut out: Vec<BorgInstr> = Vec::with_capacity(self.sel.prog.len());
        for (n, mut ins) in std::mem::take(&mut self.sel.prog).into_iter().enumerate() {
            for s in ins.srcs.iter_mut() {
                let Some(&bits) = pooled.get(s) else { continue };
                if loaded.get(s).map_or(true, |&(_, last)| n - last > GAP) {
                    let hoisted = hoist && depth > 0;
                    while ptr.1 < self.cs.pool.len() {
                        let p = self.sel.fresh();
                        let at = if hoisted { anchor } else { out.len() };
                        out.insert(at, BorgInstr { mnem: "IADD", dst: p, srcs: vec![ptr.0, one], swz: vec![0, 0] });
                        anchor += hoisted as usize;
                        ptr = (p, ptr.1 + 1);
                    }
                    let at = if hoisted { anchor } else { out.len() };
                    let v = self.sel.fresh();
                    out.insert(at, BorgInstr { mnem: "LOAD", dst: v, srcs: vec![ptr.0], swz: vec![0] });
                    anchor += hoisted as usize;
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
                    if !hoist {
                        frames.push((ptr, loaded.clone()));
                    }
                }
                // Registers do not survive the other quads' turns, a pointer stepped in a loop
                // differs between trips, and one stepped under its mask between lanes.
                "BARRIER" | "LABEL" | "BRNZ" => {
                    let r = self.cp.preset(POOL_BASE_WORDS + self.cs.pool.len() as u32);
                    ptr = (self.fixed(r), self.cs.pool.len());
                    loaded.clear();
                }
                "EXELSE" if !hoist => (ptr, loaded) = frames.last().cloned().unwrap_or((ptr, loaded)),
                "EXPOP" => {
                    depth = depth.saturating_sub(1);
                    if let Some(f) = frames.pop() {
                        (ptr, loaded) = f;
                    }
                }
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
            Some(Ubo::Fixed(_)) => self.bits_of(v),
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

    fn int(&mut self, mnem: &'static str, a: Val, b: Val) -> Val {
        match (mnem, a, b) {
            (_, Val::K(x), Val::K(y)) => Val::K(match mnem {
                "ISLT" => ((x as i32) < (y as i32)) as u32,
                "ISEQ" => (x == y) as u32,
                "IAND" => x & y,
                "IOR" => x | y,
                _ => x ^ y,
            }),
            ("IAND", Val::K(1), o) | ("IAND", o, Val::K(1)) | ("IOR", Val::K(0), o) | ("IOR", o, Val::K(0)) => o,
            ("IAND", Val::K(0), _) | ("IAND", _, Val::K(0)) => Val::K(0),
            ("IOR", Val::K(1), _) | ("IOR", _, Val::K(1)) => Val::K(1),
            _ => self.bin(mnem, a, b),
        }
    }

    /// An integer that orders like the float, with -0 equal to +0, whether it is NaN, and
    /// whether a NaN key can lie below or above every number's key.
    unsafe fn float_key(&mut self, s: &nir_alu_src) -> (Val, Val, bool, bool) {
        let d = s.src.as_def();
        if let Some(&b) = self.consts.get(&d.index).filter(|_| d.num_components == 1) {
            let m = b & 0x7fff_ffff;
            let nan = m > 0x7f80_0000;
            return (Val::K(if b >> 31 != 0 { m.wrapping_neg() } else { m }), Val::K(nan as u32), nan, nan);
        }
        let x = resolve_vm(&self.sel.vec_map, d.index, s.swizzle[0]);
        let (x, inf) = (Val::R(x.0, x.1), Val::K(0x7f80_0000));
        if d.parent_instr().as_alu().is_some_and(|p| p.op == nir_op_fabs) {
            return (x, self.int("ISLT", inf, x), false, true);
        }
        let m = self.bin("IAND", x, Val::K(0x7fff_ffff));
        let sign = self.bin("ISHR", x, Val::K(31));
        let f = self.bin("IXOR", m, sign);
        (self.bin("ISUB", f, sign), self.int("ISLT", inf, m), true, true)
    }

    /// Float compares on the bits: the FSTEP of a difference is wrong for NaN, equal infinities
    /// and differences that flush to zero.
    pub(crate) unsafe fn exact_compare(&mut self, alu: &nir_alu_instr) {
        let s = alu.srcs_as_slice();
        let same = s[0].src.as_def().index == s[1].src.as_def().index && s[0].swizzle[0] == s[1].swizzle[0];
        let (ka, na, alo, ahi) = self.float_key(&s[0]);
        let (kb, nb, blo, bhi) = if same { (ka, na, alo, ahi) } else { self.float_key(&s[1]) };
        let unord = self.int("IOR", na, nb);
        let ord = self.int("IXOR", unord, Val::K(1));
        // NaN keys lie outside the numbers' keys: a NaN where it already makes the compare false
        // needs no mask.
        let eq_ord = if !(alo || ahi) || !(blo || bhi) { Val::K(1) } else { ord };
        let lt_ord = if !alo && !bhi { Val::K(1) } else { ord };
        let ge_ord = if !ahi && !blo { Val::K(1) } else { ord };
        let r = match alu.op {
            nir_op_flt | nir_op_slt => {
                let lt = self.int("ISLT", ka, kb);
                self.int("IAND", lt, lt_ord)
            }
            nir_op_fge | nir_op_sge => {
                let lt = self.int("ISLT", ka, kb);
                let ge = self.int("IXOR", lt, Val::K(1));
                self.int("IAND", ge, ge_ord)
            }
            nir_op_feq | nir_op_seq => {
                let eq = if same { Val::K(1) } else { self.int("ISEQ", ka, kb) };
                self.int("IAND", eq, eq_ord)
            }
            _ => {
                let eq = if same { Val::K(1) } else { self.int("ISEQ", ka, kb) };
                let eq = self.int("IAND", eq, eq_ord);
                self.int("IXOR", eq, Val::K(1))
            }
        };
        match r {
            Val::K(b) => self.set(alu.def.index, &[Val::K(if b != 0 { 1.0f32.to_bits() } else { 0 })]),
            Val::R(v, c) => {
                let f = self.sel.opc("I2F", &[(v, c)]);
                self.sel.vec_map.insert(alu.def.index, vec![(f, 0)]);
            }
        }
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

    /// The group the hardware runs: a packed quad holds four workgroups.
    fn group_id(&mut self, c: u8) -> Val {
        self.cs.base_used = true;
        let u = self.sel.fresh();
        self.sel.ubo.insert(u, Ubo::Uniform(29 + c));
        let b = self.fixed(WG_BASE_REG as u8 + c);
        Val::R(self.sel.opc("IADD", &[(u, 0), (b, 0)]), 0)
    }

    fn workgroup_id(&mut self, c: u8) -> Val {
        let g = self.group_id(c);
        if !self.cs.packed || c != 0 {
            return g;
        }
        let g = self.bin("ISHL", g, Val::K(2));
        let r30 = self.fixed(30);
        self.add(g, Val::R(r30, 0))
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
                    let wg = self.group_id(c);
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
            nir_intrinsic_load_local_invocation_id | nir_intrinsic_load_local_invocation_index if self.cs.packed => {
                let n = i.def.num_components as usize;
                self.set(d, &vec![Val::K(0); n]);
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
                // One quad runs every instruction in all its lanes in order: it needs no barrier.
                if control && ws[0] * ws[1] * ws[2] > LANES {
                    self.sel.prog.push(BorgInstr { mnem: "BARRIER", dst: NO_DST, srcs: vec![], swz: vec![] });
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
            nir_intrinsic_ssbo_atomic | nir_intrinsic_shared_atomic | nir_intrinsic_ssbo_atomic_swap
            | nir_intrinsic_shared_atomic_swap => {
                let ssbo = matches!(i.intrinsic, nir_intrinsic_ssbo_atomic | nir_intrinsic_ssbo_atomic_swap);
                let a = if ssbo {
                    self.cs.atomics = true;
                    self.word_addr(i.get_src(0), i.get_src(1), 0)
                } else {
                    let o = self.val(i.get_src(0), 0);
                    let w = self.shr(o, 2);
                    let w = self.add(w, Val::K(i.base() as u32 / 4));
                    let w = self.mask(w);
                    self.add(Val::K(SLOT_BASE_WORDS + 33 * SLOT_WORDS), w)
                };
                let s = if ssbo { 2 } else { 1 };
                let swap = i.atomic_op() == nir_atomic_op_cmpxchg;
                let a = self.vreg(a);
                // A swap's sources are the comparator, then the new value.
                let x = self.val(i.get_src(s + swap as usize), 0);
                let x = self.vreg(x);
                let cmp = if swap {
                    let c = self.val(i.get_src(s), 0);
                    Some(self.vreg(c))
                } else {
                    None
                };
                let lt = match i.atomic_op() {
                    nir_atomic_op_imin | nir_atomic_op_imax => Some("ISLT"),
                    nir_atomic_op_umin | nir_atomic_op_umax => Some("ISLTU"),
                    _ => None,
                };
                let alu = match i.atomic_op() {
                    nir_atomic_op_iadd => "IADD",
                    nir_atomic_op_iand => "IAND",
                    nir_atomic_op_ior => "IOR",
                    nir_atomic_op_ixor => "IXOR",
                    nir_atomic_op_xchg | nir_atomic_op_cmpxchg => "",
                    _ if lt.is_some() => "",
                    _ => {
                        self.fail("unsupported atomic op".into());
                        return;
                    }
                };
                let is_min = matches!(i.atomic_op(), nir_atomic_op_imin | nir_atomic_op_umin);
                // One lane at a time; the same registers in every lane's turn.
                let (old, new, t, u, w) = (self.sel.fresh(), self.sel.fresh(), self.sel.fresh(), self.sel.fresh(), self.sel.fresh());
                let r30 = self.fixed(30);
                // r30 counts across the workgroup's quads; the lane is its low bits.
                let (m, q) = (self.konst(LANES - 1, Kv::Int), self.sel.fresh());
                self.sel.prog.push(BorgInstr { mnem: "IAND", dst: q, srcs: vec![r30, m], swz: vec![0, 0] });
                for lane in 0..LANES {
                    let l = self.konst(lane, Kv::Int);
                    let p = &mut self.sel.prog;
                    let mut op = |mnem: &'static str, dst: u32, srcs: Vec<(u32, u8)>| {
                        let (srcs, swz) = srcs.into_iter().unzip();
                        p.push(BorgInstr { mnem, dst, srcs, swz });
                    };
                    op("ISEQ", t, vec![(q, 0), (l, 0)]);
                    op("EXPUSH", NO_DST, vec![(t, 0)]);
                    op("LOAD", old, vec![a]);
                    let v = if !alu.is_empty() {
                        op(alu, new, vec![(old, 0), x]);
                        (new, 0)
                    } else if let Some(c) = cmp {
                        // old + (old == cmp) * (x - old)
                        op("ISEQ", t, vec![(old, 0), c]);
                        op("ISUB", u, vec![x, (old, 0)]);
                        op("IMUL", w, vec![(t, 0), (u, 0)]);
                        op("IADD", new, vec![(old, 0), (w, 0)]);
                        (new, 0)
                    } else if let Some(lt) = lt {
                        // max: old + (old < x) * (x - old); min: x + (old < x) * (old - x)
                        let (p0, q0) = if is_min { (x, (old, 0)) } else { ((old, 0), x) };
                        op(lt, t, vec![(old, 0), x]);
                        op("ISUB", u, vec![q0, p0]);
                        op("IMUL", w, vec![(t, 0), (u, 0)]);
                        op("IADD", new, vec![p0, (w, 0)]);
                        (new, 0)
                    } else {
                        x
                    };
                    op("STORE", NO_DST, vec![a, v]);
                    op("EXPOP", NO_DST, vec![]);
                }
                self.set(d, &[Val::R(old, 0)]);
            }
            _ => self.fail(format!("unsupported intrinsic {}", i.info().name())),
        }
    }
}

const OUT_OF_REGISTERS: &str = "out of registers";

/// Fewer constants in registers leave more for values: retried like the graphics spill limits,
/// last compact: loads sunk to their uses and pool loads kept inside branches.
pub(crate) unsafe fn compile_unified(nir: *mut nir_shader) -> Result<Out, String> {
    let mut r = Err(String::new());
    for compact in [false, true] {
        for limit in [None, Some(10), Some(8), Some(6), Some(4)] {
            for cap in [usize::MAX, 8, 4, 0] {
                r = compile_unified_with(nir, cap, limit, compact, true);
                // Packed runs merge by bytes changed: atomics would lose updates.
                if matches!(&r, Ok(o) if o.flags & 5 == 5) {
                    r = compile_unified_with(nir, cap, limit, compact, false);
                }
                if !matches!(&r, Err(e) if e == OUT_OF_REGISTERS) {
                    return r;
                }
            }
        }
    }
    r
}

unsafe fn compile_unified_with(nir: *mut nir_shader, cap: usize, limit: Option<usize>, compact: bool, pack: bool) -> Result<Out, String> {
    let (mut cx, entry, _) = ShaderState::start(nir, crate::config::Flags(0));
    cx.spill_limit = limit;
    let ws = (*nir).info.workgroup_size;
    cx.cs.ws = [ws[0] as u32, ws[1] as u32, ws[2] as u32];
    if pack && cx.cs.ws == [1, 1, 1] {
        (cx.cs.ws, cx.cs.packed) = ([LANES, 1, 1], true);
    }
    cx.select_entry(entry);
    if let Some(e) = cx.cs.err.take() {
        return Err(e);
    }
    if !cx.check_selected() {
        return Err(format!("unsupported: {}", cx.skipped.join(", ")));
    }
    if compact {
        cx.sink_loads();
    }
    if cx.sel.prog.iter().any(|i| i.mnem == "BARRIER") {
        crate::opt::dce(&mut cx.sel.prog, &cx.out_roots);
        cx.spill_across_barriers()?;
    }
    cx.optimise();
    if cx.spill_slots > 0 {
        eprintln!("borgc[compute]: spilled {} value(s) to keep {:?} live", cx.spill_slots, limit);
    }
    cx.pool_consts(cap, !compact)?;
    crate::set_cs_pool(std::mem::take(&mut cx.cs.pool));
    let mut reserved: Vec<u8> = cx.cp.preset_regs().to_vec();
    reserved.extend([30u8, 31, 12, 13, 14]);
    let alloc = crate::fixed::allocate(&cx, &HashMap::new(), &mut reserved).ok_or(OUT_OF_REGISTERS)?;
    let enc = crate::emit::encode_program(&cx, &alloc);
    if enc.pending > 0 || enc.n_preload > 0 {
        return Err(format!("{} operand(s) without a register", enc.pending));
    }
    if enc.words.len() > MAX_WORDS {
        return Err(format!("{} instructions exceed the {MAX_WORDS}-word program limit", enc.words.len()));
    }
    let mut regs = cx.cp.presets().to_vec();
    regs.extend(cx.cs.presets.iter().copied());
    if cx.cs.base_used {
        regs.extend((0..3).map(|c| ((WG_BASE_REG + c) as u8, 0)));
    }
    let flags = cx.cs.atomics as u32 | cx.cs.nwg.map_or(0, |r| 2 | (r as u32) << 8) | (cx.cs.packed as u32) << 2;
    Ok(Out { words: enc.words, regs, local: cx.cs.ws, flags })
}

#[no_mangle]
pub unsafe extern "C" fn borgc_compile_compute(
    nir: *mut nir_shader,
    words: *mut u32, word_cap: u32, nwords: *mut u32,
    regs: *mut u32, reg_cap: u32, nregs: *mut u32,
    local: *mut u32, flags: *mut u32,
) -> u32 {
    crate::borgc_capture(nir, 1, 0, -1);
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
    let compiled = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_unified(nir))) {
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
