// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::state::{Selector, ShaderState};
use crate::compute::{POOL_BASE_WORDS, POOL_MAX as CS_POOL_MAX};
use crate::{BorgInstr, Ubo};
use std::collections::HashMap;

/// Words of a window's pool in memory, the driver's BORGC_POOL_MAX.
const POOL_MAX: usize = 64;
/// Pooled words are tagged u100.. until the fetch replaces them.
const POOL_TAG: u8 = 100;

/// A draw shader's uniform window: words u{base}.. in creation order, more than it holds until
/// pool_window moves the rest to memory.
pub struct Window {
    base: u8,
    size: usize,
    /// The vertex attribute slot the pool is fetched from.
    slot: u32,
    words: Vec<(u8, u32)>,
    vregs: HashMap<u32, u32>,
}

impl Window {
    pub fn fragment() -> Self {
        Window { base: 20, size: 11, slot: 40, words: Vec::new(), vregs: HashMap::new() }
    }

    pub fn vertex() -> Self {
        Window { base: 25, size: 6, slot: 42, words: Vec::new(), vregs: HashMap::new() }
    }

    /// The words a shader may have: those the window keeps beside the constant 1 and the pool's
    /// control word, and a full pool.
    pub fn max(&self) -> usize {
        self.size - 2 + POOL_MAX
    }

    pub fn word(&mut self, sel: &mut Selector, bits: u32) -> u32 {
        if let Some(&v) = self.vregs.get(&bits) {
            return v;
        }
        assert!(self.words.len() < self.max(), "borgc: shader needs more than {} constants", self.max());
        let u = self.base + self.words.len() as u8;
        self.words.push((u, bits));
        let v = sel.fresh();
        sel.ubo.insert(v, Ubo::Uniform(u));
        self.vregs.insert(bits, v);
        v
    }

    /// Room for `n` more words before the pool pass.
    pub fn room(&self, n: usize) -> bool {
        self.words.len() + n < self.max()
    }
}

pub struct ConstPool {
    regs: &'static [u8],
    count: usize,
    uniforms: Vec<(u8, u32)>,
    zero: Option<u32>,
}

impl ConstPool {
    pub fn new(regs: &'static [u8]) -> Self {
        ConstPool {
            regs,
            count: 0,
            uniforms: Vec::new(),
            zero: None,
        }
    }

    /// 0, built once as r30 - r30.
    pub fn zero(&mut self, sel: &mut Selector) -> u32 {
        if let Some(z) = self.zero {
            return z;
        }
        let f = sel.fresh();
        sel.ubo.insert(f, crate::Ubo::Fixed(30));
        let z = sel.fresh();
        sel.prog.push(crate::BorgInstr { mnem: "ISUB", dst: z, srcs: vec![f, f], swz: vec![0, 0] });
        self.zero = Some(z);
        z
    }

    /// A preset register the driver fills with `bits`.
    pub fn preset(&mut self, bits: u32) -> u8 {
        let r = self.alloc_reg();
        self.uniforms.push((r, bits));
        r
    }

    /// The registers handed out as presets so far.
    pub fn preset_regs(&self) -> &[u8] {
        &self.regs[..self.count]
    }

    pub fn free_regs(&self) -> usize {
        self.regs.len() - self.count
    }

    pub fn presets(&self) -> &[(u8, u32)] {
        &self.uniforms
    }

    pub fn alloc_reg(&mut self) -> u8 {
        assert!(self.count < self.regs.len(),
            "borgc: shader needs more than {} constant registers ({:?})", self.regs.len(), self.regs);
        let reg = self.regs[self.count];
        self.count += 1;
        reg
    }
}

impl ShaderState {
    /// A register holding `bits`: a window word in a fragment or vertex shader (in memory when the
    /// window is full), a preset register or pool word in a compute shader.
    pub fn konst(&mut self, bits: u32) -> u32 {
        if self.is_compute {
            return self.cs_const(bits);
        }
        if self.is_vertex && bits == 0 {
            return self.cp.zero(&mut self.sel);
        }
        self.win.word(&mut self.sel, bits)
    }

    /// The window words the driver stages for the draw, (u, bits).
    pub fn window(&self) -> Vec<(u8, u32)> {
        self.win.words.clone()
    }

    /// The bits a constant's register holds: a placeholder, a window word or a preset register.
    pub fn bits_of(&self, v: u32) -> Option<u32> {
        if let Some(&b) = self.cs.kval.get(&v) {
            return Some(b);
        }
        match self.sel.ubo.get(&v) {
            Some(&Ubo::Uniform(u)) => self.win.words.iter().find(|&&(w, _)| w == u).map(|&(_, b)| b),
            Some(&Ubo::Fixed(r)) => self.cp.uniforms.iter().find(|&&(x, _)| x == r).map(|&(_, b)| b),
            _ => None,
        }
    }

    /// 1.0, built once: FSTEP of r30 in a fragment shader (no window word).
    pub fn one(&mut self) -> u32 {
        if let Some(v) = self.one_vreg {
            return v;
        }
        if self.is_compute {
            let v = self.cs_const(1.0f32.to_bits());
            self.one_vreg = Some(v);
            return v;
        }
        let v = self.sel.fresh();
        let v = if self.is_fragment {
            let c = self.sel.fresh();
            self.sel.ubo.insert(c, crate::Ubo::Fixed(30));
            self.per_pixel_fixed.insert(c);
            self.sel.prog.push(crate::BorgInstr { mnem: "FSTEP", dst: v, srcs: vec![c], swz: vec![0] });
            v
        } else if self.is_vertex {
            // A pinned register is not staged for a draw's vertex shader.
            self.win.word(&mut self.sel, 1.0f32.to_bits())
        } else {
            let reg = self.cp.alloc_reg();
            self.cp.uniforms.push((reg, 1.0f32.to_bits()));
            self.sel.ubo.insert(v, crate::Ubo::Fixed(reg));
            v
        };
        self.one_vreg = Some(v);
        v
    }

    /// A window with more words than it holds keeps the most read ones, the constant 1 and the
    /// pool's control word; the rest are fetched from the pool in memory.
    pub unsafe fn pool_window(&mut self) {
        if self.win.words.len() <= self.win.size || !(self.is_vertex || self.is_fragment) {
            return;
        }
        let (base, words) = (self.win.base, std::mem::take(&mut self.win.words));
        let mut uses: HashMap<u8, usize> = HashMap::new();
        for s in self.sel.prog.iter().flat_map(|i| i.srcs.iter()) {
            if let Some(&Ubo::Uniform(u)) = self.sel.ubo.get(s) {
                *uses.entry(u).or_insert(0) += 1;
            }
        }
        let one_old = words.iter().find(|w| w.1 == 1).map(|w| w.0);
        let mut others: Vec<(u8, u32)> = words.iter().copied().filter(|w| Some(w.0) != one_old).collect();
        others.sort_by_key(|&(u, _)| (std::cmp::Reverse(uses.get(&u).copied().unwrap_or(0)), u));
        let (keep, pooled) = others.split_at((self.win.size - 2).min(others.len()));
        let mut window: Vec<(Option<u8>, u32)> = keep.iter().map(|&(u, b)| (Some(u), b)).collect();
        window.sort();
        window.push((one_old, 1));
        window.push((None, (1 << 16) | (128 + self.win.slot)));
        let renum: HashMap<u8, u8> = window.iter().enumerate().filter_map(|(k, &(u, _))| Some((u?, base + k as u8))).collect();
        let tag: HashMap<u8, u8> = pooled.iter().enumerate().map(|(k, &(u, _))| (u, POOL_TAG + k as u8)).collect();
        for m in self.sel.ubo.values_mut() {
            if let Ubo::Uniform(u) = m {
                *u = renum.get(u).or(tag.get(u)).copied().unwrap_or(*u);
            }
        }
        let n = window.len() as u8;
        let ctl = self.sel.fresh();
        self.sel.ubo.insert(ctl, Ubo::Uniform(base + n - 1));
        let one = self.sel.fresh();
        self.sel.ubo.insert(one, Ubo::Uniform(base + n - 2));
        let f = self.sel.fresh();
        self.sel.ubo.insert(f, Ubo::Fixed(30));
        if self.is_fragment {
            self.per_pixel_fixed.insert(f);
        }
        let zero = self.sel.fresh();
        let is_pool = |v: &u32, ubo: &HashMap<u32, Ubo>| match ubo.get(v) {
            Some(&Ubo::Uniform(u)) if u >= POOL_TAG => Some((u - POOL_TAG) as u32),
            _ => None,
        };
        let bits: Vec<u32> = pooled.iter().map(|&(_, b)| b).collect();
        let pool = self.fetch_pool(&bits, is_pool, ctl, one, f, zero);
        assert!(pool.len() <= POOL_MAX, "borgc: a constant pool of {} words ({POOL_MAX} fit)", pool.len());
        self.win.words = window.iter().enumerate().map(|(k, &(_, b))| (base + k as u8, b)).collect();
        eprintln!("borgc: constant pool of {} word(s)", pool.len());
        self.pool = pool;
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
        if self.cs.pool.len() > CS_POOL_MAX {
            return Err(format!("a constant pool of {} words ({CS_POOL_MAX} fit)", self.cs.pool.len()));
        }
        eprintln!("borgc[compute]: constant pool of {} word(s)", self.cs.pool.len());
        Ok(())
    }

    /// Fetches the pooled words from memory four per TEX, again when a group goes unused for a while,
    /// and hoisted above the TEX results its first use waits for. Returns the pool in fetch order.
    fn fetch_pool(&mut self, words: &[u32], is_pool: impl Fn(&u32, &HashMap<u32, Ubo>) -> Option<u32>,
                  ctl: u32, one: u32, f: u32, zero: u32) -> Vec<u32> {
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            if let Some(u) = u {
                ubo.insert(v, u);
            }
            v
        };
        let mut first_use: Vec<Option<usize>> = vec![None; words.len()];
        for (at, i) in self.sel.prog.iter().enumerate() {
            for s in i.srcs.iter() {
                if let Some(k) = is_pool(s, &self.sel.ubo) {
                    first_use[k as usize].get_or_insert(at);
                }
            }
        }
        let mut order: Vec<usize> = (0..words.len()).filter(|&k| first_use[k].is_some()).collect();
        order.sort_by_key(|&k| first_use[k]);
        let mut slot_of: Vec<usize> = vec![0; words.len()];
        for (slot, &k) in order.iter().enumerate() {
            slot_of[k] = slot;
        }
        let pool: Vec<u32> = order.iter().map(|&k| words[k]).collect();
        const GAP: usize = 16;
        let ngroups = (pool.len() + 3) / 4;
        let mut last: Vec<Option<usize>> = vec![None; ngroups];
        let mut events: Vec<(usize, usize)> = Vec::new();   // (before instruction, group)
        for (at, i) in self.sel.prog.iter().enumerate() {
            for s in i.srcs.iter() {
                let Some(k) = is_pool(s, &self.sel.ubo) else { continue };
                let g = slot_of[k as usize] / 4;
                if last[g].map_or(true, |l| at - l > GAP) {
                    let mut p = at;
                    loop {
                        let Some(t) = (0..p).rev().find(|&t| self.sel.prog[t].mnem == "TEX") else { break };
                        let d = self.sel.prog[t].dst;
                        if self.sel.prog[p..].iter().any(|i| i.srcs.contains(&d)) { p = t; } else { break; }
                    }
                    events.push((p, g));
                }
                last[g] = Some(at);
            }
        }
        events.sort();
        let mut cur: Vec<[u32; 4]> = vec![[0; 4]; ngroups];
        let mut out: Vec<BorgInstr> = vec![BorgInstr { mnem: "ISUB", dst: zero, srcs: vec![f, f], swz: vec![0, 0] }];
        let mut e = 0;
        for (at, mut i) in std::mem::take(&mut self.sel.prog).into_iter().enumerate() {
            while e < events.len() && events[e].0 == at {
                let g = events[e].1;
                let mut idx = zero;
                for _ in 0..g {
                    let n = fresh(&mut self.sel.ubo, None);
                    out.push(BorgInstr { mnem: "IADD", dst: n, srcs: vec![idx, one], swz: vec![0, 0] });
                    idx = n;
                }
                let t = fresh(&mut self.sel.ubo, None);
                out.push(BorgInstr { mnem: "TEX", dst: t, srcs: vec![idx, zero, ctl], swz: vec![0, 0, 0] });
                self.tex_dsts.insert(t);
                for c in 0..4u8 {
                    let r = fresh(&mut self.sel.ubo, None);
                    out.push(BorgInstr { mnem: "IOR", dst: r, srcs: vec![t, t], swz: vec![c, c] });
                    cur[g][c as usize] = r;
                }
                e += 1;
            }
            for s in i.srcs.iter_mut() {
                if let Some(k) = is_pool(s, &self.sel.ubo) {
                    let slot = slot_of[k as usize];
                    *s = cur[slot / 4][slot % 4];
                }
            }
            out.push(i);
        }
        self.sel.prog = out;
        pool
    }
    pub fn split_window_reads(&mut self) {
        let is_win = |v: &u32, ubo: &HashMap<u32, Ubo>| matches!(ubo.get(v), Some(Ubo::Uniform(_)));
        if self.sel.prog.iter().any(|i| i.srcs.iter().filter(|v| is_win(v, &self.sel.ubo)).count() > 1) {
            let f = self.sel.next_vreg;
            let z = self.sel.next_vreg + 1;
            self.sel.next_vreg += 2;
            self.sel.ubo.insert(f, Ubo::Fixed(30));
            let mut out = vec![BorgInstr { mnem: "ISUB", dst: z, srcs: vec![f, f], swz: vec![0, 0] }];
            for mut ins in std::mem::take(&mut self.sel.prog) {
                let mut seen = false;
                for k in 0..ins.srcs.len() {
                    if !is_win(&ins.srcs[k], &self.sel.ubo) {
                        continue;
                    }
                    if !seen {
                        seen = true;
                        continue;
                    }
                    let t = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    out.push(BorgInstr { mnem: "IADD", dst: t, srcs: vec![ins.srcs[k], z], swz: vec![ins.swz[k], 0] });
                    ins.srcs[k] = t;
                    ins.swz[k] = 0;
                }
                out.push(ins);
            }
            self.sel.prog = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Flags;

    fn cx(vertex: bool) -> ShaderState {
        ShaderState::new(&[17, 18, 19], vertex, !vertex, Flags(0))
    }

    #[test]
    fn const_regs_run_out() {
        let mut c = cx(false);
        assert_eq!((c.cp.alloc_reg(), c.cp.alloc_reg(), c.cp.alloc_reg()), (17, 18, 19));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.cp.alloc_reg())).is_err());
    }

    #[test]
    fn vertex_window_words_start_at_u25_and_dedupe() {
        let mut c = cx(true);
        let a = c.konst(0x3f80_0000);
        assert_eq!(c.konst(0x3f80_0000), a);
        c.konst(0x4000_0000);
        assert_eq!(c.window(), [(25, 0x3f80_0000), (26, 0x4000_0000)]);
        assert!(matches!(c.sel.ubo[&a], crate::Ubo::Uniform(25)));
    }

    #[test]
    fn vertex_zero_is_r30_minus_r30_and_words_go_past_the_window() {
        let mut c = cx(true);
        let z = c.konst(0);
        assert_eq!((c.sel.prog[0].mnem, c.konst(0)), ("ISUB", z));
        for b in 1..=7 {
            c.konst(b);
        }
        assert_eq!(c.window().len(), 7);
    }

    #[test]
    fn fragment_window_numbers_from_u20_and_dedupes() {
        let mut c = cx(false);
        let (a, b) = (c.konst(5), c.konst(6));
        assert_eq!((c.konst(5), c.window()), (a, vec![(20, 5), (21, 6)]));
        assert!(matches!((c.sel.ubo[&a], c.sel.ubo[&b]), (crate::Ubo::Uniform(20), crate::Ubo::Uniform(21))));
    }

    #[test]
    fn fragment_window_overflow_panics() {
        let mut c = cx(false);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for b in 0..=Window::fragment().max() as u32 {
                c.konst(b);
            }
        }));
        assert!(r.is_err());
    }
}
