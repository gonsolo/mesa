// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::state::{Selector, ShaderState};
use crate::{BorgInstr, Ubo, FRAG_POOL_SLOT};
use std::collections::HashMap;

/// A draw shader's uniform window: words u{base}.. in creation order, more than it holds until
/// the pool pass moves the rest to memory.
pub struct Window {
    base: u8,
    size: usize,
    max: usize,
    words: Vec<(u8, u32)>,
    vregs: HashMap<u32, u32>,
}

impl Window {
    pub fn fragment() -> Self {
        Window { base: 20, size: 11, max: crate::FRAG_CONST_MAX as usize, words: Vec::new(), vregs: HashMap::new() }
    }

    pub fn vertex() -> Self {
        Window { base: DRAW_VS_CONST_U0, size: DRAW_VS_CONST_WORDS, max: VS_VIRTUAL_MAX, words: Vec::new(), vregs: HashMap::new() }
    }

    pub fn word(&mut self, sel: &mut Selector, bits: u32) -> u32 {
        if let Some(&v) = self.vregs.get(&bits) {
            return v;
        }
        assert!(self.words.len() < self.max, "borgc: shader needs more than {} constants", self.max);
        let u = self.base + self.words.len() as u8;
        self.words.push((u, bits));
        let v = sel.fresh();
        sel.ubo.insert(v, Ubo::Uniform(u));
        self.vregs.insert(bits, v);
        v
    }

    /// Room for `n` more words before the pool pass.
    pub fn room(&self, n: usize) -> bool {
        self.words.len() + n < self.max
    }
}

pub const DRAW_VS_CONST_U0: u8 = 25;
pub const DRAW_VS_CONST_WORDS: usize = 6;
/// Window words a vertex shader may ask for before the pool pass sorts them out.
pub const VS_VIRTUAL_MAX: usize = DRAW_VS_CONST_WORDS + 64;

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

    /// Moves the window words past the window's end to the memory pool.
    pub unsafe fn pool_window(&mut self) {
        if self.win.words.len() > self.win.size {
            if self.is_vertex {
                self.pool_vs_consts();
            } else if self.is_fragment {
                self.pool_window_consts();
            }
        }
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

    unsafe fn pool_window_consts(&mut self) {
        let first_pool = 20 + 9;
        // The window keeps the nine constants read most often: a window word costs no register.
        {
            let mut uses: HashMap<u32, usize> = HashMap::new();
            for i in self.sel.prog.iter() {
                for s in i.srcs.iter() {
                    if let Some(&Ubo::Uniform(u)) = self.sel.ubo.get(s) {
                        if u >= 20 {
                            *uses.entry(u as u32).or_insert(0) += 1;
                        }
                    }
                }
            }
            let mut by_use: Vec<(u32, u32)> = self.win.words.iter().map(|&(u, b)| (u as u32, b)).collect();
            by_use.sort_by_key(|&(u, _)| (std::cmp::Reverse(uses.get(&u).copied().unwrap_or(0)), u));
            let renum: HashMap<u32, u32> = by_use.iter().enumerate().map(|(k, &(u, _))| (u, 20 + k as u32)).collect();
            for (_, m) in self.sel.ubo.iter_mut() {
                if let Ubo::Uniform(u) = m {
                    if let Some(&n) = renum.get(&(*u as u32)) {
                        *u = n as u8;
                    }
                }
            }
            self.win.words = by_use.iter().map(|&(u, bits)| (renum[&u] as u8, bits)).collect();
        }
        let mut pool: Vec<u32> = vec![0; (20 + self.win.words.len() as u32 - first_pool) as usize];
        for &(u, bits) in self.win.words.iter() {
            if u as u32 >= first_pool {
                pool[(u as u32 - first_pool) as usize] = bits;
            }
        }
        self.win.words.retain(|&(u, _)| (u as u32) < first_pool);
        self.win.words.push((first_pool as u8, (1 << 16) | (128 + FRAG_POOL_SLOT)));
        self.win.words.push((first_pool as u8 + 1, 1));
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            if let Some(u) = u {
                ubo.insert(v, u);
            }
            v
        };
        let ctl = fresh(&mut self.sel.ubo, Some(Ubo::Uniform(first_pool as u8)));
        let one = fresh(&mut self.sel.ubo, Some(Ubo::Uniform(first_pool as u8 + 1)));
        let f = fresh(&mut self.sel.ubo, Some(Ubo::Fixed(30)));
        self.per_pixel_fixed.insert(f);
        let zero = fresh(&mut self.sel.ubo, None);
        let is_pool = |v: &u32, ubo: &HashMap<u32, Ubo>| match ubo.get(v) {
            Some(&Ubo::Uniform(u)) if u as u32 >= first_pool && *v != ctl && *v != one => Some(u as u32 - first_pool),
            _ => None,
        };
        let pool = self.fetch_pool(&pool, is_pool, ctl, one, f, zero);
        eprintln!("borgc: constant pool of {} word(s)", pool.len());
        self.pool = pool;
    }
    /// A vertex shader with more constants than its window holds: the most used stay in the window, the rest
    /// are fetched from a pool in memory, four words per TEX (the way a fragment shader's are).
    unsafe fn pool_vs_consts(&mut self) {
        const POOL_TAG: u32 = 100;
        let first = DRAW_VS_CONST_U0 as u32;
        let mut uses: HashMap<u32, usize> = HashMap::new();
        for i in self.sel.prog.iter() {
            for s in i.srcs.iter() {
                if let Some(&Ubo::Uniform(u)) = self.sel.ubo.get(s) {
                    if u as u32 >= first {
                        *uses.entry(u as u32).or_insert(0) += 1;
                    }
                }
            }
        }
        let consts = self.win.words.clone();
        let one_old = consts.iter().find(|c| c.1 == 1).map(|c| c.0 as u32);
        let mut others: Vec<(u32, u32)> = consts.iter().filter(|c| Some(c.0 as u32) != one_old).map(|&(u, b)| (u as u32, b)).collect();
        others.sort_by_key(|&(u, _)| (std::cmp::Reverse(uses.get(&u).copied().unwrap_or(0)), u));
        let cap = DRAW_VS_CONST_WORDS - 2;   // the pool's control word and the constant 1
        let (keep, pooled) = others.split_at(cap.min(others.len()));
        let mut keep: Vec<(u32, u32)> = keep.to_vec();
        keep.sort();
        let mut window: Vec<(u32, u32)> = keep.clone();   // (old id, bits)
        if let Some(o) = one_old { window.push((o, 1)); } else { window.push((u32::MAX, 1)); }
        window.push((u32::MAX - 1, (1u32 << 16) | (128 + crate::VERT_POOL_SLOT)));
        let renum: HashMap<u32, u32> = window.iter().enumerate().map(|(k, &(u, _))| (u, first + k as u32)).collect();
        let pool_tag: HashMap<u32, u32> = pooled.iter().enumerate().map(|(k, &(u, _))| (u, POOL_TAG + k as u32)).collect();
        let pool_bits: Vec<u32> = pooled.iter().map(|&(_, b)| b).collect();
        for (_, m) in self.sel.ubo.iter_mut() {
            if let Ubo::Uniform(u) = m {
                if let Some(&n) = renum.get(&(*u as u32)) { *u = n as u8; }
                else if let Some(&t) = pool_tag.get(&(*u as u32)) { *u = t as u8; }
            }
        }
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            if let Some(u) = u { ubo.insert(v, u); }
            v
        };
        let n = window.len() as u32;
        let ctl = fresh(&mut self.sel.ubo, Some(Ubo::Uniform((first + n - 1) as u8)));
        let one = fresh(&mut self.sel.ubo, Some(Ubo::Uniform(renum[&window[window.len() - 2].0] as u8)));
        let f = fresh(&mut self.sel.ubo, Some(Ubo::Fixed(30)));
        let zero = fresh(&mut self.sel.ubo, None);
        let is_pool = |v: &u32, ubo: &HashMap<u32, Ubo>| match ubo.get(v) {
            Some(&Ubo::Uniform(u)) if u as u32 >= POOL_TAG => Some(u as u32 - POOL_TAG),
            _ => None,
        };
        let pool = self.fetch_pool(&pool_bits, is_pool, ctl, one, f, zero);
        assert!(pool.len() <= 64, "borgc: vertex shader needs a constant pool of {} words (64 fit)", pool.len());
        self.win.words = window.iter().map(|&(u, bits)| (renum[&u] as u8, bits)).collect();
        eprintln!("borgc: vertex constant pool of {} word(s)", pool.len());
        self.pool = pool;
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
        for b in 1..=DRAW_VS_CONST_WORDS as u32 + 1 {
            c.konst(b);
        }
        assert_eq!(c.window().len(), DRAW_VS_CONST_WORDS + 1);
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
            for b in 0..=crate::FRAG_CONST_MAX {
                c.konst(b);
            }
        }));
        assert!(r.is_err());
    }
}
