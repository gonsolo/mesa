// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use std::collections::HashMap;
use crate::state::ShaderState;
use crate::{BorgInstr, Ubo, NO_DST};

const SCRATCH_WORDS: u32 = 16384 * 34;
const MAX_SPILL_SLOTS: u32 = 16;
const MAX_COMPUTE_SPILL_SLOTS: u32 = 64;
// The vertex lane key's two words, after the slots; the word after the second takes the other lanes' store.
const KEY_WORDS: u32 = SCRATCH_WORDS + 4 * MAX_SPILL_SLOTS;

impl ShaderState {
    fn op(&mut self, at: &mut Vec<BorgInstr>, mnem: &'static str, srcs: &[u32]) -> u32 {
        let dst = self.sel.fresh();
        at.push(BorgInstr { mnem, dst, srcs: srcs.to_vec(), swz: vec![0; srcs.len()] });
        self.no_spill.insert(dst);
        dst
    }

    /// A per-lane word in 0..3. Memory ops run lane by lane, so the last lane's VertexIndex wins a
    /// shared word: lanes holding it get 0, the rest repeat that on a second word. Lanes with equal
    /// VertexIndex share a key, and their values are equal too.
    fn vertex_lane_key(&mut self) -> Vec<BorgInstr> {
        let mut out = Vec::new();
        let vi = self.sel.fresh();
        self.sel.ubo.insert(vi, Ubo::Fixed(30));
        let (k0, k1, c1, c2) = (self.konst(KEY_WORDS), self.konst(KEY_WORDS + 2), self.konst(1), self.konst(2));
        let z = self.op(&mut out, "ISUB", &[vi, vi]);
        let a0 = self.op(&mut out, "IADD", &[z, k0]);
        out.push(BorgInstr { mnem: "STORE", dst: NO_DST, srcs: vec![a0, vi], swz: vec![0, 0] });
        let t0 = self.op(&mut out, "LOAD", &[a0]);
        let e0 = self.op(&mut out, "ISEQ", &[vi, t0]);
        let a1 = self.op(&mut out, "IADD", &[e0, k1]);
        out.push(BorgInstr { mnem: "STORE", dst: NO_DST, srcs: vec![a1, vi], swz: vec![0, 0] });
        let a2 = self.op(&mut out, "IADD", &[z, k1]);
        let t1 = self.op(&mut out, "LOAD", &[a2]);
        let e1 = self.op(&mut out, "ISEQ", &[vi, t1]);
        let n0 = self.op(&mut out, "ISUB", &[c1, e0]);
        let m = self.op(&mut out, "ISUB", &[c2, e1]);
        let key = self.op(&mut out, "IMUL", &[n0, m]);
        self.spill_lane = Some(key);
        out
    }

    /// A per-lane word in 0..4, unique within a quad; in compute the LocalInvocationIndex.
    fn lane_key(&mut self) -> Vec<BorgInstr> {
        if self.is_compute {
            let r30 = self.sel.fresh();
            self.sel.ubo.insert(r30, Ubo::Fixed(30));
            self.spill_lane = Some(r30);
            return Vec::new();
        }
        if self.is_vertex {
            return self.vertex_lane_key();
        }
        let mut out = Vec::new();
        let (f30, f31) = (self.sel.fresh(), self.sel.fresh());
        for (f, r) in [(f30, 30), (f31, 31)] {
            self.sel.ubo.insert(f, Ubo::Fixed(r));
            self.per_pixel_fixed.insert(f);
        }
        let c1 = self.konst(1);
        let xi = self.op(&mut out, "F2I", &[f30]);
        let yi = self.op(&mut out, "F2I", &[f31]);
        let xb = self.op(&mut out, "IAND", &[xi, c1]);
        let yb = self.op(&mut out, "IAND", &[yi, c1]);
        let y2 = self.op(&mut out, "ISHL", &[yb, c1]);
        let key = self.op(&mut out, "IOR", &[xb, y2]);
        self.spill_lane = Some(key);
        out
    }

    /// Helper lanes skip STOREs, so their spilled values would come back wrong: derivatives and implicit LOD need them.
    fn needs_helper_lanes(&self, i: &BorgInstr) -> bool {
        if self.is_compute {
            return false;
        }
        if self.is_vertex {
            return matches!(i.mnem, "BRZ" | "BRNZ");
        }
        match i.mnem {
            "DDX" | "DDY" | "TEXA" | "BRZ" | "BRNZ" => true,
            "TEX" => {
                let op = i.srcs.get(2).and_then(|c| match self.sel.ubo.get(c) {
                    Some(Ubo::Uniform(_)) => self.bits_of(*c).map(|bits| (bits >> 16) & 3),
                    _ => None,
                });
                !matches!(op, Some(1 | 3))
            }
            _ => false,
        }
    }

    /// The first position with more than `limit` live values, with the values that cross it.
    fn over_limit(&self, limit: usize) -> Option<Vec<(u32, usize)>> {
        let mut def_at: HashMap<u32, usize> = HashMap::new();
        let mut last: HashMap<u32, usize> = HashMap::new();
        for (at, i) in self.sel.prog.iter().enumerate() {
            if i.dst != NO_DST {
                def_at.entry(i.dst).or_insert(at);
                last.insert(i.dst, at);
            }
            for s in &i.srcs {
                last.insert(*s, at);
            }
        }
        crate::regalloc::extend_for_loops(&self.sel.prog, &mut def_at, &mut last);
        let pinned = |v: &u32| self.sel.ubo.contains_key(v);
        let vals: Vec<(u32, usize, usize)> = def_at.iter().filter(|(v, _)| !pinned(v)).map(|(&v, &d)| (v, d, last[&v])).collect();
        let at = (0..self.sel.prog.len()).find(|&p| vals.iter().filter(|&&(_, d, l)| d <= p && p <= l).count() > limit)?;
        Some(vals.iter().filter(|&&(_, d, l)| d < at && l > at).map(|&(v, _, l)| (v, l)).collect())
    }

    pub fn spill_to_pressure(&mut self, limit: usize) {
        let max = if self.is_compute { MAX_COMPUTE_SPILL_SLOTS } else { MAX_SPILL_SLOTS };
        while self.spill_slots < max {
            let Some(live) = self.over_limit(limit) else { return };
            if !self.spill_one(&live) {
                return;
            }
        }
    }

    /// Spills the live value with the furthest next use; false when nothing can be spilled.
    fn spill_one(&mut self, live: &[(u32, usize)]) -> bool {
        let vs_room = self.is_vertex && self.win.room(5);
        if !(self.is_fragment || self.is_compute || self.is_vertex && vs_room) || self.sel.prog.iter().any(|i| self.needs_helper_lanes(i)) {
            return false;
        }
        let pinned: Vec<u32> = self.out_roots.iter().copied().chain(self.tex_dsts.iter().copied()).chain(self.fattr_dsts.iter().copied()).chain(self.frag_z).chain(self.pos_out.iter().flatten().copied()).chain(self.frag_out.iter().flatten().copied()).collect();
        let single_def = |prog: &[BorgInstr], v: u32| {
            prog.iter().filter(|i| i.dst == v).count() == 1
                && prog.iter().all(|i| i.srcs.iter().zip(&i.swz).all(|(s, &c)| *s != v || c == 0))
        };
        let Some(&(v, _)) = live
            .iter()
            .filter(|(v, _)| !self.no_spill.contains(v) && !pinned.contains(v) && single_def(&self.sel.prog, *v))
            .max_by_key(|&&(v, end)| (end, std::cmp::Reverse(v)))
        else {
            return false;
        };
        self.spill(v);
        true
    }

    /// Every quad of a workgroup runs up to a BARRIER on the same registers before any goes on,
    /// so a value live across one is kept in memory.
    pub fn spill_across_barriers(&mut self) -> Result<(), String> {
        loop {
            let bars: Vec<usize> = self.sel.prog.iter().enumerate().filter(|(_, i)| i.mnem == "BARRIER").map(|(at, _)| at).collect();
            let mut def_at: HashMap<u32, usize> = HashMap::new();
            let mut last: HashMap<u32, usize> = HashMap::new();
            for (at, i) in self.sel.prog.iter().enumerate() {
                if i.dst != NO_DST {
                    def_at.entry(i.dst).or_insert(at);
                }
                for s in &i.srcs {
                    last.insert(*s, at);
                }
            }
            crate::regalloc::extend_for_loops(&self.sel.prog, &mut def_at, &mut last);
            let crossing = def_at
                .iter()
                .filter(|(v, &d)| !self.sel.ubo.contains_key(v) && last.get(v).is_some_and(|&l| bars.iter().any(|&b| d < b && b < l)))
                .map(|(&v, _)| v)
                .min();
            let Some(v) = crossing else { return Ok(()) };
            let single = self.sel.prog.iter().filter(|i| i.dst == v).count() == 1
                && self.sel.prog.iter().all(|i| i.srcs.iter().zip(&i.swz).all(|(s, &c)| *s != v || c == 0));
            if !single || self.spill_slots >= MAX_COMPUTE_SPILL_SLOTS {
                return Err("a value live across barrier() cannot be spilled".into());
            }
            self.spill(v);
        }
    }

    fn spill(&mut self, v: u32) {
        let slot_bits = if self.is_compute {
            crate::compute::SPILL_BASE_WORDS + self.spill_slots * self.invocations()
        } else {
            SCRATCH_WORDS + 4 * self.spill_slots
        };
        let mut prog = std::mem::take(&mut self.sel.prog);
        if self.spill_lane.is_none() {
            let head = self.lane_key();
            prog.splice(0..0, head);
        }
        let lane = self.spill_lane.unwrap();
        let k = self.konst(slot_bits);
        self.spill_slots += 1;
        let mut out = Vec::with_capacity(prog.len() + 8);
        for mut i in prog {
            if i.srcs.contains(&v) {
                let a = self.op(&mut out, "IADD", &[lane, k]);
                let t = self.sel.fresh();
                self.no_spill.insert(t);
                out.push(BorgInstr { mnem: "LOAD", dst: t, srcs: vec![a], swz: vec![0] });
                for s in i.srcs.iter_mut().filter(|s| **s == v) {
                    *s = t;
                }
            }
            let defines = i.dst == v;
            out.push(i);
            if defines {
                let a = self.op(&mut out, "IADD", &[lane, k]);
                out.push(BorgInstr { mnem: "STORE", dst: NO_DST, srcs: vec![a, v], swz: vec![0, 0] });
            }
        }
        self.sel.prog = out;
    }
}
