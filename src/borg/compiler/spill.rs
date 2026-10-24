// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use std::collections::HashMap;
use crate::state::{ShaderState, Stage};
use crate::{BorgInstr, Loc, NO_DST};
use crate::opcodes::Op;
use crate::Mnem;

const SCRATCH_WORDS: u32 = 16384 * 34;
const MAX_SPILL_SLOTS: u32 = 16;
const MAX_COMPUTE_SPILL_SLOTS: u32 = 64;
// The vertex lane key's two words, after the slots; the word after the second takes the other lanes' store.
const KEY_WORDS: u32 = SCRATCH_WORDS + 4 * MAX_SPILL_SLOTS;

impl ShaderState {
    fn op(&mut self, at: &mut Vec<BorgInstr>, op: Op, srcs: &[u32]) -> u32 {
        let dst = self.sel.fresh();
        at.push(BorgInstr { mnem: op.into(), dst, srcs: srcs.to_vec(), swz: vec![0; srcs.len()], imm: 0 });
        self.spill.exempt.insert(dst);
        dst
    }

    /// A per-lane word in 0..3. Memory ops run lane by lane, so the last lane's VertexIndex wins a
    /// shared word: lanes holding it get 0, the rest repeat that on a second word. Lanes with equal
    /// VertexIndex share a key, and their values are equal too.
    fn vertex_lane_key(&mut self) -> Vec<BorgInstr> {
        let mut out = Vec::new();
        let vi = self.sel.fresh();
        self.sel.loc.insert(vi, Loc::Fixed(30));
        let (k0, k1, c1, c2) = (self.konst(KEY_WORDS), self.konst(KEY_WORDS + 2), self.konst(1), self.konst(2));
        let z = self.op(&mut out, Op::Isub, &[vi, vi]);
        let a0 = self.op(&mut out, Op::Iadd, &[z, k0]);
        out.push(BorgInstr { mnem: Op::Store.into(), dst: NO_DST, srcs: vec![a0, vi], swz: vec![0, 0], imm: 0 });
        let t0 = self.op(&mut out, Op::Load, &[a0]);
        let e0 = self.op(&mut out, Op::Iseq, &[vi, t0]);
        let a1 = self.op(&mut out, Op::Iadd, &[e0, k1]);
        out.push(BorgInstr { mnem: Op::Store.into(), dst: NO_DST, srcs: vec![a1, vi], swz: vec![0, 0], imm: 0 });
        let a2 = self.op(&mut out, Op::Iadd, &[z, k1]);
        let t1 = self.op(&mut out, Op::Load, &[a2]);
        let e1 = self.op(&mut out, Op::Iseq, &[vi, t1]);
        let n0 = self.op(&mut out, Op::Isub, &[c1, e0]);
        let m = self.op(&mut out, Op::Isub, &[c2, e1]);
        let key = self.op(&mut out, Op::Imul, &[n0, m]);
        self.spill.lane = Some(key);
        out
    }

    /// A per-lane word in 0..4, unique within a quad; in compute the LocalInvocationIndex.
    fn lane_key(&mut self) -> Vec<BorgInstr> {
        match self.stage {
            Stage::Compute => {
                let r30 = self.sel.fresh();
                self.sel.loc.insert(r30, Loc::Fixed(30));
                self.spill.lane = Some(r30);
                return Vec::new();
            }
            Stage::Vertex => return self.vertex_lane_key(),
            _ => {}
        }
        let mut out = Vec::new();
        let (f30, f31) = (self.sel.fresh(), self.sel.fresh());
        for (f, r) in [(f30, 30), (f31, 31)] {
            self.sel.loc.insert(f, Loc::Fixed(r));
            self.frag.per_pixel_fixed.insert(f);
        }
        let c1 = self.konst(1);
        let xi = self.op(&mut out, Op::F2i, &[f30]);
        let yi = self.op(&mut out, Op::F2i, &[f31]);
        let xb = self.op(&mut out, Op::Iand, &[xi, c1]);
        let yb = self.op(&mut out, Op::Iand, &[yi, c1]);
        let y2 = self.op(&mut out, Op::Ishl, &[yb, c1]);
        let key = self.op(&mut out, Op::Ior, &[xb, y2]);
        self.spill.lane = Some(key);
        out
    }

    /// Helper lanes skip STOREs, so their spilled values would come back wrong: derivatives and implicit LOD need them.
    fn needs_helper_lanes(&self, i: &BorgInstr) -> bool {
        match (self.stage, i.mnem) {
            (Stage::Compute, _) => false,
            (Stage::Vertex, m) => matches!(m, Mnem::Op(Op::Brz | Op::Brnz)),
            // TEX by address: its operation is in the TEXS before it, so assume an implicit LOD.
            (_, Mnem::Op(Op::Ddx | Op::Ddy | Op::Texa | Op::Texad | Op::Texd | Op::Brz | Op::Brnz | Op::Bne | Op::Beq)) => true,
            (_, Mnem::Op(Op::Tex)) => {
                let op = i.srcs.get(2).and_then(|c| match self.sel.loc.get(c) {
                    Some(Loc::Uniform(_)) => self.bits_of(*c).map(|bits| (bits >> 16) & 3),
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
        let pinned = |v: &u32| self.sel.loc.contains_key(v);
        let vals: Vec<(u32, usize, usize)> = def_at.iter().filter(|(v, _)| !pinned(v)).map(|(&v, &d)| (v, d, last[&v])).collect();
        let at = (0..self.sel.prog.len()).find(|&p| vals.iter().filter(|&&(_, d, l)| d <= p && p <= l).count() > limit)?;
        Some(vals.iter().filter(|&&(_, d, l)| d < at && l > at).map(|&(v, _, l)| (v, l)).collect())
    }

    pub fn spill_to_pressure(&mut self, limit: usize) {
        let max = if self.stage == Stage::Compute { MAX_COMPUTE_SPILL_SLOTS } else { MAX_SPILL_SLOTS };
        while self.spill.slots < max {
            let Some(live) = self.over_limit(limit) else { return };
            if !self.spill_one(&live) {
                return;
            }
        }
    }

    /// Spills the live value with the furthest next use; false when nothing can be spilled.
    fn spill_one(&mut self, live: &[(u32, usize)]) -> bool {
        let room = match self.stage {
            Stage::Fragment | Stage::Compute => true,
            Stage::Vertex => self.win.room(5),
            Stage::Other => false,
        };
        if !room || self.sel.prog.iter().any(|i| self.needs_helper_lanes(i)) {
            return false;
        }
        let pinned: Vec<u32> = self.out_roots.iter().copied().chain(self.tex_dsts.iter().copied()).chain(self.fattr_dsts.iter().copied()).chain(self.frag.out.iter().flatten().copied()).collect();
        let single_def = |prog: &[BorgInstr], v: u32| {
            prog.iter().filter(|i| i.dst == v).count() == 1
                && prog.iter().all(|i| i.srcs.iter().zip(&i.swz).all(|(s, &c)| *s != v || c == 0))
        };
        let Some(&(v, _)) = live
            .iter()
            .filter(|(v, _)| !self.spill.exempt.contains(v) && !pinned.contains(v) && single_def(&self.sel.prog, *v))
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
            let bars: Vec<usize> = self.sel.prog.iter().enumerate().filter(|(_, i)| i.mnem == Op::Barrier).map(|(at, _)| at).collect();
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
                .filter(|(v, &d)| !self.sel.loc.contains_key(v) && last.get(v).is_some_and(|&l| bars.iter().any(|&b| d < b && b < l)))
                .map(|(&v, _)| v)
                .min();
            let Some(v) = crossing else { return Ok(()) };
            let single = self.sel.prog.iter().filter(|i| i.dst == v).count() == 1
                && self.sel.prog.iter().all(|i| i.srcs.iter().zip(&i.swz).all(|(s, &c)| *s != v || c == 0));
            if !single || self.spill.slots >= MAX_COMPUTE_SPILL_SLOTS {
                return Err("a value live across barrier() cannot be spilled".into());
            }
            self.spill(v);
        }
    }

    fn spill(&mut self, v: u32) {
        // A draw shader's lane key includes the scratch base: slot + key is a LW/SW immediate.
        let slot_bits = if self.stage == Stage::Compute {
            crate::compute::SPILL_BASE_WORDS + self.spill.slots * self.invocations()
        } else {
            4 * self.spill.slots
        };
        let mut prog = std::mem::take(&mut self.sel.prog);
        if self.spill.lane.is_none() {
            let mut head = self.lane_key();
            if self.stage != Stage::Compute {
                let base = self.konst(SCRATCH_WORDS);
                let key = self.op(&mut head, Op::Iadd, &[self.spill.lane.unwrap(), base]);
                self.spill.lane = Some(key);
            }
            prog.splice(0..0, head);
        }
        let lane = self.spill.lane.unwrap();
        let k = self.konst(slot_bits);
        self.spill.slots += 1;
        let mut out = Vec::with_capacity(prog.len() + 8);
        for mut i in prog {
            if i.srcs.contains(&v) {
                let a = self.op(&mut out, Op::Iadd, &[lane, k]);
                let t = self.sel.fresh();
                self.spill.exempt.insert(t);
                out.push(BorgInstr { mnem: Op::Load.into(), dst: t, srcs: vec![a], swz: vec![0], imm: 0 });
                for s in i.srcs.iter_mut().filter(|s| **s == v) {
                    *s = t;
                }
            }
            let defines = i.dst == v;
            out.push(i);
            if defines {
                let a = self.op(&mut out, Op::Iadd, &[lane, k]);
                out.push(BorgInstr { mnem: Op::Store.into(), dst: NO_DST, srcs: vec![a, v], swz: vec![0, 0], imm: 0 });
            }
        }
        self.sel.prog = out;
    }
}
