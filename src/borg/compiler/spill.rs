// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::state::{ShaderState, Stage};
use crate::{BorgInstr, Loc, NO_DST};
use crate::opcodes::{Op, CSR_ROOT};
use std::collections::HashMap;

const SCRATCH_WORDS: u32 = 16384 * 34;
const MAX_SPILL_SLOTS: u32 = 16;
// A draw shader's late slots: 4 words each, by a 12-bit immediate from the base.
const MAX_LATE_SLOTS: u32 = 128;
const MAX_COMPUTE_SPILL_SLOTS: u32 = 64;
// The vertex lane key's two words, after the slots; the word after the second takes the other lanes' store.
const KEY_WORDS: u32 = SCRATCH_WORDS + 4 * MAX_LATE_SLOTS;

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

    /// The first position with more than `limit` live values, with the values that cross it.
    fn over_limit(&self, limit: usize) -> Option<Vec<(u32, usize)>> {
        let pinned = |v: &u32| self.sel.loc.contains_key(v);
        let ranges = crate::block::live_ranges(&self.sel.prog);
        let vals: Vec<(u32, usize, usize)> = ranges.iter().filter(|(v, _)| !pinned(v)).map(|(&v, &(d, l))| (v, d, l)).collect();
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
        if !room {
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
            let crossing = crate::block::live_ranges(&self.sel.prog)
                .iter()
                .filter(|(v, &(d, l))| !self.sel.loc.contains_key(v) && bars.iter().any(|&b| d < b && b < l))
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
                self.spill.addrs.insert(a);
                out.push(BorgInstr { mnem: Op::Store.into(), dst: NO_DST, srcs: vec![a, v], swz: vec![0, 0], imm: 0 });
            }
        }
        self.sel.prog = out;
    }

    /// Once the passes ran: while more values are live than the free registers, the one with the
    /// furthest next use leaves its register (`spill_late`). A forced value takes no free register,
    /// unless its register is `soft` (a draw shader's TEX block, 4 while live).
    pub(crate) fn spill_to_fit(&mut self, forced: &HashMap<u32, u8>, reserved: &[u8], soft: &[u8], margin: usize) -> Result<(), String> {
        let hard: Vec<u8> = forced.values().copied().filter(|r| !soft.contains(r)).chain(reserved.iter().copied()).collect();
        let free = (1..30u8).filter(|r| !hard.contains(r)).count().saturating_sub(margin);
        let max = if self.stage == Stage::Compute { MAX_COMPUTE_SPILL_SLOTS } else { MAX_LATE_SLOTS };
        loop {
            let ranges = crate::block::live_ranges(&self.sel.prog);
            let weight = |v: &u32| match forced.get(v) {
                Some(r) if soft.contains(r) => 4,
                Some(_) => 0,
                None => !self.sel.loc.contains_key(v) as usize,
            };
            let vals: Vec<(u32, usize, usize)> = ranges.iter().filter(|(v, _)| weight(v) > 0).map(|(&v, &(d, l))| (v, d, l)).collect();
            let mut live = vec![0usize; self.sel.prog.len() + 1];
            for &(v, d, l) in &vals {
                live[d..=l].iter_mut().for_each(|n| *n += weight(&v));
            }
            let Some(at) = live.iter().position(|&n| n > free) else { return Ok(()) };
            if self.spill.base.is_none() {
                self.late_head();
                continue;
            }
            let scalar = |v: u32| self.sel.prog.iter().all(|i| i.srcs.iter().zip(&i.swz).all(|(s, &c)| *s != v || c == 0));
            let nu = crate::block::NextUse::new(&self.sel.prog);
            let pick = vals
                .iter()
                .filter(|&&(v, d, l)| {
                    d < at && l > at && weight(&v) == 1 && (!self.spill.exempt.contains(&v) || self.spill.held.contains_key(&v))
                        && !self.out_roots.contains(&v) && scalar(v)
                })
                .map(|&(v, ..)| (v, nu.after(&self.sel.prog, at, v)))
                // One read by the next instruction would be back in a register at once.
                .filter(|&(_, n)| n > 1)
                .max_by_key(|&(v, n)| (n, std::cmp::Reverse(v)))
                .map(|(v, _)| v);
            match pick {
                Some(v) if self.spill.held.contains_key(&v) => self.reload_later(v, at),
                Some(v) if self.spill.slots < max => self.spill_late(v, nu.split(at, v)),
                _ => return Err(crate::compute::OUT_OF_REGISTERS.into()),
            }
        }
    }

    /// The byte address of the lane's word of slot 0, in `spill`'s layout: compute's from
    /// LocalInvocationIndex, a fragment's from its place in the quad, a vertex's from the lane key.
    fn late_base(&mut self, base: u32) -> Vec<BorgInstr> {
        let fixed = |cx: &mut Self, r: u8| {
            let f = cx.sel.fresh();
            cx.sel.loc.insert(f, Loc::Fixed(r));
            f
        };
        let i = |m: Op, dst: u32, srcs: Vec<u32>, imm: u32| BorgInstr { mnem: m.into(), dst, swz: vec![0; srcs.len()], srcs, imm };
        let mut head = Vec::new();
        let (lane, bytes) = if self.stage == Stage::Compute {
            let (r30, t) = (fixed(self, 30), self.sel.fresh());
            head.push(i(Op::Slli, t, vec![r30], 2));
            (t, 4 * crate::compute::SPILL_BASE_WORDS)
        } else if self.stage == Stage::Vertex {
            // vertex_lane_key in RISC-V forms.
            let vi = fixed(self, 30);
            let [table, root, a0, t0, d0, e0, s0, a1, t1, d1, e1, n0, n1, k1, key, kb, t] = [(); 17].map(|_| self.sel.fresh());
            head.extend([i(Op::Csrrs, table, vec![], CSR_ROOT), i(Op::Lw, root, vec![table], 0)]);
            let kw = self.materialize(&mut head, 4 * KEY_WORDS);
            head.extend([i(Op::Add, a0, vec![root, kw], 0), i(Op::Sw, NO_DST, vec![a0, vi], 0),
                i(Op::Lw, t0, vec![a0], 0), i(Op::Sub, d0, vec![vi, t0], 0), i(Op::Sltiu, e0, vec![d0], 1),
                i(Op::Slli, s0, vec![e0], 2), i(Op::Add, a1, vec![a0, s0], 0), i(Op::Sw, NO_DST, vec![a1, vi], 8),
                i(Op::Lw, t1, vec![a0], 8), i(Op::Sub, d1, vec![vi, t1], 0), i(Op::Sltiu, e1, vec![d1], 1),
                i(Op::Xori, n0, vec![e0], 1), i(Op::Xori, n1, vec![e1], 1), i(Op::And, k1, vec![n0, n1], 0),
                i(Op::Add, key, vec![n0, k1], 0), i(Op::Slli, kb, vec![key], 2), i(Op::Add, t, vec![kb, root], 0)]);
            (t, 4 * SCRATCH_WORDS)
        } else {
            let (r30, r31) = (fixed(self, 30), fixed(self, 31));
            self.frag.per_pixel_fixed.extend([r30, r31]);
            let [xi, yi, xb, yb, y2, key, kb, table, root, t] = [(); 10].map(|_| self.sel.fresh());
            head.extend([i(Op::F2i, xi, vec![r30], 0), i(Op::F2i, yi, vec![r31], 0), i(Op::Andi, xb, vec![xi], 1),
                i(Op::Andi, yb, vec![yi], 1), i(Op::Slli, y2, vec![yb], 1), i(Op::Ior, key, vec![xb, y2], 0),
                i(Op::Slli, kb, vec![key], 2), i(Op::Csrrs, table, vec![], CSR_ROOT), i(Op::Lw, root, vec![table], 0),
                i(Op::Add, t, vec![kb, root], 0)]);
            (t, 4 * SCRATCH_WORDS)
        };
        let k = self.materialize(&mut head, bytes);
        head.push(i(Op::Add, base, vec![lane, k], 0));
        self.spill.exempt.extend(head.iter().map(|i| i.dst));
        head
    }

    /// The slots' base at the start, and in compute again after each BARRIER, which leaves no
    /// register to the next quad.
    fn late_head(&mut self) {
        let base = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        self.spill.base = Some(base);
        let head = self.late_base(base);
        let mut prog = Vec::with_capacity(self.sel.prog.len() + 8);
        prog.extend(head.iter().cloned());
        for i in std::mem::take(&mut self.sel.prog) {
            let bar = i.mnem == Op::Barrier;
            prog.push(i);
            if bar {
                prog.extend(head.iter().cloned());
            }
        }
        self.sel.prog = prog;
    }

    /// A slot's address: the base and an immediate, or past the immediate's reach a register.
    fn slot_addr(&mut self, out: &mut Vec<BorgInstr>, slot: u32) -> (u32, u32) {
        let base = self.spill.base.unwrap();
        if slot < 2048 {
            return (base, slot);
        }
        let k = self.materialize(out, slot);
        let a = self.op(out, Op::Add, &[base, k]);
        self.spill.exempt.insert(k);
        (a, 0)
    }

    fn reload(&mut self, out: &mut Vec<BorgInstr>, slot: u32) -> u32 {
        let (a, o) = self.slot_addr(out, slot);
        let t = self.sel.fresh();
        self.spill.exempt.insert(t);
        self.spill.held.insert(t, slot);
        out.push(BorgInstr { mnem: Op::Lw.into(), dst: t, srcs: vec![a], swz: vec![0], imm: o });
        t
    }

    /// `t`, which its slot holds too, leaves its register after `at`: its next read loads it again.
    fn reload_later(&mut self, t: u32, at: usize) {
        let slot = self.spill.held[&t];
        let mut out = Vec::with_capacity(self.sel.prog.len() + 2);
        let mut again = None;
        for (k, mut i) in std::mem::take(&mut self.sel.prog).into_iter().enumerate() {
            if k > at && i.srcs.contains(&t) {
                let r = match again {
                    Some(r) => r,
                    None => self.reload(&mut out, slot),
                };
                again = Some(r);
                i.srcs.iter_mut().filter(|s| **s == t).for_each(|s| *s = r);
            }
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// `v` in a slot as `spill` lays it out, stored after each definition. From `from` on, a block
    /// that reads it loads it once; a definition there gets a register of its own.
    fn spill_late(&mut self, v: u32, from: usize) {
        let slot = 4 * self.spill.slots * if self.stage == Stage::Compute { self.invocations() } else { 4 };
        let store = if self.stage == Stage::Compute { Op::Sw } else { Op::Sws };
        self.spill.slots += 1;
        let mut out = Vec::with_capacity(self.sel.prog.len() + 8);
        let addr = |cx: &mut Self, out: &mut Vec<BorgInstr>| cx.slot_addr(out, slot);
        let mut held = None;
        for (k, mut i) in std::mem::take(&mut self.sel.prog).into_iter().enumerate() {
            if crate::block::starts_block(i.mnem) {
                held = None;
            }
            let after = k >= from;
            if after && i.srcs.contains(&v) {
                let t = match held {
                    Some(t) => t,
                    None => self.reload(&mut out, slot),
                };
                held = Some(t);
                i.srcs.iter_mut().filter(|s| **s == v).for_each(|s| *s = t);
            }
            let defined = (i.dst == v).then(|| {
                if after {
                    i.dst = self.sel.fresh();
                    self.spill.exempt.insert(i.dst);
                    self.spill.held.insert(i.dst, slot);
                }
                i.dst
            });
            let ends = crate::block::ends_block(i.mnem);
            out.push(i);
            if let Some(d) = defined {
                let (a, o) = addr(self, &mut out);
                out.push(BorgInstr { mnem: store.into(), dst: NO_DST, srcs: vec![a, d], swz: vec![0, 0], imm: o });
                held = after.then_some(d);
            }
            if ends {
                held = None;
            }
        }
        self.sel.prog = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Flags;

    #[test]
    fn spill_to_fit_leaves_no_more_live_values_than_registers() {
        let mut c = ShaderState::new(&[17, 18, 19], Stage::Compute, Flags(0));
        let (f, vals): (u32, Vec<u32>) = (c.sel.fresh(), (0..6).map(|_| c.sel.fresh()).collect());
        c.sel.loc.insert(f, Loc::Fixed(30));
        let mut prog: Vec<BorgInstr> = vals.iter().map(|&v| BorgInstr { mnem: Op::Addi.into(), dst: v, srcs: vec![f], swz: vec![0], imm: v }).collect();
        let sum = c.sel.fresh();
        prog.extend(vals.iter().map(|&v| BorgInstr { mnem: Op::Add.into(), dst: sum, srcs: vec![sum, v], swz: vec![0, 0], imm: 0 }));
        c.sel.prog = prog;
        // 4 free registers: r1..r4.
        let reserved: Vec<u8> = (5..30).collect();
        c.spill_to_fit(&HashMap::new(), &reserved, &[], 0).unwrap();
        let ranges = crate::block::live_ranges(&c.sel.prog);
        let peak = (0..c.sel.prog.len())
            .map(|p| ranges.iter().filter(|(v, &(d, l))| !c.sel.loc.contains_key(v) && d <= p && p <= l).count())
            .max();
        assert!(peak <= Some(4), "{peak:?}");
        let (lw, sw) = (c.sel.prog.iter().filter(|i| i.mnem == Op::Lw).count(), c.sel.prog.iter().filter(|i| i.mnem == Op::Sw).count());
        assert_eq!((lw, sw), (c.spill.slots as usize, c.spill.slots as usize));
        assert!(c.spill.slots > 0);
    }
}
