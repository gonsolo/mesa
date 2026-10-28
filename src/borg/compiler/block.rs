// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::Op;
use crate::{BorgInstr, Mnem, NO_DST};
use compiler::cfg::{CFGBuilder, CFG};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// What the quad runs in order under one exec mask: a block starts at a loop's label, a barrier or
/// a change of the mask, and ends after a branch.
pub(crate) type Block = Vec<BorgInstr>;

pub(crate) fn starts_block(m: Mnem) -> bool {
    matches!(m, Mnem::Label | Mnem::Op(Op::Expush | Op::Exelse | Op::Expop | Op::Barrier))
}

/// BRNZ is the only branch before emission: a loop's back edge.
pub(crate) fn ends_block(m: Mnem) -> bool {
    m == Op::Brnz
}

/// The blocks as ranges of positions.
fn spans(prog: &[BorgInstr]) -> Vec<Range<usize>> {
    let (mut out, mut start) = (Vec::new(), 0);
    for (k, i) in prog.iter().enumerate() {
        if starts_block(i.mnem) && k > start {
            out.push(start..k);
            start = k;
        }
        if ends_block(i.mnem) {
            out.push(start..k + 1);
            start = k + 1;
        }
    }
    if start < prog.len() {
        out.push(start..prog.len());
    }
    out
}

pub(crate) fn blocks(prog: Vec<BorgInstr>) -> Vec<Block> {
    let s = spans(&prog);
    let mut it = prog.into_iter();
    s.iter().map(|r| it.by_ref().take(r.len()).collect()).collect()
}

/// Each instruction, and whether it starts a block.
pub(crate) fn block_starts(prog: Vec<BorgInstr>) -> impl Iterator<Item = (bool, BorgInstr)> {
    blocks(prog).into_iter().flat_map(|b| b.into_iter().enumerate().map(|(k, i)| (k == 0, i)))
}

/// The blocks the quad can run next: the following one, and a branch's label.
fn linear_succs(prog: &[BorgInstr], spans: &[Range<usize>]) -> Vec<Vec<usize>> {
    let label = |l: u32| spans.iter().position(|r| prog[r.start].mnem == Mnem::Label && prog[r.start].imm == l);
    (0..spans.len())
        .map(|b| {
            let last = &prog[spans[b].end - 1];
            let mut s: Vec<usize> = Some(last).filter(|i| ends_block(i.mnem)).and_then(|i| label(i.imm)).into_iter().collect();
            if b + 1 < spans.len() {
                s.push(b + 1);
            }
            s
        })
        .collect()
}

/// How a lane goes from block to block: it takes one side of an if, a break leaves the loop and
/// a continue goes back to its head. A lane that has left is masked where it is skipped, so the
/// blocks after a REPUSH and at a loop's end only see lanes still in the loop.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) struct Cfg {
    cfg: CFG<usize>,
    node: Vec<Option<usize>>,
}

#[cfg_attr(not(test), expect(dead_code))]
impl Cfg {
    pub(crate) fn new(prog: &[BorgInstr]) -> Cfg {
        let spans = spans(prog);
        let mut b = CFGBuilder::<usize, usize, std::hash::RandomState>::new();
        for k in 0..spans.len().max(1) {
            b.add_node(k, k);
        }
        for (p, s) in logical_edges(prog, &spans) {
            b.add_edge(p, s);
        }
        let cfg = b.as_cfg(false);
        let mut node = vec![None; spans.len().max(1)];
        for (k, n) in cfg.iter().enumerate() {
            node[**n] = Some(k);
        }
        Cfg { cfg, node }
    }

    /// Whether every lane that runs block `b` ran block `a` before it. No lane runs a block
    /// without a path to it.
    pub(crate) fn dominates(&self, a: usize, b: usize) -> bool {
        match (self.node[a], self.node[b]) {
            (Some(x), Some(y)) => self.cfg.dominates(x, y),
            _ => a == b,
        }
    }
}

/// The edges of `Cfg`. A push no lane skips is a loop's (its body, its EXANY) or a REPUSH of the
/// loop's `go`: lanes it masks have left already.
fn logical_edges(prog: &[BorgInstr], spans: &[Range<usize>]) -> Vec<(usize, usize)> {
    // Per push: for an if, the blocks its lanes came from and the end of its then side.
    let mut ifs: Vec<Option<(Vec<usize>, Option<Vec<usize>>)>> = vec![];
    // Per loop: its head, its `go` and its breaks.
    let mut loops: Vec<(usize, u32, Vec<usize>)> = vec![];
    let (mut edges, mut live) = (vec![], vec![]);
    for (b, r) in spans.iter().enumerate() {
        let i = &prog[r.start];
        let mut from = std::mem::take(&mut live);
        match i.mnem {
            Mnem::Label => loops.push((b, NO_DST, vec![])),
            Mnem::Op(Op::Expush) => {
                let body = b > 0 && prog[spans[b - 1].start].mnem == Mnem::Label;
                if let Some(l) = loops.last_mut().filter(|_| body) {
                    l.1 = i.srcs[0];
                }
                let any = prog.get(r.start + 1).is_some_and(|n| n.mnem == Op::Exany);
                let repush = loops.last().is_some_and(|l| l.1 == i.srcs[0]);
                ifs.push((!body && !any && !repush).then(|| (from.clone(), None)));
            }
            Mnem::Op(Op::Exelse) => {
                if let Some(Some((entry, then))) = ifs.last_mut() {
                    *then = Some(std::mem::replace(&mut from, entry.clone()));
                }
            }
            Mnem::Op(Op::Expop) => {
                if let Some(Some((entry, then))) = ifs.pop() {
                    from.extend(then.unwrap_or(entry));
                }
            }
            _ => {}
        }
        edges.extend(from.into_iter().map(|p| (p, b)));
        live = vec![b];
        for i in &prog[r.clone()] {
            match (i.mnem, loops.last_mut()) {
                (Mnem::Break, Some(l)) => {
                    l.2.push(b);
                    live.clear();
                }
                (Mnem::Continue, Some(l)) => {
                    edges.push((b, l.0));
                    live.clear();
                }
                (Mnem::Op(Op::Brnz), Some(_)) => {
                    let (head, _, breaks) = loops.pop().unwrap();
                    edges.extend(live.iter().map(|&p| (p, head)));
                    // Without a break the loop never ends; its exit still has a block before it.
                    if !breaks.is_empty() {
                        live = breaks;
                    }
                }
                _ => {}
            }
        }
    }
    edges
}

/// The loops around each block, innermost last: a loop runs from its LABEL's block to its BRNZ's.
fn loops_around(prog: &[BorgInstr], spans: &[Range<usize>]) -> Vec<Vec<usize>> {
    let mut open = vec![];
    spans
        .iter()
        .enumerate()
        .map(|(b, r)| {
            if prog[r.start].mnem == Mnem::Label {
                open.push(b);
            }
            let around = open.clone();
            if ends_block(prog[r.end - 1].mnem) {
                open.pop();
            }
            around
        })
        .collect()
}

/// What an edge leaving a loop adds to a next-use distance: a use after the loop is far for a value
/// the loop does not read (Braun & Hack, CC 2009).
pub(crate) const LOOP_EXIT: u32 = 1 << 16;

/// Next-use distances (Braun & Hack, CC 2009) on the logical CFG: per block, each value's distance
/// from the block's start to its next read, in the block or through its nearest successor. A value
/// written before it is read is dead there.
pub(crate) struct NextUse {
    pub(crate) at: Vec<HashMap<u32, u32>>,
    spans: Vec<Range<usize>>,
    succs: Vec<Vec<usize>>,
    around: Vec<Vec<usize>>,
}

impl NextUse {
    pub(crate) fn new(prog: &[BorgInstr]) -> NextUse {
        let spans = spans(prog);
        let around = loops_around(prog, &spans);
        let mut succs = vec![vec![]; spans.len()];
        for (p, s) in logical_edges(prog, &spans) {
            succs[p].push(s);
        }
        // Per block: the values it reads first, at their distance, and those it writes first.
        let (mut first, mut killed) = (vec![HashMap::new(); spans.len()], vec![HashSet::new(); spans.len()]);
        for (b, r) in spans.iter().enumerate() {
            for (k, i) in prog[r.clone()].iter().enumerate() {
                for &v in &i.srcs {
                    if !killed[b].contains(&v) {
                        first[b].entry(v).or_insert(k as u32);
                    }
                }
                if i.dst != NO_DST && !first[b].contains_key(&i.dst) {
                    killed[b].insert(i.dst);
                }
            }
        }
        let mut n = NextUse { at: first.clone(), spans, succs, around };
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..n.spans.len()).rev() {
                let len = n.spans[b].len() as u32;
                let mut mine = std::mem::take(&mut n.at[b]);
                // Through itself a value comes no nearer.
                for &s in n.succs[b].iter().filter(|&&s| s != b) {
                    let exit = n.exit(b, s);
                    for (&v, &d) in &n.at[s] {
                        if killed[b].contains(&v) || first[b].contains_key(&v) {
                            continue;
                        }
                        let d = d.saturating_add(len + exit);
                        if mine.get(&v).is_none_or(|&x| d < x) {
                            mine.insert(v, d);
                            changed = true;
                        }
                    }
                }
                n.at[b] = mine;
            }
        }
        n
    }

    fn exit(&self, b: usize, s: usize) -> u32 {
        if self.around[b].iter().all(|l| self.around[s].contains(l)) { 0 } else { LOOP_EXIT }
    }

    fn block_of(&self, pos: usize) -> usize {
        self.spans.partition_point(|r| r.end <= pos)
    }

    /// From after position `pos`: the distance to `v`'s next read; MAX if written first or never.
    pub(crate) fn after(&self, prog: &[BorgInstr], pos: usize, v: u32) -> u32 {
        let b = self.block_of(pos);
        for k in pos + 1..self.spans[b].end {
            if prog[k].srcs.contains(&v) {
                return (k - pos) as u32;
            }
            if prog[k].dst == v {
                return u32::MAX;
            }
        }
        let rest = (self.spans[b].end - pos) as u32;
        self.succs[b]
            .iter()
            .filter_map(|&s| self.at[s].get(&v).map(|d| d.saturating_add(rest + self.exit(b, s))))
            .min()
            .unwrap_or(u32::MAX)
    }

    /// Where `v` leaves its register to relieve `pos`: the start of the outermost loop around `pos`
    /// it enters live, as a lane runs that loop's earlier blocks again; else `pos`.
    pub(crate) fn split(&self, pos: usize, v: u32) -> usize {
        let b = self.block_of(pos);
        self.around[b].iter().find(|&&l| self.at[l].contains_key(&v)).map_or(pos, |&l| self.spans[l].start)
    }
}

/// Where each value holds its register, as a range of positions: from its first definition to
/// where it is last live on the linear CFG. A value a loop defines and needs on its next trip or
/// after it holds it for the whole loop: a lane that left early keeps its last trip's value.
pub(crate) fn live_ranges(prog: &[BorgInstr]) -> HashMap<u32, (usize, usize)> {
    let mut range: HashMap<u32, (usize, usize)> = HashMap::new();
    for (k, i) in prog.iter().enumerate() {
        for s in &i.srcs {
            if let Some(r) = range.get_mut(s) {
                r.1 = k;
            }
        }
        if i.dst != NO_DST {
            range.entry(i.dst).and_modify(|r| r.1 = k).or_insert((k, k));
        }
    }
    let spans = spans(prog);
    let succs = linear_succs(prog, &spans);
    let loops: Vec<(usize, usize)> = (0..spans.len()).filter_map(|e| succs[e].iter().find(|&&l| l <= e).map(|&l| (l, e))).collect();
    // Without a back edge a value is live from its first definition to its last use.
    if loops.is_empty() {
        return range;
    }
    let id: HashMap<u32, usize> = range.keys().enumerate().map(|(k, &v)| (v, k)).collect();
    let words = id.len().div_ceil(64);
    let set = |b: &mut Vec<u64>, v: usize| b[v / 64] |= 1 << (v % 64);
    let has = |b: &[u64], v: usize| b[v / 64] >> (v % 64) & 1 != 0;
    let n = spans.len();
    let (mut uses, mut defs) = (vec![vec![0u64; words]; n], vec![vec![0u64; words]; n]);
    for (b, r) in spans.iter().enumerate() {
        for i in &prog[r.clone()] {
            for v in i.srcs.iter().filter_map(|s| id.get(s)) {
                if !has(&defs[b], *v) {
                    set(&mut uses[b], *v);
                }
            }
            if let Some(&v) = id.get(&i.dst) {
                set(&mut defs[b], v);
            }
        }
    }
    let (mut live_in, mut live_out) = (uses.clone(), vec![vec![0u64; words]; n]);
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut out = vec![0u64; words];
            for &s in &succs[b] {
                out.iter_mut().zip(&live_in[s]).for_each(|(o, x)| *o |= x);
            }
            for w in 0..words {
                let x = uses[b][w] | out[w] & !defs[b][w];
                changed |= x != live_in[b][w];
                live_in[b][w] = x;
            }
            live_out[b] = out;
        }
    }
    let value: Vec<u32> = {
        let mut v = vec![0; id.len()];
        id.iter().for_each(|(&x, &k)| v[k] = x);
        v
    };
    let bits = |b: &[u64]| -> Vec<u32> {
        let mut out = vec![];
        for (w, &x) in b.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                out.push(value[w * 64 + x.trailing_zeros() as usize]);
                x &= x - 1;
            }
        }
        out
    };
    for (b, r) in spans.iter().enumerate() {
        for v in bits(&live_out[b]) {
            let x = range.get_mut(&v).unwrap();
            x.1 = x.1.max(r.end - 1);
        }
    }
    for &(l, e) in &loops {
        let (head, end) = (spans[l].start, spans[e].end - 1);
        let mut through = live_in[l].clone();
        if let Some(a) = live_in.get(e + 1) {
            through.iter_mut().zip(a).for_each(|(t, x)| *t |= x);
        }
        for v in bits(&through) {
            let x = range.get_mut(&v).unwrap();
            if (head..=end).contains(&x.0) {
                *x = (x.0.min(head), x.1.max(end));
            }
        }
    }
    range
}

/// Parallel copies, which all read before any writes, in an order that keeps that: a copy whose
/// destination another still reads waits, and a cycle goes through a temporary.
pub(crate) fn sequence_copies(mut todo: Vec<(u32, (u32, u8))>, fresh: &mut dyn FnMut() -> u32) -> Vec<(u32, (u32, u8))> {
    todo.retain(|&(d, s)| s != (d, 0));
    let mut out = vec![];
    while !todo.is_empty() {
        match todo.iter().position(|&(d, _)| !todo.iter().any(|&(_, (v, _))| v == d)) {
            Some(k) => out.push(todo.remove(k)),
            None => {
                let (d, t) = (todo[0].0, fresh());
                out.push((t, (d, 0)));
                todo.iter_mut().filter(|x| x.1 .0 == d).for_each(|x| x.1 .0 = t);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(mnem: impl Into<Mnem>, dst: u32, imm: u32) -> BorgInstr {
        BorgInstr { mnem: mnem.into(), dst, srcs: vec![], swz: vec![], imm }
    }

    fn mnems(b: &Block) -> Vec<Mnem> {
        b.iter().map(|i| i.mnem).collect()
    }

    fn a_loop() -> Vec<BorgInstr> {
        vec![
            op(Op::Fadd, 1, 0),
            op(Mnem::Label, NO_DST, 3),
            op(Op::Expush, NO_DST, 0),
            op(Op::Fmul, 2, 0),
            op(Op::Expop, NO_DST, 0),
            op(Op::Exany, 4, 0),
            op(Op::Brnz, NO_DST, 3),
            op(Op::Fadd, 5, 0),
        ]
    }

    #[test]
    fn blocks_start_at_labels_and_mask_changes_and_end_after_branches() {
        let b = blocks(a_loop());
        assert_eq!(b.iter().map(mnems).collect::<Vec<_>>(), vec![
            vec![Mnem::Op(Op::Fadd)],
            vec![Mnem::Label],
            vec![Mnem::Op(Op::Expush), Mnem::Op(Op::Fmul)],
            vec![Mnem::Op(Op::Expop), Mnem::Op(Op::Exany), Mnem::Op(Op::Brnz)],
            vec![Mnem::Op(Op::Fadd)],
        ]);
        assert_eq!(b.concat().len(), a_loop().len());
    }
    #[test]
    fn loop_values_live_for_the_whole_loop() {
        let v = |mnem: Op, dst: u32, srcs: &[u32]| BorgInstr { mnem: mnem.into(), dst, srcs: srcs.to_vec(), swz: vec![0; srcs.len()], imm: 0 };
        // 1 = before; LABEL; 3 = 2 + 1 (2 carried: read, then written); 4 defined, used after.
        let prog = vec![
            v(Op::Ior, 1, &[9, 9]),
            v(Op::Ior, 2, &[9, 9]),
            op(Mnem::Label, NO_DST, 0),
            v(Op::Iadd, 3, &[2, 1]),
            v(Op::Ior, 2, &[3, 3]),
            v(Op::Ior, 4, &[3, 3]),
            v(Op::Iadd, 5, &[1, 1]),
            BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![5], swz: vec![0], imm: 0 },
            v(Op::Iadd, 6, &[4, 4]),
        ];
        let r = live_ranges(&prog);
        assert_eq!(r[&1], (0, 7), "live into the loop");
        assert_eq!(r[&2], (1, 7), "carried round it");
        assert_eq!(r[&4], (2, 8), "defined in it, live after it");
        assert_eq!(r[&3], (3, 5), "a value of one trip");
    }
    fn m(mnem: impl Into<Mnem>, dst: u32, srcs: &[u32]) -> BorgInstr {
        BorgInstr { mnem: mnem.into(), dst, srcs: srcs.to_vec(), swz: vec![0; srcs.len()], imm: 0 }
    }

    fn edges(prog: &[BorgInstr]) -> Vec<(usize, usize)> {
        let mut e = logical_edges(prog, &spans(prog));
        e.sort();
        e
    }

    /// A loop as selection marks it: alive 1, go 2, any 3; `body` between its push and its end.
    fn in_loop(body: Vec<BorgInstr>) -> Vec<BorgInstr> {
        let mut p = vec![m(Op::Ior, 1, &[9, 9]), m(Mnem::Label, NO_DST, &[]), m(Op::Ior, 2, &[1, 1]), m(Op::Expush, NO_DST, &[2])];
        p.extend(body);
        p.extend([
            m(Op::Expop, NO_DST, &[]),
            m(Op::Expush, NO_DST, &[1]),
            m(Op::Exany, 3, &[]),
            m(Op::Expop, NO_DST, &[]),
            m(Op::Brnz, NO_DST, &[3]),
            m(Op::Fadd, 8, &[]),
        ]);
        p
    }

    /// `if (c) { jump; }` and the REPUSH after it, with `b` under it.
    fn jump_then(jump: Mnem) -> Vec<BorgInstr> {
        vec![
            m(Op::Fadd, 4, &[]),
            m(Op::Expush, NO_DST, &[4]),
            m(Op::Ior, 2, &[9, 9]),
            m(jump, NO_DST, &[]),
            m(Op::Exelse, NO_DST, &[]),
            m(Op::Expop, NO_DST, &[]),
            m(Op::Expush, NO_DST, &[2]),
            m(Op::Fadd, 5, &[]),
            m(Op::Expop, NO_DST, &[]),
        ]
    }

    #[test]
    fn an_if_goes_to_either_side_and_joins() {
        let p = vec![
            m(Op::Fadd, 1, &[]),
            m(Op::Expush, NO_DST, &[1]),
            m(Op::Fadd, 2, &[]),
            m(Op::Exelse, NO_DST, &[]),
            m(Op::Fadd, 3, &[]),
            m(Op::Expop, NO_DST, &[]),
            m(Op::Fadd, 4, &[]),
        ];
        assert_eq!(edges(&p), vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
        let c = Cfg::new(&p);
        assert!(c.dominates(0, 3) && !c.dominates(1, 3) && !c.dominates(2, 3));
    }
    #[test]
    fn a_nested_if_joins_before_the_outer_else() {
        let p = vec![
            m(Op::Fadd, 1, &[]),
            m(Op::Expush, NO_DST, &[1]),
            m(Op::Expush, NO_DST, &[1]),
            m(Op::Fadd, 2, &[]),
            m(Op::Exelse, NO_DST, &[]),
            m(Op::Expop, NO_DST, &[]),
            m(Op::Exelse, NO_DST, &[]),
            m(Op::Fadd, 3, &[]),
            m(Op::Expop, NO_DST, &[]),
        ];
        assert_eq!(edges(&p), vec![(0, 1), (0, 5), (1, 2), (1, 3), (2, 4), (3, 4), (4, 6), (5, 6)]);
        let c = Cfg::new(&p);
        assert!(c.dominates(1, 4) && !c.dominates(2, 4) && !c.dominates(1, 6) && c.dominates(0, 6));
    }
    #[test]
    fn a_break_leaves_the_loop() {
        // 0 pre, 1 head, 2 body, 3 break, 4 else, 5 join, 6 REPUSH, 7 its pop, 8..10 loop end, 11 after.
        let p = in_loop(jump_then(Mnem::Break));
        assert_eq!(edges(&p), vec![(0, 1), (1, 2), (2, 3), (2, 4), (3, 11), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)]);
        let c = Cfg::new(&p);
        assert!(c.dominates(2, 11) && c.dominates(3, 11) && !c.dominates(6, 11) && c.dominates(6, 10));
    }
    #[test]
    fn a_continue_goes_back_to_the_head() {
        let p = in_loop(jump_then(Mnem::Continue));
        assert_eq!(edges(&p), vec![(0, 1), (1, 2), (2, 3), (2, 4), (3, 1), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1), (10, 11)]);
        let c = Cfg::new(&p);
        assert!(c.dominates(2, 10) && c.dominates(6, 11) && !c.dominates(3, 1) && c.dominates(1, 3));
    }
    #[test]
    fn next_uses_take_the_nearer_side_and_end_at_a_write() {
        // 1 from before; 0: 5 = ..; 1: then reads 1 at 2; 2: else writes 1 first; 3: join reads 1.
        let p = vec![
            m(Op::Fadd, 5, &[]),
            m(Op::Expush, NO_DST, &[9]),
            m(Op::Fadd, 2, &[]),
            m(Op::Fadd, 3, &[1]),
            m(Op::Exelse, NO_DST, &[]),
            m(Op::Fadd, 1, &[]),
            m(Op::Expop, NO_DST, &[]),
            m(Op::Fadd, 4, &[1]),
        ];
        let n = NextUse::new(&p).at;
        assert_eq!(n[1][&1], 2, "read in the then side");
        assert_eq!(n[0][&1], 1 + 2, "the nearer side: block 0 is one long");
        assert!(!n[2].contains_key(&1), "the else side writes it first");
        assert_eq!(n[3][&1], 1);
    }
    #[test]
    fn a_use_after_a_loop_is_far_from_inside_it() {
        // 1 read in the loop, 7 defined before it and read only after it.
        let mut p = in_loop(jump_then(Mnem::Break));
        p.insert(0, m(Op::Fadd, 7, &[]));
        p.push(m(Op::Fadd, 6, &[7]));
        let n = NextUse::new(&p).at;
        let after = n.len() - 1;
        assert!(n[after].contains_key(&7));
        assert!(n[1][&7] >= LOOP_EXIT, "from the head, 7 is only used after the loop");
        assert!(n[1][&1] < LOOP_EXIT, "1 is read in the loop");
    }
    #[test]
    fn a_value_carried_round_a_loop_is_near_at_its_end() {
        // 7 read, then written again, on every trip.
        let mut p = in_loop(vec![m(Op::Fadd, 6, &[7]), m(Op::Fadd, 7, &[6])]);
        p.insert(0, m(Op::Fadd, 7, &[]));
        let n = NextUse::new(&p).at;
        let spans = spans(&p);
        let end = (0..spans.len()).find(|&b| p[spans[b].end - 1].mnem == Op::Brnz).unwrap();
        assert!(n[end][&7] < LOOP_EXIT, "the back edge does not leave the loop");
    }
    #[test]
    fn a_split_in_a_loop_starts_at_the_head_for_a_value_it_enters_with() {
        // 7 from before the loop, read in its body; 6 defined and read within one trip.
        let mut p = in_loop(vec![m(Op::Fadd, 6, &[7]), m(Op::Fadd, 5, &[6]), m(Op::Fadd, 4, &[6])]);
        p.insert(0, m(Op::Fadd, 7, &[]));
        let n = NextUse::new(&p);
        let head = p.iter().position(|i| i.mnem == Mnem::Label).unwrap();
        let mid = p.iter().position(|i| i.dst == 5).unwrap();
        assert_eq!(n.split(mid, 7), head, "7 is read again on the next trip");
        assert_eq!(n.split(mid, 6), mid, "6 is not live at the head");
        assert_eq!(n.split(0, 7), 0, "outside any loop");
        assert_eq!(n.after(&p, mid, 6), 1);
    }
    #[test]
    fn two_loops_in_a_row_are_left_one_after_the_other() {
        let mut p = in_loop(vec![]);
        let second: Vec<BorgInstr> = in_loop(vec![]).into_iter().skip(1).collect();
        p.extend(second);
        p.push(m(Op::Fadd, 6, &[1]));
        let n = NextUse::new(&p).at;
        let spans = spans(&p);
        let around = loops_around(&p, &spans);
        let heads: Vec<usize> = (0..spans.len()).filter(|&b| p[spans[b].start].mnem == Mnem::Label).collect();
        assert_eq!(heads.len(), 2);
        assert!(around[heads[1]].len() == 1 && around[heads[1]] != around[heads[0]]);
        // 1 is read by each loop's `go`: near from both heads.
        assert!(n[heads[0]][&1] < LOOP_EXIT && n[heads[1]][&1] < LOOP_EXIT);
    }
    #[test]
    fn parallel_copies_keep_their_sources() {
        let mut n = 100;
        let mut fresh = || { n += 1; n };
        // 1 <- 2 and 2 <- 3: 1 first, while 2 still holds its old value.
        assert_eq!(sequence_copies(vec![(2, (3, 0)), (1, (2, 0))], &mut fresh), vec![(1, (2, 0)), (2, (3, 0))]);
        // A swap needs a temporary; a copy of itself goes.
        assert_eq!(
            sequence_copies(vec![(1, (2, 0)), (2, (1, 0)), (3, (3, 0))], &mut fresh),
            vec![(101, (1, 0)), (1, (2, 0)), (2, (101, 0))]
        );
    }
}
