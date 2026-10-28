// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::Op;
use crate::{BorgInstr, Mnem, NO_DST};
use compiler::cfg::{CFGBuilder, CFG};
use std::collections::HashMap;
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
    for (v, x) in range.iter_mut() {
        let k = id[v];
        for (b, r) in spans.iter().enumerate() {
            if has(&live_out[b], k) {
                x.1 = x.1.max(r.end - 1);
            }
        }
        for &(l, e) in &loops {
            let (head, end) = (spans[l].start, spans[e].end - 1);
            if (head..=end).contains(&x.0) && (has(&live_in[l], k) || live_in.get(e + 1).is_some_and(|a| has(a, k))) {
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
