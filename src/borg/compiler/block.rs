// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::Op;
use crate::{BorgInstr, Mnem, NO_DST};
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
}
