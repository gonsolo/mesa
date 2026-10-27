// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::opcodes::Op;
use crate::{BorgInstr, Mnem};

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

pub(crate) fn blocks(prog: Vec<BorgInstr>) -> Vec<Block> {
    let (mut out, mut cur) = (Vec::new(), Block::new());
    for i in prog {
        if starts_block(i.mnem) && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        let end = ends_block(i.mnem);
        cur.push(i);
        if end {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Each instruction, and whether it starts a block.
pub(crate) fn block_starts(prog: Vec<BorgInstr>) -> impl Iterator<Item = (bool, BorgInstr)> {
    blocks(prog).into_iter().flat_map(|b| b.into_iter().enumerate().map(|(k, i)| (k == 0, i)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NO_DST;

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
}
