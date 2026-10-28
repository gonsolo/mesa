// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use crate::encode::encode;
use crate::opcodes::Op;
use crate::state::ShaderState;
use crate::{BorgInstr, Loc, Mnem};
use std::collections::HashMap;

/// An instruction's first three operands as registers, the first window word read through funct3.
pub(crate) struct Operands {
    pub r: [u8; 3],
    pub f3: u32,
    windows: u32,
    missing: bool,
}

/// RISC-V forms with an immediate: no window operand, and r0 is x0 there.
pub(crate) fn imm_form(m: Mnem) -> Option<Op> {
    match m {
        Mnem::Op(op @ (Op::Lw | Op::Sw | Op::Sws | Op::Slli | Op::Srli | Op::Srai | Op::Addi | Op::Andi | Op::Xori | Op::Sltiu | Op::Lui | Op::Csrrs)) => Some(op),
        _ => None,
    }
}

pub(crate) fn operands(cx: &ShaderState, alloc: &HashMap<u32, u8>, i: &BorgInstr) -> Operands {
    let mut o = Operands { r: [0; 3], f3: 0, windows: 0, missing: false };
    for (k, (s, &c)) in i.srcs.iter().zip(i.swz.iter()).take(3).enumerate() {
        // TEX and FATTR results: component c is rd + c.
        if cx.tex_dsts.contains(s) || cx.fattr_dsts.contains(s) {
            o.r[k] = alloc.get(s).map_or(0, |&r| r + c);
            continue;
        }
        match cx.sel.loc.get(s) {
            Some(&Loc::Uniform(u)) => {
                o.windows += 1;
                if o.f3 == 0 {
                    (o.r[k], o.f3) = (u, k as u32 + 1);
                }
            }
            Some(&Loc::Fixed(r)) => o.r[k] = r,
            None => match alloc.get(s) {
                Some(&r) => o.r[k] = r,
                None => o.missing = true,
            },
        }
    }
    o
}

/// Why the encoder cannot take `i` with these registers, if it cannot.
fn illegal(cx: &ShaderState, alloc: &HashMap<u32, u8>, i: &BorgInstr) -> Option<&'static str> {
    let rd = *alloc.get(&i.dst).unwrap_or(&0);
    match i.mnem {
        Mnem::Mov | Mnem::Label | Mnem::Break | Mnem::Continue => None,
        Mnem::Op(Op::Expush | Op::Exelse | Op::Expop | Op::Sout | Op::Fattr) => None,
        // BNE and CSRRW read a register, and r0 is x0.
        Mnem::Op(Op::Brnz | Op::Csrrw) => {
            let o = operands(cx, alloc, i);
            (o.missing || o.f3 != 0 || o.r[0] == 0).then_some("reads x0 or a window word")
        }
        m => {
            let o = operands(cx, alloc, i);
            if o.missing {
                return Some("has an operand without a register");
            }
            if o.windows > 1 {
                return Some("reads two window words");
            }
            if let Some(op) = imm_form(m) {
                return if o.f3 != 0 {
                    Some("reads a window word in an immediate form")
                } else if matches!(op, Op::Lw | Op::Sw | Op::Sws) && o.r[0] == 0 {
                    Some("has x0 as base")
                } else if matches!(op, Op::Sw | Op::Sws) && o.r[1] == 0 {
                    Some("stores x0")
                } else if !matches!(op, Op::Sw | Op::Sws) && rd == 0 {
                    Some("writes x0")
                } else {
                    None
                };
            }
            let w = encode(m, rd, o.r[0], o.r[1], o.r[2], o.f3);
            // A RISC-V form reads no window word.
            (w.is_some_and(|w| !matches!(w & 0x7F, 0 | 4)) && o.f3 != 0).then_some("reads a window word in a RISC-V form")
        }
    }
}

/// Every instruction is one the encoder takes with its registers; else the first that is not.
pub(crate) fn check(cx: &ShaderState, alloc: &HashMap<u32, u8>) -> Result<(), String> {
    match cx.sel.prog.iter().find_map(|i| illegal(cx, alloc, i).map(|why| (i.mnem, why))) {
        Some((m, why)) => Err(format!("{m} {why}")),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Stage;
    use crate::{Flags, NO_DST};

    fn op(o: Op, dst: u32, srcs: Vec<u32>) -> BorgInstr {
        BorgInstr { mnem: o.into(), dst, swz: vec![0; srcs.len()], srcs, imm: 0 }
    }

    #[test]
    fn each_rule_names_the_instruction() {
        let mut c = ShaderState::new(&[17, 18, 19], Stage::Fragment, Flags(0));
        let (a, b, d, u, w) = (c.sel.fresh(), c.sel.fresh(), c.sel.fresh(), c.sel.fresh(), c.sel.fresh());
        c.sel.loc.insert(u, Loc::Uniform(21));
        c.sel.loc.insert(w, Loc::Uniform(22));
        let alloc: HashMap<u32, u8> = [(a, 1), (b, 2), (d, 3)].into();
        let case = |c: &mut ShaderState, i: BorgInstr| {
            c.sel.prog = vec![i];
            check(c, &alloc)
        };
        assert_eq!(case(&mut c, op(Op::Fadd, d, vec![a, u])), Ok(()));
        assert_eq!(case(&mut c, op(Op::Fadd, d, vec![u, w])), Err("FADD reads two window words".into()));
        assert_eq!(case(&mut c, op(Op::Fadd, d, vec![a, 99])), Err("FADD has an operand without a register".into()));
        assert_eq!(case(&mut c, op(Op::Lw, d, vec![u])), Err("LW reads a window word in an immediate form".into()));
        assert_eq!(case(&mut c, op(Op::Add, d, vec![a, u])), Err("ADD reads a window word in a RISC-V form".into()));
        assert_eq!(case(&mut c, op(Op::Sw, NO_DST, vec![a, 99])), Err("SW has an operand without a register".into()));
        assert_eq!(case(&mut c, op(Op::Addi, 99, vec![a])), Err("ADDI writes x0".into()));
        assert_eq!(case(&mut c, op(Op::Brnz, NO_DST, vec![u])), Err("BRNZ reads x0 or a window word".into()));
    }
}
