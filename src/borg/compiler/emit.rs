use crate::encode::{encode, encode_fattr, encode_sout};
use crate::state::Cx;
use crate::Ubo;
use std::collections::HashMap;

pub struct Encoded {
    pub words: Vec<u32>,
    pub n_preload: usize,
    pub n_body: usize,
    pub pending: u32,
}

pub fn encode_program(cx: &Cx, alloc: &HashMap<u32, u8>) -> Encoded {
    let mut words: Vec<u32> = Vec::new();
    let mut pos_gpr: std::collections::HashMap<(u32, u8), u8> = HashMap::new();
    let mut next_pre: u8 = 24; // r24..r29: above regalloc's low-GPR fill
    for i in &cx.sel.prog {
        if i.mnem == "mov" {
            continue;
        }
        for (s, &c) in i.srcs.iter().zip(i.swz.iter()) {
            if cx.sel.ubo.get(s) == Some(&Ubo::Pos)
                && c != 3 // pos.w is the constant 1.0 (folded below), never pre-loaded
                && !pos_gpr.contains_key(&(*s, c))
                && next_pre <= 29
            {
                let g = next_pre;
                next_pre += 1;
                pos_gpr.insert((*s, c), g);
                // g = u[c] (position component c → uniform u0..u2), funct3=1.
                if let Some(w) = encode("FADD", g, c, 30, 0, 1) {
                    words.push(w);
                }
            }
        }
    }
    let n_preload = words.len();

    let mut pending = 0u32;
    // Resolve one source operand to (physical register, reads-from-uniform).
    let resolve_op = |s: u32, c: u8| -> (u8, bool) {
        // TEX result component c → rd+c (R/G/B/A in consecutive regs).
        if cx.tex_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        // FATTR result component c → rd+c (per-vertex v0/v1/v2 in consecutive regs).
        if cx.fattr_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        match cx.sel.ubo.get(&s) {
            Some(&Ubo::Pos) => (*pos_gpr.get(&(s, c)).unwrap_or(&0), false),
            Some(&Ubo::Uniform(idx)) => (idx, true),
            Some(&Ubo::Fixed(reg)) => (reg, false),
            _ => (*alloc.get(&s).unwrap_or(&0), false),
        }
    };
    for i in &cx.sel.prog {
        if i.mnem == "mov" {
            continue;
        }
        if matches!(i.mnem, "EXPUSH" | "EXELSE" | "EXPOP") {
            let rs1 = if i.srcs.is_empty() {
                0
            } else {
                resolve_op(i.srcs[0], i.swz[0]).0
            };
            if let Some(w) = encode(i.mnem, 0, rs1, 0, 0, 0) {
                words.push(w);
            }
            continue;
        }
        if i.mnem == "SOUT" {
            let index = i.swz[0] as u16;
            let rs2 = resolve_op(i.srcs[0], 0).0;
            if let Some(w) = encode_sout(rs2, index, 0) {
                words.push(w);
            }
            continue;
        }
        if i.mnem == "FATTR" {
            let index = i.swz[0] as u16;
            let rd = *alloc.get(&i.dst).unwrap_or(&0);
            if let Some(w) = encode_fattr(rd, index) {
                words.push(w);
            }
            continue;
        }
        let rd = *alloc.get(&i.dst).unwrap_or(&0);

        let posw = i
            .srcs
            .iter()
            .zip(i.swz.iter())
            .take(2) // only the multiply operands (rs1/rs2) can be pos.w
            .position(|(s, &c)| cx.sel.ubo.get(s) == Some(&Ubo::Pos) && c == 3);
        if let Some(k) = posw {
            if i.mnem == "FMUL" || i.mnem == "FMADD" {
                let other = 1 - k; // the surviving multiply operand
                let (oreg, ouni) = resolve_op(i.srcs[other], i.swz[other]);
                // FMUL → FADD(rd, other, 0); FMADD → FADD(rd, other, acc).
                let (addend, f3) = if i.mnem == "FMADD" {
                    (resolve_op(i.srcs[2], i.swz[2]).0, if ouni { 1 } else { 0 })
                } else {
                    (30, if ouni { 1 } else { 0 }) // r30 = 0 → load `other`
                };
                if let Some(w) = encode("FADD", rd, oreg, addend, 0, f3) {
                    words.push(w);
                }
                continue;
            }
        }

        let mut r = [0u8; 3];
        let mut f3 = 0u32;
        for (k, (s, &c)) in i.srcs.iter().zip(i.swz.iter()).take(3).enumerate() {
            if cx.tex_dsts.contains(s) {
                r[k] = alloc.get(s).map_or(0, |&rr| rr + c); // TEX result rd+c
                continue;
            }
            if cx.fattr_dsts.contains(s) {
                r[k] = alloc.get(s).map_or(0, |&rr| rr + c); // FATTR result rd+c
                continue;
            }
            match cx.sel.ubo.get(s) {
                // position component → its pre-loaded GPR.
                Some(&Ubo::Pos) => r[k] = *pos_gpr.get(&(*s, c)).unwrap_or(&0),
                // fragment uniform (inv_area / per-vertex varying) → funct3 read.
                Some(&Ubo::Uniform(idx)) => {
                    if f3 == 0 {
                        r[k] = idx;
                        f3 = k as u32 + 1;
                    } else {
                        pending += 1; // 2nd uniform in one op needs a preload
                    }
                }
                // edge-function attribute in a fixed register (r0/r1/r2).
                Some(&Ubo::Fixed(reg)) => r[k] = reg,
                // ALU intermediate → allocated GPR (or pending if a stray input).
                None => match alloc.get(s) {
                    Some(&phys) => r[k] = phys,
                    None => pending += 1,
                },
            }
        }
        if let Some(w) = encode(i.mnem, rd, r[0], r[1], r[2], f3) {
            words.push(w);
        }
    }
    let n_body = words.len();
    Encoded { words, n_preload, n_body, pending }
}
