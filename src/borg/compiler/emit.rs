use crate::encode::{emit_blob, encode, encode_branch, encode_fattr, encode_jmp, encode_sout};
use std::env;
use crate::state::{ShaderState, Stage};
use crate::{BorgInstr, Ubo};
use std::collections::HashMap;
use crate::opcodes::Op;
use crate::Mnem;

pub struct Encoded {
    pub words: Vec<u32>,
    pub n_preload: usize,
    pub n_body: usize,
    pub pending: u32,
}

pub fn encode_program(cx: &ShaderState, alloc: &HashMap<u32, u8>) -> Encoded {
    let mut words: Vec<u32> = Vec::new();
    let mut pos_gpr: std::collections::HashMap<(u32, u8), u8> = HashMap::new();
    let mut next_pre: u8 = 24; // r24..r29: above regalloc's low-GPR fill
    for i in &cx.sel.prog {
        if i.mnem == Mnem::Mov {
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
                if let Some(w) = encode(Op::Fadd.into(), g, c, 30, 0, 1) {
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
    let (labels, far) = branch_layout(&cx.sel.prog, n_preload);
    for (k, i) in cx.sel.prog.iter().enumerate() {
        if i.mnem == Mnem::Mov || i.mnem == Mnem::Label {
            continue;
        }
        if i.mnem == Op::Brnz {
            let t = labels[&i.swz[1]] as u32;
            let rs1 = resolve_op(i.srcs[0], i.swz[0]).0;
            match far.get(&k) {
                None => words.extend(encode_branch(Op::Brnz, rs1, (t % 1024) as u16)),
                Some(&pad) => {
                    for _ in 0..pad {
                        words.push(encode_jmp(words.len() as u32 + 1));
                    }
                    words.extend(encode_branch(Op::Brz, rs1, ((words.len() + 2) % 1024) as u16));
                    words.push(encode_jmp(t));
                }
            }
            continue;
        }
        if matches!(i.mnem, Mnem::Op(Op::Expush | Op::Exelse | Op::Expop)) {
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
        if i.mnem == Op::Sout {
            let index = i.swz[0] as u16;
            let rs2 = resolve_op(i.srcs[0], 0).0;
            if let Some(w) = encode_sout(rs2, index, 0) {
                words.push(w);
            }
            continue;
        }
        if i.mnem == Op::Fattr {
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
            if i.mnem == Op::Fmul || i.mnem == Op::Fmadd {
                let other = 1 - k; // the surviving multiply operand
                let (oreg, ouni) = resolve_op(i.srcs[other], i.swz[other]);
                // FMUL → FADD(rd, other, 0); FMADD → FADD(rd, other, acc).
                let (addend, f3) = if i.mnem == Op::Fmadd {
                    (resolve_op(i.srcs[2], i.swz[2]).0, if ouni { 1 } else { 0 })
                } else {
                    (30, if ouni { 1 } else { 0 }) // r30 = 0 → load `other`
                };
                if let Some(w) = encode(Op::Fadd.into(), rd, oreg, addend, 0, f3) {
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

/// LABEL word addresses, and the BRNZ whose target lies in another 1024-word page: those become
/// BRZ over a JMP, after JMP-to-next padding that keeps the BRZ's own target in its page.
fn branch_layout(prog: &[BorgInstr], start: usize) -> (HashMap<u8, usize>, HashMap<usize, usize>) {
    let mut far: HashMap<usize, usize> = HashMap::new();
    loop {
        let (mut a, mut labels, mut at) = (start, HashMap::new(), Vec::new());
        for (k, i) in prog.iter().enumerate() {
            match i.mnem {
                Mnem::Mov => {}
                Mnem::Label => {
                    labels.insert(i.swz[0], a);
                }
                Mnem::Op(Op::Brnz) => {
                    at.push((k, a));
                    a += far.get(&k).map_or(1, |p| p + 2);
                }
                _ => a += 1,
            }
        }
        let mut changed = false;
        for (k, a) in at {
            let t = labels[&prog[k].swz[1]];
            if far.contains_key(&k) || a / 1024 != t / 1024 {
                let pad = if a % 1024 > 1021 { 1024 - a % 1024 } else { 0 };
                changed |= far.insert(k, pad) != Some(pad);
            }
        }
        if !changed {
            return (labels, far);
        }
    }
}

pub fn build_blob(
    cx: &ShaderState,
    words: &[u32],
    n_preload: usize,
    n_body: usize,
    pending: u32,
    frag_out: &[Option<u32>; 4],
) -> Vec<u8> {
    let is_vertex = cx.stage == Stage::Vertex;
    eprintln!(
        "borgc: encoded {} word(s) = {} uniform pre-load(s) + {} op(s) + {} epilogue \
         ({} operand(s) still pending)",
        words.len(),
        n_preload,
        n_body - n_preload,
        words.len() - n_body,
        pending
    );
    if !cx.cp.presets().is_empty() {
        let cs: Vec<String> = cx.cp.presets().iter().map(|(r, v)| format!("r{r}={v:#06x}")).collect();
        eprintln!("borgc: const regs (firmware-staged via MMIO): {}", cs.join(" "));
    }
    let window = cx.window();
    if !window.is_empty() && !is_vertex {
        let cs: Vec<String> = window.iter().map(|(u, v)| format!("u{u}={v:#010x}")).collect();
        eprintln!(
            "borgc: draw-mode uniform consts (write to draw_fs_const_offset + 4*(u-20) before the draw): {}",
            cs.join(" ")
        );
    }
    if !window.is_empty() && is_vertex {
        let cs: Vec<String> = window.iter().map(|(u, v)| format!("u{u}={v:#010x}")).collect();
        eprintln!(
            "borgc: draw-mode vertex window consts (write to draw_vs_const_offset + 4*(u-25) before the draw): {}",
            cs.join(" ")
        );
    }
    if env::var("BORGC_DUMP_ISA").is_ok() {
        let hex: Vec<String> = words.iter().map(|w| format!("{w:#010x}")).collect();
        eprintln!("borgc:   first words: {}", hex.join(" "));
    }

    let outputs: Vec<u8> = if !is_vertex {
        // Fragment colour outputs r26/27/28, alpha in r24.
        let mut o: Vec<u8> = (0..3).filter(|&c| frag_out[c].is_some()).map(|c| 26 + c as u8).collect();
        if frag_out[3].is_some() {
            o.push(24);
        }
        o
    } else {
        vec![]
    };
    let num_varyings: u8 = cx.sel.prog
        .iter()
        .filter(|i| i.mnem == Op::Sout)
        .map(|i| i.swz[0] + 1)
        .max()
        .unwrap_or(0);
    let blob = emit_blob(words, &outputs, cx.cp.presets(),
                         Some((num_varyings, &window[..])));
    eprintln!(
        "borgc: .borg blob = {} bytes ({} instr, {} output regs, {} window consts, {} varyings)",
        blob.len(), words.len(), outputs.len(), window.len(), num_varyings
    );
    if let Ok(prefix) = env::var("BORGC_EMIT_BLOB") {
        let path = format!("{prefix}.{}.borg", if is_vertex { "vert" } else { "frag" });
        match std::fs::write(&path, &blob) {
            Ok(()) => eprintln!("borgc: wrote {path}"),
            Err(e) => eprintln!("borgc: failed to write {path}: {e}"),
        }
    }
    blob
}

#[cfg(test)]
mod tests {
    use super::branch_layout;
    use crate::opcodes::Op;
    use crate::{BorgInstr, Mnem, NO_DST};

    fn prog(body: usize) -> Vec<BorgInstr> {
        let mut p = vec![BorgInstr { mnem: Mnem::Label, dst: NO_DST, srcs: vec![], swz: vec![7] }];
        p.extend((0..body).map(|k| BorgInstr { mnem: Op::Iadd.into(), dst: k as u32, srcs: vec![], swz: vec![] }));
        p.push(BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![0], swz: vec![0, 7] });
        p
    }

    #[test]
    fn a_branch_in_its_targets_page_is_near() {
        let (labels, far) = branch_layout(&prog(10), 3);
        assert_eq!(labels[&7], 3);
        assert!(far.is_empty());
    }

    #[test]
    fn a_far_branch_is_padded_off_the_page_end() {
        // Label at 1000, BRNZ at 2046 (1022 in its page): its BRZ would skip into the next page.
        let p = prog(1046);
        let (_, far) = branch_layout(&p, 1000);
        assert_eq!(far[&(p.len() - 1)], 2);
        let (_, far) = branch_layout(&prog(1100), 0);
        assert_eq!(far.values().copied().collect::<Vec<_>>(), vec![0]);
    }
}
