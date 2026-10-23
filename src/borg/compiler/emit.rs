use crate::encode::{emit_blob, encode, encode_branch, encode_csrw, encode_fattr, encode_jmp, encode_sout};
use crate::state::{ShaderState, Stage};
use crate::{BorgInstr, Loc};
use std::collections::{HashMap, HashSet};
use crate::opcodes::Op;
use crate::Mnem;

pub struct Encoded {
    pub words: Vec<u32>,
    pub n_preload: usize,
    pub n_body: usize,
    pub pending: u32,
}

/// Encodes the allocated program: physical registers for every operand, and the words.
struct Emitter<'a> {
    cx: &'a ShaderState,
    alloc: &'a HashMap<u32, u8>,
    pos_gpr: HashMap<(u32, u8), u8>,
    words: Vec<u32>,
    pending: u32,
}

impl Emitter<'_> {
    fn push(&mut self, w: Option<u32>) {
        self.words.extend(w);
    }

    /// Loads the position components the program reads into r24..r29; pos.w (1.0) is folded.
    fn preload_positions(&mut self) {
        let cx = self.cx;
        let mut next_pre: u8 = 24; // above regalloc's low-GPR fill
        for i in cx.sel.prog.iter().filter(|i| i.mnem != Mnem::Mov) {
            for (s, &c) in i.srcs.iter().zip(i.swz.iter()) {
                if cx.sel.loc.get(s) == Some(&Loc::Pos) && c != 3 && !self.pos_gpr.contains_key(&(*s, c)) && next_pre <= 29 {
                    let g = next_pre;
                    next_pre += 1;
                    self.pos_gpr.insert((*s, c), g);
                    // g = u[c] (position component c → uniform u0..u2), funct3=1.
                    self.push(encode(Op::Fadd.into(), g, c, 30, 0, 1));
                }
            }
        }
    }

    /// One source operand as (physical register, read from the uniform window).
    fn operand(&self, s: u32, c: u8) -> (u8, bool) {
        let (cx, alloc) = (self.cx, self.alloc);
        // TEX and FATTR results: component c is rd + c.
        if cx.tex_dsts.contains(&s) || cx.fattr_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        match cx.sel.loc.get(&s) {
            Some(&Loc::Pos) => (*self.pos_gpr.get(&(s, c)).unwrap_or(&0), false),
            Some(&Loc::Uniform(idx)) => (idx, true),
            Some(&Loc::Fixed(reg)) => (reg, false),
            _ => (*alloc.get(&s).unwrap_or(&0), false),
        }
    }

    /// BRNZ to the LABEL named in imm; a target out of its reach is a BRZ over a JMP.
    fn branch(&mut self, k: usize, i: &BorgInstr, labels: &HashMap<u32, usize>, far: &HashSet<usize>) {
        let t = labels[&i.imm] as i32;
        let (rs1, at) = (self.operand(i.srcs[0], i.swz[0]).0, self.words.len() as i32);
        if far.contains(&k) {
            self.push(encode_branch(Op::Brz, rs1, 2));
            self.push(encode_jmp(t - (at + 1)));
        } else {
            self.push(encode_branch(Op::Brnz, rs1, t - at));
        }
    }

    /// FMUL or FMADD by pos.w (1.0) becomes FADD of the other operand; false for anything else.
    fn fold_pos_w(&mut self, i: &BorgInstr, rd: u8) -> bool {
        let cx = self.cx;
        let posw = i.srcs.iter().zip(i.swz.iter())
            .take(2) // only the multiply operands (rs1/rs2) can be pos.w
            .position(|(s, &c)| cx.sel.loc.get(s) == Some(&Loc::Pos) && c == 3);
        let Some(k) = posw else { return false };
        if i.mnem != Op::Fmul && i.mnem != Op::Fmadd {
            return false;
        }
        let (oreg, ouni) = self.operand(i.srcs[1 - k], i.swz[1 - k]);
        // FMUL → FADD(rd, other, 0); FMADD → FADD(rd, other, acc); r30 = 0.
        let addend = if i.mnem == Op::Fmadd { self.operand(i.srcs[2], i.swz[2]).0 } else { 30 };
        self.push(encode(Op::Fadd.into(), rd, oreg, addend, 0, ouni as u32));
        true
    }

    /// Any other instruction: a register per operand, the first window word read through funct3.
    fn instr(&mut self, i: &BorgInstr, rd: u8) {
        let (cx, alloc) = (self.cx, self.alloc);
        let mut r = [0u8; 3];
        let mut f3 = 0u32;
        for (k, (s, &c)) in i.srcs.iter().zip(i.swz.iter()).take(3).enumerate() {
            if cx.tex_dsts.contains(s) || cx.fattr_dsts.contains(s) {
                r[k] = alloc.get(s).map_or(0, |&rr| rr + c);
                continue;
            }
            match cx.sel.loc.get(s) {
                Some(&Loc::Pos) => r[k] = *self.pos_gpr.get(&(*s, c)).unwrap_or(&0),
                Some(&Loc::Uniform(idx)) => {
                    if f3 == 0 {
                        r[k] = idx;
                        f3 = k as u32 + 1;
                    } else {
                        self.pending += 1; // 2nd uniform in one op needs a preload
                    }
                }
                Some(&Loc::Fixed(reg)) => r[k] = reg,
                // ALU intermediate → allocated GPR (or pending if a stray input).
                None => match alloc.get(s) {
                    Some(&phys) => r[k] = phys,
                    None => self.pending += 1,
                },
            }
        }
        // LW and SW: no window operand, and r0 is x0 there.
        if matches!(i.mnem, Mnem::Op(Op::Lw | Op::Sw)) && (f3 != 0 || r[0] == 0 || (i.mnem == Op::Sw && r[1] == 0) || (i.mnem == Op::Lw && rd == 0)) {
            self.pending += 1;
        }
        self.push(encode(i.mnem, rd, r[0], r[1], r[2], f3));
    }
}

pub fn encode_program(cx: &ShaderState, alloc: &HashMap<u32, u8>) -> Encoded {
    let mut e = Emitter { cx, alloc, pos_gpr: HashMap::new(), words: Vec::new(), pending: 0 };
    e.preload_positions();
    let n_preload = e.words.len();
    let (labels, far) = branch_layout(&cx.sel.prog, n_preload);
    for (k, i) in cx.sel.prog.iter().enumerate() {
        let rd = *alloc.get(&i.dst).unwrap_or(&0);
        match i.mnem {
            Mnem::Mov | Mnem::Label => {}
            Mnem::Op(Op::Brnz) => e.branch(k, i, &labels, &far),
            Mnem::Op(Op::Expush | Op::Exelse | Op::Expop) => {
                let rs1 = if i.srcs.is_empty() { 0 } else { e.operand(i.srcs[0], i.swz[0]).0 };
                e.push(encode(i.mnem, 0, rs1, 0, 0, 0));
            }
            Mnem::Op(Op::Sout) => {
                let rs2 = e.operand(i.srcs[0], 0).0;
                e.push(encode_sout(rs2, i.imm as u16, 0));
            }
            Mnem::Op(Op::Fattr) => e.push(encode_fattr(rd, i.imm as u16)),
            Mnem::Op(Op::Csrrw) => {
                let (rs1, window) = e.operand(i.srcs[0], i.swz[0]);
                e.pending += (window || rs1 == 0) as u32; // a window word is no register, r0 is x0
                e.push(encode_csrw(rs1, i.imm));
            }
            _ => {
                if !e.fold_pos_w(i, rd) {
                    e.instr(i, rd);
                }
            }
        }
    }
    let n_body = e.words.len();
    Encoded { words: e.words, n_preload, n_body, pending: e.pending }
}

/// LABEL word addresses from the program's start, and the BRNZ whose label is out of a branch's
/// reach: those become BRZ over a JMP, one word longer, so the layout repeats until it settles.
fn branch_layout(prog: &[BorgInstr], start: usize) -> (HashMap<u32, usize>, HashSet<usize>) {
    let mut far = HashSet::new();
    loop {
        let (mut a, mut labels, mut at) = (start, HashMap::new(), Vec::new());
        for (k, i) in prog.iter().enumerate() {
            match i.mnem {
                Mnem::Mov => {}
                Mnem::Label => {
                    labels.insert(i.imm, a);
                }
                Mnem::Op(Op::Brnz) => {
                    at.push((k, a));
                    a += if far.contains(&k) { 2 } else { 1 };
                }
                _ => a += 1,
            }
        }
        let mut changed = false;
        for (k, a) in at {
            if !(-512..512).contains(&(labels[&prog[k].imm] as i64 - a as i64)) {
                changed |= far.insert(k);
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
    debug!(
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
        debug!("borgc: const regs (firmware-staged via MMIO): {}", cs.join(" "));
    }
    let window = cx.window();
    if !window.is_empty() && !is_vertex {
        let cs: Vec<String> = window.iter().map(|(u, v)| format!("u{u}={v:#010x}")).collect();
        debug!(
            "borgc: draw-mode uniform consts (write to draw_fs_const_offset + 4*(u-20) before the draw): {}",
            cs.join(" ")
        );
    }
    if !window.is_empty() && is_vertex {
        let cs: Vec<String> = window.iter().map(|(u, v)| format!("u{u}={v:#010x}")).collect();
        debug!(
            "borgc: draw-mode vertex window consts (write to draw_vs_const_offset + 4*(u-25) before the draw): {}",
            cs.join(" ")
        );
    }
    if crate::config::env().dump_isa {
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
        .map(|i| (i.imm & 0xFF) as u8 + 1)
        .max()
        .unwrap_or(0);
    let blob = emit_blob(words, &outputs, cx.cp.presets(),
                         Some((num_varyings, &window[..])));
    debug!(
        "borgc: .borg blob = {} bytes ({} instr, {} output regs, {} window consts, {} varyings)",
        blob.len(), words.len(), outputs.len(), window.len(), num_varyings
    );
    if let Some(prefix) = &crate::config::env().emit_blob {
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
        let mut p = vec![BorgInstr { mnem: Mnem::Label, dst: NO_DST, srcs: vec![], swz: vec![], imm: 7 }];
        p.extend((0..body).map(|k| BorgInstr { mnem: Op::Iadd.into(), dst: k as u32, srcs: vec![], swz: vec![], imm: 0 }));
        p.push(BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![0], swz: vec![0], imm: 7 });
        p
    }

    #[test]
    fn a_branch_back_512_words_is_near() {
        // Label at 3, BRNZ at 3 + 512: offset -512, the farthest a branch reaches back.
        let (labels, far) = branch_layout(&prog(512), 3);
        assert_eq!(labels[&7], 3);
        assert!(far.is_empty());
    }

    #[test]
    fn a_branch_back_513_words_is_a_brz_over_a_jmp() {
        let p = prog(513);
        let (_, far) = branch_layout(&p, 3);
        assert_eq!(far.into_iter().collect::<Vec<_>>(), vec![p.len() - 1]);
    }
}
