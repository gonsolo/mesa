use crate::legal::{imm_form, operands};
use crate::encode::{emit_blob, encode, encode_csrw, encode_fattr, encode_imm, encode_jal, encode_rv_branch, encode_sout};
use crate::state::{ShaderState, Stage};
use crate::{BorgInstr, Loc};
use std::collections::{HashMap, HashSet};
use crate::opcodes::Op;
use crate::Mnem;

pub struct Encoded {
    pub words: Vec<u32>,
    pub n_body: usize,
    /// Instructions without an encoding.
    pub bad: u32,
}

/// Encodes the allocated program: physical registers for every operand, and the words.
struct Emitter<'a> {
    cx: &'a ShaderState,
    alloc: &'a HashMap<u32, u8>,
    words: Vec<u32>,
    bad: u32,
}

impl Emitter<'_> {
    fn push(&mut self, w: Option<u32>) {
        self.bad += w.is_none() as u32;
        self.words.extend(w);
    }

    /// One source operand as (physical register, read from the uniform window).
    fn operand(&self, s: u32, c: u8) -> (u8, bool) {
        let (cx, alloc) = (self.cx, self.alloc);
        // TEX and FATTR results: component c is rd + c.
        if cx.tex_dsts.contains(&s) || cx.fattr_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        match cx.sel.loc.get(&s) {
            Some(&Loc::Uniform(idx)) => (idx, true),
            Some(&Loc::Fixed(reg)) => (reg, false),
            _ => (*alloc.get(&s).unwrap_or(&0), false),
        }
    }

    /// BNE rs, x0 to the LABEL named in imm; a target out of its reach is a BEQ over a JAL.
    fn branch(&mut self, k: usize, i: &BorgInstr, labels: &HashMap<u32, usize>, far: &HashSet<usize>) {
        let t = labels[&i.imm] as i32;
        let (rs1, at) = (self.operand(i.srcs[0], i.swz[0]).0, self.words.len() as i32);
        if far.contains(&k) {
            self.push(encode_rv_branch(Op::Beq, rs1, 2));
            self.push(encode_jal(t - (at + 1)));
        } else {
            self.push(encode_rv_branch(Op::Bne, rs1, t - at));
        }
    }

    /// Any other instruction: a register per operand, the first window word read through funct3.
    fn instr(&mut self, i: &BorgInstr, rd: u8) {
        let o = operands(self.cx, self.alloc, i);
        let w = match imm_form(i.mnem) {
            Some(op) => encode_imm(op, rd, o.r[0], o.r[1], i.imm),
            None => encode(i.mnem, rd, o.r[0], o.r[1], o.r[2], o.f3),
        };
        self.push(w);
    }
}

pub fn encode_program(cx: &ShaderState, alloc: &HashMap<u32, u8>) -> Encoded {
    let mut e = Emitter { cx, alloc, words: Vec::new(), bad: 0 };
    let regions = uniform_regions(&cx.sel.prog);
    let (labels, far, addr) = branch_layout(&cx.sel.prog, 0, &regions);
    for (k, i) in cx.sel.prog.iter().enumerate() {
        debug_assert!(e.bad > 0 || e.words.len() == addr[k]);
        let rd = *alloc.get(&i.dst).unwrap_or(&0);
        match i.mnem {
            Mnem::Mov | Mnem::Label | Mnem::Break | Mnem::Continue => {}
            Mnem::Op(Op::Brnz) => e.branch(k, i, &labels, &far),
            // A uniform push skips its side when no lane takes it: BEQ, and the then side's JAL.
            Mnem::Op(Op::Expush) if regions.contains_key(&k) => {
                let r = &regions[&k];
                let t = r.els.filter(|&x| r.jal(x)).map_or(addr[r.pop], |x| addr[x] + 1) as i32;
                let (rs1, at) = (e.operand(i.srcs[0], i.swz[0]).0, e.words.len() as i32);
                if far.contains(&k) {
                    e.push(encode_rv_branch(Op::Bne, rs1, 2));
                    e.push(encode_jal(t - (at + 1)));
                } else {
                    e.push(encode_rv_branch(Op::Beq, rs1, t - at));
                }
            }
            Mnem::Op(Op::Exelse | Op::Expop) if regions.values().any(|r| r.els == Some(k) || r.pop == k) => {
                if let Some(r) = regions.values().find(|r| r.els == Some(k) && r.jal(k)) {
                    e.push(encode_jal(addr[r.pop] as i32 - e.words.len() as i32));
                }
            }
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
                let rs1 = e.operand(i.srcs[0], i.swz[0]).0;
                e.push(encode_csrw(rs1, i.imm));
            }
            _ => e.instr(i, rd),
        }
    }
    let n_body = e.words.len();
    Encoded { words: e.words, n_body, bad: e.bad }
}

/// The EXELSE and EXPOP of a uniform EXPUSH (imm 1).
struct Region {
    els: Option<usize>,
    pop: usize,
    // Whether anything is emitted between the EXELSE and the EXPOP.
    full_else: bool,
}

impl Region {
    /// The then side ends with a JAL over the else side.
    fn jal(&self, els: usize) -> bool {
        self.els == Some(els) && self.full_else
    }
}

pub(crate) fn emits(m: Mnem) -> bool {
    !matches!(m, Mnem::Mov | Mnem::Label | Mnem::Break | Mnem::Continue)
}

fn uniform_regions(prog: &[BorgInstr]) -> HashMap<usize, Region> {
    let (mut open, mut out): (Vec<(usize, Option<usize>)>, HashMap<usize, Region>) = (vec![], HashMap::new());
    for (k, i) in prog.iter().enumerate() {
        match i.mnem {
            Mnem::Op(Op::Expush) => open.push((k, None)),
            Mnem::Op(Op::Exelse) => open.last_mut().unwrap().1 = Some(k),
            Mnem::Op(Op::Expop) => {
                let (p, els) = open.pop().unwrap();
                if prog[p].imm == 1 {
                    let full_else = els.is_some_and(|x| prog[x + 1..k].iter().any(|i| emits(i.mnem)));
                    out.insert(p, Region { els, pop: k, full_else });
                }
            }
            _ => {}
        }
    }
    out
}

/// LABEL word addresses from the program's start, the branches out of a BEQ's reach (a BRNZ, or a
/// uniform EXPUSH: one word longer as a branch over a JAL, so the layout repeats until it
/// settles) and each instruction's word address.
fn branch_layout(prog: &[BorgInstr], start: usize, regions: &HashMap<usize, Region>) -> (HashMap<u32, usize>, HashSet<usize>, Vec<usize>) {
    let mut far = HashSet::new();
    let skipped: HashSet<usize> = regions.values().flat_map(|r| [Some(r.pop), r.els.filter(|&x| !r.jal(x))]).flatten().collect();
    loop {
        let (mut a, mut labels, mut addr) = (start, HashMap::new(), Vec::with_capacity(prog.len()));
        for (k, i) in prog.iter().enumerate() {
            addr.push(a);
            match i.mnem {
                _ if !emits(i.mnem) => {
                    if i.mnem == Mnem::Label {
                        labels.insert(i.imm, a);
                    }
                }
                _ if skipped.contains(&k) => {}
                Mnem::Op(Op::Brnz | Op::Expush) => a += if far.contains(&k) { 2 } else { 1 },
                _ => a += 1,
            }
        }
        let mut changed = false;
        for (k, i) in prog.iter().enumerate() {
            let t = match i.mnem {
                Mnem::Op(Op::Brnz) => labels[&i.imm],
                Mnem::Op(Op::Expush) => match regions.get(&k) {
                    Some(r) => r.els.filter(|&x| r.jal(x)).map_or(addr[r.pop], |x| addr[x] + 1),
                    None => continue,
                },
                _ => continue,
            };
            if !(-1024..1024).contains(&(t as i64 - addr[k] as i64)) {
                changed |= far.insert(k);
            }
        }
        if !changed {
            return (labels, far, addr);
        }
    }
}

pub fn build_blob(
    cx: &ShaderState,
    words: &[u32],
    n_body: usize,
    frag_out: &[Option<u32>; 4],
) -> Vec<u8> {
    let is_vertex = cx.stage == Stage::Vertex;
    debug!(
        "borgc: encoded {} word(s) = {} op(s) + {} epilogue",
        words.len(),
        n_body,
        words.len() - n_body
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
    use super::{branch_layout, uniform_regions};
    use std::collections::HashMap;
    use crate::opcodes::Op;
    use crate::{BorgInstr, Mnem, NO_DST};

    fn prog(body: usize) -> Vec<BorgInstr> {
        let mut p = vec![BorgInstr { mnem: Mnem::Label, dst: NO_DST, srcs: vec![], swz: vec![], imm: 7 }];
        p.extend((0..body).map(|k| BorgInstr { mnem: Op::Iadd.into(), dst: k as u32, srcs: vec![], swz: vec![], imm: 0 }));
        p.push(BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![0], swz: vec![0], imm: 7 });
        p
    }

    #[test]
    fn a_branch_back_1024_words_is_near() {
        // Label at 3, BNE at 3 + 1024: offset -1024, the farthest a branch reaches back.
        let (labels, far, _) = branch_layout(&prog(1024), 3, &HashMap::new());
        assert_eq!(labels[&7], 3);
        assert!(far.is_empty());
    }

    #[test]
    fn a_branch_back_1025_words_is_a_beq_over_a_jal() {
        let p = prog(1025);
        let (_, far, _) = branch_layout(&p, 3, &HashMap::new());
        assert_eq!(far.into_iter().collect::<Vec<_>>(), vec![p.len() - 1]);
    }

    #[test]
    fn a_uniform_if_takes_no_mask_level() {
        let i = |m: Op, imm: u32| BorgInstr { mnem: m.into(), dst: NO_DST, srcs: vec![1], swz: vec![0], imm };
        // EXPUSH (uniform) 0, IADD 1, EXELSE as JAL 2, IADD 3, EXPOP nothing: 4.
        let p = vec![i(Op::Expush, 1), i(Op::Iadd, 0), i(Op::Exelse, 0), i(Op::Iadd, 0), i(Op::Expop, 0)];
        let r = uniform_regions(&p);
        let (_, far, addr) = branch_layout(&p, 0, &r);
        assert!(far.is_empty() && r[&0].jal(2));
        assert_eq!((addr[2] + 1, addr[4]), (3, 4));
        // An empty else side: no JAL, the BEQ goes to the end.
        let p = vec![i(Op::Expush, 1), i(Op::Iadd, 0), i(Op::Exelse, 0), i(Op::Expop, 0), i(Op::Iadd, 0)];
        let r = uniform_regions(&p);
        let (_, _, addr) = branch_layout(&p, 0, &r);
        assert!(!r[&0].jal(2));
        assert_eq!((addr[3], addr[4]), (2, 2));
        // A divergent push is no region.
        assert!(uniform_regions(&[i(Op::Expush, 0), i(Op::Expop, 0)]).is_empty());
    }
}
