// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// IR-to-IR optimization passes over the selected Vec<BorgInstr> program, run
// after instruction selection and before register allocation.

use crate::BorgInstr;
use std::collections::HashMap;

/// Dead-code elimination. Because every load_ubo is pinned to a fixed uniform,
/// the UBO byte-offset arithmetic (iadd/ishl from gl_VertexIndex, descriptor
/// index math) feeds only the dropped offsets and is dead. Mark-and-sweep from
/// the store_output roots back through the def→use chain to fixpoint.
pub(crate) fn dce(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let pre_dce = prog.len();
    let mut live: std::collections::HashSet<u32> = out_roots.iter().copied().collect();
    loop {
        let mut grew = false;
        for i in prog.iter() {
            if live.contains(&i.dst) {
                for &s in &i.srcs {
                    grew |= live.insert(s);
                }
            }
        }
        if !grew {
            break;
        }
    }
    prog.retain(|i| live.contains(&i.dst));
    if pre_dce != prog.len() {
        eprintln!(
            "borgc: DCE — dropped {} dead instr(s) (UBO address math), {} live",
            pre_dce - prog.len(),
            prog.len()
        );
    }
}

/// Peephole: re-fuse FMUL+FADD into FMADD. nir_lower_alu_to_scalar split the
/// source ffma into a multiply and an add; fuse FADD(d, m, c) where m=FMUL(a,b)
/// is used only here back into FMADD(d, a*b + c). Halves the matrix-multiply op
/// count — both faster and necessary to fit SPIRB_MAX_INSTRS (32).
pub(crate) fn fuse_fmadd(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let pre_fuse = prog.len();
    {
        let mut def_idx: HashMap<u32, usize> = HashMap::new();
        for (i, instr) in prog.iter().enumerate() {
            def_idx.insert(instr.dst, i);
        }
        let mut uses: HashMap<u32, u32> = HashMap::new();
        for instr in prog.iter() {
            for &s in &instr.srcs {
                *uses.entry(s).or_insert(0) += 1;
            }
        }
        for &r in out_roots {
            *uses.entry(r).or_insert(0) += 1; // outputs count as a use (don't fuse away)
        }
        let mut remove = vec![false; prog.len()];
        for i in 0..prog.len() {
            if prog[i].mnem != "FADD" || prog[i].srcs.len() != 2 {
                continue;
            }
            for k in 0..2 {
                let m = prog[i].srcs[k];
                let Some(&mi) = def_idx.get(&m) else { continue };
                if mi < i
                    && !remove[mi]
                    && prog[mi].mnem == "FMUL"
                    && prog[mi].srcs.len() == 2
                    && uses.get(&m) == Some(&1)
                {
                    let (a, b) = (prog[mi].srcs[0], prog[mi].srcs[1]);
                    let (sa, sb) = (prog[mi].swz[0], prog[mi].swz[1]);
                    let c = prog[i].srcs[1 - k];
                    let sc = prog[i].swz[1 - k];
                    prog[i].mnem = "FMADD";
                    prog[i].srcs = vec![a, b, c];
                    prog[i].swz = vec![sa, sb, sc];
                    remove[mi] = true;
                    break;
                }
            }
        }
        let mut idx = 0;
        prog.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
    }
    if pre_fuse != prog.len() {
        eprintln!(
            "borgc: FMADD fusion — {} FMUL+FADD pair(s) fused, {} instr(s)",
            pre_fuse - prog.len(),
            prog.len()
        );
    }
}
