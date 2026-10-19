// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::BorgInstr;
use std::collections::{HashMap, HashSet};
use crate::opcodes::Op;
use crate::Mnem;

pub(crate) fn dce(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let pre_dce = prog.len();
    let mut live: std::collections::HashSet<u32> = out_roots.iter().copied().collect();

    let side_effecting = |i: &BorgInstr| i.dst == crate::NO_DST;
    for i in prog.iter().filter(|i| side_effecting(i)) {
        for &s in &i.srcs {
            live.insert(s);
        }
    }

    loop {
        let mut grew = false;
        for i in prog.iter() {
            if live.contains(&i.dst) || side_effecting(i) {
                for &s in &i.srcs {
                    grew |= live.insert(s);
                }
            }
        }
        if !grew {
            break;
        }
    }
    prog.retain(|i| side_effecting(i) || live.contains(&i.dst));
    if pre_dce != prog.len() {
        debug!(
            "borgc: DCE — dropped {} dead instr(s) (UBO address math), {} live",
            pre_dce - prog.len(),
            prog.len()
        );
    }
}

/// Reorders the program to keep few values live at once, since registers are scarce. Each run
/// of side-effect-free instructions between TEX, FATTR and stores is scheduled on its own,
/// preferring instructions that free their sources, and the order inside a run follows the
/// dependencies.
pub(crate) fn schedule_for_pressure(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let out_roots: HashSet<u32> = out_roots.iter().copied().collect();

    let mut runs: Vec<(Vec<BorgInstr>, Option<BorgInstr>)> = Vec::new();
    {
        let mut cur: Vec<BorgInstr> = Vec::new();
        for instr in prog.drain(..) {
            if instr.dst == crate::NO_DST || instr.mnem == Op::Tex || instr.mnem == Op::Fattr {
                runs.push((std::mem::take(&mut cur), Some(instr)));
            } else {
                cur.push(instr);
            }
        }
        runs.push((cur, None));
    }

    for (run, anchor) in runs {
        prog.extend(schedule_run(run, &out_roots));
        if let Some(a) = anchor {
            prog.push(a);
        }
    }
}

fn schedule_run(run: Vec<BorgInstr>, out_roots: &HashSet<u32>) -> Vec<BorgInstr> {
    let n = run.len();
    if n <= 1 {
        return run;
    }

    let defined_here: HashSet<u32> = run.iter().map(|i| i.dst).collect();
    // The order below follows def-use edges, so a value redefined or read before its definition
    // here (a NIR register, a loop counter) keeps the run as it is.
    let mut seen = HashSet::new();
    for i in &run {
        if i.srcs.iter().any(|s| defined_here.contains(s) && !seen.contains(s)) || !seen.insert(i.dst) {
            return run;
        }
    }

    let mut remaining_uses: HashMap<u32, i32> = HashMap::new();
    for instr in &run {
        for &s in &instr.srcs {
            *remaining_uses.entry(s).or_insert(0) += 1;
        }
    }
    for instr in &run {
        if out_roots.contains(&instr.dst) {
            *remaining_uses.entry(instr.dst).or_insert(0) += 1;
        }
    }

    let mut pending_deps: Vec<i32> = run
        .iter()
        .map(|i| i.srcs.iter().filter(|s| defined_here.contains(s)).count() as i32)
        .collect();
    let mut consumers: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, instr) in run.iter().enumerate() {
        for &s in &instr.srcs {
            if defined_here.contains(&s) {
                consumers.entry(s).or_default().push(i);
            }
        }
    }

    let mut urgency: Vec<usize> = vec![0; n];
    for i in (0..n).rev() {
        urgency[i] = match consumers.get(&run[i].dst) {
            Some(cs) if !cs.is_empty() => cs.iter().map(|&c| urgency[c]).min().unwrap(),
            _ => i,
        };
    }

    let mut slots: Vec<Option<BorgInstr>> = run.into_iter().map(Some).collect();
    let mut ready: Vec<usize> = (0..n).filter(|&i| pending_deps[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut last_dst: Option<u32> = None;

    for _ in 0..n {
        let best = *ready
            .iter()
            .min_by_key(|&&i| {
                let instr = slots[i].as_ref().unwrap();
                let produces = i32::from(remaining_uses.get(&instr.dst).copied().unwrap_or(0) > 0);
                // How many of its sources die right here.
                let frees = instr
                    .srcs
                    .iter()
                    .filter(|&&s| remaining_uses.get(&s).copied().unwrap_or(0) == 1)
                    .count() as i32;
                let continues_last = !(last_dst.is_some() && instr.srcs.contains(&last_dst.unwrap()));
                (produces - frees, urgency[i], continues_last, i)
            })
            .expect("ready is non-empty while instructions remain");
        ready.retain(|&i| i != best);
        order.push(best);

        let instr = slots[best].as_ref().unwrap();
        for &s in &instr.srcs {
            if let Some(c) = remaining_uses.get_mut(&s) {
                *c -= 1;
            }
        }
        let dst = instr.dst;
        last_dst = Some(dst);
        if let Some(cs) = consumers.get(&dst) {
            for &ci in cs {
                pending_deps[ci] -= 1;
                if pending_deps[ci] == 0 {
                    ready.push(ci);
                }
            }
        }
    }

    order.into_iter().map(|i| slots[i].take().unwrap()).collect()
}

pub(crate) fn fuse_fmadd(
    prog: &mut Vec<BorgInstr>,
    out_roots: &[u32],
    uniform_like: &dyn Fn(u32) -> bool,
) {
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
            if prog[i].mnem != Op::Fadd || prog[i].srcs.len() != 2 {
                continue;
            }
            for k in 0..2 {
                let m = prog[i].srcs[k];
                let Some(&mi) = def_idx.get(&m) else { continue };
                if mi < i
                    && !remove[mi]
                    && prog[mi].mnem == Op::Fmul
                    && prog[mi].srcs.len() == 2
                    && uses.get(&m) == Some(&1)
                {
                    let (a, b) = (prog[mi].srcs[0], prog[mi].srcs[1]);
                    // The fused multiply reads its operands at the add.
                    if prog[mi + 1..i].iter().any(|x| x.dst == a || x.dst == b || x.mnem == Mnem::Label || x.mnem == Op::Brnz) {
                        continue;
                    }
                    let (sa, sb) = (prog[mi].swz[0], prog[mi].swz[1]);
                    let c = prog[i].srcs[1 - k];
                    let sc = prog[i].swz[1 - k];
                    let n_uniform = [a, b, c].iter().filter(|&&s| uniform_like(s)).count();
                    if n_uniform > 1 {
                        continue;
                    }
                    prog[i].mnem = Op::Fmadd.into();
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
        debug!(
            "borgc: FMADD fusion — {} FMUL+FADD pair(s) fused, {} instr(s)",
            pre_fuse - prog.len(),
            prog.len()
        );
    }
}

pub(crate) fn classify_uniform(prog: &[BorgInstr], roots: &HashSet<u32>) -> HashSet<u32> {
    let mut uniform: HashSet<u32> = roots.clone();
    for i in prog {
        if i.mnem == Op::Ddx || i.mnem == Op::Ddy {
            uniform.insert(i.dst);
        } else if !i.srcs.is_empty() && i.srcs.iter().all(|s| uniform.contains(s)) {
            uniform.insert(i.dst);
        }
    }
    uniform
}
