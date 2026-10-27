// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::BorgInstr;
use std::collections::{HashMap, HashSet};
use crate::opcodes::Op;
use crate::Mnem;

pub(crate) fn dce(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let pre_dce = prog.len();
    let mut live: std::collections::HashSet<u32> = out_roots.iter().copied().collect();

    let side_effecting = crate::has_effect;
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
            if crate::has_effect(&instr) || matches!(instr.mnem, Mnem::Op(Op::Tex | Op::Texd | Op::Fattr)) {
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

/// Issue slots from an instruction to one that reads its result (B9 pipeline, with the bypass).
const LATENCY: usize = 5;

/// Nothing moves across these: the core drains its pipeline for them, or they change control flow
/// or state the instructions around them see.
fn barrier(i: &BorgInstr) -> bool {
    crate::has_effect(i)
        || !matches!(i.mnem, Mnem::Op(_) | Mnem::Mov | Mnem::Tld1)
        || matches!(i.mnem, Mnem::Op(Op::Tex | Op::Texa | Op::Texd | Op::Texad | Op::Texs | Op::Fattr | Op::FattrV2
            | Op::Sout | Op::SoutV2 | Op::Ztest | Op::ZtestV2 | Op::Fexp2 | Op::Flog2 | Op::Load | Op::Store
            | Op::Lb | Op::Lh | Op::Lw | Op::Lbu | Op::Lhu | Op::Sw | Op::Jal | Op::Jmp | Op::Brz | Op::Brnz
            | Op::Beq | Op::Bne | Op::Blt | Op::Bge | Op::Bltu | Op::Bgeu | Op::Expush | Op::Exelse | Op::Expop
            | Op::Exany | Op::ExpushV2 | Op::ExelseV2 | Op::ExpopV2 | Op::ExanyV2 | Op::Barrier | Op::BarrierV2
            | Op::Csrrw | Op::Smask | Op::Attidx))
}

/// Reorders each run between barriers so that an instruction issues LATENCY slots after the ones
/// it reads, without more values live at any point than the program's peak before.
pub(crate) fn schedule_for_latency(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let mut uses: HashMap<u32, usize> = HashMap::new();
    for &s in prog.iter().flat_map(|i| i.srcs.iter()).chain(out_roots) {
        *uses.entry(s).or_insert(0) += 1;
    }
    // Live values before each instruction, in the order as it is.
    let defined: HashSet<u32> = prog.iter().map(|i| i.dst).collect();
    let mut left = uses.clone();
    let mut live = 0usize;
    let mut live_before = Vec::with_capacity(prog.len());
    for i in prog.iter() {
        live_before.push(live);
        live = live.saturating_add_signed(live_delta(i, &left, &uses, &defined));
        for s in &i.srcs {
            *left.get_mut(s).unwrap() -= 1;
        }
    }
    let cap = live_before.iter().copied().max().unwrap_or(0).max(live);

    let mut out = Vec::with_capacity(prog.len());
    let mut run = Vec::new();
    let mut start = 0;
    let mut left = uses.clone();
    for (k, i) in std::mem::take(prog).into_iter().enumerate() {
        if barrier(&i) {
            out.extend(latency_run(std::mem::take(&mut run), &mut left, &uses, &defined, cap - live_before[start]));
            for s in &i.srcs {
                *left.get_mut(s).unwrap() -= 1;
            }
            out.push(i);
            start = k + 1;
        } else {
            run.push(i);
        }
    }
    let room = cap - live_before.get(start).copied().unwrap_or(live);
    out.extend(latency_run(run, &mut left, &uses, &defined, room));
    *prog = out;
}

/// The change in live values when `i` issues: its result if anything reads it, less the sources
/// it reads last.
fn live_delta(i: &BorgInstr, left: &HashMap<u32, usize>, uses: &HashMap<u32, usize>, defined: &HashSet<u32>) -> isize {
    let mut d = isize::from(uses.get(&i.dst).is_some_and(|&n| n > 0));
    let mut seen = HashSet::new();
    for s in &i.srcs {
        if defined.contains(s) && seen.insert(*s) && left[s] == i.srcs.iter().filter(|&x| x == s).count() {
            d -= 1;
        }
    }
    d
}

fn latency_run(run: Vec<BorgInstr>, left: &mut HashMap<u32, usize>, uses: &HashMap<u32, usize>,
               defined: &HashSet<u32>, room: usize) -> Vec<BorgInstr> {
    let n = run.len();
    let take = |left: &mut HashMap<u32, usize>, run: &[BorgInstr]| {
        for s in run.iter().flat_map(|i| i.srcs.iter()) {
            *left.get_mut(s).unwrap() -= 1;
        }
    };
    let here: HashMap<u32, usize> = run.iter().enumerate().map(|(k, i)| (i.dst, k)).collect();
    let mut seen = HashSet::new();
    if n <= 2 || here.len() != n
        || run.iter().any(|i| i.srcs.iter().any(|s| here.contains_key(s) && !seen.contains(s)) || !seen.insert(i.dst)) {
        take(left, &run);
        return run;
    }
    let preds: Vec<Vec<usize>> = run.iter().map(|i| i.srcs.iter().filter_map(|s| here.get(s).copied()).collect()).collect();
    let mut height = vec![1usize; n];
    for k in (0..n).rev() {
        for &p in &preds[k] {
            height[p] = height[p].max(height[k] + LATENCY);
        }
    }
    // Issue slot of each instruction in the order given by `order`.
    let slots = |order: &[usize]| {
        let mut at = vec![0usize; n];
        let mut next = 0;
        for &k in order {
            at[k] = preds[k].iter().map(|&p| at[p] + LATENCY).max().unwrap_or(0).max(next);
            next = at[k] + 1;
        }
        next
    };

    let mut l = left.clone();
    let (mut order, mut done, mut at) = (Vec::with_capacity(n), vec![false; n], vec![0usize; n]);
    let (mut live, mut next) = (0isize, 0usize);
    while order.len() < n {
        let ready = (0..n).filter(|&k| !done[k] && preds[k].iter().all(|&p| done[p]));
        let best = ready
            .map(|k| {
                let d = live_delta(&run[k], &l, uses, defined);
                let issue = preds[k].iter().map(|&p| at[p] + LATENCY).max().unwrap_or(0).max(next);
                (live + d > room as isize, issue, std::cmp::Reverse(height[k]), k, d)
            })
            .min()
            .expect("an SSA run always has a ready instruction");
        let (over, issue, _, k, d) = best;
        if over {
            take(left, &run);
            return run;
        }
        (done[k], at[k], next, live) = (true, issue, issue + 1, live + d);
        for s in &run[k].srcs {
            *l.get_mut(s).unwrap() -= 1;
        }
        order.push(k);
    }
    take(left, &run);
    if slots(&order) >= slots(&(0..n).collect::<Vec<_>>()) {
        return run;
    }
    let mut slots_of: Vec<Option<BorgInstr>> = run.into_iter().map(Some).collect();
    order.into_iter().map(|k| slots_of[k].take().unwrap()).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn op(m: Op, dst: u32, srcs: &[u32]) -> BorgInstr {
        BorgInstr { mnem: m.into(), dst, srcs: srcs.to_vec(), swz: vec![0; srcs.len()], imm: 0 }
    }
    fn dsts(p: &[BorgInstr]) -> Vec<u32> {
        p.iter().map(|i| i.dst).collect()
    }

    #[test]
    fn interleaves_two_dependent_chains() {
        // 10 -> 11 -> 12 and 20 -> 21 -> 22, from inputs 1 and 2.
        let mut p = vec![op(Op::Fadd, 10, &[1, 1]), op(Op::Fadd, 11, &[10, 1]), op(Op::Fadd, 12, &[11, 1]),
                         op(Op::Fadd, 20, &[2, 2]), op(Op::Fadd, 21, &[20, 2]), op(Op::Fadd, 22, &[21, 2])];
        schedule_for_latency(&mut p, &[12, 22]);
        assert_eq!(dsts(&p), [10, 20, 11, 21, 12, 22]);
    }

    #[test]
    fn nothing_moves_across_a_barrier() {
        let mut p = vec![op(Op::Fadd, 10, &[1, 1]), op(Op::Fadd, 11, &[10, 1]), op(Op::Tex, 30, &[11, 11, 11]),
                         op(Op::Fadd, 20, &[2, 2]), op(Op::Fadd, 21, &[20, 2]), op(Op::Fadd, 22, &[1, 2])];
        schedule_for_latency(&mut p, &[21, 22, 30]);
        assert_eq!(dsts(&p)[..3], [10, 11, 30]);
    }

    #[test]
    fn keeps_the_order_when_it_would_need_more_registers() {
        // One value live at a time as given; interleaving the chains would hold two.
        let mut p = vec![op(Op::Fadd, 10, &[1, 1]), op(Op::Fadd, 11, &[10, 1]), op(Op::Fadd, 12, &[11, 1]),
                         op(Op::Fadd, 20, &[2, 2]), op(Op::Fadd, 21, &[20, 2]), op(Op::Fadd, 22, &[21, 2])];
        let before = dsts(&p);
        schedule_for_latency(&mut p, &[]);
        assert_eq!(dsts(&p), before);
    }
}
