// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// IR-to-IR optimization passes over the selected Vec<BorgInstr> program, run
// after instruction selection and before register allocation.

use crate::BorgInstr;
use std::collections::{HashMap, HashSet};

/// Dead-code elimination. Because every load_ubo is pinned to a fixed uniform,
/// the UBO byte-offset arithmetic (iadd/ishl from gl_VertexIndex, descriptor
/// index math) feeds only the dropped offsets and is dead. Mark-and-sweep from
/// the store_output roots back through the def→use chain to fixpoint.
pub(crate) fn dce(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let pre_dce = prog.len();
    let mut live: std::collections::HashSet<u32> = out_roots.iter().copied().collect();

    // An instruction with no destination is a SIDE EFFECT, not dead code. The
    // execution-mask ops and STORE write no register, so nothing can ever make
    // their dst reachable from an output root -- a liveness-only rule deletes
    // every one of them. That silently removed EXPUSH/EXELSE/EXPOP from a
    // shader whose `if` had otherwise lowered correctly, leaving both arms
    // running unconditionally: the exact miscompile the mask exists to fix.
    // Their sources are live too, or the condition feeding EXPUSH would be
    // dropped in turn.
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
        eprintln!(
            "borgc: DCE — dropped {} dead instr(s) (UBO address math), {} live",
            pre_dce - prog.len(),
            prog.len()
        );
    }
}

/// List-schedule `prog` to reduce peak register pressure, without changing
/// what it computes. Runs after DCE and FMADD fusion, right before
/// regalloc, and reorders only WITHIN maximal runs between side-effecting
/// instructions (STORE, EXPUSH/EXELSE/EXPOP, branches -- anything with no
/// `dst`): those keep their exact original relative order and act as full
/// barriers nothing crosses, so their own semantics never change. Within a
/// run, an instruction only ever becomes eligible once every source it
/// reads is already available (defined by an earlier run, or already
/// scheduled within this one) -- a true dependency is never violated, only
/// reordered around. Among eligible instructions, greedily picks whichever
/// one leaves the FEWEST values live immediately afterward: consuming a
/// value at its last remaining use frees a register right away, so
/// finishing a chain that is already live is always preferred over
/// starting a new, longer-lived one (typically a LOAD).
///
/// This exists because a straightforward source expression like
/// `mat4 * vec4` lowers, in NIR's own instruction order, to "load all 16
/// matrix words, THEN multiply" -- correct, but needlessly holding all 16
/// (plus the position vector) live at once when loading a column and
/// consuming it immediately would do. Caught in exactly that shape:
/// cube.vert's compiled MVP multiply needed a peak of 21 live registers
/// with nothing else touching them at that point, tight enough that
/// reserving even 3 more (for the paired fragment shader's own pinned
/// constants, a separate bug) caused real spills.
pub(crate) fn schedule_for_pressure(prog: &mut Vec<BorgInstr>, out_roots: &[u32]) {
    let out_roots: HashSet<u32> = out_roots.iter().copied().collect();

    // Split into maximal side-effect-free runs, each followed by the
    // side-effecting instruction (if any) that ended it.
    let mut runs: Vec<(Vec<BorgInstr>, Option<BorgInstr>)> = Vec::new();
    {
        let mut cur: Vec<BorgInstr> = Vec::new();
        for instr in prog.drain(..) {
            if instr.dst == crate::NO_DST {
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

    // remaining_uses[v]: how many not-yet-scheduled reads of v are left --
    // every source occurrence in this run, plus one more if v is also
    // needed after it (the trailing side-effecting instruction, a later
    // run, or a shader output). Reaching 0 means v can be freed.
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

    // pending_deps[i]: how many of instruction i's sources are defined
    // WITHIN this run and not yet scheduled. Only these gate readiness --
    // a source from an earlier run is available from the start.
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

    // urgency[i]: the original index of the earliest instruction that
    // ultimately needs instr[i]'s result, propagated backward through
    // chains -- an address-chain LOAD whose value only the very first
    // matmul term needs is urgent even though its own defining chain sits
    // right next to three OTHER, unrelated chains whose results nothing
    // needs until much later. Without this, every pure-load instruction
    // ties on (produces, frees) alone (delta = +1, nothing to free yet),
    // and the tie-break falls back to original NIR order -- i.e. no
    // reordering happens at all, because nothing distinguishes "this
    // column's load is needed by the very next multiply" from "this
    // column's load is needed sixteen instructions from now."
    let mut urgency: Vec<usize> = vec![0; n];
    for i in (0..n).rev() {
        urgency[i] = match consumers.get(&run[i].dst) {
            Some(cs) if !cs.is_empty() => cs.iter().map(|&c| urgency[c]).min().unwrap(),
            _ => i, // no consumer within this run (feeds an out_root, or dead): its own position is the best guess
        };
    }

    let mut slots: Vec<Option<BorgInstr>> = run.into_iter().map(Some).collect();
    let mut ready: Vec<usize> = (0..n).filter(|&i| pending_deps[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    // Depth-first tie-break: once several equally-urgent chains are all
    // simultaneously ready (e.g. four independent MVP-column address
    // chains), keep pulling on whichever one was just advanced instead of
    // round-robining between them -- finishing one chain fully before
    // starting the next keeps its own scratch registers alive for less
    // total time than interleaving all four at once would.
    let mut last_dst: Option<u32> = None;

    for _ in 0..n {
        let best = *ready
            .iter()
            .min_by_key(|&&i| {
                let instr = slots[i].as_ref().unwrap();
                // 1 if this instruction's own result survives past it, 0 if
                // this was its last remaining use too (dead on arrival).
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

/// Peephole: re-fuse FMUL+FADD into FMADD. nir_lower_alu_to_scalar split the
/// source ffma into a multiply and an add; fuse FADD(d, m, c) where m=FMUL(a,b)
/// is used only here back into FMADD(d, a*b + c). Halves the matrix-multiply op
/// count — both faster and necessary to fit SPIRB_MAX_INSTRS (72).
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

/// Which values are "primitive-uniform" — constant for the whole triangle,
/// safe to compute once in the setup stage instead of once per fragment.
///
/// Borg-specific axiom: DDX/DDY results are ALWAYS uniform. Borg has no
/// fixed-function interpolator — every fragment varying is computed in the
/// fragment shader itself as a barycentric-weighted sum of per-vertex uniform
/// data, i.e. every varying is (by construction) an affine function of screen
/// position within one triangle. The derivative of an affine function over
/// that triangle is a constant, regardless of which varying is being
/// differentiated — so this holds for any Borg fragment shader that uses
/// dFdx/dFdy, not just one in particular.
///
/// From that axiom, uniformity propagates the ordinary way: a value is
/// uniform iff every one of its operands is uniform — either a `root` (a
/// per-vertex-staged uniform read or a true shader constant, classified by
/// the caller from its knowledge of the `Ubo` map) or another already-uniform
/// result. `prog` must be in forward topological order (true for the
/// selection walk's own output — DCE only removes entries and FMADD fusion
/// only merges adjacent producer/consumer pairs in place, so the order
/// invariant survives both), so one linear pass suffices — no fixed point
/// needed, unlike the backward DCE walk above.
pub(crate) fn classify_uniform(prog: &[BorgInstr], roots: &HashSet<u32>) -> HashSet<u32> {
    let mut uniform: HashSet<u32> = roots.clone();
    for i in prog {
        if i.mnem == "DDX" || i.mnem == "DDY" {
            uniform.insert(i.dst);
        } else if !i.srcs.is_empty() && i.srcs.iter().all(|s| uniform.contains(s)) {
            uniform.insert(i.dst);
        }
    }
    uniform
}
