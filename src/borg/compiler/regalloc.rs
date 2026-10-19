// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::BorgInstr;
use crate::opcodes::Op;
use crate::Mnem;

/// A value live into a loop or carried round it lives for the whole loop, and so does one defined
/// in it and live after it: a lane that left early keeps its last trip's value.
pub(crate) fn extend_for_loops(
    prog: &[BorgInstr],
    def_at: &mut std::collections::HashMap<u32, usize>,
    last: &mut std::collections::HashMap<u32, usize>,
) {
    let mut heads = std::collections::HashMap::new();
    let mut spans = Vec::new();
    for (k, i) in prog.iter().enumerate() {
        match i.mnem {
            Mnem::Label => {
                heads.insert(i.swz[0], k);
            }
            Mnem::Op(Op::Brnz) => spans.push((heads[&i.swz[1]], k)),
            _ => {}
        }
    }
    let mut changed = !spans.is_empty();
    while changed {
        changed = false;
        for &(s, e) in &spans {
            let mut first_use = std::collections::HashMap::new();
            for (k, i) in prog.iter().enumerate().take(e + 1).skip(s) {
                for &v in &i.srcs {
                    first_use.entry(v).or_insert(k);
                }
            }
            for (v, d) in def_at.iter_mut() {
                let Some(l) = last.get_mut(v) else { continue };
                let inside = (s..=e).contains(d);
                let carried = inside && first_use.get(v).is_some_and(|&u| u <= *d);
                if (*d < s && *l >= s || carried) && *l < e {
                    *l = e;
                    changed = true;
                }
                if (carried || inside && *l > e) && *d > s {
                    *d = s;
                    changed = true;
                }
            }
        }
    }
}

pub(crate) fn regalloc(
    prog: &[BorgInstr],
    forced: &std::collections::HashMap<u32, u8>,
    extra_reserved: &[u8],
    soft: &[u8],
    block_len: &std::collections::HashMap<u32, u8>,
) -> std::collections::HashMap<u32, u8> {
    use std::collections::HashMap;

    const NUM_GPRS: u8 = 30; // r0..r29; r30/r31 reserved for coordinates

    // Def position (first) and last-use position for every value.
    let mut def_at: HashMap<u32, usize> = HashMap::new();
    let mut last_use: HashMap<u32, usize> = HashMap::new();
    for (i, instr) in prog.iter().enumerate() {
        if instr.dst != crate::NO_DST {
            def_at.entry(instr.dst).or_insert(i);
            last_use.insert(instr.dst, i);
        }
        for &s in &instr.srcs {
            last_use.insert(s, i);
        }
    }

    extend_for_loops(prog, &mut def_at, &mut last_use);

    // Allocate only values we define (ALU results), in def order.
    let mut defs: Vec<u32> = def_at.keys().copied().collect();
    defs.sort_by_key(|v| def_at[v]);

    let mut busy: HashMap<u8, Vec<(usize, usize)>> = HashMap::new();
    for (&v, &r) in forced {
        if !soft.contains(&r) {
            continue;
        }
        let Some(&st) = def_at.get(&v) else { continue };
        let en = if block_len.get(&v).copied().unwrap_or(1) > 1 { last_use.get(&v).copied().unwrap_or(st) } else { prog.len() };
        for k in 0..block_len.get(&v).copied().unwrap_or(1) {
            busy.entry(r + k).or_default().push((st, en));
        }
    }
    let reserved: std::collections::HashSet<u8> = forced
        .values()
        .copied()
        .filter(|r| !soft.contains(r))
        .chain(extra_reserved.iter().copied())
        .collect();
    let mut free: Vec<u8> = (0..NUM_GPRS).rev().filter(|r| !reserved.contains(r)).collect();
    let mut active: Vec<(usize, u8)> = Vec::new(); // (live_end, phys_reg)
    let mut alloc: HashMap<u32, u8> = HashMap::new();
    let mut spills = 0usize;
    let mut peak = 0usize;

    for v in defs {
        let start = def_at[&v];
        let end = *last_use.get(&v).unwrap_or(&start);

        // Pre-colored value (gl_Position component): pin to its output register.
        if let Some(&r) = forced.get(&v) {
            alloc.insert(v, r);
            if !soft.contains(&r) {
                active.push((end, r));
                peak = peak.max(active.len());
            }
            continue;
        }

        let mut keep = Vec::with_capacity(active.len());
        for &(e, r) in &active {
            if e < start && !reserved.contains(&r) {
                free.push(r);
            } else if e >= start {
                keep.push((e, r));
            }
        }
        active = keep;

        let fits = |r: u8| busy.get(&r).map_or(true, |l| l.iter().all(|&(a, b)| end < a || start > b));
        match free.iter().rposition(|&r| fits(r)) {
            Some(i) => {
                let r = free.remove(i);
                alloc.insert(v, r);
                active.push((end, r));
            }
            None => spills += 1,
        }
        peak = peak.max(active.len());
    }

    eprintln!(
        "borgc: regalloc — {} values → r0..r{}, peak pressure {}, spills {}",
        alloc.len(),
        NUM_GPRS - 1,
        peak,
        spills
    );
    alloc
}

#[cfg(test)]
mod tests {
    use super::{extend_for_loops, regalloc};
    use crate::opcodes::Op;
    use crate::{BorgInstr, Mnem, NO_DST};
    use std::collections::HashMap;

    fn op(mnem: impl Into<Mnem>, dst: u32, srcs: &[u32]) -> BorgInstr {
        BorgInstr { mnem: mnem.into(), dst, srcs: srcs.to_vec(), swz: vec![0; srcs.len()] }
    }

    #[test]
    fn loop_values_live_for_the_whole_loop() {
        // 1 = before; LABEL; 3 = 2 + 1 (2 carried: read, then written); 4 defined, used after.
        let prog = vec![
            op(Op::Ior, 1, &[9, 9]),
            op(Op::Ior, 2, &[9, 9]),
            BorgInstr { mnem: Mnem::Label, dst: NO_DST, srcs: vec![], swz: vec![0] },
            op(Op::Iadd, 3, &[2, 1]),
            op(Op::Ior, 2, &[3, 3]),
            op(Op::Ior, 4, &[3, 3]),
            op(Op::Iadd, 5, &[1, 1]),
            BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![5], swz: vec![0, 0] },
            op(Op::Iadd, 6, &[4, 4]),
        ];
        let (mut d, mut l) = (HashMap::new(), HashMap::new());
        for (k, i) in prog.iter().enumerate() {
            d.entry(i.dst).or_insert(k);
            for s in &i.srcs {
                l.insert(*s, k);
            }
        }
        extend_for_loops(&prog, &mut d, &mut l);
        assert_eq!((d[&1], l[&1]), (0, 7), "live into the loop");
        assert_eq!((d[&2], l[&2]), (1, 7), "carried round it");
        assert_eq!((d[&4], l[&4]), (2, 8), "defined in it, live after it");
        assert_eq!((d[&3], l[&3]), (3, 5), "a value of one trip");
    }

    fn chain(n: u32) -> Vec<BorgInstr> {
        (1..=n)
            .map(|d| BorgInstr { mnem: Op::Fadd.into(), dst: d, srcs: vec![d - 1, d.saturating_sub(2)], swz: vec![0, 0] })
            .collect()
    }

    #[test]
    fn every_value_gets_a_register_and_overlaps_differ() {
        let prog = chain(60);
        let a = regalloc(&prog, &HashMap::new(), &[], &[], &HashMap::new());
        assert_eq!(a.len(), 60);
        for d in 1..=58u32 {
            assert_ne!(a[&d], a[&(d + 1)]);
            assert_ne!(a[&d], a[&(d + 2)]);
        }
    }

    #[test]
    fn forced_registers_are_honoured_and_reserved_ones_avoided() {
        let prog = chain(20);
        let forced = HashMap::from([(5u32, 9u8)]);
        let a = regalloc(&prog, &forced, &[0, 1, 2, 3], &[], &HashMap::new());
        assert_eq!(a[&5], 9);
        assert!(a.iter().all(|(&v, &r)| v == 5 || ![0, 1, 2, 3].contains(&r)));
    }
}
