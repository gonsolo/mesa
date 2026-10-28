// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::BorgInstr;

pub(crate) fn regalloc(
    prog: &[BorgInstr],
    forced: &std::collections::HashMap<u32, u8>,
    extra_reserved: &[u8],
    soft: &[u8],
    block_len: &std::collections::HashMap<u32, u8>,
) -> std::collections::HashMap<u32, u8> {
    use std::collections::HashMap;

    const NUM_GPRS: u8 = 30; // r0..r29; r30/r31 reserved for coordinates

    let ranges = crate::block::live_ranges(prog);
    let def_at: HashMap<u32, usize> = ranges.iter().map(|(&v, &(d, _))| (v, d)).collect();
    let last_use: HashMap<u32, usize> = ranges.iter().map(|(&v, &(_, l))| (v, l)).collect();

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
    // r0 is x0 for ISA v2 instructions: only pinned values live there.
    let mut free: Vec<u8> = (1..NUM_GPRS).rev().filter(|r| !reserved.contains(r)).collect();
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

    debug!(
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
    use super::regalloc;
    use crate::opcodes::Op;
    use crate::BorgInstr;
    use std::collections::HashMap;

    fn chain(n: u32) -> Vec<BorgInstr> {
        (1..=n)
            .map(|d| BorgInstr { mnem: Op::Fadd.into(), dst: d, srcs: vec![d - 1, d.saturating_sub(2)], swz: vec![0, 0], imm: 0 })
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
