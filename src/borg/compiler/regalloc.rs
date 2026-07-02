// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// Linear-scan register allocation over the selected virtual-register program.

use crate::BorgInstr;

/// Linear-scan register allocation (Poletto & Sarkar) for the values our
/// instruction selection defines (the ALU results). Each is live from its def
/// to its last use; we assign physical GPRs r0..r29 (r30/r31 are the special
/// coordinate registers), reusing a register once its value dies. Reports the
/// allocation, the peak register pressure, and any spills. Values that only
/// appear as sources (shader inputs) are not allocated here — they map to
/// uniforms/attributes when the I/O lowering lands.
pub(crate) fn regalloc(
    prog: &[BorgInstr],
    forced: &std::collections::HashMap<u32, u8>,
    extra_reserved: &[u8],
) -> std::collections::HashMap<u32, u8> {
    use std::collections::HashMap;

    const NUM_GPRS: u8 = 30; // r0..r29; r30/r31 reserved for coordinates

    // Def position (first) and last-use position for every value.
    let mut def_at: HashMap<u32, usize> = HashMap::new();
    let mut last_use: HashMap<u32, usize> = HashMap::new();
    for (i, instr) in prog.iter().enumerate() {
        def_at.entry(instr.dst).or_insert(i);
        last_use.insert(instr.dst, i);
        for &s in &instr.srcs {
            last_use.insert(s, i);
        }
    }

    // Allocate only values we define (ALU results), in def order.
    let mut defs: Vec<u32> = def_at.keys().copied().collect();
    defs.sort_by_key(|v| def_at[v]);

    // Reserve the pre-colored output regs (gl_Position r0..r3) and r4 (the
    // perspective-divide scratch) from the general pool — the epilogue owns them.
    let reserved: std::collections::HashSet<u8> = forced
        .values()
        .copied()
        .chain([4])
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
            active.push((end, r));
            peak = peak.max(active.len());
            continue;
        }

        // Expire intervals that ended before this value starts. Reserved output
        // registers are never returned to the pool — the epilogue still reads them
        // after the body, beyond the body-local last-use we computed here.
        let mut keep = Vec::with_capacity(active.len());
        for &(e, r) in &active {
            if e < start && !reserved.contains(&r) {
                free.push(r);
            } else if e >= start {
                keep.push((e, r));
            }
        }
        active = keep;

        match free.pop() {
            Some(r) => {
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
