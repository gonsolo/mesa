// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT
use compiler::bindings::*;
use compiler::nir::AsDef;
use std::collections::HashMap;

/// NIR values that share one register: a phi with the phis and sources it takes without a copy.
/// Liveness is per lane (NIR's CFG): a masked lane keeps its register, so only the lanes that
/// write a value can clobber another.
pub(crate) struct Webs {
    pub(crate) webs: Vec<Vec<*mut nir_def>>,
    pub(crate) of: HashMap<u32, usize>,
}

impl Webs {
    pub(crate) unsafe fn new(imp: *mut nir_function_impl) -> Webs {
        let mut w = Webs { webs: vec![], of: HashMap::new() };
        let mut phis = vec![];
        for block in (*imp).iter_blocks() {
            for phi in block.iter_instr_list().map_while(|i| i.as_phi()) {
                w.add(&phi.def as *const nir_def as *mut nir_def);
                phis.push(phi);
            }
        }
        if !phis.is_empty() {
            nir_metadata_require(imp, nir_metadata_live_defs | nir_metadata_instr_index);
        }
        for phi in phis {
            for s in phi.iter_srcs() {
                let d = s.src.as_def() as *const nir_def as *mut nir_def;
                if matches!((*d).parent_instr().type_, nir_instr_type_load_const | nir_instr_type_undef)
                    || (*d).num_components != phi.def.num_components
                {
                    continue;
                }
                let (a, b) = (w.of[&phi.def.index], w.add(d));
                if a != b && w.can_merge(a, b) {
                    let moved = std::mem::take(&mut w.webs[b]);
                    for &m in &moved {
                        w.of.insert((*m).index, a);
                    }
                    w.webs[a].extend(moved);
                }
            }
        }
        w
    }

    fn add(&mut self, d: *mut nir_def) -> usize {
        let i = unsafe { (*d).index };
        *self.of.entry(i).or_insert_with(|| {
            self.webs.push(vec![d]);
            self.webs.len() - 1
        })
    }

    /// No member is live where another is written: at its definition, or by a phi's copy from a
    /// source outside the web at the end of a predecessor.
    unsafe fn can_merge(&self, a: usize, b: usize) -> bool {
        let all: Vec<*mut nir_def> = self.webs[a].iter().chain(&self.webs[b]).copied().collect();
        if self.webs[a].iter().any(|&x| self.webs[b].iter().any(|&y| nir_defs_interfere(x, y))) {
            return false;
        }
        all.iter().all(|&q| {
            let Some(phi) = (*q).parent_instr().as_phi() else { return true };
            phi.iter_srcs().all(|s| {
                let t = s.src.as_def() as *const nir_def as *mut nir_def;
                let pred = s.pred() as *const nir_block as *mut nir_block;
                all.contains(&t) || all.iter().all(|&m| m == q || !u_sparse_bitset_test(&mut (*pred).live_out, (*m).index))
            })
        })
    }
}
