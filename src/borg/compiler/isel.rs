// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::opcodes::Op;
use crate::Mnem;
use compiler::bindings::*;

/// The value component c of d stands for, following vec_map's copies to their end. A chain is
/// at most as long as the map; a cycle is a compiler bug.
pub(crate) fn resolve_vm(
    vec_map: &std::collections::HashMap<u32, Vec<(u32, u8)>>,
    d0: u32,
    c0: u8,
) -> (u32, u8) {
    let (mut d, mut c) = (d0, c0);
    for _ in 0..=vec_map.len() {
        match vec_map.get(&d) {
            Some(v) if (c as usize) < v.len() && v[c as usize] != (d, c) => (d, c) = v[c as usize],
            _ => return (d, c),
        }
    }
    panic!("borgc: vec_map has a cycle through {d0}");
}

/// Does this NIR ALU op map directly onto a Borg ISA opcode?
pub(crate) fn borg_isel(op: nir_op) -> Option<Mnem> {
    Some(match op {
        nir_op_fadd => Op::Fadd.into(),
        nir_op_fmul => Op::Fmul.into(),
        nir_op_ffma | nir_op_ffma_weak => Op::Fmadd.into(),
        nir_op_fneg => Op::Fneg.into(),
        nir_op_frcp => Op::Frcp.into(),
        nir_op_frsq => Op::Frsq.into(),
        nir_op_fexp2 => Op::Fexp2.into(),
        nir_op_flog2 => Op::Flog2.into(),
        // Integer ops (the Borg core now has these — enables standard addressing).
        nir_op_iadd => Op::Iadd.into(),
        nir_op_ishl => Op::Ishl.into(),
        nir_op_ishr => Op::Ishr.into(),
        nir_op_ushr => Op::Isrl.into(),
        nir_op_isub => Op::Isub.into(),
        nir_op_iand => Op::Iand.into(),
        nir_op_ior => Op::Ior.into(),
        nir_op_ixor => Op::Ixor.into(),
        nir_op_imul => Op::Imul.into(),
        nir_op_i2f16 | nir_op_i2f32 | nir_op_u2f16 | nir_op_u2f32 => Op::I2f.into(),
        nir_op_f2i16 | nir_op_f2i32 => Op::F2i.into(),
        nir_op_f2u16 | nir_op_f2u32 => Op::FcvtWuS.into(),
        // Moves / vector (de)construction become register copies, no ISA op.
        nir_op_mov | nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => Mnem::Mov,
        nir_op_i2i16 | nir_op_i2i32 | nir_op_u2u16 | nir_op_u2u32 => Mnem::Mov,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_vm;
    use std::collections::HashMap;

    #[test]
    fn follows_a_chain_longer_than_64_copies() {
        let m: HashMap<u32, Vec<(u32, u8)>> = (0..100).map(|d| (d, vec![(d + 1, 0)])).collect();
        assert_eq!(resolve_vm(&m, 0, 0), (100, 0));
    }

    #[test]
    fn a_self_copy_ends_the_chain() {
        let m: HashMap<u32, Vec<(u32, u8)>> = [(5, vec![(5, 0), (7, 2)])].into_iter().collect();
        assert_eq!((resolve_vm(&m, 5, 0), resolve_vm(&m, 5, 1)), ((5, 0), (7, 2)));
    }
}
