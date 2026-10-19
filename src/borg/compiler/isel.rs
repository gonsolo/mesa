// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::opcodes::Op;
use crate::Mnem;
use compiler::bindings::*;

pub(crate) fn resolve_vm(
    vec_map: &std::collections::HashMap<u32, Vec<(u32, u8)>>,
    d0: u32,
    c0: u8,
) -> (u32, u8) {
    let (mut d, mut c) = (d0, c0);
    for _ in 0..64 {
        match vec_map.get(&d) {
            Some(v) if (c as usize) < v.len() => {
                let (nd, nc) = v[c as usize];
                d = nd;
                c = nc;
            }
            _ => break,
        }
    }
    (d, c)
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
        nir_op_f2i16 | nir_op_f2i32 | nir_op_f2u16 | nir_op_f2u32 => Op::F2i.into(),
        // Moves / vector (de)construction become register copies, no ISA op.
        nir_op_mov | nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => Mnem::Mov,
        nir_op_i2i16 | nir_op_i2i32 | nir_op_u2u16 | nir_op_u2u32 => Mnem::Mov,
        _ => return None,
    })
}
