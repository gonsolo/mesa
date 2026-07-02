// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// Small NIR-facing instruction-selection helpers used by the main selection
// walk in lib.rs::borgc_compile_nir.

use compiler::bindings::*;

/// Resolve a (def, component) through the vec/mov construction map to the true
/// underlying scalar producer + component. Free function (not a closure) so the
/// selection walk can resolve and mutate `vec_map` in the same loop.
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
pub(crate) fn borg_isel(op: nir_op) -> Option<&'static str> {
    match op {
        nir_op_fadd => Some("FADD"),
        nir_op_fmul => Some("FMUL"),
        nir_op_ffma | nir_op_ffma_weak => Some("FMADD"),
        nir_op_fneg => Some("FNEG"),
        // Fragment transcendentals — now real Borg HW ops. (ddx/ddy are NIR
        // intrinsics, handled in the intrinsic selection path, not here.)
        nir_op_frcp => Some("FRCP"),
        nir_op_frsq => Some("FRSQ"),
        // Integer ops (the Borg core now has these — enables standard addressing).
        nir_op_iadd => Some("IADD"),
        nir_op_ishl => Some("ISHL"),
        nir_op_ishr | nir_op_ushr => Some("ISHR"),
        nir_op_imul => Some("IMUL"),
        nir_op_i2f16 | nir_op_i2f32 | nir_op_u2f16 | nir_op_u2f32 => Some("I2F"),
        nir_op_f2i16 | nir_op_f2i32 | nir_op_f2u16 | nir_op_f2u32 => Some("F2I"),
        // Moves / vector (de)construction become register copies, no ISA op.
        nir_op_mov | nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => Some("mov"),
        // Integer↔int bit-size conversions are no-ops at 16-bit (copy).
        nir_op_i2i16 | nir_op_i2i32 | nir_op_u2u16 | nir_op_u2u32 => Some("mov"),
        _ => None,
    }
}
