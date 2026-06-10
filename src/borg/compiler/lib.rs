// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// borgc — the Borg GPU shader compiler (Rust), called from the borgvk Vulkan
// driver. It lowers Mesa NIR to Borg ISA (the 7-op FP16 ISA: FADD/FMUL/FMADD/
// FNEG/FSTEP/FRCP/FTEX) and emits the .borg blob the firmware loads. Modeled on
// Mesa's NAK (src/nouveau/compiler/nak).
//
// Status: SPIR-V→NIR→borgc is live. This pass classifies every NIR instruction
// against the Borg ISA — a coverage report that drives instruction selection.
// The arithmetic core (fmul/ffma → FMUL/FMADD) selects directly; the open work
// is lowering the UBO/I-O intrinsics to the firmware's uniform model and real
// register allocation + .borg emission.

#![allow(non_upper_case_globals)]

use compiler::bindings::*;

/// Self-test: confirms the Rust crate is linked and callable across the FFI.
#[no_mangle]
pub extern "C" fn borgc_selftest() -> u32 {
    0xB0_06
}

/// Does this NIR ALU op map directly onto a Borg ISA opcode?
fn borg_isel(op: nir_op) -> Option<&'static str> {
    match op {
        nir_op_fadd => Some("FADD"),
        nir_op_fmul => Some("FMUL"),
        nir_op_ffma | nir_op_ffma_weak => Some("FMADD"),
        nir_op_fneg => Some("FNEG"),
        // Moves / vector (de)construction become register copies, no ISA op.
        nir_op_mov | nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => Some("mov"),
        _ => None,
    }
}

/// Receive the app's shader as NIR and report Borg-ISA instruction-selection
/// coverage: how many instructions map directly today vs. still need lowering or
/// new selection rules (printed by op name). Returns the instruction count.
///
/// # Safety
/// `nir` must be a valid `nir_shader` pointer from the Mesa runtime (or null).
#[no_mangle]
pub unsafe extern "C" fn borgc_compile_nir(nir: *mut nir_shader) -> u32 {
    if nir.is_null() {
        return 0;
    }
    let stage = (*nir).info.stage();

    let mut total = 0u32;
    let mut alu_ok = 0u32;
    let mut alu_todo = 0u32;
    let mut tex = 0u32;
    let mut consts = 0u32;
    let mut intrinsics = 0u32;
    let mut other = 0u32;
    let mut todo: Vec<&'static str> = Vec::new();
    let mut note = |op: &'static str| {
        if !todo.contains(&op) {
            todo.push(op);
        }
    };

    let entry = nir_shader_get_entrypoint(nir);
    if !entry.is_null() {
        for block in (*entry).iter_blocks() {
            for instr in block.iter_instr_list() {
                total += 1;
                if let Some(alu) = instr.as_alu() {
                    if borg_isel(alu.op).is_some() {
                        alu_ok += 1;
                    } else {
                        alu_todo += 1;
                        note(alu.info().name());
                    }
                } else if instr.as_tex().is_some() {
                    tex += 1; // → FTEX
                } else if let Some(i) = instr.as_intrinsic() {
                    intrinsics += 1;
                    note(i.info().name());
                } else if instr.as_load_const().is_some() {
                    consts += 1;
                } else {
                    other += 1;
                }
            }
        }
    }

    eprintln!(
        "borgc: stage={stage} instrs={total} | alu_ok={alu_ok} alu_todo={alu_todo} \
         tex={tex} const={consts} intrinsic={intrinsics} other={other}"
    );
    if !todo.is_empty() {
        eprintln!("borgc:   needs lowering/selection: {}", todo.join(", "));
    }
    total
}
