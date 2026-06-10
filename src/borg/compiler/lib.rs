// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// borgc — the Borg GPU shader compiler (Rust), called from the borgvk Vulkan
// driver. It lowers Mesa NIR to Borg ISA (the 7-op FP16 ISA: FADD/FMUL/FMADD/
// FNEG/FSTEP/FRCP/FTEX) and emits the .borg blob the firmware loads. Modeled on
// Mesa's NAK (src/nouveau/compiler/nak).
//
// Status: the build/link pipeline is proven (borgc_selftest) and NIR now flows
// in (borgc_compile_nir inspects the shader). The actual NIR→Borg lowering and
// .borg emission land next.

use compiler::bindings::*;

/// Self-test: returns a known constant so the C side can confirm the Rust crate
/// is linked into libvulkan_borg and callable across the FFI boundary.
#[no_mangle]
pub extern "C" fn borgc_selftest() -> u32 {
    0xB0_06
}

/// First NIR entry point: receive the app's shader as Mesa NIR and inspect it
/// (stage + instruction count) to prove the SPIR-V→NIR→borgc path. Returns the
/// instruction count. The real lowering to Borg ISA replaces this body next.
///
/// # Safety
/// `nir` must be a valid `nir_shader` pointer from the Mesa runtime (or null).
#[no_mangle]
pub unsafe extern "C" fn borgc_compile_nir(nir: *mut nir_shader) -> u32 {
    if nir.is_null() {
        return 0;
    }
    let shader = &*nir;
    let stage = shader.info.stage();

    let mut instrs: u32 = 0;
    let entry = nir_shader_get_entrypoint(nir);
    if !entry.is_null() {
        for block in (*entry).iter_blocks() {
            for _instr in block.iter_instr_list() {
                instrs += 1;
            }
        }
    }

    eprintln!("borgc: NIR shader stage={stage} instrs={instrs}");
    instrs
}
