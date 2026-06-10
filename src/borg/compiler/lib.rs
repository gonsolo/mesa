// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// borgc — the Borg GPU shader compiler (Rust), called from the borgvk Vulkan
// driver. It lowers Mesa NIR to Borg ISA (the 7-op FP16 ISA: FADD/FMUL/FMADD/
// FNEG/FSTEP/FRCP/FTEX) and emits the .borg blob the firmware loads. Modeled on
// Mesa's NAK (src/nouveau/compiler/nak).
//
// This file currently contains only a self-test entry point to prove the
// Rust→C build/link pipeline end to end; the NIR backend lands next.

/// Self-test: returns a known constant so the C side can confirm the Rust crate
/// is linked into libvulkan_borg and callable across the FFI boundary.
#[no_mangle]
pub extern "C" fn borgc_selftest() -> u32 {
    0xB0_06
}
