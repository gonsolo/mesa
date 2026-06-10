/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * C side of the borgvk shader compiler. The actual NIR→Borg-ISA lowering lives
 * in the Rust crate `borgc` (src/borg/compiler/); this file is the thin C shim
 * that the Mesa runtime's shader hooks call into. For now it just exposes the
 * Rust self-test so we can confirm the Rust↔C build/link pipeline works.
 */
#include "borgvk_private.h"

#include "util/log.h"

#include <stdint.h>

/* Implemented in Rust: src/borg/compiler/lib.rs. */
uint32_t borgc_selftest(void);

void
borgvk_compiler_selftest(void)
{
   mesa_logi("borgvk: borgc (Rust compiler) selftest = 0x%x", borgc_selftest());
}
