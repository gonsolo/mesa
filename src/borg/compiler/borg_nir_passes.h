/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * The SPIR-V and NIR configuration borgc expects, shared by every front end
 * that feeds it.
 *
 * There are two: the borgvk driver (compiling the application's SPIR-V at
 * vkQueueSubmit time) and the offline borgc CLI (compiling a .spv at build
 * time). They MUST agree -- a pass that runs in one and not the other changes
 * the NIR borgc sees and therefore the code it emits, so an offline-compiled
 * shader would not match what the driver produces for the same source. That is
 * exactly the kind of divergence that is invisible until an image is wrong, so
 * the configuration lives here rather than being copied.
 */
#ifndef BORG_NIR_PASSES_H
#define BORG_NIR_PASSES_H

#include "nir.h"
#include "spirv/nir_spirv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* SPIR-V → NIR options. Addressing formats mirror a software driver
 * (lavapipe); the Borg backend reads resources from the firmware's
 * uniform/descriptor model, so the exact formats only need to be
 * self-consistent for NIR construction. */
extern const struct spirv_to_nir_options borg_spirv_options;

/* NIR compiler options for the Borg target. */
extern const struct nir_shader_compiler_options borg_nir_options;

/* Lower and optimize NIR into the shape borgc's instruction selection expects.
 * Idempotent enough to call once per shader, immediately after
 * SPIR-V → NIR. */
void borg_lower_nir_for_borgc(struct nir_shader *nir);

/* Implemented in Rust (src/borg/compiler/lib.rs). Writes up to buf_cap bytes
 * into out_buf and sets *out_len to the true blob size, so *out_len > buf_cap
 * signals an undersized buffer; returns the instruction count. */
uint32_t borgc_compile_nir(struct nir_shader *nir, uint8_t *out_buf,
                           uint32_t buf_cap, uint32_t *out_len);
uint32_t borgc_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* BORG_NIR_PASSES_H */
