/*
 * Copyright © 2026 Andreas Wendleder
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
#include "nir_lower_blend.h"
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

/* Vertex shaders: fetch each input again at every use (a retry for a shader out of registers). */
void borg_lower_set_vertex_remat(bool on);

/* Blending in the fragment shader (docs/B3_colour_formats.md): the attachment's blend state for
 * nir_lower_blend, and the blend constants. */
struct borg_blend {
   nir_lower_blend_options opts;
   float konst[4];
};
void borg_lower_nir_for_borgc_blend(struct nir_shader *nir, const struct borg_blend *blend);

/* Implemented in Rust (src/borg/compiler/lib.rs). Writes up to buf_cap bytes
 * into out_buf and sets *out_len to the true blob size, so *out_len > buf_cap
 * signals an undersized buffer; returns the instruction count. */
/* vfetch: bit 31 = fetch vertex inputs with TEX (slot 128 + location); bits 0..15 = the
 * locations whose binding is instance-rate. 0 keeps the cube's uniform-buffer pulling. */
/* out_select: a fragment shader keeps only colour output k (-1: all), for a target drawn on its
 * own. info, if not NULL, gets what the driver stages beside the blob. */
#define BORGC_POOL_MAX 64
struct borgc_info {
   uint32_t pool[BORGC_POOL_MAX]; /* with a blob: words fetched from vertex slot 40 (fragment) or 42 */
   uint32_t npool;
   uint32_t spill_slots;          /* scratch slots of the draw's buffer region */
   uint32_t push_window;          /* push constants read from the heap window */
   uint32_t sample_mask;          /* the fragment shader writes gl_SampleMask */
};
uint32_t borgc_compile_nir(struct nir_shader *nir, uint8_t *out_buf,
                           uint32_t buf_cap, uint32_t *out_len, uint32_t vfetch,
                           int32_t out_select, struct borgc_info *info);
uint32_t borgc_selftest(void);

/* Compute: program words, (register, value) presets, LocalSize and the constant pool for the
 * job's slot 34. 0 on success. */
uint32_t borgc_compile_compute(struct nir_shader *nir,
                               uint32_t *words, uint32_t word_cap, uint32_t *nwords,
                               uint32_t *regs, uint32_t reg_cap, uint32_t *nregs,
                               uint32_t *local, uint32_t *flags,
                               uint32_t *pool, uint32_t pool_cap, uint32_t *npool);

#ifdef __cplusplus
}
#endif

#endif /* BORG_NIR_PASSES_H */
