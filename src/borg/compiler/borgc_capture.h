/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORGC_CAPTURE_H
#define BORGC_CAPTURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;

#define BORGC_CAPTURE_MAGIC 0x4c505242u /* "BRPL" */

enum borgc_capture_kind {
   BORGC_CAPTURE_GRAPHICS = 0,
   BORGC_CAPTURE_COMPUTE = 1,
};

/* With BORGC_REPLAY_DIR set, writes the shader borgc is given to <dir>/<hash>.nir for borgc-replay. */
void borgc_capture(const struct nir_shader *nir, uint32_t kind, uint32_t vfetch, int32_t out_select);

#ifdef __cplusplus
}
#endif

#endif
