/* SPDX-License-Identifier: MIT */
/* Serial transport for the Borg drm-shim — no Mesa/Vulkan dependencies.
 * Protocol is identical to borgvk_serial.c; this copy exists so the shim can
 * be compiled without pulling in the Mesa Vulkan runtime headers. */
#pragma once
#include <stdint.h>
#include "drm-uapi/borg_drm.h"

/* Geometry packet length (must match borgvk_serial.c BORGVK_GEOM_PKT_LEN). */
#define BORG_GEOM_PKT_LEN \
   (1 + 2 + BORG_GEOM_MAX_VERTS * 12 + BORG_GEOM_MAX_TRIS * 3 + \
    BORG_GEOM_MAX_TRIS * 24 + 1)

void borg_serial_send_geom(const float *verts, int nverts,
                           const uint8_t *idx, const float *uv, int ntris);
/* 0xAF: one row of BORG_TEX_DIM RGBA8 texels, with the sampler descriptor. */
void borg_serial_send_tex_row(int y, const uint8_t *rgba, const uint32_t sampler[4]);
void borg_serial_send_mvp(const float mvp[16]);
/* 0xB3: BLEND_CFG and BLEND_CONST register values. */
void borg_serial_send_blend(uint32_t blend_cfg, uint32_t blend_const);

/* 0xB4: STENCIL_CFG, STENCIL_FRONT, STENCIL_BACK, DEPTH_CFG, CULL_CFG. */
void borg_serial_send_state(const uint32_t reg[5]);
/* 0xB2: push-constant words. */
void borg_serial_send_push(uint32_t off_words, uint32_t n_words, const uint32_t *words);
void borg_serial_send_shader(uint8_t stage, const uint8_t *blob, uint32_t len);
