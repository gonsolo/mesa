/* SPDX-License-Identifier: MIT */
/* Serial transport for the Borg drm-shim — no Mesa/Vulkan dependencies.
 * Protocol is identical to borgvk_serial.c; this copy exists so the shim can
 * be compiled without pulling in the Mesa Vulkan runtime headers. */
#pragma once
#include <stdint.h>
#include "drm-uapi/borg_drm.h"

/* Geometry packet length (must match borgvk_serial.c BORGVK_GEOM_PKT_LEN). */
#define BORG_GEOM_PKT_LEN \
   (1 + 2 + BORG_GEOM_MAX_VERTS * 6 + BORG_GEOM_MAX_TRIS * 3 + \
    BORG_GEOM_MAX_TRIS * 12 + 1)

void borg_serial_send_geom(const float *verts, int nverts,
                           const uint8_t *idx, const float *uv, int ntris);
void borg_serial_send_tex_row(int y, const float *rgb);
void borg_serial_send_mvp(const float mvp[16]);
void borg_serial_send_shader(uint8_t stage, const uint8_t *blob, uint32_t len);
