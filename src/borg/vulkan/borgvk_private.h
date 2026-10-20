/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BORGVK_PRIVATE_H
#define BORGVK_PRIVATE_H

#include "borgc_flags.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_device_memory.h"
#include "vk_buffer.h"
#include "vk_buffer_view.h"
#include "vk_image.h"
#include "vk_sampler.h"
#include "vk_command_buffer.h"
#include "vk_descriptor_set_layout.h"

#include "wsi_common.h"

#include "borgvk_entrypoints.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BORGVK_USE_WSI_PLATFORM   /* Mesa's headless WSI backend needs no platform */



#define BORGVK_SHADER_BLOB_MAX 4096
enum borgvk_shader_stage {
   BORGVK_STAGE_VERT = 0,
   BORGVK_STAGE_FRAG = 1,
   BORGVK_SHADER_STAGE_COUNT,
};
#define BORGVK_POOL_MAX  64
#define BORGVK_POOL_SLOT 40
#define BORGVK_SRGB_SLOT 41
#define BORGVK_VPOOL_SLOT 42
struct borgvk_shader_blob {
   uint8_t  data[BORGVK_SHADER_BLOB_MAX];
   uint32_t len;   /* 0 = not compiled (or didn't fit) */
   uint32_t pool[BORGVK_POOL_MAX];
   uint32_t npool;
};

/* A viewport and scissor as the firmware takes them (packet 0xC2). */
struct borgvk_vp {
   bool     valid;
   float    v[6];           /* x, y, width, height, minDepth, maxDepth */
   uint32_t sc[4];          /* scissor x0, y0, x1, y1 */
};


#define BORGVK_MAX_COLOR_ATTACHMENTS 8

#define BORGVK_MAX_VERTEX_BINDINGS 16

#define BORGVK_MAX_BINDINGS 16
#define BORGVK_MAX_WRITEBACKS 16
#define BORGVK_SSBO_SLOT_BYTES 65536u

/* A buffer range the GPU may write, copied back to host memory after the pass. */
struct borgvk_writeback {
   uint32_t off, size;
   uint8_t *host;
};



/* The persistent simulator (borgvk_sim.c). */
#define BORGVK_CS_MAX_BUFS 35   /* 32 binding slots, the push constants and the constant pool */
#define BORGVK_CS_POOL_MAX 4096
#define BORGVK_CS_WORDS_MAX 4096
#define BORGVK_CORPUS_SEED 0xcbf29ce484222325ull
struct borgvk_writeback;

/* 0xB4: STENCIL_CFG, STENCIL_FRONT, STENCIL_BACK, DEPTH_CFG, CULL_CFG. */
#define BORGVK_TEXG_CHUNK 256
/* Pack a pipeline's stencil, depth and cull state into register layout. */

#define BORGVK_GEOM_MAX_VERTS 16
#define BORGVK_GEOM_MAX_TRIS  12

#define BORGVK_TEX_DIM 64

#define BORGVK_LINE_SLOT  16
#define BORGVK_LINE_OTHER 24





/* One mip level and layer of an image, rendered to as an image of its own. */




struct borgvk_desc_map {
   uint8_t n, num[BORGVK_MAX_BINDINGS], slot[BORGVK_MAX_BINDINGS], count[BORGVK_MAX_BINDINGS];
   uint8_t total;
   bool arrays;
};




#define BORGVK_MAX_VERTEX_ATTRS    16

struct borgvk_vertex_attr {
   uint32_t location;
   uint32_t binding;
   uint32_t offset;
   VkFormat format;
};



#ifdef __cplusplus
}
#endif

struct tex_conv {
   uint32_t bpp, obpp, cs, sch, alpha, pack16, zstride, soff;
   bool swap_rb;
};


#include "borgvk_rust.h"

#endif /* BORGVK_PRIVATE_H */
