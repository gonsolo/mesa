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

struct borgvk_instance {
   struct vk_instance vk;
};

struct borgvk_physical_device {
   struct vk_physical_device vk;

   VkPhysicalDeviceMemoryProperties memory;

   const struct vk_sync_type *sync_types[2];

   int drm_fd;

#ifdef BORGVK_USE_WSI_PLATFORM
   struct wsi_device wsi_device;
#endif
};

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

struct borgvk_device {
   struct vk_device vk;

   struct vk_device_dispatch_table cmd_dispatch;

   struct vk_queue queue;

   /* Non-owning copy of physical_device->drm_fd.  -1 → malloc+serial fallback. */
   int drm_fd;

   const struct borg_blend *blend_lower;
   struct borgvk_shader_blob shader_blob[BORGVK_SHADER_STAGE_COUNT];
   VkPipelineLayout compile_layout;
   uint32_t stage_ubo[2];
   bool stage_push[2];
   bool stage_pwin[2];   /* the stage reads push constants from the heap window */
   bool stage_spill[2];  /* the stage spills to the region's scratch slot */
   int frag_tex_binding;
   uint32_t frag_select;   /* 1 + the colour output the fragment shader keeps; 0 keeps all */
   bool frag_reads_pntc;
   uint32_t blend_cfg;
   uint32_t blend_const;
   uint32_t state_reg[5];
   struct borgvk_vp vp;      /* the viewport and scissor of the draw being queued */
   bool     state_valid;
   /* Push constants set by vkCmdPushConstants, replayed at submit (DRM path). */
   uint32_t push_words[32];
   uint32_t push_lo, push_hi;   /* word range written so far; empty when lo >= hi */
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
struct borgvk_image;
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

VK_DEFINE_HANDLE_CASTS(borgvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(borgvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(borgvk_device, vk.base, VkDevice,
                       VK_OBJECT_TYPE_DEVICE)

struct borgvk_device_memory {
   struct vk_device_memory vk;
   void    *map;         /* mmap(drm_fd, gem_offset) or malloc'd              */
   size_t   size;        /* allocation size for munmap                         */
   uint32_t gem_handle;  /* 0 = malloc path (no DRM)                          */
};

struct borgvk_buffer {
   struct vk_buffer vk;
   struct borgvk_device_memory *mem;
   VkDeviceSize offset;
};

struct borgvk_image {
   struct vk_image vk;
   VkDeviceSize size;     /* bytes needed to back this image */
   struct borgvk_device_memory *mem;
   VkDeviceSize offset;
};

/* One mip level and layer of an image, rendered to as an image of its own. */

struct borgvk_buffer_view {
   struct vk_buffer_view vk;
};

struct borgvk_sampler {
   struct vk_sampler vk;
   uint32_t desc[4];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_buffer, vk.base, VkBuffer,
                               VK_OBJECT_TYPE_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_image, vk.base, VkImage,
                               VK_OBJECT_TYPE_IMAGE)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)

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
