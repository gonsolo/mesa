/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * borgvk — a native Mesa Vulkan ICD for the Borg GPU (ULX3S FPGA), driven over
 * serial. Modeled on the v3dv (Broadcom) driver. See
 * ~/.claude/plans/atomic-questing-stream.md for the overall design.
 */
#ifndef BORGVK_PRIVATE_H
#define BORGVK_PRIVATE_H

#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_device_memory.h"
#include "vk_buffer.h"
#include "vk_image.h"
#include "vk_descriptor_set_layout.h"

#include "wsi_common.h"

#include "borgvk_entrypoints.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WSI (window-system integration) is available when at least one surface
 * platform is compiled in. borgvk uses the software (CPU) WSI path: the host
 * window is irrelevant — the real output is the FPGA's HDMI — so WSI only needs
 * to function so an app's render loop runs and submits each frame. */
#if defined(VK_USE_PLATFORM_XCB_KHR) || \
    defined(VK_USE_PLATFORM_XLIB_KHR) || \
    defined(VK_USE_PLATFORM_WAYLAND_KHR)
#define BORGVK_USE_WSI_PLATFORM
#endif

struct borgvk_instance {
   struct vk_instance vk;
};

struct borgvk_physical_device {
   struct vk_physical_device vk;

   /* Memory layout reported to the app. Borg renders on the FPGA; from the
    * host's point of view all "device" memory is plain host-visible RAM that we
    * stage and ship over serial, so we expose a single host-visible coherent
    * heap. */
   VkPhysicalDeviceMemoryProperties memory;

   /* Submit is synchronous (ship MVP over serial, return), so a single
    * immediate-success sync type (borgvk_sync_type) backs fences/semaphores. */
   const struct vk_sync_type *sync_types[2];

#ifdef BORGVK_USE_WSI_PLATFORM
   struct wsi_device wsi_device;
#endif
};

#ifdef BORGVK_USE_WSI_PLATFORM
VkResult borgvk_wsi_init(struct borgvk_physical_device *physical_device);
void borgvk_wsi_finish(struct borgvk_physical_device *physical_device);
#endif

extern const struct vk_sync_type borgvk_sync_type;

struct borgvk_device {
   struct vk_device vk;

   /* Dispatch table used when replaying recorded command queues (secondary
    * cmd buffers). We never replay for rendering, but the runtime wants it set. */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Single graphics/compute/transfer queue. cube.c and vulkaninfo both use one
    * queue from family 0; multi-queue support can come later. */
   struct vk_queue queue;
};

extern const struct vk_command_buffer_ops borgvk_cmd_buffer_ops;

/* ---- Serial transport (borgvk_serial.c) ------------------------------- *
 * The "GPU" is the ULX3S FPGA reached over the FT231X serial bridge
 * (/dev/ttyUSB0 @115200, overridable via $BORGVK_SERIAL). The submit path
 * ships the per-frame 4×4 MVP as a framed packet the firmware decodes. */
void borgvk_serial_send_mvp(const float mvp[16]);

/* Max mesh the firmware's fixed-length 0xAE geometry packet carries (must match
 * the firmware RX_GEOM_*). cube.c is 8 unique verts / 12 triangles. */
#define BORGVK_GEOM_MAX_VERTS 16
#define BORGVK_GEOM_MAX_TRIS  16

/* Ship the app's real mesh once (Phase B): nverts unique model-space positions
 * (xyz floats), ntris triangles each indexing 3 of them (idx, 3 per tri), with
 * per-triangle-vertex UVs (uv, 2 floats per tri-vertex). */
void borgvk_serial_send_geom(const float *verts, int nverts,
                             const uint8_t *idx, const float *uv, int ntris);

/* The Borg texture dimension (must match firmware TEX_WIDTH). The app's texture
 * is downsampled to this on the host (lossless at the 128x128 render size). */
#define BORGVK_TEX_DIM 64

/* Ship one texture row (Phase B): `rgb` is BORGVK_TEX_DIM texels of 3 floats
 * (R,G,B in [0,1]); converted to RGB-FP16 and framed as a 0xAF packet. */
void borgvk_serial_send_tex_row(int y, const float *rgb);

/* Queue submit hook (borgvk_queue.c): reads the MVP from the submitted command
 * buffer's bound descriptor set and ships it over serial. Wired into
 * device->queue.driver_submit by CreateDevice. */
VkResult borgvk_queue_submit(struct vk_queue *queue,
                             struct vk_queue_submit *submit);

/* Compiler shim (borgvk_compiler.c → Rust borgc crate). */
void borgvk_compiler_selftest(void);
void borgvk_compile_stage(struct borgvk_device *device,
                          const VkPipelineShaderStageCreateInfo *stage_info);

VK_DEFINE_HANDLE_CASTS(borgvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(borgvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(borgvk_device, vk.base, VkDevice,
                       VK_OBJECT_TYPE_DEVICE)

/* All "device" memory is plain host RAM. We malloc the backing bytes and hand
 * the pointer straight back from vkMapMemory; at submit time (Phase 3) the
 * driver reads the bound uniform buffer's bytes here and ships the MVP over
 * serial. */
struct borgvk_device_memory {
   struct vk_device_memory vk;
   void *map;   /* malloc'd backing store, mem->vk.size bytes */
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

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_buffer, vk.base, VkBuffer,
                               VK_OBJECT_TYPE_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_image, vk.base, VkImage,
                               VK_OBJECT_TYPE_IMAGE)

#define BORGVK_MAX_BINDINGS 8

struct borgvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count;
};

struct borgvk_descriptor_pool {
   struct vk_object_base base;
};

/* A descriptor set remembers which buffer/image is bound at each binding, so the
 * submit path can find the cube's uniform buffer (binding 0 → MVP + geometry)
 * and its texture image (binding 1 → combined image sampler). */
struct borgvk_descriptor_set {
   struct vk_object_base base;
   struct borgvk_buffer *buffers[BORGVK_MAX_BINDINGS];
   VkDeviceSize offsets[BORGVK_MAX_BINDINGS];
   struct borgvk_image *images[BORGVK_MAX_BINDINGS];
};

/* No shader compilation yet (the cube's shaders are pre-baked in firmware), so
 * a pipeline is an opaque placeholder. */
struct borgvk_pipeline {
   struct vk_object_base base;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_descriptor_set_layout, vk.base,
                               VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_pipeline, base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE)

#ifdef __cplusplus
}
#endif

#endif /* BORGVK_PRIVATE_H */
