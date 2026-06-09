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

#include "borgvk_entrypoints.h"

#ifdef __cplusplus
extern "C" {
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
};

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

/* A descriptor set just remembers which buffer is bound at each binding, so the
 * submit path can find the cube's uniform buffer (binding 0) and read the MVP. */
struct borgvk_descriptor_set {
   struct vk_object_base base;
   struct borgvk_buffer *buffers[BORGVK_MAX_BINDINGS];
   VkDeviceSize offsets[BORGVK_MAX_BINDINGS];
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
