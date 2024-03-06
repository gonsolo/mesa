/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_PHYSICAL_DEVICE_H
#define BORG_PHYSICAL_DEVICE_H

#include "vulkan/runtime/vk_physical_device.h"

struct borg_physical_device;
struct _drmDevice;

struct borg_queue_family {
   VkQueueFlags queue_flags;
   uint32_t queue_count;
};

struct borg_memory_heap {

   uint64_t size;
   uint64_t used;
   VkMemoryHeapFlags flags;
   uint64_t (*available)(struct borg_physical_device *pdev);
};

struct borg_physical_device {
   struct vk_physical_device vk;

   struct borg_memory_heap mem_heaps[1];
   uint8_t mem_heap_count;
   VkMemoryType mem_types[1];
   uint8_t mem_type_count;

   struct borg_queue_family queue_families[1];
   uint8_t queue_family_count;
};

VkResult borg_create_drm_physical_device(struct vk_instance *vk_instance,
                                          struct _drmDevice *drm_device,
                                          struct vk_physical_device **pdev_out);

void borg_physical_device_destroy(struct vk_physical_device *vk_device);

VK_DEFINE_HANDLE_CASTS(borg_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)


#endif
