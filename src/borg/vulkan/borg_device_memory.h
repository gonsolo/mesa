/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEVICE_MEMORY_H
#define BORG_DEVICE_MEMORY_H

#include "vk_device_memory.h"

struct borg_va {
   struct borg_device *dev;
   uint64_t addr;
   uint64_t size_B;
};

struct borg_mem {
   struct borg_va *va;
   void* map;
   uint64_t size_B;
   struct borg_ws_bo *bo;
};

struct borg_device_memory {
   struct vk_device_memory vk;
   struct borg_mem *mem;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_device_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)

#endif
