/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "vk_device_memory.h"

struct borg_device_memory {               
   struct vk_device_memory vk;
   void* map;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_device_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)

