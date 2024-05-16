/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEVICE_MEMORY_H
#define BORG_DEVICE_MEMORY_H

#include "vk_device_memory.h"

struct borg_device_memory {               
   struct vk_device_memory vk;
   void* map;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_device_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)

#endif
