/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEVICE_H
#define BORG_DEVICE_H

#include "borg_queue.h"

#include "util/simple_mtx.h"
#include "util/vma.h"

#include "vk_device.h"

#define BORG_HEAP_START ((uint64_t)4096)
#define BORG_HEAP_END ((uint64_t)(1ull << 38))

struct borg_device {
   struct vk_device vk;
   struct borg_ws_device *ws_dev;

   simple_mtx_t heap_mutex;
   struct util_vma_heap heap;

   struct borg_queue queue;
};

VK_DEFINE_HANDLE_CASTS(borg_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct borg_physical_device *
borg_device_physical(struct borg_device *dev)
{
   return (struct borg_physical_device *)dev->vk.physical;
}

#endif
