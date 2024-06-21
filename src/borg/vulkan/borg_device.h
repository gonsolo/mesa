/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEVICE_H
#define BORG_DEVICE_H

#include "borg_queue.h"

#include "vk_device.h"

struct borg_device {
	struct vk_device vk;
        struct borg_ws_device *ws_dev;

	//struct borg_physical_device* pdev;
	struct borg_queue queue;
};

VK_DEFINE_HANDLE_CASTS(borg_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct borg_physical_device *
borg_device_physical(struct borg_device *dev)
{
   return (struct borg_physical_device *)dev->vk.physical;
}

#endif
