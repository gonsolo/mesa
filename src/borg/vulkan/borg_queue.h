/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_QUEUE_H
#define BORG_QUEUE_H

#include "vk_queue.h"

struct borg_device;

struct borg_queue {
   struct vk_queue vk;
};

VkResult borg_queue_init(struct borg_device* dev,
		         struct borg_queue *queue,
			 const VkDeviceQueueCreateInfo *pCreateInfo,
			 uint32_t index_in_family);

#endif
