/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_device.h"
#include "borg_queue.h"

static VkResult borg_driver_submit(struct vk_queue *queue,
                            struct vk_queue_submit *submit)
{
   // TODO
   return VK_SUCCESS;
}

VkResult borg_queue_init(struct borg_device* dev,
			 struct borg_queue *queue,
			 const VkDeviceQueueCreateInfo *pCreateInfo,
			 uint32_t index_in_family)
{
   VkResult result = VK_SUCCESS;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   queue->vk.driver_submit = borg_driver_submit;

   return result;
}

