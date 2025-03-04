/*
 * Copyright © 2024-25 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include <xf86drm.h>

#include "borg_device.h"
#include "borg_queue.h"
#include "borg_ws_device.h"

#include "drm-uapi/borg_drm.h"

extern void *global_shader_pointer;
extern uint32_t global_shader_size;

static VkResult borg_driver_submit(struct vk_queue *vk_queue,
                            struct vk_queue_submit *submit)
{
   puts(__func__);
   printf("Borg: global_shader_pointer here is %p\n", global_shader_pointer);

   struct borg_queue *queue = container_of(vk_queue, struct borg_queue, vk);
   struct borg_device *dev = borg_queue_device(queue);

   struct drm_borg_submit borg_submit = {
      .dummy = 1,
      .shader_pointer = (__u64)global_shader_pointer,
      .shader_size = global_shader_size
   };
   int err = drmCommandWriteRead(dev->ws_dev->fd, DRM_BORG_SUBMIT, &borg_submit, sizeof(borg_submit));
   if (err) {
      puts("borg submit failed!");
   }
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

