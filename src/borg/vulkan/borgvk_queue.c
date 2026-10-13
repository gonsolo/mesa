/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_sync.h"

static VkResult borgvk_queue_submit_work(struct vk_queue *vk_queue, struct vk_queue_submit *submit);

VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct vk_device *vk_dev = vk_queue->base.device;

   for (uint32_t i = 0; i < submit->wait_count; i++) {
      VkResult r = vk_sync_wait(vk_dev, submit->waits[i].sync, submit->waits[i].wait_value,
                                VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (r != VK_SUCCESS)
         return r;
   }

   VkResult result = borgvk_queue_submit_work(vk_queue, submit);

   for (uint32_t i = 0; i < submit->signal_count; i++) {
      VkResult r = vk_sync_signal(vk_dev, submit->signals[i].sync, submit->signals[i].signal_value);
      if (r != VK_SUCCESS && result == VK_SUCCESS)
         result = r;
   }
   return result;
}

static VkResult
borgvk_queue_submit_work(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct borgvk_device *replay_dev =
      container_of(vk_queue->base.device, struct borgvk_device, vk);
   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      vk_cmd_queue_execute(&cb->cmd_queue, vk_command_buffer_to_handle(cb), &replay_dev->cmd_dispatch);
      borgvk_flush_draws(container_of(cb, struct borgvk_command_buffer, vk));   /* draws outside a pass end */
   }

   struct borgvk_device *device =
      container_of(vk_queue->base.device, struct borgvk_device, vk);

   if (getenv("BORGVK_SIM") || borgvk_hw_enabled()) {
      if (borgvk_submit_is_cube(submit))
         return borgvk_submit_sim_cube(device, submit);
      return VK_SUCCESS;
   }

   return borgvk_submit_cube_live(device, submit);
}
