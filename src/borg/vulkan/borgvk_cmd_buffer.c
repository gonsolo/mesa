/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"
#include <stdlib.h>
#include <string.h>

#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"

static void
borgvk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd_buffer, struct borgvk_command_buffer, vk);

   vk_command_buffer_finish(&cmd->vk);
   free(cmd->stream);
   free(cmd->state_sent);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

static VkResult
borgvk_create_cmd_buffer(struct vk_command_pool *pool,
                         VkCommandBufferLevel level,
                         struct vk_command_buffer **cmd_buffer_out)
{
   struct borgvk_command_buffer *cmd;

   cmd = vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkResult result = vk_command_buffer_init_with_params(
      &cmd->vk,
      &(struct vk_command_buffer_init_params) {
         .pool = pool,
         .ops = &borgvk_cmd_buffer_ops,
         .level = level,
         .needs_cmd_queue = true,
      });
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }

   *cmd_buffer_out = &cmd->vk;
   return VK_SUCCESS;
}

static void
borgvk_reset_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer,
                        VkCommandBufferResetFlags flags)
{
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd_buffer, struct borgvk_command_buffer, vk);

   free(cmd->stream);
   cmd->stream = NULL;
   free(cmd->state_sent);
   cmd->state_sent = NULL;
   cmd->state_sent_len = 0;
   cmd->stream_len = cmd->stream_cap = 0;
   cmd->heap_top = cmd->batch_draws = 0;
   cmd->batch_img = NULL;
   cmd->index_ptr = NULL;
   cmd->in_rendering = false;
   cmd->has_clear = false;
   cmd->gfx_pipeline = NULL;
   memset(cmd->vb, 0, sizeof(cmd->vb));
   memset(cmd->vb_avail, 0, sizeof(cmd->vb_avail));
   cmd->desc_set = NULL;
   memset(cmd->cs_sets, 0, sizeof(cmd->cs_sets));
   memset(cmd->desc_sets, 0, sizeof(cmd->desc_sets));
   cmd->generic_drawn = false;
   cmd->cs_pipeline = NULL;
   cmd->dispatched = false;
   cmd->color_attachment_count = 0;
   cmd->depth_view = NULL;
   cmd->stencil_view = NULL;

   vk_command_buffer_reset(vk_cmd_buffer);
}

const struct vk_command_buffer_ops borgvk_cmd_buffer_ops = {
   .create = borgvk_create_cmd_buffer,
   .reset = borgvk_reset_cmd_buffer,
   .destroy = borgvk_destroy_cmd_buffer,
};

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                          const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd, commandBuffer);

   vk_command_buffer_begin(cmd, pBeginInfo);
   return vk_command_buffer_get_record_result(cmd);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd, commandBuffer);

   return vk_command_buffer_end(cmd);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdPushConstants(VkCommandBuffer commandBuffer,
                        VkPipelineLayout layout,
                        VkShaderStageFlags stageFlags,
                        uint32_t offset, uint32_t size, const void *pValues)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd, commandBuffer);
   {
      struct borgvk_command_buffer *bcmd = container_of(cmd, struct borgvk_command_buffer, vk);
      if (offset + size <= sizeof(bcmd->pc))
         memcpy((uint8_t *)bcmd->pc + offset, pValues, size);
   }
   (void)layout;
   (void)stageFlags;

   struct borgvk_device *device = container_of(cmd->base.device, struct borgvk_device, vk);
   if (!size || offset + size > sizeof(device->push_words))
      return;
   memcpy((uint8_t *)device->push_words + offset, pValues, size);
   uint32_t lo = offset / 4, hi = (offset + size + 3) / 4;
   if (device->push_lo >= device->push_hi) {
      device->push_lo = lo;
      device->push_hi = hi;
   } else {
      if (lo < device->push_lo) device->push_lo = lo;
      if (hi > device->push_hi) device->push_hi = hi;
   }
}
