/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_cmd_buffer.h"
#include "borg_cmd_pool.h"

#include "vk_alloc.h"
#include "vk_command_buffer.h"

static VkResult
borg_create_cmd_buffer(struct vk_command_pool *vk_pool,
                       struct vk_command_buffer **cmd_buffer_out)
{
   struct borg_cmd_pool *pool = container_of(vk_pool, struct borg_cmd_pool, vk);
   struct borg_device *dev = borg_cmd_pool_device(pool);
   struct borg_cmd_buffer *cmd;
   VkResult result;

   cmd = vk_zalloc(&pool->vk.alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_command_buffer_init(&pool->vk, &cmd->vk,
                                   &borg_cmd_buffer_ops, 0);
   if (result != VK_SUCCESS) {
      vk_free(&pool->vk.alloc, cmd);
      return result;
   }

   *cmd_buffer_out = &cmd->vk;

   return VK_SUCCESS;
}

const struct vk_command_buffer_ops borg_cmd_buffer_ops = {
   .create = borg_create_cmd_buffer,
   //.reset = borg_reset_cmd_buffer,
   //.destroy = borg_destroy_cmd_buffer,
};

VKAPI_ATTR VkResult VKAPI_CALL
borg_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                        const VkCommandBufferBeginInfo *pBeginInfo)
{
   // TODO
   return VK_SUCCESS;
}

void
borg_cmd_bind_shaders(struct vk_command_buffer *vk_cmd,
                      uint32_t stage_count,
                      const gl_shader_stage *stages,
                      struct vk_shader ** const shaders)
{
   // TODO
}

VKAPI_ATTR void VKAPI_CALL
borg_CmdBindDescriptorSets(VkCommandBuffer commandBuffer,
                           VkPipelineBindPoint pipelineBindPoint,
                           VkPipelineLayout layout,
                           uint32_t firstSet,
                           uint32_t descriptorSetCount,
                           const VkDescriptorSet *pDescriptorSets,
                           uint32_t dynamicOffsetCount,
                           const uint32_t *pDynamicOffsets)
{
   // TODO
}
