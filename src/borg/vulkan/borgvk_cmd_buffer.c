/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Command buffers for borgvk. We do NOT replay Vulkan commands to render — the
 * cube's per-frame MVP is read straight from the bound uniform buffer at submit
 * time (Phase 3). So recording uses the runtime's generic command-queue
 * emulation (vk_cmd_enqueue_*): every vkCmd* is recorded into the command
 * buffer's vk_cmd_queue and then simply discarded on reset. Begin/End just
 * drive the common command-buffer lifecycle.
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"

struct borgvk_command_buffer {
   struct vk_command_buffer vk;
};

static void
borgvk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd_buffer, struct borgvk_command_buffer, vk);

   vk_command_buffer_finish(&cmd->vk);
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

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &borgvk_cmd_buffer_ops, level);
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
