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

/* struct borgvk_command_buffer is defined in borgvk_private.h -- it also
 * tracks render-pass/dynamic-rendering state (see the struct's comment),
 * needed by borgvk_CmdBeginRendering/CmdClearAttachments in
 * borgvk_memory.c. */

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
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd_buffer, struct borgvk_command_buffer, vk);

   cmd->in_rendering = false;
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
   /* Step 50 item 13.  The shader half of this already worked: borgc lowers
    * load_push_constant to `LOAD rd, rs1` with rs1 pinned to the field's word
    * index, and the hardware forms LS_BASE + (rs1 << 2).  What was missing was
    * anything putting real data where those loads look -- this, plus the
    * firmware's 0xB2 handler and borg_set_push_constants().
    *
    * `layout` and `stageFlags` are deliberately unused: Borg has one shader
    * core with one LS_BASE, so there is no per-stage or per-layout push-constant
    * window to select between.  A range pushed for the vertex stage is visible
    * to the fragment stage too.  That is more permissive than Vulkan requires,
    * not less, so it cannot make a conformant application read the wrong value
    * -- it would only fail to catch an application reading a stage it never
    * pushed to, which is already undefined behaviour.
    *
    * Emitted at record time rather than queued to submit: the firmware applies
    * a 0xB2 packet when it arrives, and recording necessarily precedes the
    * submit that ships the draw's MVP, so ordering holds for the
    * record-then-submit pattern borgvk targets.  KNOWN LIMITATION: a command
    * buffer recorded once and submitted repeatedly pushes only on the
    * recording, so a later submit reuses whatever LS_BASE last pointed at.
    * That matches how borgvk_serial_send_shader() already behaves and is
    * invisible to the current re-record-every-frame callers; fixing it means
    * capturing the range into struct borgvk_command_buffer and replaying it
    * from the submit path, which is the natural follow-on once anything
    * actually reuses command buffers. */
   VK_FROM_HANDLE(vk_command_buffer, cmd, commandBuffer);
   (void)cmd;
   (void)layout;
   (void)stageFlags;

   borgvk_serial_send_push_constants(offset, size, pValues);
}
