/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Queue submit for borgvk. This is the single interception point of the whole
 * driver: cube.c records a frame into a command buffer (which the runtime's
 * vk_cmd_enqueue emulation captures into cmd_buffer->cmd_queue) and submits it.
 * We never replay those commands to render on the host — instead we locate the
 * bound descriptor set, read the 4×4 MVP from its binding-0 uniform buffer (the
 * first 64 bytes of cube.c's vktexcube_vs_uniform), and ship it to the FPGA over
 * serial. The Borg sequencer renders the cube autonomously from that MVP.
 *
 * Because the FPGA render is effectively synchronous from the host's point of
 * view (we return only after the bytes are queued to the port) and our sync type
 * waits always succeed immediately, no explicit fence/semaphore signalling is
 * needed here.
 */
#include "borgvk_private.h"

#include "vk_command_buffer.h"
#include "vk_cmd_queue.h"

#include <string.h>

/* Walk a recorded command buffer for the first BindDescriptorSets and pull the
 * MVP out of descriptor set 0's binding-0 uniform buffer. Returns true on hit. */
static bool
read_mvp_from_cmd_buffer(struct vk_command_buffer *cb, float mvp[16])
{
   list_for_each_entry(struct vk_cmd_queue_entry, e, &cb->cmd_queue.cmds, cmd_link) {
      if (e->type != VK_CMD_BIND_DESCRIPTOR_SETS)
         continue;

      const struct vk_cmd_bind_descriptor_sets *b = &e->u.bind_descriptor_sets;
      if (b->descriptor_set_count == 0 || b->descriptor_sets == NULL)
         continue;

      VK_FROM_HANDLE(borgvk_descriptor_set, set, b->descriptor_sets[0]);
      if (!set)
         continue;

      struct borgvk_buffer *buf = set->buffers[0];
      if (!buf || !buf->mem || !buf->mem->map)
         continue;

      /* mvp is the first 64 bytes of the uniform; combine the buffer's own
       * memory offset with the descriptor's offset into that buffer. */
      const char *base = (const char *)buf->mem->map +
                         buf->offset + set->offsets[0];
      memcpy(mvp, base, 16 * sizeof(float));
      return true;
   }
   return false;
}

VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   float mvp[16];
   bool found = false;

   for (uint32_t i = 0; i < submit->command_buffer_count && !found; i++)
      found = read_mvp_from_cmd_buffer(submit->command_buffers[i], mvp);

   if (found)
      borgvk_serial_send_mvp(mvp);

   return VK_SUCCESS;
}
