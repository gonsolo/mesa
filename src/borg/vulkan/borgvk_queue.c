/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Queue submit for borgvk — the single interception point of the driver. cube.c
 * records a frame into a command buffer (captured into cmd_buffer->cmd_queue by
 * the runtime's vk_cmd_enqueue emulation) and submits it. We never replay those
 * commands; instead we locate the bound descriptor set's binding-0 uniform
 * buffer, which for cube.c is `struct vktexcube_vs_uniform`:
 *
 *     float mvp[4][4];          // bytes   0..63   (updated every frame)
 *     float position[12*3][4];  // bytes  64..639  (static mesh, written once)
 *     float attr[12*3][4];      // bytes 640..1215 (static UVs)
 *
 * Each frame we ship the MVP over serial (0xAD). Once, at startup, we also ship
 * the real mesh (0xAE): we dedup the 36 expanded positions to the unique corners
 * and forward them with the indexed triangle list + UVs. The GPU then renders
 * the app's actual geometry with hardware vertex transform + perspective divide.
 */
#include "borgvk_private.h"

#include "vk_command_buffer.h"
#include "vk_cmd_queue.h"

#include <string.h>

/* cube.c uniform-buffer layout (floats). */
#define UBO_MVP_FLOATS   16
#define UBO_NUM_VERTS    36                  /* 12 triangles, expanded */
#define UBO_POS_FLOAT0   UBO_MVP_FLOATS      /* position[36][4] */
#define UBO_ATTR_FLOAT0  (UBO_MVP_FLOATS + UBO_NUM_VERTS * 4)  /* attr[36][4] */
#define UBO_MIN_FLOATS   (UBO_ATTR_FLOAT0 + UBO_NUM_VERTS * 4) /* 304 = 1216 B */

/* Locate the bound descriptor-set's binding-0 uniform buffer; return its mapped
 * base (where mvp + geometry live) or NULL, and the readable float count. */
static const float *
find_ubo(struct vk_queue_submit *submit, uint32_t *floats_out)
{
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct vk_command_buffer *cb = submit->command_buffers[i];
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

         VkDeviceSize off = buf->offset + set->offsets[0];
         if (off >= buf->vk.size)
            continue;
         *floats_out = (uint32_t)((buf->vk.size - off) / sizeof(float));
         return (const float *)((const char *)buf->mem->map + off);
      }
   }
   return NULL;
}

/* Dedup the 36 expanded positions to unique corners; build the indexed triangle
 * list and per-tri-vertex UVs, then ship the mesh. */
static void
send_geometry(const float *ubo)
{
   const float *pos = ubo + UBO_POS_FLOAT0;
   const float *att = ubo + UBO_ATTR_FLOAT0;

   float verts[BORGVK_GEOM_MAX_VERTS * 3];
   uint8_t idx[UBO_NUM_VERTS];
   float uv[UBO_NUM_VERTS * 2];
   int nverts = 0;

   for (int i = 0; i < UBO_NUM_VERTS; i++) {
      float x = pos[i*4 + 0], y = pos[i*4 + 1], z = pos[i*4 + 2];
      int u = -1;
      for (int j = 0; j < nverts; j++) {
         if (verts[j*3+0] == x && verts[j*3+1] == y && verts[j*3+2] == z) {
            u = j;
            break;
         }
      }
      if (u < 0) {
         if (nverts >= BORGVK_GEOM_MAX_VERTS)
            return;  /* mesh too complex for the fixed packet — skip (cube fits) */
         u = nverts++;
         verts[u*3+0] = x; verts[u*3+1] = y; verts[u*3+2] = z;
      }
      idx[i] = (uint8_t)u;
      uv[i*2+0] = att[i*4 + 0];
      uv[i*2+1] = att[i*4 + 1];
   }

   borgvk_serial_send_geom(verts, nverts, idx, uv, UBO_NUM_VERTS / 3);
}

VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   uint32_t nfloats = 0;
   const float *ubo = find_ubo(submit, &nfloats);
   if (!ubo)
      return VK_SUCCESS;

   /* Ship the static mesh on the first frames (and rarely after, to self-heal a
    * dropped upload), otherwise the per-frame MVP. The firmware drains one
    * packet per frame, so geometry and MVP take turns; the cube is briefly
    * static while the mesh uploads, then animates. */
   static unsigned submit_no = 0;
   bool can_geom = (nfloats >= UBO_MIN_FLOATS);
   if (can_geom && (submit_no < 24 || (submit_no % 256) == 0))
      send_geometry(ubo);
   else
      borgvk_serial_send_mvp(ubo);   /* mvp = the first 16 floats */
   submit_no++;

   return VK_SUCCESS;
}
