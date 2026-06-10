/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Queue submit for borgvk — the single interception point of the driver. cube.c
 * records a frame into a command buffer (captured into cmd_buffer->cmd_queue by
 * the runtime's vk_cmd_enqueue emulation) and submits it. We never replay those
 * commands; instead we locate the bound descriptor set and forward the app's
 * real data to the FPGA over serial:
 *
 *   binding 0 (uniform buffer) = `struct vktexcube_vs_uniform`:
 *       float mvp[4][4];          // bytes   0..63   (updated every frame → 0xAD)
 *       float position[12*3][4];  // bytes  64..639  (static mesh → 0xAE)
 *       float attr[12*3][4];      // bytes 640..1215 (static UVs)
 *   binding 1 (combined image sampler) = the texture (RGBA8 → 0xAF rows)
 *
 * The firmware drains one packet per frame, so we time-multiplex: ship the mesh
 * on the first frames, then per-frame MVPs interleaved with texture rows (the
 * texture progressively, reliably fills in while the cube animates). The GPU
 * renders with hardware vertex transform + perspective divide.
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

#define GEOM_FRAMES      24   /* frames spent shipping the mesh at startup */

/* Locate the bound descriptor set (binding 0 = UBO, binding 1 = texture). */
static struct borgvk_descriptor_set *
find_set(struct vk_queue_submit *submit)
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
         if (set)
            return set;
      }
   }
   return NULL;
}

/* Dedup the 36 expanded positions to unique corners; build the indexed triangle
 * list + per-tri-vertex UVs, then ship the mesh. */
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
            return;  /* mesh too complex for the fixed packet (cube fits) */
         u = nverts++;
         verts[u*3+0] = x; verts[u*3+1] = y; verts[u*3+2] = z;
      }
      idx[i] = (uint8_t)u;
      uv[i*2+0] = att[i*4 + 0];
      uv[i*2+1] = att[i*4 + 1];
   }

   borgvk_serial_send_geom(verts, nverts, idx, uv, UBO_NUM_VERTS / 3);
}

/* Box-downsample the linear RGBA8 texture to one BORGVK_TEX_DIM-wide row of
 * normalized RGB floats and ship it. */
static void
send_texture_row(struct borgvk_image *tex, int dy)
{
   const uint8_t *base = (const uint8_t *)tex->mem->map + tex->offset;
   uint32_t sw = tex->vk.extent.width, sh = tex->vk.extent.height;
   if (sw == 0 || sh == 0)
      return;
   uint32_t pitch = sw * 4;                 /* RGBA8, linear tiling */
   int sxs = (int)(sw / BORGVK_TEX_DIM); if (sxs < 1) sxs = 1;
   int sys = (int)(sh / BORGVK_TEX_DIM); if (sys < 1) sys = 1;

   float row[BORGVK_TEX_DIM * 3];
   for (int dx = 0; dx < BORGVK_TEX_DIM; dx++) {
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int oy = 0; oy < sys; oy++) {
         uint32_t sy = (uint32_t)dy * sys + oy;
         if (sy >= sh) break;
         for (int ox = 0; ox < sxs; ox++) {
            uint32_t sx = (uint32_t)dx * sxs + ox;
            if (sx >= sw) break;
            const uint8_t *p = base + sy * pitch + sx * 4;
            r += p[0]; g += p[1]; b += p[2]; n++;
         }
      }
      if (n == 0) n = 1;
      row[dx*3+0] = (float)r / n / 255.0f;
      row[dx*3+1] = (float)g / n / 255.0f;
      row[dx*3+2] = (float)b / n / 255.0f;
   }
   borgvk_serial_send_tex_row(dy, row);
}

VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct borgvk_descriptor_set *set = find_set(submit);
   if (!set)
      return VK_SUCCESS;

   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return VK_SUCCESS;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return VK_SUCCESS;
   const float *ubo = (const float *)((const char *)ubuf->mem->map + off);
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));

   struct borgvk_image *tex = set->images[1];
   bool can_geom = nfloats >= UBO_MIN_FLOATS;
   bool can_tex  = tex && tex->mem && tex->mem->map &&
                   tex->vk.extent.width && tex->vk.extent.height;

   static unsigned submit_no = 0;
   static int tex_row = 0;

   /* Startup: ship the mesh. Then animate (MVP), slipping a texture row in every
    * 4th frame — the rows cycle so the whole texture reliably lands over time
    * even though the firmware only catches ~1 packet per frame. */
   if (can_geom && submit_no < GEOM_FRAMES) {
      send_geometry(ubo);
   } else if (can_tex && (submit_no % 4) == 3) {
      send_texture_row(tex, tex_row);
      tex_row = (tex_row + 1) % BORGVK_TEX_DIM;
   } else {
      borgvk_serial_send_mvp(ubo);   /* mvp = the first 16 floats */
   }
   submit_no++;

   return VK_SUCCESS;
}
