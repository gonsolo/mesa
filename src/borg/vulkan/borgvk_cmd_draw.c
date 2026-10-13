/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "util/format_srgb.h"
#include "borgvk_private.h"

#include "vk_format.h"

enum {
   REJECT_UNDRAWABLE = 3,
   REJECT_NO_SHADER = 4,
   REJECT_NO_TARGET_MEMORY = 5,
   REJECT_NO_INDEX_BUFFER = 6,
   REJECT_INDEX_RANGE = 7,
   REJECT_VERTEX_INPUT = 8,
   REJECT_VERTEX_RANGE = 9,
   REJECT_UPLOAD = 10,
   REJECT_MRT_FORMATS = 12,
   REJECT_MRT_EXTENT = 13,
   REJECT_MRT_NOT_SERVED = 14,
};

static bool
generic_reject(int why)
{
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: generic sim draw rejected (reason %d)", why);
   return false;
}

#define BORGVK_HEAP_BYTES 0x2000000u
#define BORGVK_POINT_SLOT 16

struct draw_heap {
   uint32_t top;
};

static uint64_t
corpus_state(uint64_t h, const struct borgvk_image *img, struct borgvk_image *const *extra, uint32_t nextra,
             const struct vk_image_view *depth, const struct vk_image_view *stencil,
             const struct borgvk_writeback *wb, uint32_t nwb)
{
   h = borgvk_corpus_image(h, img);
   for (uint32_t i = 0; i < nextra; i++)
      h = borgvk_corpus_image(h, extra[i]);
   if (depth && depth->image)
      h = borgvk_corpus_image(h, container_of(depth->image, struct borgvk_image, vk));
   if (stencil && stencil->image)
      h = borgvk_corpus_image(h, container_of(stencil->image, struct borgvk_image, vk));
   for (uint32_t i = 0; i < nwb; i++)
      h = borgvk_corpus_hash(h, wb[i].host, wb[i].size);
   return h;
}

void
borgvk_flush_draws(struct borgvk_command_buffer *cmd)
{
   if (cmd->stream_len == 0)
      return;
   uint8_t *bytes = cmd->stream;
   size_t n = cmd->stream_len;
   struct borgvk_image *img = cmd->batch_img;
   struct borgvk_image *extra[3] = { cmd->batch_extra[0], cmd->batch_extra[1], cmd->batch_extra[2] };
   const uint32_t nextra = cmd->batch_nextra;
   const bool serve = cmd->batch_serve;
   struct borgvk_writeback wb[BORGVK_MAX_WRITEBACKS];
   const uint32_t nwb = cmd->nwb;
   memcpy(wb, cmd->wb, sizeof(wb));
   cmd->nwb = 0;
   cmd->region_set = NULL;
   cmd->batch_pool = NULL;
   cmd->batch_srgb = false;
   cmd->stream = NULL;
   cmd->stream_len = cmd->stream_cap = 0;
   cmd->heap_top = 0;
   cmd->batch_draws = 0;
   cmd->batch_img = NULL;
   cmd->batch_nextra = 0;
   free(cmd->state_sent);
   cmd->state_sent = NULL;
   cmd->state_sent_len = 0;
   const bool rec = borgvk_corpus_on();
   uint64_t key = BORGVK_CORPUS_SEED;
   if (rec) {
      key = borgvk_corpus_hash(key, bytes, n);
      key = corpus_state(key, img, extra, nextra, cmd->batch_depth, cmd->batch_stencil, wb, nwb);
   }
   if (serve) {
      borgvk_sim_run_pass(bytes, n, img, extra, nextra, cmd->batch_depth, cmd->batch_stencil, wb, nwb);
      free(bytes);
   } else {
      borgvk_sim_run_stream(bytes, n, img, 0);   /* takes ownership of bytes */
   }
   if (rec)
      borgvk_corpus_log('F', key, corpus_state(BORGVK_CORPUS_SEED, img, extra, nextra,
                                               cmd->batch_depth, cmd->batch_stencil, wb, nwb));
}

static bool
stream_append(struct borgvk_command_buffer *cmd, const uint8_t *data, size_t n)
{
   if (cmd->stream_len + n > cmd->stream_cap) {
      size_t cap = MAX2(cmd->stream_cap * 2, cmd->stream_len + n + 65536);
      uint8_t *p = realloc(cmd->stream, cap);
      if (!p)
         return false;
      cmd->stream = p;
      cmd->stream_cap = cap;
   }
   memcpy(cmd->stream + cmd->stream_len, data, n);
   cmd->stream_len += n;
   return true;
}

static uint32_t
heap_upload(struct draw_heap *h, const uint8_t *data, uint32_t n)
{
   uint32_t padded = (n + 3) & ~3u;
   if (h->top + padded + 16 > BORGVK_HEAP_BYTES)
      return UINT32_MAX;
   uint32_t off = h->top;
   for (uint32_t i = 0; i < padded; i += 256) {
      uint8_t tmp[256] = { 0 };
      uint32_t c = MIN2(256u, n > i ? n - i : 0);
      if (c)
         memcpy(tmp, data + i, c);
      borgvk_serial_send_mem(off + i, tmp, MIN2(256u, padded - i));
   }
   h->top += padded + 16;   /* slack for the typed fetch's over-read */
   return off;
}

struct draw_params {
   bool          indexed;
   uint32_t      index_size;      /* 2 or 4 */
   uint32_t      count;           /* vertices or indices */
   uint32_t      instances;
   uint32_t      first;           /* first vertex or first index */
   int32_t       vertex_offset;
   uint32_t      first_instance;
};

static bool
borgvk_sim_generic_draw(struct borgvk_device *device, struct borgvk_command_buffer *cmd,
                        const struct draw_params *dp)
{
   struct borgvk_pipeline *pipeline = cmd->gfx_pipeline;
   struct borgvk_descriptor_set *set = cmd->desc_set;

   if (!pipeline || dp->count == 0 || dp->instances == 0)
      return generic_reject(REJECT_UNDRAWABLE);
   uint32_t topo;
   switch (pipeline->topology) {
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:  topo = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: topo = 1; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:   topo = 2; break;
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:     topo = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:     topo = 0; break;
   default: return generic_reject(REJECT_UNDRAWABLE);
   }
   if (pipeline->blob[BORGVK_STAGE_VERT].len == 0 || pipeline->blob[BORGVK_STAGE_FRAG].len == 0)
      return generic_reject(REJECT_NO_SHADER);
   const bool points = pipeline->topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   const uint32_t X = points ? 6 : 1;
   /* A line is a quad too, but its six rows are gathered per line (see `pairs`). */
   const bool lines = pipeline->topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
                      pipeline->topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;

   struct borgvk_image *color_img =
      cmd->color_views[0] && cmd->color_views[0]->image
         ? container_of(cmd->color_views[0]->image, struct borgvk_image, vk) : NULL;
   /* Depth only: no colour attachment, the depth or stencil image gives the size. */
   const struct vk_image_view *ds_view = cmd->depth_view ? cmd->depth_view : cmd->stencil_view;
   struct borgvk_image *size_img =
      color_img ? color_img : ds_view && ds_view->image ? container_of(ds_view->image, struct borgvk_image, vk) : NULL;
   if (!size_img || !size_img->mem || !size_img->mem->map)
      return generic_reject(REJECT_NO_TARGET_MEMORY);
   if (!color_img && !borgvk_sim_serves(size_img, true))
      return generic_reject(REJECT_NO_TARGET_MEMORY);
   struct borgvk_image *extra[3] = { NULL, NULL, NULL };
   const uint32_t nextra = pipeline->mrt ? pipeline->color_count - 1 : 0;
   if (pipeline->color_count >= 2 && !pipeline->mrt)
      return generic_reject(REJECT_MRT_FORMATS);
   for (uint32_t k = 0; k < nextra; k++) {
      struct vk_image_view *v = cmd->color_views[k + 1];
      extra[k] = v && v->image ? container_of(v->image, struct borgvk_image, vk) : NULL;
      if (!extra[k] || !extra[k]->mem || !extra[k]->mem->map ||
          extra[k]->vk.extent.width != color_img->vk.extent.width ||
          extra[k]->vk.extent.height != color_img->vk.extent.height)
         return generic_reject(REJECT_MRT_EXTENT);
   }
   if (nextra && !borgvk_sim_serves(color_img, cmd->depth_view || cmd->stencil_view))
      return generic_reject(REJECT_MRT_NOT_SERVED);
   bool same_extra = cmd->batch_nextra == nextra;
   for (uint32_t k = 0; k < nextra; k++)
      same_extra &= cmd->batch_extra[k] == extra[k];
   if (cmd->batch_draws && (!same_extra || cmd->batch_img != color_img || cmd->batch_depth != cmd->depth_view ||
                            cmd->batch_stencil != cmd->stencil_view))
      borgvk_flush_draws(cmd);

   /* The index range each binding needs. */
   const uint8_t *idx_host = NULL;
   if (dp->indexed) {
      if (!cmd->index_ptr)
         return generic_reject(REJECT_NO_INDEX_BUFFER);
      idx_host = cmd->index_ptr + (size_t)dp->first * dp->index_size;
      if ((size_t)(dp->first + dp->count) * dp->index_size > cmd->index_avail)
         return generic_reject(REJECT_NO_INDEX_BUFFER);
   }
   int64_t vmin = INT64_MAX, vmax = -1;
   if (dp->indexed) {
      const uint32_t restart_val = dp->index_size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
      for (uint32_t i = 0; i < dp->count; i++) {
         uint32_t v = dp->index_size == 2 ? ((const uint16_t *)idx_host)[i] : ((const uint32_t *)idx_host)[i];
         if (pipeline->restart && v == restart_val)
            continue;
         int64_t r = (int64_t)v + dp->vertex_offset;
         if (r < 0)
            return generic_reject(REJECT_INDEX_RANGE);
         vmin = MIN2(vmin, r);
         vmax = MAX2(vmax, r);
      }
   } else {
      vmin = dp->first;
      vmax = (int64_t)dp->first + dp->count - 1;
   }
   if (vmax < 0)
      return generic_reject(REJECT_INDEX_RANGE);
   /* Lines: the two vertex rows of each line. */
   uint32_t (*pairs)[2] = NULL, nlines = 0;
   if (lines) {
      const bool strip = pipeline->topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
      const uint32_t restart_val = dp->index_size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
      pairs = malloc(sizeof(*pairs) * MAX2(dp->count, 1));
      if (!pairs)
         return generic_reject(REJECT_UPLOAD);
      int64_t prev = -1;
      for (uint32_t i = 0; i < dp->count; i++) {
         int64_t r = (int64_t)dp->first + i;
         if (dp->indexed) {
            uint32_t v = dp->index_size == 2 ? ((const uint16_t *)idx_host)[i] : ((const uint32_t *)idx_host)[i];
            if (pipeline->restart && v == restart_val) {
               prev = -1;
               continue;
            }
            r = (int64_t)v + dp->vertex_offset;
         }
         if (prev < 0) {
            prev = r;
            continue;
         }
         pairs[nlines][0] = (uint32_t)prev;
         pairs[nlines][1] = (uint32_t)r;
         nlines++;
         prev = strip ? r : -1;
      }
      if (nlines == 0 || (uint64_t)nlines * 6 > 4096 * 4096) {
         free(pairs);
         return generic_reject(REJECT_INDEX_RANGE);
      }
   }
   static const uint8_t line_end[6] = { 0, 0, 1, 0, 1, 1 };
   static const float line_side[6] = { -1, 1, -1, 1, 1, -1 };
   int64_t imin = dp->first_instance, imax = (int64_t)dp->first_instance + dp->instances - 1;

   {
      uint64_t need = dp->indexed ? (uint64_t)dp->count * dp->index_size + 32 : 0;
      for (uint32_t b = 0; b < BORGVK_MAX_VERTEX_BINDINGS; b++) {
         bool used = false;
         for (uint32_t a = 0; a < pipeline->attr_count; a++)
            used |= pipeline->attrs[a].binding == b;
         if (!used)
            continue;
         int64_t r0 = pipeline->binding_instance[b] ? imin : vmin;
         int64_t r1 = pipeline->binding_instance[b] ? imax : vmax;
         need += (uint64_t)(r1 - r0 + 1) * X * pipeline->binding_stride[b] + 32;
      }
      if (points)
         need += (uint64_t)(vmax - vmin + 1) * X * 16 + 32;
      if (lines)
         need = (uint64_t)nlines * 6 * (16 + 2 * 16 * pipeline->attr_count) + 64;
      if (cmd->batch_draws && cmd->heap_top + need > BORGVK_HEAP_BYTES)
         borgvk_flush_draws(cmd);
      /* The constant pool's slot is one per pass. */
      if (cmd->batch_draws && pipeline->blob[BORGVK_STAGE_FRAG].npool && cmd->batch_pool && cmd->batch_pool != pipeline)
         borgvk_flush_draws(cmd);
   }
   const bool first_of_batch = cmd->batch_draws == 0;

   /* The DRM shim's ioctls bypass the capture buffer; force the serial path. */
   int saved_fd = device->drm_fd;
   device->drm_fd = -1;
   borgvk_transport_capture_begin();
   memcpy(device->shader_blob, pipeline->blob, sizeof(device->shader_blob));
   device->blend_cfg = pipeline->blend_cfg;
   device->blend_const = pipeline->blend_const;
   memcpy(device->state_reg, pipeline->state_reg, sizeof(device->state_reg));
   device->state_valid = pipeline->state_valid;
   device->vp = pipeline->vp;
   {
      const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      if (pipeline->vp_dyn && BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VP_VIEWPORTS)) {
         for (int k = 0; k < 6; k++) {
            const VkViewport *v = &dyn->vp.viewports[0];
            const float vv[6] = { v->x, v->y, v->width, v->height, v->minDepth, v->maxDepth };
            device->vp.v[k] = vv[k];
         }
         device->vp.valid = true;
      }
      if (pipeline->sc_dyn && BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VP_SCISSORS)) {
         const VkRect2D *r = &dyn->vp.scissors[0];
         device->vp.sc[0] = MAX2(r->offset.x, 0);
         device->vp.sc[1] = MAX2(r->offset.y, 0);
         device->vp.sc[2] = MAX2(r->offset.x + (int32_t)r->extent.width, 0);
         device->vp.sc[3] = MAX2(r->offset.y + (int32_t)r->extent.height, 0);
         device->vp.valid = true;
      }
   }
   borgvk_upload_shaders(device);
   borgvk_send_blend_state(device);
   size_t state_len = 0;
   uint8_t *state = borgvk_transport_capture_end(&state_len);
   const bool state_same = state && cmd->state_sent && state_len == cmd->state_sent_len &&
                           memcmp(state, cmd->state_sent, state_len) == 0;
   borgvk_transport_capture_begin();
   if (first_of_batch)
      cmd->batch_serve = borgvk_sim_serves(size_img, cmd->depth_view || cmd->stencil_view);
   const bool serve = cmd->batch_serve;
   uint32_t load = first_of_batch ? 0u : 1u;
   if (serve) {
      const bool has_z = cmd->depth_view != NULL, has_s = cmd->stencil_view != NULL;
      load = 1u | (has_z ? 2u : 0u) | (has_s ? 4u : 0u) | 8u;
      if (first_of_batch)
         borgvk_serial_send_pass(nextra ? 5 : color_img ? borgvk_sim_flush_format(color_img->vk.format) : 0,
                                 (has_z ? 1u : 0u) | (has_s ? 2u : 0u) | (color_img ? 0u : 8u) |
                                 (has_z && borgvk_sim_depth_is_d32(cmd->depth_view->image->format) ? 4u : 0u));
      if (first_of_batch) {
         const uint8_t raw8[3] = { 5, 5, 5 };
         borgvk_serial_send_att(nextra + 1, raw8);
      }
   } else if (first_of_batch) {
      uint8_t fmt = color_img->vk.format == VK_FORMAT_R8G8B8A8_UNORM ? 1 :
                    color_img->vk.format == VK_FORMAT_B8G8R8A8_UNORM ? 2 :
                    borgvk_sim_raw32(color_img->vk.format) ? 3 : 0;   /* 3 = RAW32 */
      float clear[4] = { 0, 0, 0, 0 };
      if (cmd->has_clear)
         memcpy(clear, cmd->clear_color, sizeof(clear));
      borgvk_serial_send_target(fmt, clear);
   }

   struct draw_heap heap = { cmd->heap_top };
   bool ok = true;
   for (uint32_t a = 0; ok && a < pipeline->attr_count; a++) {
      const struct borgvk_vertex_attr *at = &pipeline->attrs[a];
      uint32_t code, swz, rbytes;
      uint32_t b = at->binding;
      if (at->location >= 16 || b >= BORGVK_MAX_VERTEX_BINDINGS ||
          !borgvk_vertex_format_to_fetch(at->format, &code, &swz, &rbytes) ||
          !cmd->vb[b]) {
         ok = generic_reject(REJECT_VERTEX_INPUT);
         break;
      }
      const uint32_t stride = pipeline->binding_stride[b];
      const bool inst = pipeline->binding_instance[b];
      if (lines && !inst) {
         const uint32_t asz = vk_format_get_blocksize(at->format);
         for (uint32_t other = 0; ok && other < 2; other++) {
            uint8_t *tight = calloc((size_t)nlines * 6, rbytes);
            if (!tight) {
               ok = generic_reject(REJECT_UPLOAD);
               break;
            }
            for (uint32_t r = 0; r < nlines * 6; r++) {
               const uint64_t src = (uint64_t)pairs[r / 6][line_end[r % 6] ^ other] * stride + at->offset;
               if (src + asz <= cmd->vb_avail[b])
                  memcpy(tight + (size_t)r * rbytes, cmd->vb[b] + src, MIN2(asz, rbytes));
            }
            const uint32_t off = heap_upload(&heap, tight, nlines * 6 * rbytes);
            free(tight);
            if (off == UINT32_MAX)
               ok = generic_reject(REJECT_UPLOAD);
            else
               borgvk_serial_send_vattr(at->location + (other ? BORGVK_LINE_OTHER : 0), code, off, nlines * 6, rbytes, swz);
         }
         continue;
      }
      const int64_t row0 = inst ? imin : vmin, row1 = inst ? imax : vmax;
      const bool expand = points && !inst;
      const int64_t e0 = expand ? row0 * X : row0, e1 = expand ? row1 * X + X - 1 : row1;
      if (e1 + 1 > 4096 * 4096) {
         ok = generic_reject(REJECT_VERTEX_RANGE);
         break;
      }
      const uint32_t nrows = (uint32_t)(e1 - e0 + 1);
      const uint32_t asz = vk_format_get_blocksize(at->format);
      uint8_t *tight = calloc(nrows, rbytes);
      if (!tight) {
         ok = generic_reject(REJECT_UPLOAD);
         break;
      }
      const uint64_t avail = cmd->vb_avail[b];
      for (uint32_t r = 0; r < nrows; r++) {
         const uint64_t src = (uint64_t)(expand ? (e0 + r) / X : row0 + r) * stride + at->offset;
         if (src + asz <= avail)
            memcpy(tight + (size_t)r * rbytes, cmd->vb[b] + src, MIN2(asz, rbytes));
      }
      const uint32_t off = heap_upload(&heap, tight, nrows * rbytes);
      free(tight);
      if (off == UINT32_MAX) {
         ok = generic_reject(REJECT_UPLOAD);
         break;
      }
      const uint32_t base = off - (uint32_t)e0 * rbytes;
      const uint32_t count = (uint32_t)e1 + 1;
      borgvk_serial_send_vattr(at->location, code, base, count, rbytes, swz);
   }
   if (ok && points) {
      static const float corner[6][2] = { { -.5f, -.5f }, { .5f, -.5f }, { -.5f, .5f },
                                          { .5f, -.5f }, { .5f, .5f }, { -.5f, .5f } };
      /* (dx, dy) in clip units, then the point coordinate (u, v) of the corner. */
      uint32_t code, swz, rbytes;
      const int64_t e0 = vmin * X, e1 = vmax * X + X - 1;
      if (e1 + 1 > 4096 * 4096 || !borgvk_vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_SFLOAT, &code, &swz, &rbytes)) {
         ok = generic_reject(REJECT_VERTEX_RANGE);
      } else {
         float *c = malloc((size_t)(e1 - e0 + 1) * 16);
         if (!c) {
            ok = generic_reject(REJECT_UPLOAD);
         } else {
            for (int64_t e = e0; e <= e1; e++) {
               float *o = c + 4 * (e - e0);
               o[0] = corner[e % X][0] * 2.0f / size_img->vk.extent.width;
               o[1] = corner[e % X][1] * 2.0f / size_img->vk.extent.height;
               o[2] = corner[e % X][0] + 0.5f;
               o[3] = corner[e % X][1] + 0.5f;
            }
            const uint32_t off = heap_upload(&heap, (const uint8_t *)c, (uint32_t)(e1 - e0 + 1) * 16);
            free(c);
            if (off == UINT32_MAX)
               ok = generic_reject(REJECT_UPLOAD);
            else
               borgvk_serial_send_vattr(BORGVK_POINT_SLOT, code, off - (uint32_t)e0 * rbytes, (uint32_t)e1 + 1, rbytes, swz);
         }
      }
   }
   if (ok && pipeline->blob[BORGVK_STAGE_FRAG].npool && cmd->batch_pool != pipeline) {
      const struct borgvk_shader_blob *fb = &pipeline->blob[BORGVK_STAGE_FRAG];
      uint32_t code, swz, rbytes, words[BORGVK_POOL_MAX + 4] = { 0 };
      memcpy(words, fb->pool, fb->npool * 4);
      const uint32_t rows = (fb->npool + 3) / 4;
      const uint32_t off = heap_upload(&heap, (const uint8_t *)words, rows * 16);
      if (off == UINT32_MAX || !borgvk_vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_UINT, &code, &swz, &rbytes)) {
         ok = generic_reject(REJECT_UPLOAD);
      } else {
         borgvk_serial_send_vattr(BORGVK_POOL_SLOT, code, off, rows, rbytes, swz);
         cmd->batch_pool = pipeline;
      }
   }
   if (ok && pipeline->srgb_table && !cmd->batch_srgb) {
      float t[256];
      uint32_t code, swz, rbytes;
      for (int i = 0; i < 256; i++)
         t[i] = util_format_srgb_8unorm_to_linear_float(i);
      const uint32_t off = heap_upload(&heap, (const uint8_t *)t, sizeof(t));
      if (off == UINT32_MAX || !borgvk_vertex_format_to_fetch(VK_FORMAT_R32_SFLOAT, &code, &swz, &rbytes)) {
         ok = generic_reject(REJECT_UPLOAD);
      } else {
         borgvk_serial_send_vattr(BORGVK_SRGB_SLOT, code, off, 256, rbytes, swz);
         cmd->batch_srgb = true;
      }
   }
   if (ok && lines) {
      /* The half-pixel step in clip units for x and y, then the target's size. */
      uint32_t code, swz, rbytes;
      float *c = malloc((size_t)nlines * 6 * 16);
      if (!c || !borgvk_vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_SFLOAT, &code, &swz, &rbytes)) {
         ok = generic_reject(REJECT_UPLOAD);
      } else {
         const float w = size_img->vk.extent.width, h = size_img->vk.extent.height;
         for (uint32_t r = 0; r < nlines * 6; r++) {
            c[4 * r + 0] = line_side[r % 6] / w;
            c[4 * r + 1] = line_side[r % 6] / h;
            c[4 * r + 2] = w;
            c[4 * r + 3] = h;
         }
         const uint32_t off = heap_upload(&heap, (const uint8_t *)c, nlines * 6 * 16);
         if (off == UINT32_MAX)
            ok = generic_reject(REJECT_UPLOAD);
         else
            borgvk_serial_send_vattr(BORGVK_LINE_SLOT, code, off, nlines * 6, rbytes, swz);
      }
      free(c);
   }
   free(pairs);
   uint32_t idx_off = 0, index_code = dp->index_size == 2 ? 1 : 2, voff_out = dp->vertex_offset;
   if (ok && dp->indexed && !lines) {
      if (points) {
         uint32_t *ex = malloc((size_t)dp->count * X * 4);
         if (!ex) {
            ok = generic_reject(REJECT_UPLOAD);
         } else {
            for (uint32_t i = 0; i < dp->count; i++) {
               int64_t v = (dp->index_size == 2 ? ((const uint16_t *)idx_host)[i] : ((const uint32_t *)idx_host)[i]) +
                           (int64_t)dp->vertex_offset;
               for (uint32_t k = 0; k < X; k++)
                  ex[i * X + k] = (uint32_t)(v * X + k);
            }
            idx_off = heap_upload(&heap, (const uint8_t *)ex, dp->count * X * 4);
            free(ex);
            index_code = 2;
            voff_out = 0;
         }
      } else {
         idx_off = heap_upload(&heap, idx_host, dp->count * dp->index_size);
      }
      if (ok && idx_off == UINT32_MAX)
         ok = generic_reject(REJECT_UPLOAD);
   }
   uint32_t ubo_base = 0;
   uint32_t smask = 0;
   struct borgvk_descriptor_set *sets[4] = { set, cmd->desc_sets[1], cmd->desc_sets[2], cmd->desc_sets[3] };
   bool any_dyn = false;
   for (int si = 0; si < 4; si++) {
      if (!sets[si])
         continue;
      any_dyn |= sets[si]->dyn_mask != 0;
      for (int b = 0; b < BORGVK_MAX_BINDINGS && pipeline->set_base[si] + b < 32; b++)
         if ((sets[si]->storage_mask >> b & 1) && sets[si]->buffers[b] && sets[si]->buffers[b]->mem &&
             sets[si]->buffers[b]->mem->map)
            smask |= 1u << (pipeline->set_base[si] + b);
   }
   const uint32_t umask = pipeline->ubo_mask & ~1u;
   if (ok && (smask || umask)) {
      const uint32_t slots = 1 + util_last_bit(smask | umask);
      const bool reuse = cmd->region_set == set && cmd->batch_draws && !umask && !any_dyn && !sets[1];
      uint32_t base = cmd->region_base;
      if (!reuse) {
         base = (heap.top + 3) & ~3u;
         if (base + slots * BORGVK_SSBO_SLOT_BYTES + 16 > BORGVK_HEAP_BYTES) {
            ok = generic_reject(REJECT_UPLOAD);
         } else {
            heap.top = base + slots * BORGVK_SSBO_SLOT_BYTES + 16;
            for (int si = 0; si < 4; si++) {
               for (int b = 0; sets[si] && b < BORGVK_MAX_BINDINGS; b++) {
                  const struct borgvk_descriptor_set *ds = sets[si];
                  const int fb = pipeline->set_base[si] + b;
                  struct borgvk_buffer *ub = ds->buffers[b];
                  if (fb >= 32 || !ub || !ub->mem || !ub->mem->map)
                     continue;
                  const bool st = smask >> fb & 1;
                  if (!st && pipeline->ubo_mask && !(pipeline->ubo_mask >> fb & 1))
                     continue;
                  const VkDeviceSize *dyn = si ? cmd->dyn_offs[si] : cmd->dyn_off;
                  VkDeviceSize off = ds->offsets[b] + ((ds->dyn_mask >> b) & 1 ? dyn[b] : 0);
                  if (off >= ub->vk.size)
                     continue;
                  VkDeviceSize len = ds->ranges[b] == VK_WHOLE_SIZE || ds->ranges[b] == 0 ? ub->vk.size - off : ds->ranges[b];
                  len = MIN2(len, MIN2(ub->vk.size - off, BORGVK_SSBO_SLOT_BYTES));
                  uint8_t *host = (uint8_t *)ub->mem->map + ub->offset + off;
                  const uint32_t at = base + (st || (fb && pipeline->ubo_mask) ? (1 + fb) * BORGVK_SSBO_SLOT_BYTES : 0);
                  for (uint32_t i = 0; i < len; i += 256) {
                     uint8_t tmp[256] = { 0 };
                     memcpy(tmp, host + i, MIN2(256u, (uint32_t)len - i));
                     borgvk_serial_send_mem(at + i, tmp, (MIN2(256u, (uint32_t)len - i) + 3) & ~3u);
                  }
                  if (st && cmd->nwb < BORGVK_MAX_WRITEBACKS)
                     cmd->wb[cmd->nwb++] = (struct borgvk_writeback){ at, (uint32_t)len, host };
               }
            }
            cmd->region_set = set;
            cmd->region_base = base;
         }
      }
      ubo_base = base + 1;
   }
   for (int b = 0; ok && !smask && !umask && set && b < BORGVK_MAX_BINDINGS; b++) {
      struct borgvk_buffer *ub = set->buffers[b];
      if (!ub || !ub->mem || !ub->mem->map || (pipeline->ubo_mask && !(pipeline->ubo_mask >> b & 1)))
         continue;
      VkDeviceSize off = set->offsets[b] + ((set->dyn_mask >> b) & 1 ? cmd->dyn_off[b] : 0);
      VkDeviceSize len = set->ranges[b] == VK_WHOLE_SIZE || set->ranges[b] == 0 ? ub->vk.size - off : set->ranges[b];
      if (off >= ub->vk.size)
         break;
      len = MIN2(len, MIN2(ub->vk.size - off, 65536));
      uint32_t at = heap_upload(&heap, (const uint8_t *)ub->mem->map + ub->offset + off, (uint32_t)len);
      if (at == UINT32_MAX)
         ok = generic_reject(REJECT_UPLOAD);
      else
         ubo_base = at + 1;
      break;
   }
   if (ok) {
      int want = -1;
      if (set && pipeline->tex_binding >= 0) {
         want = set->map.n ? -2 : pipeline->tex_binding;
         for (uint32_t j = 0; j < set->map.n; j++)
            if (set->map.num[j] == pipeline->tex_binding)
               want = set->map.slot[j];
         if (want < 0 || want >= BORGVK_MAX_BINDINGS)
            want = -1;
      }
      for (int b = 0; set && b < BORGVK_MAX_BINDINGS; b++) {
         if (want >= 0 && b != want)
            continue;
         if (set->views[b] && borgvk_send_generic_texture(set->views[b], set->images[b], set->samplers[b]))
            break;
         if (set->buffer_views[b] && borgvk_send_buffer_texture(set->buffer_views[b]))
            break;
      }
      if (lines)
         borgvk_serial_send_draw(0, 0, load << 1, nlines * 6, dp->instances, 0, dp->first_instance, 0, 0, ubo_base);
      else
      borgvk_serial_send_draw(topo, dp->indexed ? index_code : 0,
                              (pipeline->restart ? 1u : 0u) | (load << 1), dp->count * X, dp->instances,
                              dp->indexed ? 0 : dp->first * X, dp->first_instance, voff_out, idx_off, ubo_base);
   }
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   device->drm_fd = saved_fd;
   if (!ok || !bytes || nbytes == 0) {
      free(bytes);
      free(state);
      return ok;
   }
   bool appended = (state_same || !state || stream_append(cmd, state, state_len)) &&
                   stream_append(cmd, bytes, nbytes);
   free(bytes);
   if (!state_same && state && appended) {
      free(cmd->state_sent);
      cmd->state_sent = state;
      cmd->state_sent_len = state_len;
   } else {
      free(state);
   }
   if (!appended)
      return false;
   cmd->heap_top = heap.top;
   cmd->batch_draws++;
   cmd->batch_img = color_img;
   cmd->batch_nextra = nextra;
   for (uint32_t k = 0; k < 3; k++)
      cmd->batch_extra[k] = extra[k];
   cmd->batch_depth = cmd->depth_view;
   cmd->batch_stencil = cmd->stencil_view;
   return true;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                       VkPipeline _pipeline)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   VK_FROM_HANDLE(borgvk_pipeline, pl, _pipeline);
   if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
      cmd->gfx_pipeline = pl;
   else if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
      cmd->cs_pipeline = pl;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                            uint32_t bindingCount, const VkBuffer *pBuffers,
                            const VkDeviceSize *pOffsets)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   for (uint32_t i = 0; i < bindingCount; i++) {
      uint32_t b = firstBinding + i;
      VK_FROM_HANDLE(borgvk_buffer, buf, pBuffers[i]);
      if (b < BORGVK_MAX_VERTEX_BINDINGS) {
         cmd->vb[b] = buf && buf->mem && buf->mem->map
            ? (const uint8_t *)buf->mem->map + buf->offset + pOffsets[i] : NULL;
         cmd->vb_avail[b] = buf && buf->vk.size > pOffsets[i] ? buf->vk.size - pOffsets[i] : 0;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                             VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
                             const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
                             const uint32_t *pDynamicOffsets)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   const bool cs = pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE;
   uint32_t k = 0;
   for (uint32_t i = 0; i < descriptorSetCount && pDescriptorSets; i++) {
      VK_FROM_HANDLE(borgvk_descriptor_set, set, pDescriptorSets[i]);
      const uint32_t si = firstSet + i;
      if (!set || si > 3)
         continue;
      VkDeviceSize *dyn = cs ? cmd->cs_dyn_offs[si] : si ? cmd->dyn_offs[si] : cmd->dyn_off;
      if (cs)
         cmd->cs_sets[si] = set;
      else if (si)
         cmd->desc_sets[si] = set;
      else
         cmd->desc_set = set;
      memset(dyn, 0, sizeof(cmd->dyn_off));
      for (uint32_t b = 0; b < BORGVK_MAX_BINDINGS && k < dynamicOffsetCount; b++)
         if (set->dyn_mask & (1ull << b))
            dyn[b] = pDynamicOffsets[k++];
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   VK_FROM_HANDLE(borgvk_buffer, buf, _buffer);
   cmd->index_ptr = NULL;
   cmd->index_avail = 0;
   cmd->index_size = indexType == VK_INDEX_TYPE_UINT16 ? 2 : indexType == VK_INDEX_TYPE_UINT32 ? 4 : 0;
   if (buf && buf->mem && buf->mem->map && buf->vk.size > offset) {
      cmd->index_ptr = (const uint8_t *)buf->mem->map + buf->offset + offset;
      cmd->index_avail = buf->vk.size - offset;
   }
}

static void
sim_draw(VkCommandBuffer commandBuffer, const struct draw_params *dp)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   struct borgvk_device *device = container_of(vk_cmd->base.device, struct borgvk_device, vk);

   /* Only the simulator renders; cube.c (UBO-driven) has its own path at submit. */
   if (!(getenv("BORGVK_SIM") || borgvk_hw_enabled()) || (cmd->desc_set && borgvk_set_is_cube(cmd->desc_set)))
      return;
   struct draw_params p = *dp;
   p.index_size = cmd->index_size;
   if (p.indexed && p.index_size == 0)
      return;
   if (borgvk_sim_generic_draw(device, cmd, &p))
      cmd->generic_drawn = true;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
               uint32_t firstVertex, uint32_t firstInstance)
{
   struct draw_params dp = { .count = vertexCount, .instances = instanceCount, .first = firstVertex,
                             .first_instance = firstInstance };
   sim_draw(commandBuffer, &dp);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                      uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
   struct draw_params dp = { .indexed = true, .count = indexCount, .instances = instanceCount,
                             .first = firstIndex, .vertex_offset = vertexOffset,
                             .first_instance = firstInstance };
   sim_draw(commandBuffer, &dp);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(borgvk_buffer, buf, _buffer);
   if (!buf || !buf->mem || !buf->mem->map)
      return;
   for (uint32_t i = 0; i < drawCount; i++) {
      VkDrawIndirectCommand c;
      memcpy(&c, (const uint8_t *)buf->mem->map + buf->offset + offset + (VkDeviceSize)i * (stride ? stride : sizeof(c)),
             sizeof(c));
      struct draw_params dp = { .count = c.vertexCount, .instances = c.instanceCount, .first = c.firstVertex,
                                .first_instance = c.firstInstance };
      sim_draw(commandBuffer, &dp);
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                              uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(borgvk_buffer, buf, _buffer);
   if (!buf || !buf->mem || !buf->mem->map)
      return;
   for (uint32_t i = 0; i < drawCount; i++) {
      VkDrawIndexedIndirectCommand c;
      memcpy(&c, (const uint8_t *)buf->mem->map + buf->offset + offset + (VkDeviceSize)i * (stride ? stride : sizeof(c)),
             sizeof(c));
      struct draw_params dp = { .indexed = true, .count = c.indexCount, .instances = c.instanceCount,
                                .first = c.firstIndex, .vertex_offset = c.vertexOffset,
                                .first_instance = c.firstInstance };
      sim_draw(commandBuffer, &dp);
   }
}
