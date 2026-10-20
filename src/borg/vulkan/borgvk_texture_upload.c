/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_format.h"

void
borgvk_send_texture_row(struct borgvk_image *tex, const struct borgvk_sampler *sampler,
                         int dy)
{
   const uint8_t *base = (const uint8_t *)tex->mem->map + tex->offset;
   uint32_t sw = tex->vk.extent.width, sh = tex->vk.extent.height;
   if (sw == 0 || sh == 0)
      return;
   uint32_t pitch = sw * 4;                 /* RGBA8, linear tiling */
   int sxs = (int)(sw / BORGVK_TEX_DIM); if (sxs < 1) sxs = 1;
   int sys = (int)(sh / BORGVK_TEX_DIM); if (sys < 1) sys = 1;

   uint8_t row[BORGVK_TEX_DIM * 4];
   for (int dx = 0; dx < BORGVK_TEX_DIM; dx++) {
      uint32_t sum[4] = { 0, 0, 0, 0 }, n = 0;
      for (int oy = 0; oy < sys; oy++) {
         uint32_t sy = (uint32_t)dy * sys + oy;
         if (sy >= sh) break;
         for (int ox = 0; ox < sxs; ox++) {
            uint32_t sx = (uint32_t)dx * sxs + ox;
            if (sx >= sw) break;
            const uint8_t *p = base + sy * pitch + sx * 4;
            for (int c = 0; c < 4; c++)
               sum[c] += p[c];
            n++;
         }
      }
      if (n == 0) n = 1;
      for (int c = 0; c < 4; c++)
         row[dx * 4 + c] = (uint8_t)((sum[c] + n / 2) / n);
   }
   borgvk_serial_send_tex_row(dy, row, borgvk_sampler_desc(sampler));
}

bool
borgvk_send_generic_texture(const struct vk_image_view *view,
                             const struct borgvk_image *img,
                             const struct borgvk_sampler *sampler)
{
   if (!img || !img->mem || !img->mem->map)
      return false;
   if (vk_format_get_blockwidth(img->vk.format) > 1 || vk_format_get_blockheight(img->vk.format) > 1) {
      struct borgvk_image dimg;
      struct borgvk_device_memory dmem;
      void *buf = borgvk_decode_compressed_image(img, &dimg, &dmem);
      if (!buf)
         return false;
      struct vk_image_view dview = *view;
      dview.format = dimg.vk.format;
      bool ok = borgvk_send_generic_texture(&dview, &dimg, sampler);
      free(buf);
      return ok;
   }
   uint32_t w = img->vk.extent.width, h = img->vk.extent.height;
   if (w == 0 || h == 0 || w > 4096 || h > 4096)
      return false;
   uint32_t type;
   const bool stencil = view->aspects & VK_IMAGE_ASPECT_STENCIL_BIT;
   uint32_t layers = 1, depth = 1;
   switch (view->view_type) {
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY: layers = view->layer_count; type = 0; h = 1; break;
   case VK_IMAGE_VIEW_TYPE_1D: type = 0; h = 1; break;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY: layers = view->layer_count; type = 1; break;
   case VK_IMAGE_VIEW_TYPE_2D: type = 1; break;
   case VK_IMAGE_VIEW_TYPE_CUBE: type = 3; layers = 6; break;
   case VK_IMAGE_VIEW_TYPE_3D: type = 2; depth = img->vk.extent.depth; break;
   default: return false;
   }
   if (layers == 0 || layers > 1024 || depth > 1024)
      return false;

   uint32_t fmt_code;
   struct tex_conv tc;
   if (!borgvk_tex_format(view->format, stencil, &fmt_code, &tc))
      return false;
   const uint32_t obpp = tc.obpp, sbpp = tc.zstride ? tc.zstride : tc.bpp;
   const uint32_t base = view->base_mip_level;
   const uint32_t levels = base < img->vk.mip_levels ? MIN2(view->level_count, img->vk.mip_levels - base) : 0;
   if (levels == 0 || levels > 13)
      return false;

   uint32_t lw[13], lh[13], ld[13];
   uint64_t soff[13], sum = 0;
   for (uint32_t l = 0; l < img->vk.mip_levels; l++) {
      uint32_t cw = MAX2(img->vk.extent.width >> l, 1), ch = type ? MAX2(img->vk.extent.height >> l, 1) : 1;
      uint32_t cd = type == 2 ? MAX2(img->vk.extent.depth >> l, 1) : 1;
      if (l >= base && l < base + levels) {
         lw[l - base] = cw;
         lh[l - base] = ch;
         ld[l - base] = cd;
         soff[l - base] = sum;
      }
      sum += (uint64_t)cw * ch * cd * sbpp;
   }
   w = lw[0];
   h = lh[0];
   if (type == 2)
      depth = ld[0];
   const uint8_t *src = (const uint8_t *)img->mem->map + img->offset;

   const bool tiled = levels > 1 || layers > 1 || depth > 1 || type == 3;
   uint32_t offs[12] = { 0 }, ltotal = 0, total = 0;
   uint8_t *out = NULL;
   if (!tiled) {
      total = align(w * h * obpp, 4);
      out = calloc(1, total);
      if (out)
         borgvk_tex_convert(&tc, out, src + (uint64_t)view->base_array_layer * sum + soff[0], w * h);
   } else {
      uint32_t lsz[13];
      for (uint32_t l = 0; l < levels; l++) {
         lsz[l] = ((lw[l] + 3) / 4) * ((lh[l] + 3) / 4) * 16 * obpp * ld[l];
         if (l)
            offs[l - 1] = ltotal;
         ltotal += lsz[l];
      }
      total = ltotal * layers;
      /* the board's texture region is 512 KB, the direct simulator's 6 MB */
      const char *direct = getenv("BORGVK_SIM_DIRECT");
      out = total <= (direct && direct[0] ? 0x600000u : 0x70000u) ? calloc(1, total) : NULL;
      uint8_t *lin = out ? malloc((size_t)w * h * depth * obpp) : NULL;
      if (!lin) {
         free(out);
         return false;
      }
      for (uint32_t layer = 0; layer < layers; layer++)
         for (uint32_t l = 0; l < levels; l++) {
            const uint32_t tw = (lw[l] + 3) / 4, th = (lh[l] + 3) / 4;
            uint8_t *lev = out + (size_t)layer * ltotal + (l ? offs[l - 1] : 0);
            borgvk_tex_convert(&tc, lin, src + (uint64_t)(view->base_array_layer + layer) * sum + soff[l],
                        lw[l] * lh[l] * ld[l]);
            for (uint32_t z = 0; z < ld[l]; z++)
               for (uint32_t y = 0; y < lh[l]; y++)
                  for (uint32_t x = 0; x < lw[l]; x++) {
                     const uint8_t *t = lin + (((size_t)z * lh[l] + y) * lw[l] + x) * obpp;
                     uint8_t *sl = lev + (size_t)z * tw * th * 16 * obpp;
                     const uint32_t tile = (y >> 2) * tw + (x >> 2), ti = (y & 3) * 4 + (x & 3);
                     if (obpp == 16) {
                        memcpy(sl + tile * 256 + ti * 8, t, 8);
                        memcpy(sl + tile * 256 + 128 + ti * 8, t + 8, 8);
                     } else {
                        memcpy(sl + ((size_t)tile * 16 + ti) * obpp, t, obpp);
                     }
                  }
         }
      free(lin);
   }
   if (!out)
      return false;

   uint32_t sw[4] = { view->swizzle.r, view->swizzle.g, view->swizzle.b, view->swizzle.a };
   const bool no_alpha = tc.sch && !util_format_has_alpha(vk_format_to_pipe_format(view->format));
   for (uint32_t i = 0; no_alpha && i < 4; i++) {
      const uint32_t src = sw[i] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R + i : sw[i];
      if (src == VK_COMPONENT_SWIZZLE_A)
         sw[i] = VK_COMPONENT_SWIZZLE_ONE;
   }
   uint32_t desc[3];
   desc[0] = (w - 1) | ((h - 1) << 16) | (type << 28) | (tiled ? 0 : 1u << 30);
   desc[1] = ((type == 2 ? depth : layers) - 1) | (fmt_code << 14) | ((levels - 1) << 10) |
             ((uint32_t)(sw[0] & 7) << 20) |
             ((uint32_t)(sw[1] & 7) << 23) |
             ((uint32_t)(sw[2] & 7) << 26) |
             ((uint32_t)(sw[3] & 7) << 29);
   desc[2] = tiled ? ltotal : w * obpp;
   if (levels > 1)
      borgvk_serial_send_texture_levels(offs);
   for (uint32_t off = 0; off < total; off += BORGVK_TEXG_CHUNK) {
      uint32_t n = total - off < BORGVK_TEXG_CHUNK ? total - off : BORGVK_TEXG_CHUNK;
      borgvk_serial_send_texture_chunk(off, out + off, n, desc, borgvk_sampler_desc(sampler));
   }
   free(out);
   return true;
}

bool
borgvk_send_buffer_texture(const struct vk_buffer_view *bv)
{
   uint32_t fmt_code;
   switch (bv->format) {
   case VK_FORMAT_R32_UINT:   fmt_code = 33; break;   /* BORG_TEX_FORMAT_R32_UINT */
   case VK_FORMAT_R32_SFLOAT: fmt_code = 35; break;
   default: return false;
   }
   struct borgvk_buffer *buf = container_of(bv->buffer, struct borgvk_buffer, vk);
   if (!buf->mem || !buf->mem->map)
      return false;
   uint32_t w = (uint32_t)(bv->range / 4);
   if (w == 0 || w > 65536)
      return false;
   uint32_t total = w * 4;
   const uint8_t *src = (const uint8_t *)buf->mem->map + buf->offset + bv->offset;
   uint32_t desc[3];
   desc[0] = (w - 1) | (0u << 28) | (1u << 30);       /* 1D, linear */
   desc[1] = fmt_code << 14;                          /* identity swizzle */
   desc[2] = total;
   for (uint32_t off = 0; off < total; off += BORGVK_TEXG_CHUNK) {
      uint32_t n = total - off < BORGVK_TEXG_CHUNK ? total - off : BORGVK_TEXG_CHUNK;
      borgvk_serial_send_texture_chunk(off, src + off, n, desc, borgvk_sampler_desc(NULL));
   }
   return true;
}
