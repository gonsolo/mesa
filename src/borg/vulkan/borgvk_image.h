/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BORGVK_IMAGE_H
#define BORGVK_IMAGE_H

#include "borgvk_private.h"

static inline uint64_t
borgvk_mip_level_size(uint32_t width, uint32_t height, uint32_t depth,
                      uint32_t level, uint32_t bs, uint32_t bw, uint32_t bh,
                      uint32_t bd, uint32_t *width_blocks_out)
{
   uint32_t w = MAX2(width >> level, 1);
   uint32_t h = MAX2(height >> level, 1);
   uint32_t d = MAX2(depth >> level, 1);
   uint32_t w_blocks = DIV_ROUND_UP(w, bw);
   uint32_t h_blocks = DIV_ROUND_UP(h, bh);
   uint32_t d_blocks = DIV_ROUND_UP(d, bd);

   if (width_blocks_out)
      *width_blocks_out = w_blocks;
   return (uint64_t)w_blocks * h_blocks * d_blocks * bs;
}

static inline uint64_t
borgvk_image_sample_plane_size(const struct borgvk_image *image, uint32_t bs,
                               uint32_t bw, uint32_t bh, uint32_t bd)
{
   uint64_t plane_size = 0;
   for (uint32_t l = 0; l < image->vk.mip_levels; l++)
      plane_size += borgvk_mip_level_size(image->vk.extent.width,
                                          image->vk.extent.height,
                                          image->vk.extent.depth,
                                          l, bs, bw, bh, bd, NULL);
   return plane_size;
}

static inline uint64_t
borgvk_image_layer_size(const struct borgvk_image *image, uint32_t bs,
                        uint32_t bw, uint32_t bh, uint32_t bd)
{
   return borgvk_image_sample_plane_size(image, bs, bw, bh, bd) *
          MAX2(image->vk.samples, 1);
}

static inline uint64_t
borgvk_image_level_offset(const struct borgvk_image *image, uint32_t level,
                          uint32_t layer, uint32_t bs, uint32_t bw, uint32_t bh,
                          uint32_t bd, uint32_t *stride_blocks_out)
{
   uint64_t layer_size = borgvk_image_layer_size(image, bs, bw, bh, bd);
   uint64_t level_offset = 0;

   for (uint32_t l = 0; l < level; l++)
      level_offset += borgvk_mip_level_size(image->vk.extent.width,
                                            image->vk.extent.height,
                                            image->vk.extent.depth,
                                            l, bs, bw, bh, bd, NULL);
   borgvk_mip_level_size(image->vk.extent.width, image->vk.extent.height,
                         image->vk.extent.depth, level, bs, bw, bh, bd,
                         stride_blocks_out);

   return (uint64_t)layer * layer_size + level_offset;
}

static inline bool
borgvk_is_z16_s8(enum pipe_format fmt)
{
   return fmt == PIPE_FORMAT_Z16_UNORM_S8_UINT;
}

static inline void
borgvk_z16_s8_pack_z(uint8_t *texel, float z)
{
   uint16_t z16 = (uint16_t)(CLAMP(z, 0.0f, 1.0f) * 65535.0f + 0.5f);
   texel[0] = z16 & 0xFF;
   texel[1] = (z16 >> 8) & 0xFF;
}

static inline float
borgvk_z16_s8_unpack_z(const uint8_t *texel)
{
   uint16_t z16 = (uint16_t)texel[0] | ((uint16_t)texel[1] << 8);
   return z16 / 65535.0f;
}

static inline bool
borgvk_is_r4g4_unorm(VkFormat fmt)
{
   return fmt == VK_FORMAT_R4G4_UNORM_PACK8;
}

static inline void
borgvk_unpack_rgba_float(VkFormat vk_fmt, enum pipe_format pfmt,
                         const uint8_t *texel, float out[4])
{
   if (borgvk_is_r4g4_unorm(vk_fmt)) {
      out[0] = ((texel[0] >> 4) & 0xF) / 15.0f;
      out[1] = (texel[0] & 0xF) / 15.0f;
      out[2] = 0.0f;
      out[3] = 1.0f;
   } else {
      util_format_unpack_rgba(pfmt, out, texel, 1);
   }
}

static inline void
borgvk_pack_rgba_float(VkFormat vk_fmt, enum pipe_format pfmt,
                       uint8_t *texel, const float in[4])
{
   if (borgvk_is_r4g4_unorm(vk_fmt)) {
      uint8_t r4 = (uint8_t)(CLAMP(in[0], 0.0f, 1.0f) * 15.0f + 0.5f);
      uint8_t g4 = (uint8_t)(CLAMP(in[1], 0.0f, 1.0f) * 15.0f + 0.5f);
      texel[0] = (uint8_t)((r4 << 4) | g4);
   } else {
      util_format_pack_rgba(pfmt, texel, in, 1);
   }
}

static inline bool
borgvk_clear_color_is_uint(enum pipe_format pfmt)
{
   if (util_format_is_pure_uint(pfmt))
      return true;
   if (!util_format_is_scaled(pfmt))
      return false;
   int ch = util_format_get_first_non_void_channel(pfmt);
   return ch >= 0 && util_format_description(pfmt)->channel[ch].type == UTIL_FORMAT_TYPE_UNSIGNED;
}

static inline bool
borgvk_clear_color_is_sint(enum pipe_format pfmt)
{
   if (util_format_is_pure_sint(pfmt))
      return true;
   if (!util_format_is_scaled(pfmt))
      return false;
   int ch = util_format_get_first_non_void_channel(pfmt);
   return ch >= 0 && util_format_description(pfmt)->channel[ch].type == UTIL_FORMAT_TYPE_SIGNED;
}

static inline void
borgvk_pack_clear_color(VkFormat vk_fmt, enum pipe_format pfmt, uint8_t *pixel,
                        const VkClearColorValue *color)
{
   int first_ch = util_format_get_first_non_void_channel(pfmt);
   if (first_ch >= 0 && util_format_is_pure_integer(pfmt) &&
       util_format_description(pfmt)->channel[first_ch].size == 64) {
      memcpy(pixel, color->uint32, vk_format_get_blocksize(vk_fmt));
      return;
   }

   if (util_format_is_pure_uint(pfmt)) {
      util_format_pack_rgba(pfmt, pixel, color->uint32, 1);
   } else if (util_format_is_pure_sint(pfmt)) {
      util_format_pack_rgba(pfmt, pixel, color->int32, 1);
   } else if (borgvk_clear_color_is_uint(pfmt)) {
      float v[4] = {(float)color->uint32[0], (float)color->uint32[1],
                    (float)color->uint32[2], (float)color->uint32[3]};
      util_format_pack_rgba(pfmt, pixel, v, 1);
   } else if (borgvk_clear_color_is_sint(pfmt)) {
      float v[4] = {(float)color->int32[0], (float)color->int32[1],
                    (float)color->int32[2], (float)color->int32[3]};
      util_format_pack_rgba(pfmt, pixel, v, 1);
   } else {
      borgvk_pack_rgba_float(vk_fmt, pfmt, pixel, color->float32);
   }
}

static inline bool
borgvk_is_partial_ds_aspect(VkFormat fmt, VkImageAspectFlags aspect)
{
   if (!vk_format_has_depth(fmt) || !vk_format_has_stencil(fmt))
      return false;
   return aspect != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
}

static inline void
borgvk_aspect_block_size(VkFormat combined_format, VkImageAspectFlags aspect,
                         uint32_t *bs_out, uint32_t *bw_out, uint32_t *bh_out)
{
   VkFormat fmt = util_bitcount(aspect) == 1 ?
      vk_format_get_aspect_format(combined_format, aspect) : combined_format;
   *bs_out = vk_format_get_blocksize(fmt);
   *bw_out = vk_format_get_blockwidth(fmt);
   *bh_out = vk_format_get_blockheight(fmt);
}

static inline VkDeviceSize
borgvk_image_size(const VkImageCreateInfo *info)
{
   uint32_t bs = vk_format_get_blocksize(info->format);
   uint32_t bw = vk_format_get_blockwidth(info->format);
   uint32_t bh = vk_format_get_blockheight(info->format);
   uint32_t bd = util_format_get_blockdepth(vk_format_to_pipe_format(info->format));
   uint64_t layer_size = 0;

   for (uint32_t l = 0; l < MAX2(info->mipLevels, 1); l++)
      layer_size += borgvk_mip_level_size(info->extent.width, info->extent.height,
                                          info->extent.depth, l, bs, bw, bh, bd, NULL);

   uint64_t size = layer_size * MAX2(info->arrayLayers, 1) * MAX2(info->samples, 1);
   return align64(size, 256);
}

#endif /* BORGVK_IMAGE_H */
