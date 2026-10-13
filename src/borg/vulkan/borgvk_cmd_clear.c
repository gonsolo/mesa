/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_format.h"

#include "borgvk_image.h"

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage _image,
                          VkImageLayout imageLayout,
                          const VkClearColorValue *pColor,
                          uint32_t rangeCount, const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(borgvk_image, image, _image);

   if (!image || !image->mem || !image->mem->map)
      return;

   enum pipe_format pfmt = vk_format_to_pipe_format(image->vk.format);
   uint32_t bs = vk_format_get_blocksize(image->vk.format);
   uint32_t bw = vk_format_get_blockwidth(image->vk.format);
   uint32_t bh = vk_format_get_blockheight(image->vk.format);
   uint32_t bd = util_format_get_blockdepth(pfmt);
   if (bd > 1)
      return;

   uint8_t pixel[16]; /* largest uncompressed colour format block is 16 B */
   borgvk_pack_clear_color(image->vk.format, pfmt, pixel, pColor);

   for (uint32_t r = 0; r < rangeCount; r++) {
      const VkImageSubresourceRange *range = &pRanges[r];
      uint32_t levelCount = vk_image_subresource_level_count(&image->vk, range);
      uint32_t layerCount = vk_image_subresource_layer_count(&image->vk, range);

      for (uint32_t lv = 0; lv < levelCount; lv++) {
         uint32_t level = range->baseMipLevel + lv;
         uint32_t stride_blocks;
         uint32_t h_blocks = DIV_ROUND_UP(MAX2(image->vk.extent.height >> level, 1), bh);
         uint32_t d = DIV_ROUND_UP(MAX2(image->vk.extent.depth >> level, 1), bd);

         for (uint32_t l = 0; l < layerCount; l++) {
            uint32_t layer = range->baseArrayLayer + l;
            uint64_t off = borgvk_image_level_offset(image, level, layer, bs, bw, bh, bd,
                                                     &stride_blocks);
            uint64_t plane_size = borgvk_image_sample_plane_size(image, bs, bw, bh, bd);
            for (uint32_t s = 0; s < MAX2(image->vk.samples, 1); s++) {
               uint8_t *dst = (uint8_t *)image->mem->map + image->offset + off +
                  (uint64_t)s * plane_size;

               for (uint64_t t = 0; t < (uint64_t)stride_blocks * h_blocks * d; t++)
                  memcpy(dst + t * bs, pixel, bs);
            }
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage _image,
                                 VkImageLayout imageLayout,
                                 const VkClearDepthStencilValue *pDepthStencil,
                                 uint32_t rangeCount, const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(borgvk_image, image, _image);

   if (!image || !image->mem || !image->mem->map)
      return;

   enum pipe_format pfmt = vk_format_to_pipe_format(image->vk.format);
   uint32_t bs = vk_format_get_blocksize(image->vk.format);
   uint32_t bw = vk_format_get_blockwidth(image->vk.format);
   uint32_t bh = vk_format_get_blockheight(image->vk.format);
   uint32_t bd = util_format_get_blockdepth(pfmt);
   if (bd > 1)
      return;
   uint8_t stencil8 = (uint8_t)pDepthStencil->stencil;

   for (uint32_t r = 0; r < rangeCount; r++) {
      const VkImageSubresourceRange *range = &pRanges[r];
      uint32_t levelCount = vk_image_subresource_level_count(&image->vk, range);
      uint32_t layerCount = vk_image_subresource_layer_count(&image->vk, range);

      for (uint32_t lv = 0; lv < levelCount; lv++) {
         uint32_t level = range->baseMipLevel + lv;
         uint32_t stride_blocks;
         uint32_t h_blocks = DIV_ROUND_UP(MAX2(image->vk.extent.height >> level, 1), bh);
         uint32_t d = DIV_ROUND_UP(MAX2(image->vk.extent.depth >> level, 1), bd);

         for (uint32_t l = 0; l < layerCount; l++) {
            uint32_t layer = range->baseArrayLayer + l;
            uint64_t off = borgvk_image_level_offset(image, level, layer, bs, bw, bh, bd,
                                                     &stride_blocks);
            uint64_t texel_count = (uint64_t)stride_blocks * h_blocks * d;

            uint64_t plane_size = borgvk_image_sample_plane_size(image, bs, bw, bh, bd);
            for (uint32_t s = 0; s < MAX2(image->vk.samples, 1); s++) {
               uint8_t *base = (uint8_t *)image->mem->map + image->offset + off +
                  (uint64_t)s * plane_size;

               for (uint64_t t = 0; t < texel_count; t++) {
                  uint8_t *texel = base + t * bs;
                  if (borgvk_is_z16_s8(pfmt)) {
                     if (range->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
                        borgvk_z16_s8_pack_z(texel, pDepthStencil->depth);
                     if (range->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
                        texel[2] = stencil8;
                     continue;
                  }
                  if (range->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
                     util_format_pack_z_float(pfmt, texel, &pDepthStencil->depth, 1);
                  if (range->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
                     util_format_pack_s_8uint(pfmt, texel, &stencil8, 1);
               }
            }
         }
      }
   }
}

void
borgvk_clear_attachment_rect(struct vk_image_view *view, VkImageAspectFlags aspectMask,
                             const VkClearValue *clearValue, VkRect2D rect,
                             uint32_t baseLayer, uint32_t layerCount)
{
   if (!view)
      return;

   struct borgvk_image *image = container_of(view->image, struct borgvk_image, vk);
   if (!image->mem || !image->mem->map)
      return;

   enum pipe_format pfmt = vk_format_to_pipe_format(image->vk.format);
   uint32_t bs = vk_format_get_blocksize(image->vk.format);
   uint32_t bw = vk_format_get_blockwidth(image->vk.format);
   uint32_t bh = vk_format_get_blockheight(image->vk.format);
   uint32_t bd = util_format_get_blockdepth(pfmt);
   if (bd > 1)
      return;

   uint32_t level = view->base_mip_level;
   uint32_t x0 = (uint32_t)rect.offset.x, y0 = (uint32_t)rect.offset.y;
   uint32_t w = rect.extent.width, h = rect.extent.height;

   uint8_t pixel[16];
   uint8_t stencil8 = (uint8_t)clearValue->depthStencil.stencil;
   if (aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
      borgvk_pack_clear_color(image->vk.format, pfmt, pixel, &clearValue->color);
   }

   for (uint32_t l = 0; l < layerCount; l++) {
      uint32_t layer = view->base_array_layer + baseLayer + l;
      uint32_t stride_texels;
      uint64_t off = borgvk_image_level_offset(image, level, layer, bs, bw, bh, bd,
                                               &stride_texels);

      uint64_t plane_size = borgvk_image_sample_plane_size(image, bs, bw, bh, bd);
      for (uint32_t s = 0; s < MAX2(image->vk.samples, 1); s++) {
         uint8_t *level_base = (uint8_t *)image->mem->map + image->offset + off +
            (uint64_t)s * plane_size;

         for (uint32_t y = y0; y < y0 + h; y++) {
            uint8_t *row = level_base + (uint64_t)y * stride_texels * bs;
            for (uint32_t x = x0; x < x0 + w; x++) {
               uint8_t *texel = row + (uint64_t)x * bs;

               if (aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
                  memcpy(texel, pixel, bs);
                  continue;
               }
               if (borgvk_is_z16_s8(pfmt)) {
                  if (aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
                     borgvk_z16_s8_pack_z(texel, clearValue->depthStencil.depth);
                  if (aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
                     texel[2] = stencil8;
                  continue;
               }
               if (aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
                  util_format_pack_z_float(pfmt, texel, &clearValue->depthStencil.depth, 1);
               if (aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
                  util_format_pack_s_8uint(pfmt, texel, &stencil8, 1);
            }
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                           const VkClearAttachment *pAttachments,
                           uint32_t rectCount, const VkClearRect *pRects)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd, struct borgvk_command_buffer, vk);

   borgvk_flush_draws(cmd);   /* the draws so far come first */
   if (!cmd->in_rendering)
      return;

   for (uint32_t a = 0; a < attachmentCount; a++) {
      const VkClearAttachment *att = &pAttachments[a];
      struct vk_image_view *view = NULL;
      VkImageAspectFlags aspect = att->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT;

      if (aspect) {
         if (att->colorAttachment >= cmd->color_attachment_count)
            continue;
         view = cmd->color_views[att->colorAttachment];
         for (uint32_t r = 0; r < rectCount; r++)
            borgvk_clear_attachment_rect(view, aspect, &att->clearValue, pRects[r].rect,
                                         pRects[r].baseArrayLayer, pRects[r].layerCount);
         continue;
      }

      for (uint32_t r = 0; r < rectCount; r++) {
         if (att->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
            borgvk_clear_attachment_rect(cmd->depth_view, VK_IMAGE_ASPECT_DEPTH_BIT,
                                         &att->clearValue, pRects[r].rect,
                                         pRects[r].baseArrayLayer, pRects[r].layerCount);
         if (att->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
            borgvk_clear_attachment_rect(cmd->stencil_view, VK_IMAGE_ASPECT_STENCIL_BIT,
                                         &att->clearValue, pRects[r].rect,
                                         pRects[r].baseArrayLayer, pRects[r].layerCount);
      }
   }
}
