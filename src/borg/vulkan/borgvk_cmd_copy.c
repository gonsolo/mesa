/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_format.h"

#include "util/format/u_format.h"

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyImageToBuffer(VkCommandBuffer commandBuffer,
                            VkImage _srcImage, VkImageLayout srcImageLayout,
                            VkBuffer _dstBuffer,
                            uint32_t regionCount, const VkBufferImageCopy *pRegions)
{
   VK_FROM_HANDLE(borgvk_image, src, _srcImage);
   VK_FROM_HANDLE(borgvk_buffer, dst, _dstBuffer);

   if (!src || !dst || !src->mem || !dst->mem || !src->mem->map || !dst->mem->map)
      return;

   uint32_t img_bs = vk_format_get_blocksize(src->vk.format);
   uint32_t img_bw = vk_format_get_blockwidth(src->vk.format);
   uint32_t img_bh = vk_format_get_blockheight(src->vk.format);
   uint32_t img_bd = util_format_get_blockdepth(vk_format_to_pipe_format(src->vk.format));
   if (img_bd > 1)
      return;

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkBufferImageCopy *reg = &pRegions[r];

      uint32_t bs, bw, bh;
      borgvk_aspect_block_size(src->vk.format, reg->imageSubresource.aspectMask,
                               &bs, &bw, &bh);
      bool partial_aspect = borgvk_is_partial_ds_aspect(
         src->vk.format, reg->imageSubresource.aspectMask);

      uint32_t w_blocks = DIV_ROUND_UP(reg->imageExtent.width, bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->imageExtent.height, bh);
      /* 0 means tightly packed (VkBufferImageCopy spec). */
      uint32_t row_len_blocks = reg->bufferRowLength ?
         DIV_ROUND_UP(reg->bufferRowLength, bw) : w_blocks;
      uint32_t img_h_blocks = reg->bufferImageHeight ?
         DIV_ROUND_UP(reg->bufferImageHeight, bh) : h_blocks;
      uint64_t layer_pitch = (uint64_t)row_len_blocks * img_h_blocks * bs;
      bool img_3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
      uint32_t sliceCount = img_3d ?
         DIV_ROUND_UP(MAX2(reg->imageExtent.depth, 1), img_bd) :
         MAX2(reg->imageSubresource.layerCount, 1);

      for (uint32_t l = 0; l < sliceCount; l++) {
         uint32_t src_stride_blocks;
         uint64_t level_off = borgvk_image_level_offset(
            src, reg->imageSubresource.mipLevel,
            img_3d ? 0 : reg->imageSubresource.baseArrayLayer + l,
            img_bs, img_bw, img_bh, img_bd, &src_stride_blocks);
         if (img_3d)
            level_off += (uint64_t)(reg->imageOffset.z / img_bd + l) *
               borgvk_mip_level_size(src->vk.extent.width, src->vk.extent.height, 1,
                                     reg->imageSubresource.mipLevel, img_bs, img_bw, img_bh,
                                     img_bd, NULL);

         const uint8_t *s = (const uint8_t *)src->mem->map + src->offset + level_off
                          + (VkDeviceSize)(reg->imageOffset.y / img_bh) * src_stride_blocks * img_bs
                          + (VkDeviceSize)(reg->imageOffset.x / img_bw) * img_bs;
         uint8_t *d = (uint8_t *)dst->mem->map + dst->offset + reg->bufferOffset
                    + (VkDeviceSize)l * layer_pitch;

         if (bs == img_bs && !partial_aspect) {
            for (uint32_t row = 0; row < h_blocks; row++) {
               memcpy(d + (size_t)row * row_len_blocks * bs,
                      s + (size_t)row * src_stride_blocks * img_bs,
                      (size_t)w_blocks * bs);
            }
         } else if (reg->imageSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) {
            enum pipe_format img_pfmt = vk_format_to_pipe_format(src->vk.format);
            bool z16_s8 = borgvk_is_z16_s8(img_pfmt);
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  const uint8_t *src_texel =
                     s + (size_t)row * src_stride_blocks * img_bs + (size_t)col * img_bs;
                  uint8_t stencil;
                  if (z16_s8)
                     stencil = src_texel[2];
                  else
                     util_format_unpack_s_8uint(img_pfmt, &stencil, src_texel, 1);
                  d[(size_t)row * row_len_blocks * bs + (size_t)col * bs] = stencil;
               }
            }
         } else {
            enum pipe_format img_pfmt = vk_format_to_pipe_format(src->vk.format);
            enum pipe_format dst_pfmt = vk_format_to_pipe_format(
               vk_format_get_aspect_format(src->vk.format, VK_IMAGE_ASPECT_DEPTH_BIT));
            bool z16_s8 = borgvk_is_z16_s8(img_pfmt);
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  const uint8_t *src_texel =
                     s + (size_t)row * src_stride_blocks * img_bs + (size_t)col * img_bs;
                  float z = z16_s8 ? borgvk_z16_s8_unpack_z(src_texel) : 0.0f;
                  if (!z16_s8)
                     util_format_unpack_z_float(img_pfmt, &z, src_texel, 1);
                  util_format_pack_z_float(dst_pfmt,
                     d + (size_t)row * row_len_blocks * bs + (size_t)col * bs, &z, 1);
               }
            }
         }
      }
   }
   if (borgvk_corpus_on()) {
      uint64_t key = borgvk_corpus_hash(BORGVK_CORPUS_SEED, &src->vk.format, sizeof(src->vk.format));
      key = borgvk_corpus_hash(key, pRegions, regionCount * sizeof(*pRegions));
      borgvk_corpus_log('R', key, borgvk_corpus_hash(BORGVK_CORPUS_SEED,
                        (const uint8_t *)dst->mem->map + dst->offset, dst->vk.size));
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyBufferToImage(VkCommandBuffer commandBuffer,
                            VkBuffer _srcBuffer, VkImage _dstImage,
                            VkImageLayout dstImageLayout,
                            uint32_t regionCount, const VkBufferImageCopy *pRegions)
{
   VK_FROM_HANDLE(borgvk_buffer, src, _srcBuffer);
   VK_FROM_HANDLE(borgvk_image, dst, _dstImage);

   if (!src || !dst || !src->mem || !dst->mem || !src->mem->map || !dst->mem->map)
      return;

   uint32_t img_bs = vk_format_get_blocksize(dst->vk.format);
   uint32_t img_bw = vk_format_get_blockwidth(dst->vk.format);
   uint32_t img_bh = vk_format_get_blockheight(dst->vk.format);
   uint32_t img_bd = util_format_get_blockdepth(vk_format_to_pipe_format(dst->vk.format));
   if (img_bd > 1)
      return;

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkBufferImageCopy *reg = &pRegions[r];

      uint32_t bs, bw, bh;
      borgvk_aspect_block_size(dst->vk.format, reg->imageSubresource.aspectMask,
                               &bs, &bw, &bh);
      bool partial_aspect = borgvk_is_partial_ds_aspect(
         dst->vk.format, reg->imageSubresource.aspectMask);

      uint32_t w_blocks = DIV_ROUND_UP(reg->imageExtent.width, bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->imageExtent.height, bh);
      uint32_t row_len_blocks = reg->bufferRowLength ?
         DIV_ROUND_UP(reg->bufferRowLength, bw) : w_blocks;
      uint32_t img_h_blocks = reg->bufferImageHeight ?
         DIV_ROUND_UP(reg->bufferImageHeight, bh) : h_blocks;
      uint64_t layer_pitch = (uint64_t)row_len_blocks * img_h_blocks * bs;
      bool img_3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;
      uint32_t sliceCount = img_3d ?
         DIV_ROUND_UP(MAX2(reg->imageExtent.depth, 1), img_bd) :
         MAX2(reg->imageSubresource.layerCount, 1);

      for (uint32_t l = 0; l < sliceCount; l++) {
         uint32_t dst_stride_blocks;
         uint64_t level_off = borgvk_image_level_offset(
            dst, reg->imageSubresource.mipLevel,
            img_3d ? 0 : reg->imageSubresource.baseArrayLayer + l,
            img_bs, img_bw, img_bh, img_bd, &dst_stride_blocks);
         if (img_3d)
            level_off += (uint64_t)(reg->imageOffset.z / img_bd + l) *
               borgvk_mip_level_size(dst->vk.extent.width, dst->vk.extent.height, 1,
                                     reg->imageSubresource.mipLevel, img_bs, img_bw, img_bh,
                                     img_bd, NULL);

         const uint8_t *s = (const uint8_t *)src->mem->map + src->offset + reg->bufferOffset
                          + (VkDeviceSize)l * layer_pitch;
         uint8_t *d = (uint8_t *)dst->mem->map + dst->offset + level_off
                    + (VkDeviceSize)(reg->imageOffset.y / img_bh) * dst_stride_blocks * img_bs
                    + (VkDeviceSize)(reg->imageOffset.x / img_bw) * img_bs;

         if (bs == img_bs && !partial_aspect) {
            for (uint32_t row = 0; row < h_blocks; row++) {
               memcpy(d + (size_t)row * dst_stride_blocks * img_bs,
                      s + (size_t)row * row_len_blocks * bs,
                      (size_t)w_blocks * bs);
            }
         } else if (reg->imageSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) {
            enum pipe_format img_pfmt = vk_format_to_pipe_format(dst->vk.format);
            bool z16_s8 = borgvk_is_z16_s8(img_pfmt);
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  uint8_t stencil = s[(size_t)row * row_len_blocks * bs + (size_t)col * bs];
                  uint8_t *dst_texel =
                     d + (size_t)row * dst_stride_blocks * img_bs + (size_t)col * img_bs;
                  if (z16_s8)
                     dst_texel[2] = stencil;
                  else
                     util_format_pack_s_8uint(img_pfmt, dst_texel, &stencil, 1);
               }
            }
         } else {
            enum pipe_format img_pfmt = vk_format_to_pipe_format(dst->vk.format);
            enum pipe_format src_pfmt = vk_format_to_pipe_format(
               vk_format_get_aspect_format(dst->vk.format, VK_IMAGE_ASPECT_DEPTH_BIT));
            bool z16_s8 = borgvk_is_z16_s8(img_pfmt);
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  float z;
                  util_format_unpack_z_float(src_pfmt, &z,
                     s + (size_t)row * row_len_blocks * bs + (size_t)col * bs, 1);
                  uint8_t *dst_texel =
                     d + (size_t)row * dst_stride_blocks * img_bs + (size_t)col * img_bs;
                  if (z16_s8)
                     borgvk_z16_s8_pack_z(dst_texel, z);
                  else
                     util_format_pack_z_float(img_pfmt, dst_texel, &z, 1);
               }
            }
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer,
                             const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VkBufferImageCopy regions[MAX2(pCopyImageToBufferInfo->regionCount, 1)];
   for (uint32_t i = 0; i < pCopyImageToBufferInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pCopyImageToBufferInfo->pRegions[i];
      regions[i] = (VkBufferImageCopy){
         .bufferOffset = r->bufferOffset, .bufferRowLength = r->bufferRowLength,
         .bufferImageHeight = r->bufferImageHeight, .imageSubresource = r->imageSubresource,
         .imageOffset = r->imageOffset, .imageExtent = r->imageExtent,
      };
   }
   borgvk_CmdCopyImageToBuffer(commandBuffer, pCopyImageToBufferInfo->srcImage,
                               pCopyImageToBufferInfo->srcImageLayout,
                               pCopyImageToBufferInfo->dstBuffer,
                               pCopyImageToBufferInfo->regionCount, regions);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer,
                             const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo)
{
   VkBufferImageCopy regions[MAX2(pCopyBufferToImageInfo->regionCount, 1)];
   for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pCopyBufferToImageInfo->pRegions[i];
      regions[i] = (VkBufferImageCopy){
         .bufferOffset = r->bufferOffset, .bufferRowLength = r->bufferRowLength,
         .bufferImageHeight = r->bufferImageHeight, .imageSubresource = r->imageSubresource,
         .imageOffset = r->imageOffset, .imageExtent = r->imageExtent,
      };
   }
   borgvk_CmdCopyBufferToImage(commandBuffer, pCopyBufferToImageInfo->srcBuffer,
                               pCopyBufferToImageInfo->dstImage,
                               pCopyBufferToImageInfo->dstImageLayout,
                               pCopyBufferToImageInfo->regionCount, regions);
}

void
borgvk_resolve_region(struct borgvk_image *src, struct borgvk_image *dst,
                      const VkImageResolve *reg)
{
   if (!src->mem || !dst->mem || !src->mem->map || !dst->mem->map)
      return;

   enum pipe_format src_pfmt = vk_format_to_pipe_format(src->vk.format);
   enum pipe_format dst_pfmt = vk_format_to_pipe_format(dst->vk.format);
   uint32_t src_bs = vk_format_get_blocksize(src->vk.format);
   uint32_t dst_bs = vk_format_get_blocksize(dst->vk.format);
   uint32_t samples = MAX2(src->vk.samples, 1);

   uint32_t layerCount = vk_image_subresource_layer_count(&src->vk, &reg->srcSubresource);

   for (uint32_t l = 0; l < layerCount; l++) {
      uint32_t src_stride, dst_stride;
      uint64_t src_off = borgvk_image_level_offset(
         src, reg->srcSubresource.mipLevel,
         reg->srcSubresource.baseArrayLayer + l,
         src_bs, 1, 1, 1, &src_stride);
      uint64_t dst_off = borgvk_image_level_offset(
         dst, reg->dstSubresource.mipLevel,
         reg->dstSubresource.baseArrayLayer + l,
         dst_bs, 1, 1, 1, &dst_stride);
      uint64_t plane_size = borgvk_image_sample_plane_size(src, src_bs, 1, 1, 1);

      const uint8_t *src_base = (const uint8_t *)src->mem->map + src->offset + src_off;
      uint8_t *dst_base = (uint8_t *)dst->mem->map + dst->offset + dst_off;

      for (uint32_t y = 0; y < reg->extent.height; y++) {
         uint32_t sy = reg->srcOffset.y + y, dy = reg->dstOffset.y + y;
         for (uint32_t x = 0; x < reg->extent.width; x++) {
            uint32_t sx = reg->srcOffset.x + x, dx = reg->dstOffset.x + x;
            uint8_t *dtexel = dst_base + ((uint64_t)dy * dst_stride + dx) * dst_bs;

            if (util_format_is_pure_integer(src_pfmt) && src_bs == dst_bs) {
               memcpy(dtexel, src_base + ((uint64_t)sy * src_stride + sx) * src_bs, src_bs);
               continue;
            }
            float acc[4] = {0, 0, 0, 0};
            for (uint32_t s = 0; s < samples; s++) {
               const uint8_t *stexel = src_base + (uint64_t)s * plane_size +
                  ((uint64_t)sy * src_stride + sx) * src_bs;
               float v[4];
               borgvk_unpack_rgba_float(src->vk.format, src_pfmt, stexel, v);
               acc[0] += v[0]; acc[1] += v[1]; acc[2] += v[2]; acc[3] += v[3];
            }
            acc[0] /= samples; acc[1] /= samples; acc[2] /= samples; acc[3] /= samples;
            borgvk_pack_rgba_float(dst->vk.format, dst_pfmt, dtexel, acc);
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdResolveImage(VkCommandBuffer commandBuffer,
                       VkImage srcImage, VkImageLayout srcImageLayout,
                       VkImage dstImage, VkImageLayout dstImageLayout,
                       uint32_t regionCount, const VkImageResolve *pRegions)
{
   VK_FROM_HANDLE(borgvk_image, src, srcImage);
   VK_FROM_HANDLE(borgvk_image, dst, dstImage);

   if (!src || !dst)
      return;

   for (uint32_t r = 0; r < regionCount; r++)
      borgvk_resolve_region(src, dst, &pRegions[r]);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdResolveImage2(VkCommandBuffer commandBuffer,
                        const VkResolveImageInfo2 *pResolveImageInfo)
{
   VkImageResolve regions[MAX2(pResolveImageInfo->regionCount, 1)];
   for (uint32_t i = 0; i < pResolveImageInfo->regionCount; i++) {
      const VkImageResolve2 *r = &pResolveImageInfo->pRegions[i];
      regions[i] = (VkImageResolve){
         .srcSubresource = r->srcSubresource, .srcOffset = r->srcOffset,
         .dstSubresource = r->dstSubresource, .dstOffset = r->dstOffset,
         .extent = r->extent,
      };
   }
   borgvk_CmdResolveImage(commandBuffer, pResolveImageInfo->srcImage,
                          pResolveImageInfo->srcImageLayout,
                          pResolveImageInfo->dstImage,
                          pResolveImageInfo->dstImageLayout,
                          pResolveImageInfo->regionCount, regions);
}
