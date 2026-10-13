/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_sampler.h"

#include "util/u_math.h"

#include "drm-uapi/borg_drm.h"

#include <math.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "borgvk_drm_compat.h"

/* ---- Device memory ---------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_AllocateMemory(VkDevice _device,
                      const VkMemoryAllocateInfo *pAllocateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_device_memory *mem;

   mem = vk_device_memory_create(&device->vk, pAllocateInfo,
                                 pAllocator, sizeof(*mem));
   if (!mem)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   size_t alloc_size = MAX2(pAllocateInfo->allocationSize, 1);
   mem->size = alloc_size;
   mem->gem_handle = 0;

   if (device->drm_fd >= 0) {
      struct drm_borg_gem_create create = { .size = (__u64)alloc_size };
      if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_GEM_CREATE, &create) < 0) {
         vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      struct drm_borg_gem_mmap mmap_arg = { .handle = create.handle };
      if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_GEM_MMAP, &mmap_arg) < 0) {
         struct drm_gem_close cl = { .handle = create.handle };
         drmIoctl(device->drm_fd, DRM_IOCTL_GEM_CLOSE, &cl);
         vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      mem->map = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, device->drm_fd, (off_t)mmap_arg.offset);
      if (mem->map == MAP_FAILED) {
         struct drm_gem_close cl = { .handle = create.handle };
         drmIoctl(device->drm_fd, DRM_IOCTL_GEM_CLOSE, &cl);
         vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      mem->gem_handle = create.handle;
   } else {
      mem->map = malloc(alloc_size);
      if (!mem->map) {
         vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   *pMem = borgvk_device_memory_to_handle(mem);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_FreeMemory(VkDevice _device, VkDeviceMemory _mem,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_device_memory, mem, _mem);

   if (!mem)
      return;

   if (mem->gem_handle != 0) {
      munmap(mem->map, mem->size);
      struct drm_gem_close cl = { .handle = mem->gem_handle };
      drmIoctl(device->drm_fd, DRM_IOCTL_GEM_CLOSE, &cl);
   } else {
      free(mem->map);
   }
   vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
}

/* No memory type is lazily allocated, so everything allocated is committed. */
VKAPI_ATTR void VKAPI_CALL
borgvk_GetDeviceMemoryCommitment(VkDevice _device, VkDeviceMemory _memory,
                                 VkDeviceSize *pCommittedMemoryInBytes)
{
   VK_FROM_HANDLE(borgvk_device_memory, mem, _memory);
   *pCommittedMemoryInBytes = mem ? mem->vk.size : 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_MapMemory2(VkDevice _device, const VkMemoryMapInfo *pMemoryMapInfo,
                  void **ppData)
{
   VK_FROM_HANDLE(borgvk_device_memory, mem, pMemoryMapInfo->memory);
   *ppData = (char *)mem->map + pMemoryMapInfo->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_UnmapMemory2(VkDevice _device, const VkMemoryUnmapInfo *pMemoryUnmapInfo)
{
   /* Coherent host memory stays valid; nothing to do. */
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_FlushMappedMemoryRanges(VkDevice _device, uint32_t count,
                               const VkMappedMemoryRange *pRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t count,
                                    const VkMappedMemoryRange *pRanges)
{
   return VK_SUCCESS;
}

/* ---- Buffers ---------------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateBuffer(VkDevice _device, const VkBufferCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_buffer *buffer;

   buffer = vk_buffer_create(&device->vk, pCreateInfo, pAllocator,
                             sizeof(*buffer));
   if (!buffer)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pBuffer = borgvk_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyBuffer(VkDevice _device, VkBuffer _buffer,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_buffer, buffer, _buffer);

   if (!buffer)
      return;

   vk_buffer_destroy(&device->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator, VkBufferView *pView)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_buffer_view *view;

   view = vk_buffer_view_create(&device->vk, pCreateInfo, pAllocator,
                                sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pView = borgvk_buffer_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyBufferView(VkDevice _device, VkBufferView _bufferView,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_buffer_view, view, _bufferView);

   if (!view)
      return;

   vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
}

/* The common GetBufferMemoryRequirements2 delegates here (maintenance4). */
VKAPI_ATTR void VKAPI_CALL
borgvk_GetDeviceBufferMemoryRequirements(
   VkDevice _device,
   const VkDeviceBufferMemoryRequirements *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   uint64_t size = pInfo->pCreateInfo->size;
   pMemoryRequirements->memoryRequirements.size =
      size > UINT64_MAX - 255 ? UINT64_MAX : align64(size, 256);
   pMemoryRequirements->memoryRequirements.alignment = 256;
   pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x1;

   vk_foreach_struct(ext_stype, ext, pMemoryRequirements->pNext) {
      switch (ext_stype) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = (void *)ext;
         dedicated->prefersDedicatedAllocation = false;
         dedicated->requiresDedicatedAllocation = false;
         break;
      }
      default:
         break;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_BindBufferMemory2(VkDevice _device, uint32_t bindInfoCount,
                         const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(borgvk_buffer, buffer, pBindInfos[i].buffer);
      VK_FROM_HANDLE(borgvk_device_memory, mem, pBindInfos[i].memory);
      buffer->mem = mem;
      buffer->offset = pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                     VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(borgvk_buffer, buffer, _buffer);

   if (!buffer || !buffer->mem || !buffer->mem->map)
      return;

   VkDeviceSize fill_size = size == VK_WHOLE_SIZE ? buffer->vk.size - dstOffset : size;
   uint8_t *dst = (uint8_t *)buffer->mem->map + buffer->offset + dstOffset;

   for (VkDeviceSize i = 0; i < fill_size / 4; i++)
      ((uint32_t *)dst)[i] = data;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer _dstBuffer,
                       VkDeviceSize dstOffset, VkDeviceSize dataSize,
                       const void *pData)
{
   VK_FROM_HANDLE(borgvk_buffer, buffer, _dstBuffer);

   if (!buffer || !buffer->mem || !buffer->mem->map || !pData)
      return;

   memcpy((uint8_t *)buffer->mem->map + buffer->offset + dstOffset, pData, dataSize);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer _srcBuffer,
                     VkBuffer _dstBuffer, uint32_t regionCount,
                     const VkBufferCopy *pRegions)
{
   VK_FROM_HANDLE(borgvk_buffer, src, _srcBuffer);
   VK_FROM_HANDLE(borgvk_buffer, dst, _dstBuffer);

   if (!src || !dst || !src->mem || !dst->mem || !src->mem->map || !dst->mem->map)
      return;

   for (uint32_t i = 0; i < regionCount; i++) {
      const VkBufferCopy *r = &pRegions[i];
      memcpy((uint8_t *)dst->mem->map + dst->offset + r->dstOffset,
             (const uint8_t *)src->mem->map + src->offset + r->srcOffset,
             (size_t)r->size);
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo)
{
   VkBufferCopy regions[MAX2(pCopyBufferInfo->regionCount, 1)];
   for (uint32_t i = 0; i < pCopyBufferInfo->regionCount; i++) {
      const VkBufferCopy2 *r = &pCopyBufferInfo->pRegions[i];
      regions[i] = (VkBufferCopy){
         .srcOffset = r->srcOffset, .dstOffset = r->dstOffset, .size = r->size,
      };
   }
   borgvk_CmdCopyBuffer(commandBuffer, pCopyBufferInfo->srcBuffer, pCopyBufferInfo->dstBuffer,
                        pCopyBufferInfo->regionCount, regions);
}

/* ---- Images ----------------------------------------------------------- */

static uint64_t
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

static uint64_t
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

static uint64_t
borgvk_image_layer_size(const struct borgvk_image *image, uint32_t bs,
                        uint32_t bw, uint32_t bh, uint32_t bd)
{
   return borgvk_image_sample_plane_size(image, bs, bw, bh, bd) *
          MAX2(image->vk.samples, 1);
}

static uint64_t
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

static bool
borgvk_is_partial_ds_aspect(VkFormat fmt, VkImageAspectFlags aspect)
{
   if (!vk_format_has_depth(fmt) || !vk_format_has_stencil(fmt))
      return false;
   return aspect != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
}

static void
borgvk_aspect_block_size(VkFormat combined_format, VkImageAspectFlags aspect,
                         uint32_t *bs_out, uint32_t *bw_out, uint32_t *bh_out)
{
   VkFormat fmt = util_bitcount(aspect) == 1 ?
      vk_format_get_aspect_format(combined_format, aspect) : combined_format;
   *bs_out = vk_format_get_blocksize(fmt);
   *bw_out = vk_format_get_blockwidth(fmt);
   *bh_out = vk_format_get_blockheight(fmt);
}

static VkDeviceSize
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

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_image *image;

   image = vk_image_create(&device->vk, pCreateInfo, pAllocator,
                           sizeof(*image));
   if (!image)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->size = borgvk_image_size(pCreateInfo);

   *pImage = borgvk_image_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyImage(VkDevice _device, VkImage _image,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_image, image, _image);

   if (!image)
      return;

   vk_image_destroy(&device->vk, pAllocator, &image->vk);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetImageMemoryRequirements2(VkDevice _device,
                                   const VkImageMemoryRequirementsInfo2 *pInfo,
                                   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(borgvk_image, image, pInfo->image);

   pMemoryRequirements->memoryRequirements.size = image->size;
   pMemoryRequirements->memoryRequirements.alignment = 256;
   pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x1;

   vk_foreach_struct(ext_stype, ext, pMemoryRequirements->pNext) {
      switch (ext_stype) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = (void *)ext;
         dedicated->prefersDedicatedAllocation = false;
         dedicated->requiresDedicatedAllocation = false;
         break;
      }
      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetImageSubresourceLayout2KHR(VkDevice _device, VkImage _image,
                                     const VkImageSubresource2KHR *pSubresource,
                                     VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(borgvk_image, image, _image);
   const VkImageSubresource *sub = &pSubresource->imageSubresource;

   uint32_t bs, bw, bh;
   borgvk_aspect_block_size(image->vk.format, sub->aspectMask, &bs, &bw, &bh);
   uint32_t bd = util_format_get_blockdepth(vk_format_to_pipe_format(image->vk.format));

   uint32_t stride_blocks;
   uint64_t offset = borgvk_image_level_offset(image, sub->mipLevel, sub->arrayLayer,
                                               bs, bw, bh, bd, &stride_blocks);
   uint32_t h_blocks = DIV_ROUND_UP(MAX2(image->vk.extent.height >> sub->mipLevel, 1), bh);
   uint32_t d_blocks = DIV_ROUND_UP(MAX2(image->vk.extent.depth >> sub->mipLevel, 1), bd);

   uint64_t row = (uint64_t)stride_blocks * bs;
   uint64_t slice = row * h_blocks;

   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset     = offset,
      .size       = slice * d_blocks,
      .rowPitch   = row,
      .arrayPitch = borgvk_image_layer_size(image, bs, bw, bh, bd),
      .depthPitch = slice,
   };

   vk_foreach_struct(ext_stype, ext, pLayout->pNext) {
      vk_debug_ignored_stype(ext_stype);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_BindImageMemory2(VkDevice _device, uint32_t bindInfoCount,
                        const VkBindImageMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(borgvk_image, image, pBindInfos[i].image);
      VK_FROM_HANDLE(borgvk_device_memory, mem, pBindInfos[i].memory);
      image->mem = mem;
      image->offset = pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

/* ---- Image views & samplers (base objects; no driver state yet) ------- */

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateImageView(VkDevice _device,
                       const VkImageViewCreateInfo *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkImageView *pView)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct vk_image_view *view;

   view = vk_image_view_create(&device->vk, pCreateInfo,
                               pAllocator, sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pView = vk_image_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyImageView(VkDevice _device, VkImageView _view,
                        const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(vk_image_view, view, _view);

   if (!view)
      return;

   vk_image_view_destroy(&device->vk, pAllocator, view);
}

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

static void
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

static void borgvk_resolve_region(struct borgvk_image *src, struct borgvk_image *dst,
                                  const VkImageResolve *reg);

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd, struct borgvk_command_buffer, vk);

   cmd->in_rendering = true;
   cmd->render_area = pRenderingInfo->renderArea;
   cmd->layer_count = MAX2(pRenderingInfo->layerCount, 1);
   cmd->color_attachment_count =
      MIN2(pRenderingInfo->colorAttachmentCount, BORGVK_MAX_COLOR_ATTACHMENTS);
   cmd->depth_view = NULL;
   cmd->stencil_view = NULL;
   /* A subpass without colour must not draw into the previous one's attachment. */
   memset(cmd->color_views, 0, sizeof(cmd->color_views));
   memset(cmd->color_resolve_views, 0, sizeof(cmd->color_resolve_views));

   uint32_t layerCount = cmd->layer_count;

   for (uint32_t i = 0; i < cmd->color_attachment_count; i++) {
      const VkRenderingAttachmentInfo *att = &pRenderingInfo->pColorAttachments[i];
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      VK_FROM_HANDLE(vk_image_view, resolve_view, att->resolveImageView);
      cmd->color_views[i] = view;
      cmd->color_resolve_views[i] = resolve_view;
      if (view && att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_COLOR_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
         if (i == 0) {
            cmd->has_clear = true;
            for (int c = 0; c < 4; c++)
               cmd->clear_color[c] = att->clearValue.color.float32[c];
         }
      }
   }

   if (pRenderingInfo->pDepthAttachment &&
       pRenderingInfo->pDepthAttachment->imageView != VK_NULL_HANDLE) {
      const VkRenderingAttachmentInfo *att = pRenderingInfo->pDepthAttachment;
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      cmd->depth_view = view;
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_DEPTH_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
   }

   if (pRenderingInfo->pStencilAttachment &&
       pRenderingInfo->pStencilAttachment->imageView != VK_NULL_HANDLE) {
      const VkRenderingAttachmentInfo *att = pRenderingInfo->pStencilAttachment;
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      cmd->stencil_view = view;
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_STENCIL_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   vk_common_CmdEndRendering(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRendering2EXT(VkCommandBuffer commandBuffer,
                           const VkRenderingEndInfoEXT *pRenderingEndInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd, struct borgvk_command_buffer, vk);

   borgvk_flush_draws(cmd);

   for (uint32_t i = 0; i < cmd->color_attachment_count; i++) {
      struct vk_image_view *view = cmd->color_views[i];
      struct vk_image_view *resolve_view = cmd->color_resolve_views[i];
      if (!view || !resolve_view)
         continue;

      struct borgvk_image *src = container_of(view->image, struct borgvk_image, vk);
      struct borgvk_image *dst = container_of(resolve_view->image, struct borgvk_image, vk);
      const VkImageResolve region = {
         .srcSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = view->base_mip_level,
            .baseArrayLayer = view->base_array_layer,
            .layerCount = cmd->layer_count,
         },
         .srcOffset = { cmd->render_area.offset.x, cmd->render_area.offset.y, 0 },
         .dstSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = resolve_view->base_mip_level,
            .baseArrayLayer = resolve_view->base_array_layer,
            .layerCount = cmd->layer_count,
         },
         .dstOffset = { cmd->render_area.offset.x, cmd->render_area.offset.y, 0 },
         .extent = {
            cmd->render_area.extent.width, cmd->render_area.extent.height, 1
         },
      };
      borgvk_resolve_region(src, dst, &region);
   }

   cmd->in_rendering = false;
   cmd->color_attachment_count = 0;
   cmd->depth_view = NULL;
   cmd->stencil_view = NULL;
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

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBeginRenderPass(VkCommandBuffer commandBuffer,
                          const VkRenderPassBeginInfo *pRenderPassBegin,
                          VkSubpassContents contents)
{
   vk_common_CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBeginRenderPass2(VkCommandBuffer commandBuffer,
                           const VkRenderPassBeginInfo *pRenderPassBegin,
                           const VkSubpassBeginInfo *pSubpassBeginInfo)
{
   vk_common_CmdBeginRenderPass2(commandBuffer, pRenderPassBegin, pSubpassBeginInfo);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents)
{
   vk_common_CmdNextSubpass(commandBuffer, contents);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdNextSubpass2(VkCommandBuffer commandBuffer,
                       const VkSubpassBeginInfo *pSubpassBeginInfo,
                       const VkSubpassEndInfo *pSubpassEndInfo)
{
   vk_common_CmdNextSubpass2(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRenderPass(VkCommandBuffer commandBuffer)
{
   vk_common_CmdEndRenderPass(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRenderPass2(VkCommandBuffer commandBuffer,
                         const VkSubpassEndInfo *pSubpassEndInfo)
{
   vk_common_CmdEndRenderPass2(commandBuffer, pSubpassEndInfo);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdCopyImage(VkCommandBuffer commandBuffer,
                    VkImage srcImage, VkImageLayout srcImageLayout,
                    VkImage dstImage, VkImageLayout dstImageLayout,
                    uint32_t regionCount, const VkImageCopy *pRegions)
{
   VK_FROM_HANDLE(borgvk_image, src, srcImage);
   VK_FROM_HANDLE(borgvk_image, dst, dstImage);

   if (!src || !dst || !src->mem || !dst->mem ||
       !src->mem->map || !dst->mem->map)
      return;

   enum pipe_format src_pfmt = vk_format_to_pipe_format(src->vk.format);
   enum pipe_format dst_pfmt = vk_format_to_pipe_format(dst->vk.format);
   uint32_t src_bs = vk_format_get_blocksize(src->vk.format);
   uint32_t src_bw = vk_format_get_blockwidth(src->vk.format);
   uint32_t src_bh = vk_format_get_blockheight(src->vk.format);
   uint32_t src_bd = util_format_get_blockdepth(src_pfmt);
   uint32_t dst_bs = vk_format_get_blocksize(dst->vk.format);
   uint32_t dst_bw = vk_format_get_blockwidth(dst->vk.format);
   uint32_t dst_bh = vk_format_get_blockheight(dst->vk.format);
   uint32_t dst_bd = util_format_get_blockdepth(dst_pfmt);
   if (src_bd > 1 || dst_bd > 1)
      return;

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkImageCopy *reg = &pRegions[r];

      uint32_t src_elem_bs, src_elem_bw, src_elem_bh;
      uint32_t dst_elem_bs, dst_elem_bw, dst_elem_bh;
      borgvk_aspect_block_size(src->vk.format, reg->srcSubresource.aspectMask,
                               &src_elem_bs, &src_elem_bw, &src_elem_bh);
      borgvk_aspect_block_size(dst->vk.format, reg->dstSubresource.aspectMask,
                               &dst_elem_bs, &dst_elem_bw, &dst_elem_bh);
      uint32_t elem_bs = MIN2(src_elem_bs, dst_elem_bs);

      uint32_t w_blocks = DIV_ROUND_UP(reg->extent.width, src_elem_bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->extent.height, src_elem_bh);

      bool src_3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
      bool dst_3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;
      uint32_t slice_bd = src_3d ? src_bd : (dst_3d ? dst_bd : 1);
      uint32_t sliceCount = (src_3d || dst_3d) ?
         DIV_ROUND_UP(reg->extent.depth, slice_bd) :
         vk_image_subresource_layer_count(&src->vk, &reg->srcSubresource);

      for (uint32_t l = 0; l < sliceCount; l++) {
         uint32_t src_stride_blocks, dst_stride_blocks;
         uint64_t src_off = borgvk_image_level_offset(
            src, reg->srcSubresource.mipLevel,
            src_3d ? 0 : reg->srcSubresource.baseArrayLayer + l,
            src_bs, src_bw, src_bh, src_bd, &src_stride_blocks);
         if (src_3d)
            src_off += (uint64_t)(reg->srcOffset.z / src_bd + l) *
               borgvk_mip_level_size(src->vk.extent.width, src->vk.extent.height, 1,
                                     reg->srcSubresource.mipLevel, src_bs, src_bw, src_bh,
                                     src_bd, NULL);

         uint64_t dst_off = borgvk_image_level_offset(
            dst, reg->dstSubresource.mipLevel,
            dst_3d ? 0 : reg->dstSubresource.baseArrayLayer + l,
            dst_bs, dst_bw, dst_bh, dst_bd, &dst_stride_blocks);
         if (dst_3d)
            dst_off += (uint64_t)(reg->dstOffset.z / dst_bd + l) *
               borgvk_mip_level_size(dst->vk.extent.width, dst->vk.extent.height, 1,
                                     reg->dstSubresource.mipLevel, dst_bs, dst_bw, dst_bh,
                                     dst_bd, NULL);

         const uint8_t *s = (const uint8_t *)src->mem->map + src->offset + src_off
                          + (VkDeviceSize)(reg->srcOffset.y / src_bh) * src_stride_blocks * src_bs
                          + (VkDeviceSize)(reg->srcOffset.x / src_bw) * src_bs;
         uint8_t *d = (uint8_t *)dst->mem->map + dst->offset + dst_off
                    + (VkDeviceSize)(reg->dstOffset.y / dst_bh) * dst_stride_blocks * dst_bs
                    + (VkDeviceSize)(reg->dstOffset.x / dst_bw) * dst_bs;

         if (src_elem_bs == src_bs && dst_elem_bs == dst_bs) {
            uint64_t src_plane = borgvk_image_sample_plane_size(src, src_bs, src_bw, src_bh, src_bd);
            uint64_t dst_plane = borgvk_image_sample_plane_size(dst, dst_bs, dst_bw, dst_bh, dst_bd);
            for (uint32_t smp = 0; smp < MAX2(src->vk.samples, 1); smp++)
               for (uint32_t row = 0; row < h_blocks; row++) {
                  memcpy(d + smp * dst_plane + (size_t)row * dst_stride_blocks * dst_bs,
                         s + smp * src_plane + (size_t)row * src_stride_blocks * src_bs,
                         (size_t)w_blocks * elem_bs);
               }
         } else {
            VkImageAspectFlags aspect = reg->srcSubresource.aspectMask;
            bool stencil = (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;

            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  const uint8_t *stexel = s + (size_t)row * src_stride_blocks * src_bs
                                            + (size_t)col * src_bs;
                  uint8_t *dtexel = d + (size_t)row * dst_stride_blocks * dst_bs
                                       + (size_t)col * dst_bs;

                  if (stencil) {
                     uint8_t val = borgvk_is_z16_s8(src_pfmt) ? stexel[2] : 0;
                     if (!borgvk_is_z16_s8(src_pfmt))
                        util_format_unpack_s_8uint(src_pfmt, &val, stexel, 1);
                     if (borgvk_is_z16_s8(dst_pfmt))
                        dtexel[2] = val;
                     else
                        util_format_pack_s_8uint(dst_pfmt, dtexel, &val, 1);
                  } else {
                     float z = borgvk_is_z16_s8(src_pfmt) ?
                        borgvk_z16_s8_unpack_z(stexel) : 0.0f;
                     if (!borgvk_is_z16_s8(src_pfmt))
                        util_format_unpack_z_float(src_pfmt, &z, stexel, 1);
                     if (borgvk_is_z16_s8(dst_pfmt))
                        borgvk_z16_s8_pack_z(dtexel, z);
                     else
                        util_format_pack_z_float(dst_pfmt, dtexel, &z, 1);
                  }
               }
            }
         }
      }
   }
}

static void *
borgvk_decode_compressed_image(const struct borgvk_image *src,
                               struct borgvk_image *out,
                               struct borgvk_device_memory *mem)
{
   enum pipe_format pfmt = vk_format_to_pipe_format(src->vk.format);
   uint32_t bs = vk_format_get_blocksize(src->vk.format);
   uint32_t bw = vk_format_get_blockwidth(src->vk.format);
   uint32_t bh = vk_format_get_blockheight(src->vk.format);
   uint32_t layers = MAX2(src->vk.array_layers, 1);

   *out = *src;
   out->vk.format = VK_FORMAT_R32G32B32A32_SFLOAT;
   uint64_t total = borgvk_image_layer_size(out, 16, 1, 1, 1) * layers;
   uint8_t *buf = calloc(1, total);
   if (!buf)
      return NULL;

   for (uint32_t layer = 0; layer < layers; layer++) {
      for (uint32_t l = 0; l < src->vk.mip_levels; l++) {
         uint32_t w = MAX2(src->vk.extent.width >> l, 1);
         uint32_t h = MAX2(src->vk.extent.height >> l, 1);
         uint32_t d = MAX2(src->vk.extent.depth >> l, 1);
         uint32_t wb, ob;
         uint64_t soff = borgvk_image_level_offset(src, l, layer, bs, bw, bh, 1, &wb);
         uint64_t doff = borgvk_image_level_offset(out, l, layer, 16, 1, 1, 1, &ob);
         uint64_t sslice = (uint64_t)wb * DIV_ROUND_UP(h, bh) * bs;
         uint64_t dslice = (uint64_t)w * h * 16;
         for (uint32_t z = 0; z < d; z++)
            util_format_unpack_rgba_rect(pfmt, buf + doff + z * dslice, w * 16,
                                         (const uint8_t *)src->mem->map + src->offset + soff + z * sslice,
                                         wb * bs, w, h);
      }
   }

   memset(mem, 0, sizeof(*mem));
   mem->map = buf;
   mem->size = total;
   out->mem = mem;
   out->offset = 0;
   return buf;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBlitImage(VkCommandBuffer commandBuffer,
                    VkImage srcImage, VkImageLayout srcImageLayout,
                    VkImage dstImage, VkImageLayout dstImageLayout,
                    uint32_t regionCount, const VkImageBlit *pRegions,
                    VkFilter filter)
{
   VK_FROM_HANDLE(borgvk_image, src, srcImage);
   VK_FROM_HANDLE(borgvk_image, dst, dstImage);

   if (!src || !dst || !src->mem || !dst->mem || !src->mem->map || !dst->mem->map)
      return;

   enum pipe_format src_pfmt = vk_format_to_pipe_format(src->vk.format);
   enum pipe_format dst_pfmt = vk_format_to_pipe_format(dst->vk.format);
   uint32_t src_bs = vk_format_get_blocksize(src->vk.format);
   uint32_t dst_bs = vk_format_get_blocksize(dst->vk.format);

   if (vk_format_get_blockwidth(dst->vk.format) > 1 ||
       vk_format_get_blockheight(dst->vk.format) > 1)
      return;
   struct borgvk_image src_decoded_img;
   struct borgvk_device_memory src_decoded_mem;
   void *src_decoded = NULL;
   if (vk_format_get_blockwidth(src->vk.format) > 1 ||
       vk_format_get_blockheight(src->vk.format) > 1) {
      src_decoded = borgvk_decode_compressed_image(src, &src_decoded_img, &src_decoded_mem);
      if (!src_decoded)
         return;
      src = &src_decoded_img;
      src_pfmt = vk_format_to_pipe_format(src->vk.format);
      src_bs = vk_format_get_blocksize(src->vk.format);
   }

   bool is_uint = util_format_is_pure_uint(src_pfmt);
   bool is_sint = util_format_is_pure_sint(src_pfmt);

   bool src_3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
   bool dst_3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkImageBlit *reg = &pRegions[r];
      bool is_depth_aspect = (reg->srcSubresource.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
      bool is_stencil_aspect = (reg->srcSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;

      int32_t dst_x0 = MIN2(reg->dstOffsets[0].x, reg->dstOffsets[1].x);
      int32_t dst_x1 = MAX2(reg->dstOffsets[0].x, reg->dstOffsets[1].x);
      int32_t dst_y0 = MIN2(reg->dstOffsets[0].y, reg->dstOffsets[1].y);
      int32_t dst_y1 = MAX2(reg->dstOffsets[0].y, reg->dstOffsets[1].y);
      int32_t dst_z0 = MIN2(reg->dstOffsets[0].z, reg->dstOffsets[1].z);
      int32_t dst_z1 = MAX2(reg->dstOffsets[0].z, reg->dstOffsets[1].z);

      float dst_ex = (float)(reg->dstOffsets[1].x - reg->dstOffsets[0].x);
      float dst_ey = (float)(reg->dstOffsets[1].y - reg->dstOffsets[0].y);
      float dst_ez = (float)(reg->dstOffsets[1].z - reg->dstOffsets[0].z);
      float src_ex = (float)(reg->srcOffsets[1].x - reg->srcOffsets[0].x);
      float src_ey = (float)(reg->srcOffsets[1].y - reg->srcOffsets[0].y);
      float src_ez = (float)(reg->srcOffsets[1].z - reg->srcOffsets[0].z);

      uint32_t layerCount = (src_3d || dst_3d) ? 1 :
         vk_image_subresource_layer_count(&src->vk, &reg->srcSubresource);

      for (uint32_t l = 0; l < layerCount; l++) {
         uint32_t src_level_w = MAX2(src->vk.extent.width >> reg->srcSubresource.mipLevel, 1);
         uint32_t src_level_h = MAX2(src->vk.extent.height >> reg->srcSubresource.mipLevel, 1);
         uint32_t src_level_d = src_3d ?
            MAX2(src->vk.extent.depth >> reg->srcSubresource.mipLevel, 1) : 1;
         uint32_t dst_level_h = MAX2(dst->vk.extent.height >> reg->dstSubresource.mipLevel, 1);

         uint32_t src_stride_texels, dst_stride_texels;
         uint64_t src_layer_off = borgvk_image_level_offset(
            src, reg->srcSubresource.mipLevel,
            src_3d ? 0 : reg->srcSubresource.baseArrayLayer + l,
            src_bs, 1, 1, 1, &src_stride_texels);
         uint64_t dst_layer_off = borgvk_image_level_offset(
            dst, reg->dstSubresource.mipLevel,
            dst_3d ? 0 : reg->dstSubresource.baseArrayLayer + l,
            dst_bs, 1, 1, 1, &dst_stride_texels);

         const uint8_t *src_base = (const uint8_t *)src->mem->map + src->offset + src_layer_off;
         uint8_t *dst_base = (uint8_t *)dst->mem->map + dst->offset + dst_layer_off;

         for (int32_t dz = dst_z0; dz < dst_z1; dz++) {
            float wz = dst_ez != 0 ? ((float)dz + 0.5f - reg->dstOffsets[0].z) / dst_ez : 0.5f;
            float sz = reg->srcOffsets[0].z + wz * src_ez;

            for (int32_t dy = dst_y0; dy < dst_y1; dy++) {
               float wy = dst_ey != 0 ? ((float)dy + 0.5f - reg->dstOffsets[0].y) / dst_ey : 0.5f;
               float sy = reg->srcOffsets[0].y + wy * src_ey;

               uint8_t *drow = dst_base +
                  ((uint64_t)dz * dst_level_h + dy) * dst_stride_texels * dst_bs;

               for (int32_t dx = dst_x0; dx < dst_x1; dx++) {
                  float wx = dst_ex != 0 ? ((float)dx + 0.5f - reg->dstOffsets[0].x) / dst_ex : 0.5f;
                  float sx = reg->srcOffsets[0].x + wx * src_ex;
                  uint8_t *dtexel = drow + (uint64_t)dx * dst_bs;

                  int32_t six = CLAMP((int32_t)floorf(sx), 0, (int32_t)src_level_w - 1);
                  int32_t siy = CLAMP((int32_t)floorf(sy), 0, (int32_t)src_level_h - 1);
                  int32_t siz = CLAMP((int32_t)floorf(sz), 0, (int32_t)src_level_d - 1);

                  if (is_stencil_aspect) {
                     const uint8_t *stexel = src_base +
                        (((uint64_t)siz * src_level_h + siy) * src_stride_texels + six) * src_bs;
                     uint8_t s = borgvk_is_z16_s8(src_pfmt) ? stexel[2] : 0;
                     if (!borgvk_is_z16_s8(src_pfmt))
                        util_format_unpack_s_8uint(src_pfmt, &s, stexel, 1);
                     if (borgvk_is_z16_s8(dst_pfmt))
                        dtexel[2] = s;
                     else
                        util_format_pack_s_8uint(dst_pfmt, dtexel, &s, 1);
                  }

                  if (is_depth_aspect) {
                     if (filter == VK_FILTER_NEAREST) {
                        const uint8_t *stexel = src_base +
                           (((uint64_t)siz * src_level_h + siy) * src_stride_texels + six) * src_bs;
                        float z = borgvk_is_z16_s8(src_pfmt) ? borgvk_z16_s8_unpack_z(stexel) : 0.0f;
                        if (!borgvk_is_z16_s8(src_pfmt))
                           util_format_unpack_z_float(src_pfmt, &z, stexel, 1);
                        if (borgvk_is_z16_s8(dst_pfmt))
                           borgvk_z16_s8_pack_z(dtexel, z);
                        else
                           util_format_pack_z_float(dst_pfmt, dtexel, &z, 1);
                     } else {
                        float fx = sx - 0.5f, fy = sy - 0.5f, fz = sz - 0.5f;
                        int32_t x0 = (int32_t)floorf(fx), y0 = (int32_t)floorf(fy), z0 = (int32_t)floorf(fz);
                        float tx = fx - (float)x0, ty = fy - (float)y0, tz = fz - (float)z0;
                        int32_t x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
                        x0 = CLAMP(x0, 0, (int32_t)src_level_w - 1);
                        x1 = CLAMP(x1, 0, (int32_t)src_level_w - 1);
                        y0 = CLAMP(y0, 0, (int32_t)src_level_h - 1);
                        y1 = CLAMP(y1, 0, (int32_t)src_level_h - 1);
                        z0 = CLAMP(z0, 0, (int32_t)src_level_d - 1);
                        z1 = CLAMP(z1, 0, (int32_t)src_level_d - 1);

                        float acc = 0;
                        int32_t xs[2] = {x0, x1}, ys[2] = {y0, y1}, zs[2] = {z0, z1};
                        float xw[2] = {1 - tx, tx}, yw[2] = {1 - ty, ty}, zw[2] = {1 - tz, tz};
                        for (int iz = 0; iz < 2; iz++) {
                           for (int iy = 0; iy < 2; iy++) {
                              for (int ix = 0; ix < 2; ix++) {
                                 float weight = xw[ix] * yw[iy] * zw[iz];
                                 if (weight == 0.0f)
                                    continue;
                                 const uint8_t *stexel = src_base +
                                    (((uint64_t)zs[iz] * src_level_h + ys[iy]) * src_stride_texels
                                     + xs[ix]) * src_bs;
                                 float z = borgvk_is_z16_s8(src_pfmt) ?
                                    borgvk_z16_s8_unpack_z(stexel) : 0.0f;
                                 if (!borgvk_is_z16_s8(src_pfmt))
                                    util_format_unpack_z_float(src_pfmt, &z, stexel, 1);
                                 acc += z * weight;
                              }
                           }
                        }
                        if (borgvk_is_z16_s8(dst_pfmt))
                           borgvk_z16_s8_pack_z(dtexel, acc);
                        else
                           util_format_pack_z_float(dst_pfmt, dtexel, &acc, 1);
                     }
                  }

                  if (!is_depth_aspect && !is_stencil_aspect) {
                     if (filter == VK_FILTER_NEAREST) {
                        const uint8_t *stexel = src_base +
                           (((uint64_t)siz * src_level_h + siy) * src_stride_texels + six) * src_bs;
                        if (is_uint) {
                           uint32_t v[4];
                           util_format_unpack_rgba(src_pfmt, v, stexel, 1);
                           util_format_pack_rgba(dst_pfmt, dtexel, v, 1);
                        } else if (is_sint) {
                           int32_t v[4];
                           util_format_unpack_rgba(src_pfmt, v, stexel, 1);
                           util_format_pack_rgba(dst_pfmt, dtexel, v, 1);
                        } else {
                           float v[4];
                           borgvk_unpack_rgba_float(src->vk.format, src_pfmt, stexel, v);
                           borgvk_pack_rgba_float(dst->vk.format, dst_pfmt, dtexel, v);
                        }
                     } else {
                        float fx = sx - 0.5f, fy = sy - 0.5f, fz = sz - 0.5f;
                        int32_t x0 = (int32_t)floorf(fx), y0 = (int32_t)floorf(fy), z0 = (int32_t)floorf(fz);
                        float tx = fx - (float)x0, ty = fy - (float)y0, tz = fz - (float)z0;
                        int32_t x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
                        x0 = CLAMP(x0, 0, (int32_t)src_level_w - 1);
                        x1 = CLAMP(x1, 0, (int32_t)src_level_w - 1);
                        y0 = CLAMP(y0, 0, (int32_t)src_level_h - 1);
                        y1 = CLAMP(y1, 0, (int32_t)src_level_h - 1);
                        z0 = CLAMP(z0, 0, (int32_t)src_level_d - 1);
                        z1 = CLAMP(z1, 0, (int32_t)src_level_d - 1);

                        float acc[4] = {0, 0, 0, 0};
                        int32_t xs[2] = {x0, x1}, ys[2] = {y0, y1}, zs[2] = {z0, z1};
                        float xw[2] = {1 - tx, tx}, yw[2] = {1 - ty, ty}, zw[2] = {1 - tz, tz};
                        for (int iz = 0; iz < 2; iz++) {
                           for (int iy = 0; iy < 2; iy++) {
                              for (int ix = 0; ix < 2; ix++) {
                                 float weight = xw[ix] * yw[iy] * zw[iz];
                                 if (weight == 0.0f)
                                    continue;
                                 const uint8_t *stexel = src_base +
                                    (((uint64_t)zs[iz] * src_level_h + ys[iy]) * src_stride_texels
                                     + xs[ix]) * src_bs;
                                 float v[4];
                                 borgvk_unpack_rgba_float(src->vk.format, src_pfmt, stexel, v);
                                 acc[0] += v[0] * weight;
                                 acc[1] += v[1] * weight;
                                 acc[2] += v[2] * weight;
                                 acc[3] += v[3] * weight;
                              }
                           }
                        }
                        borgvk_pack_rgba_float(dst->vk.format, dst_pfmt, dtexel, acc);
                     }
                  }
               }
            }
         }
      }
   }
   free(src_decoded);
}

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
borgvk_CmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *pCopyImageInfo)
{
   VkImageCopy regions[MAX2(pCopyImageInfo->regionCount, 1)];
   for (uint32_t i = 0; i < pCopyImageInfo->regionCount; i++) {
      const VkImageCopy2 *r = &pCopyImageInfo->pRegions[i];
      regions[i] = (VkImageCopy){
         .srcSubresource = r->srcSubresource, .srcOffset = r->srcOffset,
         .dstSubresource = r->dstSubresource, .dstOffset = r->dstOffset,
         .extent = r->extent,
      };
   }
   borgvk_CmdCopyImage(commandBuffer, pCopyImageInfo->srcImage, pCopyImageInfo->srcImageLayout,
                       pCopyImageInfo->dstImage, pCopyImageInfo->dstImageLayout,
                       pCopyImageInfo->regionCount, regions);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBlitImage2(VkCommandBuffer commandBuffer, const VkBlitImageInfo2 *pBlitImageInfo)
{
   VkImageBlit regions[MAX2(pBlitImageInfo->regionCount, 1)];
   for (uint32_t i = 0; i < pBlitImageInfo->regionCount; i++) {
      const VkImageBlit2 *r = &pBlitImageInfo->pRegions[i];
      regions[i] = (VkImageBlit){
         .srcSubresource = r->srcSubresource,
         .srcOffsets = {r->srcOffsets[0], r->srcOffsets[1]},
         .dstSubresource = r->dstSubresource,
         .dstOffsets = {r->dstOffsets[0], r->dstOffsets[1]},
      };
   }
   borgvk_CmdBlitImage(commandBuffer, pBlitImageInfo->srcImage, pBlitImageInfo->srcImageLayout,
                       pBlitImageInfo->dstImage, pBlitImageInfo->dstImageLayout,
                       pBlitImageInfo->regionCount, regions, pBlitImageInfo->filter);
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

static void
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

static void
borgvk_pack_sampler(const VkSamplerCreateInfo *info, uint32_t desc[4])
{
   union { float f; uint32_t u; } bias = { info->mipLodBias },
                                  lo = { info->minLod }, hi = { info->maxLod };
   desc[0] = (info->magFilter == VK_FILTER_LINEAR ? 1u << 0 : 0) |
             (info->minFilter == VK_FILTER_LINEAR ? 1u << 1 : 0) |
             (info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? 1u << 2 : 0) |
             ((uint32_t)info->addressModeU & 0x7) << 3 |
             ((uint32_t)info->addressModeV & 0x7) << 6 |
             ((uint32_t)info->addressModeW & 0x7) << 9 |
             ((uint32_t)info->borderColor & 0x7) << 12 |
             (info->compareEnable ? 1u << 15 : 0) |
             ((uint32_t)info->compareOp & 0x7) << 16 |
             (info->unnormalizedCoordinates ? 1u << 19 : 0);
   desc[1] = bias.u;
   desc[2] = lo.u;
   desc[3] = hi.u;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkSampler *pSampler)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_sampler *sampler;

   sampler = vk_sampler_create(&device->vk, pCreateInfo,
                               pAllocator, sizeof(*sampler));
   if (!sampler)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   borgvk_pack_sampler(pCreateInfo, sampler->desc);

   *pSampler = borgvk_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroySampler(VkDevice _device, VkSampler _sampler,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_sampler, sampler, _sampler);

   if (!sampler)
      return;

   vk_sampler_destroy(&device->vk, pAllocator, &sampler->vk);
}
