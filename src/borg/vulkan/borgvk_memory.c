/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Memory, buffers, images, image views and samplers for borgvk. Everything is
 * backed by plain host RAM (the real GPU is the FPGA over serial); these
 * entrypoints exist so an app can allocate/bind/map resources during init. The
 * submit path (Phase 3) reads the bound uniform buffer's mapped bytes.
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_sampler.h"

#include "util/u_math.h"

#include "drm-uapi/borg_drm.h"

#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <xf86drm.h>

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

/* ---- Buffer views -------------------------------------------------------
 * Was entirely missing: vkCreateBufferView/vkDestroyBufferView are
 * mandatory core Vulkan 1.0 entry points (not extension-gated like sparse
 * binding), so their dispatch-table slot was null and calling through it
 * crashed with a SEGV at instruction address 0 -- confirmed via
 * coredumpctl/gdb backtrace through vk::createBufferView() ->
 * BufferViewTestInstance -- rather than any real driver logic being wrong.
 */

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
   /* align64(size, 256) overflows (wraps toward 0) for a size within 255 of
    * UINT64_MAX. borgvk_CreateBuffer never rejects an oversized VkBuffer (it
    * defers all real allocation to vkAllocateMemory/vkBindBufferMemory), so
    * reporting a wrapped, too-small size here made VK-GL-CTS's
    * api.buffer.basic.size_max_uint64 fail: it requires
    * memoryRequirements.size >= the buffer's own requested size whenever
    * creation succeeds. Saturate instead of wrapping. */
   uint64_t size = pInfo->pCreateInfo->size;
   pMemoryRequirements->memoryRequirements.size =
      size > UINT64_MAX - 255 ? UINT64_MAX : align64(size, 256);
   pMemoryRequirements->memoryRequirements.alignment = 256;
   pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x1;

   vk_foreach_struct(ext, pMemoryRequirements->pNext) {
      switch (ext->sType) {
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

/* Was entirely missing (discarded into the generic vk_cmd_queue). Needed by
 * ImageClearingTestInstance::preClearImage, which cmdFillBuffer(0)s a staging
 * buffer before uploading it into an image via CmdCopyBufferToImage -- a
 * no-op fill meant that upload copied whatever garbage the staging buffer's
 * host allocation already contained instead of zero. */
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

/* ---- Images ----------------------------------------------------------- */

/* One consistent linear layout, shared by every place that reads or writes
 * image bytes (borgvk_image_size, CmdClearColorImage, CmdCopyImage,
 * CmdCopyImageToBuffer): layer-major, mip-minor -- layer 0's full mip chain
 * (level 0, level 1, ...) is contiguous, then layer 1's, etc. This is
 * internal to borgvk only (VK_IMAGE_TILING_LINEAR's real layout is
 * implementation-defined, and nothing outside this driver ever reads this
 * memory), but every reader/writer MUST agree, or an offset computed one way
 * and validated against a total computed another way silently walks off the
 * end of the allocation -- the same class of bug as the earlier CmdCopyImage
 * block-size overflow. Returns the level's size in bytes and, via
 * *width_blocks_out, its row stride (in format blocks) for callers doing
 * row-by-row copies. */
static uint64_t
borgvk_mip_level_size(uint32_t width, uint32_t height, uint32_t depth,
                      uint32_t level, uint32_t bs, uint32_t bw, uint32_t bh,
                      uint32_t *width_blocks_out)
{
   uint32_t w = MAX2(width >> level, 1);
   uint32_t h = MAX2(height >> level, 1);
   uint32_t d = MAX2(depth >> level, 1);
   uint32_t w_blocks = DIV_ROUND_UP(w, bw);
   uint32_t h_blocks = DIV_ROUND_UP(h, bh);

   if (width_blocks_out)
      *width_blocks_out = w_blocks;
   return (uint64_t)w_blocks * h_blocks * d * bs;
}

static uint64_t
borgvk_image_layer_size(const struct borgvk_image *image, uint32_t bs,
                        uint32_t bw, uint32_t bh)
{
   uint64_t layer_size = 0;
   for (uint32_t l = 0; l < image->vk.mip_levels; l++)
      layer_size += borgvk_mip_level_size(image->vk.extent.width,
                                          image->vk.extent.height,
                                          image->vk.extent.depth,
                                          l, bs, bw, bh, NULL);
   return layer_size;
}

/* Byte offset of (level, layer)'s data within the image's backing memory,
 * plus that level's row stride (in format blocks). */
static uint64_t
borgvk_image_level_offset(const struct borgvk_image *image, uint32_t level,
                          uint32_t layer, uint32_t bs, uint32_t bw, uint32_t bh,
                          uint32_t *stride_blocks_out)
{
   uint64_t layer_size = borgvk_image_layer_size(image, bs, bw, bh);
   uint64_t level_offset = 0;

   for (uint32_t l = 0; l < level; l++)
      level_offset += borgvk_mip_level_size(image->vk.extent.width,
                                            image->vk.extent.height,
                                            image->vk.extent.depth,
                                            l, bs, bw, bh, NULL);
   borgvk_mip_level_size(image->vk.extent.width, image->vk.extent.height,
                         image->vk.extent.depth, level, bs, bw, bh,
                         stride_blocks_out);

   return (uint64_t)layer * layer_size + level_offset;
}

/* For a combined depth-stencil image, a copy naming only one aspect
 * (VkBufferImageCopy::imageSubresource.aspectMask) writes/reads a buffer
 * packed with just that aspect's own element size (e.g. 1 tightly-packed
 * byte per texel for VK_IMAGE_ASPECT_STENCIL_BIT on D24_UNORM_S8_UINT), not
 * the combined format's element size (4 bytes there). Blindly using
 * vk_format_get_blocksize(image->vk.format) for that side overflowed the
 * buffer by up to 8x -- a real "double free or corruption" heap-overflow
 * abort confirmed via coredumpctl/gdb in a later borgvk_FreeMemory (found
 * running dEQP-VK.api.command_buffers.record_many_draws_primary_1, which
 * exercises a depth-stencil attachment readback). For VK_IMAGE_ASPECT_COLOR_BIT
 * this returns the image's own format unchanged, so callers that always pass
 * COLOR_BIT see no behavior change. A region can also legally name BOTH
 * VK_IMAGE_ASPECT_DEPTH_BIT and _STENCIL_BIT together (image-to-image copies
 * between two images of the same combined format): that means "copy the
 * whole combined texel" and must NOT be forwarded to
 * vk_format_get_aspect_format(), which asserts on a single-bit aspect mask
 * (confirmed via a clean assertion failure, not a crash, running
 * dEQP-VK.api.copy_and_blit...depth_stencil_aspects tests) -- so it's handled
 * as the identity case here, same as passing the format through unchanged. */
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
   uint64_t layer_size = 0;

   for (uint32_t l = 0; l < MAX2(info->mipLevels, 1); l++)
      layer_size += borgvk_mip_level_size(info->extent.width, info->extent.height,
                                          info->extent.depth, l, bs, bw, bh, NULL);

   uint64_t size = layer_size * MAX2(info->arrayLayers, 1);
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

   vk_foreach_struct(ext, pMemoryRequirements->pNext) {
      switch (ext->sType) {
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

/* Linear subresource layout. cube.c queries this to upload its staging texture
 * row by row; report a tightly-packed linear layout matching how an app maps and
 * writes the malloc-backed image memory (mip 0 / layer 0 at offset 0). */
VKAPI_ATTR void VKAPI_CALL
borgvk_GetImageSubresourceLayout2KHR(VkDevice _device, VkImage _image,
                                     const VkImageSubresource2KHR *pSubresource,
                                     VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(borgvk_image, image, _image);

   uint32_t bs = vk_format_get_blocksize(image->vk.format);
   uint64_t row = (uint64_t)image->vk.extent.width * MAX2(bs, 1);
   uint64_t slice = row * image->vk.extent.height;

   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset     = 0,
      .size       = slice * image->vk.extent.depth,
      .rowPitch   = row,
      .arrayPitch = slice,
      .depthPitch = slice,
   };

   vk_foreach_struct(ext, pLayout->pNext) {
      vk_debug_ignored_stype(ext->sType);
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

/* Every other vkCmd* is recorded into the generic vk_cmd_queue and discarded
 * on reset (see the file comment in borgvk_cmd_buffer.c) -- fine for
 * rendering, since the cube's real state is read from bound
 * buffers/descriptors at submit time, not replayed commands. But
 * CmdClearColorImage's whole observable effect IS the write it makes to image
 * memory, and CTS's image_clearing tests read that memory back afterward. Left
 * unimplemented, the clear byte pattern was whatever the host malloc arena
 * happened to already contain -- which is why the *_clamp_input tests were
 * flipping Pass/Fail between otherwise-identical runs (confirmed by diffing
 * two full CTS runs where the only driver change was unrelated: whatever
 * garbage ended up in a given allocation depends on the process's prior
 * allocation history). Fixed the same way as CmdCopyImage: execute for real
 * at record time instead of queuing, addressed via the shared
 * borgvk_image_level_offset layout so every mip level/array layer in range is
 * covered, not just level 0. util_format_pack_rgba() clamps to the
 * destination format's representable range, which is exactly the
 * "_clamp_input" semantics CTS is checking. */
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

   uint8_t pixel[16]; /* largest uncompressed colour format block is 16 B */
   if (util_format_is_pure_uint(pfmt))
      util_format_pack_rgba(pfmt, pixel, pColor->uint32, 1);
   else if (util_format_is_pure_sint(pfmt))
      util_format_pack_rgba(pfmt, pixel, pColor->int32, 1);
   else
      util_format_pack_rgba(pfmt, pixel, pColor->float32, 1);

   for (uint32_t r = 0; r < rangeCount; r++) {
      const VkImageSubresourceRange *range = &pRanges[r];
      uint32_t levelCount = vk_image_subresource_level_count(&image->vk, range);
      uint32_t layerCount = vk_image_subresource_layer_count(&image->vk, range);

      for (uint32_t lv = 0; lv < levelCount; lv++) {
         uint32_t level = range->baseMipLevel + lv;
         uint32_t stride_blocks;
         uint32_t h_blocks = DIV_ROUND_UP(MAX2(image->vk.extent.height >> level, 1), bh);
         uint32_t d = MAX2(image->vk.extent.depth >> level, 1);

         for (uint32_t l = 0; l < layerCount; l++) {
            uint32_t layer = range->baseArrayLayer + l;
            uint64_t off = borgvk_image_level_offset(image, level, layer, bs, bw, bh,
                                                     &stride_blocks);
            uint8_t *dst = (uint8_t *)image->mem->map + image->offset + off;

            for (uint64_t t = 0; t < (uint64_t)stride_blocks * h_blocks * d; t++)
               memcpy(dst + t * bs, pixel, bs);
         }
      }
   }
}

/* Image copy for readback: both images are linear host-RAM; just memcpy the
 * region, addressed via the shared borgvk_image_level_offset layout so every
 * mip level/array layer a region names is handled, not just level 0/layer 0. */
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

   /* Strides and extents must be counted in compressed blocks, not texels:
    * vk_format_get_blocksize() returns bytes per BLOCK (e.g. 16 B covering a
    * 4x4 texel block for ETC2/EAC), not bytes per texel. Multiplying texel
    * width/height directly by bs overflowed the destination allocation for
    * any block-compressed format -- surfaced as a heap "double free or
    * corruption" abort in a later borgvk_FreeMemory, not here. VkImageCopy
    * offsets/extents are always block-aligned per the spec, so the divisions
    * below are exact. For uncompressed formats bw = bh = 1 and this is
    * unchanged from the original per-texel math. */
   uint32_t src_bs = vk_format_get_blocksize(src->vk.format);
   uint32_t src_bw = vk_format_get_blockwidth(src->vk.format);
   uint32_t src_bh = vk_format_get_blockheight(src->vk.format);
   uint32_t dst_bs = vk_format_get_blocksize(dst->vk.format);
   uint32_t dst_bw = vk_format_get_blockwidth(dst->vk.format);
   uint32_t dst_bh = vk_format_get_blockheight(dst->vk.format);

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkImageCopy *reg = &pRegions[r];

      /* Element size actually moved per texel: for a combined depth-stencil
       * image with an aspect-selective region this differs from the
       * image's own (combined) block size -- see borgvk_aspect_block_size.
       * Addressing within each image still uses its own combined format
       * (that's the real backing layout); src_elem_bs/dst_elem_bs is only
       * how many bytes of each texel are actually copied. */
      uint32_t src_elem_bs, src_elem_bw, src_elem_bh;
      uint32_t dst_elem_bs, dst_elem_bw, dst_elem_bh;
      borgvk_aspect_block_size(src->vk.format, reg->srcSubresource.aspectMask,
                               &src_elem_bs, &src_elem_bw, &src_elem_bh);
      borgvk_aspect_block_size(dst->vk.format, reg->dstSubresource.aspectMask,
                               &dst_elem_bs, &dst_elem_bw, &dst_elem_bh);
      uint32_t elem_bs = MIN2(src_elem_bs, dst_elem_bs);

      uint32_t w_blocks = DIV_ROUND_UP(reg->extent.width, src_elem_bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->extent.height, src_elem_bh);

      /* A copy between a 2D (array) image and a 3D image is a spec-defined
       * special case (VUID-vkCmdCopyImage-srcImage-07743 and friends): the
       * 2D side's array layers correspond to Z slices within the 3D image's
       * SINGLE array layer, not to separate array layers of their own (a 3D
       * image always has arrayLayers=1). Treating each slice as a real array
       * layer on the 3D side -- which borgvk_image_level_offset's "layer"
       * parameter always multiplies by the image's full per-layer size --
       * walked off the end of the 3D image's allocation for slice > 0. Real
       * bug, not a corner case: confirmed via coredumpctl/gdb, a heap
       * corruption surfacing later as a SEGV inside CTS's own posix_memalign
       * while logging the (unrelated, already-corrupted) test result, found
       * running dEQP-VK.api.copy_and_blit.core.image_to_image.3d_images.*. */
      bool src_3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
      bool dst_3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;
      uint32_t sliceCount = (src_3d || dst_3d) ?
         reg->extent.depth : vk_image_subresource_layer_count(&src->vk, &reg->srcSubresource);

      for (uint32_t l = 0; l < sliceCount; l++) {
         uint32_t src_stride_blocks, dst_stride_blocks;
         uint64_t src_off = borgvk_image_level_offset(
            src, reg->srcSubresource.mipLevel,
            src_3d ? 0 : reg->srcSubresource.baseArrayLayer + l,
            src_bs, src_bw, src_bh, &src_stride_blocks);
         if (src_3d)
            src_off += (uint64_t)(reg->srcOffset.z + l) *
               borgvk_mip_level_size(src->vk.extent.width, src->vk.extent.height, 1,
                                     reg->srcSubresource.mipLevel, src_bs, src_bw, src_bh, NULL);

         uint64_t dst_off = borgvk_image_level_offset(
            dst, reg->dstSubresource.mipLevel,
            dst_3d ? 0 : reg->dstSubresource.baseArrayLayer + l,
            dst_bs, dst_bw, dst_bh, &dst_stride_blocks);
         if (dst_3d)
            dst_off += (uint64_t)(reg->dstOffset.z + l) *
               borgvk_mip_level_size(dst->vk.extent.width, dst->vk.extent.height, 1,
                                     reg->dstSubresource.mipLevel, dst_bs, dst_bw, dst_bh, NULL);

         const uint8_t *s = (const uint8_t *)src->mem->map + src->offset + src_off
                          + (VkDeviceSize)(reg->srcOffset.y / src_bh) * src_stride_blocks * src_bs
                          + (VkDeviceSize)(reg->srcOffset.x / src_bw) * src_bs;
         uint8_t *d = (uint8_t *)dst->mem->map + dst->offset + dst_off
                    + (VkDeviceSize)(reg->dstOffset.y / dst_bh) * dst_stride_blocks * dst_bs
                    + (VkDeviceSize)(reg->dstOffset.x / dst_bw) * dst_bs;

         if (src_elem_bs == src_bs && dst_elem_bs == dst_bs) {
            /* Common case (whole-texel copy, e.g. plain colour formats):
             * one memcpy per row. */
            for (uint32_t row = 0; row < h_blocks; row++) {
               memcpy(d + (size_t)row * w_blocks * dst_bs,
                      s + (size_t)row * src_stride_blocks * src_bs,
                      (size_t)w_blocks * elem_bs);
            }
         } else {
            /* Aspect-selective copy on a combined format: each destination
             * texel is smaller than its source texel's storage, so texels
             * must be copied individually rather than row-at-once. */
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  memcpy(d + (size_t)row * w_blocks * dst_elem_bs + (size_t)col * dst_elem_bs,
                         s + (size_t)row * src_stride_blocks * src_bs + (size_t)col * src_bs,
                         elem_bs);
               }
            }
         }
      }
   }
}

/* Like CmdClearColorImage, this was entirely missing -- discarded into the
 * generic vk_cmd_queue like every other unimplemented vkCmd*. That's the
 * actual reason CmdClearColorImage's fix alone didn't make image_clearing
 * tests pass: CTS's own readback path (ImageClearingTestInstance::readImage)
 * doesn't read the image directly, it always goes through
 * vkCmdCopyImageToBuffer into a host-visible staging buffer first. With this
 * missing, that staging buffer was never actually written, so verification
 * saw whatever the buffer's fresh host allocation already contained,
 * regardless of what CmdClearColorImage or anything else wrote to the image.
 * Addressed via the shared borgvk_image_level_offset layout so every mip
 * level CTS reads back (not just level 0) sees real data. */
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

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkBufferImageCopy *reg = &pRegions[r];

      /* The buffer side is packed with the SELECTED ASPECT's element size,
       * which for an aspect-selective region on a combined depth-stencil
       * image (e.g. VK_IMAGE_ASPECT_STENCIL_BIT on D24_UNORM_S8_UINT) is
       * smaller than the image's own combined element size -- see
       * borgvk_aspect_block_size. Using the combined size for the buffer
       * side overflowed it by up to 8x, a real heap-overflow "double free
       * or corruption" abort confirmed via coredumpctl/gdb. */
      uint32_t bs, bw, bh;
      borgvk_aspect_block_size(src->vk.format, reg->imageSubresource.aspectMask,
                               &bs, &bw, &bh);

      uint32_t w_blocks = DIV_ROUND_UP(reg->imageExtent.width, bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->imageExtent.height, bh);
      /* 0 means tightly packed (VkBufferImageCopy spec). */
      uint32_t row_len_blocks = reg->bufferRowLength ?
         DIV_ROUND_UP(reg->bufferRowLength, bw) : w_blocks;
      uint32_t img_h_blocks = reg->bufferImageHeight ?
         DIV_ROUND_UP(reg->bufferImageHeight, bh) : h_blocks;
      uint64_t layer_pitch = (uint64_t)row_len_blocks * img_h_blocks * bs;
      uint32_t layerCount = MAX2(reg->imageSubresource.layerCount, 1);

      for (uint32_t l = 0; l < layerCount; l++) {
         uint32_t src_stride_blocks;
         uint64_t level_off = borgvk_image_level_offset(
            src, reg->imageSubresource.mipLevel,
            reg->imageSubresource.baseArrayLayer + l, img_bs, img_bw, img_bh,
            &src_stride_blocks);

         const uint8_t *s = (const uint8_t *)src->mem->map + src->offset + level_off
                          + (VkDeviceSize)(reg->imageOffset.y / img_bh) * src_stride_blocks * img_bs
                          + (VkDeviceSize)(reg->imageOffset.x / img_bw) * img_bs;
         uint8_t *d = (uint8_t *)dst->mem->map + dst->offset + reg->bufferOffset
                    + (VkDeviceSize)l * layer_pitch;

         if (bs == img_bs) {
            for (uint32_t row = 0; row < h_blocks; row++) {
               memcpy(d + (size_t)row * row_len_blocks * bs,
                      s + (size_t)row * src_stride_blocks * img_bs,
                      (size_t)w_blocks * bs);
            }
         } else {
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  memcpy(d + (size_t)row * row_len_blocks * bs + (size_t)col * bs,
                         s + (size_t)row * src_stride_blocks * img_bs + (size_t)col * img_bs,
                         bs);
               }
            }
         }
      }
   }
}

/* Upload counterpart of CmdCopyImageToBuffer, same reasoning: was entirely
 * missing (discarded into the generic vk_cmd_queue), which broke every test
 * that pre-fills an image from a host buffer before checking a partial
 * operation against it. Concretely: ImageClearingTestInstance::preClearImage
 * cmdFillBuffer(0)s a staging buffer then vkCmdCopyBufferToImage()s it across
 * every mip level and array layer *before* the real (possibly partial-range)
 * clear -- so pixels/layers the real clear doesn't touch are supposed to read
 * back as zero. Without this, they read back whatever the image's fresh host
 * allocation already contained, which is why clear_color_image's
 * multiple_layers/remaining_array_layers* groups kept failing even after
 * CmdClearColorImage and CmdCopyImageToBuffer were both fixed. */
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

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkBufferImageCopy *reg = &pRegions[r];

      /* See the matching comment in CmdCopyImageToBuffer: the buffer side is
       * packed with the selected aspect's element size, not the image's
       * combined element size. */
      uint32_t bs, bw, bh;
      borgvk_aspect_block_size(dst->vk.format, reg->imageSubresource.aspectMask,
                               &bs, &bw, &bh);

      uint32_t w_blocks = DIV_ROUND_UP(reg->imageExtent.width, bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->imageExtent.height, bh);
      uint32_t row_len_blocks = reg->bufferRowLength ?
         DIV_ROUND_UP(reg->bufferRowLength, bw) : w_blocks;
      uint32_t img_h_blocks = reg->bufferImageHeight ?
         DIV_ROUND_UP(reg->bufferImageHeight, bh) : h_blocks;
      uint64_t layer_pitch = (uint64_t)row_len_blocks * img_h_blocks * bs;
      uint32_t layerCount = MAX2(reg->imageSubresource.layerCount, 1);

      for (uint32_t l = 0; l < layerCount; l++) {
         uint32_t dst_stride_blocks;
         uint64_t level_off = borgvk_image_level_offset(
            dst, reg->imageSubresource.mipLevel,
            reg->imageSubresource.baseArrayLayer + l, img_bs, img_bw, img_bh,
            &dst_stride_blocks);

         const uint8_t *s = (const uint8_t *)src->mem->map + src->offset + reg->bufferOffset
                          + (VkDeviceSize)l * layer_pitch;
         uint8_t *d = (uint8_t *)dst->mem->map + dst->offset + level_off
                    + (VkDeviceSize)(reg->imageOffset.y / img_bh) * dst_stride_blocks * img_bs
                    + (VkDeviceSize)(reg->imageOffset.x / img_bw) * img_bs;

         if (bs == img_bs) {
            for (uint32_t row = 0; row < h_blocks; row++) {
               memcpy(d + (size_t)row * dst_stride_blocks * img_bs,
                      s + (size_t)row * row_len_blocks * bs,
                      (size_t)w_blocks * bs);
            }
         } else {
            for (uint32_t row = 0; row < h_blocks; row++) {
               for (uint32_t col = 0; col < w_blocks; col++) {
                  memcpy(d + (size_t)row * dst_stride_blocks * img_bs + (size_t)col * img_bs,
                         s + (size_t)row * row_len_blocks * bs + (size_t)col * bs,
                         bs);
               }
            }
         }
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkSampler *pSampler)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct vk_sampler *sampler;

   sampler = vk_sampler_create(&device->vk, pCreateInfo,
                               pAllocator, sizeof(*sampler));
   if (!sampler)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pSampler = vk_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroySampler(VkDevice _device, VkSampler _sampler,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(vk_sampler, sampler, _sampler);

   if (!sampler)
      return;

   vk_sampler_destroy(&device->vk, pAllocator, sampler);
}
