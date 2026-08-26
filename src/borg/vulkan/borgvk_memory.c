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
   pMemoryRequirements->memoryRequirements.size =
      align64(pInfo->pCreateInfo->size, 256);
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

/* ---- Images ----------------------------------------------------------- */

static VkDeviceSize
borgvk_image_size(const VkImageCreateInfo *info)
{
   /* Rough linear size: enough to back the image in host RAM. We never sample
    * or render to it on the host. */
   uint32_t bs = vk_format_get_blocksize(info->format);
   uint64_t size = (uint64_t)info->extent.width * info->extent.height *
                   info->extent.depth * MAX2(info->arrayLayers, 1) * MAX2(bs, 1);
   /* crude mip allowance */
   if (info->mipLevels > 1)
      size += size / 2;
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

/* Image copy for readback: both images are linear host-RAM; just memcpy the
 * region. The CTS path (Image::copyToLinearImage) copies mip 0, layer 0 of the
 * colour attachment into a linear staging image of the same size/format. */
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

   uint32_t bs = vk_format_get_blocksize(src->vk.format);

   /* Strides and extents must be counted in compressed blocks, not texels:
    * vk_format_get_blocksize() returns bytes per BLOCK (e.g. 16 B covering a
    * 4x4 texel block for ETC2/EAC), not bytes per texel. Multiplying texel
    * width/height directly by bs overflowed the destination allocation for
    * any block-compressed format -- surfaced as a heap "double free or
    * corruption" abort in a later borgvk_FreeMemory, not here. VkImageCopy
    * offsets/extents are always block-aligned per the spec, so the divisions
    * below are exact. For uncompressed formats bw = bh = 1 and this is
    * unchanged from the original per-texel math. */
   uint32_t bw = vk_format_get_blockwidth(src->vk.format);
   uint32_t bh = vk_format_get_blockheight(src->vk.format);
   uint32_t src_stride_blocks = src->vk.extent.width / bw;
   uint32_t dst_stride_blocks = dst->vk.extent.width / bw;

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkImageCopy *reg = &pRegions[r];
      uint32_t w_blocks = DIV_ROUND_UP(reg->extent.width, bw);
      uint32_t h_blocks = DIV_ROUND_UP(reg->extent.height, bh);

      const uint8_t *s = (const uint8_t *)src->mem->map + src->offset
                       + (VkDeviceSize)(reg->srcOffset.y / bh) * src_stride_blocks * bs
                       + (VkDeviceSize)(reg->srcOffset.x / bw) * bs;
      uint8_t *d = (uint8_t *)dst->mem->map + dst->offset
                 + (VkDeviceSize)(reg->dstOffset.y / bh) * dst_stride_blocks * bs
                 + (VkDeviceSize)(reg->dstOffset.x / bw) * bs;

      for (uint32_t row = 0; row < h_blocks; row++) {
         memcpy(d + (size_t)row * w_blocks * bs,
                s + (size_t)row * src_stride_blocks * bs,
                (size_t)w_blocks * bs);
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
