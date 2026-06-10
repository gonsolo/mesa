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

#include <stdlib.h>

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

   mem->map = malloc(MAX2(pAllocateInfo->allocationSize, 1));
   if (!mem->map) {
      vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
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

   free(mem->map);
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
