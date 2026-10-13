/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_format.h"

#include "borgvk_image.h"

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
