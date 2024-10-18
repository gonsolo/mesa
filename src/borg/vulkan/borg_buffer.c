/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_buffer.h"
#include "borg_device.h"
#include "borg_device_memory.h"
#include "borg_entrypoints.h"
#include "borg_physical_device.h"
#include "borg_private.h"

#include "vk_log.h"
#include "vk_util.h"

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateBuffer(VkDevice device,
                 const VkBufferCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(borg_device, dev, device);
   struct borg_buffer *buffer;

   //puts("borg_CreateBuffer");

   if (pCreateInfo->size > BORG_MAX_BUFFER_SIZE) {
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   buffer = vk_buffer_create(&dev->vk, pCreateInfo, pAllocator,
                             sizeof(*buffer));
   if (!buffer) {
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   *pBuffer = borg_buffer_to_handle(buffer);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_DestroyBuffer(VkDevice device,
                  VkBuffer _buffer,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borg_device, dev, device);
   VK_FROM_HANDLE(borg_buffer, buffer, _buffer);

   if (!buffer)
      return;

   vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR void VKAPI_CALL
borg_GetDeviceBufferMemoryRequirements(
   VkDevice device,
   const VkDeviceBufferMemoryRequirements *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   //puts("borg_GetDeviceBufferMemoryRequirements");

   VK_FROM_HANDLE(borg_device, dev, device);
   struct borg_physical_device *pdev = borg_device_physical(dev);

   const uint32_t alignment = 4;

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements) {
      .size = align64(pInfo->pCreateInfo->size, alignment),
      .alignment = alignment,
      .memoryTypeBits = BITFIELD_MASK(pdev->mem_type_count),
   };
}

static VkResult
borg_bind_buffer_memory(struct borg_device *dev, const VkBindBufferMemoryInfo *info)
{
   VK_FROM_HANDLE(borg_device_memory, mem, info->memory);
   VK_FROM_HANDLE(borg_buffer, buffer, info->buffer);
   VkResult result = VK_SUCCESS;

   assert(mem->mem);
   assert(mem->mem->va);
   assert(info);

   buffer->addr = mem->mem->va->addr + info->memoryOffset;
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_BindBufferMemory2(VkDevice device,
                      uint32_t bindInfoCount,
                      const VkBindBufferMemoryInfo *pBindInfos)
{
   //puts("borg_BindBufferMemory2");
   VK_FROM_HANDLE(borg_device, dev, device);
   VkResult first_error_or_success = VK_SUCCESS;

   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VkResult result = borg_bind_buffer_memory(dev, &pBindInfos[i]);
        const VkBindMemoryStatusKHR *status =
           vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
        if (status != NULL && status->pResult != NULL)
           *status->pResult = result;

        if (first_error_or_success == VK_SUCCESS)
           first_error_or_success = result;
     }
   return first_error_or_success;
}
