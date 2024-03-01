/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_buffer.h"
#include "borg_device.h"
#include "borg_entrypoints.h"
#include "vk_log.h"

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateBuffer(VkDevice device,
                 const VkBufferCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(borg_device, dev, device);
   struct borg_buffer *buffer;

   buffer = vk_buffer_create(&dev->vk, pCreateInfo, pAllocator,
                             sizeof(*buffer));
   if (!buffer)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

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
   // TODO
}
