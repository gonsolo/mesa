/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include <alloca.h>
#include <sys/mman.h>
#include "borg_device.h"
#include "borg_device_memory.h"
#include "borg_entrypoints.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_object.h"

VKAPI_ATTR VkResult VKAPI_CALL
borg_AllocateMemory(VkDevice device,
                   const VkMemoryAllocateInfo *pAllocateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(borg_device, dev, device);

   struct borg_device_memory *mem;

   printf("borg_AllocateMemory: allocationSize: %li\n", pAllocateInfo->allocationSize);

   mem = vk_device_memory_create(&dev->vk, pAllocateInfo,
                                pAllocator, sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->map = NULL;

   mem->todo_bo_size = pAllocateInfo->allocationSize;
   mem->todo_bo_mem = calloc(1, mem->todo_bo_size);

   *pMem = borg_device_memory_to_handle(mem);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_MapMemory2KHR(VkDevice device,
                   const VkMemoryMapInfoKHR *pMemoryMapInfo,
                   void **ppData)
{
   puts("borg_MapMemory2KHR");
   VK_FROM_HANDLE(borg_device_memory, mem, pMemoryMapInfo->memory);

   off_t offset = 0;
   mem->map = mem->todo_bo_mem;
   *ppData = mem->map + offset;
   printf("  address or ppData: %p\n", ppData);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_UnmapMemory2KHR(VkDevice device,
                     const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(borg_device_memory, mem, pMemoryUnmapInfo->memory);
   if (mem == NULL)
      return VK_SUCCESS;
   mem->map = NULL;
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_FreeMemory(VkDevice device,
                VkDeviceMemory _mem,
                const VkAllocationCallbacks *pAllocator)
{
   // TODO
}
