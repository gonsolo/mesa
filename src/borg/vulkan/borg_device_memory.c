/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include <alloca.h>
#include <sys/mman.h>

#include "borg_bo.h"

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
   VkResult result = VK_SUCCESS;

   printf("borg_AllocateMemory: allocationSize: %li\n", pAllocateInfo->allocationSize);

   mem = vk_device_memory_create(&dev->vk, pAllocateInfo,
                                pAllocator, sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->map = NULL;

   mem->bo = borg_ws_bo_new(dev->ws_dev, pAllocateInfo->allocationSize);
   if (!mem->bo) {
          result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
          goto fail_alloc;
   }

   //mem->todo_bo_size = pAllocateInfo->allocationSize;
   //mem->todo_bo_mem = calloc(1, mem->todo_bo_size);

   *pMem = borg_device_memory_to_handle(mem);

   return VK_SUCCESS;

fail_alloc:
     vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
     return result;
}

VKAPI_ATTR void VKAPI_CALL
borg_FreeMemory(VkDevice device,
                VkDeviceMemory _mem,
                const VkAllocationCallbacks *pAllocator)
{
        VK_FROM_HANDLE(borg_device, dev, device);
        VK_FROM_HANDLE(borg_device_memory, mem, _mem);

        if (!mem)
                return;

        if (mem->map)
               borg_ws_bo_unmap(mem->bo, mem->map);

        borg_ws_bo_destroy(mem->bo);

        vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}
        
VKAPI_ATTR VkResult VKAPI_CALL
borg_MapMemory2KHR(VkDevice device,
                   const VkMemoryMapInfoKHR *pMemoryMapInfo,
                   void **ppData)
{
   puts("borg_MapMemory2KHR");
   VK_FROM_HANDLE(borg_device, dev, device);
   VK_FROM_HANDLE(borg_device_memory, mem, pMemoryMapInfo->memory);

   off_t offset = 0;
   //mem->map = mem->todo_bo_mem;
   mem->map = borg_ws_bo_map(mem->bo);
   if (mem->map == NULL) {
     return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                         "Memory object couldn't be mapped.");
   }

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

   borg_ws_bo_unmap(mem->bo, mem->map);

   mem->map = NULL;
   return VK_SUCCESS;
}


