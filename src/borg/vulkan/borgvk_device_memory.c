/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "drm-uapi/borg_drm.h"

#include <sys/mman.h>
#include "borgvk_drm_compat.h"

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
