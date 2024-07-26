/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include <alloca.h>
#include <sys/mman.h>

#include "drm-uapi/borg_drm.h"

#include "borg_bo.h"
#include "borg_device.h"
#include "borg_device_memory.h"
#include "borg_entrypoints.h"

#include "util/u_memory.h"
#include "util/vma.h"

#include "vk_log.h"
#include "vk_object.h"

static VkResult MUST_CHECK
alloc_heap_addr_locked(struct borg_device *dev,
                       struct vk_object_base *log_obj,
                       uint64_t size_B,
                       uint64_t *addr_out)
{
   uint64_t align_B = 4;

   *addr_out = util_vma_heap_alloc(&dev->heap, size_B, align_B);
   if (*addr_out == 0)
      return vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY, "Failed to allocate virtual address range");

   return VK_SUCCESS;
}

static VkResult
alloc_heap_addr(struct borg_device *dev,
                struct vk_object_base *log_obj,
                uint64_t size_B,
                uint64_t *addr_out)
{
   simple_mtx_lock(&dev->heap_mutex);
   VkResult result = alloc_heap_addr_locked(dev, log_obj, size_B, addr_out);
   simple_mtx_unlock(&dev->heap_mutex);
   return result;
}

static VkResult
borg_alloc_va(struct borg_device *dev, struct vk_object_base *log_obj, uint64_t size_B, struct borg_va **va_out)
{
   VkResult result;

   struct borg_va *va = CALLOC_STRUCT(borg_va);
   if (va == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = alloc_heap_addr(dev, log_obj, size_B, &va->addr);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   *va_out = va;

fail_alloc:
   FREE(va);
   return result;
}

static VkResult
create_mem_or_close_bo(struct borg_device* dev,
                       struct borg_ws_bo *bo,
                       struct vk_object_base *log_obj)
{
   const uint64_t size_B = bo->size;
   VkResult result;

   struct borg_mem *mem = CALLOC_STRUCT(borg_mem);
   if (mem == NULL) {
      result = vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_bo;
   }
   mem->bo = bo;

   result = borg_alloc_va(dev, log_obj, size_B, &mem->va);
   if (result != VK_SUCCESS)
      goto fail_mem;

fail_mem:
   FREE(mem);
fail_bo:
   borg_ws_bo_destroy(mem->bo);

   return result;
}

static VkResult
borg_alloc_mem(struct borg_device *dev,
               struct vk_object_base *log_obj,
               uint64_t size)
{
   struct borg_ws_bo *bo = borg_ws_bo_new(dev->ws_dev, size);
   if (bo == NULL)
        return vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY, "%m");

   return create_mem_or_close_bo(dev, bo, log_obj);
}

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

   result = borg_alloc_mem(dev, &dev->vk.base, pAllocateInfo->allocationSize);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   *pMem = borg_device_memory_to_handle(mem);

   return result;

fail_alloc:
     vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
     return result;
}

static void borg_mem_unmap(struct borg_mem *mem)
{
   if (mem->map != NULL) {
      munmap(mem->map, mem->size_B);
      mem->map = NULL;
   }
}

static VkResult MUST_CHECK
vm_bind(struct borg_device *dev,
        struct vk_object_base *log_obj,
        struct drm_borg_vm_bind_op *op)
{
   struct drm_borg_vm_bind vmbind = {
      .op_count = 1,
      .op_ptr = (uint64_t)(uintptr_t)(void *)op,
   };
   int err = drmCommandWriteRead(dev->ws_dev->fd, DRM_BORG_VM_BIND,
                                 &vmbind, sizeof(vmbind));
   if (err)
      return vk_errorf(log_obj, VK_ERROR_UNKNOWN, "vm_bind failed: %m");

   return VK_SUCCESS;
}

static void
free_heap_addr(struct borg_device *dev,
               uint64_t addr, uint64_t size_B)
{
   simple_mtx_lock(&dev->heap_mutex);
   assert(addr >= BORG_HEAP_START);
   assert(addr <= BORG_HEAP_END);
   assert(addr + size_B <= BORG_HEAP_END);
   util_vma_heap_free(&dev->heap, addr, size_B);
   simple_mtx_unlock(&dev->heap_mutex);
}

static void borg_va_free(struct borg_va *va)
{
   VkResult result = VK_SUCCESS;
   struct borg_device* dev = va->dev;

   {
      struct drm_borg_vm_bind_op bind = {
         .op = DRM_BORG_VM_BIND_OP_UNMAP,
         .addr = va->addr,
         .range = va->size_B,
      };
      result |= vm_bind(dev, NULL, &bind);
   }
  /* If unbinding fails, we leak the VA range */
   if (result == VK_SUCCESS)
      free_heap_addr(dev, va->addr, va->size_B);

   FREE(va);

}

static void borg_mem_free(struct borg_mem *mem)
{
   borg_va_free(mem->va);
   borg_ws_bo_destroy(mem->bo);
   FREE(mem);
}

static void
borg_mem_unref(struct borg_mem *mem)
{
   borg_mem_unmap(mem);
   borg_mem_free(mem);
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

        borg_mem_unref(mem->mem);

        vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

static inline VkResult MUST_CHECK
borg_mem_map(struct borg_device *dev, struct borg_mem *mem, struct vk_object_base *log_obj, void **map_out)
{
   assert(mem->map == NULL);

   void *fixed_addr = NULL;
   int prot = 0;
   prot |= PROT_READ;
   prot |= PROT_WRITE;
   int flags = MAP_SHARED;

   void *map = mmap(fixed_addr, mem->size_B, prot, flags,
                    dev->ws_dev->fd, mem->bo->map_handle);
   if (map == MAP_FAILED)
      return vk_error(log_obj, VK_ERROR_MEMORY_MAP_FAILED);

   mem->map = map;

   *map_out = mem->map;

   return VK_SUCCESS;
}


VKAPI_ATTR VkResult VKAPI_CALL
borg_MapMemory2KHR(VkDevice device,
                   const VkMemoryMapInfoKHR *pMemoryMapInfo,
                   void **ppData)
{
   puts("borg_MapMemory2KHR");

   VkResult result;

   VK_FROM_HANDLE(borg_device, dev, device);
   VK_FROM_HANDLE(borg_device_memory, mem, pMemoryMapInfo->memory);

   if (mem == NULL) {
      *ppData = NULL;
      return VK_SUCCESS;
   }

   const VkDeviceSize offset = pMemoryMapInfo->offset;
   const VkDeviceSize size = vk_device_memory_range(&mem->vk,
                                                    pMemoryMapInfo->offset,
                                                    pMemoryMapInfo->size);
   assert(size > 0);
   assert(offset + size <= mem->mem->size_B);

   if (size != (size_t)size) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "requested size 0x%"PRIx64" does not fit in %u bits",
                       size, (unsigned)(sizeof(size_t) * 8));
   }
   if (mem->mem->map != NULL) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "Memory object already mapped.");
   }


   void *mem_map = NULL;
   result = borg_mem_map(dev, mem->mem, &mem->vk.base, &mem_map);
   if (result != VK_SUCCESS)
      return result;

   *ppData = mem_map + offset;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_UnmapMemory2KHR(VkDevice device,
                     const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(borg_device_memory, mem, pMemoryUnmapInfo->memory);
   if (mem == NULL)
      return VK_SUCCESS;

   borg_mem_unmap(mem->mem);

   return VK_SUCCESS;
}

