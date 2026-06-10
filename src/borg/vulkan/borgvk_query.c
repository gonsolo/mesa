/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Minimal query-pool stubs. borgvk does no host-side timestamping or occlusion
 * queries, but Mesa's common WSI references vkCreateQueryPool/vkDestroyQueryPool
 * for swapchain present-timing bookkeeping (and calls DestroyQueryPool even when
 * no pool was created). There's no vk_common implementation for these, so the
 * driver must provide non-NULL entrypoints. They allocate an opaque object and
 * report zeroed results.
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_log.h"

#include <string.h>

struct borgvk_query_pool {
   struct vk_object_base base;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_query_pool, base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateQueryPool(VkDevice _device,
                       const VkQueryPoolCreateInfo *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);

   struct borgvk_query_pool *pool =
      vk_object_zalloc(&device->vk, pAllocator, sizeof(*pool),
                       VK_OBJECT_TYPE_QUERY_POOL);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pQueryPool = borgvk_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyQueryPool(VkDevice _device, VkQueryPool _pool,
                        const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_query_pool, pool, _pool);

   if (!pool)
      return;

   vk_object_free(&device->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_GetQueryPoolResults(VkDevice _device, VkQueryPool queryPool,
                           uint32_t firstQuery, uint32_t queryCount,
                           size_t dataSize, void *pData, VkDeviceSize stride,
                           VkQueryResultFlags flags)
{
   if (pData && dataSize)
      memset(pData, 0, dataSize);
   return VK_SUCCESS;
}
