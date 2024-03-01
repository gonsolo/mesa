/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_device.h"

#include "borg_entrypoints.h"
#include "borg_instance.h"
#include "borg_physical_device.h"
#include "vulkan/runtime/vk_object.h"
#include "vk_alloc.h"
#include "vk_log.h"

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateDevice(VkPhysicalDevice physicalDevice,
                 const VkDeviceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkDevice *pDevice)
{
   VK_FROM_HANDLE(borg_physical_device, pdev, physicalDevice);
   struct borg_device *dev;

   dev = vk_zalloc2(&pdev->vk.instance->alloc, pAllocator,
                    sizeof(*dev), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);


   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &borg_device_entrypoints, true);

   VkResult result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table,
                           pCreateInfo, pAllocator);
   if (result != VK_SUCCESS)
      return result;

  *pDevice = borg_device_to_handle(dev);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   // TODO
}
