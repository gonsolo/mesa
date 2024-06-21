/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_cmd_buffer.h"
#include "borg_device.h"
#include "borg_entrypoints.h"
#include "borg_physical_device.h"
#include "borg_shader.h"

#include "vulkan/runtime/vk_object.h"
#include "vulkan/wsi/wsi_common.h"

#include "vk_alloc.h"
#include "vk_log.h"

#include <xf86drm.h>

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
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                               &wsi_device_entrypoints, false);

   VkResult result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table,
                           pCreateInfo, pAllocator);
   if (result != VK_SUCCESS)
      return result;

   dev->vk.shader_ops = &borg_device_shader_ops;

   drmDevicePtr drm_device = NULL;
   int ret = drmGetDeviceFromDevId(pdev->render_dev, 0, &drm_device);
   if (ret != 0) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to get DRM device: %m");
      return result;
   }

   dev->vk.command_buffer_ops = &borg_cmd_buffer_ops;

   result = borg_queue_init(dev, &dev->queue, &pCreateInfo->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS)
      return result;

   //dev->pdev = pdev;

   *pDevice = borg_device_to_handle(dev);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   // TODO
}
