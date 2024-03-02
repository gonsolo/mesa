/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_entrypoints.h"
#include "borg_instance.h"
#include "borg_physical_device.h"
#include "borg_private.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_util.h"
#include <xf86drm.h>

VkResult borg_create_drm_physical_device(struct vk_instance *vk_instance,
                                          struct _drmDevice *drm_device,
                                          struct vk_physical_device **pdev_out)
{
   VkResult result;

   struct borg_instance *instance = (struct borg_instance *)vk_instance;

   struct borg_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev),
                8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (pdev == NULL) {
      result = vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      return result;
   }

   //struct vk_device_extension_table supported_extensions;
   //struct vk_features supported_features;
   struct vk_properties properties = {
      .apiVersion = VK_MAKE_VERSION(1, 0, VK_HEADER_VERSION),
      .driverVersion = vk_get_driver_version(),
      .vendorID = 0x666,
      .deviceID = 0x9000,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,

      // Vulkan 1.0 limits
      .maxComputeSharedMemorySize = BORG_MAX_SHARED_SIZE,
   };
   snprintf(properties.deviceName, sizeof(properties.deviceName), "%s", "Borg 9000");

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &borg_physical_device_entrypoints, true);

   result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                      NULL, //&supported_extensions,
                                      NULL, //&supported_features,
                                      &properties,
                                      &dispatch_table
				    );

   *pdev_out = &pdev->vk;
   return result;
}

void borg_physical_device_destroy(struct vk_physical_device *vk_device)
{
   //vk_free(&device->instance->vk.alloc, device);
   vk_logd(VK_LOG_OBJS(vk_device), "borg physical device destroyed");
}

VKAPI_ATTR void VKAPI_CALL
  borg_GetPhysicalDeviceMemoryProperties2(
     VkPhysicalDevice physicalDevice,
     VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   // TODO
}

VKAPI_ATTR void VKAPI_CALL
  borg_GetPhysicalDeviceQueueFamilyProperties2(
     VkPhysicalDevice physicalDevice,
     uint32_t *pQueueFamilyPropertyCount,
     VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   *pQueueFamilyPropertyCount = 1;
   // TODO
}

