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
      .maxMemoryAllocationCount = 4096,
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

   pdev->mem_heap_count = 1;
   pdev->mem_heaps[0] = (struct borg_memory_heap) {
      .size = 1024,
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT

   };
   pdev->mem_type_count = 1;
   pdev->mem_types[0] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      .heapIndex = 0
   };

   pdev->queue_families[pdev->queue_family_count++] = (struct borg_queue_family) {
      .queue_flags = VK_QUEUE_COMPUTE_BIT,
      .queue_count = 1,
   };
   assert(pdev->queue_family_count <= ARRAY_SIZE(pdev->queue_families));

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
   VK_FROM_HANDLE(borg_physical_device, pdev, physicalDevice);

   pMemoryProperties->memoryProperties.memoryHeapCount = pdev->mem_heap_count;
   for (int i = 0; i < pdev->mem_heap_count; i++) {
      pMemoryProperties->memoryProperties.memoryHeaps[i] = (VkMemoryHeap) {
         .size = pdev->mem_heaps[i].size,
         .flags = pdev->mem_heaps[i].flags,
      };
   }
   pMemoryProperties->memoryProperties.memoryTypeCount = pdev->mem_type_count;
   for (int i = 0; i < pdev->mem_type_count; i++) {
      pMemoryProperties->memoryProperties.memoryTypes[i] = pdev->mem_types[i];
    }
}

VKAPI_ATTR void VKAPI_CALL
  borg_GetPhysicalDeviceQueueFamilyProperties2(
     VkPhysicalDevice physicalDevice,
     uint32_t *pQueueFamilyPropertyCount,
     VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(borg_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                            pQueueFamilyPropertyCount);

   for (uint8_t i = 0; i < pdev->queue_family_count; i++) {
      const struct borg_queue_family *queue_family = &pdev->queue_families[i];

      vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
         p->queueFamilyProperties.queueFlags = queue_family->queue_flags;
         p->queueFamilyProperties.queueCount = queue_family->queue_count;
         p->queueFamilyProperties.timestampValidBits = 64;
         p->queueFamilyProperties.minImageTransferGranularity =
            (VkExtent3D){1, 1, 1};
      }
   }
}

