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

#include <sys/stat.h>
#include <xf86drm.h>

#define BORG_VENDOR_ID 0x3333 // seems to be unused

VkResult dummy_syncobj_init(struct vk_device *device,
                    struct vk_sync *sync,
                    uint64_t initial_value);
VkResult dummy_syncobj_init(struct vk_device *device,
                    struct vk_sync *sync,
                    uint64_t initial_value)
{
   return VK_SUCCESS;
}

VkResult dummy_syncobj_wait(struct vk_device *device,
                            struct vk_sync *sync,
                            uint64_t wait_value,
                            enum vk_sync_wait_flags wait_flags,
                            uint64_t abs_timeout_ns);
VkResult dummy_syncobj_wait(struct vk_device *device,
                            struct vk_sync *sync,
                            uint64_t wait_value,
                            enum vk_sync_wait_flags wait_flags,
                            uint64_t abs_timeout_ns)
{
   return VK_SUCCESS;
}

VkResult dumm_syncobj_reset(struct vk_device *device,
                            struct vk_sync *sync);
VkResult dumm_syncobj_reset(struct vk_device *device,
                            struct vk_sync *sync)
{
   return VK_SUCCESS;
}

void dummy_syncobj_finish(struct vk_device *device,
                          struct vk_sync *sync);
void dummy_syncobj_finish(struct vk_device *device,
                          struct vk_sync *sync)
{}

VkResult borg_create_drm_physical_device(struct vk_instance *vk_instance,
                                          struct _drmDevice *drm_device,
                                          struct vk_physical_device **pdev_out)
{
   VkResult result;

   puts(__func__);

   struct borg_instance *instance = (struct borg_instance *)vk_instance;

   if (!(drm_device->available_nodes & (1 << DRM_NODE_RENDER))) {
      puts("Borg: No render node!");
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   switch (drm_device->bustype) {
      case DRM_BUS_PCI:
         if (drm_device->deviceinfo.pci->vendor_id != BORG_VENDOR_ID) {
            puts("Borg: PCI vendor id is not Borg!");
            return VK_ERROR_INCOMPATIBLE_DRIVER;
         }
         break;
      case DRM_BUS_PLATFORM: {
         const char *compat_prefix = "borg,";
         bool found = false;
         for (int i = 0; drm_device->deviceinfo.platform->compatible[i] != NULL; i++) {
            if (strncmp(drm_device->deviceinfo.platform->compatible[0], compat_prefix, strlen(compat_prefix)) == 0) {
               found = true;
               break;
            }
         }
         if (!found) {
            puts("Borg: No compatible device found!");
            return VK_ERROR_INCOMPATIBLE_DRIVER;
         }
         break;
      }
      default:
         puts("Borg: Unknown bus type!");
         return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   struct stat st;
   if (stat(drm_device->nodes[DRM_NODE_RENDER], &st)) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                         "fstat() failed on %s: %m",
                         drm_device->nodes[DRM_NODE_RENDER]);
      perror("gonso 1");
      return result;
   }
   const dev_t render_dev = st.st_rdev;

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
      .maxBoundDescriptorSets = 4,
      .maxComputeSharedMemorySize = BORG_MAX_SHARED_SIZE,
      .maxComputeWorkGroupCount = {65535,65535,65535},
      .maxComputeWorkGroupInvocations = 128,
      .maxDescriptorSetStorageBuffers =  96,
      .maxMemoryAllocationCount = 4096,
      .maxPerStageDescriptorStorageBuffers = 4,
      .maxPerStageResources = 128,
      .maxStorageBufferRange = 1 << 27,
      .maxComputeWorkGroupSize = {32, 1, 1},
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

   pdev->syncobj_sync_type = (struct vk_sync_type) {
      .size = sizeof(struct vk_sync),
      .features = VK_SYNC_FEATURE_BINARY |
                  VK_SYNC_FEATURE_CPU_WAIT |
                  VK_SYNC_FEATURE_CPU_RESET,
      .init = dummy_syncobj_init,
      .finish = dummy_syncobj_finish,
      //.signal = vk_drm_syncobj_signal,
      .reset = dumm_syncobj_reset,
      //.move = vk_drm_syncobj_move,
      //.import_opaque_fd = dummy_import_opaque_fd,
      //.export_opaque_fd = vk_drm_syncobj_export_opaque_fd,
      //.import_sync_file = vk_drm_syncobj_import_sync_file,
      //.export_sync_file = vk_drm_syncobj_export_sync_file,
      .wait = dummy_syncobj_wait,
     };

   pdev->sync_types[0] = &pdev->syncobj_sync_type;
   pdev->sync_types[1] = NULL;
   pdev->vk.supported_sync_types = pdev->sync_types;

   pdev->render_dev = render_dev;

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

