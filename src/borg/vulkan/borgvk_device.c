/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Instance + physical-device bring-up for borgvk. Phase 1 goal: a single fake
 * "Borg (ULX3S)" physical device that `vulkaninfo` can enumerate. There is no
 * DRM device — the real GPU is the FPGA reached over serial — so device
 * enumeration unconditionally creates one physical device.
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_util.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"

#include "util/log.h"

#include <stdio.h>

#define BORGVK_API_VERSION VK_API_VERSION_1_3

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   *pApiVersion = BORGVK_API_VERSION;
   return VK_SUCCESS;
}

/* Instance extensions advertised by borgvk. The WSI surface extensions let an
 * app create a swapchain so its render loop runs; the actual pixels go to the
 * FPGA over serial, the host window is just a formality (software WSI path). */
static const struct vk_instance_extension_table instance_extensions = {
   .KHR_device_group_creation           = true,
   .KHR_external_fence_capabilities     = true,
   .KHR_external_memory_capabilities    = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .EXT_debug_report                    = true,
   .EXT_debug_utils                     = true,
#ifdef BORGVK_USE_WSI_PLATFORM
   .KHR_surface                         = true,
   .KHR_get_surface_capabilities2       = true,
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
   .KHR_xcb_surface                     = true,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
   .KHR_xlib_surface                    = true,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
   .KHR_wayland_surface                 = true,
#endif
};

/* Device extensions advertised by borgvk: the swapchain so apps can present. */
static const struct vk_device_extension_table device_extensions = {
#ifdef BORGVK_USE_WSI_PLATFORM
   .KHR_swapchain                       = true,
#endif
};

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                            uint32_t *pPropertyCount,
                                            VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(
      &instance_extensions, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                        VkLayerProperties *pProperties)
{
   if (pProperties == NULL) {
      *pPropertyCount = 0;
      return VK_SUCCESS;
   }
   return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);
}

/* One queue family: graphics + compute + transfer, a single queue. */
static const VkQueueFamilyProperties borgvk_queue_family_properties = {
   .queueFlags = VK_QUEUE_GRAPHICS_BIT |
                 VK_QUEUE_COMPUTE_BIT |
                 VK_QUEUE_TRANSFER_BIT,
   .queueCount = 1,
   .timestampValidBits = 0,
   .minImageTransferGranularity = { 1, 1, 1 },
};

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice,
   uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out,
                          pQueueFamilyProperties, pQueueFamilyPropertyCount);

   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = borgvk_queue_family_properties;
      vk_foreach_struct(s, p->pNext) {
         vk_debug_ignored_stype(s->sType);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(borgvk_physical_device, device, physicalDevice);

   pMemoryProperties->memoryProperties = device->memory;

   vk_foreach_struct(ext, pMemoryProperties->pNext) {
      vk_debug_ignored_stype(ext->sType);
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceFormatProperties2(
   VkPhysicalDevice physicalDevice,
   VkFormat format,
   VkFormatProperties2 *pFormatProperties)
{
   /* borgvk doesn't render on the host — it forwards an MVP to the FPGA — so we
    * don't actually consume these formats. Report a generous, uniform feature
    * set so WSI can pick a swapchain format and apps (cube.c) pass their format
    * capability checks (color/depth attachment, sampling, blit, transfer). */
   const VkFormatFeatureFlags img =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_BLIT_SRC_BIT |
      VK_FORMAT_FEATURE_BLIT_DST_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   const VkFormatFeatureFlags buf =
      VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT |
      VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
      VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;

   pFormatProperties->formatProperties = (VkFormatProperties){
      .linearTilingFeatures  = img,
      .optimalTilingFeatures = img,
      .bufferFeatures        = buf,
   };

   vk_foreach_struct(ext, pFormatProperties->pNext) {
      vk_debug_ignored_stype(ext->sType);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_GetPhysicalDeviceImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
   VkImageFormatProperties2 *pImageFormatProperties)
{
   /* Permissive: every (format, type, tiling, usage) the app or WSI asks for is
    * "supported" with generous limits. We never allocate device-real images. */
   const uint32_t max2d = 16384;
   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
      .maxExtent = {
         .width  = pImageFormatInfo->type == VK_IMAGE_TYPE_1D ? max2d : max2d,
         .height = pImageFormatInfo->type == VK_IMAGE_TYPE_1D ? 1 : max2d,
         .depth  = pImageFormatInfo->type == VK_IMAGE_TYPE_3D ? 2048 : 1,
      },
      .maxMipLevels   = 15,
      .maxArrayLayers = 2048,
      .sampleCounts   = VK_SAMPLE_COUNT_1_BIT,
      .maxResourceSize = (VkDeviceSize)1 << 31,
   };

   vk_foreach_struct(ext, pImageFormatProperties->pNext) {
      vk_debug_ignored_stype(ext->sType);
   }
   return VK_SUCCESS;
}

static void
borgvk_get_properties(struct vk_properties *p)
{
   *p = (struct vk_properties){
      .apiVersion = BORGVK_API_VERSION,
      .driverVersion = VK_MAKE_VERSION(0, 1, 0),
      .vendorID = 0x1de0,  /* not a registered ID; placeholder for "Borg" */
      .deviceID = 0xb049,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER,

      /* Vulkan 1.0 limits — modest, sane values so vulkaninfo and apps see a
       * coherent device. Tightened/grown as real features come online. */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      .maxImageDimension3D = 2048,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 2048,
      .maxTexelBufferElements = 65536,
      .maxUniformBufferRange = 65536,
      .maxStorageBufferRange = UINT32_MAX,
      .maxPushConstantsSize = 128,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 1,
      .maxBoundDescriptorSets = 8,
      .maxPerStageResources = 128,
      .maxVertexInputAttributes = 16,
      .maxVertexInputBindings = 16,
      .maxVertexInputAttributeOffset = 2047,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 64,
      .maxFragmentInputComponents = 64,
      .maxFragmentOutputAttachments = 4,
      .maxColorAttachments = 4,
      .maxFramebufferWidth = 16384,
      .maxFramebufferHeight = 16384,
      .maxFramebufferLayers = 256,
      .maxViewports = 1,
      .maxViewportDimensions = { 16384, 16384 },
      .viewportBoundsRange = { -32768.0f, 32767.0f },
      .framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSamplerAnisotropy = 1.0f,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT32_MAX,
      .pointSizeRange = { 1.0f, 1.0f },
      .lineWidthRange = { 1.0f, 1.0f },
      .pointSizeGranularity = 0.0f,
      .lineWidthGranularity = 0.0f,

      /* Vulkan 1.2 driver identification */
      .driverID = VK_DRIVER_ID_MESA_DOZEN,  /* placeholder until a Borg ID exists */
      .conformanceVersion = { 1, 3, 0, 0 },
   };

   snprintf(p->deviceName, sizeof(p->deviceName), "Borg (ULX3S)");
   snprintf(p->driverName, sizeof(p->driverName), "borgvk");
   snprintf(p->driverInfo, sizeof(p->driverInfo), "Mesa borgvk (serial/FPGA)");
}

static VkResult
create_physical_device(struct borgvk_instance *instance)
{
   struct borgvk_physical_device *device =
      vk_zalloc(&instance->vk.alloc, sizeof(*device), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!device)
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &borgvk_physical_device_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_physical_device_entrypoints, false);
#endif

   struct vk_features features = { 0 };
   struct vk_properties properties;
   borgvk_get_properties(&properties);

   VkResult result =
      vk_physical_device_init(&device->vk, &instance->vk, &device_extensions,
                              &features, &properties, &dispatch_table);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, device);
      return result;
   }

   /* Single host-visible coherent heap (host RAM staged to the FPGA). */
   device->memory.memoryHeapCount = 1;
   device->memory.memoryHeaps[0].size = 256ull * 1024 * 1024;
   device->memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
   device->memory.memoryTypeCount = 1;
   device->memory.memoryTypes[0].propertyFlags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   device->memory.memoryTypes[0].heapIndex = 0;

   device->sync_types[0] = &borgvk_sync_type;
   device->sync_types[1] = NULL;
   device->vk.supported_sync_types = device->sync_types;

   /* WSI init must follow supported_sync_types: wsi_device_init queries external
    * semaphore/fence properties, which walk that list. */
#ifdef BORGVK_USE_WSI_PLATFORM
   result = borgvk_wsi_init(device);
   if (result != VK_SUCCESS) {
      vk_physical_device_finish(&device->vk);
      vk_free(&instance->vk.alloc, device);
      return result;
   }
#endif

   list_addtail(&device->vk.link, &instance->vk.physical_devices.list);

   return VK_SUCCESS;
}

static VkResult
enumerate_devices(struct vk_instance *vk_instance)
{
   struct borgvk_instance *instance =
      container_of(vk_instance, struct borgvk_instance, vk);

   return create_physical_device(instance);
}

static void
destroy_physical_device(struct vk_physical_device *pdev)
{
#ifdef BORGVK_USE_WSI_PLATFORM
   borgvk_wsi_finish(container_of(pdev, struct borgvk_physical_device, vk));
#endif
   vk_physical_device_finish(pdev);
   vk_free(&pdev->instance->alloc, pdev);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkInstance *pInstance)
{
   struct borgvk_instance *instance;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);

   if (pAllocator == NULL)
      pAllocator = vk_default_allocator();

   instance = vk_alloc(pAllocator, sizeof(*instance), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &borgvk_instance_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_instance_entrypoints, false);
#endif

   result = vk_instance_init(&instance->vk, &instance_extensions,
                             &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return vk_error(NULL, result);
   }

   instance->vk.physical_devices.enumerate = enumerate_devices;
   instance->vk.physical_devices.destroy = destroy_physical_device;

   *pInstance = borgvk_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyInstance(VkInstance _instance,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_instance, instance, _instance);

   if (!instance)
      return;

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateDevice(VkPhysicalDevice physicalDevice,
                    const VkDeviceCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkDevice *pDevice)
{
   VK_FROM_HANDLE(borgvk_physical_device, physical_device, physicalDevice);
   struct borgvk_device *device;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);

   device = vk_zalloc2(&physical_device->vk.instance->alloc, pAllocator,
                       sizeof(*device), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Main device dispatch: our entrypoints, then fill every vkCmd* we don't
    * implement with the runtime's record-into-queue emulation. */
   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &borgvk_device_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);
#endif
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vk_cmd_enqueue_device_entrypoints, false);

   /* Dispatch used to replay a recorded queue (secondary cmd buffers). */
   vk_device_dispatch_table_from_entrypoints(&device->cmd_dispatch,
                                             &borgvk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&device->cmd_dispatch,
                                             &vk_common_device_entrypoints, false);

   result = vk_device_init(&device->vk, &physical_device->vk,
                           &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&physical_device->vk.instance->alloc, pAllocator, device);
      return vk_error(physical_device, result);
   }

   device->vk.command_dispatch_table = &device->cmd_dispatch;
   device->vk.command_buffer_ops = &borgvk_cmd_buffer_ops;

   /* One queue from family 0. */
   assert(pCreateInfo->queueCreateInfoCount >= 1);
   result = vk_queue_init(&device->queue, &device->vk,
                          &pCreateInfo->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS) {
      vk_device_finish(&device->vk);
      vk_free2(&physical_device->vk.instance->alloc, pAllocator, device);
      return vk_error(physical_device, result);
   }

   /* On every vkQueueSubmit, read the MVP from the bound uniform buffer and
    * ship it to the FPGA over serial. This is the driver's whole render path. */
   device->queue.driver_submit = borgvk_queue_submit;

   *pDevice = borgvk_device_to_handle(device);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);

   if (!device)
      return;

   vk_queue_finish(&device->queue);
   vk_device_finish(&device->vk);
   vk_free2(&device->vk.alloc, pAllocator, device);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
borgvk_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(borgvk_instance, instance, _instance);
   return vk_instance_get_proc_addr(instance ? &instance->vk : NULL,
                                    &borgvk_instance_entrypoints, pName);
}

/* With version 1+ of the loader interface the ICD should expose
 * vk_icdGetInstanceProcAddr to work around certain LD_PRELOAD issues. */
PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return borgvk_GetInstanceProcAddr(instance, pName);
}
