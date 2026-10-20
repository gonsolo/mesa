/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_util.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"

#include "util/log.h"
#include "drm-uapi/borg_drm.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include "borgvk_drm_compat.h"

#define BORGVK_API_VERSION VK_API_VERSION_1_0

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   *pApiVersion = BORGVK_API_VERSION;
   return VK_SUCCESS;
}

static const struct vk_instance_extension_table instance_extensions = {
   .KHR_external_fence_capabilities     = true,
   .KHR_external_memory_capabilities    = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .EXT_debug_report                    = true,
   .EXT_debug_utils                     = true,
#ifdef BORGVK_USE_WSI_PLATFORM
   .KHR_surface                         = true,
   .KHR_get_surface_capabilities2       = true,
   .EXT_headless_surface                = true,
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

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkInstance *pInstance)
{
   struct borgvk_instance *instance;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);

   /* A Vulkan 1.0 implementation rejects a requested apiVersion it does not support. */
   if (pCreateInfo->pApplicationInfo && pCreateInfo->pApplicationInfo->apiVersion != 0) {
      const uint32_t req = pCreateInfo->pApplicationInfo->apiVersion;
      if (VK_API_VERSION_MAJOR(req) != VK_API_VERSION_MAJOR(BORGVK_API_VERSION) ||
          VK_API_VERSION_MINOR(req) > VK_API_VERSION_MINOR(BORGVK_API_VERSION))
         return vk_error(NULL, VK_ERROR_INCOMPATIBLE_DRIVER);
   }

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

   instance->vk.physical_devices.enumerate = borgvk_enumerate_devices;
   instance->vk.physical_devices.destroy = borgvk_destroy_physical_device;

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

/* Body for vkCmd* entries borgvk does not implement; see the cmd_dispatch fill-in. */
static void
borgvk_noop_cmd(void)
{
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

   vk_foreach_struct_const(f_stype, f, pCreateInfo->pNext) {
      switch (f_stype) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES:
         if (((const VkPhysicalDeviceProtectedMemoryFeatures *)f)->protectedMemory)
            return vk_error(physical_device, VK_ERROR_FEATURE_NOT_PRESENT);
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES:
         if (((const VkPhysicalDeviceShaderDrawParametersFeatures *)f)->shaderDrawParameters)
            return vk_error(physical_device, VK_ERROR_FEATURE_NOT_PRESENT);
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
         const VkPhysicalDeviceVulkan11Features *v11 = (const void *)f;
         if (v11->protectedMemory || v11->shaderDrawParameters)
            return vk_error(physical_device, VK_ERROR_FEATURE_NOT_PRESENT);
         break;
      }
      default:
         break;
      }
   }

   device = vk_zalloc2(&physical_device->vk.instance->alloc, pAllocator,
                       sizeof(*device), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &borgvk_device_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);
#endif
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vk_cmd_enqueue_device_entrypoints, false);
#define BORGVK_RECORD_CMD(name) dispatch_table.name = vk_cmd_enqueue_##name
   BORGVK_RECORD_CMD(CmdPushConstants);
   BORGVK_RECORD_CMD(CmdBeginRendering);   BORGVK_RECORD_CMD(CmdEndRendering);
   BORGVK_RECORD_CMD(CmdClearAttachments);
   BORGVK_RECORD_CMD(CmdClearColorImage);  BORGVK_RECORD_CMD(CmdClearDepthStencilImage);
   BORGVK_RECORD_CMD(CmdBlitImage);        BORGVK_RECORD_CMD(CmdBlitImage2);
   BORGVK_RECORD_CMD(CmdCopyBuffer);       BORGVK_RECORD_CMD(CmdCopyBuffer2);
   BORGVK_RECORD_CMD(CmdCopyBufferToImage); BORGVK_RECORD_CMD(CmdCopyBufferToImage2);
   BORGVK_RECORD_CMD(CmdCopyImage);        BORGVK_RECORD_CMD(CmdCopyImage2);
   BORGVK_RECORD_CMD(CmdCopyImageToBuffer); BORGVK_RECORD_CMD(CmdCopyImageToBuffer2);
   BORGVK_RECORD_CMD(CmdResolveImage);     BORGVK_RECORD_CMD(CmdResolveImage2);
   BORGVK_RECORD_CMD(CmdFillBuffer);       BORGVK_RECORD_CMD(CmdUpdateBuffer);
   BORGVK_RECORD_CMD(CmdSetEvent);         BORGVK_RECORD_CMD(CmdResetEvent);
   BORGVK_RECORD_CMD(CmdSetEvent2);        BORGVK_RECORD_CMD(CmdResetEvent2);
   BORGVK_RECORD_CMD(CmdBindPipeline);     BORGVK_RECORD_CMD(CmdBindVertexBuffers);
   BORGVK_RECORD_CMD(CmdBindDescriptorSets); BORGVK_RECORD_CMD(CmdDraw);
   BORGVK_RECORD_CMD(CmdDrawIndexed);      BORGVK_RECORD_CMD(CmdBindIndexBuffer);
   BORGVK_RECORD_CMD(CmdDispatch);
   BORGVK_RECORD_CMD(CmdDrawIndirect);     BORGVK_RECORD_CMD(CmdDrawIndexedIndirect);
#undef BORGVK_RECORD_CMD
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vk_common_device_entrypoints, false);

   /* Dispatch used to replay a recorded queue (secondary cmd buffers). */
   vk_device_dispatch_table_from_entrypoints(&device->cmd_dispatch,
                                             &borgvk_device_entrypoints, true);
   if (!device->cmd_dispatch.CmdExecuteCommands)
      device->cmd_dispatch.CmdExecuteCommands = vk_common_CmdExecuteCommands;
   {
      void **slots = (void **)&device->cmd_dispatch;
      for (size_t i = 0; i < sizeof(device->cmd_dispatch) / sizeof(void *); i++)
         if (!slots[i])
            slots[i] = (void *)borgvk_noop_cmd;
   }

   result = vk_device_init(&device->vk, &physical_device->vk,
                           &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&physical_device->vk.instance->alloc, pAllocator, device);
      return vk_error(physical_device, result);
   }

   vk_device_enable_threaded_submit(&device->vk);

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

   /* Non-owning copy — lifetime is managed by the physical device. */
   device->drm_fd = physical_device->drm_fd;

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

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return borgvk_GetInstanceProcAddr(instance, pName);
}
