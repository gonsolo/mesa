/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_instance.h"
#include "borg_entrypoints.h"
#include "borg_physical_device.h"
#include "vulkan/runtime/vk_instance.h"
#include "vulkan/runtime/vk_log.h"
#include "vulkan/util/vk_alloc.h"
#include "vulkan/util/vk_util.h"

static const struct vk_instance_extension_table instance_extensions = {
   .KHR_get_physical_device_properties2   = true,
   .EXT_debug_report                      = true,
   .EXT_debug_utils                       = true,
};

VKAPI_ATTR VkResult VKAPI_CALL
borg_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   uint32_t version_override = vk_get_version_override();
   *pApiVersion = version_override ? version_override :
                  VK_MAKE_VERSION(1, 3, VK_HEADER_VERSION);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                           uint32_t *pPropertyCount,
                                           VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(
      &instance_extensions, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkInstance *pInstance)
{
   puts(__func__);

   struct borg_instance *instance;
   VkResult result;

   if (pAllocator == NULL)
      pAllocator = vk_default_allocator();

   instance = vk_alloc(pAllocator, sizeof(*instance), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &borg_instance_entrypoints, true);

   result = vk_instance_init(&instance->vk, &instance_extensions,
                             &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return result;
   }

   instance->vk.physical_devices.try_create_for_drm =
      borg_create_drm_physical_device;
   instance->vk.physical_devices.destroy = borg_physical_device_destroy;

   *pInstance = borg_instance_to_handle(instance);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_DestroyInstance(VkInstance _instance,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borg_instance, instance, _instance);

   if (!instance)
      return;

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
borg_GetInstanceProcAddr(VkInstance _instance,
                        const char *pName)
{
   VK_FROM_HANDLE(borg_instance, instance, _instance);
   return vk_instance_get_proc_addr(&instance->vk,
                                    &borg_instance_entrypoints,
                                    pName);
}

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance,
                          const char *pName);
PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance,
                          const char *pName)
{
   return borg_GetInstanceProcAddr(instance, pName);
}

