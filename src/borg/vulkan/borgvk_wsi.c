/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#ifdef BORGVK_USE_WSI_PLATFORM

#include "vk_instance.h"
#include "vulkan/vk_icd.h"

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
borgvk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(borgvk_physical_device, pdevice, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(pdevice->vk.instance, pName);
}

VkResult
borgvk_wsi_init(struct borgvk_physical_device *physical_device)
{
   VkResult result;

   result = wsi_device_init(&physical_device->wsi_device,
                            borgvk_physical_device_to_handle(physical_device),
                            borgvk_wsi_proc_addr,
                            &physical_device->vk.instance->alloc,
                            -1, NULL,
                            &(struct wsi_device_options){ .sw_device = true });
   if (result != VK_SUCCESS)
      return result;

   /* Route all swapchains through the headless backend — no display blitting. */
   physical_device->wsi_device.force_headless_swapchain = true;
   physical_device->vk.wsi_device = &physical_device->wsi_device;

   return VK_SUCCESS;
}

void
borgvk_wsi_finish(struct borgvk_physical_device *physical_device)
{
   physical_device->vk.wsi_device = NULL;
   wsi_device_finish(&physical_device->wsi_device,
                     &physical_device->vk.instance->alloc);
}

static VkResult
make_headless_surface(VkInstance _instance,
                      const VkAllocationCallbacks *pAllocator,
                      VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   VkIcdSurfaceHeadless *surface =
      vk_alloc2(&instance->alloc, pAllocator, sizeof(*surface), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!surface)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   surface->base.platform = VK_ICD_WSI_PLATFORM_HEADLESS;
   *pSurface = VkIcdSurfaceBase_to_handle(&surface->base);
   return VK_SUCCESS;
}

#ifdef VK_USE_PLATFORM_XLIB_KHR
VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateXlibSurfaceKHR(VkInstance instance,
                             const VkXlibSurfaceCreateInfoKHR *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator,
                             VkSurfaceKHR *pSurface)
{
   return make_headless_surface(instance, pAllocator, pSurface);
}
#endif

#ifdef VK_USE_PLATFORM_XCB_KHR
VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateXcbSurfaceKHR(VkInstance instance,
                            const VkXcbSurfaceCreateInfoKHR *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkSurfaceKHR *pSurface)
{
   return make_headless_surface(instance, pAllocator, pSurface);
}
#endif

#endif /* BORGVK_USE_WSI_PLATFORM */
