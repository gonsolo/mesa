/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * WSI (window-system integration) for borgvk.  borgvk's real output is the
 * FPGA's HDMI monitor, reached over serial at submit time.  WSI only needs to
 * function (surfaces, swapchains, acquire/present all succeed) so an unmodified
 * app (e.g. Vulkan-Tools/cube.c) runs its render loop and submits each frame.
 *
 * No host window is created.  borgvk overrides every platform surface-creation
 * entrypoint to silently return a headless surface (VK_ICD_WSI_PLATFORM_HEADLESS)
 * regardless of what WSI the app requests.  Mesa's headless WSI backend (always
 * initialized by wsi_device_init) then handles swapchain/acquire/present without
 * opening any display connection.  The app's window-system call succeeds, its
 * extension check passes, and nothing appears on the workstation screen.
 *
 * wsi_device_init + the wsi_*_entrypoints still provide GetSurfaceCapabilities,
 * GetSurfaceFormats, etc.; only surface *creation* is overridden here.
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

/* ── Surface-creation overrides ─────────────────────────────────────────────
 * These are picked up automatically by borgvk_instance_entrypoints (which is
 * loaded with priority=true before wsi_instance_entrypoints in CreateInstance).
 * Each function ignores its platform-specific parameters and returns a minimal
 * headless surface so no window-system connection is ever opened. */

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

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateWaylandSurfaceKHR(VkInstance instance,
                                const VkWaylandSurfaceCreateInfoKHR *pCreateInfo,
                                const VkAllocationCallbacks *pAllocator,
                                VkSurfaceKHR *pSurface)
{
   return make_headless_surface(instance, pAllocator, pSurface);
}
#endif

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
