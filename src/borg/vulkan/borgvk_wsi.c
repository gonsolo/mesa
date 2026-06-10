/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * WSI (window-system integration) for borgvk, using Mesa's common WSI with the
 * software (CPU) path — exactly like lavapipe. borgvk has no host GPU allocator;
 * the real output is the FPGA's HDMI, reached over serial at submit time. WSI
 * only needs to *function* (surfaces, swapchains, acquire/present all succeed)
 * so an unmodified app (e.g. Vulkan-Tools/cube.c) runs its render loop and
 * submits each frame. The host X window that WSI blits into is irrelevant.
 *
 * All surface/swapchain entrypoints come for free from the runtime's wsi_common
 * (merged into the dispatch tables in borgvk_device.c); this file only stands up
 * the wsi_device on the physical device.
 */
#include "borgvk_private.h"

#ifdef BORGVK_USE_WSI_PLATFORM

#include "vk_instance.h"

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

   /* Software path: linear images we can mmap and hand to xcb_put_image. */
   physical_device->wsi_device.wants_linear = true;
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

#endif /* BORGVK_USE_WSI_PLATFORM */
