/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_PHYSICAL_DEVICE_H
#define BORG_PHYSICAL_DEVICE_H

#include "vulkan/runtime/vk_physical_device.h"

struct borg_physical_device {
   struct vk_physical_device vk;

   /* Driver-specific stuff */
};

struct _drmDevice;

VkResult borg_create_drm_physical_device(struct vk_instance *vk_instance,
                                          struct _drmDevice *drm_device,
                                          struct vk_physical_device **pdev_out);

void borg_physical_device_destroy(struct vk_physical_device *vk_device);

VK_DEFINE_HANDLE_CASTS(borg_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)


#endif
