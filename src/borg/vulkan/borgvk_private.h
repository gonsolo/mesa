/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * borgvk — a native Mesa Vulkan ICD for the Borg GPU (ULX3S FPGA), driven over
 * serial. Modeled on the v3dv (Broadcom) driver. See
 * ~/.claude/plans/atomic-questing-stream.md for the overall design.
 */
#ifndef BORGVK_PRIVATE_H
#define BORGVK_PRIVATE_H

#include "vk_instance.h"
#include "vk_physical_device.h"

#include "borgvk_entrypoints.h"

#ifdef __cplusplus
extern "C" {
#endif

struct borgvk_instance {
   struct vk_instance vk;
};

struct borgvk_physical_device {
   struct vk_physical_device vk;

   /* Memory layout reported to the app. Borg renders on the FPGA; from the
    * host's point of view all "device" memory is plain host-visible RAM that we
    * stage and ship over serial, so we expose a single host-visible coherent
    * heap. */
   VkPhysicalDeviceMemoryProperties memory;
};

VK_DEFINE_HANDLE_CASTS(borgvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(borgvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

#ifdef __cplusplus
}
#endif

#endif /* BORGVK_PRIVATE_H */
