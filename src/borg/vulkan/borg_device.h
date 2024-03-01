/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEVICE_H
#define BORG_DEVICE_H

#include "vk_device.h"

struct borg_device {
	struct vk_device vk;
	struct borg_physical_device* pdev;
};

VK_DEFINE_HANDLE_CASTS(borg_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

#endif
