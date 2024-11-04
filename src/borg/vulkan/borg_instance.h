/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_INSTANCE_H
#define BORG_INSTANCE_H

#include "borg_debug.h"

#include "vk_instance.h"

struct borg_instance {
   struct vk_instance vk;

   enum borg_debug debug_flags;
};

VK_DEFINE_HANDLE_CASTS(borg_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)

#endif
