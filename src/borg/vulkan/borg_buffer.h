/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_BUFFER_H
#define BORG_BUFFER_H

#include "vulkan/runtime/vk_buffer.h"

struct borg_buffer {
	struct vk_buffer vk;
        uint64_t addr;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)

#endif
