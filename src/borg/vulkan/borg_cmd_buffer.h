/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "vk_command_buffer.h"

extern const struct vk_command_buffer_ops borg_cmd_buffer_ops;
 
struct borg_cmd_buffer {
   struct vk_command_buffer vk;
};

VK_DEFINE_HANDLE_CASTS(borg_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

