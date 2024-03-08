/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_CMD_BUFFER_H
#define BORG_CMD_BUFFER_H

#include "vk_command_buffer.h"
#include "vk_shader.h"

#include "shader_enums.h"

extern const struct vk_command_buffer_ops borg_cmd_buffer_ops;
 
struct borg_cmd_buffer {
   struct vk_command_buffer vk;
};

VK_DEFINE_HANDLE_CASTS(borg_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

void borg_cmd_bind_shaders(struct vk_command_buffer *vk_cmd,
                           uint32_t stage_count,
                           const gl_shader_stage *stages,
                           struct vk_shader ** const shaders);

#endif
