/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_CMD_BUFFER_H
#define BORG_CMD_BUFFER_H

#include "borg_private.h"
#include "borg_descriptor_set.h"

#include "shader_enums.h"

#include "vk_command_buffer.h"
#include "vk_shader.h"



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

struct borg_root_descriptor_table {

   struct borg_buffer_address sets[BORG_MAX_SETS];

   // TODO
};

#define borg_root_descriptor_offset(member)\
     offsetof(struct borg_root_descriptor_table, member)

#endif
