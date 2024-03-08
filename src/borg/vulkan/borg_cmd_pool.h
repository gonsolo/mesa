/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_CMD_POOL_H
#define BORG_CMD_POOL_H

#include "borg_entrypoints.h"

#include "vk_command_pool.h"

struct borg_cmd_pool {
   struct vk_command_pool vk;
};   

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_cmd_pool, vk.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)

static inline struct borg_device *
borg_cmd_pool_device(struct borg_cmd_pool *pool)
{
   return (struct borg_device *)pool->vk.base.device;
}

#endif
