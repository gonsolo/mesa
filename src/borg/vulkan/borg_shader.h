/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "vk_shader.h"

extern const struct vk_device_shader_ops borg_device_shader_ops;

struct borg_shader {
   struct vk_shader vk;
};
