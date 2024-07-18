/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_SHADER_H
#define BORG_SHADER_H

#include "vk_shader.h"

#include "borg_physical_device.h"

extern const struct vk_device_shader_ops borg_device_shader_ops;

struct borg_shader {
   struct vk_shader vk;
   struct bak_shader_bin *bak;
   const void *code_ptr;
   uint32_t code_size;
};

bool
borg_nir_lower_descriptors(nir_shader *nir,
                           const struct borg_physical_device *pdev,
                           uint32_t set_layout_count,
                           struct vk_descriptor_set_layout * const *set_layouts);
#endif
