/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_entrypoints.h"

#include "vk_descriptor_set_layout.h"

struct borg_descriptor_set_binding_layout {
   VkDescriptorType type;
   VkDescriptorBindingFlags flags;
   uint32_t array_size;
   uint8_t dynamic_buffer_index;
   struct borg_sampler **immutable_samplers;
   uint32_t offset;
   uint8_t stride;
};

struct borg_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;

   uint32_t binding_count;

   struct borg_descriptor_set_binding_layout binding[0];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_descriptor_set_layout, vk.base,
                                 VkDescriptorSetLayout,
                                 VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)

