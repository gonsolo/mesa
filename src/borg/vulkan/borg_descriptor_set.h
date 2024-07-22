/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DESCRIPTOR_SET_H
#define BORG_DESCRIPTOR_SET_H

#include "vk_object.h"

struct borg_descriptor_pool {
   struct vk_object_base base;
   uint64_t size;
   uint32_t max_entry_count;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

struct borg_descriptor_pool_entry {

};

struct borg_descriptor_set {
   struct vk_object_base base;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_descriptor_set, base, VkDescriptorSet,
                       VK_OBJECT_TYPE_DESCRIPTOR_SET)

struct borg_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero; /* Must be zero! */
};

#endif
