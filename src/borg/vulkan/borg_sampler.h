/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_SAMPLER_H
#define BORG_SAMPLER_H

#include "vk_object.h"
#include "vk_sampler.h"

struct borg_sampler {
   struct vk_sampler vk;
};     

VK_DEFINE_NONDISP_HANDLE_CASTS(borg_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)

#endif
