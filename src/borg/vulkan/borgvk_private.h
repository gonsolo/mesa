/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BORGVK_PRIVATE_H
#define BORGVK_PRIVATE_H

#include "borgc_flags.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_device_memory.h"
#include "vk_buffer.h"
#include "vk_buffer_view.h"
#include "vk_image.h"
#include "vk_sampler.h"
#include "vk_command_buffer.h"
#include "vk_descriptor_set_layout.h"

#include "wsi_common.h"

#include "borgvk_entrypoints.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BORGVK_USE_WSI_PLATFORM   /* Mesa's headless WSI backend needs no platform */



#ifdef __cplusplus
}
#endif



#include "borgvk_rust.h"

#endif /* BORGVK_PRIVATE_H */
