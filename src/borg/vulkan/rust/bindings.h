/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"
#include "util/format/u_formats.h"
#include "util/log.h"
#include "vk_format.h"
#include "vk_pipeline_layout.h"
#include "vk_sampler.h"
#include "vk_buffer.h"
#include "vk_buffer_view.h"
#include "vk_device_memory.h"
#include "vk_sync.h"
#include "vk_queue.h"
#include "vk_cmd_queue.h"
#include "vk_instance.h"
#include "wsi_common.h"
#include "vulkan/vk_icd.h"
#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_shader_module.h"
#include "vk_descriptor_set_layout.h"
#include "util/list.h"
#include "util/os_time.h"
#include "drm-uapi/borg_drm.h"
#include "borgvk_drm_compat.h"
#include "borgc_flags.h"

#include "software/borg/borg_core.h"
#include "software/borg/borg_hw.h"
#include "software/borg/borg_layout.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <termios.h>

/* The firmware headers have their own FP16 constants; NIR brings Mesa's. */
#undef FP16_ONE
/* Mesa's NIR crate only knows these names; here the structs, under their own names. */
#define nir_builder borgvk_nir_builder
#define nir_lower_blend_channel borgvk_blend_channel
#define nir_lower_blend_rt borgvk_blend_rt
#define nir_lower_blend_options borgvk_blend_options
#include "borg_nir_passes.h"
#include "vk_blend.h"
#include "vk_render_pass.h"

#ifdef BORGVK_NIR
#include "nir_builder.h"
#include "vk_pipeline.h"
#endif
