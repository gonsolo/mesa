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

#ifdef BORGVK_NIR
/* The firmware headers have their own FP16 constants; NIR brings Mesa's. */
#undef FP16_ONE
/* Mesa's NIR crate only knows the name; here the struct, under its own name. */
#define nir_builder borgvk_nir_builder
#include "nir_builder.h"
#include "borg_nir_passes.h"
#include "vk_pipeline.h"
#endif
