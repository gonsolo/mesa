/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"
#include "util/format/u_formats.h"
#include "util/log.h"
#include "vk_format.h"
#include "vk_pipeline_layout.h"
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
