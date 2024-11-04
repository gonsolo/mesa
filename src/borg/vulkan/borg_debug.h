/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEBUG_H
#define BORG_DEBUG_H

enum borg_debug {
   // Skips all drm calls
   BORG_SKIP_DRM = 1ull << 0,
};

#endif

