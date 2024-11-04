/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_DEBUG_H
#define BORG_DEBUG_H

enum borg_debug {
   // Skips all drm calls.
   // This can be handy when for example working on the compiler; there is no need for a RISC-V
   // emulator, compilation and debugging can be done on an x86 machine.
   BORG_SKIP_DRM = 1ull << 0,
};

#endif

