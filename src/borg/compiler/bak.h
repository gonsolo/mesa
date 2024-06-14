/* 
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BAK_H
#define BAK_H

#include "nir.h"

struct bak_shader_bin {
        const void *code;
        uint32_t code_size;
};

struct bak_shader_bin *
bak_compile_shader(nir_shader *nir);

#endif
