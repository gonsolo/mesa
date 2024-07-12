/* 
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BAK_H
#define BAK_H

#include "nir.h"

struct bak_compiler;

struct bak_shader_bin {
        const void *code;
        uint32_t code_size;
};

struct bak_shader_bin *
bak_compile_shader(nir_shader *nir, const struct bak_compiler *bak);

const struct nir_shader_compiler_options *
bak_nir_options(const struct bak_compiler *bak);

struct bak_compiler *bak_compiler_create();

void bak_postprocess_nir(nir_shader *nir, const struct bak_compiler *nak);

#endif
