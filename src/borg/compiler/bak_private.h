/* 
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BAK_PRIVATE_H
#define BAK_PRIVATE_H

#include "bak.h"
#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

struct bak_compiler {
        struct nir_shader_compiler_options nir_options;
};

void bak_optimize_nir(nir_shader *nir, const struct bak_compiler *nak);
void bak_print_instr(const nir_instr*);
void bak_print_defer_instr(const nir_instr*);

#ifdef __cplusplus
}
#endif


#endif
