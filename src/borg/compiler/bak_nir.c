/* 
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "bak_private.h"

#include "nir_builder.h"

#define OPT(nir, pass, ...) ({                             \
     bool this_progress = false;                           \
     NIR_PASS(this_progress, nir, pass, ##__VA_ARGS__);    \
     if (this_progress)                                    \
        progress = true;                                   \
     this_progress;                                        \
})

static void
optimize_nir(nir_shader *nir, const struct bak_compiler *bak)
{
        bool progress;

        puts("optimize_nir");

        do {
                progress = false;

                OPT(nir, nir_shrink_vec_array_vars, nir_var_function_temp);
                OPT(nir, nir_opt_deref);
                if (OPT(nir, nir_opt_memcpy))
                   OPT(nir, nir_split_var_copies);

                OPT(nir, nir_lower_system_values);
                nir_lower_compute_system_values_options lower_csv_options = {
                        .has_base_workgroup_id = nir->info.stage == MESA_SHADER_COMPUTE,
                };
                OPT(nir, nir_lower_compute_system_values, &lower_csv_options);

                OPT(nir, nir_lower_vars_to_ssa);

                OPT(nir, nir_opt_copy_prop_vars);
                OPT(nir, nir_opt_dead_write_vars);
                OPT(nir, nir_opt_combine_stores, nir_var_all);

                OPT(nir, nir_lower_phis_to_scalar, false);
                OPT(nir, nir_lower_frexp);
                OPT(nir, nir_copy_prop);
                OPT(nir, nir_opt_dce);
                OPT(nir, nir_opt_cse);

                OPT(nir, nir_opt_peephole_select, 0, false, false);
                OPT(nir, nir_opt_intrinsics);
                OPT(nir, nir_opt_idiv_const, 32);
                OPT(nir, nir_opt_algebraic);
                OPT(nir, nir_lower_constant_convert_alu_types);
                OPT(nir, nir_opt_constant_folding);

                OPT(nir, nir_opt_dead_cf);
                if (OPT(nir, nir_opt_loop)) {
                   /* If nir_opt_loop makes progress, then we need to clean things up
                    * if we want any hope of nir_opt_if or nir_opt_loop_unroll to make
                    * progress.
                    */
                   OPT(nir, nir_copy_prop);
                   OPT(nir, nir_opt_dce);
                }
                OPT(nir, nir_opt_if, nir_opt_if_optimize_phi_true_false);
                OPT(nir, nir_opt_conditional_discard);
                OPT(nir, nir_opt_remove_phis);
                OPT(nir, nir_opt_gcm, false);
                OPT(nir, nir_opt_undef);
                OPT(nir, nir_lower_pack);
     } while (progress);

     OPT(nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
}

void
bak_optimize_nir(nir_shader *nir, const struct bak_compiler *bak)
{
        optimize_nir(nir, bak);
}

static bool
bak_nir_lower_workgroup_intrin(nir_builder *b, nir_intrinsic_instr *intrin,
                                    void *data)
{
        b->cursor = nir_before_instr(&intrin->instr);
        nir_def *val;
        switch (intrin->intrinsic) {
        case nir_intrinsic_load_base_workgroup_id:
        case nir_intrinsic_load_workgroup_id: {
                unsigned components = 3;
                unsigned bit_size = 32;
                val = nir_imm_zero(b, components, bit_size);
                break;
        }
        default:
                return false;
        }
        nir_def_rewrite_uses(&intrin->def, val);
        return true;
}

static bool
bak_nir_lower_workgroup(nir_shader *nir, const struct bak_compiler *bak)
{
        return nir_shader_intrinsics_pass(nir, bak_nir_lower_workgroup_intrin,
                                          nir_metadata_none,
                                          (void *)bak);
}

void
bak_postprocess_nir(nir_shader *nir, const struct bak_compiler *bak)
{
        puts("bak_postprocess_nir");
        UNUSED bool progress = false;
        bak_optimize_nir(nir, bak);

        OPT(nir, nir_opt_shrink_vectors, true);
        bak_optimize_nir(nir, bak);

        OPT(nir, bak_nir_lower_workgroup, bak);

        OPT(nir, nir_lower_doubles, NULL, bak->nir_options.lower_doubles_options);
        OPT(nir, nir_lower_int64);
        bak_optimize_nir(nir, bak);

        nir_convert_to_lcssa(nir, true, true);
        nir_divergence_analysis(nir);
}

