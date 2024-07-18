/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "bak.h"
#include "borg_cmd_buffer.h"
#include "borg_device.h"
#include "borg_entrypoints.h"
#include "borg_physical_device.h"
#include "borg_shader.h"

#include "vk_shader.h"

static const struct vk_shader_ops borg_shader_ops = {
        //.destroy = nvk_shader_destroy,
        //.serialize = nvk_shader_serialize,
        //.get_executable_properties = nvk_shader_get_executable_properties,
        //.get_executable_statistics = nvk_shader_get_executable_statistics,
        //.get_executable_internal_representations =
        //     nvk_shader_get_executable_internal_representations,
};

static const nir_shader_compiler_options *
borg_get_nir_options(struct vk_physical_device *vk_pdev,
                gl_shader_stage stage,
                UNUSED const struct vk_pipeline_robustness_state *rs)
{
        puts("borg_get_nir_options");
        const struct borg_physical_device *pdev = container_of(vk_pdev, struct borg_physical_device, vk);
        return bak_nir_options(pdev->bak);
}

static struct spirv_to_nir_options
borg_get_spirv_options(struct vk_physical_device *vk_pdev,
                UNUSED gl_shader_stage stage,
                const struct vk_pipeline_robustness_state *rs)
{
        puts("borg_get_spirv_options TODO");
        return (struct spirv_to_nir_options) { /* TODO */ };
}

static VkResult
borg_compile_nir(struct borg_device *dev,
                nir_shader *nir,
                VkShaderCreateFlagsEXT shader_flags,
                const struct vk_pipeline_robustness_state *rs,
                struct borg_shader *shader)
{
        puts("borg_compile_nir");
        struct borg_physical_device *pdev = borg_device_physical(dev);
        shader->bak = bak_compile_shader(nir, pdev->bak);
        //shader->code_ptr = shader->bak->code;
        //shader->code_size = shader->bak->code_size;
        return VK_SUCCESS;
}

static void
borg_shader_destroy(struct vk_device *vk_dev,
                struct vk_shader *vk_shader,
                const VkAllocationCallbacks* pAllocator)
{
        puts("borg_shader_destroy TODO");
        // TODO
}

static void
borg_lower_nir(struct borg_device *dev,
               nir_shader *nir,
               uint32_t set_layout_count,
               struct vk_descriptor_set_layout * const *set_layouts)
{
        puts("borg_lower_nir");

        struct borg_physical_device *pdev = borg_device_physical(dev);

        nir_lower_compute_system_values_options csv_options = {
                .has_base_workgroup_id = true,
        };
        NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

        /* Lower push constants before lower_descriptors */
        NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
                 nir_address_format_32bit_offset);

        /* Lower non-uniform access before lower_descriptors */
        enum nir_lower_non_uniform_access_type lower_non_uniform_access_types =
                nir_lower_non_uniform_ubo_access;

        /* In practice, most shaders do not have non-uniform-qualified accesses
         * thus a cheaper and likely to fail check is run first.
         */
        if (nir_has_non_uniform_access(nir, lower_non_uniform_access_types)) {
                struct nir_lower_non_uniform_access_options opts = {
                        .types = lower_non_uniform_access_types,
                        .callback = NULL,
                };
                NIR_PASS(_, nir, nir_opt_non_uniform_access);
                NIR_PASS(_, nir, nir_lower_non_uniform_access, &opts);
        }

        /* Large constant support assumes cbufs */
        NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

        NIR_PASS(_, nir, borg_nir_lower_descriptors, pdev, set_layout_count, set_layouts);
        //NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            //nir_address_format_64bit_global);
        //NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            //nvk_ssbo_addr_format(pdev, rs->storage_buffers));
        //NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            //nvk_ubo_addr_format(pdev, rs->uniform_buffers));
        //NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            //lower_load_intrinsic, nir_metadata_none, NULL);

        NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
                 nir_address_format_32bit_offset);

        if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
                /* QMD::SHARED_MEMORY_SIZE requires an alignment of 256B so it's safe to
                 * align everything up to 16B so we can write whole vec4s.
                 */
                nir->info.shared_size = align(nir->info.shared_size, 16);
                NIR_PASS(_, nir, nir_zero_initialize_shared_memory,
                         nir->info.shared_size, 16);

                /* We need to call lower_compute_system_values again because
                 * nir_zero_initialize_shared_memory generates load_invocation_id which
                 * has to be lowered to load_invocation_index.
                 */
                NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
        }
}

static VkResult
borg_compile_shader(struct borg_device *dev,
                struct vk_shader_compile_info *info,
                const struct vk_graphics_pipeline_state *state,
                const VkAllocationCallbacks* pAllocator,
                struct vk_shader **shader_out)
{
        struct borg_shader *shader;
        VkResult result;

        puts("borg_compile_shader");

        shader = vk_shader_zalloc(&dev->vk, &borg_shader_ops, info->stage,
                        pAllocator, sizeof(*shader));

        nir_shader *nir = info->nir;

        borg_lower_nir(dev, nir, info->set_layout_count, info->set_layouts);

        result = borg_compile_nir(dev, nir, info->flags, info->robustness, shader);
        ralloc_free(nir);
        if (result != VK_SUCCESS) {
                borg_shader_destroy(&dev->vk, &shader->vk, pAllocator);
                return result;
        }

        *shader_out = &shader->vk;

        return VK_SUCCESS;
}

static VkResult
borg_compile_shaders(struct vk_device *vk_dev,
                uint32_t shader_count,
                struct vk_shader_compile_info *infos,
                const struct vk_graphics_pipeline_state *state,
                const VkAllocationCallbacks* pAllocator,
                struct vk_shader **shaders_out)
{
        puts("borg_compile_shaders");
        struct borg_device *dev = container_of(vk_dev, struct borg_device, vk);

        for (uint32_t i = 0; i < shader_count; i++) {
                VkResult result = borg_compile_shader(dev, &infos[i], state, pAllocator, &shaders_out[i]);
                if (result != VK_SUCCESS) {
                        /* Clean up all the shaders before this point */
                        for (uint32_t j = 0; j < i; j++)
                                borg_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

                        /* Clean up all the NIR after this point */
                        for (uint32_t j = i + 1; j < shader_count; j++)
                                ralloc_free(infos[j].nir);

                        /* Memset the output array */
                        memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

                        return result;
                }
        }
        return VK_SUCCESS;
}

const struct vk_device_shader_ops borg_device_shader_ops = {
        .get_nir_options = borg_get_nir_options,
        .get_spirv_options = borg_get_spirv_options,
        //.preprocess_nir = nvk_preprocess_nir,
        //.hash_graphics_state = nvk_hash_graphics_state,
        .compile = borg_compile_shaders,
        //.deserialize = nvk_deserialize_shader,
        //.cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
        .cmd_bind_shaders = borg_cmd_bind_shaders,
};

