/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_entrypoints.h"

#include "borg_shader.h"

#include "vk_shader.h"

static const nir_shader_compiler_options *
borg_get_nir_options(struct vk_physical_device *vk_pdev,
                     gl_shader_stage stage,
                     UNUSED const struct vk_pipeline_robustness_state *rs)
{
   // TODO
   return NULL;
}

const struct vk_device_shader_ops borg_device_shader_ops = {
   .get_nir_options = borg_get_nir_options,
   //.get_spirv_options = nvk_get_spirv_options,
   //.preprocess_nir = nvk_preprocess_nir,
   //.hash_graphics_state = nvk_hash_graphics_state,
   //.compile = nvk_compile_shaders,
   //.deserialize = nvk_deserialize_shader,
   //.cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   //.cmd_bind_shaders = nvk_cmd_bind_shaders,
};

