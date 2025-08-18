#include <cstdio>
#include <unistd.h>

#include "pybind11/detail/common.h"
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

#include "glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nouveau/compiler/nak_private.h"
#include "nouveau/compiler/nak.h"
#include "nouveau/winsys/nouveau_device.h"

#include <xf86drm.h>
#include <stdio.h>
#include <stdbool.h>

// Fix the 'class' keyword conflict
#define class _class
#include <libdrm/nouveau_drm.h>
#undef class

void test_gonsolo() {
      glsl_type_singleton_init_or_ref();
      gl_shader_stage stage = MESA_SHADER_COMPUTE;
      nir_shader_compiler_options options = {};
      nir_builder builder = nir_builder_init_simple_shader(stage, &options, "simple");

      nir_def* val1 = nir_imm_int(&builder, 3);
      assert(val1->num_components == 1);
      assert(val1->bit_size == 32);

      nir_def* val2 = nir_imm_int(&builder, 5);
      nir_def* sum = nir_iadd(&builder, val1, val2);

      (void)sum;

      //printf("--- Original Shader ---\n");
      //nir_print_shader(builder.shader, stdout);
      //printf("\n\n");

      nir_metadata_require(builder.impl, nir_metadata_block_index | nir_metadata_dominance);

      nir_opt_algebraic(builder.shader);
      nir_opt_constant_folding(builder.shader);
      nir_opt_dce(builder.shader);

      //printf("--- Optimized Shader ---\n");
      //nir_print_shader(builder.shader, stdout);

      ralloc_free(builder.shader);
}

namespace py = pybind11;

struct nouveau_ws_device* nouveau_ws_device_new_wrapper(uintptr_t drm_device_address) {
    // Cast the integer memory address back to a pointer to the struct
    struct _drmDevice* drm_device_ptr = reinterpret_cast<struct _drmDevice*>(drm_device_address);

    // Call the original C function with the correctly typed pointer
    return nouveau_ws_device_new(drm_device_ptr);
}

static nir_builder
nir_builder_init_simple_shader_wrapper(gl_shader_stage stage,
                                      const nir_shader_compiler_options *options,
                                      const char *name)
{
    return nir_builder_init_simple_shader(stage, options, "%s", name);
}

nir_def *
wrap_nir_imm_int(nir_builder *build, int x)
{
  assert(build);
  nir_def *result = nir_imm_int(build, x);
  assert(result);
  return result;
}

nir_def *
wrap_nir_imm_float(nir_builder *build, float x)
{
  assert(build);
  nir_def *result = nir_imm_float(build, x);
  assert(result);
  return result;
}

nir_def *
wrap_nir_iadd(nir_builder *build, nir_def *src0, nir_def *src1)
{
  assert(build);
  nir_def *result = nir_iadd(build, src0, src1);
  assert(result);
  return result;
}

nir_def *
wrap_nir_imul(nir_builder *build, nir_def *src0, nir_def *src1)
{
  assert(build);
  nir_def *result = nir_imul(build, src0, src1);
  assert(result);
  return result;
}

void
wrap_nir_metadata_require(nir_function_impl *impl, int required)
{
  nir_metadata_require(impl, static_cast<nir_metadata>(required));
}

void
wrap_nir_print_shader(nir_shader *shader, int fd)
{
    FILE *fp = fdopen(dup(fd), "w");
    if (fp) {
        nir_print_shader(shader, fp);
        fclose(fp);
    }
}


void wrap_is_device_info_initialized()
{
    int drm_fd = drmOpen("nouveau", NULL);
    if (drm_fd < 0) {
        perror("Failed to open DRM device");
        return;
    }
    printf("Successfully opened DRM device with file descriptor: %d\n", drm_fd);

    drmDevice* device;
    if (drmGetDevice(drm_fd, &device) < 0) {
        perror("Failed to get DRM device object");
        drmClose(drm_fd);
        return;
    }
    printf("Successfully retrieved device object.\n");

    nouveau_ws_device *dev = nouveau_ws_device_new(device);
    if (!dev) {
        printf("Failed to create nouveau_ws_device.\n");
        drmFreeDevice(&device);
        drmClose(drm_fd);
        return;
    }
    printf("Successfully retrieved nouveau ws device object.\n");
    printf("Device: %s\n", dev->info.device_name);

    nouveau_ws_device_destroy(dev);
    drmFreeDevice(&device);
    drmClose(drm_fd);
}

py::object wrap_nir_load_ssbo(py::object builder, int num_components, int bit_size, py::object src0, py::object src1) {
    nir_builder* b = py::cast<nir_builder*>(builder);
    nir_def* s0 = py::cast<nir_def*>(src0);
    nir_def* s1 = py::cast<nir_def*>(src1);

    return py::cast(nir_load_ssbo(b, num_components, bit_size, s0, s1));
}

struct lower_ycbcr_state {
     uint32_t set_layout_count;
     struct vk_descriptor_set_layout * const *set_layouts;
};

static bool
lower_load_intrinsic(nir_builder *b, nir_intrinsic_instr *load,
                     UNUSED void *_data)
{
   switch (load->intrinsic) {
   case nir_intrinsic_load_ubo: {
      b->cursor = nir_before_instr(&load->instr);

      nir_def *index = load->src[0].ssa;
      nir_def *offset = load->src[1].ssa;
      const enum gl_access_qualifier access = nir_intrinsic_access(load);
      const uint32_t align_mul = nir_intrinsic_align_mul(load);
      const uint32_t align_offset = nir_intrinsic_align_offset(load);

      nir_def *val;
      if (load->src[0].ssa->num_components == 1) {
         val = nir_ldc_nv(b, load->num_components, load->def.bit_size,
                           index, offset, .access = access,
                           .align_mul = align_mul,
                           .align_offset = align_offset);
      } else if (load->src[0].ssa->num_components == 2) {
         nir_def *handle = nir_pack_64_2x32(b, load->src[0].ssa);
         val = nir_ldcx_nv(b, load->num_components, load->def.bit_size,
                           handle, offset, .access = access,
                           .align_mul = align_mul,
                           .align_offset = align_offset);
      } else {
         unreachable("Invalid UBO index");
      }
      nir_def_rewrite_uses(&load->def, val);
      return true;
   }

   case nir_intrinsic_load_global_constant_offset:
   case nir_intrinsic_load_global_constant_bounded: {
      b->cursor = nir_before_instr(&load->instr);

      nir_def *base_addr = load->src[0].ssa;
      nir_def *offset = load->src[1].ssa;

      nir_def *zero = NULL;
      if (load->intrinsic == nir_intrinsic_load_global_constant_bounded) {
         nir_def *bound = load->src[2].ssa;

         unsigned bit_size = load->def.bit_size;
         assert(bit_size >= 8 && bit_size % 8 == 0);
         unsigned byte_size = bit_size / 8;

         zero = nir_imm_zero(b, load->num_components, bit_size);

         unsigned load_size = byte_size * load->num_components;

         nir_def *sat_offset =
            nir_umin(b, offset, nir_imm_int(b, UINT32_MAX - (load_size - 1)));
         nir_def *in_bounds =
            nir_ilt(b, nir_iadd_imm(b, sat_offset, load_size - 1), bound);

         nir_push_if(b, in_bounds);
      }

      nir_def *val =
         nir_build_load_global_constant(b, load->def.num_components,
                                        load->def.bit_size,
                                        nir_iadd(b, base_addr, nir_u2u64(b, offset)),
                                        .align_mul = nir_intrinsic_align_mul(load),
                                        .align_offset = nir_intrinsic_align_offset(load));

      if (load->intrinsic == nir_intrinsic_load_global_constant_bounded) {
         nir_pop_if(b, NULL);
         val = nir_if_phi(b, val, zero);
      }

      nir_def_rewrite_uses(&load->def, val);
      return true;
   }

   default:
      return false;
   }
}

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

static void tinygrad_lower_nir(nir_shader *nir)
{
   //if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
   //   NIR_PASS(_, nir, nir_lower_patch_vertices,
   //            nir->info.tess.tcs_vertices_out, NULL);
   //}

   //const struct lower_ycbcr_state ycbcr_state = {
   //   .set_layout_count = set_layout_count,
   //   .set_layouts = set_layouts,
   //};
   //NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex,
   //         lookup_ycbcr_conversion, &ycbcr_state);

   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   //struct nvk_cbuf_map *cbuf_map = NULL;
   //if (!(pdev->debug_flags & NVK_DEBUG_NO_CBUF)) {
   //   cbuf_map = cbuf_map_out;

   //   /* Large constant support assumes cbufs */
   //   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);
   //} else {
   //   *cbuf_map_out = (struct nvk_cbuf_map) {
   //      .cbuf_count = 1,
   //      .cbufs = {
   //         { .type = NVK_CBUF_TYPE_ROOT_DESC },
   //      }
   //   };
   //}

   nir_opt_access_options opt_access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &opt_access_options);

   /* On Kepler, we have to lower images to addresses */
   //if (pdev->info.cls_eng3d < MAXWELL_A)
   //   NIR_PASS(_, nir, nak_nir_lower_image_addrs, pdev->nak);

   //NIR_PASS(_, nir, nvk_nir_lower_descriptors, pdev, shader_flags, rs,
   //         set_layout_count, set_layouts, cbuf_map);

   //if (nvk_use_bindless_cbuf(&pdev->info)) {
   //   /* On Turing+ where we have bindless cbufs, we use ACCESS_NON_UNIFORM to
   //    * determine whether or not it's safe to assume a uniform handle so we
   //    * want to optimize it away whenever possible.
   //    */
   //   if (nir_has_non_uniform_access(nir, nir_lower_non_uniform_ubo_access))
   //      NIR_PASS(_, nir, nir_opt_non_uniform_access);
   //}

   //if (pdev->info.cls_eng3d < TURING_A) {
   //   /* NOTE: This does nothing for images on Kepler since those are lowered
   //    * to suldga/sustga before we get here.  That's fine, though, because
   //    * our nil_su_info fetches and calculations work fine with non-uniform
   //    * descriptors.
   //    */
   //   struct nir_lower_non_uniform_access_options opts = {
   //      .types = nir_lower_non_uniform_texture_access |
   //               nir_lower_non_uniform_image_access,
   //      .callback = NULL,
   //   };
   //   /* In practice, most shaders do not have non-uniform-qualified accesses
   //    * thus a cheaper and likely to fail check is run first.
   //    */
   //   if (nir_has_non_uniform_access(nir, opts.types)) {
   //      NIR_PASS(_, nir, nir_opt_non_uniform_access);
   //      NIR_PASS(_, nir, nir_lower_non_uniform_access, &opts);
   //   }
   //}

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            lower_load_intrinsic, nir_metadata_none, NULL);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_mem_shared, shared_var_info);
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


void register_classes(py::module &m) {
    py::class_<nir_function_impl>(m, "nir_function_impl")
        .def_readonly("body", &nir_function_impl::body);

    py::class_<nir_shader_compiler_options>(m, "nir_shader_compiler_options")
        .def(py::init<>());

    py::class_<nir_builder>(m, "nir_builder")
        .def_readwrite("cursor", &nir_builder::cursor)
        .def_readwrite("impl", &nir_builder::impl)
        .def_property_readonly("shader", [](nir_builder *builder) {
            return builder->shader;
        },
        py::return_value_policy::reference);

    py::class_<nir_def>(m, "nir_def")
        .def_readonly("bit_size", &nir_def::bit_size)
        .def_readonly("num_components", &nir_def::num_components);

    py::class_<nir_shader>(m, "nir_shader")
        .def(py::init<>());

    py::class_<glsl_type>(m, "glsl_type")
       .def_property_readonly("name",
           [](const glsl_type &self) {
               return glsl_get_type_name(&self);
           })
       .def_property_readonly("array",
           [](const glsl_type &self) {
               return self.fields.array;
           },
           py::return_value_policy::reference_internal)
       .def_property_readonly("structure",
           [](const glsl_type &self) {
               return self.fields.structure;
           },
           py::return_value_policy::reference_internal)
       .def("is_array", &glsl_type_is_array)
       .def("is_struct", &glsl_type_is_struct);

    py::class_<nir_variable::nir_variable_data>(m, "nir_variable_data")
        .def_readwrite("binding", &nir_variable::nir_variable_data::binding)
        .def_property("explicit_binding",
            [](const nir_variable::nir_variable_data& self) { return self.explicit_binding; },
            [](nir_variable::nir_variable_data& self, bool value) { self.explicit_binding = value; });

    py::class_<nir_variable, std::unique_ptr<nir_variable, py::nodelete>>(m, "nir_variable")
        .def_readwrite("data", &nir_variable::data)
        .def_readwrite("type", &nir_variable::type);

    py::class_<nir_deref_instr>(m, "nir_deref_instr")
        .def(py::init<>());

    py::class_<nv_device_info> nv_device_info_class(m, "nv_device_info");

    py::class_<nouveau_ws_device>(m, "nouveau_ws_device")
        .def_readonly("info", &nouveau_ws_device::info);

    py::class_<nak_compiler> nv_compiler_class(m, "nak_compiler");

    py::class_<nak_fs_key> nak_fs_key_class(m, "nak_fs_key");

    py::class_<nak_shader_bin> nak_shader_bin_class(m, "nak_shader_bin");
}

void register_enums(py::module &m) {
    py::native_enum<gl_shader_stage>(m, "gl_shader_stage", "enum.IntEnum")
        .value("COMPUTE", MESA_SHADER_COMPUTE)
        .export_values()
        .finalize();

    py::native_enum<nir_metadata>(m, "nir_metadata", "enum.IntEnum")
        .value("nir_metadata_block_index", nir_metadata_block_index)
        .value("nir_metadata_dominance", nir_metadata_dominance)
        .export_values()
        .finalize();

    py::enum_<nir_variable_mode>(m, "nir_variable_mode")
        .value("nir_var_system_value", nir_var_system_value)
        .value("nir_var_uniform", nir_var_uniform)
        .value("nir_var_shader_in", nir_var_shader_in)
        .value("nir_var_shader_out", nir_var_shader_out)
        // ... 
        .value("nir_var_mem_ssbo", nir_var_mem_ssbo)
        .export_values();

    py::enum_<nir_address_format>(m, "nir_address_format")
        .value("nir_address_format_64bit_global", nir_address_format_64bit_global)
        .export_values();
}

void register_functions(py::module &m) {
    m.def("glsl_type_singleton_init_or_ref", &glsl_type_singleton_init_or_ref,
        "Initializes or references the GLSL type singleton.");

    m.def("test_gonsolo", &test_gonsolo);

    m.def("nir_builder_init_simple_shader", &nir_builder_init_simple_shader_wrapper,
        py::arg("stage"),
        py::arg("options"),
        py::arg("name"));

    m.def("nir_imm_int", &wrap_nir_imm_int,
        "Wrapper for nir_imm_int",
        py::arg("builder"),
        py::arg("x"),
        py::return_value_policy::reference,
        py::keep_alive<1, 0>());

    m.def("nir_imm_float", &wrap_nir_imm_float,
        "Wrapper for nir_imm_float",
        py::arg("builder"),
        py::arg("x"),
        py::return_value_policy::reference,
        py::keep_alive<1, 0>());

    m.def("nir_iadd", &wrap_nir_iadd,
        "Wrapper for nir_iadd",
        py::arg("builder"),
        py::arg("src0"),
        py::arg("src1"),
        py::return_value_policy::reference,
        py::keep_alive<1, 0>());

    m.def("nir_imul", &wrap_nir_imul,
        "Wrapper for nir_iadd",
        py::arg("builder"),
        py::arg("src0"),
        py::arg("src1"),
        py::return_value_policy::reference,
        py::keep_alive<1, 0>());

    m.def("nir_metadata_require", &wrap_nir_metadata_require,
        py::arg("impl"),
        py::arg("required"),
        py::return_value_policy::reference);

    m.def("nir_opt_algebraic", &nir_opt_algebraic,
        py::arg("shader"));

    m.def("nir_opt_constant_folding", &nir_opt_constant_folding,
        py::arg("shader"));

    m.def("nir_opt_dce", &nir_opt_dce,
        py::arg("shader"));

    m.def("ralloc_free", &ralloc_free,
        py::arg("shader"));

    m.def("nir_print_shader", &wrap_nir_print_shader,
        py::arg("shader"),
        py::arg("fp"));

    m.def("glsl_int_type", &glsl_int_type,
          py::return_value_policy::reference);
    m.def("glsl_float_type", &glsl_float_type,
        py::return_value_policy::reference);

    m.def("glsl_array_type", &glsl_array_type,
        py::arg("element"),
        py::arg("array_size"),
        py::arg("explicit_stride"),
        py::return_value_policy::reference);

    m.def("nir_variable_create", &nir_variable_create,
        py::arg("shader"),
        py::arg("mode"),
        py::arg("type"),
        py::arg("name"),
        py::return_value_policy::reference);

    m.def("glsl_get_explicit_size", &glsl_get_explicit_size,
        py::arg("type"),
        py::arg("align_to_stride"));

    m.def("nir_imul_imm", &nir_imul_imm,
        py::arg("builder"),
        py::arg("x"),
        py::arg("y"),
        py::return_value_policy::reference);

    m.def("glsl_get_vector_elements", &glsl_get_vector_elements,
        py::arg("type"));

    m.def("glsl_get_bit_size", &glsl_get_bit_size,
        py::arg("type"));

    m.def("nir_load_ssbo", &wrap_nir_load_ssbo, "A C++ wrapper for the nir_load_ssbo macro.",
        py::arg("builder"),
        py::arg("num_components"),
        py::arg("bit_size"),
        py::arg("src0"),
        py::arg("src1"));

    m.def("nir_load_var", &nir_load_var,
        py::arg("builder"),
        py::arg("var"),
        py::return_value_policy::reference);

    m.def("nir_build_deref_var", &nir_build_deref_var,
        py::arg("builder"),
        py::arg("var"),
        py::return_value_policy::reference);

    m.def("nir_build_deref_array", &nir_build_deref_array,
        py::arg("builder"),
        py::arg("parent"),
        py::arg("index"),
        py::return_value_policy::reference);

    m.def("nir_load_deref", &nir_load_deref,
        py::arg("builder"),
        py::arg("deref"),
        py::return_value_policy::reference);

    m.def("nir_store_deref", &nir_store_deref,
        py::arg("builder"),
        py::arg("deref"),
        py::arg("value"),
        py::arg("writemask"));

    m.def("nir_load_local_invocation_id", &_nir_build_load_local_invocation_id,
        py::arg("builder"),
        py::return_value_policy::reference);

    m.def("nir_channel", &nir_channel,
        py::arg("builder"),
        py::arg("def"),
        py::arg("c"),
        py::return_value_policy::reference);

    m.def("nir_load_global_invocation_id", &_nir_build_load_global_invocation_id,
        py::arg("builder"),
        py::arg("bit_size"),
        py::return_value_policy::reference);

    m.def("nir_ine_imm", &nir_ine_imm,
        py::arg("builder"),
        py::arg("src1"),
        py::arg("src2"),
        py::return_value_policy::reference);

    m.def("nir_bcsel", &nir_bcsel,
        py::arg("builder"),
        py::arg("src0"),
        py::arg("src1"),
        py::arg("src2"),
        py::return_value_policy::reference);

    m.def("nir_ilt", &nir_ilt,
        py::arg("builder"),
        py::arg("src0"),
        py::arg("src1"),
        py::return_value_policy::reference);

    m.def("is_device_info_initialized", wrap_is_device_info_initialized);

    m.def("nouveau_ws_device_new", &nouveau_ws_device_new_wrapper,
      py::arg("drm_device"),
      py::return_value_policy::reference);

    m.def("nak_compiler_create", &nak_compiler_create,
      py::arg("dev"),
      py::return_value_policy::reference);

    m.def("nak_compile_shader", &nak_compile_shader,
      py::arg("nir"),
      py::arg("dump_asm"),
      py::arg("nak"),
      py::arg("robust2_modes"),
      py::arg("fs_key"),
      py::return_value_policy::reference);

    m.def("nir_lower_vars_to_ssa", &nir_lower_vars_to_ssa,
      py::arg("shader"));

    m.def("nak_optimize_nir", &nak_optimize_nir,
      py::arg("nir"),
      py::arg("nak"));

    m.def("nak_preprocess_nir", &nak_preprocess_nir,
      py::arg("nir"),
      py::arg("nak"));

    m.def("nak_postprocess_nir", &nak_postprocess_nir,
      py::arg("nir"),
      py::arg("nak"),
      py::arg("robust2_modes"),
      py::arg("fs_key"));

    m.def("nir_lower_explicit_io", &nir_lower_explicit_io,
      py::arg("shader"),
      py::arg("modes"),
      py::arg("addr_format"));

     m.def("nir_build_deref_ptr_as_array", &nir_build_deref_ptr_as_array,
          "Builds a nir_deref_instr that indexes a pointer as an array.",
          py::arg("builder"),
          py::arg("parent"),
          py::arg("index"),
          py::return_value_policy::reference);

    m.def("tinygrad_lower_nir", &tinygrad_lower_nir,
      py::arg("nir"));
}

PYBIND11_MODULE(mesa3d, m) {
    m.doc() = "pybind11 bindings for selected Mesa3D functions";

    register_classes(m);
    register_enums(m);
    register_functions(m);
}
