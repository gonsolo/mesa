#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

#include "glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"

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

PYBIND11_MODULE(mesabindings, m) {
    m.doc() = "pybind11 bindings for selected Mesa3D functions";

    m.def("glsl_type_singleton_init_or_ref", &glsl_type_singleton_init_or_ref,
      "Initializes or references the GLSL type singleton.");

    m.def("test_gonsolo", &test_gonsolo);

    py::native_enum<gl_shader_stage>(m, "gl_shader_stage", "enum.IntEnum")
      .value("COMPUTE", MESA_SHADER_COMPUTE)
      .export_values()
      .finalize();

    py::class_<nir_shader_compiler_options>(m, "nir_shader_compiler_options")
      .def(py::init<>());
}

