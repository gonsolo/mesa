#include <pybind11/pybind11.h>

// Include the header where glsl_type_singleton_init_or_ref is declared
// Adjust this path based on your Mesa3D source tree
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

      // Um die Warnung "Variable wird nicht verwendet" zu beheben
      (void)sum;

      printf("--- Original Shader ---\n");
      nir_print_shader(builder.shader, stdout);
      printf("\n\n");

      // Korrekter Aufruf: builder.impl statt builder.shader
      nir_metadata_require(builder.impl, nir_metadata_block_index | nir_metadata_dominance);

        // Rufen Sie jeden Pass einzeln auf
      nir_opt_algebraic(builder.shader);
      nir_opt_constant_folding(builder.shader);
      nir_opt_dce(builder.shader);

      printf("--- Optimized Shader ---\n");
      nir_print_shader(builder.shader, stdout);

      ralloc_free(builder.shader);
}
namespace py = pybind11;

PYBIND11_MODULE(mesabindings, m) {
    m.doc() = "pybind11 bindings for selected Mesa3D functions"; // Optional module docstring

    m.def("glsl_type_singleton_init_or_ref", &glsl_type_singleton_init_or_ref,
          "Initializes or references the GLSL type singleton.");

    m.def("test_gonsolo", &test_gonsolo);

}

