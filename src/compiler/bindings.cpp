#include <cstdio>
#include <unistd.h>

#include "pybind11/detail/common.h"
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
wrap_nir_iadd(nir_builder *build, nir_def *src0, nir_def *src1)
{
  assert(build);
  nir_def *result = nir_iadd(build, src0, src1);
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

namespace py = pybind11;

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

    m.def("nir_iadd", &wrap_nir_iadd,
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
}

PYBIND11_MODULE(mesa3d, m) {
    m.doc() = "pybind11 bindings for selected Mesa3D functions";

    register_classes(m);
    register_enums(m);
    register_functions(m);
}
