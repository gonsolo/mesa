import mesa3d
import sys

def test_python_functions():
    print("Calling functions from Python...")
    try:
        mesa3d.glsl_type_singleton_init_or_ref()
        stage = mesa3d.gl_shader_stage.COMPUTE
        options = mesa3d.nir_shader_compiler_options()
        builder = mesa3d.nir_builder_init_simple_shader(stage, options, "simple")
        val1 = mesa3d.nir_imm_int(builder, 3);
        assert val1.num_components == 1, "val1 num_components is not 1"
        assert val1.bit_size == 32, "val1 bit_size is not 32"
        val2 = mesa3d.nir_imm_int(builder, 5);
        assert val2.num_components == 1, "val2 num_components is not 1"
        assert val2.bit_size == 32, "val2 bit_size is not 32"
        val3 = mesa3d.nir_iadd(builder, val1, val2);
        assert val3.num_components == 1, "val3 num_components is not 1"
        assert val3.bit_size == 32, "val3 bit_size is not 32"

        print("Original shader:")
        mesa3d.nir_print_shader(builder.shader, sys.stdout.fileno())

        mesa3d.nir_metadata_require(builder.impl, mesa3d.nir_metadata_block_index | mesa3d.nir_metadata_dominance);
        mesa3d.nir_opt_algebraic(builder.shader);
        mesa3d.nir_opt_constant_folding(builder.shader);
        mesa3d.nir_opt_dce(builder.shader);

        print("Optimized shader:");
        mesa3d.nir_print_shader(builder.shader, sys.stdout.fileno());

        mesa3d.ralloc_free(builder.shader);

        print("Functions called successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_gonsolo_function():
    print("Calling test_gonsolo from Python...")
    try:
        mesa3d.test_gonsolo()
        print("Function called successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    mesa3d.is_device_info_initialized()

    #test_python_functions()
    #test_gonsolo_function()
