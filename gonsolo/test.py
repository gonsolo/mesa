import mesabindings

def test_python_functions():
    print("Calling functions from Python...")
    try:
        mesabindings.glsl_type_singleton_init_or_ref()
        stage = mesabindings.gl_shader_stage.COMPUTE
        options = mesabindings.nir_shader_compiler_options()
        builder = mesabindings.nir_builder_init_simple_shader(stage, options, "simple")
        val1 = mesabindings.nir_imm_int(builder, 3);
        print(f"val1 num_components: {val1.num_components}")
        print(f"val1 bit_size: {val1.bit_size}")
        print("Functions called successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_gonsolo_function():
    print("Calling test_gonsolo from Python...")
    try:
        mesabindings.test_gonsolo()
        print("Function called successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_python_functions()
    test_gonsolo_function()
