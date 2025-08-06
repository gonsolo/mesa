import mesabindings

def test_python_functions():
    print("Calling functions from Python...")
    try:
        mesabindings.glsl_type_singleton_init_or_ref()
        stage = mesabindings.gl_shader_stage.COMPUTE
        options = mesabindings.nir_shader_compiler_options()
        builder = mesabindings.nir_builder_init_simple_shader(stage, options, "simple")
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
