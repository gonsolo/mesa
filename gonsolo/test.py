import mesabindings

def test_glsl_type_singleton():
    """
    Tests the glsl_type_singleton_init_or_ref binding.
    """
    print("Calling glsl_type_singleton_init_or_ref from Python...")
    try:
        mesabindings.glsl_type_singleton_init_or_ref()
        print("Function called successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_gonsolo_function():
    """
    Tests test_gonsolo
    """
    print("Calling test_gonsolo from Python...")
    try:
        mesabindings.test_gonsolo()
        print("Function called successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_glsl_type_singleton()
    test_gonsolo_function()
