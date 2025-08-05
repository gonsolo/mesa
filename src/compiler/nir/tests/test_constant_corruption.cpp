#include "compiler/nir/tests/nir_test.h"

class nir_constant_corruption_test : public nir_test {
public:
   // Explicitly define a constructor that calls the base class constructor.
   // This is what the compiler expects.
   nir_constant_corruption_test()
      : nir_test("nir_constant_corruption_test")
   {}
};

TEST_F(nir_constant_corruption_test, constant_corruption)
{
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "test");
   
   nir_def* val1 = nir_imm_int(&builder, 3);
   
   ASSERT_EQ(val1->num_components, 1);
   ASSERT_EQ(val1->bit_size, 32);

   ralloc_free(builder.shader);
}
