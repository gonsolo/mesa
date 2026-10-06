/* SPDX-License-Identifier: MIT */
/* Stand-ins for the C++ parts of src/util in a build with -Dutil-without-cpp=true. */
#include <stdint.h>
#include <stdlib.h>

#include "texcompress_astc.h"
#include "texcompress_astc_luts_wrap.h"
#include "u_qsort.h"

void
util_tls_qsort_r(void *base, size_t nmemb, size_t size,
                 int (*compar)(const void *, const void *, void *), void *arg)
{
   (void)base; (void)nmemb; (void)size; (void)compar; (void)arg;
   abort();
}

void
_mesa_init_astc_decoder_luts(astc_decoder_lut_holder *holder)
{
   (void)holder;
}

void *
_mesa_get_astc_decoder_partition_table(uint32_t block_width, uint32_t block_height,
                                       unsigned *lut_width, unsigned *lut_height)
{
   (void)block_width; (void)block_height; (void)lut_width; (void)lut_height;
   return NULL;
}

void
_mesa_unpack_astc_2d_ldr(uint8_t *dst_row, unsigned dst_stride, const uint8_t *src_row,
                         unsigned src_stride, unsigned src_width, unsigned src_height,
                         enum pipe_format format)
{
   (void)dst_row; (void)dst_stride; (void)src_row; (void)src_stride;
   (void)src_width; (void)src_height; (void)format;
   abort();
}
