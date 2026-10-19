/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borgc_capture.h"

#include "nir_serialize.h"
#include "util/blob.h"
#include "util/mesa-blake3.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void
borgc_capture(const struct nir_shader *nir, uint32_t kind, uint32_t vfetch, int32_t out_select)
{
   const char *dir = getenv("BORGC_REPLAY_DIR");
   if (!dir || !nir)
      return;
   struct blob b;
   blob_init(&b);
   const uint32_t head[] = { BORGC_CAPTURE_MAGIC, kind, vfetch, (uint32_t)out_select };
   blob_write_bytes(&b, head, sizeof head);
   nir_serialize(&b, nir, false);
   if (!b.out_of_memory) {
      blake3_hash hash;
      char hex[BLAKE3_HEX_LEN], path[4096], tmp[4096 + 32];
      _mesa_blake3_compute(b.data, b.size, hash);
      _mesa_blake3_format(hex, hash);
      snprintf(path, sizeof path, "%s/%s.nir", dir, hex);
      snprintf(tmp, sizeof tmp, "%s.%d", path, (int)getpid());
      FILE *f = access(path, F_OK) ? fopen(tmp, "wb") : NULL;
      if (f) {
         bool ok = fwrite(b.data, 1, b.size, f) == b.size;
         ok &= fclose(f) == 0;
         if (!ok || rename(tmp, path))
            unlink(tmp);
      }
   }
   blob_finish(&b);
}
