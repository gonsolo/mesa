/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * borgc-replay IN OUT: recompiles every IN/<hash>.nir that borgc_capture wrote and stores all
 * of borgc's results in OUT/<hash>.out, with the driver's buffer sizes. Comparing two OUT
 * directories checks a compiler change byte for byte without running the CTS.
 */
#include "borg_nir_passes.h"
#include "borgc_capture.h"

#include "compiler/glsl_types.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "util/ralloc.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOB_MAX 4096
#define POOL_MAX 64
#define CS_WORDS_MAX 4096
#define CS_REGS_MAX 32
#define CS_POOL_MAX 4096

static void
put(struct blob *o, const void *p, size_t n)
{
   blob_write_bytes(o, p, n);
}

static void
put32(struct blob *o, uint32_t v)
{
   blob_write_uint32(o, v);
}

static void
compile_graphics(struct blob *o, nir_shader *nir, uint32_t vfetch, int32_t out_select)
{
   static uint8_t buf[BLOB_MAX];
   struct borgc_info info;
   uint32_t len = 0;
   uint32_t ret = borgc_compile_nir(nir, buf, BLOB_MAX, &len, vfetch, out_select, &info);
   put32(o, ret);
   put32(o, len);
   put(o, buf, len <= BLOB_MAX ? len : 0);
   put32(o, info.push_window);
   put32(o, info.spill_slots);
   /* The pool of the shader's stage, read only for a blob the driver keeps. */
   uint32_t n = len > 0 && len <= BLOB_MAX ? MIN2(info.npool, POOL_MAX) : 0;
   put32(o, n);
   put(o, info.pool, 4 * n);
}

static void
compile_compute(struct blob *o, nir_shader *nir)
{
   static uint32_t words[CS_WORDS_MAX], regs[2 * CS_REGS_MAX], pool[CS_POOL_MAX];
   uint32_t nwords = 0, nregs = 0, local[3] = { 0 }, flags = 0, npool = 0;
   uint32_t rc = borgc_compile_compute(nir, words, CS_WORDS_MAX, &nwords, regs, CS_REGS_MAX,
                                       &nregs, local, &flags, pool, CS_POOL_MAX, &npool);
   put32(o, rc);
   if (rc == 0) {
      put32(o, nwords);
      put(o, words, 4 * nwords);
      put32(o, nregs);
      put(o, regs, 8 * nregs);
      put(o, local, sizeof local);
      put32(o, flags);
   }
   put32(o, rc == 0 ? npool : 0);
   put(o, pool, rc == 0 ? 4 * npool : 0);
}

static bool
replay(const char *in, const char *out)
{
   FILE *f = fopen(in, "rb");
   if (!f)
      return false;
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *data = sz > 16 ? malloc(sz) : NULL;
   bool ok = data && fread(data, 1, sz, f) == (size_t)sz;
   fclose(f);
   uint32_t head[4];
   if (ok)
      memcpy(head, data, sizeof head);
   if (!ok || head[0] != BORGC_CAPTURE_MAGIC) {
      free(data);
      return false;
   }
   struct blob_reader r;
   blob_reader_init(&r, (const uint8_t *)data + sizeof head, sz - sizeof head);
   nir_shader *nir = nir_deserialize(NULL, &borg_nir_options, &r);
   /* Captures from before borgc took phis hold NIR registers. */
   nir_lower_reg_intrinsics_to_ssa(nir);
   struct blob o;
   blob_init(&o);
   if (head[1] == BORGC_CAPTURE_COMPUTE)
      compile_compute(&o, nir);
   else
      compile_graphics(&o, nir, head[2], (int32_t)head[3]);
   ralloc_free(nir);
   free(data);
   f = fopen(out, "wb");
   ok = f && fwrite(o.data, 1, o.size, f) == o.size;
   if (f)
      ok &= fclose(f) == 0;
   blob_finish(&o);
   return ok;
}

int
main(int argc, char **argv)
{
   if (argc != 3) {
      fprintf(stderr, "usage: %s IN_DIR OUT_DIR\n", argv[0]);
      return 2;
   }
   DIR *d = opendir(argv[1]);
   if (!d) {
      fprintf(stderr, "borgc-replay: cannot open %s\n", argv[1]);
      return 2;
   }
   glsl_type_singleton_init_or_ref();
   unsigned done = 0, failed = 0;
   struct dirent *e;
   while ((e = readdir(d))) {
      size_t n = strlen(e->d_name);
      if (n < 5 || strcmp(e->d_name + n - 4, ".nir"))
         continue;
      char in[4096], out[4096];
      snprintf(in, sizeof in, "%s/%s", argv[1], e->d_name);
      snprintf(out, sizeof out, "%s/%.*s.out", argv[2], (int)(n - 4), e->d_name);
      if (replay(in, out))
         done++;
      else
         failed++;
   }
   closedir(d);
   glsl_type_singleton_decref();
   fprintf(stderr, "borgc-replay: %u shader(s) replayed, %u unreadable\n", done, failed);
   return failed ? 1 : 0;
}
