/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borgvk_private.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
emit_burst(const uint8_t *vert, size_t vert_len,
           const uint8_t *frag, size_t frag_len,
           const float color[4], const char *out_path)
{
   static const float verts[3 * 3] = {
      -1.0f, -1.0f, 0.0f,
       3.0f, -1.0f, 0.0f,
      -1.0f,  3.0f, 0.0f,
   };
   static const uint8_t idx[2 * 3] = { 0, 1, 2,   0, 2, 1 };
   static const float uv[2 * 3 * 2] = { 0 };

   static const float mvp[16] = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
   };

   borgvk_transport_capture_begin();

   borgvk_serial_send_shader(0, vert, (uint32_t)vert_len);
   borgvk_serial_send_shader(1, frag, (uint32_t)frag_len);
   borgvk_serial_send_geom(verts, 3, idx, uv, 2);
   borgvk_serial_send_push_constants(32, 16, color);
   borgvk_serial_send_mvp(mvp);

   size_t len = 0;
   uint8_t *buf = borgvk_transport_capture_end(&len);
   if (!buf || len == 0) {
      fprintf(stderr, "borgvk_burst_gen: capture produced nothing\n");
      free(buf);
      return 1;
   }

   FILE *f = fopen(out_path, "wb");
   if (!f) {
      fprintf(stderr, "borgvk_burst_gen: cannot write %s\n", out_path);
      free(buf);
      return 1;
   }
   size_t wrote = fwrite(buf, 1, len, f);
   fclose(f);
   free(buf);

   if (wrote != len) {
      fprintf(stderr, "borgvk_burst_gen: short write to %s\n", out_path);
      return 1;
   }

   printf("borgvk_burst_gen: wrote %zu bytes to %s "
          "(colour %.3f %.3f %.3f %.3f)\n",
          len, out_path, color[0], color[1], color[2], color[3]);
   return 0;
}

static uint8_t *
read_blob(const char *path, size_t *out_len)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "borgvk_burst_gen: cannot open %s\n", path);
      return NULL;
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (sz <= 0) {
      fclose(f);
      fprintf(stderr, "borgvk_burst_gen: %s is empty\n", path);
      return NULL;
   }
   uint8_t *buf = malloc((size_t)sz);
   if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
      fclose(f);
      free(buf);
      fprintf(stderr, "borgvk_burst_gen: short read of %s\n", path);
      return NULL;
   }
   fclose(f);
   *out_len = (size_t)sz;
   return buf;
}

int
main(int argc, char **argv)
{
   if (argc != 8) {
      fprintf(stderr,
              "usage: %s <vert.borg> <frag.borg> <out.bin> <r> <g> <b> <a>\n",
              argv[0]);
      return 2;
   }

   size_t vert_len = 0, frag_len = 0;
   uint8_t *vert = read_blob(argv[1], &vert_len);
   uint8_t *frag = read_blob(argv[2], &frag_len);
   if (!vert || !frag) {
      free(vert);
      free(frag);
      return 1;
   }

   float color[4];
   for (int i = 0; i < 4; i++)
      color[i] = strtof(argv[4 + i], NULL);

   int rc = emit_burst(vert, vert_len, frag, frag_len, color, argv[3]);
   free(vert);
   free(frag);
   return rc;
}
