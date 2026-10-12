/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int
corpus_fd(void)
{
   static int fd = -2;
   if (fd == -2) {
      const char *dir = getenv("BORGVK_CORPUS_DIR");
      char path[512];
      snprintf(path, sizeof(path), "%s/%d.log", dir ? dir : "", (int)getpid());
      fd = dir ? open(path, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1;
   }
   return fd;
}

bool
borgvk_corpus_on(void)
{
   return corpus_fd() >= 0;
}

uint64_t
borgvk_corpus_hash(uint64_t h, const void *p, size_t n)
{
   const uint8_t *b = p;
   for (size_t i = 0; i < n; i++)
      h = (h ^ b[i]) * 0x100000001b3ull;
   return h;
}

uint64_t
borgvk_corpus_image(uint64_t h, const struct borgvk_image *img)
{
   if (img && img->mem && img->mem->map)
      h = borgvk_corpus_hash(h, (const uint8_t *)img->mem->map + img->offset, img->size);
   return h;
}

void
borgvk_corpus_log(char kind, uint64_t key, uint64_t content)
{
   char line[64];
   const int n = snprintf(line, sizeof(line), "%c %016llx %016llx\n", kind,
                          (unsigned long long)key, (unsigned long long)content);
   if (write(corpus_fd(), line, n) < 0)
      return;
}
