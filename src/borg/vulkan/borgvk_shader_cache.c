/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * borgvk_shader_cache.c — compiled shaders keyed by their SPIR-V.
 *
 * A board running borgvk compiles slowly. $BORGVK_SHADER_CACHE names a file of compiled
 * blobs; a stage whose SPIR-V is in it skips SPIR-V → NIR → borgc altogether.
 * $BORGVK_SHADER_CACHE_RECORD=<file> appends every compiled stage to such a file.
 *
 * Entry: u64 key, u8 flags (bit 0: reads gl_PointCoord), u16 len, blob.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "borgvk_private.h"
#include "vk_shader_module.h"
#include "vk_util.h"

struct entry {
   uint64_t key;
   uint8_t flags;
   uint16_t len;
   uint8_t data[BORGVK_SHADER_BLOB_MAX];
};

static struct { struct entry *e; unsigned n; bool loaded; } cache;

static uint64_t
fnv(uint64_t h, const void *p, size_t n)
{
   const uint8_t *b = p;
   while (n--)
      h = (h ^ *b++) * 0x100000001b3ull;
   return h;
}

static bool
stage_key(const VkPipelineShaderStageCreateInfo *si, uint32_t vfetch, uint64_t *key)
{
   const void *code = NULL;
   size_t size = 0;
   if (si->module != VK_NULL_HANDLE) {
      struct vk_shader_module *m = vk_shader_module_from_handle(si->module);
      code = m->data;
      size = m->size;
   } else {
      const VkShaderModuleCreateInfo *ci = vk_find_struct_const(si->pNext, SHADER_MODULE_CREATE_INFO);
      if (ci) {
         code = ci->pCode;
         size = ci->codeSize;
      }
   }
   if (!code || !size)
      return false;
   uint64_t h = 0xcbf29ce484222325ull;
   h = fnv(h, code, size);
   h = fnv(h, si->pName, strlen(si->pName));
   h = fnv(h, &si->stage, sizeof(si->stage));
   h = fnv(h, &vfetch, sizeof(vfetch));
   *key = h;
   return true;
}

static void
load(void)
{
   cache.loaded = true;
   const char *path = getenv("BORGVK_SHADER_CACHE");
   FILE *f = path && path[0] ? fopen(path, "rb") : NULL;
   if (!f)
      return;
   for (;;) {
      struct entry e;
      if (fread(&e.key, 8, 1, f) != 1 || fread(&e.flags, 1, 1, f) != 1 || fread(&e.len, 2, 1, f) != 1 ||
          e.len > BORGVK_SHADER_BLOB_MAX || fread(e.data, 1, e.len, f) != e.len)
         break;
      struct entry *g = realloc(cache.e, (cache.n + 1) * sizeof(*g));
      if (!g)
         break;
      cache.e = g;
      cache.e[cache.n++] = e;
   }
   fclose(f);
}

bool
borgvk_shader_cache_get(const VkPipelineShaderStageCreateInfo *si, uint32_t vfetch,
                        struct borgvk_shader_blob *blob, bool *reads_pntc)
{
   uint64_t key;
   if (!cache.loaded)
      load();
   if (!cache.n || !stage_key(si, vfetch, &key))
      return false;
   for (unsigned i = 0; i < cache.n; i++) {
      if (cache.e[i].key == key) {
         memcpy(blob->data, cache.e[i].data, cache.e[i].len);
         blob->len = cache.e[i].len;
         *reads_pntc = cache.e[i].flags & 1;
         return true;
      }
   }
   return false;
}

void
borgvk_shader_cache_record(const VkPipelineShaderStageCreateInfo *si, uint32_t vfetch,
                           const struct borgvk_shader_blob *blob, bool reads_pntc)
{
   const char *path = getenv("BORGVK_SHADER_CACHE_RECORD");
   uint64_t key;
   if (!path || !path[0] || !blob->len || !stage_key(si, vfetch, &key))
      return;
   FILE *f = fopen(path, "ab");
   if (!f)
      return;
   uint8_t flags = reads_pntc;
   uint16_t len = blob->len;
   fwrite(&key, 8, 1, f);
   fwrite(&flags, 1, 1, f);
   fwrite(&len, 2, 1, f);
   fwrite(blob->data, 1, len, f);
   fclose(f);
}
