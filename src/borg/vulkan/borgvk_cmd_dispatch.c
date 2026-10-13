/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_format.h"

#include <sys/wait.h>

#define BORGVK_CS_SLOT_BYTES (0x80000u * 4)
#define BORGVK_CS_BASE_BYTES (0x40000u * 4)
#define BORGVK_MAX_CS_PARTS 64

static void
put_u32(uint8_t **p, uint32_t v)
{
   memcpy(*p, &v, 4);
   *p += 4;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDispatch(VkCommandBuffer commandBuffer, uint32_t gx, uint32_t gy, uint32_t gz)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   const struct borgvk_pipeline *pl = cmd->cs_pipeline;
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (!pl || !pl->cs_ok || !gx || !gy || !gz || !direct || !direct[0])
      return;
   struct { uint8_t *host; uint8_t *decoded; bool wb; uint32_t skip, size, addr;
            uint8_t *texels; enum pipe_format pf; } bufs[34];
   memset(bufs, 0, sizeof(bufs));
   uint32_t nbufs = 0, total = 0;
   for (uint32_t si = 0; si < 4; si++) {
   struct borgvk_descriptor_set *set = cmd->cs_sets[si];
   const VkDeviceSize *dyn_off = cmd->cs_dyn_offs[si];
   for (uint32_t b = 0; set && b < BORGVK_MAX_BINDINGS && pl->set_base[si] + b < 32; b++) {
      const struct vk_buffer_view *bv = set->buffer_views[b];
      if (bv && !set->buffers[b]) {
         struct borgvk_buffer *tb = container_of(bv->buffer, struct borgvk_buffer, vk);
         enum pipe_format pf = vk_format_to_pipe_format(bv->format);
         uint32_t bs = util_format_get_blocksize(pf);
         if (!tb->mem || !tb->mem->map || pf == PIPE_FORMAT_NONE || bs == 0)
            continue;
         uint32_t n = (uint32_t)MIN2(bv->range / bs, BORGVK_CS_SLOT_BYTES / 16);
         uint8_t *dec = calloc(MAX2(n, 1), 16);
         if (!dec)
            continue;
         const uint8_t *src = (const uint8_t *)tb->mem->map + tb->offset + bv->offset;
         for (uint32_t t = 0; t < n; t++)
            util_format_unpack_rgba(pf, dec + (size_t)t * 16, src + (size_t)t * bs, 1);
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].texels = (uint8_t *)tb->mem->map + tb->offset + bv->offset;
         bufs[nbufs].pf = pf;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = n * 16;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
         total += 8 + n * 16;
         nbufs++;
         continue;
      }
      struct borgvk_buffer *buf = set->buffers[b];
      struct borgvk_image *simg = set->images[b];
      if (!buf && !bv && simg && (set->sampled_mask >> b & 1) && simg->mem && simg->mem->map &&
          simg->vk.image_type == VK_IMAGE_TYPE_2D && simg->vk.samples == 1) {
         enum pipe_format pf = vk_format_to_pipe_format(simg->vk.format);
         uint32_t w = simg->vk.extent.width, h = simg->vk.extent.height;
         uint32_t bs = util_format_get_blocksize(pf);
         if (pf == PIPE_FORMAT_NONE || bs == 0 || util_format_is_compressed(pf))
            continue;
         uint32_t n = MIN2(w * h, BORGVK_CS_SLOT_BYTES / 16 - 1);
         uint8_t *dec = calloc(1 + n, 16);
         if (!dec)
            continue;
         memcpy(dec, &w, 4);
         const uint8_t *src = (const uint8_t *)simg->mem->map + simg->offset;
         for (uint32_t t = 0; t < n; t++)
            util_format_unpack_rgba(pf, dec + 16 + (size_t)t * 16, src + (size_t)t * bs, 1);
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = (1 + n) * 16;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
         total += 8 + (1 + n) * 16;
         nbufs++;
         continue;
      }
      if (!buf && !bv && simg && simg->mem && simg->mem->map && simg->vk.image_type == VK_IMAGE_TYPE_2D &&
          vk_format_get_blocksize(simg->vk.format) == 4 && simg->vk.samples == 1) {
         uint32_t w = simg->vk.extent.width, h = simg->vk.extent.height;
         uint32_t n = MIN2(w * h, BORGVK_CS_SLOT_BYTES / 4 - 1);
         uint8_t *dec = calloc(1 + n, 4);
         if (!dec)
            continue;
         memcpy(dec, &w, 4);
         memcpy(dec + 4, (uint8_t *)simg->mem->map + simg->offset, (size_t)n * 4);
         bufs[nbufs].host = (uint8_t *)simg->mem->map + simg->offset;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = true;
         bufs[nbufs].skip = 4;
         bufs[nbufs].size = (1 + n) * 4;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
         total += 8 + (1 + n) * 4;
         nbufs++;
         continue;
      }
      if (!buf || !buf->mem || !buf->mem->map)
         continue;
      VkDeviceSize boff = set->offsets[b] + dyn_off[b];
      VkDeviceSize avail = buf->vk.size > boff ? buf->vk.size - boff : 0;
      VkDeviceSize size = set->ranges[b] == VK_WHOLE_SIZE || set->ranges[b] > avail ? avail : set->ranges[b];
      if (size > BORGVK_CS_SLOT_BYTES)
         size = BORGVK_CS_SLOT_BYTES;
      bufs[nbufs].host = (uint8_t *)buf->mem->map + buf->offset + boff;
      bufs[nbufs].decoded = NULL;
      bufs[nbufs].wb = true;
      bufs[nbufs].skip = 0;
      bufs[nbufs].size = (uint32_t)size;
      bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
      total += 8 + ((uint32_t)size + 3) / 4 * 4;
      nbufs++;
   }
   }

   /* Push constants: one more window, slot 32. */
   {
      uint8_t *dec = calloc(1, sizeof(cmd->pc));
      if (dec) {
         memcpy(dec, cmd->pc, sizeof(cmd->pc));
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = sizeof(cmd->pc);
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + 32 * BORGVK_CS_SLOT_BYTES;
         total += 8 + sizeof(cmd->pc);
         nbufs++;
      }
   }

   const uint32_t HDR = 12;
   size_t cap = HDR * 4 + pl->cs_nwords * 4 + pl->cs_nregs * 8 + total;
   uint8_t *job = malloc(cap), *w = job;
   put_u32(&w, pl->cs_nwords); put_u32(&w, gx); put_u32(&w, gy); put_u32(&w, gz);
   put_u32(&w, pl->cs_local[0]); put_u32(&w, pl->cs_local[1]); put_u32(&w, pl->cs_local[2]);
   put_u32(&w, pl->cs_nregs); put_u32(&w, nbufs);
   put_u32(&w, 0); put_u32(&w, 0); put_u32(&w, 0);
   for (uint32_t i = 0; i < pl->cs_nwords; i++) put_u32(&w, pl->cs_words[i]);
   for (uint32_t i = 0; i < pl->cs_nregs; i++) { put_u32(&w, pl->cs_regs[2 * i]); put_u32(&w, pl->cs_regs[2 * i + 1]); }
   for (uint32_t i = 0; i < nbufs; i++) {
      put_u32(&w, bufs[i].addr); put_u32(&w, bufs[i].size);
      memcpy(w, bufs[i].decoded ? bufs[i].decoded : bufs[i].host, bufs[i].size);
      memset(w + bufs[i].size, 0, ((bufs[i].size + 3) & ~3u) - bufs[i].size);
      w += (bufs[i].size + 3) & ~3u;
   }
   const size_t job_len = (size_t)(w - job);

   uint32_t axis = gy > 1 && gy >= gx ? 1 : gx > 1 ? 0 : 2;
   uint32_t extent = axis == 0 ? gx : axis == 1 ? gy : gz;
   uint32_t nparts = 1;
   if (!(pl->cs_flags & 1) && (uint64_t)gx * gy * gz >= 4096 && extent > 1) {
      long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
      nparts = (uint32_t)MIN2((long)extent, MAX2(ncpu, 1));
      if (getenv("BORGVK_CS_PARTS"))
         nparts = (uint32_t)MAX2(1, MIN2((long)extent, atol(getenv("BORGVK_CS_PARTS"))));
   }

   char jpath[BORGVK_MAX_CS_PARTS][32], opath[BORGVK_MAX_CS_PARTS][32];
   pid_t pids[BORGVK_MAX_CS_PARTS];
   nparts = MIN2(nparts, BORGVK_MAX_CS_PARTS);
   bool ran = true;
   for (uint32_t k = 0; k < nparts; k++) {
      snprintf(jpath[k], sizeof jpath[k], "/tmp/borgvk_cs_job_XXXXXX");
      snprintf(opath[k], sizeof opath[k], "/tmp/borgvk_cs_out_XXXXXX");
      pids[k] = -1;
      int jfd = mkstemp(jpath[k]), ofd = mkstemp(opath[k]);
      if (jfd < 0 || ofd < 0) { ran = false; continue; }
      uint32_t lo = extent * k / nparts, hi = extent * (k + 1) / nparts;
      uint32_t hdr[HDR];
      memcpy(hdr, job, sizeof hdr);
      hdr[1 + axis] = hi - lo;        /* gx gy gz */
      hdr[9 + axis] = lo;             /* bx by bz */
      if (write(jfd, hdr, sizeof hdr) != (ssize_t)sizeof hdr) ran = false;
      for (size_t o = sizeof hdr; o < job_len; ) {
         ssize_t n = write(jfd, job + o, job_len - o);
         if (n <= 0) break;
         o += (size_t)n;
      }
      close(jfd);
      close(ofd);
      pids[k] = fork();
      if (pids[k] == 0) {
         setenv("BORG_SIM_CFG", "simt", 1);
         execlp(direct, direct, "--compute", jpath[k], opath[k], (char *)NULL);
         _exit(127);
      }
   }
   uint8_t *merged[BORGVK_MAX_BINDINGS + 1] = { 0 };
   for (uint32_t i = 0; i < nbufs; i++) {
      merged[i] = malloc(((bufs[i].size + 3) & ~3u) + 1);
      memcpy(merged[i], bufs[i].decoded ? bufs[i].decoded : bufs[i].host, bufs[i].size);
   }
   for (uint32_t k = 0; k < nparts; k++) {
      int st = 0;
      if (pids[k] > 0)
         waitpid(pids[k], &st, 0);
      FILE *f = pids[k] > 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0 ? fopen(opath[k], "rb") : NULL;
      if (!f) {
         mesa_logw("borgvk: compute sim failed (status %d)", st);
         ran = false;
         continue;
      }
      for (uint32_t i = 0; i < nbufs; i++) {
         uint8_t *tmp = malloc(((bufs[i].size + 3) & ~3u) + 1);
         const uint8_t *orig = bufs[i].decoded ? bufs[i].decoded : bufs[i].host;
         if (tmp && fread(tmp, 1, bufs[i].size, f) == bufs[i].size)
            for (uint32_t b = 0; b < bufs[i].size; b++)
               if (tmp[b] != orig[b])
                  merged[i][b] = tmp[b];
         free(tmp);
         fseek(f, ((bufs[i].size + 3) & ~3u) - bufs[i].size, SEEK_CUR);
      }
      fclose(f);
   }
   if (borgvk_corpus_on()) {
      uint64_t out = borgvk_corpus_hash(BORGVK_CORPUS_SEED, &ran, sizeof(ran));
      for (uint32_t i = 0; i < nbufs; i++)
         out = borgvk_corpus_hash(out, merged[i], bufs[i].size);
      borgvk_corpus_log('D', borgvk_corpus_hash(BORGVK_CORPUS_SEED, job, job_len), out);
   }
   if (ran) {
      for (uint32_t i = 0; i < nbufs; i++)
         if (bufs[i].wb)
            memcpy(bufs[i].host, merged[i] + bufs[i].skip, bufs[i].size - bufs[i].skip);
      /* A texel buffer: the texels a shader stored, encoded back into the view's format. */
      for (uint32_t i = 0; i < nbufs; i++) {
         if (!bufs[i].texels)
            continue;
         const uint32_t bs = util_format_get_blocksize(bufs[i].pf);
         for (uint32_t t = 0; t < bufs[i].size / 16; t++)
            if (memcmp(merged[i] + t * 16, bufs[i].decoded + t * 16, 16) != 0)
               util_format_pack_rgba(bufs[i].pf, bufs[i].texels + (size_t)t * bs, merged[i] + t * 16, 1);
      }
      cmd->dispatched = true;
   }
   for (uint32_t k = 0; k < nparts; k++) {
      if (!getenv("BORGVK_KEEP_JOB"))
         unlink(jpath[k]);
      unlink(opath[k]);
   }
   for (uint32_t i = 0; i < nbufs; i++) {
      free(merged[i]);
      free(bufs[i].decoded);
   }
   free(job);
}
