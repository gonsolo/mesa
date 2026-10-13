/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "drm-uapi/borg_drm.h"

#include <fcntl.h>
#include "borgvk_drm_compat.h"

#define UBO_MVP_FLOATS   16
#define UBO_NUM_VERTS    36                  /* 12 triangles, expanded */

#define UBO_POS_FLOAT0   UBO_MVP_FLOATS      /* position[36][4] */
#define UBO_ATTR_FLOAT0  (UBO_MVP_FLOATS + UBO_NUM_VERTS * 4)  /* attr[36][4] */

#define UBO_MIN_FLOATS   (UBO_ATTR_FLOAT0 + UBO_NUM_VERTS * 4) /* 304 = 1216 B */

/* Locate the first render-pass colour attachment image (for sim readback). */
static struct borgvk_image *
find_color_attachment(struct vk_queue_submit *submit)
{
   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      struct borgvk_command_buffer *bcb =
         container_of(cb, struct borgvk_command_buffer, vk);
      if (bcb->color_views[0] &&
          bcb->color_views[0]->image)
         return container_of(bcb->color_views[0]->image, struct borgvk_image, vk);
   }
   return NULL;
}

static void
send_geometry(const float *ubo)
{
   const float *pos = ubo + UBO_POS_FLOAT0;
   const float *att = ubo + UBO_ATTR_FLOAT0;

   float verts[BORGVK_GEOM_MAX_VERTS * 3];
   uint8_t idx[UBO_NUM_VERTS];
   float uv[UBO_NUM_VERTS * 2];
   int nverts = 0;

   for (int i = 0; i < UBO_NUM_VERTS; i++) {
      float x = pos[i*4 + 0], y = pos[i*4 + 1], z = pos[i*4 + 2];
      int u = -1;
      for (int j = 0; j < nverts; j++) {
         if (verts[j*3+0] == x && verts[j*3+1] == y && verts[j*3+2] == z) {
            u = j;
            break;
         }
      }
      if (u < 0) {
         if (nverts >= BORGVK_GEOM_MAX_VERTS)
            return;  /* mesh too complex for the fixed packet (cube fits) */
         u = nverts++;
         verts[u*3+0] = x; verts[u*3+1] = y; verts[u*3+2] = z;
      }
      idx[i] = (uint8_t)u;
      uv[i*2+0] = att[i*4 + 0];
      uv[i*2+1] = att[i*4 + 1];
   }

   borgvk_serial_send_geom(verts, nverts, idx, uv, UBO_NUM_VERTS / 3);
}

static void
sentinel_path(char *buf, size_t n)
{
   const char *dev = getenv("BORGVK_SERIAL");
   if (!dev || !dev[0])
      dev = "/dev/ttyUSB0";
   const char *base = strrchr(dev, '/');
   base = base ? base + 1 : dev;
   snprintf(buf, n, "/tmp/borgvk_%s_setup", base);
}

static bool
setup_already_done(void)
{
   if (getenv("BORGVK_FORCE_UPLOAD"))
      return false;
   char path[256];
   sentinel_path(path, sizeof(path));
   return access(path, F_OK) == 0;
}

static void
mark_setup_done(void)
{
   char path[256];
   sentinel_path(path, sizeof(path));
   int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
   if (fd >= 0)
      close(fd);
}

/* True when this descriptor set is cube.c's UBO-driven frame (vs a CTS VBO draw). */
bool
borgvk_set_is_cube(struct borgvk_descriptor_set *set)
{
   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return false;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return false;
   VkDeviceSize avail = ubuf->vk.size - off;
   if (set->ranges[0] != VK_WHOLE_SIZE && set->ranges[0] < avail)
      avail = set->ranges[0];
   return avail / sizeof(float) >= UBO_MIN_FLOATS;
}

bool
borgvk_submit_is_cube(struct vk_queue_submit *submit)
{
   struct borgvk_descriptor_set *set = borgvk_find_set(submit);
   return set && borgvk_set_is_cube(set);
}

VkResult
borgvk_submit_sim_cube(struct borgvk_device *device,
                        struct vk_queue_submit *submit)
{
   const char *sim_bin = getenv("BORGVK_SIM");
   const char *sim_fw  = getenv("BORGVK_SIM_FW");
   if (!borgvk_hw_enabled() && (!sim_bin || !sim_fw))
      return VK_SUCCESS;

   struct borgvk_descriptor_set *set = borgvk_find_set(submit);
   if (!set)
      return VK_SUCCESS;
   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return VK_SUCCESS;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return VK_SUCCESS;
   const float *ubo = (const float *)((const char *)ubuf->mem->map + off);
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));
   if (nfloats < UBO_MIN_FLOATS)
      return VK_SUCCESS;
   struct borgvk_image *tex = set->images[1];

   /* Capture one frame's full wire stream (shaders + geom + texture + MVP). */
   borgvk_transport_capture_begin();
   borgvk_upload_shaders(device);
   send_geometry(ubo);
   borgvk_send_blend_state(device);
   if (tex && tex->mem && tex->mem->map &&
       tex->vk.extent.width && tex->vk.extent.height)
      for (int row = 0; row < BORGVK_TEX_DIM; row++)
         borgvk_send_texture_row(tex, set->samplers[1], row);
   borgvk_serial_send_mvp(ubo);
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   if (!bytes || nbytes == 0) {
      free(bytes);
      return VK_SUCCESS;
   }

   const char *dump = getenv("BORGVK_SIM_DUMP");
   if (dump && dump[0]) {
      int dfd = open(dump, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (dfd >= 0) {
         for (size_t w = 0; w < nbytes; ) {
            ssize_t n = write(dfd, bytes + w, nbytes - w);
            if (n <= 0) break;
            w += (size_t)n;
         }
         close(dfd);
      }
      mesa_logi("borgvk: dumped %zu-byte sim stream to %s", nbytes, dump);
      free(bytes);
      return VK_SUCCESS;
   }

   struct borgvk_image *color_img = find_color_attachment(submit);
   const bool rec = borgvk_corpus_on();
   const uint64_t key = rec ? borgvk_corpus_hash(BORGVK_CORPUS_SEED, bytes, nbytes) : 0;
   const VkResult res = borgvk_sim_run_stream(bytes, nbytes, color_img, 0);
   if (rec)
      borgvk_corpus_log('C', key, borgvk_corpus_image(BORGVK_CORPUS_SEED, color_img));
   return res;
}

VkResult
borgvk_submit_cube_live(struct borgvk_device *device, struct vk_queue_submit *submit)
{
   struct borgvk_descriptor_set *set = borgvk_find_set(submit);
   if (!set)
      return VK_SUCCESS;

   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return VK_SUCCESS;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return VK_SUCCESS;
   const float *ubo = (const float *)((const char *)ubuf->mem->map + off);
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));

   struct borgvk_image *tex = set->images[1];
   bool can_geom = nfloats >= UBO_MIN_FLOATS;
   bool can_tex  = tex && tex->mem && tex->mem->map &&
                   tex->vk.extent.width && tex->vk.extent.height;

   static bool g_setup_done = false;

   if (device->drm_fd >= 0) {
      bool has_gem = ubuf->mem->gem_handle != 0 &&
                     tex && tex->mem && tex->mem->gem_handle != 0;

      if (!g_setup_done && has_gem) {
         borgvk_upload_shaders(device);   /* borgc-compiled shaders, before geom/tex */
         struct drm_borg_setup s = {
            .ubo_handle = ubuf->mem->gem_handle,
            .tex_handle = tex->mem->gem_handle,
            .tex_offset = (__u64)tex->offset,
            .tex_width  = tex->vk.extent.width,
            .tex_height = tex->vk.extent.height,
         };
         memcpy(s.sampler, borgvk_sampler_desc(set->samplers[1]), sizeof(s.sampler));
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SETUP, &s) == 0)
            g_setup_done = true;
         else
            mesa_logw("borgvk: DRM_IOCTL_BORG_SETUP failed");
      }

      if (g_setup_done) {
         borgvk_send_blend_state(device);
         struct drm_borg_submit sub = { .ubo_handle = ubuf->mem->gem_handle };
         drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SUBMIT, &sub);
      }
   } else {
      if (!g_setup_done) {
         if (setup_already_done()) {
            mesa_logi("borgvk: skipping upload (sentinel present); "
                      "set BORGVK_FORCE_UPLOAD=1 to re-upload");
            g_setup_done = true;
         } else if (can_geom && can_tex) {
            mesa_logi("borgvk: uploading shaders + geometry + texture burst...");
            borgvk_upload_shaders(device);   /* borgc-compiled shaders, before geom/tex */
            send_geometry(ubo);
            for (int row = 0; row < BORGVK_TEX_DIM; row++)
               borgvk_send_texture_row(tex, set->samplers[1], row);
            mark_setup_done();
            mesa_logi("borgvk: upload complete");
            g_setup_done = true;
         }
      }

      if (g_setup_done) {
         borgvk_send_blend_state(device);
         borgvk_serial_send_mvp(ubo);
      }
   }

   return VK_SUCCESS;
}
