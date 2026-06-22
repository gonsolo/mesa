/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Queue submit for borgvk — the single interception point of the driver. cube.c
 * records a frame into a command buffer (captured into cmd_buffer->cmd_queue by
 * the runtime's vk_cmd_enqueue emulation) and submits it. We never replay those
 * commands; instead we locate the bound descriptor set and forward the app's
 * real data to the FPGA over serial:
 *
 *   binding 0 (uniform buffer) = `struct vktexcube_vs_uniform`:
 *       float mvp[4][4];          // bytes   0..63   (updated every frame → 0xAD)
 *       float position[12*3][4];  // bytes  64..639  (static mesh → 0xAE)
 *       float attr[12*3][4];      // bytes 640..1215 (static UVs)
 *   binding 1 (combined image sampler) = the texture (RGBA8 → 0xAF rows)
 *
 * Startup: geometry + texture are burst-uploaded once on the first submit, then
 * every subsequent submit ships only the MVP.  A sentinel file in /tmp records
 * that the FPGA already holds the mesh+texture so subsequent runs in the same
 * power cycle skip the upload entirely (~0 s startup instead of ~2 s).
 * Set BORGVK_FORCE_UPLOAD=1 to ignore the sentinel and re-upload unconditionally.
 */
#include "borgvk_private.h"

#include "vk_command_buffer.h"
#include "vk_cmd_queue.h"
#include "vk_framebuffer.h"
#include "vk_image.h"

#include "drm-uapi/borg_drm.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <xf86drm.h>

/* Firmware layout constants (mailbox offsets, max verts/tris, magic). */
#include "software/borg/borg_layout.h"

/* cube.c uniform-buffer layout (floats). */
#define UBO_MVP_FLOATS   16
#define UBO_NUM_VERTS    36                  /* 12 triangles, expanded */
#define UBO_POS_FLOAT0   UBO_MVP_FLOATS      /* position[36][4] */
#define UBO_ATTR_FLOAT0  (UBO_MVP_FLOATS + UBO_NUM_VERTS * 4)  /* attr[36][4] */
#define UBO_MIN_FLOATS   (UBO_ATTR_FLOAT0 + UBO_NUM_VERTS * 4) /* 304 = 1216 B */

#define GEOM_FRAMES      24                      /* startup frames shipping the mesh */
/* Texture-upload window, in host submits (each cycles one more of the 64 rows).
 * The firmware drops bytes during its ~300 ms render, so a row sent only during
 * render gaps is lost that cycle; with the firmware's greedy texture drain, 2
 * cycles reached 54/64 distinct rows (measured).  6 cycles over-provisions so the
 * stragglers — different ones each cycle as render/send timing drifts — all land. */
#define TEX_FRAMES       (BORGVK_TEX_DIM * 6)    /* then switch to per-frame MVP */

/* ---- Arcilator sim draw path (BORGVK_SIM) --------------------------------
 * Renders one frame via arcilator_sim --cts-draw and writes the result into
 * the color attachment's backing host memory.  Activated when BORGVK_SIM and
 * BORGVK_SIM_FW are set in the environment.
 *
 * Geom file format (binary, little-endian):
 *   uint32 magic = 0x42475254 ("BGRT")
 *   uint32 nverts, uint32 ntris
 *   float  mvp[16]       (identity for CTS — positions are already in NDC)
 *   float  pos[nverts*3] (NDC xyz)
 *   float  col[nverts*3] (rgb 0..1)
 *   uint32 idx[ntris*3]
 *
 * CTS vertex buffer format (PositionColorVertex @ stride 32):
 *   offset  0: float x,y,z,w  (clip position, z=1,w=1 for CTS)
 *   offset 16: float r,g,b,a  (color)
 *
 * Winding: Borg's culler keeps CW triangles in screen space (y-down).
 * Vulkan front-face is CCW, so most CTS triangles are CCW in NDC =
 * CW in screen (after y-flip) = front-facing = passes.  However the
 * RNG can produce any orientation, so we check the signed screen-space
 * area and reverse the index order of back-facing triangles.
 */

#define GEOM_MAGIC 0x42475254u  /* "BGRT" */

/* Signed 2D area of triangle p0→p1→p2.
 * In Y-down coordinates (Vulkan NDC and framebuffer both Y-down):
 *   area > 0  →  CW  (clockwise on screen)
 *   area < 0  →  CCW (counter-clockwise on screen)
 * Borg's hardware culler keeps area < 0 (CCW = "front-facing"). */
static float
tri_signed_area(const float *p0, const float *p1, const float *p2)
{
   return (p1[0] - p0[0]) * (p2[1] - p0[1])
        - (p2[0] - p0[0]) * (p1[1] - p0[1]);
}

/* Interpolate scalar s at barycentric point (px,py) inside triangle
 * (p0,p1,p2) in 2D NDC.  Returns -1 if the point is outside (+eps slack). */
static float
tri_interp_scalar(float px, float py,
                  const float *p0, const float *p1, const float *p2,
                  float s0, float s1, float s2)
{
   float d = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]);
   if (fabsf(d) < 1e-10f) return -1.0f;
   float w0 = ((p1[0]-px)*(p2[1]-py) - (p2[0]-px)*(p1[1]-py)) / d;
   float w1 = ((p2[0]-px)*(p0[1]-py) - (p0[0]-px)*(p2[1]-py)) / d;
   float w2 = 1.0f - w0 - w1;
   if (w0 < -0.01f || w1 < -0.01f || w2 < -0.01f) return -1.0f;
   float v = w0*s0 + w1*s1 + w2*s2;
   return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

static VkResult
borgvk_submit_sim_draw(struct vk_queue_submit *submit)
{
   const char *sim_bin = getenv("BORGVK_SIM");
   const char *sim_fw  = getenv("BORGVK_SIM_FW");
   if (!sim_bin || !sim_fw)
      return VK_SUCCESS;

   /* Walk the command queue to collect: colour attachment, VBO, draw params,
    * and the bound pipeline (for cull mode / front-face winding). */
   struct borgvk_image    *color_img   = NULL;
   struct borgvk_buffer   *vbo         = NULL;
   struct borgvk_pipeline *pipeline    = NULL;
   VkDeviceSize            vbo_offset  = 0;
   uint32_t                vert_count  = 0;
   uint32_t                first_vert  = 0;

   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      list_for_each_entry(struct vk_cmd_queue_entry, e, &cb->cmd_queue.cmds, cmd_link) {
         switch (e->type) {
         case VK_CMD_BEGIN_RENDER_PASS: {
            const VkRenderPassBeginInfo *rp = e->u.begin_render_pass.render_pass_begin;
            VK_FROM_HANDLE(vk_framebuffer, fb, rp->framebuffer);
            if (fb && fb->attachment_count > 0) {
               VK_FROM_HANDLE(vk_image_view, view, fb->attachments[0]);
               if (view && view->image)
                  color_img = container_of(view->image, struct borgvk_image, vk);
            }
            break;
         }
         case VK_CMD_BIND_PIPELINE: {
            const struct vk_cmd_bind_pipeline *bp = &e->u.bind_pipeline;
            if (bp->pipeline_bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
               VK_FROM_HANDLE(borgvk_pipeline, pl, bp->pipeline);
               pipeline = pl;
            }
            break;
         }
         case VK_CMD_BIND_VERTEX_BUFFERS: {
            const struct vk_cmd_bind_vertex_buffers *bv = &e->u.bind_vertex_buffers;
            if (bv->binding_count > 0) {
               VK_FROM_HANDLE(borgvk_buffer, buf, bv->buffers[0]);
               vbo        = buf;
               vbo_offset = bv->offsets[0];
            }
            break;
         }
         case VK_CMD_DRAW: {
            const struct vk_cmd_draw *d = &e->u.draw;
            vert_count = d->vertex_count;
            first_vert = d->first_vertex;
            break;
         }
         default:
            break;
         }
      }
   }

   if (!color_img || !vbo || vert_count == 0)
      return VK_SUCCESS;
   if (!color_img->mem || !color_img->mem->map)
      return VK_SUCCESS;
   if (!vbo->mem || !vbo->mem->map)
      return VK_SUCCESS;

   uint32_t width  = color_img->vk.extent.width;
   uint32_t height = color_img->vk.extent.height;

   /* Extract PositionColorVertex array.  Stride = 2 × Vec4 = 32 bytes.
    * pos[4] at offset 0, col[4] at offset 16. */
   const uint8_t *vbase =
      (const uint8_t *)vbo->mem->map + vbo->offset + vbo_offset;
   const uint32_t stride = 32;

   uint32_t nverts = vert_count;  /* expanded: each triangle is 3 unique entries */
   uint32_t ntris  = vert_count / 3;

   if (nverts > BORG_CTS_MAX_VERTS || ntris > BORG_CTS_MAX_TRIS)
      return VK_SUCCESS;  /* mesh too large for the fixed mailbox */

   float *pos     = malloc(nverts * 3 * sizeof(float));
   float *col     = malloc(nverts * 3 * sizeof(float));
   float *alpha_v = malloc(nverts *     sizeof(float));  /* per-vertex alpha for host interp */
   uint32_t *idx  = malloc(ntris  * 3 * sizeof(uint32_t));
   if (!pos || !col || !alpha_v || !idx) {
      free(pos); free(col); free(alpha_v); free(idx);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   for (uint32_t vi = 0; vi < nverts; vi++) {
      const float *vp = (const float *)(vbase + (first_vert + vi) * stride);
      /* CTS positions are Vulkan NDC (Y-down: ndc_y=-1=top, +1=bottom).
       * The firmware's cache_ts_mvp maps ndc_y=-1→screen_y=0 and
       * ndc_y=+1→screen_y=H with no Y flip, so we pass NDC verbatim. */
      pos[vi*3+0] = vp[0];
      pos[vi*3+1] = vp[1];
      pos[vi*3+2] = vp[2];
      /* color at offset 16 = float index 4; alpha at index 7 */
      col[vi*3+0] = vp[4];
      col[vi*3+1] = vp[5];
      col[vi*3+2] = vp[6];
      alpha_v[vi] = vp[7];
      if (getenv("BORGVK_DEBUG"))
         fprintf(stderr, "[borgvk_sim] vert[%u] (abs %u) pos=(%.4f,%.4f,%.4f,%.4f) col=(%.3f,%.3f,%.3f,%.3f)\n",
                 vi, first_vert+vi, vp[0], vp[1], vp[2], vp[3], vp[4], vp[5], vp[6], vp[7]);
   }

   if (getenv("BORGVK_DEBUG"))
      fprintf(stderr, "[borgvk_sim] nverts=%u ntris=%u first_vert=%u vert_count=%u\n",
              nverts, ntris, first_vert, vert_count);

   /* Build index list.  Borg's culler keeps triangles with screen-space
    * signed area < 0 (CCW in Y-down = what Borg calls front-facing).
    * In Y-down NDC: area > 0 = CW = would be culled by Borg.
    * When VK_CULL_MODE_NONE, reverse any CW triangle so it passes. */
   bool apply_winding_fix = (pipeline == NULL ||
                             pipeline->cull_mode == VK_CULL_MODE_NONE);
   for (uint32_t t = 0; t < ntris; t++) {
      uint32_t a = t*3+0, b = t*3+1, c = t*3+2;
      const float *p0 = pos + a*3, *p1 = pos + b*3, *p2 = pos + c*3;
      float area = tri_signed_area(p0, p1, p2);
      /* area >= 0 → CW in Y-down → Borg culls → reverse to CCW.
       * area <  0 → CCW in Y-down → Borg keeps → pass through. */
      bool reversed = (apply_winding_fix && area >= 0.0f);
      if (reversed) {
         idx[t*3+0] = a; idx[t*3+1] = c; idx[t*3+2] = b;
      } else {
         idx[t*3+0] = a; idx[t*3+1] = b; idx[t*3+2] = c;
      }
      if (getenv("BORGVK_DEBUG"))
         fprintf(stderr, "[borgvk_sim] tri[%u] area=%.4f reversed=%d (apply_fix=%d)\n",
                 t, area, reversed, apply_winding_fix);
   }

   /* Write geometry file. */
   char geom_path[] = "/tmp/borgvk_geom_XXXXXX";
   int geom_fd = mkstemp(geom_path);
   if (geom_fd < 0) {
      free(pos); free(col); free(alpha_v); free(idx);
      return VK_SUCCESS;
   }

   uint32_t magic = GEOM_MAGIC;
   float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
   ssize_t nw;
   nw  = write(geom_fd, &magic,    4);
   nw += write(geom_fd, &nverts,   4);
   nw += write(geom_fd, &ntris,    4);
   nw += write(geom_fd, identity,  sizeof(identity));
   nw += write(geom_fd, pos,       nverts * 3 * 4);
   nw += write(geom_fd, col,       nverts * 3 * 4);
   nw += write(geom_fd, idx,       ntris  * 3 * 4);
   (void)nw;
   close(geom_fd);
   free(col); col = NULL;   /* col not needed beyond this point; keep pos/idx/alpha_v */

   /* Fork arcilator_sim --cts-draw, capture stdout = raw RGB888. */
   char w_str[16], h_str[16];
   snprintf(w_str, sizeof(w_str), "%u", width);
   snprintf(h_str, sizeof(h_str), "%u", height);

   int pfd[2];
   if (pipe(pfd) < 0) {
      unlink(geom_path);
      return VK_SUCCESS;
   }

   pid_t pid = fork();
   if (pid == 0) {
      close(pfd[0]);
      dup2(pfd[1], STDOUT_FILENO);
      close(pfd[1]);
      execlp(sim_bin, sim_bin, "--cts-draw", geom_path, sim_fw,
             w_str, h_str, (char *)NULL);
      _exit(127);
   }
   close(pfd[1]);

   /* Read RGB888 pixels from pipe. */
   size_t expected = (size_t)width * height * 3;
   uint8_t *rgb = malloc(expected);
   size_t got = 0;
   if (rgb) {
      while (got < expected) {
         ssize_t n = read(pfd[0], rgb + got, expected - got);
         if (n <= 0) break;
         got += (size_t)n;
      }
   }
   close(pfd[0]);
   int wstatus = 0;
   waitpid(pid, &wstatus, 0);
   unlink(geom_path);

   /* Convert RGB888 → R8G8B8A8_UNORM and write into the image's backing store.
    * Image layout is linear: row-major, 4 bytes/pixel (R, G, B, A).
    * Alpha is NOT stored in the Borg TBR (RGB565), so we compute it on the
    * host: for each pixel, test membership in each triangle and interpolate
    * the per-vertex alpha from the CTS vertex buffer. Background pixels keep
    * A=255 (clear alpha). */
   if (rgb && got == expected && color_img->mem && color_img->mem->map) {
      uint8_t *dst = (uint8_t *)color_img->mem->map + color_img->offset;
      /* CTS clear color is (0,0,0,1) → RGBA=(0,0,0,255). */
      for (size_t i = 0; i < (size_t)width * height; i++) {
         dst[i*4+0] = rgb[i*3+0];
         dst[i*4+1] = rgb[i*3+1];
         dst[i*4+2] = rgb[i*3+2];

         /* Pixel centre in NDC: x∈[-1,+1], y∈[-1,+1] (Y-down). */
         uint32_t px = (uint32_t)(i % width);
         uint32_t py = (uint32_t)(i / width);
         float nx = ((float)px + 0.5f) / (float)width  * 2.0f - 1.0f;
         float ny = ((float)py + 0.5f) / (float)height * 2.0f - 1.0f;

         float alpha = 1.0f;  /* default: clear A=1 */
         for (uint32_t t = 0; t < ntris; t++) {
            uint32_t ia = idx[t*3+0], ib = idx[t*3+1], ic = idx[t*3+2];
            float a = tri_interp_scalar(nx, ny,
                                        pos + ia*3, pos + ib*3, pos + ic*3,
                                        alpha_v[ia], alpha_v[ib], alpha_v[ic]);
            if (a >= 0.0f) { alpha = a; break; }  /* first triangle wins */
         }
         dst[i*4+3] = (uint8_t)(alpha * 255.0f + 0.5f);
      }
   }
   free(rgb);
   free(pos); free(idx); free(alpha_v);

   return VK_SUCCESS;
}

/* Locate the bound descriptor set (binding 0 = UBO, binding 1 = texture). */
static struct borgvk_descriptor_set *
find_set(struct vk_queue_submit *submit)
{
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct vk_command_buffer *cb = submit->command_buffers[i];
      list_for_each_entry(struct vk_cmd_queue_entry, e, &cb->cmd_queue.cmds, cmd_link) {
         if (e->type != VK_CMD_BIND_DESCRIPTOR_SETS)
            continue;
         const struct vk_cmd_bind_descriptor_sets *b = &e->u.bind_descriptor_sets;
         if (b->descriptor_set_count == 0 || b->descriptor_sets == NULL)
            continue;
         VK_FROM_HANDLE(borgvk_descriptor_set, set, b->descriptor_sets[0]);
         if (set)
            return set;
      }
   }
   return NULL;
}

/* Locate the first render-pass colour attachment image (for sim readback). */
static struct borgvk_image *
find_color_attachment(struct vk_queue_submit *submit)
{
   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      list_for_each_entry(struct vk_cmd_queue_entry, e, &cb->cmd_queue.cmds, cmd_link) {
         if (e->type != VK_CMD_BEGIN_RENDER_PASS)
            continue;
         const VkRenderPassBeginInfo *rp = e->u.begin_render_pass.render_pass_begin;
         VK_FROM_HANDLE(vk_framebuffer, fb, rp->framebuffer);
         if (fb && fb->attachment_count > 0) {
            VK_FROM_HANDLE(vk_image_view, view, fb->attachments[0]);
            if (view && view->image)
               return container_of(view->image, struct borgvk_image, vk);
         }
      }
   }
   return NULL;
}

/* Dedup the 36 expanded positions to unique corners; build the indexed triangle
 * list + per-tri-vertex UVs, then ship the mesh. */
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

/* Box-downsample the linear RGBA8 texture to one BORGVK_TEX_DIM-wide row of
 * normalized RGB floats and ship it. */
static void
send_texture_row(struct borgvk_image *tex, int dy)
{
   const uint8_t *base = (const uint8_t *)tex->mem->map + tex->offset;
   uint32_t sw = tex->vk.extent.width, sh = tex->vk.extent.height;
   if (sw == 0 || sh == 0)
      return;
   uint32_t pitch = sw * 4;                 /* RGBA8, linear tiling */
   int sxs = (int)(sw / BORGVK_TEX_DIM); if (sxs < 1) sxs = 1;
   int sys = (int)(sh / BORGVK_TEX_DIM); if (sys < 1) sys = 1;

   float row[BORGVK_TEX_DIM * 3];
   for (int dx = 0; dx < BORGVK_TEX_DIM; dx++) {
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int oy = 0; oy < sys; oy++) {
         uint32_t sy = (uint32_t)dy * sys + oy;
         if (sy >= sh) break;
         for (int ox = 0; ox < sxs; ox++) {
            uint32_t sx = (uint32_t)dx * sxs + ox;
            if (sx >= sw) break;
            const uint8_t *p = base + sy * pitch + sx * 4;
            r += p[0]; g += p[1]; b += p[2]; n++;
         }
      }
      if (n == 0) n = 1;
      row[dx*3+0] = (float)r / n / 255.0f;
      row[dx*3+1] = (float)g / n / 255.0f;
      row[dx*3+2] = (float)b / n / 255.0f;
   }
   borgvk_serial_send_tex_row(dy, row);
}

/* Sentinel path: /tmp/borgvk_<devbasename>_setup — presence means the FPGA
 * already holds the current session's geometry+texture in PSRAM. */
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

/* Ship the borgc-compiled shader blobs (captured at pipeline creation) to the
 * firmware once, as part of setup. DRM path → inline ioctl (the shim/kernel owns
 * the serial port); serial fallback → direct send. The firmware stages each blob
 * into PSRAM in place of its baked borgc_vert/frag_shader[] array. */
static void
upload_shaders(struct borgvk_device *device)
{
   for (int st = 0; st < BORGVK_SHADER_STAGE_COUNT; st++) {
      struct borgvk_shader_blob *b = &device->shader_blob[st];
      if (b->len == 0)
         continue;   /* stage not compiled (app didn't bind it) */
      if (device->drm_fd >= 0) {
         struct drm_borg_shader s = { .stage = (uint32_t)st, .len = b->len };
         memcpy(s.data, b->data, b->len);
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SHADER, &s) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_SHADER failed (stage %d)", st);
      } else {
         borgvk_serial_send_shader((uint8_t)st, b->data, b->len);
      }
   }
}

/* True when this submit is cube.c's UBO-driven frame (vs a CTS VBO draw). */
static bool
submit_is_cube(struct vk_queue_submit *submit)
{
   struct borgvk_descriptor_set *set = find_set(submit);
   if (!set)
      return false;
   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return false;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return false;
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));
   return nfloats >= UBO_MIN_FLOATS;
}

/* Sim path for cube.c: capture the EXACT serial byte stream borgvk would put on
 * the wire (0xB0 borgc shaders + 0xAE geom + 0xAF texture + 0xAD MVP), feed it to
 * `arcilator_sim --cts-uart`, and write the rendered pixels into the colour
 * attachment.  Unlike the live serial path there is no once-only sentinel: every
 * frame ships the full stream because the sim boots fresh per invocation.  This
 * exercises the same borgc shaders + protocol as the FPGA, with no serial port. */
static VkResult
borgvk_submit_sim_cube(struct borgvk_device *device,
                       struct vk_queue_submit *submit)
{
   const char *sim_bin = getenv("BORGVK_SIM");
   const char *sim_fw  = getenv("BORGVK_SIM_FW");
   if (!sim_bin || !sim_fw)
      return VK_SUCCESS;

   struct borgvk_descriptor_set *set = find_set(submit);
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

   struct borgvk_image *color_img = find_color_attachment(submit);
   if (!color_img || !color_img->mem || !color_img->mem->map)
      return VK_SUCCESS;
   uint32_t width  = color_img->vk.extent.width;
   uint32_t height = color_img->vk.extent.height;
   if (width == 0 || height == 0)
      return VK_SUCCESS;

   /* The firmware renders at its fixed native size (128² fallback for the cube),
    * which need not match the swapchain extent.  Render the sim at that size and
    * nearest-upscale into the attachment.  Override with BORGVK_SIM_DIM. */
   uint32_t sdim = 128;
   const char *dim_env = getenv("BORGVK_SIM_DIM");
   if (dim_env && dim_env[0]) {
      int d = atoi(dim_env);
      if (d > 0)
         sdim = (uint32_t)d;
   }

   /* Capture one frame's full wire stream (shaders + geom + texture + MVP). */
   borgvk_transport_capture_begin();
   upload_shaders(device);
   send_geometry(ubo);
   if (tex && tex->mem && tex->mem->map &&
       tex->vk.extent.width && tex->vk.extent.height)
      for (int row = 0; row < BORGVK_TEX_DIM; row++)
         send_texture_row(tex, row);
   borgvk_serial_send_mvp(ubo);
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   if (!bytes || nbytes == 0) {
      free(bytes);
      return VK_SUCCESS;
   }

   /* Debug bisect: dump the captured wire stream and skip the sim fork.  Lets us
    * verify borgvk produces the correct protocol bytes independent of arcilator.
    *   BORGVK_SIM_DUMP=/path  → write one frame's stream there, then return. */
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
      mesa_logi("borgvk: dumped %zu-byte sim stream to %s (%ux%u)",
                nbytes, dump, width, height);
      free(bytes);
      return VK_SUCCESS;
   }

   /* Hand the byte stream to arcilator_sim --cts-uart via a temp file. */
   char uart_path[] = "/tmp/borgvk_uart_XXXXXX";
   int ufd = mkstemp(uart_path);
   if (ufd < 0) {
      free(bytes);
      return VK_SUCCESS;
   }
   for (size_t w = 0; w < nbytes; ) {
      ssize_t n = write(ufd, bytes + w, nbytes - w);
      if (n <= 0)
         break;
      w += (size_t)n;
   }
   close(ufd);
   free(bytes);

   char w_str[16], h_str[16];
   snprintf(w_str, sizeof(w_str), "%u", sdim);
   snprintf(h_str, sizeof(h_str), "%u", sdim);

   int pfd[2];
   if (pipe(pfd) < 0) {
      unlink(uart_path);
      return VK_SUCCESS;
   }
   pid_t pid = fork();
   if (pid == 0) {
      close(pfd[0]);
      dup2(pfd[1], STDOUT_FILENO);
      close(pfd[1]);
      execlp(sim_bin, sim_bin, "--cts-uart", uart_path, sim_fw,
             w_str, h_str, (char *)NULL);
      _exit(127);
   }
   close(pfd[1]);

   size_t expected = (size_t)sdim * sdim * 3;
   uint8_t *rgb = malloc(expected);
   size_t got = 0;
   if (rgb) {
      while (got < expected) {
         ssize_t n = read(pfd[0], rgb + got, expected - got);
         if (n <= 0)
            break;
         got += (size_t)n;
      }
   }
   close(pfd[0]);
   int wstatus = 0;
   waitpid(pid, &wstatus, 0);
   unlink(uart_path);

   /* RGB888 sdim×sdim (sim stdout) → R8G8B8A8_UNORM attachment (width×height),
    * nearest-neighbour upscale, opaque alpha. */
   if (rgb && got == expected) {
      uint8_t *dst = (uint8_t *)color_img->mem->map + color_img->offset;
      for (uint32_t y = 0; y < height; y++) {
         uint32_t sy = (uint32_t)((uint64_t)y * sdim / height);
         for (uint32_t x = 0; x < width; x++) {
            uint32_t sx = (uint32_t)((uint64_t)x * sdim / width);
            const uint8_t *s = rgb + ((size_t)sy * sdim + sx) * 3;
            uint8_t *d = dst + ((size_t)y * width + x) * 4;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
         }
      }
   }
   free(rgb);
   return VK_SUCCESS;
}

VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct borgvk_device *device =
      container_of(vk_queue->base.device, struct borgvk_device, vk);

   /* Sim path: if BORGVK_SIM is set, render via arcilator_sim and write pixels
    * into the colour attachment instead of shipping to the FPGA.
    *   - cube.c (UBO-driven): borgvk_submit_sim_cube captures the full serial
    *     stream incl. 0xB0 borgc shaders and replays it via --cts-uart, so the
    *     sim runs the exact protocol + real shaders as the FPGA.
    *   - CTS draws (VBO-driven): borgvk_submit_sim_draw, the mailbox --cts-draw
    *     path (carries per-vertex colour; still baked-shader for now). */
   if (getenv("BORGVK_SIM"))
      return submit_is_cube(submit) ? borgvk_submit_sim_cube(device, submit)
                                    : borgvk_submit_sim_draw(submit);

   struct borgvk_descriptor_set *set = find_set(submit);
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
      /* DRM path: delegate geometry/texture upload and per-frame MVP to the
       * shim (or kernel driver) via ioctls.  The shim manages the sentinel
       * and serial transport; we just pass the GEM handles. */
      bool has_gem = ubuf->mem->gem_handle != 0 &&
                     tex && tex->mem && tex->mem->gem_handle != 0;

      if (!g_setup_done && has_gem) {
         upload_shaders(device);   /* borgc-compiled shaders, before geom/tex */
         struct drm_borg_setup s = {
            .ubo_handle = ubuf->mem->gem_handle,
            .tex_handle = tex->mem->gem_handle,
            .tex_offset = (__u64)tex->offset,
            .tex_width  = tex->vk.extent.width,
            .tex_height = tex->vk.extent.height,
         };
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SETUP, &s) == 0)
            g_setup_done = true;
         else
            mesa_logw("borgvk: DRM_IOCTL_BORG_SETUP failed");
      }

      if (g_setup_done) {
         struct drm_borg_submit sub = { .ubo_handle = ubuf->mem->gem_handle };
         drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SUBMIT, &sub);
      }
   } else {
      /* Serial fallback: no DRM device (shim not loaded).  Run the legacy
       * direct-serial path exactly as before. */
      if (!g_setup_done) {
         if (setup_already_done()) {
            mesa_logi("borgvk: skipping upload (sentinel present); "
                      "set BORGVK_FORCE_UPLOAD=1 to re-upload");
            g_setup_done = true;
         } else if (can_geom && can_tex) {
            mesa_logi("borgvk: uploading shaders + geometry + texture burst...");
            upload_shaders(device);   /* borgc-compiled shaders, before geom/tex */
            send_geometry(ubo);
            for (int row = 0; row < BORGVK_TEX_DIM; row++)
               send_texture_row(tex, row);
            mark_setup_done();
            mesa_logi("borgvk: upload complete");
            g_setup_done = true;
         }
      }

      if (g_setup_done)
         borgvk_serial_send_mvp(ubo);
   }

   return VK_SUCCESS;
}
