/*
 * Copyright © 2026 Andreas Wendleder
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
#include "vk_sync.h"
#include "vk_framebuffer.h"
#include "vk_image.h"
#include "vk_format.h"
#include "util/format/u_format.h"

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

   setenv("CTS_MAX_CYCLES", "60000000", 0);
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
      /* Mesa emulates the legacy render-pass entrypoints on top of
       * CmdBeginRendering, which does not enqueue; the command buffer
       * remembers the colour view it last rendered to instead. */
      struct borgvk_command_buffer *bcb =
         container_of(cb, struct borgvk_command_buffer, vk);
      if (bcb->color_views[0] &&
          bcb->color_views[0]->image)
         return container_of(bcb->color_views[0]->image, struct borgvk_image, vk);
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

/* Box-downsample the linear RGBA8 texture to one BORGVK_TEX_DIM-wide RGBA8
 * row and ship it with the sampler's packed descriptor. With no sampler
 * recorded (an immutable one, which borgvk does not track yet) it sends the
 * firmware's own default: nearest, CLAMP_TO_EDGE, LOD clamped to 0. */
static const uint32_t default_sampler[4] = { (2u << 3) | (2u << 6) | (2u << 9), 0, 0, 0 };

static const uint32_t *
sampler_desc(const struct borgvk_sampler *sampler)
{
   return sampler ? sampler->desc : default_sampler;
}

static void
send_texture_row(struct borgvk_image *tex, const struct borgvk_sampler *sampler,
                 int dy)
{
   const uint8_t *base = (const uint8_t *)tex->mem->map + tex->offset;
   uint32_t sw = tex->vk.extent.width, sh = tex->vk.extent.height;
   if (sw == 0 || sh == 0)
      return;
   uint32_t pitch = sw * 4;                 /* RGBA8, linear tiling */
   int sxs = (int)(sw / BORGVK_TEX_DIM); if (sxs < 1) sxs = 1;
   int sys = (int)(sh / BORGVK_TEX_DIM); if (sys < 1) sys = 1;

   uint8_t row[BORGVK_TEX_DIM * 4];
   for (int dx = 0; dx < BORGVK_TEX_DIM; dx++) {
      uint32_t sum[4] = { 0, 0, 0, 0 }, n = 0;
      for (int oy = 0; oy < sys; oy++) {
         uint32_t sy = (uint32_t)dy * sys + oy;
         if (sy >= sh) break;
         for (int ox = 0; ox < sxs; ox++) {
            uint32_t sx = (uint32_t)dx * sxs + ox;
            if (sx >= sw) break;
            const uint8_t *p = base + sy * pitch + sx * 4;
            for (int c = 0; c < 4; c++)
               sum[c] += p[c];
            n++;
         }
      }
      if (n == 0) n = 1;
      for (int c = 0; c < 4; c++)
         row[dx * 4 + c] = (uint8_t)((sum[c] + n / 2) / n);
   }
   borgvk_serial_send_tex_row(dy, row, sampler_desc(sampler));
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

/* Ship the pipeline's colour-blend state (BLEND_CFG/BLEND_CONST). blend_cfg is
 * never 0 once a graphics pipeline has set it (the write mask resets to 0xF),
 * so 0 means "no pipeline yet": leave the hardware at its reset state. */
static void
send_blend_state(const struct borgvk_device *device)
{
   if (device->state_valid) {
      if (device->drm_fd >= 0) {
         struct drm_borg_state s;
         memcpy(s.reg, device->state_reg, sizeof(s.reg));
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_STATE, &s) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_STATE failed");
      } else {
         borgvk_serial_send_state(device->state_reg);
      }
   }
   if (!device->blend_cfg)
      return;
   if (device->drm_fd >= 0) {
      struct drm_borg_blend b = { .cfg = device->blend_cfg,
                                  .constant = device->blend_const };
      if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_BLEND, &b) != 0)
         mesa_logw("borgvk: DRM_IOCTL_BORG_BLEND failed");
   } else {
      borgvk_serial_send_blend(device->blend_cfg, device->blend_const);
   }
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

/* True when this descriptor set is cube.c's UBO-driven frame (vs a CTS VBO draw). */
static bool
set_is_cube(struct borgvk_descriptor_set *set)
{
   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return false;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return false;
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));
   return nfloats >= UBO_MIN_FLOATS;
}

static bool
submit_is_cube(struct vk_queue_submit *submit)
{
   struct borgvk_descriptor_set *set = find_set(submit);
   return set && set_is_cube(set);
}

/* Feed a captured wire stream to `arcilator_sim --cts-uart` and write the
 * rendered RGB888 pixels into the colour attachment (nearest-neighbour scaled,
 * opaque alpha). `sim_dim` = 0 uses the vkcube default (128, BORGVK_SIM_DIM);
 * otherwise the simulator renders sim_dim x sim_dim. Takes ownership of `bytes`. */
static VkResult
sim_run_stream(uint8_t *bytes, size_t nbytes, struct borgvk_image *color_img,
               uint32_t sim_dim)
{
   const char *sim_bin = getenv("BORGVK_SIM");
   const char *sim_fw  = getenv("BORGVK_SIM_FW");
   if (!color_img || !color_img->mem || !color_img->mem->map)
      { free(bytes); return VK_SUCCESS; }
   uint32_t width  = color_img->vk.extent.width;
   uint32_t height = color_img->vk.extent.height;
   if (width == 0 || height == 0)
      { free(bytes); return VK_SUCCESS; }

   /* The firmware renders at its fixed native size (128² fallback for the cube),
    * which need not match the swapchain extent.  Render the sim at that size and
    * nearest-upscale into the attachment.  Override with BORGVK_SIM_DIM. */
   uint32_t sdim = sim_dim ? sim_dim : 128;
   /* A RAW32 attachment (R32_UINT) comes back as its raw words and is never resampled. */
   const bool raw32 = color_img->vk.format == VK_FORMAT_R32_UINT && width == height;
   if (raw32)
      sdim = width;
   /* BORGVK_SIM_DIRECT=<direct_sim>: Borg alone driven by a host-side driver
    * (simulation/direct) -- no firmware, so the target size is free (power of
    * two, 4..256, square) and a draw takes well under a second. */
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (direct && direct[0]) {
      uint32_t d = color_img->vk.extent.width;
      if (d >= 4 && d <= 256 && (d & (d - 1)) == 0 &&
          color_img->vk.extent.height == d)
         sdim = d;
   }
   const char *dim_env = getenv("BORGVK_SIM_DIM");
   if (!sim_dim && dim_env && dim_env[0]) {
      int d = atoi(dim_env);
      if (d > 0)
         sdim = (uint32_t)d;
   }

   /* BORGVK_SIM_KEEP=<path>: also keep the stream, for replaying it by hand
    * with scripts/cts_uart_render.py. */
   const char *keep = getenv("BORGVK_SIM_KEEP");
   if (keep && keep[0]) {
      int kfd = open(keep, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (kfd >= 0) {
         for (size_t w = 0; w < nbytes; ) {
            ssize_t n = write(kfd, bytes + w, nbytes - w);
            if (n <= 0) break;
            w += (size_t)n;
         }
         close(kfd);
      }
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

   /* A target over 128 pixels is rendered as 128-pixel windows (the hardware draws a large
    * framebuffer as several windows, docs/B1_geometry_front_end.md); the windows are
    * independent, so with the direct simulator they are dealt out to one process per core
    * and the pieces merged. Window w (row-major) belongs to process w % nparts. */
   const uint32_t WIN = 128;
   uint32_t nwx = sdim > WIN ? sdim / WIN : 1;
   uint32_t nparts = 1;
   if (direct && direct[0] && nwx > 1) {
      long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
      nparts = (uint32_t)(ncpu > 1 ? ncpu : 1);
      if (nparts > nwx * nwx)
         nparts = nwx * nwx;
   }

   const size_t bpp = raw32 ? 4 : 3;
   size_t expected = (size_t)sdim * sdim * bpp;
   uint8_t *rgb = malloc(expected);
   size_t got = 0;
   int pfds[64][2];
   pid_t pids[64];
   if (nparts > 64)
      nparts = 64;
   for (uint32_t k = 0; k < nparts; k++) {
      pids[k] = -1;
      if (pipe(pfds[k]) < 0)
         continue;
      pid_t pid = fork();
      if (pid == 0) {
         close(pfds[k][0]);
         dup2(pfds[k][1], STDOUT_FILENO);
         close(pfds[k][1]);
         if (direct && direct[0]) {
            if (nwx > 1) {
               char part[32];
               snprintf(part, sizeof(part), "%u/%u", k, nparts);
               setenv("BORG_PART", part, 1);
               setenv("BORG_WINDOW_TILES", "32", 1);
            }
            execlp(direct, direct, uart_path, w_str, h_str, (char *)NULL);
         }
         execlp(sim_bin, sim_bin, "--cts-uart", uart_path, sim_fw,
                w_str, h_str, (char *)NULL);
         _exit(127);
      }
      close(pfds[k][1]);
      pids[k] = pid;
   }
   uint8_t *piece = nparts > 1 ? malloc(expected) : NULL;
   uint32_t pieces_ok = 0;
   for (uint32_t k = 0; k < nparts && rgb; k++) {
      if (pids[k] < 0)
         continue;
      uint8_t *dst = nparts > 1 ? piece : rgb;
      size_t n_got = 0;
      while (n_got < expected) {
         ssize_t n = read(pfds[k][0], dst + n_got, expected - n_got);
         if (n <= 0)
            break;
         n_got += (size_t)n;
      }
      close(pfds[k][0]);
      int wstatus = 0;
      waitpid(pids[k], &wstatus, 0);
      if (nparts > 1) {
         if (n_got == expected) {
            for (uint32_t y = 0; y < sdim; y++)
               for (uint32_t x = 0; x < sdim; x++)
                  if (((y / WIN) * nwx + x / WIN) % nparts == k)
                     memcpy(rgb + ((size_t)y * sdim + x) * bpp,
                            piece + ((size_t)y * sdim + x) * bpp, bpp);
            pieces_ok++;
         }
      } else {
         got = n_got;
      }
   }
   if (nparts > 1)
      got = pieces_ok == nparts ? expected : 0;
   free(piece);
   unlink(uart_path);
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: sim returned %zu of %zu bytes", got, expected);

   /* RGB888 sdim×sdim (sim stdout) → R8G8B8A8_UNORM attachment (width×height),
    * nearest-neighbour upscale, opaque alpha. */
   /* The write below assumes a 4-byte texel; an attachment of another size (or unmapped
    * memory) must not be written past its backing. */
   bool fits = color_img->mem && color_img->mem->map &&
               (uint64_t)width * height * 4 * MAX2(color_img->vk.samples, 1) <= color_img->size;
   if (rgb && got == expected && !fits) {
      /* leave the image untouched */
   } else if (rgb && got == expected && raw32) {
      memcpy((uint8_t *)color_img->mem->map + color_img->offset, rgb, expected);
   } else if (rgb && got == expected) {
      /* The sim returns the resolved pixel. A multisampled image packs one plane per
       * sample (see borgvk_image_layer_size): put the resolved value in every plane,
       * so a later vkCmdResolveImage averages it back to itself. */
      uint64_t plane_size = (uint64_t)width * height * 4;
      for (uint32_t smp = 0; smp < MAX2(color_img->vk.samples, 1); smp++) {
         uint8_t *dst = (uint8_t *)color_img->mem->map + color_img->offset + smp * plane_size;
         for (uint32_t y = 0; y < height; y++) {
            uint32_t sy = (uint32_t)((uint64_t)(2 * y + 1) * sdim / (2 * (uint64_t)height));
            for (uint32_t x = 0; x < width; x++) {
               uint32_t sx = (uint32_t)((uint64_t)(2 * x + 1) * sdim / (2 * (uint64_t)width));
               const uint8_t *s = rgb + ((size_t)sy * sdim + sx) * 3;
               uint8_t *d = dst + ((size_t)y * width + x) * 4;
               d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
            }
         }
      }
   }
   free(rgb);
   return VK_SUCCESS;
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

   /* Capture one frame's full wire stream (shaders + geom + texture + MVP). */
   borgvk_transport_capture_begin();
   upload_shaders(device);
   send_geometry(ubo);
   send_blend_state(device);
   if (tex && tex->mem && tex->mem->map &&
       tex->vk.extent.width && tex->vk.extent.height)
      for (int row = 0; row < BORGVK_TEX_DIM; row++)
         send_texture_row(tex, set->samplers[1], row);
   borgvk_serial_send_mvp(ubo);
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   if (!bytes || nbytes == 0) {
      free(bytes);
      return VK_SUCCESS;
   }

   /* Dump the captured wire stream and skip the sim fork -- how the simulation
    * goldens' borgvk_capture.bin is produced. Needs no mappable colour
    * attachment, so it works against a real WSI swapchain.
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
      mesa_logi("borgvk: dumped %zu-byte sim stream to %s", nbytes, dump);
      free(bytes);
      return VK_SUCCESS;
   }

   struct borgvk_image *color_img = find_color_attachment(submit);
   return sim_run_stream(bytes, nbytes, color_img, 0);
}

/* Convert a sampled image to texels the texture unit reads (RGBA8, linear,
 * one level) and ship it as 0xB5 chunks. Returns false for formats not handled
 * yet, in which case nothing is sent. */
static bool
send_generic_texture(const struct vk_image_view *view,
                     const struct borgvk_image *img,
                     const struct borgvk_sampler *sampler)
{
   if (!img || !img->mem || !img->mem->map)
      return false;
   uint32_t w = img->vk.extent.width, h = img->vk.extent.height;
   if (w == 0 || h == 0 || w > 4096 || h > 4096)
      return false;
   uint32_t type;
   switch (view->view_type) {
   case VK_IMAGE_VIEW_TYPE_1D: type = 0; h = 1; break;
   case VK_IMAGE_VIEW_TYPE_2D: type = 1; break;
   default: return false;
   }

   /* source texel layout -> RGBA8 + TexFormat code (borg_isa.h) */
   uint32_t bpp, fmt_code;
   bool swap_rb = false, alpha_fill = false, raw = false;
   uint8_t alpha_val = 0;
   switch (view->format) {
#define RAWFMT(vk, code, bytes) case VK_FORMAT_##vk: bpp = bytes; fmt_code = code; raw = true; break;
   RAWFMT(R16_UNORM, 18, 2) RAWFMT(R16_SNORM, 19, 2) RAWFMT(R16_UINT, 20, 2) RAWFMT(R16_SINT, 21, 2)
   RAWFMT(R16_SFLOAT, 22, 2) RAWFMT(R16G16_UNORM, 23, 4) RAWFMT(R16G16_SNORM, 24, 4)
   RAWFMT(R16G16_UINT, 25, 4) RAWFMT(R16G16_SINT, 26, 4) RAWFMT(R16G16_SFLOAT, 27, 4)
   RAWFMT(R16G16B16A16_UNORM, 28, 8) RAWFMT(R16G16B16A16_SNORM, 29, 8) RAWFMT(R16G16B16A16_UINT, 30, 8)
   RAWFMT(R16G16B16A16_SINT, 31, 8) RAWFMT(R16G16B16A16_SFLOAT, 32, 8)
   RAWFMT(R32_UINT, 33, 4) RAWFMT(R32_SINT, 34, 4) RAWFMT(R32_SFLOAT, 35, 4)
   RAWFMT(R32G32_UINT, 36, 8) RAWFMT(R32G32_SINT, 37, 8) RAWFMT(R32G32_SFLOAT, 38, 8)
   RAWFMT(R32G32B32A32_UINT, 39, 16) RAWFMT(R32G32B32A32_SINT, 40, 16) RAWFMT(R32G32B32A32_SFLOAT, 41, 16)
   RAWFMT(A2B10G10R10_UNORM_PACK32, 42, 4) RAWFMT(A2B10G10R10_UINT_PACK32, 43, 4)
   RAWFMT(R5G6B5_UNORM_PACK16, 44, 2) RAWFMT(A1R5G5B5_UNORM_PACK16, 45, 2)
   RAWFMT(B4G4R4A4_UNORM_PACK16, 46, 2) RAWFMT(B10G11R11_UFLOAT_PACK32, 47, 4)
   RAWFMT(E5B9G9R9_UFLOAT_PACK32, 48, 4)
#undef RAWFMT
   case VK_FORMAT_R8G8B8A8_UNORM: bpp = 4; fmt_code = 11; swap_rb = false; alpha_fill = false; alpha_val = 0; break;
   case VK_FORMAT_R8G8B8A8_SNORM: bpp = 4; fmt_code = 12; swap_rb = false; alpha_fill = false; alpha_val = 0; break;
   case VK_FORMAT_B8G8R8A8_UNORM: bpp = 4; fmt_code = 11; swap_rb = true;  alpha_fill = false; alpha_val = 0; break;
   case VK_FORMAT_B8G8R8A8_SNORM: bpp = 4; fmt_code = 12; swap_rb = true;  alpha_fill = false; alpha_val = 0; break;
   case VK_FORMAT_R8G8B8_UNORM:   bpp = 3; fmt_code = 11; swap_rb = false; alpha_fill = true;  alpha_val = 255; break;
   case VK_FORMAT_R8G8B8_SNORM:   bpp = 3; fmt_code = 12; swap_rb = false; alpha_fill = true;  alpha_val = 127; break;
   case VK_FORMAT_B8G8R8_UNORM:   bpp = 3; fmt_code = 11; swap_rb = true;  alpha_fill = true;  alpha_val = 255; break;
   case VK_FORMAT_B8G8R8_SNORM:   bpp = 3; fmt_code = 12; swap_rb = true;  alpha_fill = true;  alpha_val = 127; break;
   default: return false;
   }
   uint32_t total = raw ? w * h * bpp : w * h * 4;
   uint8_t *rgba = malloc(total);
   if (!rgba)
      return false;
   const uint8_t *src = (const uint8_t *)img->mem->map + img->offset;
   if (raw)
      memcpy(rgba, src, total);
   for (uint32_t i = 0; !raw && i < w * h; i++) {
      const uint8_t *p = src + (size_t)i * bpp;
      uint8_t *d = rgba + (size_t)i * 4;
      d[0] = swap_rb ? p[2] : p[0];
      d[1] = p[1];
      d[2] = swap_rb ? p[0] : p[2];
      d[3] = alpha_fill ? alpha_val : p[3];
   }

   /* descriptor words 1..3: linear layout, one level, view swizzle (the
    * VkComponentSwizzle values are the hardware's own encoding). */
   uint32_t desc[3];
   desc[0] = (w - 1) | ((h - 1) << 16) | (type << 28) | (1u << 30);
   desc[1] = (fmt_code << 14) |
             ((uint32_t)(view->swizzle.r & 7) << 20) |
             ((uint32_t)(view->swizzle.g & 7) << 23) |
             ((uint32_t)(view->swizzle.b & 7) << 26) |
             ((uint32_t)(view->swizzle.a & 7) << 29);
   desc[2] = w * (raw ? bpp : 4);
   for (uint32_t off = 0; off < total; off += BORGVK_TEXG_CHUNK) {
      uint32_t n = total - off < BORGVK_TEXG_CHUNK ? total - off : BORGVK_TEXG_CHUNK;
      borgvk_serial_send_texture_chunk(off, rgba + off, n, desc, sampler_desc(sampler));
   }
   free(rgba);
   return true;
}

/* A texel buffer as a 1D texture of the same texels (linear layout, one level), so a
 * texelFetch reads it through the texture unit. Only 32-bit single-channel formats so far. */
static bool
send_buffer_texture(const struct vk_buffer_view *bv)
{
   uint32_t fmt_code;
   switch (bv->format) {
   case VK_FORMAT_R32_UINT:   fmt_code = 33; break;   /* BORG_TEX_FORMAT_R32_UINT */
   case VK_FORMAT_R32_SFLOAT: fmt_code = 35; break;
   default: return false;
   }
   struct borgvk_buffer *buf = container_of(bv->buffer, struct borgvk_buffer, vk);
   if (!buf->mem || !buf->mem->map)
      return false;
   uint32_t w = (uint32_t)(bv->range / 4);
   if (w == 0 || w > 65536)
      return false;
   uint32_t total = w * 4;
   const uint8_t *src = (const uint8_t *)buf->mem->map + buf->offset + bv->offset;
   uint32_t desc[3];
   desc[0] = (w - 1) | (0u << 28) | (1u << 30);       /* 1D, linear */
   desc[1] = fmt_code << 14;                          /* identity swizzle */
   desc[2] = total;
   for (uint32_t off = 0; off < total; off += BORGVK_TEXG_CHUNK) {
      uint32_t n = total - off < BORGVK_TEXG_CHUNK ? total - off : BORGVK_TEXG_CHUNK;
      borgvk_serial_send_texture_chunk(off, src + off, n, desc, sampler_desc(NULL));
   }
   return true;
}

/* Read one float attribute of one vertex (R32..R32G32B32A32_SFLOAT only).
 * Missing components follow the Vulkan defaults: 0, 0, 0, 1. */
static bool
fetch_vertex_attr(const struct borgvk_pipeline *pl,
                  const struct borgvk_vertex_attr *a,
                  const uint8_t *const vb[BORGVK_MAX_VERTEX_BINDINGS],
                  const VkDeviceSize vb_avail[BORGVK_MAX_VERTEX_BINDINGS],
                  uint32_t vertex, float out[4])
{
   uint32_t n;
   switch (a->format) {
   case VK_FORMAT_R32_SFLOAT:          n = 1; break;
   case VK_FORMAT_R32G32_SFLOAT:       n = 2; break;
   case VK_FORMAT_R32G32B32_SFLOAT:    n = 3; break;
   case VK_FORMAT_R32G32B32A32_SFLOAT: n = 4; break;
   default: return false;
   }
   if (a->binding >= BORGVK_MAX_VERTEX_BINDINGS || !vb[a->binding])
      return false;
   const uint64_t at = (uint64_t)vertex * pl->binding_stride[a->binding] + a->offset;
   out[0] = out[1] = out[2] = 0.0f;
   out[3] = 1.0f;
   /* robustBufferAccess: a fetch past the end of the bound range reads zero. */
   if (at + n * sizeof(float) > vb_avail[a->binding]) {
      out[3] = 0.0f;
      return true;
   }
   memcpy(out, vb[a->binding] + at, n * sizeof(float));
   return true;
}

/* Sim path for a generic draw (CTS): one graphics pipeline, vertex buffers
 * described by its vertex-input state, location 0 = position (clip space,
 * w taken as 1), location 1 = texture coordinate, a TRIANGLE_LIST draw and,
 * optionally, a sampled image at some descriptor binding. The draw goes through
 * the same wire protocol as vkcube (shaders, geometry, identity MVP, state),
 * so it runs on the real firmware and hardware model.
 * Returns true when it handled the submit. */
static bool
generic_reject(int why)
{
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: generic sim draw rejected (reason %d)", why);
   return false;
}

static bool
borgvk_sim_generic_draw(struct borgvk_device *device, struct borgvk_command_buffer *cmd,
                        uint32_t vert_count, uint32_t first_vert)
{
   /* State as the replay left it: the bound pipeline, vertex buffers and (optionally, for a
    * texture) descriptor set. */
   struct borgvk_pipeline *pipeline = cmd->gfx_pipeline;
   const uint8_t *const *vb = cmd->vb;
   struct borgvk_descriptor_set *set = cmd->desc_set;

   if (!pipeline || vert_count == 0 || vert_count % 3 != 0 ||
       pipeline->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
       vert_count / 3 > BORGVK_GEOM_MAX_TRIS)
      return generic_reject(3);

   const struct borgvk_vertex_attr *a_pos = NULL, *a_uv = NULL;
   for (uint32_t i = 0; i < pipeline->attr_count; i++) {
      if (pipeline->attrs[i].location == 0) a_pos = &pipeline->attrs[i];
      if (pipeline->attrs[i].location == 1) a_uv  = &pipeline->attrs[i];
   }
   if (!a_pos)
      return generic_reject(4);

   struct borgvk_image *color_img =
      cmd->color_views[0] && cmd->color_views[0]->image
         ? container_of(cmd->color_views[0]->image, struct borgvk_image, vk) : NULL;
   if (!color_img || !color_img->mem || !color_img->mem->map)
      return generic_reject(5);

   /* Dedup positions into the packet's shared vertex table, keep UVs per
    * triangle corner (the packet carries them that way). */
   float verts[BORGVK_GEOM_MAX_VERTS * 3];
   uint8_t idx[BORGVK_GEOM_MAX_TRIS * 3];
   float uv[BORGVK_GEOM_MAX_TRIS * 3 * 2];
   float attr4[BORGVK_GEOM_MAX_TRIS * 3 * 4];
   int nverts = 0;
   for (uint32_t i = 0; i < vert_count; i++) {
      float pos[4], tc[4] = { 0, 0, 0, 1 };
      if (!fetch_vertex_attr(pipeline, a_pos, vb, cmd->vb_avail, first_vert + i, pos))
         return generic_reject(6);
      if (a_uv && !fetch_vertex_attr(pipeline, a_uv, vb, cmd->vb_avail, first_vert + i, tc))
         return generic_reject(7);
      int u = -1;
      for (int j = 0; j < nverts; j++)
         if (verts[j*3+0] == pos[0] && verts[j*3+1] == pos[1] && verts[j*3+2] == pos[2]) {
            u = j;
            break;
         }
      if (u < 0) {
         if (nverts >= BORGVK_GEOM_MAX_VERTS)
            return generic_reject(8);
         u = nverts++;
         memcpy(&verts[u*3], pos, 3 * sizeof(float));
      }
      idx[i] = (uint8_t)u;
      uv[i*2+0] = tc[0];
      uv[i*2+1] = tc[1];
      memcpy(&attr4[i*4], tc, sizeof(tc));
   }

   static const float identity[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

   /* The DRM shim's ioctls bypass the capture buffer; force the serial path. */
   int saved_fd = device->drm_fd;
   device->drm_fd = -1;
   borgvk_transport_capture_begin();
   upload_shaders(device);
   send_blend_state(device);
   {
      /* Render target: the attachment's format and the clear colour its render pass asked for. */
      uint8_t fmt = color_img->vk.format == VK_FORMAT_R8G8B8A8_UNORM ? 1 :
                    color_img->vk.format == VK_FORMAT_B8G8R8A8_UNORM ? 2 :
                    color_img->vk.format == VK_FORMAT_R32_UINT ? 3 : 0;   /* 3 = RAW32 */
      float clear[4] = { 0, 0, 0, 0 };
      if (cmd->has_clear)
         memcpy(clear, cmd->clear_color, sizeof(clear));
      borgvk_serial_send_target(fmt, clear);
   }
   borgvk_serial_send_geom(verts, nverts, idx, uv, (int)(vert_count / 3));
   if (a_uv && (a_uv->format == VK_FORMAT_R32G32B32_SFLOAT || a_uv->format == VK_FORMAT_R32G32B32A32_SFLOAT))
      borgvk_serial_send_attr4(attr4, (int)vert_count);
   for (int b = 0; set && b < BORGVK_MAX_BINDINGS; b++) {
      if (set->views[b] && send_generic_texture(set->views[b], set->images[b], set->samplers[b]))
         break;
      if (set->buffer_views[b] && send_buffer_texture(set->buffer_views[b]))
         break;
   }
   borgvk_serial_send_mvp(identity);
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   device->drm_fd = saved_fd;
   if (!bytes || nbytes == 0) {
      free(bytes);
      return true;
   }
   for (uint32_t k = 0; k < pipeline->attr_count; k++)
      mesa_logi("borgvk: attr loc %u fmt %d off %u bind %u", pipeline->attrs[k].location, pipeline->attrs[k].format, pipeline->attrs[k].offset, pipeline->attrs[k].binding);
   mesa_logi("borgvk: generic sim draw: %u verts -> %d unique, %zu byte stream, target %ux%u",
             vert_count, nverts, nbytes, color_img->vk.extent.width,
             color_img->vk.extent.height);
   sim_run_stream(bytes, nbytes, color_img, 0);   /* firmware renders 128^2; resample to the target */
   return true;
}

/* ---- Draw state and the draw itself, run in submission order -------------------------------- *
 * The application-facing table only records these (see borgvk_CreateDevice); cmd_dispatch runs
 * them when the command buffer is submitted. Tracking the state here, instead of scanning the
 * queue afterwards, is what lets a draw happen at its place in the command stream: a
 * vkCmdCopyImageToBuffer recorded after it sees the rendered image. */
VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                       VkPipeline _pipeline)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   VK_FROM_HANDLE(borgvk_pipeline, pl, _pipeline);
   if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
      cmd->gfx_pipeline = pl;
   else if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
      cmd->cs_pipeline = pl;
}

/* Compute (simulator only): the compute shader's buffers live in fixed windows of GPU memory,
 * one per binding, which the host fills before the dispatch and reads back afterwards (the
 * addresses are baked into the program by borgc's compute backend, see compute.rs). The job
 * goes to `direct_sim --compute`. Runs when the recorded commands replay at submit, so it
 * lands in command order. */
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
   struct borgvk_descriptor_set *set = cmd->desc_set;

   /* wb: copy the window back to the app's buffer afterwards.  A texel buffer is decoded into a
    * private array (the canonical 4-word texel the compiler reads, see borg_nir_passes.c) and
    * never written back. */
   struct { uint8_t *host; uint8_t *decoded; bool wb; uint32_t skip, size, addr; } bufs[BORGVK_MAX_BINDINGS + 1];
   uint32_t nbufs = 0, total = 0;
   for (uint32_t b = 0; set && b < BORGVK_MAX_BINDINGS; b++) {
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
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = n * 16;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + b * BORGVK_CS_SLOT_BYTES;
         total += 8 + n * 16;
         nbufs++;
         continue;
      }
      struct borgvk_buffer *buf = set->buffers[b];
      struct borgvk_image *simg = set->images[b];
      if (!buf && !bv && simg && simg->mem && simg->mem->map && simg->vk.image_type == VK_IMAGE_TYPE_2D &&
          vk_format_get_blocksize(simg->vk.format) == 4 && simg->vk.samples == 1) {
         /* A storage image: word 0 = width, then level 0 / layer 0 texel by texel (4-byte
          * formats; the compiler addresses it as y * width + x + 1). */
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
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + b * BORGVK_CS_SLOT_BYTES;
         total += 8 + (1 + n) * 4;
         nbufs++;
         continue;
      }
      if (!buf || !buf->mem || !buf->mem->map)
         continue;
      VkDeviceSize boff = set->offsets[b] + cmd->dyn_off[b];
      VkDeviceSize avail = buf->vk.size > boff ? buf->vk.size - boff : 0;
      VkDeviceSize size = set->ranges[b] == VK_WHOLE_SIZE || set->ranges[b] > avail ? avail : set->ranges[b];
      if (size > BORGVK_CS_SLOT_BYTES)
         size = BORGVK_CS_SLOT_BYTES;
      bufs[nbufs].host = (uint8_t *)buf->mem->map + buf->offset + boff;
      bufs[nbufs].decoded = NULL;
      bufs[nbufs].wb = true;
      bufs[nbufs].skip = 0;
      bufs[nbufs].size = (uint32_t)size;
      bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + b * BORGVK_CS_SLOT_BYTES;
      total += 8 + ((uint32_t)size + 3) / 4 * 4;
      nbufs++;
   }

   /* Push constants: one more window, slot 8. */
   {
      uint8_t *dec = calloc(1, sizeof(cmd->pc));
      if (dec) {
         memcpy(dec, cmd->pc, sizeof(cmd->pc));
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = sizeof(cmd->pc);
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + BORGVK_MAX_BINDINGS * BORGVK_CS_SLOT_BYTES;
         total += 8 + sizeof(cmd->pc);
         nbufs++;
      }
   }

   /* Job header: nprog gx gy gz lx ly lz nregs nbufs bx by bz (b* = the grid origin, which the
    * compiled program adds to the workgroup id). */
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

   /* A big grid is cut along one axis into slices that run as separate simulator processes
    * (the simulator is cycle-accurate: a 512x512 grid of one-invocation workgroups takes
    * minutes in one process). Each slice starts from the same buffers and the changed bytes
    * are merged afterwards, so it is only done for shaders without atomics, whose result does
    * not depend on how the workgroups are ordered. */
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
   /* Merged result per buffer: the original bytes with every changed byte of every slice. */
   uint8_t *merged[BORGVK_MAX_BINDINGS] = { 0 };
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
   if (ran) {
      for (uint32_t i = 0; i < nbufs; i++)
         if (bufs[i].wb)
            memcpy(bufs[i].host, merged[i] + bufs[i].skip, bufs[i].size - bufs[i].skip);
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

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                            uint32_t bindingCount, const VkBuffer *pBuffers,
                            const VkDeviceSize *pOffsets)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   for (uint32_t i = 0; i < bindingCount; i++) {
      uint32_t b = firstBinding + i;
      VK_FROM_HANDLE(borgvk_buffer, buf, pBuffers[i]);
      if (b < BORGVK_MAX_VERTEX_BINDINGS) {
         cmd->vb[b] = buf && buf->mem && buf->mem->map
            ? (const uint8_t *)buf->mem->map + buf->offset + pOffsets[i] : NULL;
         cmd->vb_avail[b] = buf && buf->vk.size > pOffsets[i] ? buf->vk.size - pOffsets[i] : 0;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                             VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
                             const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
                             const uint32_t *pDynamicOffsets)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   if (descriptorSetCount > 0 && pDescriptorSets) {
      VK_FROM_HANDLE(borgvk_descriptor_set, set, pDescriptorSets[0]);
      if (set) {
         cmd->desc_set = set;
         memset(cmd->dyn_off, 0, sizeof(cmd->dyn_off));
         uint32_t k = 0;
         for (uint32_t b = 0; b < BORGVK_MAX_BINDINGS && k < dynamicOffsetCount; b++)
            if (set->dyn_mask & (1ull << b))
               cmd->dyn_off[b] = pDynamicOffsets[k++];
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
               uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   struct borgvk_device *device = container_of(vk_cmd->base.device, struct borgvk_device, vk);

   /* Only the simulator renders; cube.c (UBO-driven) has its own path at submit. */
   if (!getenv("BORGVK_SIM") || (cmd->desc_set && set_is_cube(cmd->desc_set)))
      return;
   if (borgvk_sim_generic_draw(device, cmd, vertexCount, firstVertex))
      cmd->generic_drawn = true;
}

static VkResult borgvk_queue_submit_work(struct vk_queue *vk_queue, struct vk_queue_submit *submit);

/* driver_submit: honour the submit's semaphore/fence waits and signals (the runtime leaves both
 * to the driver for a sync type with GPU_WAIT), around the actual work. */
VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct vk_device *vk_dev = vk_queue->base.device;

   for (uint32_t i = 0; i < submit->wait_count; i++) {
      VkResult r = vk_sync_wait(vk_dev, submit->waits[i].sync, submit->waits[i].wait_value,
                                VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (r != VK_SUCCESS)
         return r;
   }

   VkResult result = borgvk_queue_submit_work(vk_queue, submit);

   for (uint32_t i = 0; i < submit->signal_count; i++) {
      VkResult r = vk_sync_signal(vk_dev, submit->signals[i].sync, submit->signals[i].signal_value);
      if (r != VK_SUCCESS && result == VK_SUCCESS)
         result = r;
   }
   return result;
}

static VkResult
borgvk_queue_submit_work(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   /* Run the recorded commands (copies, clears, blits, events, push constants, render-pass
    * state, secondaries) in submission order. The application-facing dispatch only records;
    * cmd_dispatch holds the real implementations (see borgvk_CreateDevice). */
   struct borgvk_device *replay_dev =
      container_of(vk_queue->base.device, struct borgvk_device, vk);
   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      vk_cmd_queue_execute(&cb->cmd_queue, vk_command_buffer_to_handle(cb), &replay_dev->cmd_dispatch);
   }

   struct borgvk_device *device =
      container_of(vk_queue->base.device, struct borgvk_device, vk);

   /* Sim path: if BORGVK_SIM is set, render via arcilator_sim and write pixels
    * into the colour attachment instead of shipping to the FPGA.
    *   - cube.c (UBO-driven): borgvk_submit_sim_cube captures the full serial
    *     stream incl. 0xB0 borgc shaders and replays it via --cts-uart, so the
    *     sim runs the exact protocol + real shaders as the FPGA.
    *   - CTS draws (VBO-driven): borgvk_submit_sim_draw, the mailbox --cts-draw
    *     path (carries per-vertex colour; still baked-shader for now). */
   if (getenv("BORGVK_SIM")) {
      if (submit_is_cube(submit))
         return borgvk_submit_sim_cube(device, submit);
      /* Generic draws already ran, at their place in the replay above. */
      for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++)
         if (container_of(submit->command_buffers[ci], struct borgvk_command_buffer, vk)->generic_drawn ||
             container_of(submit->command_buffers[ci], struct borgvk_command_buffer, vk)->dispatched)
            return VK_SUCCESS;
      return borgvk_submit_sim_draw(submit);
   }

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
         memcpy(s.sampler, sampler_desc(set->samplers[1]), sizeof(s.sampler));
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SETUP, &s) == 0)
            g_setup_done = true;
         else
            mesa_logw("borgvk: DRM_IOCTL_BORG_SETUP failed");
      }

      if (g_setup_done) {
         send_blend_state(device);
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
               send_texture_row(tex, set->samplers[1], row);
            mark_setup_done();
            mesa_logi("borgvk: upload complete");
            g_setup_done = true;
         }
      }

      if (g_setup_done) {
         send_blend_state(device);
         borgvk_serial_send_mvp(ubo);
      }
   }

   return VK_SUCCESS;
}
