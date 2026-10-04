/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * Serial transport for borgvk. The real GPU is the ULX3S FPGA, reached over the
 * board's FT231X USB-serial bridge (/dev/ttyUSB0 @115200). Each frame the submit
 * path hands us the 4×4 model-view-projection matrix; we frame it and write it to
 * the port. The firmware (software/borg/borg_kernel.c) decodes the packet and
 * renders the frame through the autonomous TBR sequencer.
 *
 * Packet format ("0xAD" full-MVP):
 *   byte 0      : 0xAD marker
 *   bytes 1..64 : 16 little-endian float32, row-major mvp[4][4] (cube.c order)
 *   byte 65     : XOR checksum of bytes 1..64
 * Total 66 bytes. Unlike the legacy 0xAC rotation matrix, MVP entries are not
 * bounded to [-1,1] (projection scales them), so the firmware validates with the
 * checksum instead of a magnitude range.
 *
 * Sim transport: when $BORGVK_SIM_SOCKET is set, packets go to a Unix domain
 * socket instead of the serial port — the interactive verilator/arcilator
 * viewer (simulation/verilator/viewer.py) listens there and injects the same
 * bytes into the simulated hardware UART RXD line.  Same wire protocol, same
 * borg_kernel.c firmware image; only the transport differs.
 */
#include "borgvk_private.h"

#include "util/log.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define BORGVK_SERIAL_DEFAULT "/dev/ttyUSB0"
#define BORGVK_MARKER_MVP     0xAD
#define BORGVK_MARKER_GEOM    0xAE

/* The geometry packet is a fixed length with padded, fixed-offset regions so the
 * UART drain reads a constant byte count (must match the firmware RX_GEOM_*):
 * marker, nverts, ntris, MAX_VERTS*3 float32 positions, MAX_TRIS*3 u8 indices,
 * MAX_TRIS*3*2 float32 UVs, XOR checksum -- 520 B.
 * BORGVK_GEOM_MAX_VERTS/TRIS come from borgvk_private.h. */
#define BORGVK_GEOM_PKT_LEN \
  (1 + 2 + BORGVK_GEOM_MAX_VERTS * 12 + BORGVK_GEOM_MAX_TRIS * 3 + \
   BORGVK_GEOM_MAX_TRIS * 24 + 1)

/* Opened lazily on first submit and kept open for the process lifetime. -1 = not
 * yet attempted, -2 = open failed (don't retry every frame). */
static int borgvk_serial_fd = -1;

static int
borgvk_serial_open(void)
{
   if (borgvk_serial_fd >= 0)
      return borgvk_serial_fd;
   if (borgvk_serial_fd == -2)
      return -1;

   const char *path = getenv("BORGVK_SERIAL");
   if (!path || !path[0])
      path = BORGVK_SERIAL_DEFAULT;

   int fd = open(path, O_WRONLY | O_NOCTTY | O_CLOEXEC);
   if (fd < 0) {
      mesa_logw("borgvk: cannot open serial port %s (set $BORGVK_SERIAL); "
                "frames will not reach the FPGA", path);
      borgvk_serial_fd = -2;
      return -1;
   }

   struct termios tio;
   if (tcgetattr(fd, &tio) == 0) {
      cfmakeraw(&tio);
      cfsetispeed(&tio, B115200);
      cfsetospeed(&tio, B115200);
      tio.c_cflag |= (CLOCAL | CREAD);
      tio.c_cflag &= ~CRTSCTS;
      tcsetattr(fd, TCSANOW, &tio);
   }

   borgvk_serial_fd = fd;
   mesa_logi("borgvk: streaming MVP to %s @115200", path);
   return fd;
}

/* -1 = not yet attempted, -2 = connect failed (don't retry every frame). */
static int borgvk_sim_socket_fd = -1;

/* Connect to the interactive sim viewer's Unix domain socket (see
 * simulation/verilator/viewer.py).  Lazily opened on first submit, kept open
 * for the process lifetime — same lifecycle as borgvk_serial_open(). */
static int
borgvk_sim_socket_open(const char *path)
{
   if (borgvk_sim_socket_fd >= 0)
      return borgvk_sim_socket_fd;
   if (borgvk_sim_socket_fd == -2)
      return -1;

   int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
   if (fd < 0) {
      mesa_logw("borgvk: socket() failed (%s)", strerror(errno));
      borgvk_sim_socket_fd = -2;
      return -1;
   }

   struct sockaddr_un addr = { .sun_family = AF_UNIX };
   snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
   if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
      mesa_logw("borgvk: cannot connect to sim socket %s (%s); "
                "is the viewer running?", path, strerror(errno));
      close(fd);
      borgvk_sim_socket_fd = -2;
      return -1;
   }

   borgvk_sim_socket_fd = fd;
   mesa_logi("borgvk: streaming to sim socket %s", path);
   return fd;
}

/* A float32 as its IEEE bits, little-endian: positions and UVs are shader
 * datapath values, and the datapath is FP32 -- no conversion. */
static void
put_f32_le(uint8_t *dst, float f)
{
   union { float f; uint32_t u; } in = { f };
   dst[0] = (uint8_t)(in.u & 0xff);
   dst[1] = (uint8_t)((in.u >> 8) & 0xff);
   dst[2] = (uint8_t)((in.u >> 16) & 0xff);
   dst[3] = (uint8_t)((in.u >> 24) & 0xff);
}

/* Write the full packet, then idle so the receiver sees an inter-packet gap
 * to sync on (the firmware's gap-sync waits for the line to go idle before
 * trusting the next marker byte — see borg_kernel.c).  `is_socket` selects
 * tcdrain() (serial, flushes the kernel TTY write buffer) vs a plain sleep
 * (socket writes are already synchronous once write() returns). */
static void
borgvk_transport_write_paced(int fd, const uint8_t *pkt, size_t len, bool is_socket)
{
   size_t off = 0;
   while (off < len) {
      ssize_t n = write(fd, &pkt[off], len - off);
      if (n < 0) {
         if (errno == EINTR)
            continue;
         mesa_logw("borgvk: %s write failed (%s)",
                   is_socket ? "sim socket" : "serial", strerror(errno));
         return;
      }
      off += (size_t)n;
   }
   if (!is_socket)
      tcdrain(fd);
   usleep(3000);
}

/* ---- Transport sink: serial port / sim socket (default) or capture buffer ---
 * Selection: capture (during a frame wrapped in capture_begin/end) takes
 * priority; otherwise $BORGVK_SIM_SOCKET routes to the interactive sim
 * viewer; otherwise the real serial port. */
static bool     borgvk_capture_active = false;
static uint8_t *borgvk_capture_buf    = NULL;
static size_t   borgvk_capture_len    = 0;
static size_t   borgvk_capture_cap    = 0;

/* Single emit chokepoint for every framed packet.  In capture mode the bytes
 * are appended to the growable buffer (no serial port, no pacing — the sim
 * injects them all at once); otherwise they are paced out to the socket or
 * serial fd. */
static void
borgvk_transport_emit(const uint8_t *pkt, size_t len)
{
   if (borgvk_capture_active) {
      if (borgvk_capture_len + len > borgvk_capture_cap) {
         size_t ncap = borgvk_capture_cap ? borgvk_capture_cap : 4096;
         while (ncap < borgvk_capture_len + len)
            ncap *= 2;
         uint8_t *nbuf = realloc(borgvk_capture_buf, ncap);
         if (!nbuf) {
            mesa_logw("borgvk: transport capture realloc(%zu) failed", ncap);
            return;
         }
         borgvk_capture_buf = nbuf;
         borgvk_capture_cap = ncap;
      }
      memcpy(borgvk_capture_buf + borgvk_capture_len, pkt, len);
      borgvk_capture_len += len;
      return;
   }

   const char *sim_socket = getenv("BORGVK_SIM_SOCKET");
   if (sim_socket && sim_socket[0]) {
      int fd = borgvk_sim_socket_open(sim_socket);
      if (fd < 0)
         return;
      borgvk_transport_write_paced(fd, pkt, len, true);
      return;
   }

   int fd = borgvk_serial_open();
   if (fd < 0)
      return;
   borgvk_transport_write_paced(fd, pkt, len, false);
}

void
borgvk_transport_capture_begin(void)
{
   borgvk_capture_len = 0;     /* reuse any existing allocation */
   borgvk_capture_active = true;
}

uint8_t *
borgvk_transport_capture_end(size_t *out_len)
{
   borgvk_capture_active = false;
   uint8_t *buf = borgvk_capture_buf;
   size_t   len = borgvk_capture_len;
   borgvk_capture_buf = NULL;  /* transfer ownership to caller */
   borgvk_capture_cap = 0;
   borgvk_capture_len = 0;
   if (out_len)
      *out_len = len;
   return buf;
}

/* Ship the app's real mesh: nverts unique model-space positions, ntris triangles
 * each indexing 3 of them, with per-triangle-vertex UVs.  Fixed-offset padded
 * regions keep the packet a constant length (see firmware RX_GEOM_*). */
void
borgvk_serial_send_geom(const float *verts, int nverts,
                        const uint8_t *idx, const float *uv, int ntris)
{
   if (nverts < 1 || nverts > BORGVK_GEOM_MAX_VERTS ||
       ntris  < 1 || ntris  > BORGVK_GEOM_MAX_TRIS)
      return;

   uint8_t pkt[BORGVK_GEOM_PKT_LEN];
   memset(pkt, 0, sizeof(pkt));
   pkt[0] = BORGVK_MARKER_GEOM;
   pkt[1] = (uint8_t)nverts;
   pkt[2] = (uint8_t)ntris;

   const int vbase = 3;
   const int ibase = vbase + BORGVK_GEOM_MAX_VERTS * 12;
   const int ubase = ibase + BORGVK_GEOM_MAX_TRIS * 3;

   for (int i = 0; i < nverts * 3; i++)
      put_f32_le(&pkt[vbase + i * 4], verts[i]);
   for (int i = 0; i < ntris * 3; i++)
      pkt[ibase + i] = idx[i];
   for (int i = 0; i < ntris * 6; i++)
      put_f32_le(&pkt[ubase + i * 4], uv[i]);

   uint8_t csum = 0;
   for (int i = 1; i < BORGVK_GEOM_PKT_LEN - 1; i++)
      csum ^= pkt[i];
   pkt[BORGVK_GEOM_PKT_LEN - 1] = csum;

   borgvk_transport_emit(pkt, sizeof(pkt));
}

#define BORGVK_MARKER_TEX  0xAF
/* 0xAF: marker, y, sampler descriptor (4 words LE), BORGVK_TEX_DIM RGBA8
 * texels, checksum. Every row carries the sampler, so any row that arrives
 * intact delivers it. */
#define BORGVK_TEX_PKT_LEN (1 + 1 + 16 + BORGVK_TEX_DIM * 4 + 1)

void
borgvk_serial_send_tex_row(int y, const uint8_t *rgba, const uint32_t sampler[4])
{
   uint8_t pkt[BORGVK_TEX_PKT_LEN];
   pkt[0] = BORGVK_MARKER_TEX;
   pkt[1] = (uint8_t)y;
   for (int w = 0; w < 4; w++)
      for (int b = 0; b < 4; b++)
         pkt[2 + w * 4 + b] = (uint8_t)(sampler[w] >> (8 * b));
   memcpy(&pkt[2 + 16], rgba, BORGVK_TEX_DIM * 4);
   uint8_t csum = 0;
   for (int i = 1; i < BORGVK_TEX_PKT_LEN - 1; i++)
      csum ^= pkt[i];
   pkt[BORGVK_TEX_PKT_LEN - 1] = csum;

   borgvk_transport_emit(pkt, sizeof(pkt));
}

#define BORGVK_MARKER_SHADER  0xB0
/* 0xB0 shader upload: marker, stage(1B), len(2B LE), blob padded to
 * BORGVK_SHADER_BLOB_MAX, checksum. Fixed length so the firmware drain reads a
 * constant byte count (like 0xAE/0xAF); `len` says how many blob bytes are valid.
 * 517 B total < the firmware's 0xAF drain buffer, so no RX buffer growth needed. */
#define BORGVK_SHADER_PKT_LEN (1 + 1 + 2 + BORGVK_SHADER_BLOB_MAX + 1)

void
borgvk_serial_send_shader(uint8_t stage, const uint8_t *blob, uint32_t len)
{
   if (!blob || len == 0 || len > BORGVK_SHADER_BLOB_MAX) {
      mesa_logw("borgvk: refusing to send shader (stage %u, len %u)", stage, len);
      return;
   }

   uint8_t pkt[BORGVK_SHADER_PKT_LEN];
   memset(pkt, 0, sizeof(pkt));
   pkt[0] = BORGVK_MARKER_SHADER;
   pkt[1] = stage;
   pkt[2] = (uint8_t)(len & 0xff);
   pkt[3] = (uint8_t)(len >> 8);
   memcpy(&pkt[4], blob, len);   /* remainder stays zero-padded */

   uint8_t csum = 0;
   for (int i = 1; i < BORGVK_SHADER_PKT_LEN - 1; i++)
      csum ^= pkt[i];
   pkt[BORGVK_SHADER_PKT_LEN - 1] = csum;

   borgvk_transport_emit(pkt, sizeof(pkt));
   mesa_logi("borgvk: %s %s shader (%u bytes)",
             borgvk_capture_active ? "captured" : "uploaded",
             stage == 0 ? "vertex" : "fragment", len);
}

#define BORGVK_MARKER_PUSH    0xB2
/* 0xB2 push constants: marker, off_words(1B), n_words(1B), data padded to
 * BORGVK_PUSH_MAX_WORDS words, checksum.  0xB1 is the firmware's serial-reload
 * trigger, hence 0xB2.  Fixed length for the same reason as 0xAE/0xAF/0xB0:
 * the firmware drain reads a constant byte count per marker, and `n_words`
 * says how much of the payload is valid.
 *
 * 32 words = 128 B = Vulkan 1.0's minimum maxPushConstantsSize, which is what
 * this driver advertises.  Raising that limit means growing this AND the
 * firmware's BORG_PUSH_CONST_MAX_WORDS together -- the firmware clamps, so a
 * mismatch truncates silently rather than corrupting, but it would still be a
 * wrong answer. */
#define BORGVK_PUSH_MAX_WORDS 32
#define BORGVK_PUSH_PKT_LEN   (1 + 1 + 1 + BORGVK_PUSH_MAX_WORDS * 4 + 1)

void
borgvk_serial_send_push_constants(uint32_t offset, uint32_t size,
                                  const void *values)
{
   /* Vulkan requires both to be multiples of 4 (VUID-vkCmdPushConstants-offset
    * -00368 / -size-00369); check rather than assume, because the whole
    * word-index mapping below is built on it. */
   if (!values || size == 0 || (offset & 3) || (size & 3)) {
      mesa_logw("borgvk: bad push-constant range (offset %u, size %u)",
                offset, size);
      return;
   }

   uint32_t off_words = offset / 4;
   uint32_t n_words   = size / 4;
   if (off_words >= BORGVK_PUSH_MAX_WORDS ||
       n_words > BORGVK_PUSH_MAX_WORDS - off_words) {
      mesa_logw("borgvk: push-constant range %u..%u B exceeds the %u B limit",
                offset, offset + size, BORGVK_PUSH_MAX_WORDS * 4);
      return;
   }

   uint8_t pkt[BORGVK_PUSH_PKT_LEN];
   memset(pkt, 0, sizeof(pkt));
   pkt[0] = BORGVK_MARKER_PUSH;
   pkt[1] = (uint8_t)off_words;
   pkt[2] = (uint8_t)n_words;
   /* memcpy, not a u32 store loop: `values` is void* from the application and
    * carries no alignment guarantee, and the wire format is little-endian
    * bytes either way. */
   memcpy(&pkt[3], values, size);

   uint8_t csum = 0;
   for (int i = 1; i < BORGVK_PUSH_PKT_LEN - 1; i++)
      csum ^= pkt[i];
   pkt[BORGVK_PUSH_PKT_LEN - 1] = csum;

   borgvk_transport_emit(pkt, sizeof(pkt));
   mesa_logi("borgvk: %s %u B of push constants at offset %u",
             borgvk_capture_active ? "captured" : "sent", size, offset);
}

#define BORGVK_MARKER_BLEND   0xB3
/* 0xB3 blend state: marker, BLEND_CFG (LE u32), BLEND_CONST (LE u32), csum. */
void
borgvk_serial_send_blend(uint32_t blend_cfg, uint32_t blend_const)
{
   uint8_t pkt[10];
   pkt[0] = BORGVK_MARKER_BLEND;
   for (int i = 0; i < 4; i++) {
      pkt[1 + i] = (uint8_t)(blend_cfg   >> (8 * i));
      pkt[5 + i] = (uint8_t)(blend_const >> (8 * i));
   }
   uint8_t csum = 0;
   for (int i = 1; i < 9; i++)
      csum ^= pkt[i];
   pkt[9] = csum;
   borgvk_transport_emit(pkt, sizeof(pkt));
}

static uint32_t
unorm8(float f)
{
   if (!(f > 0.0f)) return 0;
   if (f >= 1.0f) return 255;
   return (uint32_t)(f * 255.0f + 0.5f);
}

void
borgvk_blend_pack(const VkPipelineColorBlendStateCreateInfo *cb,
                  uint32_t *cfg, uint32_t *konst)
{
   /* Reset state: blending off, all channels written. */
   *cfg = 0xFu << 27;
   *konst = 0;
   if (!cb || cb->attachmentCount < 1 || !cb->pAttachments)
      return;
   const VkPipelineColorBlendAttachmentState *a = &cb->pAttachments[0];
   *cfg = (a->blendEnable ? 1u : 0u) |
          ((uint32_t)a->srcColorBlendFactor & 0x1F) << 1 |
          ((uint32_t)a->dstColorBlendFactor & 0x1F) << 6 |
          ((uint32_t)a->colorBlendOp & 7) << 11 |
          ((uint32_t)a->srcAlphaBlendFactor & 0x1F) << 14 |
          ((uint32_t)a->dstAlphaBlendFactor & 0x1F) << 19 |
          ((uint32_t)a->alphaBlendOp & 7) << 24 |
          ((uint32_t)a->colorWriteMask & 0xF) << 27;
   *konst = unorm8(cb->blendConstants[0]) |
            unorm8(cb->blendConstants[1]) << 8 |
            unorm8(cb->blendConstants[2]) << 16 |
            unorm8(cb->blendConstants[3]) << 24;
}

void
borgvk_serial_send_mvp(const float mvp[16])
{
   uint8_t pkt[66];
   pkt[0] = BORGVK_MARKER_MVP;
   memcpy(&pkt[1], mvp, 16 * sizeof(float));

   uint8_t csum = 0;
   for (int i = 1; i <= 64; i++)
      csum ^= pkt[i];
   pkt[65] = csum;

   /* Paced write: the firmware aligns to packets via the inter-packet IDLE GAP,
    * so a blocking app's back-to-back submits would otherwise stream gaplessly
    * and freeze the cube (it locks onto the first packet and never re-syncs). */
   borgvk_transport_emit(pkt, sizeof(pkt));

   /* Live-display frame pacing is meaningless when capturing for the sim. */
   if (borgvk_capture_active)
      return;

   /* Steady frame pacing (BORGVK_FRAME_MS): hold each host frame to a fixed
    * wall-clock period.  cube.c spins by a CONSTANT angle per frame, so emitting
    * MVPs at a steady rate at/below the FPGA's sustained render rate makes the
    * displayed motion advance by exactly one angle step per shown frame —
    * removing the judder from subsampling ~60 host fps down to ~15 with a
    * variable stride.  Because re-rendering the same MVP yields an identical
    * image, the FPGA's per-frame time variance no longer affects the motion.
    * Drift-free: the deadline advances by a fixed period, not from "now".
    * Unset = no throttle (legacy free-run behaviour). */
   const char *ms_env = getenv("BORGVK_FRAME_MS");
   if (ms_env && ms_env[0]) {
      long period_ns = atol(ms_env) * 1000000L;
      if (period_ns > 0) {
         static struct timespec next;
         static int armed;
         struct timespec now;
         clock_gettime(CLOCK_MONOTONIC, &now);
         if (armed) {
            long wait_ns = (next.tv_sec - now.tv_sec) * 1000000000L +
                           (next.tv_nsec - now.tv_nsec);
            if (wait_ns > 0) {
               struct timespec ts = { wait_ns / 1000000000L,
                                      wait_ns % 1000000000L };
               nanosleep(&ts, NULL);
            } else {
               next = now;   /* fell behind (slow frame): resync, don't bank debt */
            }
         } else {
            next = now;
            armed = 1;
         }
         next.tv_nsec += period_ns;
         while (next.tv_nsec >= 1000000000L) {
            next.tv_nsec -= 1000000000L;
            next.tv_sec++;
         }
      }
   }
}

#define BORGVK_MARKER_STATE   0xB4
/* 0xB4 raster state: marker, 5 register values (LE u32), csum. */
void
borgvk_serial_send_state(const uint32_t reg[5])
{
   uint8_t pkt[22];
   pkt[0] = BORGVK_MARKER_STATE;
   for (int r = 0; r < 5; r++)
      for (int i = 0; i < 4; i++)
         pkt[1 + 4 * r + i] = (uint8_t)(reg[r] >> (8 * i));
   uint8_t csum = 0;
   for (int i = 1; i < 21; i++)
      csum ^= pkt[i];
   pkt[21] = csum;
   borgvk_transport_emit(pkt, sizeof(pkt));
}

#define BORGVK_MARKER_TARGET  0xB6
/* 0xB6 render target: marker, flush format (0 R5G6B5, 1 R8G8B8A8, 2 B8G8R8A8), the clear colour
 * (4 x float32 LE), csum. Only the simulator's host driver acts on it. */
void
borgvk_serial_send_target(uint8_t flush_format, const float clear[4])
{
   uint8_t pkt[19];
   pkt[0] = BORGVK_MARKER_TARGET;
   pkt[1] = flush_format;
   for (int i = 0; i < 4; i++)
      put_f32_le(&pkt[2 + 4 * i], clear[i]);
   uint8_t csum = 0;
   for (int i = 1; i < 18; i++)
      csum ^= pkt[i];
   pkt[18] = csum;
   borgvk_transport_emit(pkt, sizeof(pkt));
}

static void
put_u32_le(uint8_t *d, uint32_t v)
{
   d[0] = v & 0xff; d[1] = (v >> 8) & 0xff; d[2] = (v >> 16) & 0xff; d[3] = (v >> 24) & 0xff;
}

static void
emit_checked(uint8_t *pkt, size_t len)
{
   uint8_t csum = 0;
   for (size_t i = 1; i < len - 1; i++)
      csum ^= pkt[i];
   pkt[len - 1] = csum;
   borgvk_transport_emit(pkt, len);
}

/* Host-simulator only. 0xB8: write n (<= 256, a multiple of 4 is enough; the rest is padded)
 * bytes at heap offset `off` (BORG_HEAP_SPI + off in the device's memory). */
void
borgvk_serial_send_mem(uint32_t off, const uint8_t *data, uint32_t n)
{
   uint8_t pkt[1 + 4 + 2 + 256 + 1];
   memset(pkt, 0, sizeof(pkt));
   pkt[0] = 0xB8;
   put_u32_le(&pkt[1], off);
   pkt[5] = n & 0xff; pkt[6] = n >> 8;
   memcpy(&pkt[7], data, n);
   emit_checked(pkt, sizeof(pkt));
}

/* Host-simulator only. 0xB9: vertex attribute `slot` is a typed fetch (texture format code
 * `fmt`, component swizzle bits `swz` as in the texture descriptor) of `count` rows `stride`
 * bytes apart from heap offset `base` (two's complement, may precede the uploaded data). */
void
borgvk_serial_send_vattr(uint32_t slot, uint32_t fmt, uint32_t base, uint32_t count,
                         uint32_t stride, uint32_t swz)
{
   uint8_t pkt[1 + 1 + 1 + 4 + 4 + 4 + 4 + 1];
   pkt[0] = 0xB9;
   pkt[1] = (uint8_t)slot;
   pkt[2] = (uint8_t)fmt;
   put_u32_le(&pkt[3], base);
   put_u32_le(&pkt[7], count);
   put_u32_le(&pkt[11], stride);
   put_u32_le(&pkt[15], swz);
   emit_checked(pkt, sizeof(pkt));
}

/* Host-simulator only. 0xBA: run a draw. topology 0 list / 1 strip / 2 fan, index_type 0 none /
 * 1 16-bit / 2 32-bit, index_base = heap offset of the first index. */
void
borgvk_serial_send_draw(uint32_t topology, uint32_t index_type, uint32_t restart,
                        uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                        uint32_t first_instance, int32_t vertex_offset, uint32_t index_base)
{
   uint8_t pkt[1 + 3 + 6 * 4 + 1];
   pkt[0] = 0xBA;
   pkt[1] = (uint8_t)topology; pkt[2] = (uint8_t)index_type; pkt[3] = (uint8_t)restart;
   put_u32_le(&pkt[4], vertex_count);
   put_u32_le(&pkt[8], instance_count);
   put_u32_le(&pkt[12], first_vertex);
   put_u32_le(&pkt[16], first_instance);
   put_u32_le(&pkt[20], (uint32_t)vertex_offset);
   put_u32_le(&pkt[24], index_base);
   emit_checked(pkt, sizeof(pkt));
}

/* Host-simulator only: the pass's attachments. flags bit 0 depth, 1 stencil, 2 D32_SFLOAT depth. */
void
borgvk_serial_send_pass(uint8_t flush_format, uint8_t flags)
{
   uint8_t pkt[4] = { 0xBB, flush_format, flags, 0 };
   emit_checked(pkt, sizeof(pkt));
}

/* Host-simulator only: vec4 attribute at location 1 for each of n corners. */
void
borgvk_serial_send_attr4(const float *attr, int n)
{
   uint8_t pkt[1 + 1 + 36 * 16 + 1];
   memset(pkt, 0, sizeof(pkt));
   pkt[0] = 0xB7;
   pkt[1] = (uint8_t)n;
   for (int i = 0; i < n * 4; i++)
      put_f32_le(&pkt[2 + 4 * i], attr[i]);
   uint8_t csum = 0;
   for (int i = 1; i < (int)sizeof(pkt) - 1; i++)
      csum ^= pkt[i];
   pkt[sizeof(pkt) - 1] = csum;
   borgvk_transport_emit(pkt, sizeof(pkt));
}

static uint32_t
stencil_face_pack(const VkStencilOpState *f)
{
   return ((uint32_t)f->compareMask & 0xFF) |
          ((uint32_t)f->writeMask & 0xFF) << 8 |
          ((uint32_t)f->reference & 0xFF) << 16;
}

void
borgvk_state_pack(const VkGraphicsPipelineCreateInfo *ci, uint32_t reg[5])
{
   /* Reset state: no stencil, depth LESS + write, cull back faces. */
   reg[0] = 0;
   reg[1] = reg[2] = 0;
   reg[3] = 1u | (1u << 3);
   reg[4] = 2u;

   const VkPipelineDepthStencilStateCreateInfo *ds = ci->pDepthStencilState;
   if (ds) {
      /* Vulkan compare and stencil-op enums pass through unchanged. */
      uint32_t cmp = ds->depthTestEnable ? ((uint32_t)ds->depthCompareOp & 7) : 7u;
      reg[3] = cmp | ((ds->depthTestEnable && ds->depthWriteEnable) ? 1u << 3 : 0u);
      if (ds->stencilTestEnable) {
         const VkStencilOpState *f = &ds->front, *b = &ds->back;
         reg[0] = 1u |
                  ((uint32_t)f->compareOp & 7) << 1 | ((uint32_t)f->failOp & 7) << 4 |
                  ((uint32_t)f->passOp & 7) << 7 | ((uint32_t)f->depthFailOp & 7) << 10 |
                  ((uint32_t)b->compareOp & 7) << 13 | ((uint32_t)b->failOp & 7) << 16 |
                  ((uint32_t)b->passOp & 7) << 19 | ((uint32_t)b->depthFailOp & 7) << 22;
         reg[1] = stencil_face_pack(f);
         reg[2] = stencil_face_pack(b);
      }
   }
   const VkPipelineRasterizationStateCreateInfo *rs = ci->pRasterizationState;
   if (rs) {
      /* VkCullModeFlags bit 0 = front, bit 1 = back, as in CULL_CFG. cube.c's
       * CCW winding is the hardware's unmodified facing; CW inverts it. */
      reg[4] = ((uint32_t)rs->cullMode & 3) |
               (rs->frontFace == VK_FRONT_FACE_CLOCKWISE ? 1u << 2 : 0u);
   }
}

#define BORGVK_MARKER_TEXG 0xB5
/* 0xB5: marker, off (LE u32), n (LE u16), desc words 1..3, sampler words,
 * BORGVK_TEXG_CHUNK data bytes (zero padded), csum. */
void
borgvk_serial_send_texture_chunk(uint32_t off, const uint8_t *data, uint32_t n,
                                 const uint32_t desc_w123[3],
                                 const uint32_t sampler[4])
{
   uint8_t pkt[1 + 4 + 2 + 12 + 16 + BORGVK_TEXG_CHUNK + 1] = { 0 };
   if (n > BORGVK_TEXG_CHUNK)
      n = BORGVK_TEXG_CHUNK;
   pkt[0] = BORGVK_MARKER_TEXG;
   for (int i = 0; i < 4; i++)
      pkt[1 + i] = (uint8_t)(off >> (8 * i));
   pkt[5] = (uint8_t)n;
   pkt[6] = (uint8_t)(n >> 8);
   for (int w = 0; w < 3; w++)
      for (int i = 0; i < 4; i++)
         pkt[7 + 4 * w + i] = (uint8_t)(desc_w123[w] >> (8 * i));
   for (int w = 0; w < 4; w++)
      for (int i = 0; i < 4; i++)
         pkt[19 + 4 * w + i] = (uint8_t)(sampler[w] >> (8 * i));
   memcpy(&pkt[35], data, n);
   uint8_t csum = 0;
   for (size_t i = 1; i < sizeof(pkt) - 1; i++)
      csum ^= pkt[i];
   pkt[sizeof(pkt) - 1] = csum;
   borgvk_transport_emit(pkt, sizeof(pkt));
}
