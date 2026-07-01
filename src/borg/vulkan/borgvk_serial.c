/*
 * Copyright © 2026 Borg GPU project
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
 * UART drain reads a constant byte count (must match the firmware RX_GEOM_*).
 * BORGVK_GEOM_MAX_VERTS/TRIS come from borgvk_private.h. */
#define BORGVK_GEOM_PKT_LEN \
  (1 + 2 + BORGVK_GEOM_MAX_VERTS * 6 + BORGVK_GEOM_MAX_TRIS * 3 + \
   BORGVK_GEOM_MAX_TRIS * 12 + 1)

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

/* IEEE-754 float32 → float16 (round-to-nearest-even). Inputs here are positions
 * in [-1,1] and UVs in [0,1], all comfortably in the FP16 normal range. */
static uint16_t
f32_to_f16(float f)
{
   union { float f; uint32_t u; } in = { f };
   uint32_t x = in.u;
   uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
   int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
   uint32_t mant = x & 0x7fffffu;

   if (((x >> 23) & 0xff) == 0xff)          /* inf / nan */
      return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
   if (exp >= 0x1f)                         /* overflow → inf */
      return (uint16_t)(sign | 0x7c00u);
   if (exp <= 0) {                          /* subnormal / underflow → flush to 0 */
      if (exp < -10)
         return sign;
      mant |= 0x800000u;
      uint32_t shift = (uint32_t)(14 - exp);
      uint16_t h = (uint16_t)(mant >> shift);
      if ((mant >> (shift - 1)) & 1u)       /* round */
         h++;
      return (uint16_t)(sign | h);
   }
   uint16_t h = (uint16_t)(sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13));
   if (mant & 0x1000u) {                    /* round-to-nearest-even */
      if ((mant & 0x0fffu) || (h & 1u))
         h++;
   }
   return h;
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
   const int ibase = vbase + BORGVK_GEOM_MAX_VERTS * 6;
   const int ubase = ibase + BORGVK_GEOM_MAX_TRIS * 3;

   for (int i = 0; i < nverts * 3; i++) {
      uint16_t h = f32_to_f16(verts[i]);
      pkt[vbase + i*2]     = (uint8_t)(h & 0xff);
      pkt[vbase + i*2 + 1] = (uint8_t)(h >> 8);
   }
   for (int i = 0; i < ntris * 3; i++)
      pkt[ibase + i] = idx[i];
   for (int i = 0; i < ntris * 6; i++) {
      uint16_t h = f32_to_f16(uv[i]);
      pkt[ubase + i*2]     = (uint8_t)(h & 0xff);
      pkt[ubase + i*2 + 1] = (uint8_t)(h >> 8);
   }

   uint8_t csum = 0;
   for (int i = 1; i < BORGVK_GEOM_PKT_LEN - 1; i++)
      csum ^= pkt[i];
   pkt[BORGVK_GEOM_PKT_LEN - 1] = csum;

   borgvk_transport_emit(pkt, sizeof(pkt));
}

#define BORGVK_MARKER_TEX  0xAF
/* 0xAF: marker, y, BORGVK_TEX_DIM texels RGB-FP16 (dim*6 B), checksum. */
#define BORGVK_TEX_PKT_LEN (1 + 1 + BORGVK_TEX_DIM * 6 + 1)

void
borgvk_serial_send_tex_row(int y, const float *rgb)
{
   uint8_t pkt[BORGVK_TEX_PKT_LEN];
   pkt[0] = BORGVK_MARKER_TEX;
   pkt[1] = (uint8_t)y;
   for (int i = 0; i < BORGVK_TEX_DIM * 3; i++) {
      uint16_t h = f32_to_f16(rgb[i]);
      pkt[2 + i*2]     = (uint8_t)(h & 0xff);
      pkt[2 + i*2 + 1] = (uint8_t)(h >> 8);
   }
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
