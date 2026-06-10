/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Serial transport for borgvk. The real GPU is the ULX3S FPGA, reached over the
 * board's FT231X USB-serial bridge (/dev/ttyUSB0 @115200). Each frame the submit
 * path hands us the 4×4 model-view-projection matrix; we frame it and write it to
 * the port. The firmware (software/borg/borg_vkcube.c) decodes the packet and
 * renders the cube through the autonomous TBR sequencer.
 *
 * Packet format ("0xAD" full-MVP):
 *   byte 0      : 0xAD marker
 *   bytes 1..64 : 16 little-endian float32, row-major mvp[4][4] (cube.c order)
 *   byte 65     : XOR checksum of bytes 1..64
 * Total 66 bytes. Unlike the legacy 0xAC rotation matrix, MVP entries are not
 * bounded to [-1,1] (projection scales them), so the firmware validates with the
 * checksum instead of a magnitude range.
 */
#include "borgvk_private.h"

#include "util/log.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
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

static void
borgvk_serial_write_paced(int fd, const uint8_t *pkt, size_t len)
{
   size_t off = 0;
   while (off < len) {
      ssize_t n = write(fd, &pkt[off], len - off);
      if (n < 0) {
         if (errno == EINTR)
            continue;
         mesa_logw("borgvk: serial write failed (%s)", strerror(errno));
         return;
      }
      off += (size_t)n;
   }
   /* Drain + idle so the firmware sees an inter-packet gap to sync on. */
   tcdrain(fd);
   usleep(3000);
}

/* Ship the app's real mesh: nverts unique model-space positions, ntris triangles
 * each indexing 3 of them, with per-triangle-vertex UVs.  Fixed-offset padded
 * regions keep the packet a constant length (see firmware RX_GEOM_*). */
void
borgvk_serial_send_geom(const float *verts, int nverts,
                        const uint8_t *idx, const float *uv, int ntris)
{
   int fd = borgvk_serial_open();
   if (fd < 0)
      return;
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

   borgvk_serial_write_paced(fd, pkt, sizeof(pkt));
}

#define BORGVK_MARKER_TEX  0xAF
/* 0xAF: marker, y, BORGVK_TEX_DIM texels RGB-FP16 (dim*6 B), checksum. */
#define BORGVK_TEX_PKT_LEN (1 + 1 + BORGVK_TEX_DIM * 6 + 1)

void
borgvk_serial_send_tex_row(int y, const float *rgb)
{
   int fd = borgvk_serial_open();
   if (fd < 0)
      return;

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

   borgvk_serial_write_paced(fd, pkt, sizeof(pkt));
}

void
borgvk_serial_send_mvp(const float mvp[16])
{
   int fd = borgvk_serial_open();
   if (fd < 0)
      return;

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
   borgvk_serial_write_paced(fd, pkt, sizeof(pkt));
}
