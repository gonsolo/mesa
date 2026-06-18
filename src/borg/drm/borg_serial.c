/* SPDX-License-Identifier: MIT */
/*
 * Serial transport for the Borg drm-shim.  Functionally identical to
 * borgvk_serial.c; adapted here to have no Mesa / Vulkan header dependency
 * so it can be compiled into libborg_drm_shim.so without pulling in the full
 * Mesa Vulkan runtime.  Log output goes to stderr via fprintf.
 */
#include "borg_serial.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define BORG_SERIAL_DEFAULT "/dev/ttyUSB0"
#define BORG_MARKER_MVP     0xAD
#define BORG_MARKER_GEOM    0xAE
#define BORG_MARKER_TEX     0xAF
#define BORG_TEX_PKT_LEN    (1 + 1 + BORG_TEX_DIM * 6 + 1)

static int g_serial_fd = -1;  /* -1 = not tried, -2 = failed */

static int
borg_serial_open(void)
{
   if (g_serial_fd >= 0)
      return g_serial_fd;
   if (g_serial_fd == -2)
      return -1;

   const char *path = getenv("BORGVK_SERIAL");
   if (!path || !path[0])
      path = BORG_SERIAL_DEFAULT;

   int fd = open(path, O_WRONLY | O_NOCTTY | O_CLOEXEC);
   if (fd < 0) {
      fprintf(stderr, "borg-shim: cannot open %s (%s); set $BORGVK_SERIAL\n",
              path, strerror(errno));
      g_serial_fd = -2;
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

   g_serial_fd = fd;
   fprintf(stderr, "borg-shim: streaming to %s @115200\n", path);
   return fd;
}

/* IEEE-754 float32 → float16, round-to-nearest-even. */
static uint16_t
f32_to_f16(float f)
{
   union { float f; uint32_t u; } in = { f };
   uint32_t x = in.u;
   uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
   int32_t exp   = (int32_t)((x >> 23) & 0xff) - 127 + 15;
   uint32_t mant = x & 0x7fffffu;

   if (((x >> 23) & 0xff) == 0xff)
      return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
   if (exp >= 0x1f)
      return (uint16_t)(sign | 0x7c00u);
   if (exp <= 0) {
      if (exp < -10) return sign;
      mant |= 0x800000u;
      uint32_t shift = (uint32_t)(14 - exp);
      uint16_t h = (uint16_t)(mant >> shift);
      if ((mant >> (shift - 1)) & 1u) h++;
      return (uint16_t)(sign | h);
   }
   uint16_t h = (uint16_t)(sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13));
   if (mant & 0x1000u) {
      if ((mant & 0x0fffu) || (h & 1u)) h++;
   }
   return h;
}

static void
borg_serial_write_paced(int fd, const uint8_t *pkt, size_t len)
{
   size_t off = 0;
   while (off < len) {
      ssize_t n = write(fd, &pkt[off], len - off);
      if (n < 0) {
         if (errno == EINTR) continue;
         fprintf(stderr, "borg-shim: serial write failed (%s)\n", strerror(errno));
         return;
      }
      off += (size_t)n;
   }
   tcdrain(fd);
   usleep(3000);
}

void
borg_serial_send_geom(const float *verts, int nverts,
                      const uint8_t *idx, const float *uv, int ntris)
{
   int fd = borg_serial_open();
   if (fd < 0) return;
   if (nverts < 1 || nverts > BORG_GEOM_MAX_VERTS ||
       ntris  < 1 || ntris  > BORG_GEOM_MAX_TRIS)
      return;

   uint8_t pkt[BORG_GEOM_PKT_LEN];
   memset(pkt, 0, sizeof(pkt));
   pkt[0] = BORG_MARKER_GEOM;
   pkt[1] = (uint8_t)nverts;
   pkt[2] = (uint8_t)ntris;

   const int vbase = 3;
   const int ibase = vbase + BORG_GEOM_MAX_VERTS * 6;
   const int ubase = ibase + BORG_GEOM_MAX_TRIS  * 3;

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
   for (int i = 1; i < BORG_GEOM_PKT_LEN - 1; i++) csum ^= pkt[i];
   pkt[BORG_GEOM_PKT_LEN - 1] = csum;
   borg_serial_write_paced(fd, pkt, sizeof(pkt));
}

void
borg_serial_send_tex_row(int y, const float *rgb)
{
   int fd = borg_serial_open();
   if (fd < 0) return;

   uint8_t pkt[BORG_TEX_PKT_LEN];
   pkt[0] = BORG_MARKER_TEX;
   pkt[1] = (uint8_t)y;
   for (int i = 0; i < BORG_TEX_DIM * 3; i++) {
      uint16_t h = f32_to_f16(rgb[i]);
      pkt[2 + i*2]     = (uint8_t)(h & 0xff);
      pkt[2 + i*2 + 1] = (uint8_t)(h >> 8);
   }
   uint8_t csum = 0;
   for (int i = 1; i < BORG_TEX_PKT_LEN - 1; i++) csum ^= pkt[i];
   pkt[BORG_TEX_PKT_LEN - 1] = csum;
   borg_serial_write_paced(fd, pkt, sizeof(pkt));
}

void
borg_serial_send_mvp(const float mvp[16])
{
   int fd = borg_serial_open();
   if (fd < 0) return;

   uint8_t pkt[66];
   pkt[0] = BORG_MARKER_MVP;
   memcpy(&pkt[1], mvp, 64);

   uint8_t csum = 0;
   for (int i = 1; i <= 64; i++) csum ^= pkt[i];
   pkt[65] = csum;
   borg_serial_write_paced(fd, pkt, sizeof(pkt));

   /* Steady frame pacing — same logic as borgvk_serial.c. */
   const char *ms_env = getenv("BORGVK_FRAME_MS");
   if (ms_env && ms_env[0]) {
      long period_ns = atol(ms_env) * 1000000L;
      if (period_ns > 0) {
         static struct timespec next;
         static int armed;
         struct timespec now;
         clock_gettime(CLOCK_MONOTONIC, &now);
         if (armed) {
            long wait_ns = (next.tv_sec  - now.tv_sec)  * 1000000000L +
                           (next.tv_nsec - now.tv_nsec);
            if (wait_ns > 0) {
               struct timespec ts = { wait_ns / 1000000000L,
                                      wait_ns % 1000000000L };
               nanosleep(&ts, NULL);
            } else {
               next = now;
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
