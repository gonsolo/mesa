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

   size_t off = 0;
   while (off < sizeof(pkt)) {
      ssize_t n = write(fd, &pkt[off], sizeof(pkt) - off);
      if (n < 0) {
         if (errno == EINTR)
            continue;
         mesa_logw("borgvk: serial write failed (%s)", strerror(errno));
         return;
      }
      off += (size_t)n;
   }

   /* The firmware aligns to packets via the inter-packet IDLE GAP. A blocking
    * app (cube.c) submits frames back-to-back, so without pacing the packets
    * stream gaplessly and the firmware locks onto the first one and never
    * re-syncs (cube freezes). Drain the TX, then idle briefly to guarantee a
    * gap; this also caps the frame rate sensibly (the FPGA renders far slower). */
   tcdrain(fd);
   usleep(3000);
}
