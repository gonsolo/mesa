/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
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

static _Thread_local bool     borgvk_capture_active = false;
static _Thread_local uint8_t *borgvk_capture_buf    = NULL;
static _Thread_local size_t   borgvk_capture_len    = 0;
static _Thread_local size_t   borgvk_capture_cap    = 0;

void
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

bool
borgvk_transport_capturing(void)
{
   return borgvk_capture_active;
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
