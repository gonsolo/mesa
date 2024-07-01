#include "borg_ws_device.h"

#include "util/u_memory.h"

#include <fcntl.h>
#include <xf86drm.h>

struct borg_ws_device *
borg_ws_device_new(drmDevicePtr drm_device)
{
        puts("Mesa Borg: borg_ws_device_new.");

        const char *path = drm_device->nodes[DRM_NODE_RENDER];
        struct borg_ws_device *device = CALLOC_STRUCT(borg_ws_device);
        drmVersionPtr ver = NULL;

        int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0)
                goto out_open;
        puts("Mesa Borg: borg_ws_device_new: fd ok.");

        ver = drmGetVersion(fd);
        if (!ver)
                goto out_err;
        puts("Mesa Borg: borg_ws_device_new: ver ok.");

        if (strncmp("borg", ver->name, ver->name_len) != 0) {
                fprintf(stderr,
                        "DRM kernel driver '%.*s' in use. Borg requires borg.\n",
                        ver->name_len, ver->name);
                goto out_err;
        }
        puts("Mesa Borg: borg_ws_device_new: borg on borg ok.");

        device->fd = fd;

        simple_mtx_init(&device->bos_lock, mtx_plain);
out_err:
        if (ver)
                drmFreeVersion(ver);
out_open:
        FREE(device);
        close(fd);
        return NULL;
}
