#include "borg_ws_device.h"

#include "util/u_memory.h"

#include <xf86drm.h>

struct borg_ws_device *
borg_ws_device_new(drmDevicePtr drm_device)
{
        // TODO
        struct borg_ws_device *device = CALLOC_STRUCT(borg_ws_device);

        // TODO
        simple_mtx_init(&device->bos_lock, mtx_plain);
        // TODO
        return NULL;
}
