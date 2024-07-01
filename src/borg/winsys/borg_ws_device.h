#ifndef BORG_DEVICE
#define BORG_DEVICE 1

#include "util/simple_mtx.h"

#include <xf86drm.h>

struct borg_ws_device {

        int fd;

        simple_mtx_t bos_lock;
};
 
struct borg_ws_device *
borg_ws_device_new(drmDevicePtr drm_device);

#endif
