#ifndef BORG_BO // E: Unknown argument: '-mtls-dialect=gnu2'
#define BORG_BO 1      

#include <stdint.h>
#include "borg_ws_device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct borg_ws_bo * borg_ws_bo_new_locked(struct borg_ws_device *dev);

struct borg_ws_bo *borg_ws_bo_new(struct borg_ws_device *dev);

#ifdef __cplusplus
}
#endif

#endif
