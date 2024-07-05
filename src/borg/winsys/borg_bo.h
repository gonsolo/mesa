#ifndef BORG_BO // E: Unknown argument: '-mtls-dialect=gnu2'
#define BORG_BO 1      

#include <stdint.h>
#include <sys/mman.h>
#include "borg_ws_device.h"

#ifdef __cplusplus
#include <atomic>
using std::atomic_uint_fast32_t;
#else
#include <stdatomic.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct borg_ws_bo {
        uint64_t size;
        uint64_t map_handle;
        struct borg_ws_device *dev;
        uint32_t handle;
        atomic_uint_fast32_t refcnt;
};

struct borg_ws_bo * borg_ws_bo_new_locked(struct borg_ws_device *dev,
                                          uint64_t size);

struct borg_ws_bo *borg_ws_bo_new(struct borg_ws_device *dev,
                                  uint64_t size);
 
void *borg_ws_bo_map(struct borg_ws_bo *);

static inline void
borg_ws_bo_unmap(struct borg_ws_bo *bo, void *ptr)
{
        munmap(ptr, bo->size);
}

void borg_ws_bo_destroy(struct borg_ws_bo*);

#ifdef __cplusplus
}
#endif

#endif
