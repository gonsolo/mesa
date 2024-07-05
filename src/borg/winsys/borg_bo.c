#include <sys/mman.h>
#include "drm-uapi/borg_drm.h"
#include "borg_bo.h"
#include "util/u_memory.h"

struct borg_ws_bo * borg_ws_bo_new_locked(struct borg_ws_device *dev,
                                          uint64_t size)
{
        puts("Mesa Borg: borg_ws_bo_new_locked.");

        struct drm_borg_gem_new req = {};

        printf("  size: %li\n", size);

        req.info.size = size;

        puts("Mesa Borg: drmCommandWriteRead");
        int ret = drmCommandWriteRead(dev->fd, DRM_BORG_GEM_NEW, &req, sizeof(req));
        if (ret != 0) {
                puts("Mesa Borg: drmCommandWriteRead NULL");
                return NULL;
        }
        puts("Mesa Borg: drmCommandWriteRead successful");

        struct borg_ws_bo *bo = CALLOC_STRUCT(borg_ws_bo);
        bo->size = size;
        bo->handle = req.info.handle;
        bo->map_handle = req.info.map_handle;
        bo->dev = dev;
        bo->refcnt = 1;

        return bo;
}

struct borg_ws_bo *borg_ws_bo_new(struct borg_ws_device *dev,
                                  uint64_t size)
{
   struct borg_ws_bo *bo;

   puts("Mesa Borg: borg_ws_bo_new.");
  
   simple_mtx_lock(&dev->bos_lock);
   bo = borg_ws_bo_new_locked(dev, size);
   simple_mtx_unlock(&dev->bos_lock);
 
   return bo;
}

void *borg_ws_bo_map(struct borg_ws_bo *bo)
{
        puts("Mesa Borg: borg_ws_bo_map.");

        int prot = PROT_READ | PROT_WRITE;
        int map_flags = MAP_SHARED;

        printf("Mesa Borg: Before mmap. bo: %p, size: %li, dev: %p, fd: %i\n",
                bo, bo->size, bo->dev, bo->dev->fd);
        void *res = mmap(NULL, bo->size, prot, map_flags,
                      bo->dev->fd, bo->map_handle);
        if (res == MAP_FAILED) {
                puts("Mesa Borg: mmap failed.");
                return NULL;
        }

        return res;
}


static bool
atomic_dec_not_one(atomic_uint_fast32_t *counter)
{
        uint_fast32_t old = *counter;
        while (1) {
                assert(old != 0);
                if (old == 1)
                        return false;

                if (atomic_compare_exchange_weak(counter, &old, old - 1))
                        return true;
        }
}

void borg_ws_bo_destroy(struct borg_ws_bo *bo)
{
        if (atomic_dec_not_one(&bo->refcnt))
                return;
     
        struct borg_ws_device *dev = bo->dev;
        
        /* Lock the device before we drop the final reference */
        simple_mtx_lock(&dev->bos_lock);
        
        if (--bo->refcnt == 0) {
                // TODO _mesa_hash_table_remove_key(dev->bos, (void *)(uintptr_t)bo->handle);
        
                drmCloseBufferHandle(bo->dev->fd, bo->handle);
                FREE(bo);
        }
  
        simple_mtx_unlock(&dev->bos_lock);
}

