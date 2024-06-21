#include "borg_bo.h"

struct borg_ws_bo * borg_ws_bo_new_locked(struct borg_ws_device *dev)
{
        // TODO
        return NULL;
}

struct borg_ws_bo *borg_ws_bo_new(struct borg_ws_device *dev)
{
   struct borg_ws_bo *bo;
  
   simple_mtx_lock(&dev->bos_lock);
   bo = borg_ws_bo_new_locked(dev);
   simple_mtx_unlock(&dev->bos_lock);
 
   return bo;
}

