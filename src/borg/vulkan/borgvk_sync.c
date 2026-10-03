/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * Synchronization for borgvk. vkQueueSubmit ships the frame to the FPGA over
 * serial and returns once it's been sent, so from the host's perspective all
 * GPU work is already complete by the time anyone waits. That makes every sync
 * operation trivial: init/signal/reset/wait all succeed immediately. This one
 * type advertises every feature fences and (binary) semaphores need, so no
 * timeline/binary wrappers are required.
 */
#include "borgvk_private.h"

#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_sync.h"

/* ---- Binary sync (fences, binary semaphores) -------------------------------------------- *
 * A signalled flag per object under one process-wide lock and condition variable: waiters are
 * few and short-lived, and a single condvar lets wait_many sleep on several syncs at once.
 * Real state matters now that submission can be threaded (vk_device_enable_threaded_submit):
 * a fence or semaphore must stay unsignalled until the queue has actually run the work, and
 * a wait must block until then. "GPU" waits and signals are done by borgvk_queue_submit itself
 * (borgvk's GPU is driven from the submit thread, in order): it waits on submit->waits before
 * running the work and signals submit->signals after. A wait on a sync whose signal has not
 * been submitted yet just blocks until it is, which is what WAIT_PENDING asks for. */
struct borgvk_sync {
   struct vk_sync base;
   bool signaled;
};

static pthread_mutex_t borgvk_sync_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t borgvk_sync_cond;
static pthread_once_t borgvk_sync_once = PTHREAD_ONCE_INIT;

static void
borgvk_sync_cond_init(void)
{
   pthread_condattr_t attr;
   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);   /* vk_sync timeouts are CLOCK_MONOTONIC */
   pthread_cond_init(&borgvk_sync_cond, &attr);
   pthread_condattr_destroy(&attr);
}

static VkResult
borgvk_sync_init(struct vk_device *device, struct vk_sync *sync,
                 uint64_t initial_value)
{
   struct borgvk_sync *s = container_of(sync, struct borgvk_sync, base);
   pthread_once(&borgvk_sync_once, borgvk_sync_cond_init);
   s->signaled = initial_value != 0;
   return VK_SUCCESS;
}

static void
borgvk_sync_finish(struct vk_device *device, struct vk_sync *sync)
{
}

static VkResult
borgvk_sync_signal(struct vk_device *device, struct vk_sync *sync,
                   uint64_t value)
{
   struct borgvk_sync *s = container_of(sync, struct borgvk_sync, base);
   pthread_mutex_lock(&borgvk_sync_lock);
   s->signaled = true;
   pthread_cond_broadcast(&borgvk_sync_cond);
   pthread_mutex_unlock(&borgvk_sync_lock);
   return VK_SUCCESS;
}

static VkResult
borgvk_sync_reset(struct vk_device *device, struct vk_sync *sync)
{
   struct borgvk_sync *s = container_of(sync, struct borgvk_sync, base);
   pthread_mutex_lock(&borgvk_sync_lock);
   s->signaled = false;
   pthread_mutex_unlock(&borgvk_sync_lock);
   return VK_SUCCESS;
}

/* Moves the signalled state from src to dst and resets src (required of binary types
 * once the device can submit on a thread). */
static VkResult
borgvk_sync_move(struct vk_device *device, struct vk_sync *dst,
                 struct vk_sync *src)
{
   struct borgvk_sync *d = container_of(dst, struct borgvk_sync, base);
   struct borgvk_sync *s = container_of(src, struct borgvk_sync, base);
   pthread_mutex_lock(&borgvk_sync_lock);
   d->signaled = s->signaled;
   s->signaled = false;
   pthread_cond_broadcast(&borgvk_sync_cond);
   pthread_mutex_unlock(&borgvk_sync_lock);
   return VK_SUCCESS;
}

static VkResult
borgvk_sync_wait_many(struct vk_device *device, uint32_t wait_count,
                      const struct vk_sync_wait *waits,
                      enum vk_sync_wait_flags wait_flags,
                      uint64_t abs_timeout_ns)
{
   const bool any = (wait_flags & VK_SYNC_WAIT_ANY) != 0;
   struct timespec ts = {
      .tv_sec = abs_timeout_ns / 1000000000ull,
      .tv_nsec = abs_timeout_ns % 1000000000ull,
   };
   VkResult result = VK_SUCCESS;

   pthread_mutex_lock(&borgvk_sync_lock);
   for (;;) {
      uint32_t n_signaled = 0;
      for (uint32_t i = 0; i < wait_count; i++) {
         const struct borgvk_sync *s =
            container_of(waits[i].sync, struct borgvk_sync, base);
         n_signaled += s->signaled;
      }
      if (any ? n_signaled > 0 : n_signaled == wait_count)
         break;
      /* abs_timeout_ns == UINT64_MAX means "forever". */
      int r = (abs_timeout_ns == UINT64_MAX)
         ? pthread_cond_wait(&borgvk_sync_cond, &borgvk_sync_lock)
         : pthread_cond_timedwait(&borgvk_sync_cond, &borgvk_sync_lock, &ts);
      if (r == ETIMEDOUT) {
         result = VK_TIMEOUT;
         break;
      }
   }
   pthread_mutex_unlock(&borgvk_sync_lock);
   return result;
}

const struct vk_sync_type borgvk_sync_type = {
   .size = sizeof(struct borgvk_sync),
   .features = VK_SYNC_FEATURE_BINARY |
               VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_GPU_MULTI_WAIT |
               VK_SYNC_FEATURE_WAIT_PENDING |
               VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_ANY,
   .init = borgvk_sync_init,
   .finish = borgvk_sync_finish,
   .signal = borgvk_sync_signal,
   .reset = borgvk_sync_reset,
   .move = borgvk_sync_move,
   .wait_many = borgvk_sync_wait_many,
};


/* ---- Events ---------------------------------------------------------- *
 * VkEvent is a separate mandatory core-1.0 object (not a vk_sync type), and
 * was entirely missing -- same story as vkCreateBufferView: a null dispatch
 * slot that SEGVs the moment CTS's command_buffers.* tests call through it.
 * Host-only storage, modeled on lavapipe's lvp_event: Borg has no device-side
 * event wait/signal, and every command-buffer op here is already complete by
 * the time an app can observe it (see borgvk_sync_wait_many above). */

struct borgvk_event {
   struct vk_object_base base;
   volatile uint64_t event_storage;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateEvent(VkDevice _device, const VkEventCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_event *event = vk_alloc2(&device->vk.alloc, pAllocator,
                                          sizeof(*event), 8,
                                          VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!event)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   vk_object_base_init(&device->vk, &event->base, VK_OBJECT_TYPE_EVENT);
   event->event_storage = 0;

   *pEvent = borgvk_event_to_handle(event);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyEvent(VkDevice _device, VkEvent _event,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_event, event, _event);

   if (!event)
      return;

   vk_object_base_finish(&event->base);
   vk_free2(&device->vk.alloc, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_GetEventStatus(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(borgvk_event, event, _event);
   return event->event_storage ? VK_EVENT_SET : VK_EVENT_RESET;
}

/* Used by the queue when it replays vkCmdSetEvent / vkCmdResetEvent at submit. */
void
borgvk_event_set_status(VkEvent _event, bool set)
{
   VK_FROM_HANDLE(borgvk_event, event, _event);
   if (event)
      event->event_storage = set ? 1 : 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_SetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(borgvk_event, event, _event);
   event->event_storage = 1;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_ResetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(borgvk_event, event, _event);
   event->event_storage = 0;
   return VK_SUCCESS;
}

/* ---- Event commands, run when the command buffer is submitted ---------------------------- *
 * (cmd_dispatch in borgvk_device.c; the application-facing table only records them.) The v1
 * and v2 forms are both implemented because the runtime records whichever the app called. */
VKAPI_ATTR void VKAPI_CALL
borgvk_CmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
   borgvk_event_set_status(event, true);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
   borgvk_event_set_status(event, false);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent event, const VkDependencyInfo *pDependencyInfo)
{
   borgvk_event_set_status(event, true);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags2 stageMask)
{
   borgvk_event_set_status(event, false);
}
