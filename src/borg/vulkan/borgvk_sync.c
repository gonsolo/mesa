/*
 * Copyright © 2026 Borg GPU project
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

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_sync.h"

static VkResult
borgvk_sync_init(struct vk_device *device, struct vk_sync *sync,
                 uint64_t initial_value)
{
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
   return VK_SUCCESS;
}

static VkResult
borgvk_sync_reset(struct vk_device *device, struct vk_sync *sync)
{
   return VK_SUCCESS;
}

static VkResult
borgvk_sync_wait_many(struct vk_device *device, uint32_t wait_count,
                      const struct vk_sync_wait *waits,
                      enum vk_sync_wait_flags wait_flags,
                      uint64_t abs_timeout_ns)
{
   /* Work is already complete (synchronous submit) — nothing to wait for. */
   return VK_SUCCESS;
}

const struct vk_sync_type borgvk_sync_type = {
   .size = sizeof(struct vk_sync),
   .features = VK_SYNC_FEATURE_BINARY |
               VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_ANY |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = borgvk_sync_init,
   .finish = borgvk_sync_finish,
   .signal = borgvk_sync_signal,
   .reset = borgvk_sync_reset,
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
