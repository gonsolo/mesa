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
