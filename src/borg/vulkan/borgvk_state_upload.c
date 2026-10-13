/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "drm-uapi/borg_drm.h"

#include "borgvk_drm_compat.h"

/* Locate the bound descriptor set (binding 0 = UBO, binding 1 = texture). */
struct borgvk_descriptor_set *
borgvk_find_set(struct vk_queue_submit *submit)
{
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct vk_command_buffer *cb = submit->command_buffers[i];
      list_for_each_entry(struct vk_cmd_queue_entry, e, &cb->cmd_queue.cmds, cmd_link) {
         if (e->type != VK_CMD_BIND_DESCRIPTOR_SETS)
            continue;
         const struct vk_cmd_bind_descriptor_sets *b = &e->u.bind_descriptor_sets;
         if (b->descriptor_set_count == 0 || b->descriptor_sets == NULL)
            continue;
         VK_FROM_HANDLE(borgvk_descriptor_set, set, b->descriptor_sets[0]);
         if (set)
            return set;
      }
   }
   return NULL;
}

static const uint32_t default_sampler[4] = { (2u << 3) | (2u << 6) | (2u << 9), 0, 0, 0 };

const uint32_t *
borgvk_sampler_desc(const struct borgvk_sampler *sampler)
{
   return sampler ? sampler->desc : default_sampler;
}

void
borgvk_send_blend_state(const struct borgvk_device *device)
{
   if (device->state_valid) {
      if (device->drm_fd >= 0) {
         struct drm_borg_state s;
         memcpy(s.reg, device->state_reg, sizeof(s.reg));
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_STATE, &s) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_STATE failed");
      } else {
         borgvk_serial_send_state(device->state_reg);
      }
   }
   if (device->push_lo < device->push_hi) {
      const uint32_t n = device->push_hi - device->push_lo;
      if (device->drm_fd >= 0) {
         struct drm_borg_push p = { .off_words = device->push_lo, .n_words = n };
         memcpy(p.words, device->push_words + device->push_lo, n * 4);
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_PUSH, &p) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_PUSH failed");
      } else {
         borgvk_serial_send_push_constants(device->push_lo * 4, n * 4,
                                           device->push_words + device->push_lo);
      }
   }
   if (device->vp.valid && device->drm_fd < 0)
      borgvk_serial_send_viewport(&device->vp);
   if (!device->blend_cfg)
      return;
   if (device->drm_fd >= 0) {
      struct drm_borg_blend b = { .cfg = device->blend_cfg,
                                  .constant = device->blend_const };
      if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_BLEND, &b) != 0)
         mesa_logw("borgvk: DRM_IOCTL_BORG_BLEND failed");
   } else {
      borgvk_serial_send_blend(device->blend_cfg, device->blend_const);
   }
}

void
borgvk_upload_shaders(struct borgvk_device *device)
{
   for (int st = 0; st < BORGVK_SHADER_STAGE_COUNT; st++) {
      struct borgvk_shader_blob *b = &device->shader_blob[st];
      if (b->len == 0)
         continue;   /* stage not compiled (app didn't bind it) */
      if (device->drm_fd >= 0) {
         struct drm_borg_shader s = { .stage = (uint32_t)st, .len = b->len };
         memcpy(s.data, b->data, b->len);
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SHADER, &s) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_SHADER failed (stage %d)", st);
      } else {
         borgvk_serial_send_shader((uint8_t)st, b->data, b->len);
      }
   }
}
