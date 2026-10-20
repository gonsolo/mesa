/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_common_entrypoints.h"
#include "vk_format.h"
#include "util/format/u_format.h"

void
borgvk_free_targets(struct borgvk_command_buffer *cmd)
{
   while (cmd->targets) {
      struct borgvk_target *t = cmd->targets;
      cmd->targets = t->next;
      free(t);
   }
}

/* A view of a mip level above 0 or a layer above 0 renders as the single-level, single-layer image it is. */
static struct vk_image_view *
target_view(struct borgvk_command_buffer *cmd, struct vk_image_view *view)
{
   if (!view || !view->image || (view->base_mip_level == 0 && view->base_array_layer == 0))
      return view;
   struct borgvk_image *img = container_of(view->image, struct borgvk_image, vk);
   if (img->vk.image_type == VK_IMAGE_TYPE_3D || view->layer_count != 1 || !img->mem)
      return view;
   for (struct borgvk_target *t = cmd->targets; t; t = t->next)
      if (t->orig == view)
         return &t->view;
   const uint32_t bs = vk_format_get_blocksize(img->vk.format);
   const uint32_t bw = vk_format_get_blockwidth(img->vk.format), bh = vk_format_get_blockheight(img->vk.format);
   if (bw != 1 || bh != 1)
      return view;
   struct borgvk_target *t = calloc(1, sizeof(*t));
   if (!t)
      return view;
   uint32_t stride;
   const uint64_t off = borgvk_image_level_offset(img, view->base_mip_level, view->base_array_layer, bs, 1, 1, 1, &stride);
   t->orig = view;
   t->image = *img;
   t->image.offset += off;
   t->image.size = img->size > off ? img->size - off : 0;
   t->image.vk.extent.width = MAX2(img->vk.extent.width >> view->base_mip_level, 1);
   t->image.vk.extent.height = MAX2(img->vk.extent.height >> view->base_mip_level, 1);
   t->image.vk.mip_levels = 1;
   t->image.vk.array_layers = 1;
   t->view = *view;
   t->view.image = &t->image.vk;
   t->view.base_mip_level = 0;
   t->view.base_array_layer = 0;
   t->next = cmd->targets;
   cmd->targets = t;
   return &t->view;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd, struct borgvk_command_buffer, vk);

   cmd->in_rendering = true;
   cmd->render_area = pRenderingInfo->renderArea;
   cmd->layer_count = MAX2(pRenderingInfo->layerCount, 1);
   cmd->color_attachment_count =
      MIN2(pRenderingInfo->colorAttachmentCount, BORGVK_MAX_COLOR_ATTACHMENTS);
   cmd->depth_view = NULL;
   cmd->stencil_view = NULL;
   /* A subpass without colour must not draw into the previous one's attachment. */
   memset(cmd->color_views, 0, sizeof(cmd->color_views));
   memset(cmd->color_resolve_views, 0, sizeof(cmd->color_resolve_views));

   uint32_t layerCount = cmd->layer_count;

   for (uint32_t i = 0; i < cmd->color_attachment_count; i++) {
      const VkRenderingAttachmentInfo *att = &pRenderingInfo->pColorAttachments[i];
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      VK_FROM_HANDLE(vk_image_view, resolve_view, att->resolveImageView);
      cmd->color_views[i] = target_view(cmd, view);
      cmd->color_resolve_views[i] = resolve_view;
      if (view && att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_COLOR_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
         if (i == 0) {
            cmd->has_clear = true;
            for (int c = 0; c < 4; c++)
               cmd->clear_color[c] = att->clearValue.color.float32[c];
         }
      }
   }

   if (pRenderingInfo->pDepthAttachment &&
       pRenderingInfo->pDepthAttachment->imageView != VK_NULL_HANDLE) {
      const VkRenderingAttachmentInfo *att = pRenderingInfo->pDepthAttachment;
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      cmd->depth_view = target_view(cmd, view);
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_DEPTH_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
   }

   if (pRenderingInfo->pStencilAttachment &&
       pRenderingInfo->pStencilAttachment->imageView != VK_NULL_HANDLE) {
      const VkRenderingAttachmentInfo *att = pRenderingInfo->pStencilAttachment;
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      cmd->stencil_view = target_view(cmd, view);
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_STENCIL_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   vk_common_CmdEndRendering(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRendering2EXT(VkCommandBuffer commandBuffer,
                           const VkRenderingEndInfoEXT *pRenderingEndInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd =
      container_of(vk_cmd, struct borgvk_command_buffer, vk);

   borgvk_flush_draws(cmd);

   for (uint32_t i = 0; i < cmd->color_attachment_count; i++) {
      struct vk_image_view *view = cmd->color_views[i];
      struct vk_image_view *resolve_view = cmd->color_resolve_views[i];
      if (!view || !resolve_view)
         continue;

      struct borgvk_image *src = container_of(view->image, struct borgvk_image, vk);
      struct borgvk_image *dst = container_of(resolve_view->image, struct borgvk_image, vk);
      const VkImageResolve region = {
         .srcSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = view->base_mip_level,
            .baseArrayLayer = view->base_array_layer,
            .layerCount = cmd->layer_count,
         },
         .srcOffset = { cmd->render_area.offset.x, cmd->render_area.offset.y, 0 },
         .dstSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = resolve_view->base_mip_level,
            .baseArrayLayer = resolve_view->base_array_layer,
            .layerCount = cmd->layer_count,
         },
         .dstOffset = { cmd->render_area.offset.x, cmd->render_area.offset.y, 0 },
         .extent = {
            cmd->render_area.extent.width, cmd->render_area.extent.height, 1
         },
      };
      borgvk_resolve_region(src, dst, &region);
   }

   cmd->in_rendering = false;
   cmd->color_attachment_count = 0;
   cmd->depth_view = NULL;
   cmd->stencil_view = NULL;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBeginRenderPass(VkCommandBuffer commandBuffer,
                          const VkRenderPassBeginInfo *pRenderPassBegin,
                          VkSubpassContents contents)
{
   vk_common_CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBeginRenderPass2(VkCommandBuffer commandBuffer,
                           const VkRenderPassBeginInfo *pRenderPassBegin,
                           const VkSubpassBeginInfo *pSubpassBeginInfo)
{
   vk_common_CmdBeginRenderPass2(commandBuffer, pRenderPassBegin, pSubpassBeginInfo);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents)
{
   vk_common_CmdNextSubpass(commandBuffer, contents);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdNextSubpass2(VkCommandBuffer commandBuffer,
                       const VkSubpassBeginInfo *pSubpassBeginInfo,
                       const VkSubpassEndInfo *pSubpassEndInfo)
{
   vk_common_CmdNextSubpass2(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRenderPass(VkCommandBuffer commandBuffer)
{
   vk_common_CmdEndRenderPass(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdEndRenderPass2(VkCommandBuffer commandBuffer,
                         const VkSubpassEndInfo *pSubpassEndInfo)
{
   vk_common_CmdEndRenderPass2(commandBuffer, pSubpassEndInfo);
}
