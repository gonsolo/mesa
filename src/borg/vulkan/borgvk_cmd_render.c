/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_common_entrypoints.h"
#include "vk_format.h"

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
      cmd->color_views[i] = view;
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
      cmd->depth_view = view;
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         borgvk_clear_attachment_rect(view, VK_IMAGE_ASPECT_DEPTH_BIT, &att->clearValue,
                                      cmd->render_area, 0, layerCount);
   }

   if (pRenderingInfo->pStencilAttachment &&
       pRenderingInfo->pStencilAttachment->imageView != VK_NULL_HANDLE) {
      const VkRenderingAttachmentInfo *att = pRenderingInfo->pStencilAttachment;
      VK_FROM_HANDLE(vk_image_view, view, att->imageView);
      cmd->stencil_view = view;
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
