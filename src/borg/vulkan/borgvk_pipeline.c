/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * Pipelines for borgvk. The cube's vertex/fragment shaders are already
 * hand-compiled to SPIR-B in the firmware (vert_borg/frag_borg), so there is no
 * host-side shader compilation: a VkPipeline is an opaque placeholder that lets
 * pipeline creation and binding succeed. The actual draw is realized at submit
 * time by shipping the MVP to the FPGA.
 */
#include "borgvk_private.h"
#include "vk_render_pass.h"

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_format.h"

static VkResult
borgvk_create_pipeline(struct borgvk_device *device,
                       const VkAllocationCallbacks *pAllocator,
                       VkPipeline *pPipeline)
{
   struct borgvk_pipeline *pipeline =
      vk_object_zalloc(&device->vk, pAllocator, sizeof(*pipeline),
                       VK_OBJECT_TYPE_PIPELINE);
   if (!pipeline)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pPipeline = borgvk_pipeline_to_handle(pipeline);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateGraphicsPipelines(VkDevice _device, VkPipelineCache cache,
                               uint32_t count,
                               const VkGraphicsPipelineCreateInfo *pCreateInfos,
                               const VkAllocationCallbacks *pAllocator,
                               VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VkResult result = VK_SUCCESS;
   uint32_t i;

   for (i = 0; i < count; i++) {
      device->compile_layout = pCreateInfos[i].layout;
      const VkPipelineVertexInputStateCreateInfo *vi = pCreateInfos[i].pVertexInputState;
      uint32_t vfetch = 0;
      if (vi && vi->vertexAttributeDescriptionCount > 0) {
         vfetch = 0x80000000u;
         for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++) {
            const VkVertexInputBindingDescription *vb = &vi->pVertexBindingDescriptions[b];
            if (vb->inputRate != VK_VERTEX_INPUT_RATE_INSTANCE)
               continue;
            for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount; a++) {
               const VkVertexInputAttributeDescription *va = &vi->pVertexAttributeDescriptions[a];
               if (va->binding == vb->binding && va->location < 16)
                  vfetch |= 1u << va->location;
            }
         }
      }

      const bool points = pCreateInfos[i].pInputAssemblyState &&
                          pCreateInfos[i].pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      if (points)
         vfetch |= 0xC0000000u;   /* fetching, and each point is a quad of two triangles */

      /* Compile each shader stage's SPIR-V → NIR → Borg ISA (borgc). The blobs land in the
       * device (the cube's path uploads those) and are kept in the pipeline for generic draws. */
      for (int st = 0; st < BORGVK_SHADER_STAGE_COUNT; st++)
         device->shader_blob[st].len = 0;
      const VkPipelineRenderingCreateInfo *ri = vk_get_pipeline_rendering_create_info(&pCreateInfos[i]);
      /* Several colour attachments: every one an R8 format, whose class the fragment stage packs
       * (1 UNORM, 2 SNORM, 3 UINT or SINT). */
      uint32_t color_count = ri ? ri->colorAttachmentCount : 0, mrt_opt = 0;
      bool mrt = color_count >= 2 && color_count <= 4;
      for (uint32_t k = 0; mrt && k < color_count; k++) {
         const VkFormat f = ri->pColorAttachmentFormats[k];
         const uint32_t cls = f == VK_FORMAT_R8_UNORM ? 1 : f == VK_FORMAT_R8_SNORM ? 2 :
                              f == VK_FORMAT_R8_UINT || f == VK_FORMAT_R8_SINT ? 3 : 0;
         mrt = cls != 0;
         mrt_opt |= cls << (12 + 3 * k);
      }
      const uint32_t frag_opt =
         mrt ? 0x100u | ((color_count - 1) << 9) | mrt_opt :
         ri && ri->colorAttachmentCount && borgvk_sim_bytes_packed(ri->pColorAttachmentFormats[0]) ? 1 :
         ri && ri->colorAttachmentCount && ri->pColorAttachmentFormats[0] == VK_FORMAT_R8_UNORM ? 2 :
         ri && ri->colorAttachmentCount && borgvk_sim_packed_opt(ri->pColorAttachmentFormats[0]) ? borgvk_sim_packed_opt(ri->pColorAttachmentFormats[0]) :
         ri && ri->colorAttachmentCount && borgvk_sim_half1(ri->pColorAttachmentFormats[0]) ? 0x28 :
         ri && ri->colorAttachmentCount && borgvk_sim_half2(ri->pColorAttachmentFormats[0]) ? 8 :
         ri && ri->colorAttachmentCount && vk_format_has_alpha(ri->pColorAttachmentFormats[0]) ? 4 : 0;   /* 4: alpha is an output */
      for (uint32_t s = 0; s < pCreateInfos[i].stageCount; s++)
         borgvk_compile_stage(device, pCreateInfos[i].pStages[s].stage == VK_SHADER_STAGE_FRAGMENT_BIT ? frag_opt : vfetch,
                              &pCreateInfos[i].pStages[s]);
      if (points) {
         /* gl_PointCoord: the fragment stage reads two varyings placed after the vertex stage's own. */
         const VkPipelineShaderStageCreateInfo *vs = NULL, *fs = NULL;
         for (uint32_t s = 0; s < pCreateInfos[i].stageCount; s++) {
            if (pCreateInfos[i].pStages[s].stage == VK_SHADER_STAGE_VERTEX_BIT)
               vs = &pCreateInfos[i].pStages[s];
            if (pCreateInfos[i].pStages[s].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
               fs = &pCreateInfos[i].pStages[s];
         }
         if (vs && fs) {
            const uint32_t base = borgvk_blob_num_varyings(&device->shader_blob[BORGVK_STAGE_VERT]);
            borgvk_compile_stage(device, frag_opt | (base << 24), fs);
            if (device->frag_reads_pntc && base + 2 <= 255) {
               borgvk_compile_stage(device, vfetch | 0x20000000u, vs);
            } else {
               borgvk_compile_stage(device, frag_opt, fs);
            }
         }
      }

      result = borgvk_create_pipeline(device, pAllocator, &pPipelines[i]);
      if (result != VK_SUCCESS) {
         pPipelines[i] = VK_NULL_HANDLE;
         break;
      }

      /* Capture rasterization state so the sim submit path can apply it. */
      VK_FROM_HANDLE(borgvk_pipeline, pl, pPipelines[i]);
      memcpy(pl->blob, device->shader_blob, sizeof(pl->blob));
      pl->vfetch = vfetch;
      pl->color_count = color_count;
      pl->mrt = mrt;
      const VkPipelineRasterizationStateCreateInfo *rs =
         pCreateInfos[i].pRasterizationState;
      pl->cull_mode  = rs ? rs->cullMode  : VK_CULL_MODE_NONE;
      pl->front_face = rs ? rs->frontFace : VK_FRONT_FACE_COUNTER_CLOCKWISE;
      const VkPipelineInputAssemblyStateCreateInfo *ia =
         pCreateInfos[i].pInputAssemblyState;
      pl->topology = ia ? ia->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      pl->restart  = ia && ia->primitiveRestartEnable;
      if (vi) {
         for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++) {
            const VkVertexInputBindingDescription *vb = &vi->pVertexBindingDescriptions[b];
            if (vb->binding < BORGVK_MAX_VERTEX_BINDINGS) {
               pl->binding_stride[vb->binding] = vb->stride;
               pl->binding_instance[vb->binding] = vb->inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
            }
         }
         for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount &&
                              pl->attr_count < BORGVK_MAX_VERTEX_ATTRS; a++) {
            const VkVertexInputAttributeDescription *va = &vi->pVertexAttributeDescriptions[a];
            pl->attrs[pl->attr_count++] = (struct borgvk_vertex_attr){
               .location = va->location, .binding = va->binding,
               .offset = va->offset, .format = va->format };
         }
      }
      /* With rasterizer discard the spec ignores the viewport, multisample, depth/stencil
       * and colour-blend pointers (they may be invalid), so never read them. */
      VkGraphicsPipelineCreateInfo ci = pCreateInfos[i];
      if (rs && rs->rasterizerDiscardEnable) {
         ci.pViewportState = NULL;
         ci.pMultisampleState = NULL;
         ci.pDepthStencilState = NULL;
         ci.pColorBlendState = NULL;
      }
      borgvk_blend_pack(ci.pColorBlendState, &device->blend_cfg, &device->blend_const);
      borgvk_state_pack(&ci, device->state_reg);
      if (points)
         device->state_reg[4] &= ~3u;   /* points are not culled */
      device->state_valid = true;
      pl->blend_cfg = device->blend_cfg;
      pl->blend_const = device->blend_const;
      memcpy(pl->state_reg, device->state_reg, sizeof(pl->state_reg));
      pl->state_valid = true;
   }
   for (; i < count; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateComputePipelines(VkDevice _device, VkPipelineCache cache,
                              uint32_t count,
                              const VkComputePipelineCreateInfo *pCreateInfos,
                              const VkAllocationCallbacks *pAllocator,
                              VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VkResult result = VK_SUCCESS;
   uint32_t i;

   for (i = 0; i < count; i++) {
      result = borgvk_create_pipeline(device, pAllocator, &pPipelines[i]);
      if (result != VK_SUCCESS) {
         pPipelines[i] = VK_NULL_HANDLE;
         break;
      }
      VK_FROM_HANDLE(borgvk_pipeline, pl, pPipelines[i]);
      device->compile_layout = pCreateInfos[i].layout;
      borgvk_compile_compute_stage(device, &pCreateInfos[i].stage, pl);
   }
   for (; i < count; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyPipeline(VkDevice _device, VkPipeline _pipeline,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_pipeline, pipeline, _pipeline);

   if (!pipeline)
      return;

   vk_object_free(&device->vk, pAllocator, pipeline);
}
