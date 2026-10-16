/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "vk_blend.h"
#include "borg_nir_passes.h"
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

static VkResult
borgvk_build_graphics(struct borgvk_device *device, const VkAllocationCallbacks *pAllocator,
                      const VkGraphicsPipelineCreateInfo *cinfo, VkPipeline *out)
{
   VkResult result = VK_SUCCESS;
   device->compile_layout = (*cinfo).layout;
   const VkPipelineVertexInputStateCreateInfo *vi = (*cinfo).pVertexInputState;
   uint32_t vfetch = 0;
   if (vi && vi->vertexAttributeDescriptionCount > 0) {
      vfetch = BORGC_VS_FETCH;
      for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++) {
         const VkVertexInputBindingDescription *vb = &vi->pVertexBindingDescriptions[b];
         if (vb->inputRate != VK_VERTEX_INPUT_RATE_INSTANCE)
            continue;
         for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount; a++) {
            const VkVertexInputAttributeDescription *va = &vi->pVertexAttributeDescriptions[a];
            if (va->binding == vb->binding && va->location < 16)
               vfetch |= BORGC_VS_INSTANCED(va->location);
         }
      }
   }

   const bool points = (*cinfo).pInputAssemblyState &&
                       (*cinfo).pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   if (points)
      vfetch |= BORGC_VS_FETCH | BORGC_VS_POINT_QUADS;
   const VkPrimitiveTopology topology_in =
      (*cinfo).pInputAssemblyState ? (*cinfo).pInputAssemblyState->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   const bool lines = topology_in == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || topology_in == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
   if (lines)
      vfetch |= BORGC_VS_FETCH | BORGC_VS_LINES_BIT;

   for (int st = 0; st < BORGVK_SHADER_STAGE_COUNT; st++)
      device->shader_blob[st].len = 0;
   const VkPipelineRenderingCreateInfo *ri = vk_get_pipeline_rendering_create_info(&(*cinfo));
   uint32_t color_count = ri ? ri->colorAttachmentCount : 0, mrt_opt = 0;
   bool mrt = color_count >= 2 && color_count <= 4;
   for (uint32_t k = 0; mrt && k < color_count; k++) {
      const VkFormat f = ri->pColorAttachmentFormats[k];
      const uint32_t cls = f == VK_FORMAT_R8_UNORM ? 1 : f == VK_FORMAT_R8_SNORM ? 2 :
                           f == VK_FORMAT_R8_UINT || f == VK_FORMAT_R8_SINT ? 3 : 0;
      mrt = cls != 0;
      mrt_opt |= cls << (BORGC_FS_TARGET_CLASS_SHIFT + 3 * k);
   }
   const uint32_t frag_opt =
      mrt ? BORGC_FS_MULTI_TARGET | ((color_count - 1) << BORGC_FS_TARGETS_SHIFT) | mrt_opt :
      ri && ri->colorAttachmentCount && borgvk_sim_bytes_packed(ri->pColorAttachmentFormats[0]) ? BORGC_FS_BYTES_PACKED :
      ri && ri->colorAttachmentCount && ri->pColorAttachmentFormats[0] == VK_FORMAT_R8_UNORM ? BORGC_FS_R8_UNORM :
      ri && ri->colorAttachmentCount && borgvk_sim_packed_opt(ri->pColorAttachmentFormats[0]) ? borgvk_sim_packed_opt(ri->pColorAttachmentFormats[0]) :
      ri && ri->colorAttachmentCount && borgvk_sim_half1(ri->pColorAttachmentFormats[0]) ? BORGC_FS_HALF | BORGC_FS_HALF1 :
      ri && ri->colorAttachmentCount && borgvk_sim_half2(ri->pColorAttachmentFormats[0]) ? BORGC_FS_HALF :
      ri && ri->colorAttachmentCount && vk_format_has_alpha(ri->pColorAttachmentFormats[0]) ? BORGC_FS_ALPHA_OUTPUT : 0;   /* 4: alpha is an output */
   struct borg_blend blend = { 0 };
   device->blend_lower = NULL;
   {
      const VkPipelineRasterizationStateCreateInfo *rz = (*cinfo).pRasterizationState;
      const VkPipelineColorBlendStateCreateInfo *cb = rz && rz->rasterizerDiscardEnable ? NULL : (*cinfo).pColorBlendState;
      const VkFormat f0 = ri && ri->colorAttachmentCount ? ri->pColorAttachmentFormats[0] : VK_FORMAT_UNDEFINED;
      const bool fixed = f0 == VK_FORMAT_R8G8B8A8_UNORM || f0 == VK_FORMAT_B8G8R8A8_UNORM;
      if (cb && cb->attachmentCount >= 1 && cb->pAttachments && !mrt && f0 != VK_FORMAT_UNDEFINED && !fixed) {
         const VkPipelineColorBlendAttachmentState *a = &cb->pAttachments[0];
         const bool enable = a->blendEnable && !vk_format_is_int(f0);
         if (enable || (a->colorWriteMask & 0xF) != 0xF) {
            nir_lower_blend_rt *rt = &blend.opts.rt[0];
            rt->format = vk_format_to_pipe_format(f0);
            rt->colormask = a->colorWriteMask & 0xF;
            rt->rgb.func = enable ? vk_blend_op_to_pipe(a->colorBlendOp) : PIPE_BLEND_ADD;
            rt->rgb.src_factor = enable ? vk_blend_factor_to_pipe(a->srcColorBlendFactor) : PIPE_BLENDFACTOR_ONE;
            rt->rgb.dst_factor = enable ? vk_blend_factor_to_pipe(a->dstColorBlendFactor) : PIPE_BLENDFACTOR_ZERO;
            rt->alpha.func = enable ? vk_blend_op_to_pipe(a->alphaBlendOp) : PIPE_BLEND_ADD;
            rt->alpha.src_factor = enable ? vk_blend_factor_to_pipe(a->srcAlphaBlendFactor) : PIPE_BLENDFACTOR_ONE;
            rt->alpha.dst_factor = enable ? vk_blend_factor_to_pipe(a->dstAlphaBlendFactor) : PIPE_BLENDFACTOR_ZERO;
            for (unsigned k = 1; k < 8; k++)
               blend.opts.rt[k] = (nir_lower_blend_rt){ .format = PIPE_FORMAT_NONE, .colormask = 0xF,
                  .rgb = { PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO },
                  .alpha = { PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO } };
            memcpy(blend.konst, cb->blendConstants, sizeof(blend.konst));
            device->blend_lower = &blend;
         }
      }
   }
   for (uint32_t s = 0; s < (*cinfo).stageCount; s++)
      borgvk_compile_stage(device, (*cinfo).pStages[s].stage == VK_SHADER_STAGE_FRAGMENT_BIT ? frag_opt : vfetch,
                           &(*cinfo).pStages[s]);
   bool has_fs = false;
   for (uint32_t s = 0; s < (*cinfo).stageCount; s++)
      has_fs |= (*cinfo).pStages[s].stage == VK_SHADER_STAGE_FRAGMENT_BIT;
   if (!has_fs)
      borgvk_compile_empty_frag(device, frag_opt);
   if (points) {
      const VkPipelineShaderStageCreateInfo *vs = NULL, *fs = NULL;
      for (uint32_t s = 0; s < (*cinfo).stageCount; s++) {
         if ((*cinfo).pStages[s].stage == VK_SHADER_STAGE_VERTEX_BIT)
            vs = &(*cinfo).pStages[s];
         if ((*cinfo).pStages[s].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
            fs = &(*cinfo).pStages[s];
      }
      if (vs && fs) {
         const uint32_t base = borgvk_blob_num_varyings(&device->shader_blob[BORGVK_STAGE_VERT]);
         borgvk_compile_stage(device, frag_opt | (base << BORGC_FS_PNTC_BASE_SHIFT), fs);
         if (device->frag_reads_pntc && base + 2 <= 255) {
            borgvk_compile_stage(device, vfetch | BORGC_VS_PNTC_VARYINGS, vs);
         } else {
            borgvk_compile_stage(device, frag_opt, fs);
         }
      }
   }

   device->blend_lower = NULL;
   result = borgvk_create_pipeline(device, pAllocator, &(*out));
   if (result != VK_SUCCESS) {
      (*out) = VK_NULL_HANDLE;
      return result;
   }

   /* Capture rasterization state so the sim submit path can apply it. */
   VK_FROM_HANDLE(borgvk_pipeline, pl, (*out));
   memcpy(pl->blob, device->shader_blob, sizeof(pl->blob));
   pl->vfetch = vfetch;
   pl->tex_binding = device->frag_tex_binding;
   borgvk_layout_set_base(device, pl->set_base);
   pl->ubo_mask = device->stage_ubo[0] | device->stage_ubo[1];
   pl->push_used = device->stage_push[0] || device->stage_push[1];
   pl->push_window = device->stage_pwin[0] || device->stage_pwin[1];
   pl->spills = device->frag_spills;
   pl->push_stage = (device->stage_push[0] && !device->stage_ubo[0] && !device->stage_pwin[0]) |
                    (device->stage_push[1] && !device->stage_ubo[1] && !device->stage_pwin[1]) << 1;
   pl->color_count = color_count;
   pl->mrt = mrt;
   const VkPipelineRasterizationStateCreateInfo *rs =
      (*cinfo).pRasterizationState;
   const VkPipelineInputAssemblyStateCreateInfo *ia =
      (*cinfo).pInputAssemblyState;
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
   VkGraphicsPipelineCreateInfo ci = (*cinfo);
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
   if (lines)
      device->state_reg[4] &= ~3u;
   device->state_valid = true;
   pl->srgb_table = blend.opts.rt[0].format != PIPE_FORMAT_NONE && util_format_is_srgb(blend.opts.rt[0].format);
   pl->blend_cfg = device->blend_cfg;
   pl->blend_const = device->blend_const;
   memcpy(pl->state_reg, device->state_reg, sizeof(pl->state_reg));
   pl->state_valid = true;
   pl->vp_dyn = pl->sc_dyn = false;
   for (uint32_t d = 0; ci.pDynamicState && d < ci.pDynamicState->dynamicStateCount; d++) {
      const VkDynamicState ds = ci.pDynamicState->pDynamicStates[d];
      pl->vp_dyn |= ds == VK_DYNAMIC_STATE_VIEWPORT || ds == VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT;
      pl->sc_dyn |= ds == VK_DYNAMIC_STATE_SCISSOR || ds == VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT;
   }
   memset(&pl->vp, 0, sizeof(pl->vp));
   if (ci.pViewportState && !rs->rasterizerDiscardEnable) {
      const VkPipelineViewportStateCreateInfo *vs = ci.pViewportState;
      if (vs->pViewports && vs->viewportCount && !pl->vp_dyn) {
         const VkViewport *v = &vs->pViewports[0];
         const float vv[6] = { v->x, v->y, v->width, v->height, v->minDepth, v->maxDepth };
         memcpy(pl->vp.v, vv, sizeof(vv));
         pl->vp.valid = true;
      }
      if (vs->pScissors && vs->scissorCount && !pl->sc_dyn) {
         const VkRect2D *r = &vs->pScissors[0];
         pl->vp.sc[0] = MAX2(r->offset.x, 0);
         pl->vp.sc[1] = MAX2(r->offset.y, 0);
         pl->vp.sc[2] = MAX2(r->offset.x + (int32_t)r->extent.width, 0);
         pl->vp.sc[3] = MAX2(r->offset.y + (int32_t)r->extent.height, 0);
         pl->vp.valid = true;
      }
   }
   return result;
}

static bool
mrt_formats(const VkPipelineRenderingCreateInfo *ri)
{
   for (uint32_t k = 0; k < ri->colorAttachmentCount; k++) {
      const VkFormat f = ri->pColorAttachmentFormats[k];
      if (f != VK_FORMAT_R8_UNORM && f != VK_FORMAT_R8_SNORM && f != VK_FORMAT_R8_UINT && f != VK_FORMAT_R8_SINT)
         return false;
   }
   return true;
}

/* Several wide colour targets: one pipeline per target, drawn one after the other. */
static VkResult
borgvk_build_emulated(struct borgvk_device *device, const VkAllocationCallbacks *pAllocator,
                      const VkGraphicsPipelineCreateInfo *ci, const VkPipelineRenderingCreateInfo *ri, VkPipeline *out)
{
   const VkPipelineColorBlendStateCreateInfo *cb = ci->pColorBlendState;
   VkPipeline sub[4] = { VK_NULL_HANDLE };
   VkResult result = VK_SUCCESS;
   for (uint32_t k = 0; k < ri->colorAttachmentCount && result == VK_SUCCESS; k++) {
      VkPipelineRenderingCreateInfo rk = *ri;
      rk.colorAttachmentCount = 1;
      rk.pColorAttachmentFormats = &ri->pColorAttachmentFormats[k];
      rk.pNext = ci->pNext;
      VkPipelineColorBlendStateCreateInfo cbk;
      VkGraphicsPipelineCreateInfo ck = *ci;
      ck.pNext = &rk;
      ck.renderPass = VK_NULL_HANDLE;
      ck.subpass = 0;
      if (cb && cb->attachmentCount > k) {
         cbk = *cb;
         cbk.attachmentCount = 1;
         cbk.pAttachments = &cb->pAttachments[k];
         ck.pColorBlendState = &cbk;
      }
      device->frag_select = k + 1;
      result = borgvk_build_graphics(device, pAllocator, &ck, &sub[k]);
   }
   device->frag_select = 0;
   if (result != VK_SUCCESS) {
      for (uint32_t k = 0; k < 4; k++)
         if (sub[k])
            borgvk_DestroyPipeline(borgvk_device_to_handle(device), sub[k], pAllocator);
      *out = VK_NULL_HANDLE;
      return result;
   }
   VK_FROM_HANDLE(borgvk_pipeline, first, sub[0]);
   first->emul_n = ri->colorAttachmentCount;
   for (uint32_t k = 1; k < first->emul_n; k++)
      first->emul[k - 1] = sub[k];
   *out = sub[0];
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
      const VkPipelineRenderingCreateInfo *ri = vk_get_pipeline_rendering_create_info(&pCreateInfos[i]);
      const bool emulate = ri && ri->colorAttachmentCount >= 2 && ri->colorAttachmentCount <= 4 && !mrt_formats(ri);
      result = emulate ? borgvk_build_emulated(device, pAllocator, &pCreateInfos[i], ri, &pPipelines[i])
                       : borgvk_build_graphics(device, pAllocator, &pCreateInfos[i], &pPipelines[i]);
      if (result != VK_SUCCESS)
         break;
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

   for (uint32_t k = 0; k < 3; k++)
      if (pipeline->emul[k])
         borgvk_DestroyPipeline(_device, pipeline->emul[k], pAllocator);
   vk_object_free(&device->vk, pAllocator, pipeline);
}
