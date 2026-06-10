/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Pipelines for borgvk. The cube's vertex/fragment shaders are already
 * hand-compiled to SPIR-B in the firmware (vert_borg/frag_borg), so there is no
 * host-side shader compilation: a VkPipeline is an opaque placeholder that lets
 * pipeline creation and binding succeed. The actual draw is realized at submit
 * time by shipping the MVP to the FPGA.
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_log.h"

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
      /* Compile each shader stage's SPIR-V → NIR → Borg ISA (borgc). */
      for (uint32_t s = 0; s < pCreateInfos[i].stageCount; s++)
         borgvk_compile_stage(device, &pCreateInfos[i].pStages[s]);

      result = borgvk_create_pipeline(device, pAllocator, &pPipelines[i]);
      if (result != VK_SUCCESS) {
         pPipelines[i] = VK_NULL_HANDLE;
         break;
      }
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
