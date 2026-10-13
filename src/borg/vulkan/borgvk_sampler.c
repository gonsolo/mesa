/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_format.h"

static void
borgvk_pack_sampler(const VkSamplerCreateInfo *info, uint32_t desc[4])
{
   union { float f; uint32_t u; } bias = { info->mipLodBias },
                                  lo = { info->minLod }, hi = { info->maxLod };
   desc[0] = (info->magFilter == VK_FILTER_LINEAR ? 1u << 0 : 0) |
             (info->minFilter == VK_FILTER_LINEAR ? 1u << 1 : 0) |
             (info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? 1u << 2 : 0) |
             ((uint32_t)info->addressModeU & 0x7) << 3 |
             ((uint32_t)info->addressModeV & 0x7) << 6 |
             ((uint32_t)info->addressModeW & 0x7) << 9 |
             ((uint32_t)info->borderColor & 0x7) << 12 |
             (info->compareEnable ? 1u << 15 : 0) |
             ((uint32_t)info->compareOp & 0x7) << 16 |
             (info->unnormalizedCoordinates ? 1u << 19 : 0);
   desc[1] = bias.u;
   desc[2] = lo.u;
   desc[3] = hi.u;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkSampler *pSampler)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_sampler *sampler;

   sampler = vk_sampler_create(&device->vk, pCreateInfo,
                               pAllocator, sizeof(*sampler));
   if (!sampler)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   borgvk_pack_sampler(pCreateInfo, sampler->desc);

   *pSampler = borgvk_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroySampler(VkDevice _device, VkSampler _sampler,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_sampler, sampler, _sampler);

   if (!sampler)
      return;

   vk_sampler_destroy(&device->vk, pAllocator, &sampler->vk);
}
