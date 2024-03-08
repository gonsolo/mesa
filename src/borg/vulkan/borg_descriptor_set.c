/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_entrypoints.h"

VKAPI_ATTR VkResult VKAPI_CALL
borg_AllocateDescriptorSets(VkDevice device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   // TODO

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateDescriptorPool(VkDevice _device,
                           const VkDescriptorPoolCreateInfo *pCreateInfo,
                           const VkAllocationCallbacks *pAllocator,
                           VkDescriptorPool *pDescriptorPool)
{
   // TODO
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_UpdateDescriptorSets(VkDevice device,
                          uint32_t descriptorWriteCount,
                          const VkWriteDescriptorSet *pDescriptorWrites,
                          uint32_t descriptorCopyCount,
                          const VkCopyDescriptorSet *pDescriptorCopies)
{
   // TODO
}

VKAPI_ATTR void VKAPI_CALL
borg_DestroyDescriptorPool(VkDevice device,
                           VkDescriptorPool _pool,
                           const VkAllocationCallbacks *pAllocator)

{
   // TODO
}
