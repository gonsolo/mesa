/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_descriptor_set.h"
#include "borg_descriptor_set_layout.h"
#include "borg_device.h"
#include "borg_entrypoints.h"
#include "borg_physical_device.h"

#include "vk_log.h"
#include "vk_util.h"

static VkResult
borg_descriptor_set_create(struct borg_device *dev,
                          struct borg_descriptor_pool *pool,
                          struct borg_descriptor_set_layout *layout,
                          uint32_t variable_count,
                          struct borg_descriptor_set **out_set)
{
   struct borg_descriptor_set *set;

   uint32_t dummy_mem_size = 32;

   set = vk_object_zalloc(&dev->vk, NULL, dummy_mem_size,
                          VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   // TODO

   *out_set = set;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_AllocateDescriptorSets(VkDevice device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(borg_device, dev, device);
   VK_FROM_HANDLE(borg_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   struct borg_descriptor_set *set = NULL;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_desc_count =
      vk_find_struct_const(pAllocateInfo->pNext,
                           DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   /* allocate a set of buffers for each shader to contain descriptors */
   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(borg_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      /* If descriptorSetCount is zero or this structure is not included in
       * the pNext chain, then the variable lengths are considered to be zero.
       */
      const uint32_t variable_count =
         var_desc_count && var_desc_count->descriptorSetCount > 0 ?
         var_desc_count->pDescriptorCounts[i] : 0;

      result = borg_descriptor_set_create(dev, pool, layout,
                                         variable_count, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = borg_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      borg_FreeDescriptorSets(device, pAllocateInfo->descriptorPool, i, pDescriptorSets);
      for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
         pDescriptorSets[i] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateDescriptorPool(VkDevice _device,
                           const VkDescriptorPoolCreateInfo *pCreateInfo,
                           const VkAllocationCallbacks *pAllocator,
                           VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(borg_device, dev, _device);
   struct borg_physical_device *pdev = borg_device_physical(dev);
   struct borg_descriptor_pool *pool;
   uint64_t size = sizeof(struct borg_descriptor_pool);
   uint64_t bo_size = 0;

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t max_align = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
            type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      borg_descriptor_stride_align_for_type(pdev,
                                           pCreateInfo->pPoolSizes[i].type,
                                           type_list, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
            type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      borg_descriptor_stride_align_for_type(pdev,
                                           pCreateInfo->pPoolSizes[i].type,
                                           type_list, &stride, &alignment);
      bo_size += MAX2(stride, max_align) *
                 pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   /* Individual descriptor sets are aligned to the min UBO alignment to
    * ensure that we don't end up with unaligned data access in any shaders.
    * This means that each descriptor buffer allocated may burn up to 16B of
    * extra space to get the right alignment.  (Technically, it's at most 28B
    * because we're always going to start at least 4B aligned but we're being
    * conservative here.)  Allocate enough extra space that we can chop it
    * into maxSets pieces and align each one of them to 32B.
    */
   uint32_t cbuf_alignment = 32;
   bo_size += cbuf_alignment * pCreateInfo->maxSets;

   uint64_t entries_size = sizeof(struct borg_descriptor_pool_entry) *
                           pCreateInfo->maxSets;
   size += entries_size;

   pool = vk_object_zalloc(&dev->vk, pAllocator, size,
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   // TODO: borg_ws_bo_new_mapped

   pool->size = bo_size;
   pool->max_entry_count = pCreateInfo->maxSets;

   *pDescriptorPool = borg_descriptor_pool_to_handle(pool);
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
