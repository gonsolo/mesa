/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * Descriptor set layouts, pools and sets for borgvk. We don't drive shaders
 * with descriptors on the host; a set simply records which buffer is bound at
 * each binding so the submit path can locate the cube's uniform buffer
 * (binding 0) and read the per-frame MVP from its mapped memory.
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_descriptor_set_layout.h"
#include "vk_log.h"

/* ---- Descriptor set layout -------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateDescriptorSetLayout(VkDevice _device,
                                 const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                 const VkAllocationCallbacks *pAllocator,
                                 VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_descriptor_set_layout *layout;

   layout = vk_descriptor_set_layout_zalloc(&device->vk, sizeof(*layout),
                                            pCreateInfo);
   if (!layout)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->binding_count = pCreateInfo->bindingCount;

   *pSetLayout = borgvk_descriptor_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

/* DestroyDescriptorSetLayout is the common (refcounted) entrypoint. */

/* ---- Descriptor pool -------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateDescriptorPool(VkDevice _device,
                            const VkDescriptorPoolCreateInfo *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   struct borgvk_descriptor_pool *pool;

   pool = vk_object_zalloc(&device->vk, pAllocator, sizeof(*pool),
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pDescriptorPool = borgvk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool _pool,
                             const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VK_FROM_HANDLE(borgvk_descriptor_pool, pool, _pool);

   if (!pool)
      return;

   vk_object_free(&device->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_ResetDescriptorPool(VkDevice _device, VkDescriptorPool _pool,
                           VkDescriptorPoolResetFlags flags)
{
   return VK_SUCCESS;
}

/* ---- Descriptor sets -------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_AllocateDescriptorSets(VkDevice _device,
                              const VkDescriptorSetAllocateInfo *pAllocateInfo,
                              VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);
   VkResult result = VK_SUCCESS;
   uint32_t i;

   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      struct borgvk_descriptor_set *set =
         vk_object_zalloc(&device->vk, NULL, sizeof(*set),
                          VK_OBJECT_TYPE_DESCRIPTOR_SET);
      if (!set) {
         result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
         break;
      }
      pDescriptorSets[i] = borgvk_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      for (uint32_t j = 0; j < i; j++) {
         VK_FROM_HANDLE(borgvk_descriptor_set, set, pDescriptorSets[j]);
         vk_object_free(&device->vk, NULL, set);
         pDescriptorSets[j] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_FreeDescriptorSets(VkDevice _device, VkDescriptorPool pool,
                          uint32_t count, const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);

   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(borgvk_descriptor_set, set, pDescriptorSets[i]);
      if (set)
         vk_object_free(&device->vk, NULL, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_UpdateDescriptorSets(VkDevice _device,
                            uint32_t descriptorWriteCount,
                            const VkWriteDescriptorSet *pDescriptorWrites,
                            uint32_t descriptorCopyCount,
                            const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *w = &pDescriptorWrites[i];
      VK_FROM_HANDLE(borgvk_descriptor_set, set, w->dstSet);

      if (w->dstBinding >= BORGVK_MAX_BINDINGS)
         continue;

      /* Record buffer bindings (the cube's MVP UBO lives at binding 0). */
      if (w->pBufferInfo &&
          (w->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
           w->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
           w->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
           w->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)) {
         VK_FROM_HANDLE(borgvk_buffer, buffer, w->pBufferInfo[0].buffer);
         set->buffers[w->dstBinding] = buffer;
         set->offsets[w->dstBinding] = w->pBufferInfo[0].offset;
      }

      /* Record image bindings (the cube's texture is a combined image sampler
       * at binding 1); the submit path reads its mapped RGBA8 to upload it. */
      if (w->pImageInfo &&
          (w->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
           w->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) &&
          w->pImageInfo[0].imageView != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(vk_image_view, view, w->pImageInfo[0].imageView);
         if (view)
            set->images[w->dstBinding] =
               container_of(view->image, struct borgvk_image, vk);
      }
   }

   /* Copies are unused by the cube; ignore. */
}
