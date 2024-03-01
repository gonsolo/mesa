/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_entrypoints.h"

VKAPI_ATTR VkResult VKAPI_CALL
borg_CreateImage(VkDevice device,
                const VkImageCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator,
                VkImage *pImage)
{
   // TODO
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borg_DestroyImage(VkDevice device,
                 VkImage _image,
                 const VkAllocationCallbacks *pAllocator)
{
   // TODO
}

VKAPI_ATTR void VKAPI_CALL
borg_GetImageMemoryRequirements2(VkDevice device,
                                const VkImageMemoryRequirementsInfo2 *pInfo,
                                VkMemoryRequirements2 *pMemoryRequirements)
{
   // TODO
}

VKAPI_ATTR VkResult VKAPI_CALL
  borg_GetPhysicalDeviceImageFormatProperties2(
     VkPhysicalDevice physicalDevice,
     const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
     VkImageFormatProperties2 *pImageFormatProperties)
{
   // TODO
   return VK_SUCCESS;
}
