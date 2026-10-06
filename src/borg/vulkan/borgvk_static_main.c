/* SPDX-License-Identifier: MIT */
/* Link probe for a static borgvk (no ICD loader): resolves the entrypoints the way an app would. */
#include <stdio.h>
#include <vulkan/vulkan.h>

PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance, const char *);

int
main(void)
{
   PFN_vkCreateInstance create = (PFN_vkCreateInstance)vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance");
   VkInstanceCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
   VkInstance inst;
   VkResult r = create ? create(&ci, NULL, &inst) : VK_ERROR_INITIALIZATION_FAILED;
   printf("vkCreateInstance: %d\n", r);
   return r != VK_SUCCESS;
}
