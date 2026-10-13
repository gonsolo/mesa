/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_util.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"

#include "util/log.h"
#include "drm-uapi/borg_drm.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include "borgvk_drm_compat.h"

#define BORGVK_API_VERSION VK_API_VERSION_1_0

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   *pApiVersion = BORGVK_API_VERSION;
   return VK_SUCCESS;
}

static const struct vk_instance_extension_table instance_extensions = {
   .KHR_external_fence_capabilities     = true,
   .KHR_external_memory_capabilities    = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .EXT_debug_report                    = true,
   .EXT_debug_utils                     = true,
#ifdef BORGVK_USE_WSI_PLATFORM
   .KHR_surface                         = true,
   .KHR_get_surface_capabilities2       = true,
   .EXT_headless_surface                = true,
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
   .KHR_xcb_surface                     = true,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
   .KHR_xlib_surface                    = true,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
   .KHR_wayland_surface                 = true,
#endif
};

/* Device extensions advertised by borgvk: the swapchain so apps can present. */
static const struct vk_device_extension_table device_extensions = {
#ifdef BORGVK_USE_WSI_PLATFORM
   .KHR_swapchain                       = true,
#endif
};

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                            uint32_t *pPropertyCount,
                                            VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(
      &instance_extensions, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                        VkLayerProperties *pProperties)
{
   if (pProperties == NULL) {
      *pPropertyCount = 0;
      return VK_SUCCESS;
   }
   return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);
}

/* One queue family: graphics + compute + transfer, a single queue. */
static const VkQueueFamilyProperties borgvk_queue_family_properties = {
   .queueFlags = VK_QUEUE_GRAPHICS_BIT |
                 VK_QUEUE_COMPUTE_BIT |
                 VK_QUEUE_TRANSFER_BIT,
   .queueCount = 1,
   .timestampValidBits = 0,
   .minImageTransferGranularity = { 1, 1, 1 },
};

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice,
   uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out,
                          pQueueFamilyProperties, pQueueFamilyPropertyCount);

   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = borgvk_queue_family_properties;
      vk_foreach_struct(s_stype, s, p->pNext) {
         vk_debug_ignored_stype(s_stype);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(borgvk_physical_device, device, physicalDevice);

   pMemoryProperties->memoryProperties = device->memory;

   vk_foreach_struct(ext_stype, ext, pMemoryProperties->pNext) {
      vk_debug_ignored_stype(ext_stype);
   }
}

static bool
is_depth_stencil_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_D16_UNORM:
   case VK_FORMAT_X8_D24_UNORM_PACK32:
   case VK_FORMAT_D32_SFLOAT:
   case VK_FORMAT_S8_UINT:
   case VK_FORMAT_D16_UNORM_S8_UINT:
   case VK_FORMAT_D24_UNORM_S8_UINT:
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
   default:
      return false;
   }
}

static bool
is_ycbcr_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_G8B8G8R8_422_UNORM:
   case VK_FORMAT_B8G8R8G8_422_UNORM:
   case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
   case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
   case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM:
   case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:
   case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM:
   case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM:
   case VK_FORMAT_R10X6_UNORM_PACK16:
   case VK_FORMAT_R10X6G10X6_UNORM_2PACK16:
   case VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16:
   case VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16:
   case VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16:
   case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16:
   case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
   case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16:
   case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16:
   case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16:
   case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16:
   case VK_FORMAT_R12X4_UNORM_PACK16:
   case VK_FORMAT_R12X4G12X4_UNORM_2PACK16:
   case VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16:
   case VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16:
   case VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16:
   case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16:
   case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16:
   case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16:
   case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16:
   case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16:
   case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16:
   case VK_FORMAT_G16B16G16R16_422_UNORM:
   case VK_FORMAT_B16G16R16G16_422_UNORM:
   case VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM:
   case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
   case VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM:
   case VK_FORMAT_G16_B16R16_2PLANE_422_UNORM:
   case VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM:
   case VK_FORMAT_G16_B16R16_2PLANE_444_UNORM:
      return true;
   default:
      return false;
   }
}

static bool
is_compressed_format(VkFormat format)
{
   return (format >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK &&
           format <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK) ||
          (format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK &&
           format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
          (format >= 1000288000 && format <= 1000288029) ||
          util_format_get_blockdepth(vk_format_to_pipe_format(format)) > 1;
}

static bool
is_bc_format(VkFormat format)
{
   return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK &&
          format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

#define BORGVK_SC_MSAA      (VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT)
#define BORGVK_SC_INTEGER   VK_SAMPLE_COUNT_1_BIT
#define BORGVK_SC_STORAGE   VK_SAMPLE_COUNT_1_BIT

/* Compute optimal-tiling format features for a given VkFormat. */
static VkFormatFeatureFlags
borgvk_optimal_features(VkFormat format)
{
   if (format == VK_FORMAT_UNDEFINED || is_ycbcr_format(format) ||
       is_compressed_format(format))
      return 0;

   if (vk_format_is_scaled(format))
      return 0;

   if (format == VK_FORMAT_A2R10G10B10_SNORM_PACK32 || format == VK_FORMAT_A2B10G10R10_SNORM_PACK32 ||
       format == VK_FORMAT_A2R10G10B10_SINT_PACK32 || format == VK_FORMAT_A2B10G10R10_SINT_PACK32)
      return 0;

   if (format >= VK_FORMAT_R64_UINT && format <= VK_FORMAT_R64G64B64A64_SFLOAT)
      return 0;

   if (is_bc_format(format)) {
      return VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
             VK_FORMAT_FEATURE_BLIT_SRC_BIT |
             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
             VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
             VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   }

   const VkFormatFeatureFlags common =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_BLIT_SRC_BIT |
      VK_FORMAT_FEATURE_BLIT_DST_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

   if (is_depth_stencil_format(format)) {
      return common | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
   }

   VkFormatFeatureFlags f =
      common |
      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
   switch (format) {
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      f |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
      break;
   default:
      break;
   }
   if (!vk_format_is_int(format) && !vk_format_is_scaled(format))
      f |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   if (format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT)
      f |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
   return f;
}

static VkFormatFeatureFlags
borgvk_texel_buffer_features(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
      return VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
             VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT |
             ((format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT) ?
                 VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT : 0);
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_UNORM:
      return VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT;
   default:
      return 0;
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceFormatProperties2(
   VkPhysicalDevice physicalDevice,
   VkFormat format,
   VkFormatProperties2 *pFormatProperties)
{
   const VkFormatFeatureFlags opt = borgvk_optimal_features(format);
   VkFormatFeatureFlags lin =
      (opt != 0 && !is_bc_format(format)) ? (VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                    VK_FORMAT_FEATURE_TRANSFER_DST_BIT) : 0;
   if ((opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
       !is_depth_stencil_format(format) && !is_bc_format(format)) {
      lin |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
      if (opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
         lin |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   }
   VkFormatFeatureFlags buf = 0;
   if ((opt != 0 || vk_format_is_scaled(format)) &&
       !is_depth_stencil_format(format) && !is_bc_format(format)) {
      buf = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT | borgvk_texel_buffer_features(format);
   }

   pFormatProperties->formatProperties = (VkFormatProperties){
      .linearTilingFeatures  = lin,
      .optimalTilingFeatures = opt,
      .bufferFeatures        = buf,
   };

   vk_foreach_struct(ext_stype, ext, pFormatProperties->pNext) {
      switch (ext_stype) {
      case VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3: {
         VkFormatProperties3 *fp3 = (VkFormatProperties3 *)ext;
         fp3->linearTilingFeatures  = lin;
         fp3->optimalTilingFeatures = opt;
         fp3->bufferFeatures        = buf;

         if (vk_format_has_depth(format) &&
             (opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
            fp3->optimalTilingFeatures |=
               VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
         break;
      }
      default:
         vk_debug_ignored_stype(ext_stype);
         break;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_GetPhysicalDeviceImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
   VkImageFormatProperties2 *pImageFormatProperties)
{
   /* Derive supported usages from the format's reported features. */
   VkFormatProperties2 fp2 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
   borgvk_GetPhysicalDeviceFormatProperties2(physicalDevice,
                                             pImageFormatInfo->format, &fp2);
   const VkFormatFeatureFlags feats =
      (pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR)
      ? fp2.formatProperties.linearTilingFeatures
      : fp2.formatProperties.optimalTilingFeatures;

   /* Map requested usage flags to required format feature bits. */
   const VkImageUsageFlags usage = pImageFormatInfo->usage;
   VkFormatFeatureFlags required = 0;
   if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
      required |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
   if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
      required |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
   if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
      required |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
   if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
      required |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
   if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
      required |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
   if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
      required |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
      required |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

   pImageFormatProperties->imageFormatProperties =
      (VkImageFormatProperties){ 0 };

   if ((feats & required) != required)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (pImageFormatInfo->type == VK_IMAGE_TYPE_3D &&
       util_format_get_blockdepth(vk_format_to_pipe_format(pImageFormatInfo->format)) > 1)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) &&
       !(feats & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                  VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const bool cube_compatible =
      (pImageFormatInfo->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0;
   const bool has_attach_feature =
      (feats & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0;
   VkSampleCountFlags samples = VK_SAMPLE_COUNT_1_BIT;
   if (pImageFormatInfo->type == VK_IMAGE_TYPE_2D &&
       pImageFormatInfo->tiling == VK_IMAGE_TILING_OPTIMAL &&
       !cube_compatible && has_attach_feature) {
      const VkFormat fmt = pImageFormatInfo->format;
      const bool is_ds = is_depth_stencil_format(fmt);
      VkSampleCountFlags sc = ~(VkSampleCountFlags)0;
      if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
         sc &= BORGVK_SC_STORAGE;
      if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
         sc &= is_ds ? BORGVK_SC_MSAA
                     : (vk_format_is_int(fmt) ? BORGVK_SC_INTEGER : BORGVK_SC_MSAA);
      if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
         sc &= BORGVK_SC_MSAA;
      if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
         sc &= BORGVK_SC_MSAA;
      /* No usage bit narrows it: only single-sampled is required. */
      samples = (sc == ~(VkSampleCountFlags)0) ? VK_SAMPLE_COUNT_1_BIT
                                               : (sc | VK_SAMPLE_COUNT_1_BIT);
   }

   const uint32_t max2d = 4096;   /* BorgSampler: up to 4096 */
   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
      .maxExtent = {
         .width  = max2d,
         .height = pImageFormatInfo->type == VK_IMAGE_TYPE_1D ? 1 : max2d,
         .depth  = pImageFormatInfo->type == VK_IMAGE_TYPE_3D ? 2048 : 1,
      },
      .maxMipLevels   = 13,
      /* Spec §12.5: 3D images must have maxArrayLayers = 1. */
      .maxArrayLayers = pImageFormatInfo->type == VK_IMAGE_TYPE_3D ? 1 : 2048,
      .sampleCounts   = samples,
      .maxResourceSize = (VkDeviceSize)1 << 31,
   };

   vk_foreach_struct(ext_stype, ext, pImageFormatProperties->pNext) {
      vk_debug_ignored_stype(ext_stype);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceSparseImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
   uint32_t *pPropertyCount,
   VkSparseImageFormatProperties2 *pProperties)
{
   *pPropertyCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_GetPhysicalDeviceExternalBufferProperties(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
   VkExternalBufferProperties *pExternalBufferProperties)
{
   pExternalBufferProperties->externalMemoryProperties = (VkExternalMemoryProperties){
      .externalMemoryFeatures = 0,
      .exportFromImportedHandleTypes = 0,
      .compatibleHandleTypes = 0,
   };
}

static void
borgvk_get_properties(struct vk_properties *p)
{
   *p = (struct vk_properties){
      .apiVersion = BORGVK_API_VERSION,
      .driverVersion = VK_MAKE_VERSION(0, 1, 0),
      .vendorID = 0x1de0,  /* not a registered ID; placeholder for "Borg" */
      .deviceID = 0xb049,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER,

      .maxImageDimension1D = 4096,
      .maxImageDimension2D = 4096,
      .maxImageDimension3D = 256,
      .maxImageDimensionCube = 4096,
      .maxImageArrayLayers = 256,
      .maxTexelBufferElements = 65536,
      .maxUniformBufferRange = 65536,
      .maxStorageBufferRange = UINT32_MAX,
      .maxPushConstantsSize = 128,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 1,
      .maxBoundDescriptorSets = 8,
      .maxPerStageResources = 128,
      .maxVertexInputAttributes = 16,
      .maxVertexInputBindings = 16,
      .maxVertexInputAttributeOffset = 2047,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 64,
      .maxFragmentInputComponents = 64,
      .maxFragmentOutputAttachments = 4,
      .maxFragmentCombinedOutputResources = 4,
      .maxColorAttachments = 4,
      .maxFramebufferWidth = 4096,
      .maxFramebufferHeight = 4096,
      .maxFramebufferLayers = 256,
      .maxViewports = 1,
      .maxViewportDimensions = { 4096, 4096 },
      .viewportBoundsRange = { -8192.0f, 8191.0f },

      /* Per-stage and per-set descriptor limits (Vulkan spec table minimums). */
      .maxPerStageDescriptorSamplers          = 16,
      .maxPerStageDescriptorUniformBuffers    = 12,
      .maxPerStageDescriptorStorageBuffers    = 4,
      .maxPerStageDescriptorSampledImages     = 16,
      .maxPerStageDescriptorStorageImages     = 4,
      .maxPerStageDescriptorInputAttachments  = 4,
      .maxDescriptorSetSamplers               = 96,
      .maxDescriptorSetUniformBuffers         = 72,
      .maxDescriptorSetUniformBuffersDynamic  = 8,
      .maxDescriptorSetStorageBuffers         = 24,
      .maxDescriptorSetStorageBuffersDynamic  = 4,
      .maxDescriptorSetSampledImages          = 96,
      .maxDescriptorSetStorageImages          = 24,
      .maxDescriptorSetInputAttachments       = 4,

      /* Compute limits (spec minimums). */
      .maxComputeSharedMemorySize    = 16384,
      .maxComputeWorkGroupCount      = {65535, 65535, 65535},
      .maxComputeWorkGroupInvocations = 128,
      .maxComputeWorkGroupSize       = {128, 128, 64},

      /* Texel gather / LOD / offset. */
      .maxSamplerLodBias   = 15.0f,   /* BorgSampler.MaxLodBias */
      .minTexelOffset      = -8,
      .maxTexelOffset      = 7,

      /* Alignment requirements (must be >= 1 per spec). */
      .minMemoryMapAlignment           = 64,
      .minTexelBufferOffsetAlignment   = 64,
      .minUniformBufferOffsetAlignment = 256,
      .minStorageBufferOffsetAlignment = 256,
      .nonCoherentAtomSize             = 64,

      /* VK_EXT_texel_buffer_alignment (Vulkan 1.3): must be a power of two. */
      .storageTexelBufferOffsetAlignmentBytes  = 256,
      .uniformTexelBufferOffsetAlignmentBytes  = 256,
      .storageTexelBufferOffsetSingleTexelAlignment  = VK_TRUE,
      .uniformTexelBufferOffsetSingleTexelAlignment  = VK_TRUE,

      /* Spec requires at least 1x and 4x sample support (bitmask >= 5). */
      .framebufferColorSampleCounts = BORGVK_SC_MSAA,
      .framebufferDepthSampleCounts = BORGVK_SC_MSAA,
      .framebufferStencilSampleCounts = BORGVK_SC_MSAA,
      .framebufferNoAttachmentsSampleCounts = BORGVK_SC_MSAA,
      .sampledImageColorSampleCounts = BORGVK_SC_MSAA,
      .sampledImageIntegerSampleCounts = BORGVK_SC_INTEGER,
      .sampledImageDepthSampleCounts = BORGVK_SC_MSAA,
      .sampledImageStencilSampleCounts = BORGVK_SC_MSAA,
      .storageImageSampleCounts = BORGVK_SC_STORAGE,
      .maxSampleMaskWords       = 1,
      .discreteQueuePriorities  = 2,

      .maxSamplerAnisotropy = 1.0f,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT32_MAX,
      .pointSizeRange = { 1.0f, 64.0f },
      .lineWidthRange = { 1.0f, 1.0f },
      .pointSizeGranularity = 0.125f,
      .lineWidthGranularity = 0.0f,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .nonCoherentAtomSize = 1,
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits   = 8,
      .subPixelInterpolationOffsetBits = 8,

      /* Vulkan 1.1 properties. */
      .maxMultiviewViewCount    = 6,
      .maxMultiviewInstanceIndex = (1u << 27) - 1,
      .maxPerSetDescriptors     = 1024,
      .maxMemoryAllocationSize  = (VkDeviceSize)1 << 30,  /* 1 GB */

      /* Vulkan 1.2 properties. */
      .framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxTimelineSemaphoreValueDifference = (uint64_t)INT64_MAX,

      /* Vulkan 1.3 properties. */
      .maxInlineUniformTotalSize = 256,

      /* Vulkan 1.2 driver identification. */
      .driverID = VK_DRIVER_ID_MESA_DOZEN,  /* placeholder until a Borg ID exists */
      .conformanceVersion = { 1, 3, 0, 0 },
   };

   snprintf(p->deviceName, sizeof(p->deviceName), "Borg (ULX3S)");
   snprintf(p->driverName, sizeof(p->driverName), "borgvk");
   snprintf(p->driverInfo, sizeof(p->driverInfo), "Mesa borgvk (serial/FPGA)");
}

static VkResult
create_physical_device(struct borgvk_instance *instance, int drm_fd)
{
   struct borgvk_physical_device *device =
      vk_zalloc(&instance->vk.alloc, sizeof(*device), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!device)
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   device->drm_fd = drm_fd;

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &borgvk_physical_device_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_physical_device_entrypoints, false);
#endif

   struct vk_features features = {
      .robustBufferAccess = true,
      /* Vulkan 1.3 spec §43 mandates at least one compressed texture family. */
      .textureCompressionBC = true,
      .largePoints = true,
   };
   struct vk_properties properties;
   borgvk_get_properties(&properties);

   VkResult result =
      vk_physical_device_init(&device->vk, &instance->vk, &device_extensions,
                              &features, &properties, &dispatch_table);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, device);
      return result;
   }

   /* Single host-visible coherent heap (host RAM staged to the FPGA). */
   device->memory.memoryHeapCount = 1;
   device->memory.memoryHeaps[0].size = 256ull * 1024 * 1024;
   device->memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
   device->memory.memoryTypeCount = 1;
   device->memory.memoryTypes[0].propertyFlags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   device->memory.memoryTypes[0].heapIndex = 0;

   device->sync_types[0] = &borgvk_sync_type;
   device->sync_types[1] = NULL;
   device->vk.supported_sync_types = device->sync_types;

#ifdef BORGVK_USE_WSI_PLATFORM
   result = borgvk_wsi_init(device);
   if (result != VK_SUCCESS) {
      vk_physical_device_finish(&device->vk);
      vk_free(&instance->vk.alloc, device);
      return result;
   }
#endif

   list_addtail(&device->vk.link, &instance->vk.physical_devices.list);

   return VK_SUCCESS;
}

static int
find_borg_drm_device(void)
{
   for (int i = 128; i < 192; i++) {
      char path[64];
      snprintf(path, sizeof(path), "/dev/dri/renderD%d", i);
      int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0)
         continue;
      drmVersionPtr ver = drmGetVersion(fd);
      if (ver) {
         int match = ver->name && strcmp(ver->name, "borg") == 0;
         drmFreeVersion(ver);
         if (match)
            return fd;
      }
      close(fd);
   }
   return -1;
}

static VkResult
enumerate_devices(struct vk_instance *vk_instance)
{
   struct borgvk_instance *instance =
      container_of(vk_instance, struct borgvk_instance, vk);

   int drm_fd = find_borg_drm_device();
   if (drm_fd >= 0)
      mesa_logi("borgvk: found borg DRM device (fd %d)", drm_fd);
   else
      mesa_logi("borgvk: no borg DRM device; using malloc+serial fallback");

   return create_physical_device(instance, drm_fd);
}

static void
destroy_physical_device(struct vk_physical_device *pdev)
{
   struct borgvk_physical_device *pdevice =
      container_of(pdev, struct borgvk_physical_device, vk);
#ifdef BORGVK_USE_WSI_PLATFORM
   borgvk_wsi_finish(pdevice);
#endif
   if (pdevice->drm_fd >= 0)
      close(pdevice->drm_fd);
   vk_physical_device_finish(pdev);
   vk_free(&pdev->instance->alloc, pdev);
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkInstance *pInstance)
{
   struct borgvk_instance *instance;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);

   /* A Vulkan 1.0 implementation rejects a requested apiVersion it does not support. */
   if (pCreateInfo->pApplicationInfo && pCreateInfo->pApplicationInfo->apiVersion != 0) {
      const uint32_t req = pCreateInfo->pApplicationInfo->apiVersion;
      if (VK_API_VERSION_MAJOR(req) != VK_API_VERSION_MAJOR(BORGVK_API_VERSION) ||
          VK_API_VERSION_MINOR(req) > VK_API_VERSION_MINOR(BORGVK_API_VERSION))
         return vk_error(NULL, VK_ERROR_INCOMPATIBLE_DRIVER);
   }

   if (pAllocator == NULL)
      pAllocator = vk_default_allocator();

   instance = vk_alloc(pAllocator, sizeof(*instance), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &borgvk_instance_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_instance_entrypoints, false);
#endif

   result = vk_instance_init(&instance->vk, &instance_extensions,
                             &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return vk_error(NULL, result);
   }

   instance->vk.physical_devices.enumerate = enumerate_devices;
   instance->vk.physical_devices.destroy = destroy_physical_device;

   *pInstance = borgvk_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyInstance(VkInstance _instance,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_instance, instance, _instance);

   if (!instance)
      return;

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

/* Body for vkCmd* entries borgvk does not implement; see the cmd_dispatch fill-in. */
static void
borgvk_noop_cmd(void)
{
}

VKAPI_ATTR VkResult VKAPI_CALL
borgvk_CreateDevice(VkPhysicalDevice physicalDevice,
                    const VkDeviceCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkDevice *pDevice)
{
   VK_FROM_HANDLE(borgvk_physical_device, physical_device, physicalDevice);
   struct borgvk_device *device;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);

   vk_foreach_struct_const(f_stype, f, pCreateInfo->pNext) {
      switch (f_stype) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES:
         if (((const VkPhysicalDeviceProtectedMemoryFeatures *)f)->protectedMemory)
            return vk_error(physical_device, VK_ERROR_FEATURE_NOT_PRESENT);
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES:
         if (((const VkPhysicalDeviceShaderDrawParametersFeatures *)f)->shaderDrawParameters)
            return vk_error(physical_device, VK_ERROR_FEATURE_NOT_PRESENT);
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
         const VkPhysicalDeviceVulkan11Features *v11 = (const void *)f;
         if (v11->protectedMemory || v11->shaderDrawParameters)
            return vk_error(physical_device, VK_ERROR_FEATURE_NOT_PRESENT);
         break;
      }
      default:
         break;
      }
   }

   device = vk_zalloc2(&physical_device->vk.instance->alloc, pAllocator,
                       sizeof(*device), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &borgvk_device_entrypoints, true);
#ifdef BORGVK_USE_WSI_PLATFORM
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);
#endif
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vk_cmd_enqueue_device_entrypoints, false);
#define BORGVK_RECORD_CMD(name) dispatch_table.name = vk_cmd_enqueue_##name
   BORGVK_RECORD_CMD(CmdPushConstants);
   BORGVK_RECORD_CMD(CmdBeginRendering);   BORGVK_RECORD_CMD(CmdEndRendering);
   BORGVK_RECORD_CMD(CmdClearAttachments);
   BORGVK_RECORD_CMD(CmdClearColorImage);  BORGVK_RECORD_CMD(CmdClearDepthStencilImage);
   BORGVK_RECORD_CMD(CmdBlitImage);        BORGVK_RECORD_CMD(CmdBlitImage2);
   BORGVK_RECORD_CMD(CmdCopyBuffer);       BORGVK_RECORD_CMD(CmdCopyBuffer2);
   BORGVK_RECORD_CMD(CmdCopyBufferToImage); BORGVK_RECORD_CMD(CmdCopyBufferToImage2);
   BORGVK_RECORD_CMD(CmdCopyImage);        BORGVK_RECORD_CMD(CmdCopyImage2);
   BORGVK_RECORD_CMD(CmdCopyImageToBuffer); BORGVK_RECORD_CMD(CmdCopyImageToBuffer2);
   BORGVK_RECORD_CMD(CmdResolveImage);     BORGVK_RECORD_CMD(CmdResolveImage2);
   BORGVK_RECORD_CMD(CmdFillBuffer);       BORGVK_RECORD_CMD(CmdUpdateBuffer);
   BORGVK_RECORD_CMD(CmdSetEvent);         BORGVK_RECORD_CMD(CmdResetEvent);
   BORGVK_RECORD_CMD(CmdSetEvent2);        BORGVK_RECORD_CMD(CmdResetEvent2);
   BORGVK_RECORD_CMD(CmdBindPipeline);     BORGVK_RECORD_CMD(CmdBindVertexBuffers);
   BORGVK_RECORD_CMD(CmdBindDescriptorSets); BORGVK_RECORD_CMD(CmdDraw);
   BORGVK_RECORD_CMD(CmdDrawIndexed);      BORGVK_RECORD_CMD(CmdBindIndexBuffer);
   BORGVK_RECORD_CMD(CmdDispatch);
   BORGVK_RECORD_CMD(CmdDrawIndirect);     BORGVK_RECORD_CMD(CmdDrawIndexedIndirect);
#undef BORGVK_RECORD_CMD
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vk_common_device_entrypoints, false);

   /* Dispatch used to replay a recorded queue (secondary cmd buffers). */
   vk_device_dispatch_table_from_entrypoints(&device->cmd_dispatch,
                                             &borgvk_device_entrypoints, true);
   if (!device->cmd_dispatch.CmdExecuteCommands)
      device->cmd_dispatch.CmdExecuteCommands = vk_common_CmdExecuteCommands;
   {
      void **slots = (void **)&device->cmd_dispatch;
      for (size_t i = 0; i < sizeof(device->cmd_dispatch) / sizeof(void *); i++)
         if (!slots[i])
            slots[i] = (void *)borgvk_noop_cmd;
   }

   result = vk_device_init(&device->vk, &physical_device->vk,
                           &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&physical_device->vk.instance->alloc, pAllocator, device);
      return vk_error(physical_device, result);
   }

   vk_device_enable_threaded_submit(&device->vk);

   device->vk.command_dispatch_table = &device->cmd_dispatch;
   device->vk.command_buffer_ops = &borgvk_cmd_buffer_ops;

   /* One queue from family 0. */
   assert(pCreateInfo->queueCreateInfoCount >= 1);
   result = vk_queue_init(&device->queue, &device->vk,
                          &pCreateInfo->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS) {
      vk_device_finish(&device->vk);
      vk_free2(&physical_device->vk.instance->alloc, pAllocator, device);
      return vk_error(physical_device, result);
   }

   /* Non-owning copy — lifetime is managed by the physical device. */
   device->drm_fd = physical_device->drm_fd;

   device->queue.driver_submit = borgvk_queue_submit;

   *pDevice = borgvk_device_to_handle(device);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(borgvk_device, device, _device);

   if (!device)
      return;

   vk_queue_finish(&device->queue);
   vk_device_finish(&device->vk);
   vk_free2(&device->vk.alloc, pAllocator, device);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
borgvk_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(borgvk_instance, instance, _instance);
   return vk_instance_get_proc_addr(instance ? &instance->vk : NULL,
                                    &borgvk_instance_entrypoints, pName);
}

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return borgvk_GetInstanceProcAddr(instance, pName);
}
