// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The physical device: its properties, queue family, memory and format queries, and the
//! search for a Borg DRM device.

use crate::entrypoint::{chain, entrypoint, vk_error, Handle};
use crate::ffi::*;
use crate::image::blocks;
use crate::sync::SYNC_TYPE;
use crate::util::log;
use crate::wsi::{borgvk_wsi_finish, borgvk_wsi_init};
use std::ffi::CStr;
use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, IntoRawFd};

pub const API_VERSION: u32 = 1 << 22; // VK_API_VERSION_1_0

pub const SC_MSAA: VkSampleCountFlags = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
pub const SC_INTEGER: VkSampleCountFlags = VK_SAMPLE_COUNT_1_BIT;
pub const SC_STORAGE: VkSampleCountFlags = VK_SAMPLE_COUNT_1_BIT;

fn is_depth_stencil(f: VkFormat) -> bool {
    matches!(
        f,
        VK_FORMAT_D16_UNORM
            | VK_FORMAT_X8_D24_UNORM_PACK32
            | VK_FORMAT_D32_SFLOAT
            | VK_FORMAT_S8_UINT
            | VK_FORMAT_D16_UNORM_S8_UINT
            | VK_FORMAT_D24_UNORM_S8_UINT
            | VK_FORMAT_D32_SFLOAT_S8_UINT
    )
}

fn is_ycbcr(f: VkFormat) -> bool {
    (VK_FORMAT_G8B8G8R8_422_UNORM..=VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM).contains(&f)
        || (VK_FORMAT_G8_B8R8_2PLANE_444_UNORM..=VK_FORMAT_G16_B16R16_2PLANE_444_UNORM).contains(&f)
}

fn is_compressed(f: VkFormat) -> bool {
    (VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK..=VK_FORMAT_EAC_R11G11_SNORM_BLOCK).contains(&f)
        || (VK_FORMAT_ASTC_4x4_UNORM_BLOCK..=VK_FORMAT_ASTC_12x12_SRGB_BLOCK).contains(&f)
        || (1000288000..=1000288029).contains(&f)
        || blocks(f).bd > 1
}

fn is_bc(f: VkFormat) -> bool {
    (VK_FORMAT_BC1_RGB_UNORM_BLOCK..=VK_FORMAT_BC7_SRGB_BLOCK).contains(&f)
}

fn is_scaled(f: VkFormat) -> bool {
    unsafe { util_format_is_scaled(vk_format_to_pipe_format(f)) }
}

/// Optimal-tiling format features.
fn optimal_features(f: VkFormat) -> VkFormatFeatureFlags {
    if f == VK_FORMAT_UNDEFINED || is_ycbcr(f) || is_compressed(f) || is_scaled(f) {
        return 0;
    }
    if matches!(
        f,
        VK_FORMAT_A2R10G10B10_SNORM_PACK32 | VK_FORMAT_A2B10G10R10_SNORM_PACK32 | VK_FORMAT_A2R10G10B10_SINT_PACK32 | VK_FORMAT_A2B10G10R10_SINT_PACK32
    ) || (VK_FORMAT_R64_UINT..=VK_FORMAT_R64G64B64A64_SFLOAT).contains(&f)
    {
        return 0;
    }
    if is_bc(f) {
        return VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
            | VK_FORMAT_FEATURE_BLIT_SRC_BIT
            | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
            | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
            | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }
    let common = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
        | VK_FORMAT_FEATURE_BLIT_SRC_BIT
        | VK_FORMAT_FEATURE_BLIT_DST_BIT
        | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if is_depth_stencil(f) {
        return common | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    let mut r = common | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    if matches!(
        f,
        VK_FORMAT_R5G6B5_UNORM_PACK16
            | VK_FORMAT_A1R5G5B5_UNORM_PACK16
            | VK_FORMAT_R8_UNORM
            | VK_FORMAT_R8G8_UNORM
            | VK_FORMAT_R8G8B8A8_UNORM
            | VK_FORMAT_R8G8B8A8_SRGB
            | VK_FORMAT_B8G8R8A8_UNORM
            | VK_FORMAT_B8G8R8A8_SRGB
            | VK_FORMAT_A8B8G8R8_UNORM_PACK32
            | VK_FORMAT_A8B8G8R8_SRGB_PACK32
            | VK_FORMAT_A2B10G10R10_UNORM_PACK32
            | VK_FORMAT_R16_SFLOAT
            | VK_FORMAT_R16G16_SFLOAT
            | VK_FORMAT_R16G16B16A16_SFLOAT
    ) {
        r |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
    }
    if unsafe { !vk_format_is_int(f) } {
        r |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }
    if matches!(f, VK_FORMAT_R32_UINT | VK_FORMAT_R32_SINT) {
        r |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
    }
    r
}

fn texel_buffer_features(f: VkFormat) -> VkFormatFeatureFlags {
    match f {
        VK_FORMAT_A8B8G8R8_SINT_PACK32
        | VK_FORMAT_A8B8G8R8_SNORM_PACK32
        | VK_FORMAT_A8B8G8R8_UINT_PACK32
        | VK_FORMAT_A8B8G8R8_UNORM_PACK32
        | VK_FORMAT_R16G16B16A16_SFLOAT
        | VK_FORMAT_R16G16B16A16_SINT
        | VK_FORMAT_R16G16B16A16_UINT
        | VK_FORMAT_R32_SFLOAT
        | VK_FORMAT_R32_SINT
        | VK_FORMAT_R32_UINT
        | VK_FORMAT_R32G32_SFLOAT
        | VK_FORMAT_R32G32_SINT
        | VK_FORMAT_R32G32_UINT
        | VK_FORMAT_R32G32B32A32_SFLOAT
        | VK_FORMAT_R32G32B32A32_SINT
        | VK_FORMAT_R32G32B32A32_UINT
        | VK_FORMAT_R8G8B8A8_SINT
        | VK_FORMAT_R8G8B8A8_SNORM
        | VK_FORMAT_R8G8B8A8_UINT
        | VK_FORMAT_R8G8B8A8_UNORM => {
            let atomic = if matches!(f, VK_FORMAT_R32_UINT | VK_FORMAT_R32_SINT) { VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT } else { 0 };
            VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT | VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT | atomic
        }
        VK_FORMAT_A2B10G10R10_UINT_PACK32
        | VK_FORMAT_A2B10G10R10_UNORM_PACK32
        | VK_FORMAT_B10G11R11_UFLOAT_PACK32
        | VK_FORMAT_B8G8R8A8_UNORM
        | VK_FORMAT_R16_SFLOAT
        | VK_FORMAT_R16_SINT
        | VK_FORMAT_R16_UINT
        | VK_FORMAT_R16G16_SFLOAT
        | VK_FORMAT_R16G16_SINT
        | VK_FORMAT_R16G16_UINT
        | VK_FORMAT_R8_SINT
        | VK_FORMAT_R8_SNORM
        | VK_FORMAT_R8_UINT
        | VK_FORMAT_R8_UNORM
        | VK_FORMAT_R8G8_SINT
        | VK_FORMAT_R8G8_SNORM
        | VK_FORMAT_R8G8_UINT
        | VK_FORMAT_R8G8_UNORM => VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT,
        _ => 0,
    }
}

/// Linear-tiling, optimal-tiling and buffer features.
fn format_properties(f: VkFormat) -> VkFormatProperties {
    let opt = optimal_features(f);
    let (ds, bc) = (is_depth_stencil(f), is_bc(f));
    let mut lin = if opt != 0 && !bc { VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT } else { 0 };
    if opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT != 0 && !ds && !bc {
        lin |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }
    let buf = if (opt != 0 || is_scaled(f)) && !ds && !bc { VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT | texel_buffer_features(f) } else { 0 };
    VkFormatProperties { linearTilingFeatures: lin, optimalTilingFeatures: opt, bufferFeatures: buf }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceFormatProperties2(_pd: VkPhysicalDevice, format: VkFormat, props: *mut VkFormatProperties2) {
        let p = format_properties(format);
        (*props).formatProperties = p;
        for s in chain((*props).pNext) {
            if (*s).sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 {
                let p3 = &mut *s.cast::<VkFormatProperties3>();
                p3.linearTilingFeatures = p.linearTilingFeatures as u64;
                p3.optimalTilingFeatures = p.optimalTilingFeatures as u64;
                p3.bufferFeatures = p.bufferFeatures as u64;
                if vk_format_has_depth(format) && p.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT != 0 {
                    p3.optimalTilingFeatures |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
                }
            }
        }
    }
}

fn image_format_properties(info: &VkPhysicalDeviceImageFormatInfo2) -> Result<VkImageFormatProperties, VkResult> {
    let fp = format_properties(info.format);
    let feats = if info.tiling == VK_IMAGE_TILING_LINEAR { fp.linearTilingFeatures } else { fp.optimalTilingFeatures };
    let usage = info.usage;
    let need = [
        (VK_IMAGE_USAGE_SAMPLED_BIT, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT),
        (VK_IMAGE_USAGE_STORAGE_BIT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT),
        (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT),
        (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT),
        (VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT),
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_FORMAT_FEATURE_TRANSFER_SRC_BIT),
        (VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FORMAT_FEATURE_TRANSFER_DST_BIT),
    ];
    let required = need.iter().filter(|(u, _)| usage & u != 0).fold(0, |r, (_, f)| r | f);
    let attach = feats & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    if feats & required != required
        || (info.type_ == VK_IMAGE_TYPE_3D && blocks(info.format).bd > 1)
        || (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT != 0 && !attach)
    {
        return Err(VK_ERROR_FORMAT_NOT_SUPPORTED);
    }
    let cube = info.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT != 0;
    let mut samples = VK_SAMPLE_COUNT_1_BIT;
    if info.type_ == VK_IMAGE_TYPE_2D && info.tiling == VK_IMAGE_TILING_OPTIMAL && !cube && attach {
        let sampled = if is_depth_stencil(info.format) || unsafe { !vk_format_is_int(info.format) } { SC_MSAA } else { SC_INTEGER };
        let narrow = [
            (VK_IMAGE_USAGE_STORAGE_BIT, SC_STORAGE),
            (VK_IMAGE_USAGE_SAMPLED_BIT, sampled),
            (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, SC_MSAA),
            (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, SC_MSAA),
        ];
        let sc = narrow.iter().filter(|(u, _)| usage & u != 0).fold(!0, |s, (_, c)| s & c);
        // No usage bit narrows it: only single-sampled is required.
        samples = if sc == !0 { VK_SAMPLE_COUNT_1_BIT } else { sc | VK_SAMPLE_COUNT_1_BIT };
    }
    let (d1, d3) = (info.type_ == VK_IMAGE_TYPE_1D, info.type_ == VK_IMAGE_TYPE_3D);
    Ok(VkImageFormatProperties {
        maxExtent: VkExtent3D { width: 4096, height: if d1 { 1 } else { 4096 }, depth: if d3 { 2048 } else { 1 } },
        maxMipLevels: 13,
        maxArrayLayers: if d3 { 1 } else { 2048 }, // 3D images have one layer
        sampleCounts: samples,
        maxResourceSize: 1 << 31,
    })
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceImageFormatProperties2(_pd: VkPhysicalDevice, info: *const VkPhysicalDeviceImageFormatInfo2, props: *mut VkImageFormatProperties2) -> VkResult {
        let r = image_format_properties(&*info);
        (*props).imageFormatProperties = *r.as_ref().unwrap_or(&std::mem::zeroed());
        r.map_or_else(|e| e, |_| VK_SUCCESS)
    }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceSparseImageFormatProperties2(_pd: VkPhysicalDevice, _info: *const VkPhysicalDeviceSparseImageFormatInfo2, count: *mut u32, _props: *mut VkSparseImageFormatProperties2) {
        *count = 0;
    }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceExternalBufferProperties(_pd: VkPhysicalDevice, _info: *const VkPhysicalDeviceExternalBufferInfo, props: *mut VkExternalBufferProperties) {
        (*props).externalMemoryProperties = std::mem::zeroed();
    }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceQueueFamilyProperties2(_pd: VkPhysicalDevice, count: *mut u32, props: *mut VkQueueFamilyProperties2) {
        // One family: graphics, compute and transfer, a single queue.
        if props.is_null() {
            *count = 1;
            return;
        }
        if *count == 0 {
            return;
        }
        *count = 1;
        (*props).queueFamilyProperties = VkQueueFamilyProperties {
            queueFlags: VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
            queueCount: 1,
            timestampValidBits: 0,
            minImageTransferGranularity: VkExtent3D { width: 1, height: 1, depth: 1 },
        };
    }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceMemoryProperties2(pd: VkPhysicalDevice, props: *mut VkPhysicalDeviceMemoryProperties2) {
        (*props).memoryProperties = pd.get().unwrap().memory;
    }
}

fn set_name<const N: usize>(dst: &mut [std::ffi::c_char; N], s: &str) {
    for (d, &b) in dst.iter_mut().zip(s.as_bytes().iter().chain(&[0])) {
        *d = b as _;
    }
}

fn properties() -> vk_properties {
    let mut p: vk_properties = unsafe { std::mem::zeroed() };
    p.apiVersion = API_VERSION;
    p.driverVersion = 1 << 12; // 0.1.0
    p.vendorID = 0x1de0; // not a registered ID
    p.deviceID = 0xb049;
    p.deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;

    (p.maxImageDimension1D, p.maxImageDimension2D, p.maxImageDimension3D, p.maxImageDimensionCube) = (4096, 4096, 256, 4096);
    (p.maxImageArrayLayers, p.maxTexelBufferElements, p.maxUniformBufferRange, p.maxStorageBufferRange) = (256, 65536, 65536, u32::MAX);
    (p.maxPushConstantsSize, p.maxMemoryAllocationCount, p.maxSamplerAllocationCount) = (128, 4096, 4000);
    (p.bufferImageGranularity, p.maxBoundDescriptorSets, p.maxPerStageResources) = (1, 8, 128);
    (p.maxVertexInputAttributes, p.maxVertexInputBindings, p.maxVertexInputAttributeOffset, p.maxVertexInputBindingStride) = (16, 16, 2047, 2048);
    (p.maxVertexOutputComponents, p.maxFragmentInputComponents) = (64, 64);
    (p.maxFragmentOutputAttachments, p.maxFragmentCombinedOutputResources, p.maxColorAttachments) = (4, 4, 4);
    (p.maxFramebufferWidth, p.maxFramebufferHeight, p.maxFramebufferLayers) = (4096, 4096, 256);
    (p.maxViewports, p.maxViewportDimensions, p.viewportBoundsRange) = (1, [4096, 4096], [-8192.0, 8191.0]);

    // Descriptor limits: the spec's minimums.
    (p.maxPerStageDescriptorSamplers, p.maxPerStageDescriptorUniformBuffers, p.maxPerStageDescriptorStorageBuffers) = (16, 12, 4);
    (p.maxPerStageDescriptorSampledImages, p.maxPerStageDescriptorStorageImages, p.maxPerStageDescriptorInputAttachments) = (16, 4, 4);
    (p.maxDescriptorSetSamplers, p.maxDescriptorSetUniformBuffers, p.maxDescriptorSetUniformBuffersDynamic) = (96, 72, 8);
    (p.maxDescriptorSetStorageBuffers, p.maxDescriptorSetStorageBuffersDynamic, p.maxDescriptorSetSampledImages) = (24, 4, 96);
    (p.maxDescriptorSetStorageImages, p.maxDescriptorSetInputAttachments) = (24, 4);

    (p.maxComputeSharedMemorySize, p.maxComputeWorkGroupCount) = (16384, [65535; 3]);
    (p.maxComputeWorkGroupInvocations, p.maxComputeWorkGroupSize) = (128, [128, 128, 64]);

    (p.maxSamplerLodBias, p.minTexelOffset, p.maxTexelOffset) = (15.0, -8, 7); // BorgSampler.MaxLodBias

    (p.minMemoryMapAlignment, p.minTexelBufferOffsetAlignment) = (64, 64);
    (p.minUniformBufferOffsetAlignment, p.minStorageBufferOffsetAlignment, p.nonCoherentAtomSize) = (256, 256, 1);
    (p.storageTexelBufferOffsetAlignmentBytes, p.uniformTexelBufferOffsetAlignmentBytes) = (256, 256);
    (p.storageTexelBufferOffsetSingleTexelAlignment, p.uniformTexelBufferOffsetSingleTexelAlignment) = (VK_TRUE, VK_TRUE);

    (p.framebufferColorSampleCounts, p.framebufferDepthSampleCounts) = (SC_MSAA, SC_MSAA);
    (p.framebufferStencilSampleCounts, p.framebufferNoAttachmentsSampleCounts) = (SC_MSAA, SC_MSAA);
    (p.sampledImageColorSampleCounts, p.sampledImageIntegerSampleCounts) = (SC_MSAA, SC_INTEGER);
    (p.sampledImageDepthSampleCounts, p.sampledImageStencilSampleCounts, p.storageImageSampleCounts) = (SC_MSAA, SC_MSAA, SC_STORAGE);
    (p.maxSampleMaskWords, p.discreteQueuePriorities) = (1, 2);

    (p.maxSamplerAnisotropy, p.maxDrawIndexedIndexValue, p.maxDrawIndirectCount) = (1.0, u32::MAX, u32::MAX);
    (p.pointSizeRange, p.lineWidthRange, p.pointSizeGranularity, p.lineWidthGranularity) = ([1.0, 64.0], [1.0, 1.0], 0.125, 0.0);
    (p.optimalBufferCopyOffsetAlignment, p.optimalBufferCopyRowPitchAlignment) = (1, 1);
    (p.subPixelPrecisionBits, p.subTexelPrecisionBits, p.mipmapPrecisionBits, p.subPixelInterpolationOffsetBits) = (8, 8, 8, 8);

    (p.maxMultiviewViewCount, p.maxMultiviewInstanceIndex, p.maxPerSetDescriptors) = (6, (1 << 27) - 1, 1024);
    p.maxMemoryAllocationSize = 1 << 30;
    p.framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    p.maxTimelineSemaphoreValueDifference = i64::MAX as u64;
    p.maxInlineUniformTotalSize = 256;
    p.driverID = VK_DRIVER_ID_MESA_DOZEN; // until Borg has an ID
    p.conformanceVersion = VkConformanceVersion { major: 1, minor: 3, subminor: 0, patch: 0 };

    set_name(&mut p.deviceName, "Borg (ULX3S)");
    set_name(&mut p.driverName, "borgvk");
    set_name(&mut p.driverInfo, "Mesa borgvk (serial/FPGA)");
    p
}

unsafe fn create(instance: &mut borgvk_instance, drm_fd: i32) -> VkResult {
    let d = unsafe { vk_zalloc(&instance.vk.alloc, size_of::<borgvk_physical_device>(), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE) };
    let Some(d) = (unsafe { d.cast::<borgvk_physical_device>().as_mut() }) else { return vk_error!(instance, VK_ERROR_OUT_OF_HOST_MEMORY).0 };
    d.drm_fd = drm_fd;
    let mut dispatch: vk_physical_device_dispatch_table = unsafe { std::mem::zeroed() };
    let mut extensions: vk_device_extension_table = unsafe { std::mem::zeroed() };
    let mut features: vk_features = unsafe { std::mem::zeroed() };
    unsafe {
        vk_physical_device_dispatch_table_from_entrypoints(&mut dispatch, &borgvk_physical_device_entrypoints, true);
        vk_physical_device_dispatch_table_from_entrypoints(&mut dispatch, &wsi_physical_device_entrypoints, false);
        extensions.__bindgen_anon_1.__bindgen_anon_1.KHR_swapchain = true;
    }
    features.robustBufferAccess = true;
    features.textureCompressionBC = true; // the spec wants one compressed family
    features.largePoints = true;
    let r = unsafe { vk_physical_device_init(&mut d.vk, &mut instance.vk, &extensions, &features, &properties(), &dispatch) };
    if r != VK_SUCCESS {
        unsafe { vk_free(&instance.vk.alloc, std::ptr::from_mut(d).cast()) };
        return r;
    }
    // One host-visible coherent heap: host RAM, staged to the board.
    let m = &mut d.memory;
    m.memoryHeapCount = 1;
    m.memoryHeaps[0].size = 256 << 20;
    m.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    m.memoryTypeCount = 1;
    m.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    m.memoryTypes[0].heapIndex = 0;
    d.sync_types = [&SYNC_TYPE, std::ptr::null()];
    d.vk.supported_sync_types = d.sync_types.as_ptr();
    let r = unsafe { borgvk_wsi_init(d) };
    if r != VK_SUCCESS {
        unsafe {
            vk_physical_device_finish(&mut d.vk);
            vk_free(&instance.vk.alloc, std::ptr::from_mut(d).cast());
        }
        return r;
    }
    unsafe { list_addtail(&mut d.vk.link, &mut instance.vk.physical_devices.list) };
    VK_SUCCESS
}

fn find_borg_drm_device() -> i32 {
    for i in 128..192 {
        let Ok(f) = OpenOptions::new().read(true).write(true).open(format!("/dev/dri/renderD{i}")) else { continue };
        let ver = unsafe { drmGetVersion(f.as_raw_fd()) };
        if ver.is_null() {
            continue;
        }
        let name = unsafe { (*ver).name };
        let borg = !name.is_null() && unsafe { CStr::from_ptr(name) } == c"borg";
        unsafe { drmFreeVersion(ver) };
        if borg {
            return f.into_raw_fd();
        }
    }
    -1
}

/// # Safety
/// `instance` is a borgvk instance.
pub unsafe extern "C" fn enumerate_devices(instance: *mut vk_instance) -> VkResult {
    let fd = find_borg_drm_device();
    if fd >= 0 {
        log(MESA_LOG_INFO, format!("borgvk: found borg DRM device (fd {fd})"));
    } else {
        log(MESA_LOG_INFO, "borgvk: no borg DRM device; using malloc+serial fallback".into());
    }
    unsafe { create(&mut *instance.cast(), fd) }
}

/// # Safety
/// `pdev` is a borgvk physical device.
pub unsafe extern "C" fn destroy_physical_device(pdev: *mut vk_physical_device) {
    let d = pdev.cast::<borgvk_physical_device>();
    unsafe {
        borgvk_wsi_finish(d);
        if (*d).drm_fd >= 0 {
            close((*d).drm_fd);
        }
        vk_physical_device_finish(pdev);
        vk_free(&(*(*pdev).instance).alloc, pdev.cast());
    }
}
