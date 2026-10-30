// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The instance and the logical device.

use crate::cmd_buffer::CMD_BUFFER_OPS;
use crate::entrypoint::{chain, entrypoint, handle, vk_error, Handle};
use crate::ffi::*;
use crate::object::{allocate, Owned, Zeroed};
use crate::physical_device::{destroy_physical_device, enumerate_devices, API_VERSION};
use crate::queue::borgvk_queue_submit;
use std::ffi::{c_char, c_void};
use std::ptr::null;
use std::sync::{LazyLock, Mutex};

handle!(VkInstance, Instance, VK_OBJECT_TYPE_INSTANCE);
handle!(VkDevice, Device, VK_OBJECT_TYPE_DEVICE);

#[repr(C)]
pub struct Instance {
    pub vk: vk_instance,
}

#[repr(C)]
pub struct Device {
    pub vk: vk_device,
    pub cmd_dispatch: vk_device_dispatch_table,
    pub queue: vk_queue,
    pub drm_fd: i32, // the physical device's; -1 is the malloc and serial fallback
    // Locked: the runtime's submit thread replays draws while the app builds pipelines.
    pub scratch: Mutex<Scratch>,
}

/// What pipeline creation compiles into, and draws and submits upload from.
pub struct Scratch {
    pub blend_lower: *const borg_blend,
    pub shader_blob: [ShaderBlob; BORGVK_SHADER_STAGE_COUNT as usize],
    pub compile_layout: VkPipelineLayout,
    pub stage_ubo: [u32; 2],
    pub stage_push: [bool; 2],
    pub stage_pwin: [bool; 2],  // the stage reads push constants from the heap window
    pub stage_spill: [bool; 2], // the stage spills to the region's scratch slot
    pub frag_sample_mask: bool, // the fragment shader writes gl_SampleMask
    pub frag_tex_binding: i32,
    pub vert_tex_binding: i32,
    pub tex_pairs: Vec<[u8; 2]>, // the pipeline's descriptor indices: texture and sampler flat slots
    pub frag_select: u32, // 1 + the colour output the fragment shader keeps; 0 keeps all
    pub frag_reads_pntc: bool,
    pub blend_cfg: u32,
    pub blend_const: u32,
    pub state_reg: [u32; 5],
    pub vp: Vp, // the viewport and scissor of the draw being queued
    pub bias: [f32; 2],
    pub state_valid: bool,
    pub push_words: [u32; 32], // set by vkCmdPushConstants, replayed at submit (DRM path)
    pub push_lo: u32,          // word range written so far; empty when lo >= hi
    pub push_hi: u32,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            blend_lower: null(),
            shader_blob: Default::default(),
            compile_layout: std::ptr::null_mut(),
            stage_ubo: [0; 2],
            stage_push: [false; 2],
            stage_pwin: [false; 2],
            stage_spill: [false; 2],
            frag_sample_mask: false,
            frag_tex_binding: 0,
            vert_tex_binding: -1,
            tex_pairs: Default::default(),
            frag_select: 0,
            frag_reads_pntc: false,
            blend_cfg: 0,
            blend_const: 0,
            state_reg: [0; 5],
            vp: Vp::default(),
            bias: [0.0; 2],
            state_valid: false,
            push_words: [0; 32],
            push_lo: 0,
            push_hi: 0,
        }
    }
}

impl Device {
    /// For the runtime's calls; they synchronize what they change themselves.
    pub fn vk_ptr(&self) -> *mut vk_device {
        std::ptr::from_ref(&self.vk).cast_mut()
    }
}

// The runtime's instance, which vk_instance_init zeroes.
unsafe impl Zeroed for Instance {}

// By hand: bindgen leaves out the runtime's hidden tables.
extern "C" {
    static vk_cmd_enqueue_device_entrypoints: vk_device_entrypoint_table;
    static vk_common_device_entrypoints: vk_device_entrypoint_table;
}

fn error(r: VkResult) -> VkResult {
    unsafe { __vk_errorf(null(), r, concat!(file!(), "\0").as_ptr().cast(), line!() as i32, null()) }
}

// The runtime keeps a pointer to it.
static INSTANCE_EXTENSIONS: LazyLock<vk_instance_extension_table> = LazyLock::new(|| {
    let mut t: vk_instance_extension_table = unsafe { std::mem::zeroed() };
    let e = unsafe { &mut t.__bindgen_anon_1.__bindgen_anon_1 };
    e.KHR_external_fence_capabilities = true;
    e.KHR_external_memory_capabilities = true;
    e.KHR_external_semaphore_capabilities = true;
    e.KHR_get_physical_device_properties2 = true;
    e.EXT_debug_report = true;
    e.EXT_debug_utils = true;
    e.KHR_surface = true;
    e.KHR_get_surface_capabilities2 = true;
    e.EXT_headless_surface = true;
    #[cfg(vk_x11)]
    {
        e.KHR_xcb_surface = true;
        e.KHR_xlib_surface = true;
    }
    #[cfg(vk_wayland)]
    {
        e.KHR_wayland_surface = true;
    }
    t
});

entrypoint! {
    fn borgvk_EnumerateInstanceVersion(version: &mut u32) -> VkResult {
        *version = API_VERSION;
        VK_SUCCESS
    }
}

entrypoint! {
    unsafe fn borgvk_EnumerateInstanceExtensionProperties(layer: *const c_char, count: *mut u32, props: *mut VkExtensionProperties) -> VkResult {
        if !layer.is_null() {
            return error(VK_ERROR_LAYER_NOT_PRESENT);
        }
        vk_enumerate_instance_extension_properties(&*INSTANCE_EXTENSIONS, count, props)
    }
}

entrypoint! {
    fn borgvk_EnumerateInstanceLayerProperties(count: &mut u32, props: *mut VkLayerProperties) -> VkResult {
        if props.is_null() {
            *count = 0;
            return VK_SUCCESS;
        }
        error(VK_ERROR_LAYER_NOT_PRESENT)
    }
}

entrypoint! {
    unsafe fn borgvk_CreateInstance(info: *const VkInstanceCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkInstance) -> VkResult {
        // A Vulkan 1.0 implementation rejects a newer requested version.
        if let Some(app) = (*info).pApplicationInfo.as_ref().filter(|a| a.apiVersion != 0) {
            let major = |v: u32| v >> 22 & 0x7f;
            let minor = |v: u32| v >> 12 & 0x3ff;
            if major(app.apiVersion) != major(API_VERSION) || minor(app.apiVersion) > minor(API_VERSION) {
                return error(VK_ERROR_INCOMPATIBLE_DRIVER);
            }
        }
        let alloc = if alloc.is_null() { vk_default_allocator() } else { alloc };
        let Some(inst) = allocate::<Instance>(|n| vk_zalloc(alloc, n, 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE)) else {
            return error(VK_ERROR_OUT_OF_HOST_MEMORY);
        };
        let mut inst = Owned::new(inst, |i| vk_free(alloc, std::ptr::from_mut(i).cast()));
        let mut dispatch: vk_instance_dispatch_table = std::mem::zeroed();
        vk_instance_dispatch_table_from_entrypoints(&mut dispatch, &borgvk_instance_entrypoints, true);
        vk_instance_dispatch_table_from_entrypoints(&mut dispatch, &wsi_instance_entrypoints, false);
        let r = vk_instance_init(&mut inst.vk, &*INSTANCE_EXTENSIONS, &dispatch, info, alloc);
        if r != VK_SUCCESS {
            return error(r);
        }
        inst.vk.physical_devices.enumerate = Some(enumerate_devices);
        inst.vk.physical_devices.destroy = Some(destroy_physical_device);
        *out = VkInstance::from_object(inst.keep());
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_DestroyInstance(instance: Option<&mut Instance>, _alloc: *const VkAllocationCallbacks) {
        let Some(i) = instance else { return };
        unsafe {
            vk_instance_finish(&mut i.vk);
            vk_free(&i.vk.alloc, std::ptr::from_mut(i).cast());
        }
    }
}

/// The body of the vkCmd* entries borgvk does not implement, for replay.
unsafe extern "C" fn noop_cmd() {}

// The common CmdSetDepthBias goes through the device's table, which records: replay calls 2EXT itself.
unsafe extern "C" fn replay_set_depth_bias(cb: VkCommandBuffer, constant: f32, clamp: f32, slope: f32) {
    let i = VkDepthBiasInfoEXT {
        sType: VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT,
        pNext: null(),
        depthBiasConstantFactor: constant,
        depthBiasClamp: clamp,
        depthBiasSlopeFactor: slope,
    };
    unsafe { vk_common_device_entrypoints.CmdSetDepthBias2EXT.unwrap()(cb, &i) };
}

entrypoint! {
    unsafe fn borgvk_CreateDevice(physical_device: VkPhysicalDevice, info: *const VkDeviceCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkDevice) -> VkResult {
        let pdev = physical_device.get().unwrap();
        for s in chain((*info).pNext.cast_mut()) {
            let unsupported = match (*s).sType {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES => (*s.cast::<VkPhysicalDeviceProtectedMemoryFeatures>()).protectedMemory != 0,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => (*s.cast::<VkPhysicalDeviceShaderDrawParametersFeatures>()).shaderDrawParameters != 0,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                    let v = &*s.cast::<VkPhysicalDeviceVulkan11Features>();
                    v.protectedMemory != 0 || v.shaderDrawParameters != 0
                }
                _ => false,
            };
            if unsupported {
                return vk_error!(pdev, VK_ERROR_FEATURE_NOT_PRESENT).0;
            }
        }
        let ialloc = &(*pdev.vk.instance).alloc;
        let d = vk_zalloc2(ialloc, alloc, size_of::<Device>(), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE).cast::<Device>();
        if d.is_null() {
            return vk_error!(pdev, VK_ERROR_OUT_OF_HOST_MEMORY).0;
        }
        std::ptr::addr_of_mut!((*d).scratch).write(Mutex::new(Scratch::default()));
        let mut d = Owned::new(&mut *d, |d| {
            std::ptr::drop_in_place(&mut d.scratch);
            vk_free2(ialloc, alloc, std::ptr::from_mut(d).cast());
        });

        // These record, and replay through cmd_dispatch at submit; they win over borgvk's.
        let e = &vk_cmd_enqueue_device_entrypoints;
        let mut rec: vk_device_entrypoint_table = std::mem::zeroed();
        macro_rules! record {
            ($($n:ident),*) => { $(rec.$n = e.$n;)* };
        }
        record!(
            CmdPushConstants, CmdBeginRendering, CmdEndRendering, CmdClearAttachments, CmdClearColorImage,
            CmdClearDepthStencilImage, CmdBlitImage, CmdBlitImage2, CmdCopyBuffer, CmdCopyBuffer2, CmdCopyBufferToImage,
            CmdCopyBufferToImage2, CmdCopyImage, CmdCopyImage2, CmdCopyImageToBuffer, CmdCopyImageToBuffer2,
            CmdResolveImage, CmdResolveImage2, CmdFillBuffer, CmdUpdateBuffer, CmdSetEvent, CmdResetEvent, CmdSetEvent2,
            CmdResetEvent2, CmdBindPipeline, CmdBindVertexBuffers, CmdBindDescriptorSets, CmdDraw, CmdDrawIndexed,
            CmdBindIndexBuffer, CmdDispatch, CmdDispatchIndirect, CmdDrawIndirect, CmdDrawIndexedIndirect
        );
        let mut dt: vk_device_dispatch_table = std::mem::zeroed();
        vk_device_dispatch_table_from_entrypoints(&mut dt, &rec, true);
        vk_device_dispatch_table_from_entrypoints(&mut dt, &borgvk_device_entrypoints, false);
        vk_device_dispatch_table_from_entrypoints(&mut dt, &wsi_device_entrypoints, false);
        vk_device_dispatch_table_from_entrypoints(&mut dt, e, false);
        vk_device_dispatch_table_from_entrypoints(&mut dt, &vk_common_device_entrypoints, false);

        // Replay of a recorded queue: borgvk's own entry points, no-ops for the rest.
        vk_device_dispatch_table_from_entrypoints(&mut d.cmd_dispatch, &borgvk_device_entrypoints, true);
        let cd = &mut d.cmd_dispatch;
        if cd.CmdExecuteCommands.is_none() {
            cd.CmdExecuteCommands = vk_common_device_entrypoints.CmdExecuteCommands;
        }
        // Dynamic state goes to the command buffer's vk_dynamic_graphics_state, which draws read.
        let vc = &vk_common_device_entrypoints;
        (cd.CmdSetViewport, cd.CmdSetScissor, cd.CmdSetLineWidth, cd.CmdSetBlendConstants) =
            (vc.CmdSetViewport, vc.CmdSetScissor, vc.CmdSetLineWidth, vc.CmdSetBlendConstants);
        cd.CmdSetDepthBias = Some(replay_set_depth_bias);
        (cd.CmdSetDepthBounds, cd.CmdSetStencilCompareMask, cd.CmdSetStencilWriteMask, cd.CmdSetStencilReference) =
            (vc.CmdSetDepthBounds, vc.CmdSetStencilCompareMask, vc.CmdSetStencilWriteMask, vc.CmdSetStencilReference);
        let n = size_of::<vk_device_dispatch_table>() / size_of::<*const c_void>();
        let slots = std::slice::from_raw_parts_mut(std::ptr::from_mut(&mut d.cmd_dispatch).cast::<*const c_void>(), n);
        for s in slots.iter_mut().filter(|s| s.is_null()) {
            *s = noop_cmd as *const c_void;
        }

        let r = vk_device_init(&mut d.vk, &mut pdev.vk, &dt, info, alloc);
        if r != VK_SUCCESS {
            return vk_error!(pdev, r).0;
        }
        let mut d = d.then(|d| vk_device_finish(&mut d.vk));
        vk_device_enable_threaded_submit(&mut d.vk);
        d.vk.command_dispatch_table = &raw const d.cmd_dispatch;
        d.vk.command_buffer_ops = &CMD_BUFFER_OPS;
        let r = vk_queue_init(&mut d.queue, &mut d.vk, (*info).pQueueCreateInfos, 0);
        if r != VK_SUCCESS {
            return vk_error!(pdev, r).0;
        }
        d.drm_fd = pdev.drm_fd; // owned by the physical device
        d.queue.driver_submit = Some(borgvk_queue_submit);
        *out = VkDevice::from_object(d.keep());
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_DestroyDevice(device: Option<&mut Device>, alloc: *const VkAllocationCallbacks) {
        let Some(d) = device else { return };
        unsafe {
            vk_queue_finish(&mut d.queue);
            vk_device_finish(&mut d.vk);
            std::ptr::drop_in_place(&mut d.scratch);
            vk_free2(&d.vk.alloc, alloc, std::ptr::from_mut(d).cast());
        }
    }
}

/// # Safety
/// `name` is a C string; `instance` is null or a borgvk instance.
#[no_mangle]
pub unsafe extern "C" fn borgvk_GetInstanceProcAddr(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    let vk = unsafe { instance.get() }.map_or(std::ptr::null_mut(), |i| &mut i.vk);
    unsafe { vk_instance_get_proc_addr(vk, &borgvk_instance_entrypoints, name) }
}

/// # Safety
/// As `borgvk_GetInstanceProcAddr`.
#[no_mangle]
pub unsafe extern "C" fn vk_icdGetInstanceProcAddr(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { borgvk_GetInstanceProcAddr(instance, name) }
}
