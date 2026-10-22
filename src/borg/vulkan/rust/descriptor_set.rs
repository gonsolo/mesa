// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Descriptor set layouts, pools and sets. A layout maps its (binding, array element) pairs to
//! flat slots, at most BORGVK_MAX_BINDINGS of them; a set holds one descriptor per slot.

use crate::entrypoint::{entrypoint, handle, pointee, vk_error, Handle};
use crate::ffi::*;
use crate::vk::*;
use crate::memory::Buffer;
use crate::sampler::Sampler;
use crate::image::Image;
use crate::device::Device;
use crate::object::{allocate, Owned, Zeroed};
use std::ptr::null;

handle!(VkDescriptorSetLayout, DescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
handle!(VkDescriptorPool, DescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
handle!(VkDescriptorSet, DescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET);

#[repr(C)]
pub struct DescriptorSetLayout {
    pub vk: vk_descriptor_set_layout,
    pub binding_count: u32,
    pub dyn_mask: u64, // slots of *_BUFFER_DYNAMIC descriptors
    pub map: DescMap,
}

#[repr(C)]
pub struct DescriptorPool {
    pub base: vk_object_base,
    pub sets: list_head, // every live set allocated from this pool
    pub max_sets: u32,
    pub num_sets: u32,
}

#[repr(C)]
pub struct DescriptorSet {
    pub base: vk_object_base,
    pub link: list_head, // in the pool's list
    pub pool: *mut DescriptorPool,
    pub buffers: [*mut Buffer; BINDINGS],
    pub offsets: [VkDeviceSize; BINDINGS],
    pub ranges: [VkDeviceSize; BINDINGS],
    pub images: [*mut Image; BINDINGS],
    pub views: [*mut vk_image_view; BINDINGS], // behind images[]: view type, format, mip/layer range, swizzle
    pub samplers: [*mut Sampler; BINDINGS],
    pub buffer_views: [*mut vk_buffer_view; BINDINGS], // texel buffers
    pub dyn_mask: u64,      // from the layout: dynamic-offset slots
    pub storage_mask: u32,  // slots whose buffer is a storage buffer
    pub sampled_mask: u32,  // slots whose image is sampled (not storage)
    pub map: DescMap,
}

const BINDINGS: usize = BORGVK_MAX_BINDINGS as usize;

// A set's slots hold objects the app keeps alive while the set is in use.
impl DescriptorSet {
    pub fn buffer(&self, b: usize) -> Option<&Buffer> {
        unsafe { self.buffers[b].as_ref() }
    }

    pub fn image(&self, b: usize) -> Option<&Image> {
        unsafe { self.images[b].as_ref() }
    }

    pub fn view(&self, b: usize) -> Option<&vk_image_view> {
        unsafe { self.views[b].as_ref() }
    }

    pub fn sampler(&self, b: usize) -> Option<&Sampler> {
        unsafe { self.samplers[b].as_ref() }
    }

    pub fn buffer_view(&self, b: usize) -> Option<&vk_buffer_view> {
        unsafe { self.buffer_views[b].as_ref() }
    }
}

// Numbers, raw pointers, the binding map and runtime bases the runtime zero-allocates.
unsafe impl Zeroed for DescriptorSetLayout {}
unsafe impl Zeroed for DescriptorPool {}
unsafe impl Zeroed for DescriptorSet {}

const MAX: usize = BORGVK_MAX_BINDINGS as usize;

fn bit(slot: u32) -> u32 {
    1u32.wrapping_shl(slot)
}

/// The slot map of a layout's bindings and the slots of its dynamic buffers: bindings in
/// order, each array taking as many slots as it has elements. Bindings past the slot count
/// follow the others.
pub fn layout_map(bindings: &[VkDescriptorSetLayoutBinding]) -> (DescMap, u64) {
    let mut sorted: Vec<_> = bindings.iter().map(|b| {
        let dynamic = matches!(b.descriptorType, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
        (b.binding, b.descriptorCount.max(1).min(MAX as u32), dynamic)
    }).collect();
    sorted.sort_by_key(|b| b.0);
    let (mut m, mut dyn_mask, mut extra) = (DescMap::default(), 0u64, 0u32);
    for (b, count, dynamic) in sorted {
        let slot = if b < MAX as u32 { b + extra } else { m.total as u32 };
        if slot >= MAX as u32 {
            break;
        }
        let cnt = count.min(MAX as u32 - slot);
        let n = m.n as usize;
        (m.num[n], m.slot[n], m.count[n]) = (b, slot as u8, cnt as u8);
        m.n += 1;
        m.arrays |= cnt > 1;
        if dynamic {
            dyn_mask |= ((1u64 << cnt) - 1) << slot;
        }
        extra += cnt - 1;
        m.total = m.total.max((slot + cnt) as u8);
    }
    (m, dyn_mask)
}

/// The slot of element `elem` of `binding`; past the end of an array the next bindings follow.
pub fn desc_slot(m: &DescMap, binding: u32, mut elem: u32) -> Option<u32> {
    if m.n == 0 {
        let slot = binding + elem;
        return (slot < MAX as u32).then_some(slot);
    }
    let n = m.n as usize;
    let mut j = (0..n).find(|&j| m.num[j] as u32 == binding)?;
    while elem >= m.count[j] as u32 {
        elem -= m.count[j] as u32;
        j += 1;
        if j == n {
            return None;
        }
    }
    Some(m.slot[j] as u32 + elem)
}

pointee!(VkDescriptorSetLayoutCreateInfo, VkDescriptorSetLayoutSupport, VkDescriptorPoolCreateInfo, VkDescriptorSetAllocateInfo);

entrypoint! {
    fn borgvk_CreateDescriptorSetLayout(device: &Device, info: &VkDescriptorSetLayoutCreateInfo, _alloc: *const VkAllocationCallbacks, out: &mut VkDescriptorSetLayout) -> Result<(), VkError> {
        let l = unsafe { allocate::<DescriptorSetLayout>(|n| vk_descriptor_set_layout_zalloc(device.vk_ptr(), n, info)) };
        let l = l.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?;
        l.binding_count = info.bindingCount;
        (l.map, l.dyn_mask) = layout_map(info.bindings());
        *out = VkDescriptorSetLayout::from_object(l);
        Ok(())
    }
}

// DestroyDescriptorSetLayout is the common, refcounted entry point.

entrypoint! {
    fn borgvk_GetDescriptorSetLayoutSupport(_device: VkDevice, info: &VkDescriptorSetLayoutCreateInfo, support: &mut VkDescriptorSetLayoutSupport) {
        support.supported = (info.bindingCount <= BORGVK_MAX_BINDINGS) as VkBool32;
    }
}

entrypoint! {
    fn borgvk_CreateDescriptorPool(device: &Device, info: &VkDescriptorPoolCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkDescriptorPool) -> Result<(), VkError> {
        let p = unsafe { allocate::<DescriptorPool>(|n| vk_object_zalloc(device.vk_ptr(), alloc, n, VK_OBJECT_TYPE_DESCRIPTOR_POOL)) };
        let p = p.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?;
        unsafe { list_inithead(&mut p.sets) };
        p.max_sets = info.maxSets;
        *out = VkDescriptorPool::from_object(p);
        Ok(())
    }
}

/// Unlinks a set from its pool and frees it.
unsafe fn free_set(dev: &Device, set: *mut DescriptorSet) {
    unsafe {
        list_del(&mut (*set).link);
        (*(*set).pool).num_sets = (*(*set).pool).num_sets.wrapping_sub(1);
        vk_object_free(dev.vk_ptr(), null(), set.cast());
    }
}

fn free_pool_sets(dev: &Device, pool: &mut DescriptorPool) {
    let head: *mut list_head = &mut pool.sets;
    let mut p = pool.sets.next;
    while p != head {
        let next = unsafe { (*p).next };
        let set = unsafe { p.byte_sub(std::mem::offset_of!(DescriptorSet, link)) }.cast::<DescriptorSet>();
        unsafe { free_set(dev, set) };
        p = next;
    }
    pool.num_sets = 0;
}

entrypoint! {
    fn borgvk_DestroyDescriptorPool(device: &Device, pool: Option<&mut DescriptorPool>, alloc: *const VkAllocationCallbacks) {
        if let Some(p) = pool {
            free_pool_sets(device, p);
            unsafe { vk_object_free(device.vk_ptr(), alloc, std::ptr::from_mut(p).cast()) };
        }
    }
}

entrypoint! {
    fn borgvk_ResetDescriptorPool(device: &Device, pool: &mut DescriptorPool, _flags: VkDescriptorPoolResetFlags) -> VkResult {
        free_pool_sets(device, pool);
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_AllocateDescriptorSets(device: &Device, info: &VkDescriptorSetAllocateInfo, out: *mut VkDescriptorSet) -> Result<(), VkError> {
        let (dev, i) = (device, info);
        // The pool is externally synchronized.
        let pool = unsafe { i.descriptorPool.get() }.unwrap();
        if pool.num_sets.wrapping_add(i.descriptorSetCount) > pool.max_sets {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_POOL_MEMORY));
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out, i.descriptorSetCount as usize) };
        out.fill(VkDescriptorSet::from_object(std::ptr::null_mut()));
        let mut out = Owned::new(out, |out| {
            for h in out.iter_mut().filter(|h| !h.ptr().is_null()) {
                unsafe { free_set(dev, h.ptr()) };
                *h = VkDescriptorSet::from_object(std::ptr::null_mut());
            }
        });
        for k in 0..out.len() {
            let set = unsafe { allocate::<DescriptorSet>(|n| vk_object_zalloc(dev.vk_ptr(), null(), n, VK_OBJECT_TYPE_DESCRIPTOR_SET)) };
            let set = set.ok_or_else(|| vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY))?;
            if !i.pSetLayouts.is_null() {
                if let Some(l) = unsafe { (*i.pSetLayouts.add(k)).get_ref() } {
                    (set.dyn_mask, set.map) = (l.dyn_mask, l.map);
                }
            }
            set.pool = pool;
            unsafe { list_addtail(&mut set.link, &mut pool.sets) };
            pool.num_sets = pool.num_sets.wrapping_add(1);
            out[k] = VkDescriptorSet::from_object(set);
        }
        out.keep();
        Ok(())
    }
}

entrypoint! {
    fn borgvk_FreeDescriptorSets(device: &Device, _pool: VkDescriptorPool, _count: u32, sets: &[VkDescriptorSet]) -> VkResult {
        for s in sets.iter().filter(|s| !s.is_null()) {
            unsafe { free_set(device, s.ptr()) };
        }
        VK_SUCCESS
    }
}

fn set_bit(mask: &mut u32, slot: u32, on: bool) {
    *mask = (*mask & !bit(slot)) | if on { bit(slot) } else { 0 };
}

/// Copies slot `s` of `src` to slot `d` of `dst`, which may be the same set.
unsafe fn copy_slot(dst: *mut DescriptorSet, d: usize, src: *const DescriptorSet, s: usize) {
    let from = unsafe { &*src };
    let v = (from.buffers[s], from.offsets[s], from.ranges[s], from.images[s], from.views[s]);
    let (sampled, sampler, bview) = (from.sampled_mask & bit(s as u32) != 0, from.samplers[s], from.buffer_views[s]);
    let to = unsafe { &mut *dst };
    (to.buffers[d], to.offsets[d], to.ranges[d], to.images[d], to.views[d]) = v;
    set_bit(&mut to.sampled_mask, d as u32, sampled);
    (to.samplers[d], to.buffer_views[d]) = (sampler, bview);
}

unsafe fn write(w: &VkWriteDescriptorSet) {
    let set = unsafe { w.dstSet.get() }.unwrap();
    let ty = w.descriptorType;
    for k in 0..w.descriptorCount as usize {
        let Some(slot) = desc_slot(&set.map, w.dstBinding, w.dstArrayElement + k as u32) else {
            break;
        };
        let s = slot as usize;
        if !w.pBufferInfo.is_null()
            && matches!(
                ty,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                    | VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                    | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
            )
        {
            let b = unsafe { &*w.pBufferInfo.add(k) };
            (set.buffers[s], set.offsets[s], set.ranges[s]) = (b.buffer.ptr(), b.offset, b.range);
            let storage = matches!(ty, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
            set_bit(&mut set.storage_mask, slot, storage);
        }
        // Texel buffers: the view carries the buffer, format, offset and range.
        if !w.pTexelBufferView.is_null()
            && matches!(ty, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER | VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        {
            set.buffer_views[s] = unsafe { *w.pTexelBufferView.add(k) }.ptr().cast();
        }
        let image = if w.pImageInfo.is_null() { None } else { Some(unsafe { &*w.pImageInfo.add(k) }) };
        if let Some(info) = image {
            if matches!(ty, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER | VK_DESCRIPTOR_TYPE_SAMPLER) && !info.sampler.is_null() {
                set.samplers[s] = info.sampler.ptr();
            }
            let is_image = matches!(
                ty,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    | VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                    | VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
                    | VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            );
            if is_image {
                if let Some(view) = unsafe { info.imageView.get() } {
                    set.images[s] = view.image.cast();
                    set.views[s] = view;
                    set_bit(&mut set.sampled_mask, slot, ty != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_UpdateDescriptorSets(_device: VkDevice, _nwrites: u32, writes: &[VkWriteDescriptorSet], _ncopies: u32, copies: &[VkCopyDescriptorSet]) {
        for w in writes {
            unsafe { write(w) };
        }
        for c in copies {
            let (src, dst) = (c.srcSet.ptr(), c.dstSet.ptr());
            for k in 0..c.descriptorCount {
                let s = desc_slot(unsafe { &(*src).map }, c.srcBinding, c.srcArrayElement + k);
                let d = desc_slot(unsafe { &(*dst).map }, c.dstBinding, c.dstArrayElement + k);
                let (Some(s), Some(d)) = (s, d) else {
                    break;
                };
                unsafe { copy_slot(dst, d as usize, src, s as usize) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(binding: u32, count: u32, ty: VkDescriptorType) -> VkDescriptorSetLayoutBinding {
        VkDescriptorSetLayoutBinding { binding, descriptorCount: count, descriptorType: ty, ..Default::default() }
    }

    #[test]
    fn arrays_take_consecutive_slots() {
        let b = [
            binding(0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
            binding(1, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC),
            binding(4, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        ];
        let (m, dyn_mask) = layout_map(&b);
        assert_eq!((m.n, &m.num[..3], &m.slot[..3], &m.count[..3]), (3, &[0, 1, 4][..], &[0, 1, 6][..], &[1, 3, 1][..]));
        assert_eq!((m.total, m.arrays, dyn_mask), (7, true, 0b1110));
        assert_eq!(desc_slot(&m, 1, 2), Some(3));
        assert_eq!(desc_slot(&m, 1, 3), Some(6));
        assert_eq!(desc_slot(&m, 4, 1), None);
        assert_eq!(desc_slot(&m, 2, 0), None);
    }

    #[test]
    fn large_binding_numbers_follow_the_others() {
        let ty = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        let (m, _) = layout_map(&[binding(65534, 1, ty), binding(2, 2, ty), binding(32766, 1, ty)]);
        assert_eq!((m.n, &m.num[..3], &m.slot[..3], m.total), (3, &[2, 32766, 65534][..], &[2, 4, 5][..], 6));
        assert_eq!(desc_slot(&m, 65534, 0), Some(5));
    }

    #[test]
    fn without_a_map_bindings_are_slots() {
        let m = DescMap::default();
        assert_eq!(desc_slot(&m, 3, 2), Some(5));
        assert_eq!(desc_slot(&m, 15, 1), None);
    }

    #[test]
    fn slots_stop_at_the_limit() {
        let (m, _) = layout_map(&[binding(0, 20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE), binding(1, 1, VK_DESCRIPTOR_TYPE_SAMPLER)]);
        assert_eq!((m.n, m.count[0], m.total), (1, 16, 16));
    }
}
