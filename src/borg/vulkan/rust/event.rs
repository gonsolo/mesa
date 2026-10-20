// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkEvent: a flag the host and command buffers set and reset.

use crate::device::Device;
use crate::entrypoint::{entrypoint, handle, vk_error, Handle};
use crate::ffi::*;
use crate::object::{allocate, Zeroed};
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(C)]
pub struct Event {
    base: vk_object_base,
    set: AtomicU64,
}

handle!(VkEvent, Event, VK_OBJECT_TYPE_EVENT);

impl Event {
    fn store(&self, set: bool) {
        self.set.store(set as u64, Ordering::Relaxed);
    }
}

// The runtime's object base, which it zero-allocates, and an atomic.
unsafe impl Zeroed for Event {}

entrypoint! {
    fn borgvk_CreateEvent(device: &Device, _info: *const VkEventCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkEvent) -> Result<(), VkError> {
        let e = unsafe { allocate::<Event>(|n| vk_object_zalloc(device.vk_ptr(), alloc, n, VK_OBJECT_TYPE_EVENT)) };
        *out = VkEvent::from_object(e.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyEvent(device: &Device, event: Option<&mut Event>, alloc: *const VkAllocationCallbacks) {
        if let Some(e) = event {
            unsafe { vk_object_free(device.vk_ptr(), alloc, std::ptr::from_mut(e).cast()) };
        }
    }
}

entrypoint! {
    fn borgvk_GetEventStatus(_device: VkDevice, event: &Event) -> VkResult {
        if event.set.load(Ordering::Relaxed) != 0 { VK_EVENT_SET } else { VK_EVENT_RESET }
    }
}

entrypoint! {
    fn borgvk_SetEvent(_device: VkDevice, event: &Event) -> Result<(), VkError> {
        event.store(true);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_ResetEvent(_device: VkDevice, event: &Event) -> Result<(), VkError> {
        event.store(false);
        Ok(())
    }
}

// Commands set events when they are recorded.
entrypoint! {
    fn borgvk_CmdSetEvent(_cmd: VkCommandBuffer, event: Option<&Event>, _stages: VkPipelineStageFlags) {
        if let Some(e) = event {
            e.store(true);
        }
    }
}

entrypoint! {
    fn borgvk_CmdResetEvent(_cmd: VkCommandBuffer, event: Option<&Event>, _stages: VkPipelineStageFlags) {
        if let Some(e) = event {
            e.store(false);
        }
    }
}

entrypoint! {
    fn borgvk_CmdSetEvent2(_cmd: VkCommandBuffer, event: Option<&Event>, _dep: *const VkDependencyInfo) {
        if let Some(e) = event {
            e.store(true);
        }
    }
}

entrypoint! {
    fn borgvk_CmdResetEvent2(_cmd: VkCommandBuffer, event: Option<&Event>, _stages: VkPipelineStageFlags2) {
        if let Some(e) = event {
            e.store(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_and_commands_set_and_reset() {
        let mut e = Event { base: Default::default(), set: AtomicU64::new(0) };
        e.base.type_ = VK_OBJECT_TYPE_EVENT;
        let h = VkEvent::from_object(&mut e);
        let dev = std::ptr::null_mut();
        unsafe {
            assert_eq!(borgvk_GetEventStatus(dev, h), VK_EVENT_RESET);
            assert_eq!(borgvk_SetEvent(dev, h), VK_SUCCESS);
            assert_eq!(borgvk_GetEventStatus(dev, h), VK_EVENT_SET);
            borgvk_CmdResetEvent2(std::ptr::null_mut(), h, 0);
            assert_eq!(borgvk_GetEventStatus(dev, h), VK_EVENT_RESET);
            borgvk_CmdSetEvent(std::ptr::null_mut(), VkEvent::from_object(std::ptr::null_mut()), 0);
        }
    }
}
