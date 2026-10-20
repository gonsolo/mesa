// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkEvent: a flag the host and command buffers set and reset.

use crate::entrypoint::{entrypoint, handle, vk_error, Handle};
use crate::ffi::*;
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

/// Commands set events when they are recorded.
fn record(event: VkEvent, set: bool) {
    if let Some(e) = unsafe { event.get() } {
        e.store(set);
    }
}

entrypoint! {
    fn borgvk_CreateEvent(device: VkDevice, _info: *const VkEventCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkEvent) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let e = vk_object_zalloc(&mut dev.vk, alloc, size_of::<Event>(), VK_OBJECT_TYPE_EVENT);
        if e.is_null() {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        }
        *out = VkEvent::from_object(e.cast());
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyEvent(device: VkDevice, event: VkEvent, alloc: *const VkAllocationCallbacks) {
        if let Some(e) = event.get() {
            vk_object_free(&mut device.get().unwrap().vk, alloc, (e as *mut Event).cast());
        }
    }
}

entrypoint! {
    fn borgvk_GetEventStatus(_device: VkDevice, event: VkEvent) -> VkResult {
        if event.get().unwrap().set.load(Ordering::Relaxed) != 0 { VK_EVENT_SET } else { VK_EVENT_RESET }
    }
}

entrypoint! {
    fn borgvk_SetEvent(_device: VkDevice, event: VkEvent) -> Result<(), VkError> {
        event.get().unwrap().store(true);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_ResetEvent(_device: VkDevice, event: VkEvent) -> Result<(), VkError> {
        event.get().unwrap().store(false);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_CmdSetEvent(_cmd: VkCommandBuffer, event: VkEvent, _stages: VkPipelineStageFlags) {
        record(event, true);
    }
}

entrypoint! {
    fn borgvk_CmdResetEvent(_cmd: VkCommandBuffer, event: VkEvent, _stages: VkPipelineStageFlags) {
        record(event, false);
    }
}

entrypoint! {
    fn borgvk_CmdSetEvent2(_cmd: VkCommandBuffer, event: VkEvent, _dep: *const VkDependencyInfo) {
        record(event, true);
    }
}

entrypoint! {
    fn borgvk_CmdResetEvent2(_cmd: VkCommandBuffer, event: VkEvent, _stages: VkPipelineStageFlags2) {
        record(event, false);
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
