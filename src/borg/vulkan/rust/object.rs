// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Driver objects live where the runtime's allocator put them: zeroed memory, initialized in
//! place and never moved.

use crate::ffi::*;

/// A type whose all-zero bit pattern is a valid value.
///
/// # Safety
/// Every field must accept all zeros: integers, floats, bools, raw pointers, `Option` of
/// references or boxes, and runtime structs the runtime itself zero-allocates.
pub unsafe trait Zeroed: Sized {}

/// A zeroed `T` from `alloc`, to be initialized in place.
pub fn zalloc<'a, T: Zeroed>(alloc: *const VkAllocationCallbacks, scope: VkSystemAllocationScope) -> Option<&'a mut T> {
    unsafe { vk_zalloc(alloc, size_of::<T>(), align_of::<T>(), scope).cast::<T>().as_mut() }
}
