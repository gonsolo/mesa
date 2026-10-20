// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Driver objects live where the runtime's allocators put them: zeroed memory, initialized in
//! place and never moved.

use std::ffi::c_void;

/// A type whose all-zero bit pattern is a valid value.
///
/// # Safety
/// Every field must accept all zeros: integers, floats, bools, raw pointers, `Option` of
/// references or boxes, and runtime structs the runtime itself zero-allocates.
pub unsafe trait Zeroed: Sized {}

/// The zeroed `T` a runtime allocator returns for `size_of::<T>()` bytes, or `None`.
///
/// # Safety
/// `alloc` returns null or zeroed memory of the size it is given, aligned for `T`.
pub unsafe fn allocate<'a, T: Zeroed>(alloc: impl FnOnce(usize) -> *mut c_void) -> Option<&'a mut T> {
    unsafe { alloc(size_of::<T>()).cast::<T>().as_mut() }
}
