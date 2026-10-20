// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Driver objects live where the runtime's allocators put them: zeroed memory, initialized in
//! place and never moved.

use crate::ffi::*;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};

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

/// A new object with its teardown, run on drop unless `keep` hands the object out.
pub struct Owned<'a, T: ?Sized, F: FnOnce(&mut T)>(Option<(&'a mut T, F)>);

impl<'a, T: ?Sized, F: FnOnce(&mut T)> Owned<'a, T, F> {
    pub fn new(obj: &'a mut T, free: F) -> Self {
        Self(Some((obj, free)))
    }

    /// Adds a step that undoes the latest init, run before the earlier ones.
    pub fn then(mut self, undo: impl FnOnce(&mut T)) -> Owned<'a, T, impl FnOnce(&mut T)> {
        let (obj, free) = self.0.take().unwrap();
        Owned::new(obj, move |t: &mut T| {
            undo(t);
            free(t)
        })
    }

    pub fn keep(mut self) -> &'a mut T {
        self.0.take().unwrap().0
    }
}

impl<T: ?Sized, F: FnOnce(&mut T)> Deref for Owned<'_, T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_ref().unwrap().0
    }
}

impl<T: ?Sized, F: FnOnce(&mut T)> DerefMut for Owned<'_, T, F> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_mut().unwrap().0
    }
}

impl<T: ?Sized, F: FnOnce(&mut T)> Drop for Owned<'_, T, F> {
    fn drop(&mut self) {
        if let Some((obj, free)) = self.0.take() {
            free(obj)
        }
    }
}

/// A growable byte buffer from an object's allocator (the default one if null), freed on drop.
pub struct VkBuf {
    alloc: *const VkAllocationCallbacks,
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

impl VkBuf {
    /// # Safety
    /// `alloc` is null or outlives the buffer.
    pub unsafe fn new(alloc: *const VkAllocationCallbacks) -> Self {
        Self { alloc, ptr: std::ptr::null_mut(), len: 0, cap: 0 }
    }

    fn callbacks(&self) -> *const VkAllocationCallbacks {
        if self.alloc.is_null() { unsafe { vk_default_allocator() } } else { self.alloc }
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The contents, leaving an empty buffer on the same allocator.
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Self { alloc: self.alloc, ptr: std::ptr::null_mut(), len: 0, cap: 0 })
    }

    /// Appends `data`; false if the allocator fails.
    pub fn extend(&mut self, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        if self.len + data.len() > self.cap {
            let cap = (self.cap * 2).max(self.len + data.len()).max(4096);
            let p = unsafe { vk_realloc(self.callbacks(), self.ptr.cast(), cap, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) };
            if p.is_null() {
                return false;
            }
            (self.ptr, self.cap) = (p.cast(), cap);
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(self.len), data.len()) };
        self.len += data.len();
        true
    }

    /// Replaces the contents; empty if the allocator fails.
    pub fn set(&mut self, data: &[u8]) -> bool {
        self.len = 0;
        let ok = self.extend(data);
        if !ok {
            self.len = 0;
        }
        ok
    }
}

impl Drop for VkBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { vk_free(self.callbacks(), self.ptr.cast()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn owned_tears_down_in_reverse_unless_kept() {
        let log = RefCell::new(Vec::new());
        let (mut a, mut b) = (0, 0);
        {
            let o = Owned::new(&mut a, |_| log.borrow_mut().push("free"));
            let _o = o.then(|_| log.borrow_mut().push("finish"));
        }
        assert_eq!(*log.borrow(), ["finish", "free"]);
        *Owned::new(&mut b, |_| log.borrow_mut().push("kept")).keep() = 1;
        assert_eq!((log.borrow().len(), b), (2, 1));
    }

    #[test]
    fn vkbuf_grows_and_takes() {
        let mut b = unsafe { VkBuf::new(std::ptr::null()) };
        assert!(b.extend(&[1, 2]) && b.extend(&[3; 5000]) && b.extend(&[]));
        assert_eq!((b.as_slice().len(), b.as_slice()[..3].to_vec()), (5002, vec![1, 2, 3]));
        let t = b.take();
        assert!(b.is_empty() && t.as_slice().len() == 5002);
        assert!(b.set(&[7]) && b.set(&[8, 9]) && b.as_slice() == [8, 9]);
    }
}
