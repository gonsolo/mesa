// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Driver objects live where the runtime's allocators put them: zeroed memory, initialized in
//! place and never moved.

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn owned_tears_down_unless_kept() {
        let log = RefCell::new(Vec::new());
        let (mut a, mut b) = (0, 0);
        {
            let _o = Owned::new(&mut a, |_| log.borrow_mut().push("free"));
        }
        assert_eq!(*log.borrow(), ["free"]);
        *Owned::new(&mut b, |_| log.borrow_mut().push("kept")).keep() = 1;
        assert_eq!((log.borrow().len(), b), (1, 1));
    }
}
