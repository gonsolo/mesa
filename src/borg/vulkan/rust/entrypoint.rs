// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! What the entry points share: handles to driver objects, errors to `VkResult`.

use crate::ffi::*;

/// A failed call: the result, already logged by vk_error.
pub struct VkError(pub VkResult);

/// `vk_error(obj, result)`: logs the error against the object and returns it.
macro_rules! vk_error {
    ($obj:expr, $r:expr) => {
        $crate::entrypoint::VkError(unsafe {
            $crate::ffi::__vk_errorf(
                std::ptr::from_ref(&*$obj).cast(),
                $r,
                concat!(file!(), "\0").as_ptr().cast(),
                line!() as i32,
                std::ptr::null(),
            )
        })
    };
}
pub(crate) use vk_error;

/// A Vulkan handle and the driver object behind it, whose `vk_object_base` comes first.
pub trait Handle: Copy {
    type Object;
    const TYPE: VkObjectType;

    fn ptr(self) -> *mut Self::Object;

    fn from_object(o: *mut Self::Object) -> Self;

    /// # Safety
    /// The handle is null or a live object of this type that nothing else changes meanwhile.
    unsafe fn get<'a>(self) -> Option<&'a mut Self::Object> {
        let o = self.ptr();
        debug_assert!(o.is_null() || unsafe { (*o.cast::<vk_object_base>()).type_ } == Self::TYPE);
        unsafe { o.as_mut() }
    }

    /// # Safety
    /// The handle is null or a live object of this type.
    unsafe fn get_ref<'a>(self) -> Option<&'a Self::Object> {
        unsafe { self.get() }.map(|o| &*o)
    }
}

macro_rules! handle {
    ($h:ty, $obj:ty, $t:expr) => {
        impl $crate::entrypoint::Handle for $h {
            type Object = $obj;
            const TYPE: $crate::ffi::VkObjectType = $t;

            fn ptr(self) -> *mut $obj {
                self.cast()
            }

            fn from_object(o: *mut $obj) -> Self {
                o.cast()
            }
        }

        impl $crate::entrypoint::Pointee for $obj {
            type Raw = $h;
            unsafe fn ptr(raw: $h) -> *mut $obj {
                let o: *mut $obj = raw.cast();
                debug_assert!(o.is_null() || unsafe { (*o.cast::<$crate::ffi::vk_object_base>()).type_ } == $t);
                o
            }
        }

        impl $crate::entrypoint::Pointee for $h {
            type Raw = *mut $h;
            unsafe fn ptr(raw: *mut $h) -> *mut $h {
                raw
            }
        }
    };
}
pub(crate) use handle;


/// What a parameter of a safe entry point is converted from at the boundary: a handle, or a
/// pointer to a Vulkan struct or an out value.
pub trait Pointee {
    type Raw: Copy;

    /// # Safety
    /// `raw` is null or points at a live `Self`.
    unsafe fn ptr(raw: Self::Raw) -> *mut Self;
}

/// A parameter of a safe entry point, converted from its C type once, at the boundary. The
/// parameters convert in order, and an array takes its length from the last `u32` before it
/// (check_rust_signatures.py holds that against vk.xml).
pub trait Param: Sized {
    type Raw;

    /// # Safety
    /// `raw` meets the Vulkan valid usage of the command; `len` is the last `u32` so far.
    unsafe fn from_raw(raw: Self::Raw, len: &mut u32) -> Self;
}

impl<T: Pointee> Param for &T {
    type Raw = T::Raw;
    unsafe fn from_raw(raw: T::Raw, _: &mut u32) -> Self {
        let p = unsafe { T::ptr(raw) };
        debug_assert!(!p.is_null());
        unsafe { &*p }
    }
}

/// For the externally synchronized parameters, which nothing else touches during the call.
impl<T: Pointee> Param for &mut T {
    type Raw = T::Raw;
    unsafe fn from_raw(raw: T::Raw, _: &mut u32) -> Self {
        let p = unsafe { T::ptr(raw) };
        debug_assert!(!p.is_null());
        unsafe { &mut *p }
    }
}

impl<T: Pointee> Param for Option<&T> {
    type Raw = T::Raw;
    unsafe fn from_raw(raw: T::Raw, _: &mut u32) -> Self {
        unsafe { T::ptr(raw).as_ref() }
    }
}

impl<T: Pointee> Param for Option<&mut T> {
    type Raw = T::Raw;
    unsafe fn from_raw(raw: T::Raw, _: &mut u32) -> Self {
        unsafe { T::ptr(raw).as_mut() }
    }
}

impl<T> Param for &[T] {
    type Raw = *const T;
    unsafe fn from_raw(raw: *const T, len: &mut u32) -> Self {
        unsafe { slice(raw, *len) }
    }
}

impl<T> Param for &mut [T] {
    type Raw = *mut T;
    unsafe fn from_raw(raw: *mut T, len: &mut u32) -> Self {
        if *len == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(raw, *len as usize) } }
    }
}

impl<T> Param for *const T {
    type Raw = Self;
    unsafe fn from_raw(raw: Self, _: &mut u32) -> Self {
        raw
    }
}

impl<T> Param for *mut T {
    type Raw = Self;
    unsafe fn from_raw(raw: Self, _: &mut u32) -> Self {
        raw
    }
}

impl Param for u32 {
    type Raw = Self;
    unsafe fn from_raw(raw: Self, len: &mut u32) -> Self {
        *len = raw;
        raw
    }
}

macro_rules! by_value {
    ($($t:ty),*) => {
        $(impl Param for $t {
            type Raw = Self;
            unsafe fn from_raw(raw: Self, _: &mut u32) -> Self {
                raw
            }
        })*
    };
}
by_value!(u8, i32, u64, f32, usize);

/// Vulkan structs and plain values passed by pointer.
macro_rules! pointee {
    ($($t:ty),* $(,)?) => {
        $(impl $crate::entrypoint::Pointee for $t {
            type Raw = *mut $t;
            unsafe fn ptr(raw: *mut $t) -> *mut $t {
                raw
            }
        })*
    };
}
pub(crate) use pointee;
pointee!(u32, u64);

/// An entry point the dispatch table calls by name. `fn` converts the parameters at the
/// boundary (see `Param`) and runs a safe body; `unsafe fn` takes them raw, its body in an
/// unsafe block. A body of `Result<(), VkError>` returns `VK_SUCCESS` or the error.
macro_rules! entrypoint {
    (unsafe fn $name:ident($($p:ident: $t:ty),* $(,)?) -> Result<(), VkError> $body:block) => {
        /// # Safety
        /// The arguments meet the Vulkan valid usage of the command.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($p: $t),*) -> $crate::ffi::VkResult {
            #[allow(unused_unsafe)]
            let r: Result<(), $crate::entrypoint::VkError> = (|| unsafe { $body })();
            r.map_or_else(|e| e.0, |()| $crate::ffi::VK_SUCCESS)
        }
    };
    (unsafe fn $name:ident($($p:ident: $t:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        /// # Safety
        /// The arguments meet the Vulkan valid usage of the command.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($p: $t),*) $(-> $ret)? {
            #[allow(unused_unsafe)]
            unsafe { $body }
        }
    };
    (fn $name:ident($($p:ident: $t:ty),* $(,)?) -> Result<(), VkError> $body:block) => {
        /// # Safety
        /// The arguments meet the Vulkan valid usage of the command.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($p: <$t as $crate::entrypoint::Param>::Raw),*) -> $crate::ffi::VkResult {
            fn body($($p: $t),*) -> Result<(), $crate::entrypoint::VkError> $body
            let mut len = 0u32;
            $(let $p = unsafe { <$t as $crate::entrypoint::Param>::from_raw($p, &mut len) };)*
            body($($p),*).map_or_else(|e| e.0, |()| $crate::ffi::VK_SUCCESS)
        }
    };
    (fn $name:ident($($p:ident: $t:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        /// # Safety
        /// The arguments meet the Vulkan valid usage of the command.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($p: <$t as $crate::entrypoint::Param>::Raw),*) $(-> $ret)? {
            fn body($($p: $t),*) $(-> $ret)? $body
            let mut len = 0u32;
            $(let $p = unsafe { <$t as $crate::entrypoint::Param>::from_raw($p, &mut len) };)*
            body($($p),*)
        }
    };
}
pub(crate) use entrypoint;

/// The `n` elements at `p`; empty when `n` is 0, where Vulkan allows `p` to be anything.
///
/// # Safety
/// For `n` > 0, `p` points at `n` valid elements.
pub unsafe fn slice<'a, T>(p: *const T, n: u32) -> &'a [T] {
    if n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n as usize) } }
}

/// The output struct of type `stype` in a `pNext` chain.
///
/// # Safety
/// `p` is null or the head of a valid chain, and the struct of type `stype` is a `T`.
pub unsafe fn find_out<'a, T>(p: *mut std::ffi::c_void, stype: VkStructureType) -> Option<&'a mut T> {
    unsafe { chain(p).find(|&s| (*s).sType == stype).map(|s| &mut *s.cast::<T>()) }
}

/// The structs chained to an output struct by `pNext`.
///
/// # Safety
/// `p` is null or the head of a valid chain.
pub unsafe fn chain(p: *mut std::ffi::c_void) -> impl Iterator<Item = *mut VkBaseOutStructure> {
    let next = |s: *mut VkBaseOutStructure| (!s.is_null()).then_some(s);
    std::iter::successors(next(p.cast()), move |&s| next(unsafe { (*s).pNext }))
}
