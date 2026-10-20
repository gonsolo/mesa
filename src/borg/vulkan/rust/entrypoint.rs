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
    };
}
pub(crate) use handle;

handle!(VkDevice, borgvk_device, VK_OBJECT_TYPE_DEVICE);

/// An entry point the dispatch table calls by name, its body in an unsafe block. A body of
/// `Result<(), VkError>` returns `VK_SUCCESS` or the error, so `?` works.
macro_rules! entrypoint {
    (fn $name:ident($($p:ident: $t:ty),* $(,)?) -> Result<(), VkError> $body:block) => {
        /// # Safety
        /// The arguments meet the Vulkan valid usage of the command.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($p: $t),*) -> $crate::ffi::VkResult {
            #[allow(unused_unsafe)]
            let r: Result<(), $crate::entrypoint::VkError> = (|| unsafe { $body })();
            r.map_or_else(|e| e.0, |()| $crate::ffi::VK_SUCCESS)
        }
    };
    (fn $name:ident($($p:ident: $t:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        /// # Safety
        /// The arguments meet the Vulkan valid usage of the command.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($p: $t),*) $(-> $ret)? {
            #[allow(unused_unsafe)]
            unsafe { $body }
        }
    };
}
pub(crate) use entrypoint;
