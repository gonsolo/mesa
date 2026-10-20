// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

// Linked for its C entry points until the compiler glue calls it directly.
#[cfg(borgc)]
extern crate borgc_rs;

mod ffi {
    pub use borgvk_bindings_rs::*;
}

mod compiler;
mod format;
mod hw;
#[cfg(borgc)]
mod nir_info;
#[cfg(borgc)]
mod nir_passes;
mod image;
mod serial;
mod sim;
mod sim_process;
mod transport;
mod util;
