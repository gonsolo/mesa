// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

mod ffi {
    pub use borgvk_bindings_rs::*;
}

#[cfg(borgc)]
mod compile;
mod compiler;
mod entrypoint;
mod event;
mod format;
mod hw;
#[cfg(borgc)]
mod nir_info;
#[cfg(borgc)]
mod nir_passes;
mod image;
mod memory;
mod query;
mod sampler;
mod serial;
mod sim;
mod sim_process;
mod transport;
mod util;
