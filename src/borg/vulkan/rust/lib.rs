// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

mod ffi {
    pub use borgvk_bindings_rs::*;
}

mod cmd_buffer;
#[cfg(borgc)]
mod compile;
mod compiler;
mod descriptor_set;
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
mod pipeline;
mod query;
mod sampler;
mod serial;
mod shader_cache;
mod sim;
mod sim_process;
mod sync;
mod transport;
mod util;
