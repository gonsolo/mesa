// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

mod ffi {
    pub use crate::types::*;
    pub use borgvk_bindings_rs::*;
}

mod clear;
mod cmd_buffer;
#[cfg(borgc)]
mod compile;
mod compiler;
mod copy;
mod corpus;
mod cube;
mod descriptor_set;
mod device;
mod dispatch;
mod draw;
mod drm;
mod entrypoint;
mod event;
mod format;
mod hw;
#[cfg(borgc)]
mod nir_info;
#[cfg(borgc)]
mod nir_passes;
mod object;
mod image;
mod memory;
mod physical_device;
mod pipeline;
mod query;
mod queue;
mod render;
mod sampler;
mod serial;
mod shader_cache;
mod state_upload;
mod sim;
mod sim_process;
mod sync;
mod texel;
mod texture_upload;
mod transport;
mod types;
mod util;
mod wsi;
