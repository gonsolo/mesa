# Copyright © 2026 Borg GPU project
# SPDX-License-Identifier: MIT
#
# borgc: the Borg shader compiler crate (Rust), exposing a C ABI so the borgvk
# Vulkan driver can call it. Modeled on Mesa's NAK (src/nouveau/compiler).

libborgc = static_library(
  'borgc',
  files('lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'c',
  rust_args : rust_global_args,
  dependencies : [
    idep_compiler_rs,   # Mesa's NIR Rust bindings (compiler::bindings / ::nir)
  ],
)

# The SPIR-V options and NIR pass list borgc expects, shared by the driver and
# the offline CLI so the two cannot drift (see borg_nir_passes.h).
libborg_nir_passes = static_library(
  'borg_nir_passes',
  files('borg_nir_passes.c'),
  include_directories : [inc_include, inc_src, inc_compiler],
  dependencies : [idep_nir, idep_mesautil],
  gnu_symbol_visibility : 'hidden',
)

idep_borg_nir_passes = declare_dependency(
  link_with : libborg_nir_passes,
  include_directories : include_directories('.'),
)

idep_borgc = declare_dependency(
  link_with : libborgc,
  dependencies : [idep_borg_nir_passes],
)

# Offline compiler. Lets borgc be exercised (and shaders baked into the
# firmware) without standing up a Vulkan device -- see borgc_cli.c's header for
# the two verification gaps that closes.
borgc_cli = executable(
  'borgc-cli',
  files('borgc_cli.c'),
  include_directories : [inc_include, inc_src, inc_compiler],
  dependencies : [idep_borgc, idep_vtn, idep_nir, idep_mesautil],
  link_with : [libborgc],
  install : false,
)

# Unit tests for the instruction encoders. These matter more than most: the
# encodings are a hand-maintained mirror of the hardware's Instructions.scala,
# and a divergence is SILENT -- the shader would execute a different opcode
# rather than fail to load. Same sanitizer caveat as src/compiler/rust.
if with_tests and get_option('b_sanitize') == 'none'
  rust.test(
    'borgc_test',
    libborgc,
    suite : ['borg', 'rs'],
  )
endif
