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

idep_borgc = declare_dependency(
  link_with : libborgc,
)
