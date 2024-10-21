# Copyright © 2024 Andreas Wendleder
# SPDX-License-Identifier: MIT

add_languages('rust', required: true)
rust = import('rust')

libbak_deps = [
  idep_mesautil,
  idep_nir_headers
]

_bak_bindings_rs = rust.bindgen(
  input : ['bak_bindings.h'],
  output : 'bak_bindings.rs',
  c_args : [
    pre_args,
  ],
  args : [
    compiler_rs_bindgen_blocklist,
    '--raw-line', '#![allow(non_camel_case_types)]',
    '--raw-line', '#![allow(non_snake_case)]',
    '--raw-line', '#![allow(non_upper_case_globals)]',
    '--raw-line', 'use compiler::bindings::*;',
    '--allowlist-function', 'bak_.*',
    '--no-prepend-enum-name',
  ],
  dependencies : libbak_deps,
)

_libbak_bindings_rs = static_library(
  'bak_bindings',
  _bak_bindings_rs,
  gnu_symbol_visibility : 'hidden',
  dependencies : [
    idep_compiler_rs,
  ],
  rust_abi : 'rust',
)


bak_rust_args = [
  '-Aclippy::identity_op',
  '-Aclippy::len_zero',
  '-Aclippy::manual_range_contains',
  # normally this is a good one, but we use it where the "better" code is worse
  '-Aclippy::needless_range_loop',
  '-Aclippy::redundant_field_names',
  '-Aclippy::upper_case_acronyms',
  '-Aclippy::vec_box',
  '-Aclippy::write_with_newline',
  '-Anon_snake_case',
  '-Anon_camel_case_types',
]

_libbak_rs = static_library(
  'bak_rs',
  files('bak/lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'c',
  rust_args : bak_rust_args,
  dependencies : [
    idep_compiler_rs
  ],
  link_with: [
    _libbak_bindings_rs
  ],
)

libbak_c_files = files(
  'bak.h',
  'bak_nir.c',
  'bak_print.c',
)

_libbak = static_library(
  'bak',
  [libbak_c_files],
  include_directories : [inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium],
  dependencies : libbak_deps,
  link_with : [_libbak_rs],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
)

idep_bak = declare_dependency(
  include_directories : include_directories('.'),
  link_with : _libbak,
)   
