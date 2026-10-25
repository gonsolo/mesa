# Copyright © 2026 Andreas Wendleder
# SPDX-License-Identifier: MIT

libborg_layout = static_library(
  'borg_layout',
  structured_sources([
    files('lib.rs', 'descriptor.rs', 'image.rs', 'tex_format.rs', 'tile.rs'),
    borg_tex_format_rs,
  ]),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : rust_global_args,
)

if with_tests and get_option('b_sanitize') == 'none'
  rust.test('borg_layout', libborg_layout, suite : ['borg', 'rs'])
endif
