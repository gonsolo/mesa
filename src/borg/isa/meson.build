# Copyright © 2026 Andreas Wendleder
# SPDX-License-Identifier: MIT
#
# A copy of Borg's hardware/borg/isa (Borg checks the two are identical); the tables are
# generated from it at build time.

borg_opcodes_rs = custom_target(
  'borg_opcodes_rs',
  input : files('isa_gen.py', 'fields.csv', 'opcodes', 'rv_i', 'rv_zicsr', 'csrs.csv', '../compiler/ir_ops'),
  output : 'opcodes.rs',
  command : [prog_python, '@INPUT0@', meson.current_source_dir(), '--rust', '@OUTPUT@', '--ir', '@INPUT6@'],
)
borg_tex_format_rs = custom_target(
  'borg_tex_format_rs',
  input : files('tex_format_gen.py', 'tex_formats.txt'),
  output : 'tex_format_table.rs',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '--rust', '@OUTPUT@'],
)
