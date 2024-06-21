libborg_ws = static_library(
  'borg_ws',
  [
    'borg_bo.h',
    'borg_bo.c',
    'borg_ws_device.h',
    'borg_ws_device.c',
  ],
  include_directories : [
    inc_include,
    inc_src,
  ],
  c_args: [ cc.get_supported_arguments('-Wno-gnu-variable-sized-type-not-at-end') ],
  dependencies : [
    dep_libdrm,
  ],
  gnu_symbol_visibility : 'hidden',
)

idep_borg_ws = declare_dependency(
  link_with: libborg_ws,
  include_directories : include_directories('.')
)

