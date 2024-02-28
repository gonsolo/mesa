# Copyright © 2024 Andreas Wendleder
# SPDX-License-Identifier: MIT

borg_files = files(
  'borg_device.c'
)

borg_deps = [ ]

icd_lib_path = join_paths(get_option('prefix'), get_option('libdir'))
icd_file_name = 'libvulkan_borg.so'

borg_icd = custom_target(
  'borg_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'borg_icd.@0@.json'.format(host_machine.cpu()),
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.3', '--xml', '@INPUT1@',
    '--lib-path', join_paths(icd_lib_path, icd_file_name),
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

libvulkan_borg = shared_library(
  'vulkan_borg',
  [ borg_files ],
  include_directories : [ inc_include, inc_src, ],
  dependencies : borg_deps,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

