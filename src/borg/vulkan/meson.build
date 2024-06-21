# Copyright © 2024 Andreas Wendleder
# SPDX-License-Identifier: MIT

borg_files = files(
  'borg_buffer.c',
  'borg_buffer.h',
  'borg_cmd_buffer.h',
  'borg_cmd_buffer.c',
  'borg_cmd_dispatch.c',
  'borg_cmd_pool.c',
  'borg_cmd_pool.h',
  'borg_descriptor_set.c',
  'borg_descriptor_set.h',
  'borg_descriptor_set_layout.c',
  'borg_descriptor_set_layout.h',
  'borg_device.c',
  'borg_device.h',
  'borg_device_memory.c',
  'borg_device_memory.h',
  'borg_format.c',
  'borg_image.c',
  'borg_instance.c',
  'borg_instance.h',
  'borg_physical_device.c',
  'borg_physical_device.h',
  'borg_pipeline.c',
  'borg_private.h',
  'borg_queue.c',
  'borg_queue.h',
  'borg_sampler.h',
  'borg_shader.c',
  'borg_shader.h'
)

borg_entrypoints = custom_target(
  'borg_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['borg_entrypoints.h', 'borg_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'borg',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

borg_deps = [
  dep_libdrm,
  dep_valgrind,
  idep_bak,
  idep_borg_ws,
  idep_nir_headers,
  idep_vulkan_runtime,
  idep_vulkan_util,
  idep_vulkan_wsi
]

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
  [
    borg_entrypoints,
    borg_files
  ],
  include_directories : [
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
  ],
  dependencies : borg_deps,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

