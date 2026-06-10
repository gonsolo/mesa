# Copyright © 2026 Borg GPU project
# SPDX-License-Identifier: MIT

borgvk_entrypoints = custom_target(
  'borgvk_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['borgvk_entrypoints.h', 'borgvk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'borgvk',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

libborgvk_files = files(
  'borgvk_cmd_buffer.c',
  'borgvk_compiler.c',
  'borgvk_descriptor_set.c',
  'borgvk_device.c',
  'borgvk_memory.c',
  'borgvk_pipeline.c',
  'borgvk_query.c',
  'borgvk_queue.c',
  'borgvk_serial.c',
  'borgvk_sync.c',
  'borgvk_wsi.c',
  'borgvk_private.h',
)

borgvk_deps = [
  dep_dl,
  idep_borgc,
  idep_nir,
  idep_vulkan_util,
  idep_vulkan_runtime,
  idep_vulkan_wsi,
  idep_mesautil,
]

libvulkan_borg = shared_library(
  'vulkan_borg',
  [libborgvk_files, borgvk_entrypoints],
  include_directories : [
    inc_include, inc_src, inc_util,
  ],
  dependencies : borgvk_deps,
  link_args : [vulkan_icd_link_args, ld_args_build_id, ld_args_bsymbolic, ld_args_gc_sections],
  link_depends : vulkan_icd_link_depends,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

if with_symbols_check
  test(
    'borgvk symbols check',
    symbols_check,
    args : [
      '--lib', libvulkan_borg,
      '--symbols-file', vulkan_icd_symbols,
      symbols_check_args,
    ],
    suite : ['borg'],
  )
endif

icd_file_name = libname_prefix + 'vulkan_borg.' + libname_suffix

borg_icd = custom_target(
  'borg_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'borg_icd.' + vulkan_manifest_suffix,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.3', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

_dev_icdname = 'borg_devenv_icd.@0@.json'.format(host_machine.cpu())
_dev_icd = custom_target(
  'borg_devenv_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : _dev_icdname,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.3', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)

devenv.append('VK_DRIVER_FILES', _dev_icd.full_path())
