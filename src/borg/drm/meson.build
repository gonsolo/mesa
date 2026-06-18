# Copyright © 2026 Borg GPU project
# SPDX-License-Identifier: MIT
#
# Builds libborg_drm_shim.so — a Mesa drm-shim that lets the borgvk Vulkan
# driver run without a real kernel driver.  Enabled by -Dtools=drm-shim.

libborg_drm_shim = shared_library(
  'borg_drm_shim',
  ['borg_shim.c', 'borg_serial.c'],
  include_directories : [inc_include, inc_src],
  dependencies : dep_drm_shim,
  gnu_symbol_visibility : 'hidden',
  install : true,
)
