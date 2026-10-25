# Copyright © 2026 Andreas Wendleder
# SPDX-License-Identifier: MIT

subdir('isa')
if with_borgc
  subdir('compiler')
endif
subdir('layout')
subdir('vulkan')
if with_tools.contains('drm-shim')
  subdir('drm')
endif
