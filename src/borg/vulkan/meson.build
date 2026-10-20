# Copyright © 2026 Andreas Wendleder
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
  'borgvk_private.h',
)

# The Borg's own driver code (software/borg), host build, with its hooks over the DRM render node.
borg_sw = '../../../../software/borg/'
borg_sw_inc = include_directories(
  '../../../../software/borg',
  '../../../../software/linux/overlay/include/uapi',
  '../../../../out/hardware/borg/rdl',
  '../../../include/drm-uapi',
)
libborg_host = static_library(
  'borg_host',
  files(borg_sw + 'borg_core.c', borg_sw + 'borg_hw.c', borg_sw + 'borg_spirb.c'),
  c_args : ['-DBORG_HOST', '-Wno-unused-parameter'],
  include_directories : borg_sw_inc,
)
idep_borg_host = declare_dependency(link_with : libborg_host, include_directories : borg_sw_inc)

if with_borgc
  borgvk_compiler_deps = [idep_borg_nir_passes, idep_nir]
else
  borgvk_compiler_deps = [declare_dependency(include_directories : [inc_compiler, include_directories('../compiler')])]
endif

borgvk_deps = [
  borgvk_compiler_deps,
  idep_borg_host,
  dep_dl,
  dep_libdrm,
  idep_vulkan_util,
  idep_vulkan_runtime,
  idep_vulkan_wsi,
  idep_mesautil,
]

borgvk_inc = [
  inc_include, inc_src, inc_util,
  include_directories('../../../../'),   # Borg repo root → software/borg/borg_layout.h
  include_directories('../compiler'),
]

prog_cbindgen = find_program('cbindgen', required : true, native : true, version : '>= 0.25')

borgvk_rust_h = custom_target(
  'borgvk_rust.h',
  input : [files('rust/cbindgen.toml'), 'rust/lib.rs'],
  output : ['borgvk_rust.h'],
  command : [
    prog_cbindgen, '-q', '--config', '@INPUT0@', '--lang', 'c',
    '--output', '@OUTPUT0@', '--depfile', '@DEPFILE@',
    '--', '@INPUT1@',
  ],
  depfile : 'borgvk_rust.h.d',
)
libborgvk_files += borgvk_rust_h

borgvk_bindgen_args = []
borgvk_bindgen_c_args = []
borgvk_bindgen_deps = [idep_nir_headers]
if with_borgc
  # NIR from Mesa's NIR crate, as NAK does; nir_builder, Borg's NIR passes and SPIR-V to NIR here.
  borgvk_bindgen_c_args += ['-DBORGVK_NIR']
  borgvk_bindgen_args += [
    compiler_rs_bindgen_blocklist,
    '--raw-line', 'pub use compiler::bindings::*;',
    '--allowlist-function', 'nir_builder_at|nir_(before|after)_impl|nir_build_call|nir_(load|store)_var',
    '--allowlist-function', 'nir_(channel|channels|trim_vector|imm_float|vec4|fdiv|fmul|fsub|fadd|fge|fabs|bcsel)',
    '--allowlist-function', 'nir_function_set_impl|nir_progress',
    '--allowlist-function', 'borg_lower_.*|vk_pipeline_shader_stage_to_nir|nir_builder_init_simple_shader|ralloc_free',
    '--allowlist-var', 'borg_(spirv|nir)_options|BORGVK_LINE_.*',
    '--opaque-type', 'spirv_to_nir_options.*',
    '--blocklist-type', 'spirv_to_nir_options__.*',
  ]
endif
_borgvk_bindings_rs = rust.bindgen(
  input : ['rust/bindings.h', borgvk_entrypoints[0], borgvk_rust_h],
  output : 'borgvk_bindings.rs',
  output_inline_wrapper : 'borgvk_inline_wrappers.c',
  c_args : [pre_args, borgvk_bindgen_c_args],
  args : [
    bindgen_output_args, borgvk_bindgen_args, '--no-prepend-enum-name', '--with-derive-default',
    '--allowlist-type', 'borgvk_device|borgvk_descriptor_set_layout|borgvk_desc_map|vk_pipeline_layout|borgvk_shader_blob|VkFormat|tex_conv|borgvk_image|borgvk_vp|borgvk_writeback|pipe_format',
    '--allowlist-type', 'borgvk_shader_stage|borgvk_pipeline|VkPipelineShaderStageCreateInfo|VkGraphicsPipelineCreateInfo|VkPipelineColorBlendStateCreateInfo|mesa_log_level',
    '--allowlist-type', 'vk_image_view|VkExtent2D|VkExtent3D|VkResult|borgvk_device_memory|pipe_swizzle|util_format_type|termios',
    '--allowlist-function', 'util_format_description|util_format_get_block(size|width|height|depth)|vk_format_to_pipe_format',
    '--allowlist-function', 'util_format_(un)?pack_(rgba|z_float|s_8uint)',
    '--allowlist-function', 'borg_core_.*|borg_hw_.*|bh_.*',
    '--allowlist-function', 'mesa_log|malloc|realloc|free|tcgetattr|tcsetattr|cfmakeraw|cfsetispeed|cfsetospeed|tcdrain',
    '--allowlist-function', 'memfd_create|close|signal|sysconf',
    '--allowlist-function', 'vk_object_zalloc|vk_object_free|__vk_errorf|vk_sampler_(create|destroy)',
    '--allowlist-function', 'vk_device_memory_(create|destroy)|vk_buffer_(view_)?(create|destroy)|drmIoctl|munmap',
    '--allowlist-type', 'vk_shader_module|VkShaderModuleCreateInfo|borgvk_buffer|borgvk_buffer_view|VkMemoryAllocateInfo|VkMemoryMapInfo|VkMemoryUnmapInfo|VkMappedMemoryRange',
    '--allowlist-type', 'VkBufferCreateInfo|VkBufferViewCreateInfo|VkDeviceBufferMemoryRequirements|VkMemoryRequirements2',
    '--allowlist-type', 'VkMemoryDedicatedRequirements|VkBindBufferMemoryInfo|VkBaseOutStructure|drm_borg_(gem_create|gem_mmap|shader|blend|state|push|setup|submit)|drm_gem_close',
    '--allowlist-var', 'DRM_COMMAND_BASE|DRM_BORG_(GEM_.*|SHADER|BLEND|STATE|PUSH|SETUP|SUBMIT)|VK_(TRUE|FALSE)',
    '--allowlist-function', 'vk_image_(view_)?(create|destroy)|vk_format_get_aspect_format|os_time_get_nano',
    '--allowlist-function', 'vk_descriptor_set_layout_zalloc|list_(inithead|addtail|del)',
    '--allowlist-function', 'vk_blend_(op|factor)_to_pipe|vk_get_pipeline_rendering_create_info|vk_format_(is_int|has_alpha)|util_format_is_srgb',
    '--allowlist-function', 'vk_(zalloc|free)|vk_command_buffer_(init_with_params|reset|finish|begin|end|get_record_result)',
    '--allowlist-function', 'util_format_(is_scaled|get_first_non_void_channel|is_pure_(integer|uint|sint))|vk_format_has_(depth|stencil)|util_format_has_alpha|util_format_is_compressed|util_format_unpack_rgba_rect|calloc',
    '--allowlist-type', 'VkClear(ColorValue|DepthStencilValue|Value|Attachment|Rect)|VkImageSubresourceRange|util_format_type',
    '--allowlist-var', 'VK_REMAINING_(MIP_LEVELS|ARRAY_LAYERS)',
    '--allowlist-function', 'wsi_device_(init|finish)|vk_instance_get_proc_addr_unchecked|vk_alloc2',
    '--allowlist-type', 'borgvk_physical_device|vk_instance|wsi_device_options|VkIcdSurfaceHeadless|VkSurfaceKHR|VkInstance',
    '--allowlist-function', 'vk_cmd_queue_execute|vk_sync_(wait|signal)',
    '--allowlist-type', 'vk_queue|vk_queue_submit|vk_cmd_queue_entry',
    '--allowlist-type', 'VkRenderPassBeginInfo|VkSubpassContents|VkSubpass(Begin|End)Info',
    '--allowlist-type', 'vk_buffer_view|VkComponentSwizzle|VkBufferCopy|VkCopyBufferInfo2|VkImageCopy|VkCopyImageInfo2|VkImageBlit|VkBlitImageInfo2|VkFilter',
    '--allowlist-var', 'BORGVK_TEXG_CHUNK|BORGVK_TEX_DIM|VK_WHOLE_SIZE',
    '--allowlist-function', 'util_format_(linear|is_depth_or_stencil|srgb_8unorm_to_linear_float)',
    '--allowlist-type', 'mesa_vk_dynamic_graphics_state|VkDraw(Indexed)?IndirectCommand',
    '--allowlist-type', 'VkFormatFeatureFlagBits|VkImageCreateFlagBits|VkFormatProperties[23]|VkPhysicalDeviceImageFormatInfo2|VkImageFormatProperties2|VkPhysicalDeviceSparseImageFormatInfo2|VkSparseImageFormatProperties2|VkPhysicalDeviceExternalBufferInfo|VkExternalBufferProperties',
    '--allowlist-var', 'VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT',
    '--allowlist-function', 'vk_physical_device_(init|finish|dispatch_table_from_entrypoints)|drm(Get|Free)Version|close',
    '--allowlist-type', 'borgvk_instance|vk_properties|vk_features|vk_device_extension_table|VkQueueFamilyProperties2|VkPhysicalDeviceMemoryProperties2|VkQueueFlagBits|VkMemoryHeapFlagBits|VkMemoryPropertyFlagBits|VkDriverId|VkPhysicalDeviceType',
    '--allowlist-var', '(borgvk|wsi)_(instance|physical_device|device)_entrypoints',
    '--allowlist-function', 'vk_(instance|device)_(init|finish|dispatch_table_from_entrypoints)|vk_instance_get_proc_addr|vk_enumerate_instance_extension_properties|vk_default_allocator|vk_(alloc|zalloc2|free2)|vk_queue_(init|finish)|vk_device_enable_threaded_submit',
    '--allowlist-type', 'vk_instance_extension_table|vk_device_entrypoint_table|VkPhysicalDevice(ProtectedMemory|ShaderDrawParameters|Vulkan11)Features|VkLayerProperties|VkExtensionProperties',
    '--allowlist-type', 'borgvk_target|VkRenderingInfo|VkRenderingAttachmentInfo|VkImageResolve|VkRenderingEndInfoEXT',
    '--allowlist-type', 'borgvk_command_buffer|vk_command_buffer_ops|vk_command_pool|VkCommandBufferBeginInfo|borg_blend|VkComputePipelineCreateInfo|VkPipelineRenderingCreateInfo|VkPipelineCache',
    '--allowlist-type', 'borgvk_descriptor_pool|borgvk_descriptor_set|VkDescriptorSetLayoutCreateInfo|VkDescriptorSetLayoutSupport|VkDescriptorPoolCreateInfo',
    '--allowlist-type', 'VkDescriptorSetAllocateInfo|VkWriteDescriptorSet|VkCopyDescriptorSet|VkDescriptorPoolResetFlags',
    '--allowlist-type', 'vk_sync_type|vk_sync_wait|vk_sync_features|vk_sync_wait_flags|VkQueryPoolCreateInfo|VkQueryPool|VkImageMemoryRequirementsInfo2|VkImageSubresource2KHR|VkSubresourceLayout2KHR|VkBindImageMemoryInfo|VkImageViewCreateInfo',
    '--allowlist-type', 'borgvk_sampler|VkSamplerCreateInfo|VkImageCreateInfo|vk_object_base|VkEventCreateInfo|VkDependencyInfo|VkCommandBuffer|VkEvent|VkPipelineStageFlags2?',
    '--allowlist-var', 'BORGC_.*|BORGVK_.*|BC_.*|BORG_(ZB|SB|HEAP)_SPI|DRAM_OUT_BASE_SPI',
    '--allowlist-var', 'B115200|TCSANOW|CLOCAL|CREAD|CRTSCTS|O_NOCTTY',
    '--allowlist-var', 'PROT_(READ|WRITE)|MAP_(PRIVATE|ANONYMOUS|NORESERVE|SHARED)|SIGPIPE|_SC_NPROCESSORS_ONLN',
  ],
  include_directories : [borgvk_inc, include_directories('.')],
  dependencies : [
    borgvk_bindgen_deps,
    borgvk_deps,
    idep_vulkan_runtime_headers,
    idep_vulkan_util_headers,
    idep_vulkan_wsi_headers,
  ],
)
_libborgvk_bindings_rs_extern = static_library(
  'borgvk_bindings_rs_extern',
  _borgvk_bindings_rs[1],
  c_args : [no_override_init_args, '-Wno-missing-prototypes', borgvk_bindgen_c_args],
  include_directories : [borgvk_inc, include_directories('.')],
  dependencies : [
    borgvk_bindgen_deps,
    borgvk_deps,
    idep_vulkan_runtime_headers,
    idep_vulkan_util_headers,
    idep_vulkan_wsi_headers,
  ],
)
_libborgvk_bindings_rs = static_library(
  'borgvk_bindings_rs',
  _borgvk_bindings_rs[0],
  link_with : [_libborgvk_bindings_rs_extern],
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : rust_global_args,
  dependencies : with_borgc ? [idep_compiler_rs] : [],
)
borgvk_rs_args = rust_global_args
if with_platform_x11
  borgvk_rs_args += ['--cfg', 'vk_x11']
endif
if with_platform_wayland
  borgvk_rs_args += ['--cfg', 'vk_wayland']
endif
borgvk_rs_link = [_libborgvk_bindings_rs, libborg_layout]
borgvk_rs_deps = []
if with_borgc
  borgvk_rs_args += ['--cfg', 'borgc']
  borgvk_rs_link += [libborgc_rs]
  borgvk_rs_deps += [idep_compiler_rs]
endif
libborgvk_rs = static_library(
  'borgvk_rs',
  files('rust/lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'c',
  rust_args : borgvk_rs_args,
  link_with : borgvk_rs_link,
  dependencies : borgvk_rs_deps,
)

if with_tests and get_option('b_sanitize') == 'none'
  rust.test(
    'borgvk_rs',
    libborgvk_rs,
    suite : ['borg', 'rs'],
    rust_args : ['-C', 'default-linker-libraries'],
    dependencies : [idep_vulkan_util, idep_mesautil],
  )
endif

libvulkan_borg = shared_library(
  'vulkan_borg',
  [libborgvk_files, borgvk_entrypoints],
  include_directories : borgvk_inc,
  link_whole : [libborgvk_rs],   # the dispatch table's references are weak and pull nothing in
  dependencies : borgvk_deps,
  link_args : [vulkan_icd_link_args, ld_args_build_id, ld_args_bsymbolic, ld_args_gc_sections],
  link_depends : vulkan_icd_link_depends,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

test(
  'borgvk rust entrypoints',
  find_program('check_rust_entrypoints.sh'),
  args : [libvulkan_borg, meson.current_source_dir() / 'rust'],
  suite : ['borg'],
)

# Generates a replayable wire burst for the simulator's push-constant
# end-to-end test, with the driver's own senders rather than a reimplementation.
borgvk_burst_gen = executable(
  'borgvk_burst_gen',
  ['borgvk_burst_gen.c', borgvk_rust_h],
  include_directories : [
    inc_include, inc_src, inc_util,
    include_directories('../../../../'),   # Borg repo root → software/borg/borg_layout.h
  ],
  link_with : [libborgvk_rs],
  dependencies : borgvk_deps,
  link_args : ld_args_gc_sections,
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

# A static borgvk (no ICD loader, no dlopen) for the board; see software/vkcube.
borgvk_static = executable(
  'borgvk_static',
  [get_option('borg-app-objects').length() > 0 ? [] : 'borgvk_static_main.c', libborgvk_files, borgvk_entrypoints],
  objects : get_option('borg-app-objects'),
  include_directories : [
    inc_include, inc_src, inc_util,
    include_directories('../../../../'),
  ],
  link_whole : [libborgvk_rs],
  dependencies : borgvk_deps,
  link_args : ['-static', '-Wl,--gc-sections'],
  build_by_default : false,
)
