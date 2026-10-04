/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * borgvk — a native Mesa Vulkan ICD for the Borg GPU (ULX3S FPGA), driven over
 * serial. Modeled on the v3dv (Broadcom) driver. See
 * ~/.claude/plans/atomic-questing-stream.md for the overall design.
 */
#ifndef BORGVK_PRIVATE_H
#define BORGVK_PRIVATE_H

#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_device_memory.h"
#include "vk_buffer.h"
#include "vk_buffer_view.h"
#include "vk_image.h"
#include "vk_sampler.h"
#include "vk_command_buffer.h"
#include "vk_descriptor_set_layout.h"

#include "wsi_common.h"

#include "borgvk_entrypoints.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WSI (window-system integration) is available when at least one surface
 * platform is compiled in. borgvk uses the software (CPU) WSI path: the host
 * window is irrelevant — the real output is the FPGA's HDMI — so WSI only needs
 * to function so an app's render loop runs and submits each frame. */
#if defined(VK_USE_PLATFORM_XCB_KHR) || \
    defined(VK_USE_PLATFORM_XLIB_KHR) || \
    defined(VK_USE_PLATFORM_WAYLAND_KHR)
#define BORGVK_USE_WSI_PLATFORM
#endif

struct borgvk_instance {
   struct vk_instance vk;
};

struct borgvk_physical_device {
   struct vk_physical_device vk;

   /* Memory layout reported to the app. Borg renders on the FPGA; from the
    * host's point of view all "device" memory is plain host-visible RAM that we
    * stage and ship over serial, so we expose a single host-visible coherent
    * heap. */
   VkPhysicalDeviceMemoryProperties memory;

   /* Submit is synchronous (ship MVP over serial, return), so a single
    * immediate-success sync type (borgvk_sync_type) backs fences/semaphores. */
   const struct vk_sync_type *sync_types[2];

   /* DRM device fd opened during enumeration.  -1 when no borg DRM device is
    * found (drm-shim not loaded or no kernel driver).  In that case the driver
    * falls back to direct malloc + serial (legacy path). */
   int drm_fd;

#ifdef BORGVK_USE_WSI_PLATFORM
   struct wsi_device wsi_device;
#endif
};

#ifdef BORGVK_USE_WSI_PLATFORM
VkResult borgvk_wsi_init(struct borgvk_physical_device *physical_device);
void borgvk_wsi_finish(struct borgvk_physical_device *physical_device);
#endif

extern const struct vk_sync_type borgvk_sync_type;

/* Compiled Borg-ISA shader blobs (borgc output). Captured per stage at pipeline
 * creation and shipped to the firmware over serial (0xB0) at submit-setup, where
 * they replace the firmware's baked borgc_vert/frag_shader[] arrays. Only the
 * vertex and fragment stages come from the app; the rasterize stage stays baked
 * (it is a fixed-function Borg shader, not derived from the app). */
#define BORGVK_SHADER_BLOB_MAX 512   /* frag = 255 B today; headroom */
enum borgvk_shader_stage {
   BORGVK_STAGE_VERT = 0,
   BORGVK_STAGE_FRAG = 1,
   BORGVK_SHADER_STAGE_COUNT,
};
struct borgvk_shader_blob {
   uint8_t  data[BORGVK_SHADER_BLOB_MAX];
   uint32_t len;   /* 0 = not compiled (or didn't fit) */
};

struct borgvk_device {
   struct vk_device vk;

   /* Dispatch table used when replaying recorded command queues (secondary
    * cmd buffers). We never replay for rendering, but the runtime wants it set. */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Single graphics/compute/transfer queue. cube.c and vulkaninfo both use one
    * queue from family 0; multi-queue support can come later. */
   struct vk_queue queue;

   /* Non-owning copy of physical_device->drm_fd.  -1 → malloc+serial fallback. */
   int drm_fd;

   /* borgc-compiled shader blobs captured at pipeline creation, uploaded to the
    * firmware over serial (0xB0) on the first submit. */
   struct borgvk_shader_blob shader_blob[BORGVK_SHADER_STAGE_COUNT];
   /* Colour-blend state of the last graphics pipeline created, already packed
    * in BLEND_CFG / BLEND_CONST register layout (see borgvk_blend_pack()). */
   uint32_t blend_cfg;
   uint32_t blend_const;
   /* Stencil / depth / cull state of the last graphics pipeline created, in
    * register layout (see borgvk_state_pack()); state_valid once set. */
   uint32_t state_reg[5];
   bool     state_valid;
};

/* A render pass/framebuffer or dynamic-rendering scope currently open on this
 * command buffer, captured in borgvk_CmdBeginRendering(). Mesa's own
 * vk_common_CmdBeginRenderPass2 (borgvk never overrides RenderPass or
 * Framebuffer creation, nor the legacy Begin/EndRenderPass entrypoints --
 * Mesa's runtime emulates all of that generically) always resolves down to a
 * VkRenderingInfo and calls this driver's CmdBeginRendering, whether the app
 * used the legacy API or called it directly, so this is the one place that
 * needs to remember "what image backs each attachment right now" --
 * CmdClearAttachments has no other way to resolve its attachment indices
 * back to real images. */
#define BORGVK_MAX_COLOR_ATTACHMENTS 8

#define BORGVK_MAX_VERTEX_BINDINGS 4

#define BORGVK_MAX_BINDINGS 8

struct borgvk_command_buffer {
   struct vk_command_buffer vk;

   bool in_rendering;
   VkRect2D render_area;
   uint32_t layer_count;
   uint32_t color_attachment_count;
   struct vk_image_view *color_views[BORGVK_MAX_COLOR_ATTACHMENTS];
   struct vk_image_view *depth_view;
   struct vk_image_view *stencil_view;
   /* Multisample resolve targets captured from VkRenderingAttachmentInfo::
    * resolveImageView, NULL when that attachment isn't multisampled or
    * requested no resolve. Performed at CmdEndRendering2EXT -- see its
    * comment for why the legacy render-pass path (which is how CTS's
    * clear_color_attachment MSAA tests actually exercise this, via
    * VkSubpassDescription::pResolveAttachments) needs this at all. */
   struct vk_image_view *color_resolve_views[BORGVK_MAX_COLOR_ATTACHMENTS];
   /* The first colour attachment's load-op clear colour, for the simulator's render target. */
   bool has_clear;
   float clear_color[4];
   /* Draw state tracked as the recorded commands replay at submit (borgvk_queue.c). */
   struct borgvk_pipeline *gfx_pipeline;
   const uint8_t *vb[BORGVK_MAX_VERTEX_BINDINGS];
   VkDeviceSize vb_avail[BORGVK_MAX_VERTEX_BINDINGS];   /* bytes from vb[] to the buffer's end */
   struct borgvk_descriptor_set *desc_set;
   bool generic_drawn;
   const struct borgvk_pipeline *cs_pipeline;   /* bound compute pipeline */
   bool dispatched;                             /* ran a compute dispatch at replay */
   /* Draws of the open render pass, as one wire stream the simulator runs at its end. */
   uint8_t *stream;
   size_t stream_len, stream_cap;
   uint32_t heap_top, batch_draws;
   /* Shader and blend packets last sent in this batch; a draw repeats them only when they change. */
   uint8_t *state_sent;
   size_t state_sent_len;
   struct borgvk_image *batch_img;
   /* The batch runs on the persistent simulator, with these depth and stencil attachments. */
   bool batch_serve;
   struct vk_image_view *batch_depth, *batch_stencil;
   const uint8_t *index_ptr;                    /* bound index buffer, at its offset */
   VkDeviceSize index_avail;
   uint32_t index_size;                         /* 2, 4, or 0 for an unsupported type */
   uint32_t pc[32];                             /* push-constant bytes, for compute */
   VkDeviceSize dyn_off[BORGVK_MAX_BINDINGS];   /* dynamic offsets of desc_set, by binding */
};

extern const struct vk_command_buffer_ops borgvk_cmd_buffer_ops;
struct borgvk_command_buffer;
void borgvk_flush_draws(struct borgvk_command_buffer *cmd);

/* The persistent simulator (borgvk_sim.c). */
struct borgvk_image;
bool borgvk_sim_serves(const struct borgvk_image *color);
uint8_t borgvk_sim_flush_format(VkFormat f);
bool borgvk_sim_bytes_packed(VkFormat f);
bool borgvk_sim_depth_is_d32(VkFormat f);
VkResult borgvk_sim_run_pass(const uint8_t *stream, size_t n, struct borgvk_image *color,
                             const struct vk_image_view *depth_view,
                             const struct vk_image_view *stencil_view);

/* ---- Serial transport (borgvk_serial.c) ------------------------------- *
 * The "GPU" is the ULX3S FPGA reached over the FT231X serial bridge
 * (/dev/ttyUSB0 @115200, overridable via $BORGVK_SERIAL). The submit path
 * ships the per-frame 4×4 MVP as a framed packet the firmware decodes. */
void borgvk_serial_send_mvp(const float mvp[16]);
/* 0xB3: BLEND_CFG and BLEND_CONST register values, applied by the firmware
 * before the next draw. */
void borgvk_serial_send_blend(uint32_t blend_cfg, uint32_t blend_const);
/* Pack a pipeline's colour-blend state into the BLEND_CFG / BLEND_CONST layout
 * (hardware/rdl/borg.rdl); the Vulkan enum values pass through unchanged. */
void borgvk_blend_pack(const VkPipelineColorBlendStateCreateInfo *cb,
                       uint32_t *cfg, uint32_t *konst);
/* 0xB4: STENCIL_CFG, STENCIL_FRONT, STENCIL_BACK, DEPTH_CFG, CULL_CFG. */
void borgvk_serial_send_state(const uint32_t reg[5]);
void borgvk_serial_send_target(uint8_t flush_format, const float clear[4]);
void borgvk_serial_send_pass(uint8_t flush_format, uint8_t flags);
void borgvk_serial_send_attr4(const float *attr, int n);
/* 0xB8 / 0xB9 / 0xBA (simulator): heap write, vertex-attribute fetch descriptor, draw. */
void borgvk_serial_send_mem(uint32_t off, const uint8_t *data, uint32_t n);
void borgvk_serial_send_vattr(uint32_t slot, uint32_t fmt, uint32_t base, uint32_t count,
                              uint32_t stride, uint32_t swz);
void borgvk_serial_send_draw(uint32_t topology, uint32_t index_type, uint32_t restart,
                             uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                             uint32_t first_instance, int32_t vertex_offset, uint32_t index_base,
                             uint32_t ubo_base);
/* 0xB5: one chunk (<= 256 B at byte offset `off`) of a texture's texels plus
 * the texture descriptor words 1..3 and the four sampler words
 * (docs/B2_texture_unit.md); descriptor and sampler ride with every chunk. */
#define BORGVK_TEXG_CHUNK 256
void borgvk_serial_send_texture_chunk(uint32_t off, const uint8_t *data, uint32_t n,
                                      const uint32_t desc_w123[3],
                                      const uint32_t sampler[4]);
/* Pack a pipeline's stencil, depth and cull state into register layout. */
void borgvk_state_pack(const VkGraphicsPipelineCreateInfo *ci, uint32_t reg[5]);

/* Max mesh the firmware's fixed-length 0xAE geometry packet carries (must match
 * the firmware RX_GEOM_*). cube.c is 8 unique verts / 12 triangles. */
#define BORGVK_GEOM_MAX_VERTS 16
#define BORGVK_GEOM_MAX_TRIS  12

/* Ship the app's real mesh once (Phase B): nverts unique model-space positions
 * (xyz floats), ntris triangles each indexing 3 of them (idx, 3 per tri), with
 * per-triangle-vertex UVs (uv, 2 floats per tri-vertex). */
void borgvk_serial_send_geom(const float *verts, int nverts,
                             const uint8_t *idx, const float *uv, int ntris);

/* The Borg texture dimension (must match firmware TEX_WIDTH). The app's texture
 * is downsampled to this on the host (lossless at the 128x128 render size). */
#define BORGVK_TEX_DIM 64

/* Ship one texture row (Phase B): `rgba` is BORGVK_TEX_DIM RGBA8 texels, framed
 * as a 0xAF packet together with the four-word sampler descriptor `sampler`
 * (struct borgvk_sampler::desc), which the firmware installs as sampler 0. */
void borgvk_serial_send_tex_row(int y, const uint8_t *rgba, const uint32_t sampler[4]);

/* Ship a borgc-compiled Borg-ISA shader blob to the firmware (0xB0 packet):
 * `stage` selects the firmware shader slot (0 = vertex, 1 = fragment), `blob` is
 * the .borg bytes (spirb_parse format). The firmware stages it to PSRAM in place
 * of its baked borgc_vert/frag_shader[] array before the next render. */
void borgvk_serial_send_shader(uint8_t stage, const uint8_t *blob, uint32_t len);

/* Ship a push-constant range to the firmware (0xB2 packet).  `offset` and
 * `size` are BYTES, as Vulkan gives them, and both are required by the spec to
 * be multiples of 4; the packet carries words because that is what the shader
 * side addresses (LS_BASE + rs1<<2, with borgc pinning rs1 to the field's word
 * index).  The firmware stages the bytes and points LS_BASE at them, so a
 * borgc-compiled LOAD lands on the right word. */
void borgvk_serial_send_push_constants(uint32_t offset, uint32_t size,
                                       const void *values);

/* ---- Transport capture (serial ↔ sim unification) --------------------- *
 * By default every send_* packet above is written to the serial port (paced).
 * Wrapping a sequence between capture_begin/capture_end redirects the identical
 * framed bytes into a growable in-memory buffer instead — exactly the byte
 * stream borgvk would have put on the wire.  The BORGVK_SIM path captures one
 * frame's packets this way and feeds them to `arcilator_sim --cts-uart`, so the
 * simulator exercises the same protocol (incl. 0xB0 borgc shaders) as the FPGA,
 * with no serial port involved.  Not re-entrant: one capture at a time. */
void borgvk_transport_capture_begin(void);
/* End capture: returns the malloc'd captured bytes (caller frees) and sets
 * *out_len; restores serial mode.  Returns NULL with *out_len=0 if empty. */
uint8_t *borgvk_transport_capture_end(size_t *out_len);

/* Queue submit hook (borgvk_queue.c): reads the MVP from the submitted command
 * buffer's bound descriptor set and ships it over serial. Wired into
 * device->queue.driver_submit by CreateDevice. */
VkResult borgvk_queue_submit(struct vk_queue *queue,
                             struct vk_queue_submit *submit);

/* Compiler shim (borgvk_compiler.c → Rust borgc crate). */
void borgvk_compiler_selftest(void);
void borgvk_compile_compute_stage(struct borgvk_device *device,
                                  const VkPipelineShaderStageCreateInfo *stage_info,
                                  struct borgvk_pipeline *pipeline);
void borgvk_compile_stage(struct borgvk_device *device, uint32_t vfetch,
                          const VkPipelineShaderStageCreateInfo *stage_info);

VK_DEFINE_HANDLE_CASTS(borgvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(borgvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(borgvk_device, vk.base, VkDevice,
                       VK_OBJECT_TYPE_DEVICE)

/* Device memory: backed by a GEM BO (drm_fd >= 0) or plain malloc.
 * In both cases `map` is a CPU-writable pointer to the allocation. */
struct borgvk_device_memory {
   struct vk_device_memory vk;
   void    *map;         /* mmap(drm_fd, gem_offset) or malloc'd              */
   size_t   size;        /* allocation size for munmap                         */
   uint32_t gem_handle;  /* 0 = malloc path (no DRM)                          */
};

struct borgvk_buffer {
   struct vk_buffer vk;
   struct borgvk_device_memory *mem;
   VkDeviceSize offset;
};

struct borgvk_image {
   struct vk_image vk;
   VkDeviceSize size;     /* bytes needed to back this image */
   struct borgvk_device_memory *mem;
   VkDeviceSize offset;
};

/* Texel buffer view: format + range over a VkBuffer, used by
 * VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER / STORAGE_TEXEL_BUFFER. No
 * driver-specific state needed yet -- vk_buffer_view already tracks
 * buffer/format/offset/range; the shader-visible side is future work
 * (texel buffers aren't wired into the texture unit path yet). This
 * object existing and being destroyable correctly is what CTS's basic
 * lifetime/API tests need. */
struct borgvk_buffer_view {
   struct vk_buffer_view vk;
};

/* A sampler, packed at creation into the texture unit's four-word sampler
 * descriptor (docs/B2_texture_unit.md in the Borg repository). The firmware
 * stores the words in its sampler table as they are. */
struct borgvk_sampler {
   struct vk_sampler vk;
   uint32_t desc[4];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_buffer, vk.base, VkBuffer,
                               VK_OBJECT_TYPE_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_image, vk.base, VkImage,
                               VK_OBJECT_TYPE_IMAGE)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)


struct borgvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count;
   uint64_t dyn_mask;           /* bindings that are *_BUFFER_DYNAMIC */
};

struct borgvk_descriptor_pool {
   struct vk_object_base base;
};

/* A descriptor set remembers which buffer/image/sampler is bound at each
 * binding, so the submit path can find the cube's uniform buffer (binding 0 →
 * MVP + geometry) and its texture and sampler (binding 1 → combined image
 * sampler). */
struct borgvk_descriptor_set {
   struct vk_object_base base;
   struct borgvk_buffer *buffers[BORGVK_MAX_BINDINGS];
   VkDeviceSize offsets[BORGVK_MAX_BINDINGS];
   VkDeviceSize ranges[BORGVK_MAX_BINDINGS];
   struct borgvk_image *images[BORGVK_MAX_BINDINGS];
   /* The view behind images[]: view type, format, mip/layer range, swizzle. */
   struct vk_image_view *views[BORGVK_MAX_BINDINGS];
   struct borgvk_sampler *samplers[BORGVK_MAX_BINDINGS];
   struct vk_buffer_view *buffer_views[BORGVK_MAX_BINDINGS];   /* texel buffers */
   uint64_t dyn_mask;           /* from the layout: dynamic-offset bindings */
};

#define BORGVK_MAX_VERTEX_ATTRS    8

struct borgvk_vertex_attr {
   uint32_t location;
   uint32_t binding;
   uint32_t offset;
   VkFormat format;
};

struct borgvk_pipeline {
   struct vk_object_base base;
   VkCullModeFlags cull_mode;      /* from VkPipelineRasterizationStateCreateInfo */
   VkFrontFace     front_face;
   /* VkPipelineVertexInputStateCreateInfo, so a generic draw can find
    * position and texture coordinates in the bound vertex buffers. */
   VkPrimitiveTopology       topology;
   uint32_t                  binding_stride[BORGVK_MAX_VERTEX_BINDINGS];
   uint32_t                  attr_count;
   struct borgvk_vertex_attr attrs[BORGVK_MAX_VERTEX_ATTRS];
   bool                      binding_instance[BORGVK_MAX_VERTEX_BINDINGS];   /* per-instance rate */
   bool                      restart;      /* primitiveRestartEnable */
   /* borgc option word for this pipeline's vertex shader (see borgc_compile_nir): vertex inputs
    * are typed fetches whenever the pipeline declares any attribute. */
   uint32_t                  vfetch;
   /* The pipeline's own shader blobs and fixed-function state, as compiled/packed at creation. */
   struct borgvk_shader_blob blob[BORGVK_SHADER_STAGE_COUNT];
   uint32_t                  blend_cfg, blend_const, state_reg[5];
   bool                      state_valid;
   /* Compute: the borgc program, the registers the driver presets (register,
    * value pairs) and LocalSize. cs_ok is false when borgc refused the shader. */
   bool     cs_ok;
   uint32_t cs_nwords, cs_nregs;
   uint32_t cs_words[512];
   uint32_t cs_regs[64];
   uint32_t cs_local[3];
   uint32_t cs_flags;   /* bit 0: the shader uses atomics (its grid cannot be split) */
};

VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_descriptor_set_layout, vk.base,
                               VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_DEFINE_NONDISP_HANDLE_CASTS(borgvk_pipeline, base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE)

#ifdef __cplusplus
}
#endif


/* borgvk_sync.c: set or reset a VkEvent from the host (queue replay of vkCmdSetEvent / vkCmdResetEvent). */
void borgvk_event_set_status(VkEvent event, bool set);

#endif /* BORGVK_PRIVATE_H */
