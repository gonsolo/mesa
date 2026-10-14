/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BORGVK_PRIVATE_H
#define BORGVK_PRIVATE_H

#include "borgc_flags.h"
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

#define BORGVK_USE_WSI_PLATFORM   /* Mesa's headless WSI backend needs no platform */

struct borgvk_instance {
   struct vk_instance vk;
};

struct borgvk_physical_device {
   struct vk_physical_device vk;

   VkPhysicalDeviceMemoryProperties memory;

   const struct vk_sync_type *sync_types[2];

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

#define BORGVK_SHADER_BLOB_MAX 4096
enum borgvk_shader_stage {
   BORGVK_STAGE_VERT = 0,
   BORGVK_STAGE_FRAG = 1,
   BORGVK_SHADER_STAGE_COUNT,
};
#define BORGVK_POOL_MAX  64
#define BORGVK_POOL_SLOT 40
#define BORGVK_SRGB_SLOT 41
#define BORGVK_VPOOL_SLOT 42
struct borgvk_shader_blob {
   uint8_t  data[BORGVK_SHADER_BLOB_MAX];
   uint32_t len;   /* 0 = not compiled (or didn't fit) */
   uint32_t pool[BORGVK_POOL_MAX];
   uint32_t npool;
};

/* A viewport and scissor as the firmware takes them (packet 0xC2). */
struct borgvk_vp {
   bool     valid;
   float    v[6];           /* x, y, width, height, minDepth, maxDepth */
   uint32_t sc[4];          /* scissor x0, y0, x1, y1 */
};

struct borgvk_device {
   struct vk_device vk;

   struct vk_device_dispatch_table cmd_dispatch;

   struct vk_queue queue;

   /* Non-owning copy of physical_device->drm_fd.  -1 → malloc+serial fallback. */
   int drm_fd;

   const struct borg_blend *blend_lower;
   struct borgvk_shader_blob shader_blob[BORGVK_SHADER_STAGE_COUNT];
   VkPipelineLayout compile_layout;
   uint32_t stage_ubo[2];
   int frag_tex_binding;
   bool frag_reads_pntc;
   uint32_t blend_cfg;
   uint32_t blend_const;
   uint32_t state_reg[5];
   struct borgvk_vp vp;      /* the viewport and scissor of the draw being queued */
   bool     state_valid;
   /* Push constants set by vkCmdPushConstants, replayed at submit (DRM path). */
   uint32_t push_words[32];
   uint32_t push_lo, push_hi;   /* word range written so far; empty when lo >= hi */
};

#define BORGVK_MAX_COLOR_ATTACHMENTS 8

#define BORGVK_MAX_VERTEX_BINDINGS 16

#define BORGVK_MAX_BINDINGS 16
#define BORGVK_MAX_WRITEBACKS 16
#define BORGVK_SSBO_SLOT_BYTES 65536u

struct borgvk_command_buffer {
   struct vk_command_buffer vk;

   bool in_rendering;
   VkRect2D render_area;
   uint32_t layer_count;
   struct borgvk_target *targets;
   uint32_t color_attachment_count;
   struct vk_image_view *color_views[BORGVK_MAX_COLOR_ATTACHMENTS];
   struct vk_image_view *depth_view;
   struct vk_image_view *stencil_view;
   struct vk_image_view *color_resolve_views[BORGVK_MAX_COLOR_ATTACHMENTS];
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
   uint8_t *state_sent;
   size_t state_sent_len;
   struct borgvk_image *batch_img;
   struct borgvk_image *batch_extra[3];   /* colour attachments 1-3 of the batch */
   uint32_t batch_nextra;
   bool batch_serve;
   struct vk_image_view *batch_depth, *batch_stencil;
   struct borgvk_writeback { uint32_t off, size; uint8_t *host; } wb[BORGVK_MAX_WRITEBACKS];
   uint32_t nwb;
   bool batch_srgb;
   const void *batch_pool, *batch_vpool;
   struct borgvk_descriptor_set *region_set;   /* the set whose buffer region is at region_base */
   uint32_t region_base;
   const uint8_t *index_ptr;                    /* bound index buffer, at its offset */
   VkDeviceSize index_avail;
   uint32_t index_size;                         /* 2, 4, or 0 for an unsupported type */
   uint32_t pc[32];                             /* push-constant bytes, for compute */
   VkDeviceSize dyn_off[BORGVK_MAX_BINDINGS];   /* dynamic offsets of desc_set, by binding */
   /* The compute bind point has its own set. */
   struct borgvk_descriptor_set *cs_sets[4];
   VkDeviceSize cs_dyn_offs[4][BORGVK_MAX_BINDINGS];
   struct borgvk_descriptor_set *desc_sets[4];   /* [0] is desc_set; dyn_offs[0] is unused (dyn_off) */
   VkDeviceSize dyn_offs[4][BORGVK_MAX_BINDINGS];
};

extern const struct vk_command_buffer_ops borgvk_cmd_buffer_ops;
struct borgvk_command_buffer;
void borgvk_flush_draws(struct borgvk_command_buffer *cmd);

/* The persistent simulator (borgvk_sim.c). */
struct borgvk_image;
#define BORGVK_CS_MAX_BUFS 34   /* 32 binding slots and the push constants */
bool borgvk_hw_enabled(void);
size_t borgvk_hw_render(const uint8_t *stream, size_t n, uint32_t w, uint32_t h, uint8_t *out);
int borgvk_hw_compute(const uint8_t *job, size_t len, const char *outf);
bool borgvk_sim_serves(const struct borgvk_image *color, bool has_depth_stencil);
uint8_t borgvk_sim_flush_format(VkFormat f);
bool borgvk_sim_bytes_packed(VkFormat f);
bool borgvk_sim_half2(VkFormat f);
bool borgvk_sim_half1(VkFormat f);
bool borgvk_sim_raw32(VkFormat f);
uint32_t borgvk_sim_packed_opt(VkFormat f);
bool borgvk_sim_depth_is_d32(VkFormat f);
#define BORGVK_CORPUS_SEED 0xcbf29ce484222325ull
bool borgvk_corpus_on(void);
uint64_t borgvk_corpus_hash(uint64_t h, const void *p, size_t n);
uint64_t borgvk_corpus_image(uint64_t h, const struct borgvk_image *img);
void borgvk_corpus_log(char kind, uint64_t key, uint64_t content);
struct borgvk_writeback;
VkResult borgvk_sim_run_pass(const uint8_t *stream, size_t n, struct borgvk_image *color,
                             struct borgvk_image *const *extra, uint32_t nextra,
                             const struct vk_image_view *depth_view,
                             const struct vk_image_view *stencil_view,
                             const struct borgvk_writeback *wb, uint32_t nwb);

void borgvk_serial_send_mvp(const float mvp[16]);
void borgvk_serial_send_blend(uint32_t blend_cfg, uint32_t blend_const);
void borgvk_blend_pack(const VkPipelineColorBlendStateCreateInfo *cb,
                       uint32_t *cfg, uint32_t *konst);
/* 0xB4: STENCIL_CFG, STENCIL_FRONT, STENCIL_BACK, DEPTH_CFG, CULL_CFG. */
void borgvk_serial_send_state(const uint32_t reg[5]);
void borgvk_serial_send_viewport(const struct borgvk_vp *vp);
void borgvk_serial_send_target(uint8_t flush_format, const float clear[4]);
void borgvk_serial_send_pass(uint8_t flush_format, uint8_t flags);
void borgvk_serial_send_att(uint8_t count, const uint8_t fmt[3]);
void borgvk_serial_send_mem(uint32_t off, const uint8_t *data, uint32_t n);
void borgvk_serial_send_vattr(uint32_t slot, uint32_t fmt, uint32_t base, uint32_t count,
                              uint32_t stride, uint32_t swz);
void borgvk_serial_send_draw(uint32_t topology, uint32_t index_type, uint32_t restart,
                             uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                             uint32_t first_instance, int32_t vertex_offset, uint32_t index_base,
                             uint32_t ubo_base);
#define BORGVK_TEXG_CHUNK 256
void borgvk_serial_send_texture_chunk(uint32_t off, const uint8_t *data, uint32_t n,
                                      const uint32_t desc_w123[3],
                                      const uint32_t sampler[4]);
void borgvk_serial_send_texture_levels(const uint32_t offs[12]);
/* Pack a pipeline's stencil, depth and cull state into register layout. */
void borgvk_state_pack(const VkGraphicsPipelineCreateInfo *ci, uint32_t reg[5]);

#define BORGVK_GEOM_MAX_VERTS 16
#define BORGVK_GEOM_MAX_TRIS  12

void borgvk_serial_send_geom(const float *verts, int nverts,
                             const uint8_t *idx, const float *uv, int ntris);

#define BORGVK_TEX_DIM 64

void borgvk_serial_send_tex_row(int y, const uint8_t *rgba, const uint32_t sampler[4]);

void borgvk_serial_send_shader(uint8_t stage, const uint8_t *blob, uint32_t len);

void borgvk_serial_send_push_constants(uint32_t offset, uint32_t size,
                                       const void *values);

void borgvk_transport_capture_begin(void);
uint8_t *borgvk_transport_capture_end(size_t *out_len);

VkResult borgvk_queue_submit(struct vk_queue *queue,
                             struct vk_queue_submit *submit);

/* Compiler shim (borgvk_compiler.c → Rust borgc crate). */
void borgvk_layout_set_base(struct borgvk_device *device, uint8_t base[4]);
void borgvk_compile_compute_stage(struct borgvk_device *device,
                                  const VkPipelineShaderStageCreateInfo *stage_info,
                                  struct borgvk_pipeline *pipeline);
bool borgvk_shader_cache_get(const VkPipelineShaderStageCreateInfo *si, uint32_t vfetch,
                             struct borgvk_shader_blob *blob, bool *reads_pntc);
void borgvk_shader_cache_record(const VkPipelineShaderStageCreateInfo *si, uint32_t vfetch,
                                const struct borgvk_shader_blob *blob, bool reads_pntc);
uint32_t borgvk_blob_num_varyings(const struct borgvk_shader_blob *b);
#define BORGVK_LINE_SLOT  16
#define BORGVK_LINE_OTHER 24
void borgvk_compile_empty_frag(struct borgvk_device *device, uint32_t opt);
void borgvk_compile_stage(struct borgvk_device *device, uint32_t vfetch,
                          const VkPipelineShaderStageCreateInfo *stage_info);

VK_DEFINE_HANDLE_CASTS(borgvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(borgvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(borgvk_device, vk.base, VkDevice,
                       VK_OBJECT_TYPE_DEVICE)

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

/* One mip level and layer of an image, rendered to as an image of its own. */
struct borgvk_target {
   struct borgvk_target *next;
   const struct vk_image_view *orig;
   struct borgvk_image image;
   struct vk_image_view view;
};

struct borgvk_buffer_view {
   struct vk_buffer_view vk;
};

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

struct borgvk_desc_map {
   uint8_t n, num[BORGVK_MAX_BINDINGS], slot[BORGVK_MAX_BINDINGS], count[BORGVK_MAX_BINDINGS];
   uint8_t total;
   bool arrays;
};

struct borgvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count;
   uint64_t dyn_mask;           /* slots of *_BUFFER_DYNAMIC descriptors */
   struct borgvk_desc_map map;
};

struct borgvk_descriptor_pool {
   struct vk_object_base base;
   struct list_head sets;       /* every live set allocated from this pool */
   uint32_t max_sets;
   uint32_t num_sets;
};

struct borgvk_descriptor_set {
   struct vk_object_base base;
   struct list_head link;       /* in the pool's list */
   struct borgvk_descriptor_pool *pool;
   struct borgvk_buffer *buffers[BORGVK_MAX_BINDINGS];
   VkDeviceSize offsets[BORGVK_MAX_BINDINGS];
   VkDeviceSize ranges[BORGVK_MAX_BINDINGS];
   struct borgvk_image *images[BORGVK_MAX_BINDINGS];
   /* The view behind images[]: view type, format, mip/layer range, swizzle. */
   struct vk_image_view *views[BORGVK_MAX_BINDINGS];
   struct borgvk_sampler *samplers[BORGVK_MAX_BINDINGS];
   struct vk_buffer_view *buffer_views[BORGVK_MAX_BINDINGS];   /* texel buffers */
   uint64_t dyn_mask;           /* from the layout: dynamic-offset slots */
   uint32_t storage_mask;       /* slots whose buffer is a storage buffer */
   uint32_t sampled_mask;       /* slots whose image is sampled (not storage) */
   struct borgvk_desc_map map;
};

#define BORGVK_MAX_VERTEX_ATTRS    16

struct borgvk_vertex_attr {
   uint32_t location;
   uint32_t binding;
   uint32_t offset;
   VkFormat format;
};

struct borgvk_pipeline {
   struct vk_object_base base;
   VkPrimitiveTopology       topology;
   uint32_t                  binding_stride[BORGVK_MAX_VERTEX_BINDINGS];
   uint32_t                  attr_count;
   struct borgvk_vertex_attr attrs[BORGVK_MAX_VERTEX_ATTRS];
   bool                      binding_instance[BORGVK_MAX_VERTEX_BINDINGS];   /* per-instance rate */
   bool                      restart;      /* primitiveRestartEnable */
   uint32_t                  color_count;  /* colour attachments of the pipeline's render pass */
   bool                      mrt;
   uint32_t                  ubo_mask;      /* the uniform buffer bindings the draw loads from */
   uint8_t                   set_base[4];
   int                       tex_binding;   /* binding of the texture the fragment shader samples */
   uint32_t                  vfetch;
   struct borgvk_shader_blob blob[BORGVK_SHADER_STAGE_COUNT];
   uint32_t                  blend_cfg, blend_const, state_reg[5];
   struct borgvk_vp          vp;                 /* static viewport/scissor, if the pipeline has them */
   bool                      vp_dyn, sc_dyn;     /* taken from the command buffer */
   bool                      srgb_table;   /* the fragment shader decodes the destination */
   bool                      state_valid;
   bool     cs_ok;
   uint32_t cs_nwords, cs_nregs;
   uint32_t cs_words[512];
   uint32_t cs_regs[64];
   uint32_t cs_local[3];
   uint32_t cs_flags;
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

struct borgvk_descriptor_set *borgvk_find_set(struct vk_queue_submit *submit);
const uint32_t *borgvk_sampler_desc(const struct borgvk_sampler *sampler);
void borgvk_send_texture_row(struct borgvk_image *tex, const struct borgvk_sampler *sampler, int dy);
void borgvk_send_blend_state(const struct borgvk_device *device);
void borgvk_upload_shaders(struct borgvk_device *device);
void borgvk_upload_shaders_serial(struct borgvk_device *device);
void borgvk_send_blend_state_serial(const struct borgvk_device *device);
bool borgvk_set_is_cube(struct borgvk_descriptor_set *set);
bool borgvk_submit_is_cube(struct vk_queue_submit *submit);
VkResult borgvk_sim_run_stream(uint8_t *bytes, size_t nbytes, struct borgvk_image *color_img, uint32_t sim_dim);
VkResult borgvk_submit_sim_cube(struct borgvk_device *device, struct vk_queue_submit *submit);
VkResult borgvk_submit_cube_live(struct borgvk_device *device, struct vk_queue_submit *submit);
struct tex_conv {
   uint32_t bpp, obpp, cs, sch, alpha, pack16, zstride, soff;
   bool swap_rb;
};
void borgvk_tex_convert(const struct tex_conv *c, uint8_t *dst, const uint8_t *src, uint32_t n);
void *borgvk_decode_compressed_image(const struct borgvk_image *src, struct borgvk_image *out, struct borgvk_device_memory *mem);
bool borgvk_send_generic_texture(const struct vk_image_view *view, const struct borgvk_image *img,
                                 const struct borgvk_sampler *sampler);
bool borgvk_send_buffer_texture(const struct vk_buffer_view *bv);
bool borgvk_vertex_format_to_fetch(VkFormat f, uint32_t *code, uint32_t *swz, uint32_t *read_bytes);

void borgvk_resolve_region(struct borgvk_image *src, struct borgvk_image *dst, const VkImageResolve *reg);
void borgvk_free_targets(struct borgvk_command_buffer *cmd);
void borgvk_clear_attachment_rect(struct vk_image_view *view, VkImageAspectFlags aspectMask,
                                  const VkClearValue *clearValue, VkRect2D rect,
                                  uint32_t baseLayer, uint32_t layerCount);

void borgvk_event_set_status(VkEvent event, bool set);

#endif /* BORGVK_PRIVATE_H */
