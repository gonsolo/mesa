/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * See borg_nir_passes.h for why this is shared rather than duplicated.
 */
#include "borg_nir_passes.h"
#include "borgc_flags.h"
#include "util/format/u_format.h"

#include "glsl_types.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"
#include "vulkan/vulkan_core.h"
#include "util/ralloc.h"

#include <stdlib.h>

const struct spirv_to_nir_options borg_spirv_options = {
   .environment            = NIR_SPIRV_VULKAN,
   .ubo_addr_format        = nir_address_format_vec2_index_32bit_offset,
   .ssbo_addr_format       = nir_address_format_vec2_index_32bit_offset,
   .phys_ssbo_addr_format  = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_logical,
   .shared_addr_format     = nir_address_format_32bit_offset,
   .constant_addr_format   = nir_address_format_64bit_global,
};

const struct nir_shader_compiler_options borg_nir_options = {
   .lower_fdiv = true,
   .lower_fceil = true,
   .lower_fsat = true,
   .lower_flrp32 = true,
   .lower_ffract = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_pack_unorm_4x8 = true,
   .max_unroll_iterations = 32,   /* the ISA has no backward branch: constant loops are unrolled */
};

/* Location (vec4-slot) sizing for nir_lower_io of varyings. */
static unsigned
borg_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Compute: texel buffers.  A texelFetch on a textureBuffer / imageLoad on an imageBuffer
 * becomes four word loads from the binding's window, 16 bytes per texel.  The buffer view's
 * format is only known at dispatch, so the driver decodes the view into that canonical
 * four-word layout (RGBA as the shader's declared type: float, uint or int bits) before the
 * dispatch; the shader never sees the format. */
static bool
borg_texel_buffer_filter(const nir_instr *instr, const void *data)
{
   if (instr->type == nir_instr_type_tex) {
      const nir_tex_instr *tex = nir_instr_as_tex(instr);
      return tex->op == nir_texop_txf &&
             (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF || (tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array));
   }
   if (instr->type == nir_instr_type_intrinsic) {
      const nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
      return ((i->intrinsic == nir_intrinsic_image_deref_load || i->intrinsic == nir_intrinsic_image_deref_store) &&
              nir_intrinsic_image_dim(i) == GLSL_SAMPLER_DIM_BUF) ||
             ((i->intrinsic == nir_intrinsic_image_deref_store || i->intrinsic == nir_intrinsic_image_deref_load) &&
              nir_intrinsic_image_dim(i) == GLSL_SAMPLER_DIM_2D &&
              !nir_intrinsic_image_array(i));
   }
   return false;
}

/* A flat float varying is the provoking vertex's bits, like an integer one: the load says so. */
static bool
borg_flat_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   const unsigned slot = sem.location - VARYING_SLOT_VAR0 + (nir_src_is_const(intr->src[0]) ? nir_src_as_uint(intr->src[0]) : 0);
   if (sem.location < VARYING_SLOT_VAR0 || slot > 63 || !(*(uint64_t *)data >> slot & 1) ||
       nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) != nir_type_float)
      return false;
   nir_intrinsic_set_dest_type(intr, nir_type_uint | intr->def.bit_size);
   return true;
}

static bool
borg_frag_image_filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_intrinsic && borg_texel_buffer_filter(instr, data) &&
          nir_intrinsic_image_dim(nir_instr_as_intrinsic(instr)) == GLSL_SAMPLER_DIM_2D;
}

static nir_def *
borg_texel_buffer_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_deref_instr *deref;
   nir_def *coord;
   if (instr->type == nir_instr_type_intrinsic &&
       (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_image_deref_store ||
        nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_image_deref_load) &&
       nir_intrinsic_image_dim(nir_instr_as_intrinsic(instr)) != GLSL_SAMPLER_DIM_BUF) {
      /* A 2D image store (single 32-bit texel formats): the binding's window holds the image's
       * width in word 0, then the texels row by row; the host copies the image in and back. */
      nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
      nir_variable *ivar = nir_deref_instr_get_variable(nir_src_as_deref(st->src[0]));
      if (!ivar)
         return NULL;
      const nir_address_format afmt = borg_spirv_options.ssbo_addr_format;
      nir_def *ridx = nir_vulkan_resource_index(b, nir_address_format_num_components(afmt),
                                                nir_address_format_bit_size(afmt), nir_imm_int(b, 0),
                                                .desc_set = ivar->data.descriptor_set,
                                                .binding = ivar->data.binding,
                                                .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      nir_def *rdesc = nir_load_vulkan_descriptor(b, nir_address_format_num_components(afmt),
                                                  nir_address_format_bit_size(afmt), ridx,
                                                  .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      nir_def *buf = nir_channel(b, rdesc, 0);
      nir_def *width = nir_load_ssbo(b, 1, 32, buf, nir_imm_int(b, 0), .align_mul = 4);
      nir_def *x = nir_channel(b, st->src[1].ssa, 0);
      nir_def *y = nir_channel(b, st->src[1].ssa, 1);
      nir_def *word = nir_iadd(b, nir_iadd(b, nir_imul(b, y, width), x), nir_imm_int(b, 1));
      const enum pipe_format fmt = nir_intrinsic_format(st);
      const bool rgba8 = fmt == PIPE_FORMAT_R8G8B8A8_UNORM;
      const bool rgba8i = fmt == PIPE_FORMAT_R8G8B8A8_UINT || fmt == PIPE_FORMAT_R8G8B8A8_SINT;
      if (st->intrinsic == nir_intrinsic_image_deref_load) {
         nir_def *v = nir_load_ssbo(b, 1, 32, buf, nir_ishl_imm(b, word, 2), .align_mul = 4);
         if (rgba8i) {
            nir_def *c[4];
            for (unsigned i = 0; i < 4; i++) {
               nir_def *s = nir_ushr_imm(b, v, 8 * i);
               c[i] = fmt == PIPE_FORMAT_R8G8B8A8_SINT ? nir_ishr_imm(b, nir_ishl_imm(b, s, 24), 24)
                                                       : nir_iand_imm(b, s, 255);
            }
            return nir_vec(b, c, 4);
         }
         return rgba8 ? nir_unpack_unorm_4x8(b, v) : nir_vec4(b, v, nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0));
      }
      nir_def *val;
      if (rgba8i) {
         val = nir_iand_imm(b, nir_channel(b, st->src[3].ssa, 3), 255);
         for (int i = 2; i >= 0; i--)
            val = nir_iadd(b, nir_ishl_imm(b, val, 8), nir_iand_imm(b, nir_channel(b, st->src[3].ssa, i), 255));
      } else {
         val = rgba8 ? nir_pack_unorm_4x8(b, st->src[3].ssa) : nir_channel(b, st->src[3].ssa, 0);
      }
      nir_store_ssbo(b, val, buf, nir_ishl_imm(b, word, 2), .write_mask = 1, .align_mul = 4);
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      int di = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (di < 0 || ci < 0)
         return NULL;
      deref = nir_src_as_deref(tex->src[di].src);
      coord = nir_channel(b, tex->src[ci].src.ssa, 0);
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_2D) {
         /* A sampled 2D image (level 0): the window is 16 bytes of header (word 0 = width),
          * then the canonical texels row by row. */
         nir_variable *tvar = deref ? nir_deref_instr_get_variable(deref) : NULL;
         if (!tvar)
            return NULL;
         const nir_address_format tf = borg_spirv_options.ssbo_addr_format;
         nir_def *ti = nir_vulkan_resource_index(b, nir_address_format_num_components(tf),
                                                 nir_address_format_bit_size(tf), nir_imm_int(b, 0),
                                                 .desc_set = tvar->data.descriptor_set,
                                                 .binding = tvar->data.binding,
                                                 .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
         nir_def *td = nir_load_vulkan_descriptor(b, nir_address_format_num_components(tf),
                                                  nir_address_format_bit_size(tf), ti,
                                                  .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
         nir_def *tb = nir_channel(b, td, 0);
         nir_def *w = nir_load_ssbo(b, 1, 32, tb, nir_imm_int(b, 0), .align_mul = 16);
         nir_def *y = nir_channel(b, tex->src[ci].src.ssa, 1);
         nir_def *texel = nir_iadd_imm(b, nir_iadd(b, nir_imul(b, y, w), coord), 1);
         return nir_load_ssbo(b, 4, 32, tb, nir_imul_imm(b, texel, 16), .align_mul = 16);
      }
   } else {
      nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
      deref = nir_src_as_deref(i->src[0]);
      coord = nir_channel(b, i->src[1].ssa, 0);
   }
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var)
      return NULL;

   const nir_address_format fmt = borg_spirv_options.ssbo_addr_format;
   nir_def *idx = nir_vulkan_resource_index(b, nir_address_format_num_components(fmt),
                                            nir_address_format_bit_size(fmt), nir_imm_int(b, 0),
                                            .desc_set = var->data.descriptor_set,
                                            .binding = var->data.binding,
                                            .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *desc = nir_load_vulkan_descriptor(b, nir_address_format_num_components(fmt),
                                              nir_address_format_bit_size(fmt), idx,
                                              .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *off = nir_imul_imm(b, coord, 16);
   if (instr->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_image_deref_store) {
      /* The driver encodes the changed texels back into the view's format after the dispatch. */
      nir_store_ssbo(b, nir_instr_as_intrinsic(instr)->src[3].ssa, nir_channel(b, desc, 0), off,
                     .write_mask = 0xf, .align_mul = 16);
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }
   return nir_load_ssbo(b, 4, 32, nir_channel(b, desc, 0), off, .align_mul = 16);
}

/* Cube sampling: the hardware takes the face (TEXA's layer) and the face coordinates s, t in
 * [0, 1]; the direction's major-axis projection is done here (Vulkan 16.5.4). The cube
 * becomes a 2D array sample whose layer is the face. An implicit-LOD sample gets an explicit
 * LOD from the direction's derivatives, which stay continuous where the quad's lanes pick
 * different faces (their projected (s, t) do not); gradients go through nir_lower_tex. */
static bool
borg_lower_cube(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || tex->is_array)
      return false;
   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (ci < 0)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *c = tex->src[ci].src.ssa;
   nir_def *x = nir_channel(b, c, 0), *y = nir_channel(b, c, 1), *z = nir_channel(b, c, 2);
   /* Non-negative floats order like their bits: the comparisons are integer ones, two
    * instructions each. A sign test is the sign bit. */
   nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);
   nir_def *xneg = nir_ilt_imm(b, x, 0), *yneg = nir_ilt_imm(b, y, 0), *zneg = nir_ilt_imm(b, z, 0);
   nir_def *nx = nir_fneg(b, x), *ny = nir_fneg(b, y), *nz = nir_fneg(b, z);
   /* per major axis: |ma|, sc, tc, face */
   nir_def *X[4] = { ax, nir_bcsel(b, xneg, z, nz), ny, nir_b2f32(b, xneg) };
   nir_def *Y[4] = { ay, x, nir_bcsel(b, yneg, nz, z), nir_fadd_imm(b, nir_b2f32(b, yneg), 2.0) };
   nir_def *Z[4] = { az, nir_bcsel(b, zneg, nx, x), ny, nir_fadd_imm(b, nir_b2f32(b, zneg), 4.0) };
   nir_def *x_over_y = nir_ult(b, ay, ax);                          /* X beats Y; ties go to Y */
   nir_def *not_z = nir_ior(b, nir_ult(b, az, ax), nir_ult(b, az, ay));   /* Z wins ties */
   nir_def *r[4];
   for (int i = 0; i < 4; i++)
      r[i] = nir_bcsel(b, not_z, nir_bcsel(b, x_over_y, X[i], Y[i]), Z[i]);
   nir_def *half_inv = nir_fmul_imm(b, nir_frcp(b, r[0]), 0.5f);
   nir_def *half = nir_imm_float(b, 0.5f);
   nir_def *st = nir_vec3(b, nir_ffma(b, r[1], half_inv, half), nir_ffma(b, r[2], half_inv, half), r[3]);
   if (tex->op == nir_texop_tex) {
      /* With w = c x dc, d(s, t) * 2 |ma|^2 are w's two components off the major axis, so
       * |ds| + |dt| = (|wx| + |wy| + |wz| - |w[major]|) / (2 |ma|^2). */
      nir_def *rho = NULL;
      for (int axis = 0; axis < 2; axis++) {
         nir_def *d = axis ? nir_ddy(b, c) : nir_ddx(b, c);
         nir_def *dx = nir_channel(b, d, 0), *dy = nir_channel(b, d, 1), *dz = nir_channel(b, d, 2);
         nir_def *w[3] = { nir_fsub(b, nir_fmul(b, y, dz), nir_fmul(b, z, dy)),
                           nir_fsub(b, nir_fmul(b, z, dx), nir_fmul(b, x, dz)),
                           nir_fsub(b, nir_fmul(b, x, dy), nir_fmul(b, y, dx)) };
         nir_def *major = nir_bcsel(b, not_z, nir_bcsel(b, x_over_y, w[0], w[1]), w[2]);
         nir_def *sum = nir_fadd(b, nir_fadd(b, nir_fabs(b, w[0]), nir_fabs(b, w[1])), nir_fabs(b, w[2]));
         nir_def *off = nir_fsub(b, sum, nir_fabs(b, major));
         rho = rho ? nir_fmax(b, rho, off) : off;
      }
      nir_def *width = nir_i2f32(b, nir_channel(b, nir_get_texture_size(b, tex), 0));
      /* rho in face units * 2 |ma|^2 -> texels per pixel: * width / (2 |ma|^2) */
      rho = nir_fmul(b, nir_fmul(b, rho, width), nir_fmul_imm(b, nir_fmul(b, half_inv, half_inv), 2.0f));
      nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_flog2(b, rho));
      tex->op = nir_texop_txl;
   }
   nir_src_rewrite(&tex->src[ci].src, st);
   tex->coord_components = 3;
   tex->is_array = true;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   return true;
}

/* log2(x) for x > 0: the exponent plus a polynomial of the mantissa (error under 0.0011). */
static bool
borg_flog2_filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_alu && nir_instr_as_alu(instr)->op == nir_op_flog2;
}

static nir_def *
borg_flog2_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_def *x = nir_ssa_for_alu_src(b, nir_instr_as_alu(instr), 0);
   nir_def *e = nir_iadd_imm(b, nir_ushr_imm(b, nir_iand_imm(b, x, 0x7fffffff), 23), -127);
   nir_def *m = nir_ior_imm(b, nir_iand_imm(b, x, 0x7fffff), 0x3f800000);
   nir_def *t = nir_fadd_imm(b, m, -1.0f);
   nir_def *q = nir_ffma(b, t, nir_imm_float(b, -0.15638611f), nir_imm_float(b, 0.42086454f));
   nir_def *tt = nir_fsub(b, t, nir_fmul(b, t, t));
   return nir_fadd(b, nir_i2f32(b, e), nir_ffma(b, tt, q, t));
}

#ifdef BORG_SMALL_NIR
/* nir_opt_algebraic's tables are 8.6 MB; the on-board build lowers only what the ISA lacks. */
static bool
borg_basic_alu_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_op op = nir_instr_as_alu(instr)->op;
   return op == nir_op_fsub || op == nir_op_fdiv || op == nir_op_fsqrt;
}

static nir_def *
borg_basic_alu_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   switch (alu->op) {
   case nir_op_fsub:
      return nir_fadd(b, x, nir_fneg(b, nir_ssa_for_alu_src(b, alu, 1)));
   case nir_op_fdiv:
      return nir_fmul(b, x, nir_frcp(b, nir_ssa_for_alu_src(b, alu, 1)));
   default:
      return nir_frcp(b, nir_frsq(b, x));
   }
}
#endif

/* Integer division through the float reciprocal, one correction step: exact for operands below 2^24. */
/* A fragment shader that loads many inputs and uniform vectors up front runs out of registers:
 * load the components one at a time and sink each load to its first use. */
static bool
borg_wide_loads(nir_shader *nir)
{
   unsigned comps = 0;
   nir_foreach_function_impl(impl, nir)
      nir_foreach_block(blk, impl)
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
            if (i->intrinsic == nir_intrinsic_load_input || i->intrinsic == nir_intrinsic_load_ubo)
               comps += i->def.num_components;
         }
   return comps >= 16;
}

static bool
borg_ubo_split_filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_ubo &&
          nir_instr_as_intrinsic(instr)->def.num_components > 1;
}

static nir_def *
borg_ubo_split_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_intrinsic_instr *ld = nir_instr_as_intrinsic(instr);
   nir_def *c[4];
   for (unsigned i = 0; i < ld->def.num_components; i++)
      c[i] = nir_load_ubo(b, 1, 32, ld->src[0].ssa, nir_iadd_imm(b, ld->src[1].ssa, 4 * i),
                          .access = nir_intrinsic_access(ld), .align_mul = 4, .align_offset = 0,
                          .range_base = nir_intrinsic_range_base(ld) + 4 * i, .range = 4);
   return nir_vec(b, c, ld->def.num_components);
}

/* Repeat a scalar input or uniform load at each ALU use: reloading is cheaper than holding the register. */
static int
borg_use_order(const void *a, const void *b)
{
   const nir_instr *x = nir_src_use_instr(*(nir_src *const *)a), *y = nir_src_use_instr(*(nir_src *const *)b);
   return x->index < y->index ? -1 : x->index > y->index;
}

static bool
borg_dup_load(nir_builder *b, nir_intrinsic_instr *ld, void *data)
{
   const bool vec_input = ld->intrinsic == nir_intrinsic_load_input && b->shader->info.stage == MESA_SHADER_VERTEX;
   if ((ld->intrinsic != nir_intrinsic_load_input && ld->intrinsic != nir_intrinsic_load_ubo) ||
       (ld->def.num_components != 1 && !vec_input))
      return false;
   unsigned n = 0;
   if (nir_def_used_by_if(&ld->def))
      return false;
   nir_foreach_use(use, &ld->def) {
      if (nir_src_use_instr(use)->type != nir_instr_type_alu)
         return false;
      n++;
   }
   if (n < 2)
      return false;
   nir_src *uses[64];
   unsigned k = 0;
   nir_foreach_use(use, &ld->def)
      if (k < 64)
         uses[k++] = use;
   if (vec_input)
      qsort(uses, k, sizeof(uses[0]), borg_use_order);
   /* A vertex input is a whole fetch: uses close together share one. */
   nir_instr *cur = nir_src_use_instr(uses[0]);
   nir_def *def = &ld->def;
   bool changed = false;
   for (unsigned i = 1; i < k; i++) {
      nir_instr *u = nir_src_use_instr(uses[i]);
      if (vec_input && u->block == cur->block && u->index - cur->index <= 16) {
         nir_src_rewrite(uses[i], def);
         continue;
      }
      b->cursor = nir_before_instr(u);
      nir_instr *c = nir_instr_clone(b->shader, &ld->instr);
      nir_builder_instr_insert(b, c);
      def = &nir_instr_as_intrinsic(c)->def;
      nir_src_rewrite(uses[i], def);
      cur = u;
      changed = true;
   }
   return changed;
}

/* Same for the constant + descriptor address a UBO load adds, so it does not stay live between loads. */
static bool
borg_dup_addr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_iadd || !nir_src_is_const(alu->src[0].src))
      return false;
   unsigned k = 0;
   nir_src *uses[64];
   nir_foreach_use(use, &alu->def) {
      nir_instr *u = nir_src_use_instr(use);
      if (u->type != nir_instr_type_intrinsic || nir_instr_as_intrinsic(u)->intrinsic != nir_intrinsic_load_ubo ||
          use != &nir_instr_as_intrinsic(u)->src[1] || k == 64)
         return false;
      uses[k++] = use;
   }
   for (unsigned i = 1; i < k; i++) {
      b->cursor = nir_before_instr(nir_src_use_instr(uses[i]));
      nir_instr *c = nir_instr_clone(b->shader, &alu->instr);
      nir_builder_instr_insert(b, c);
      nir_src_rewrite(uses[i], &nir_instr_as_alu(c)->def);
   }
   return k > 1;
}

static bool
borg_idiv_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_op op = nir_instr_as_alu(instr)->op;
   return op == nir_op_idiv || op == nir_op_udiv || op == nir_op_imod || op == nir_op_umod || op == nir_op_irem;
}

static nir_def *
borg_idiv_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0), *y = nir_ssa_for_alu_src(b, alu, 1);
   const bool sgn = alu->op == nir_op_idiv || alu->op == nir_op_imod || alu->op == nir_op_irem;
   nir_def *sx = nir_ishr_imm(b, x, 31), *sy = nir_ishr_imm(b, y, 31);
   nir_def *a = sgn ? nir_isub(b, nir_ixor(b, x, sx), sx) : x;
   nir_def *d = sgn ? nir_isub(b, nir_ixor(b, y, sy), sy) : y;
   nir_def *q = nir_f2i32(b, nir_fmul(b, nir_i2f32(b, a), nir_frcp(b, nir_i2f32(b, d))));
   nir_def *r = nir_isub(b, a, nir_imul(b, q, d));
   nir_def *lo = nir_ilt_imm(b, r, 0);
   q = nir_bcsel(b, lo, nir_iadd_imm(b, q, -1), q);
   r = nir_bcsel(b, lo, nir_iadd(b, r, d), r);
   nir_def *hi = nir_uge(b, r, d);
   q = nir_bcsel(b, hi, nir_iadd_imm(b, q, 1), q);
   r = nir_isub(b, a, nir_imul(b, q, d));
   const bool quot = alu->op == nir_op_idiv || alu->op == nir_op_udiv;
   if (!sgn)
      return quot ? q : r;
   if (quot) {
      nir_def *sq = nir_ixor(b, sx, sy);
      return nir_isub(b, nir_ixor(b, q, sq), sq);
   }
   r = nir_isub(b, nir_ixor(b, r, sx), sx);
   if (alu->op == nir_op_irem)
      return r;
   nir_def *nz = nir_ishr_imm(b, nir_ior(b, r, nir_ineg(b, r)), 31);
   return nir_iadd(b, r, nir_iand(b, nir_iand(b, nz, nir_ishr_imm(b, nir_ixor(b, r, y), 31)), y));
}

/* F2I truncates: trunc(x) = float(int(x)), floor(x) = trunc(x) - (x < trunc(x)); fmod(x, y) = x - y * floor(x / y). */
static bool
borg_trunc_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_op op = nir_instr_as_alu(instr)->op;
   return op == nir_op_ftrunc || op == nir_op_ffloor || op == nir_op_fmod;
}

static nir_def *
borg_trunc_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   if (alu->op == nir_op_fmod) {
      nir_def *y = nir_ssa_for_alu_src(b, alu, 1);
      return nir_fsub(b, x, nir_fmul(b, y, nir_ffloor(b, nir_fmul(b, x, nir_frcp(b, y)))));
   }
   const uint32_t fp = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_exact;   /* else nir_opt_algebraic turns i2f(f2i(x)) back into ftrunc, forever */
   nir_def *t = nir_i2f32(b, nir_f2i32(b, x));
   b->fp_math_ctrl = fp;
   if (alu->op == nir_op_ftrunc)
      return t;
   return nir_fsub(b, t, nir_b2f32(b, nir_flt(b, x, t)));
}

/* No multiview: gl_ViewIndex is 0. */
static bool
borg_view_index_filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_intrinsic &&
          (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_view_index ||
           nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_layer_id);
}

static nir_def *
borg_view_index_lower(nir_builder *b, nir_instr *instr, void *data)
{
   return nir_imm_int(b, 0);
}

static bool
borg_blend_const_filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_blend_const_color_rgba;
}

static nir_def *
borg_blend_const_lower(nir_builder *b, nir_instr *instr, void *data)
{
   const float *k = data;
   return nir_imm_vec4(b, k[0], k[1], k[2], k[3]);
}

/* The vertex id, instance id and base instance are the same everywhere in the shader: one load at the top,
 * so a branch holds nothing nir_opt_peephole_select refuses to flatten. */
static bool
borg_hoist_vertex_ids(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_def *first[nir_num_intrinsics] = { 0 };
   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
         if (i->intrinsic != nir_intrinsic_load_vertex_id && i->intrinsic != nir_intrinsic_load_instance_id &&
             i->intrinsic != nir_intrinsic_load_base_instance)
            continue;
         if (!first[i->intrinsic]) {
            nir_builder b = nir_builder_at(nir_before_impl(impl));
            first[i->intrinsic] = i->intrinsic == nir_intrinsic_load_base_instance ?
               nir_imm_int(&b, 0) : nir_load_system_value(&b, i->intrinsic, 0, 1, 32);
         }
         if (first[i->intrinsic] == &i->def)
            continue;
         nir_def_rewrite_uses(&i->def, first[i->intrinsic]);
         nir_instr_remove(instr);
         progress = true;
      }
   }
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/* The blended colour is clamped to what the format holds (the pack block only rounds). */
static bool
borg_clamp_color(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const enum pipe_format *format = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   const unsigned loc = nir_intrinsic_io_semantics(intr).location;
   if (loc != FRAG_RESULT_DATA0 && loc != FRAG_RESULT_COLOR)
      return false;
   const bool unorm = util_format_is_unorm(*format) || util_format_is_srgb(*format);
   if (!unorm && !util_format_is_snorm(*format))
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *v = intr->src[0].ssa;
   v = unorm ? nir_fsat(b, v) : nir_fmin(b, nir_fmax(b, v, nir_imm_float(b, -1.0f)), nir_imm_float(b, 1.0f));
   nir_src_rewrite(&intr->src[0], v);
   return true;
}

static bool remat_vertex_inputs;

void
borg_lower_set_vertex_remat(bool on)
{
   remat_vertex_inputs = on;
}

void
borg_lower_nir_for_borgc(struct nir_shader *nir)
{
   borg_lower_nir_for_borgc_blend(nir, NULL);
}

void
borg_lower_nir_for_borgc_blend(struct nir_shader *nir, const struct borg_blend *blend)
{
   /* One invocation per workgroup: `shared` variables are private ones. */
   if (nir->info.stage == MESA_SHADER_COMPUTE && !nir->info.workgroup_size_variable &&
       nir->info.workgroup_size[0] * nir->info.workgroup_size[1] * nir->info.workgroup_size[2] == 1) {
      nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared)
         var->data.mode = nir_var_shader_temp;
      nir_fixup_deref_modes(nir);
   }
   /* The Vulkan-runtime preamble.
    *
    * The driver gets this from vk_spirv_to_nir before its own passes run; the
    * offline CLI cannot, because vk_spirv_to_nir dereferences
    * device->physical->properties and standing up a device is exactly what the
    * CLI exists to avoid. So it lives here, where both front ends get it and
    * neither can drift. Every pass below is idempotent, so the driver running
    * it a second time costs a walk and changes nothing.
    *
    * Skipping it is not subtle: cube.frag calls linearToSrgb(), and WITHOUT
    * nir_inline_functions that call is never inlined, the output store stays a
    * store_deref borgc does not understand, and dead-code elimination then
    * deletes almost the whole shader -- 66 selected instructions emitting 7.
    * Local constant initializers have to be lowered before inlining or they
    * get initialized in the caller instead of the callee. */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_lower_input_attachments, &(nir_input_attachment_options){ 0 });
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out | nir_var_system_value, NULL);
   NIR_PASS(_, nir, nir_lower_var_copies);

   /* SSA + system values (gl_VertexIndex → load_vertex_id), then lower the UBO
    * deref I/O to load_ubo with byte offsets. The Borg core has integer ops
    * (iadd/ishl), so the offset address arithmetic is selectable. */
   /* A constant array indexed by gl_VertexIndex (the usual vertex-less fullscreen
    * triangle/quad) becomes a bcsel chain. */
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_function_temp | nir_var_shader_temp, UINT32_MAX);
   /* A vertex shader that writes an output in a branch: turn the branches into selects. */
   if (nir->info.stage == MESA_SHADER_VERTEX || nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir), nir_var_shader_out);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
   }
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      bool progress;
      NIR_PASS(_, nir, borg_hoist_vertex_ids);
      do {
         progress = false;
         const nir_opt_peephole_select_options sel = { .limit = 1024, .indirect_load_ok = true, .expensive_alu_ok = true };
         NIR_PASS(progress, nir, nir_opt_peephole_select, &sel);
         NIR_PASS(progress, nir, nir_opt_dce);
         NIR_PASS(progress, nir, nir_opt_dead_cf);
      } while (progress);
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* A storage image the shader only reads is sampled with texelFetch. */
      const nir_opt_access_options access = { .is_vulkan = true };
      NIR_PASS(_, nir, nir_opt_access, &access);
      NIR_PASS(_, nir, nir_lower_readonly_images_to_tex, false);
   }
   NIR_PASS(_, nir, nir_shader_lower_instructions, borg_view_index_filter, borg_view_index_lower, NULL);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            borg_spirv_options.ubo_addr_format);
   /* Storage buffers (compute): load/store/atomic_ssbo intrinsics. */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            borg_spirv_options.ssbo_addr_format);
   if (nir->info.stage == MESA_SHADER_COMPUTE)
      NIR_PASS(_, nir, nir_shader_lower_instructions, borg_texel_buffer_filter,
               borg_texel_buffer_lower, NULL);
   else if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_shader_lower_instructions, borg_frag_image_filter, borg_texel_buffer_lower, NULL);
   /* Push constants. Confirmed empirically (not assumed): with
    * push_const_addr_format left at its default, a `layout(push_constant)`
    * access arrives here as a plain load_deref, same as any other pointer,
    * and the flat selection walk below has no deref handling -- the whole
    * shader silently DCE'd to nothing (0 instructions selected) the first
    * time this was tried. Lowering to an explicit byte offset turns it into
    * nir_intrinsic_load_push_constant, which IS something a flat walk can
    * select from directly. */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   if (nir->info.stage == MESA_SHADER_COMPUTE) {
      /* `shared` variables: one more memory window (compute.rs, slot 33). */
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   }
   uint64_t flat_slots = 0;
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      nir_foreach_shader_in_variable(var, nir)
         if (var->data.interpolation == INTERP_MODE_FLAT && var->data.location >= VARYING_SLOT_VAR0) {
            const unsigned n = glsl_count_attribute_slots(var->type, false);
            for (unsigned i = 0; i < n; i++)
               flat_slots |= 1ull << (var->data.location - VARYING_SLOT_VAR0 + i);
         }
   /* Varying I/O → load_input/store_output(location). vec4-slot sizing. */
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            borg_type_size, 0);
   if (flat_slots)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, borg_flat_input, nir_metadata_control_flow, &flat_slots);
   if (blend && nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_blend, &blend->opts);
      NIR_PASS(_, nir, nir_shader_lower_instructions, borg_blend_const_filter, borg_blend_const_lower,
               (void *)blend->konst);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, borg_clamp_color, nir_metadata_control_flow,
               (void *)&blend->opts.rt[0].format);
   }

   /* Lower/optimize toward the Borg ISA: scalarize, then fold and lower ALU ops
    * (fsub→fadd, fdiv→fmul·frcp via lower_fdiv, fdot→fmul+ffma, constant
    * folding) so the backend sees the small supported op set. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      const nir_lower_tex_options tex_options = { .lower_txd_cube_map = true };
      NIR_PASS(_, nir, nir_lower_tex, &tex_options);
      NIR_PASS(_, nir, nir_shader_instructions_pass, borg_lower_cube, nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, nir_shader_lower_instructions, borg_flog2_filter, borg_flog2_lower, NULL);
   }
   NIR_PASS(_, nir, nir_shader_lower_instructions, borg_idiv_filter, borg_idiv_lower, NULL);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   const bool wide = (nir->info.stage == MESA_SHADER_FRAGMENT || (nir->info.stage == MESA_SHADER_VERTEX && remat_vertex_inputs)) &&
                     borg_wide_loads(nir);
   if (wide && nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_lower_instructions, borg_ubo_split_filter, borg_ubo_split_lower, NULL);
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_in, NULL, NULL);
   }
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      /* An `if` on a constant (a specialization constant, say) loses its dead arm. */
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
      NIR_PASS(progress, nir, nir_shader_lower_instructions, borg_trunc_filter, borg_trunc_lower, NULL);
#ifdef BORG_SMALL_NIR
      NIR_PASS(progress, nir, nir_shader_lower_instructions, borg_basic_alu_filter,
               borg_basic_alu_lower, NULL);
#else
      NIR_PASS(progress, nir, nir_opt_algebraic);
#endif
      /* Flatten if/else into bcsel selects where that is cheaper than
       * predication.
       *
       * Borg DOES have control flow now -- BRZ/BRNZ plus an execution mask
       * (EXPUSH/EXELSE/EXPOP), which borgc lowers nir_if onto. Flattening is
       * still preferable for a small `if`, since one bcsel beats five
       * instructions plus both arms executing. The limit is what decides
       * where that trade tips, and it is deliberately still high: lowering it
       * would change codegen for every existing shader, which is a separate
       * change with its own image comparison. Raising the mask's share of
       * control flow is a tuning exercise to do against measurements, not a
       * side effect of adding the capability.
       *
       * expensive_alu_ok so cube.frag's linearToSrgb pow branch does not block
       * flattening -- the whole select then collapses to one FSRGB op. */
      /* BORGC_NO_FLATTEN keeps every `if` as control flow, to exercise the mask path. */
      const nir_opt_peephole_select_options peephole_opts = {
         .limit = getenv("BORGC_NO_FLATTEN") ? 1 : 64, .indirect_load_ok = true, .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_opts);
   } while (progress);

   /* Compute: booleans as 32-bit integers. The backend has no select; both arms
    * of an `if` store into the phi's register. */
   if (nir->info.stage == MESA_SHADER_COMPUTE)
      NIR_PASS(_, nir, nir_lower_bool_to_int32);
   if (wide) {
      nir_foreach_function_impl(impl, nir)
         nir_index_instrs(impl);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, borg_dup_load, nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, nir_shader_alu_pass, borg_dup_addr, nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, nir_opt_move, nir_move_load_ubo | nir_move_load_input | nir_move_const_undef | nir_move_alu);
   }
   /* Phis become registers for every stage. */
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);

   if (getenv("BORGC_DUMP_NIR"))
      nir_print_shader(nir, stderr);
}
