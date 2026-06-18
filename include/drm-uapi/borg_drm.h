/* SPDX-License-Identifier: MIT */
/*
 * Borg GPU DRM uAPI — shared between the borgvk Mesa Vulkan driver and
 * libborg_drm_shim.so.  When a real Linux kernel driver exists, this header
 * will be a copy of the in-kernel <uapi/drm/borg_drm.h>.
 *
 * Layout of cube.c's vktexcube_vs_uniform (binding 0 UBO):
 *   float mvp[4][4];           bytes   0.. 63   (16 floats, updated per frame)
 *   float position[36][4];     bytes  64..639   (static mesh positions)
 *   float attr[36][4];         bytes 640..1215  (static UVs)
 */
#ifndef BORG_DRM_H
#define BORG_DRM_H

#include <stdint.h>

/* Allow borg_drm.h to be included before or after xf86drm.h. */
#ifndef DRM_IOWR
#  include <sys/ioctl.h>
#  define BORG_DRM_IOCTL_BASE 'd'
#  define DRM_IOCTL_BASE  BORG_DRM_IOCTL_BASE
#  define DRM_COMMAND_BASE 0x40
#  define DRM_IOW(nr, type)   _IOW(BORG_DRM_IOCTL_BASE, nr, type)
#  define DRM_IOWR(nr, type)  _IOWR(BORG_DRM_IOCTL_BASE, nr, type)
#endif

/* ---- UBO byte offsets ------------------------------------------------- */
#define BORG_UBO_MVP_BYTE_OFFSET    0u
#define BORG_UBO_POS_BYTE_OFFSET    64u   /* after 16 MVP floats            */
#define BORG_UBO_ATTR_BYTE_OFFSET   640u  /* after 16 + 36×4 = 160 floats  */
#define BORG_UBO_MIN_BYTES          1216u /* minimum valid UBO size         */
#define BORG_UBO_NUM_VERTS          36    /* expanded triangles (12 × 3)    */

/* ---- Geometry packet limits (firmware RX_GEOM_* must agree) ----------- */
#define BORG_GEOM_MAX_VERTS   16
#define BORG_GEOM_MAX_TRIS    12

/* ---- Texture (firmware BORGVK_TEX_DIM must agree) --------------------- */
#define BORG_TEX_DIM   64

/* ---- Driver-specific ioctl offsets ------------------------------------ */
#define DRM_BORG_GEM_CREATE   0x00
#define DRM_BORG_GEM_MMAP     0x01
#define DRM_BORG_SETUP        0x02   /* one-time mesh+texture upload        */
#define DRM_BORG_SUBMIT       0x03   /* per-frame MVP send                  */

/* Allocate a GEM buffer object backed by anonymous memory. */
struct drm_borg_gem_create {
   uint64_t size;     /* in:  allocation size in bytes           */
   uint32_t handle;   /* out: GEM handle                         */
   uint32_t pad;
};

/* Get the mmap offset for a GEM handle. Pass offset to mmap(drm_fd). */
struct drm_borg_gem_mmap {
   uint32_t handle;   /* in:  GEM handle                         */
   uint32_t pad;
   uint64_t offset;   /* out: mmap offset                        */
};

/* Upload mesh + texture to the FPGA (called once per power cycle).
 * The shim reads geometry from the UBO BO at BORG_UBO_POS/ATTR_BYTE_OFFSET
 * and texture (RGBA8 linear) from the tex BO at tex_offset.
 * A sentinel file prevents re-upload when the FPGA already has the data. */
struct drm_borg_setup {
   uint32_t ubo_handle;
   uint32_t tex_handle;
   uint64_t tex_offset;    /* byte offset of pixel data within tex BO */
   uint32_t tex_width;
   uint32_t tex_height;
};

/* Send the per-frame MVP (16 floats at UBO byte offset 0) to the FPGA. */
struct drm_borg_submit {
   uint32_t ubo_handle;
   uint32_t pad;
};

#define DRM_IOCTL_BORG_GEM_CREATE \
   DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GEM_CREATE, struct drm_borg_gem_create)
#define DRM_IOCTL_BORG_GEM_MMAP \
   DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GEM_MMAP,   struct drm_borg_gem_mmap)
#define DRM_IOCTL_BORG_SETUP \
   DRM_IOW (DRM_COMMAND_BASE + DRM_BORG_SETUP,       struct drm_borg_setup)
#define DRM_IOCTL_BORG_SUBMIT \
   DRM_IOW (DRM_COMMAND_BASE + DRM_BORG_SUBMIT,      struct drm_borg_submit)

#endif /* BORG_DRM_H */
