/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#ifndef BORGC_FLAGS_H
#define BORGC_FLAGS_H

#define BORGC_VS_FETCH              0x80000000u
#define BORGC_VS_POINT_QUADS        0x40000000u
#define BORGC_VS_PNTC_VARYINGS      0x20000000u
#define BORGC_VS_LINES_BIT          0x10000000u
#define BORGC_VS_INSTANCED(loc)     (1u << (loc))

#define BORGC_FS_BYTES_PACKED       0x1u
#define BORGC_FS_R8_UNORM           0x2u
#define BORGC_FS_ALPHA_OUTPUT       0x4u
#define BORGC_FS_HALF               0x8u
#define BORGC_FS_PACKED             0x10u
#define BORGC_FS_HALF1              0x20u
#define BORGC_FS_HALF3              0x40u
#define BORGC_FS_HALF4              0x80u
#define BORGC_FS_MULTI_TARGET       0x100u
#define BORGC_FS_TARGETS_SHIFT      9
#define BORGC_FS_TARGET_CLASS_SHIFT 12
#define BORGC_FS_PACKED_KIND_SHIFT  16
#define BORGC_FS_PACKED_SIZE_SHIFT  19
#define BORGC_FS_PACKED_N_SHIFT     21
#define BORGC_FS_PACKED_SWAP_SHIFT  23
#define BORGC_FS_PNTC_BASE_SHIFT    24

#endif
