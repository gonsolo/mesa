/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * Unit tests for wire packets: push constants (0xB2) and geometry (0xAE).
 *
 * This exists because the simulator-side test
 * (simulation/verilator/main.cpp --push-const-test) BUILDS its own packets
 * rather than calling this sender, so on its own it proves the firmware
 * parses a correctly-formed packet -- not that borgvk emits one. A byte-order
 * slip or an offset miscalculation here would sail straight past it.
 *
 * So the expected bytes below are written out literally, in the same shape
 * the simulator test constructs, and compared against what the real sender
 * produces through the capture transport. The two tests together cover the
 * host->firmware path end to end: this one pins the bytes borgvk emits, and
 * the simulator one proves the firmware stages exactly those bytes correctly.
 */

#include "borgvk_private.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PKT_LEN     132   /* 1 marker + 1 off + 1 n + 32*4 payload + 1 csum */
#define MAX_WORDS   32

static int failures = 0;

static void
check(bool cond, const char *what)
{
   if (!cond) {
      fprintf(stderr, "FAIL: %s\n", what);
      failures++;
   }
}

/* Drive the real sender and hand back whatever it put on the wire. */
static uint8_t *
emit(uint32_t offset, uint32_t size, const void *values, size_t *out_len)
{
   borgvk_transport_capture_begin();
   borgvk_serial_send_push_constants(offset, size, values);
   return borgvk_transport_capture_end(out_len);
}

static void
test_well_formed_range(void)
{
   /* Word offset 8, four words -- deliberately the same range the simulator
    * test's second packet uses, so the two tests pin the same bytes. */
   const uint32_t words[4] = {
      0xB2B20000u, 0xB2B20001u, 0xB2B20002u, 0xB2B20003u,
   };

   size_t len = 0;
   uint8_t *pkt = emit(8 * 4, sizeof(words), words, &len);

   check(pkt != NULL, "a valid range emits a packet");
   if (!pkt)
      return;

   check(len == PKT_LEN, "packet is the fixed 132-byte length");
   check(pkt[0] == 0xB2, "marker is 0xB2 (0xB1 is the serial-reload trigger)");
   check(pkt[1] == 8, "byte offset 32 is sent as word offset 8");
   check(pkt[2] == 4, "byte size 16 is sent as 4 words");

   /* Payload is little-endian, at the head of the buffer regardless of the
    * offset field -- the offset says where it lands, not where it sits. */
   for (unsigned i = 0; i < 4; i++) {
      uint32_t v = words[i];
      bool ok = pkt[3 + i * 4 + 0] == (uint8_t)(v & 0xff) &&
                pkt[3 + i * 4 + 1] == (uint8_t)((v >> 8) & 0xff) &&
                pkt[3 + i * 4 + 2] == (uint8_t)((v >> 16) & 0xff) &&
                pkt[3 + i * 4 + 3] == (uint8_t)((v >> 24) & 0xff);
      check(ok, "payload word is little-endian at the head of the packet");
   }

   /* Everything past the supplied words must be zero: the firmware reads a
    * constant byte count, so the tail is padding it checksums but ignores. */
   bool padded = true;
   for (int i = 3 + 4 * 4; i < PKT_LEN - 1; i++)
      if (pkt[i] != 0)
         padded = false;
   check(padded, "unused payload is zero-padded to the fixed length");

   uint8_t csum = 0;
   for (int i = 1; i < PKT_LEN - 1; i++)
      csum ^= pkt[i];
   check(csum == pkt[PKT_LEN - 1],
         "checksum is XOR over bytes 1..len-2, matching the firmware's span");

   free(pkt);
}

static void
test_full_range(void)
{
   uint32_t words[MAX_WORDS];
   for (unsigned i = 0; i < MAX_WORDS; i++)
      words[i] = 0xA5A50000u + i;

   size_t len = 0;
   uint8_t *pkt = emit(0, sizeof(words), words, &len);

   check(pkt != NULL, "the full 128-byte range is accepted");
   if (!pkt)
      return;
   check(len == PKT_LEN, "full range still fits the fixed length");
   check(pkt[1] == 0 && pkt[2] == MAX_WORDS, "full range is 32 words at offset 0");
   free(pkt);
}

static void
test_rejected_ranges(void)
{
   const uint32_t v[2] = { 1, 2 };
   size_t len;
   uint8_t *pkt;

   /* Vulkan requires offset and size to be multiples of 4
    * (VUID-vkCmdPushConstants-offset-00368 / -size-00369). The word-index
    * mapping depends on it, so the sender checks rather than assuming. */
   len = 0;
   pkt = emit(2, 8, v, &len);
   check(len == 0, "an unaligned offset emits nothing");
   free(pkt);

   len = 0;
   pkt = emit(0, 6, v, &len);
   check(len == 0, "an unaligned size emits nothing");
   free(pkt);

   /* Past the advertised maxPushConstantsSize of 128 bytes. */
   len = 0;
   pkt = emit(0, (MAX_WORDS + 1) * 4, v, &len);
   check(len == 0, "a range beyond 128 bytes emits nothing");
   free(pkt);

   len = 0;
   pkt = emit(MAX_WORDS * 4, 4, v, &len);
   check(len == 0, "an offset at the end of the range emits nothing");
   free(pkt);

   len = 0;
   pkt = emit(0, 0, v, &len);
   check(len == 0, "a zero-size push emits nothing");
   free(pkt);

   len = 0;
   pkt = emit(0, 4, NULL, &len);
   check(len == 0, "a NULL value pointer emits nothing");
   free(pkt);
}

/* 0xAE geometry: positions and UVs are shader datapath values, so they must
 * go out as float32 bits (the FP16 packing they used to have reads as
 * denormals on the FP32 datapath). Layout mirrors borg_kernel.c RX_GEOM_*. */
#define GEOM_MAX_VERTS 16
#define GEOM_MAX_TRIS  12
#define GEOM_PKT_LEN   (1 + 2 + GEOM_MAX_VERTS * 12 + GEOM_MAX_TRIS * 3 + \
                        GEOM_MAX_TRIS * 24 + 1)

static bool
f32_at(const uint8_t *p, uint32_t bits)
{
   return p[0] == (uint8_t)(bits & 0xff) && p[1] == (uint8_t)((bits >> 8) & 0xff) &&
          p[2] == (uint8_t)((bits >> 16) & 0xff) && p[3] == (uint8_t)(bits >> 24);
}

static void
test_geometry_packet(void)
{
   const float verts[2 * 3] = { -1.0f, 0.5f, 0.1f,   1.0f, -0.25f, 3.0f };
   const uint8_t idx[1 * 3] = { 0, 1, 1 };
   const float uv[1 * 3 * 2] = { 0.0f, 1.0f,   0.75f, 0.5f,   0.1f, 0.9f };

   borgvk_transport_capture_begin();
   borgvk_serial_send_geom(verts, 2, idx, uv, 1);
   size_t len = 0;
   uint8_t *pkt = borgvk_transport_capture_end(&len);

   check(pkt != NULL && len == GEOM_PKT_LEN, "geometry packet is the fixed 520-byte length");
   if (!pkt || len != GEOM_PKT_LEN) {
      free(pkt);
      return;
   }
   check(pkt[0] == 0xAE && pkt[1] == 2 && pkt[2] == 1, "marker, nverts, ntris");

   const int vbase = 3;
   const int ibase = vbase + GEOM_MAX_VERTS * 12;
   const int ubase = ibase + GEOM_MAX_TRIS * 3;
   check(f32_at(&pkt[vbase + 0], 0xBF800000u), "position -1.0f as float32 LE");
   check(f32_at(&pkt[vbase + 4], 0x3F000000u), "position 0.5f as float32 LE");
   check(f32_at(&pkt[vbase + 8], 0x3DCCCCCDu), "position 0.1f keeps all 32 bits");
   check(f32_at(&pkt[vbase + 20], 0x40400000u), "second vertex z 3.0f");
   check(pkt[ibase] == 0 && pkt[ibase + 1] == 1 && pkt[ibase + 2] == 1, "indices");
   check(f32_at(&pkt[ubase + 8], 0x3F400000u), "UV 0.75f as float32 LE");
   check(f32_at(&pkt[ubase + 12], 0x3F000000u), "UV 0.5f as float32 LE");

   bool padded = true;
   for (int i = vbase + 2 * 12; i < ibase; i++) padded &= pkt[i] == 0;
   for (int i = ubase + 6 * 4; i < GEOM_PKT_LEN - 1; i++) padded &= pkt[i] == 0;
   check(padded, "unused vertex and UV slots are zero");

   uint8_t csum = 0;
   for (int i = 1; i < GEOM_PKT_LEN - 1; i++)
      csum ^= pkt[i];
   check(csum == pkt[GEOM_PKT_LEN - 1], "geometry checksum is XOR over bytes 1..len-2");
   free(pkt);
}

/* 0xAF: the firmware (borg_kernel.c) reads y at [1], the sampler descriptor
 * as four little-endian words at [2..17], the RGBA8 texels from [18]. */
static void
test_texture_row_packet(void)
{
   const int len_expected = 1 + 1 + 16 + BORGVK_TEX_DIM * 4 + 1;
   uint8_t rgba[BORGVK_TEX_DIM * 4];
   for (int i = 0; i < BORGVK_TEX_DIM * 4; i++)
      rgba[i] = (uint8_t)(i * 7);
   const uint32_t sampler[4] = { 0x000A0493u, 0x3F800000u, 0u, 0x447A0000u };

   borgvk_transport_capture_begin();
   borgvk_serial_send_tex_row(5, rgba, sampler);
   size_t len = 0;
   uint8_t *pkt = borgvk_transport_capture_end(&len);

   check(pkt != NULL && len == (size_t)len_expected, "texture row packet is 275 bytes");
   if (!pkt || len != (size_t)len_expected) {
      free(pkt);
      return;
   }
   check(pkt[0] == 0xAF && pkt[1] == 5, "marker and row");
   for (int w = 0; w < 4; w++)
      check(f32_at(&pkt[2 + w * 4], sampler[w]), "sampler descriptor word, LE");
   check(memcmp(&pkt[18], rgba, sizeof(rgba)) == 0, "RGBA8 texels follow the sampler");
   uint8_t csum = 0;
   for (int i = 1; i < len_expected - 1; i++)
      csum ^= pkt[i];
   check(csum == pkt[len_expected - 1], "texture checksum is XOR over bytes 1..len-2");
   free(pkt);
}

int
main(void)
{
   test_texture_row_packet();
   test_geometry_packet();
   test_well_formed_range();
   test_full_range();
   test_rejected_ranges();

   if (failures) {
      fprintf(stderr, "%d check(s) failed\n", failures);
      return 1;
   }
   printf("borgvk wire packets: all checks passed\n");
   return 0;
}
