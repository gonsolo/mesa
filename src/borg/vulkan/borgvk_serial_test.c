/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * Unit test for the push-constant wire packet (0xB2).
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

int
main(void)
{
   test_well_formed_range();
   test_full_range();
   test_rejected_ranges();

   if (failures) {
      fprintf(stderr, "%d check(s) failed\n", failures);
      return 1;
   }
   printf("borgvk push-constant packet: all checks passed\n");
   return 0;
}
