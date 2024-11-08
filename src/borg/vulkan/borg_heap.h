/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#ifndef BORG_HEAP_H
#define BORG_HEAP_H

#include "borg_private.h"

#include "stddef.h"
#include "stdint.h"

struct borg_device;

struct borg_heap {

};


VkResult borg_heap_upload(struct borg_device *dev, struct borg_heap *heap,
                         const void *data, size_t size, uint64_t *addr_out);

#endif

