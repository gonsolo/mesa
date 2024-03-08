/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include "borg_entrypoints.h"

VKAPI_ATTR void VKAPI_CALL
borg_CmdDispatchBase(VkCommandBuffer commandBuffer,
                     uint32_t baseGroupX,
                     uint32_t baseGroupY,
                     uint32_t baseGroupZ,
                     uint32_t groupCountX,
                     uint32_t groupCountY,
                     uint32_t groupCountZ)
{
   // TODO
}
