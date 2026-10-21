// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The arrays and optional pointers of the Vulkan structs borgvk reads.

use crate::entrypoint::members;
use crate::ffi::*;

members! {
    trait CopyBufferInfo for VkCopyBufferInfo2 { regions: pRegions[regionCount] -> VkBufferCopy2 }
    trait CopyImageInfo for VkCopyImageInfo2 { regions: pRegions[regionCount] -> VkImageCopy2 }
    trait CopyBufferToImageInfo for VkCopyBufferToImageInfo2 { regions: pRegions[regionCount] -> VkBufferImageCopy2 }
    trait CopyImageToBufferInfo for VkCopyImageToBufferInfo2 { regions: pRegions[regionCount] -> VkBufferImageCopy2 }
    trait BlitImageInfo for VkBlitImageInfo2 { regions: pRegions[regionCount] -> VkImageBlit2 }
    trait ResolveImageInfo for VkResolveImageInfo2 { regions: pRegions[regionCount] -> VkImageResolve2 }

    trait RenderingInfo for VkRenderingInfo {
        color_attachments: pColorAttachments[colorAttachmentCount] -> VkRenderingAttachmentInfo,
        depth_attachment: pDepthAttachment -> VkRenderingAttachmentInfo,
        stencil_attachment: pStencilAttachment -> VkRenderingAttachmentInfo,
    }

    trait VertexInputState for VkPipelineVertexInputStateCreateInfo {
        bindings: pVertexBindingDescriptions[vertexBindingDescriptionCount] -> VkVertexInputBindingDescription,
        attributes: pVertexAttributeDescriptions[vertexAttributeDescriptionCount] -> VkVertexInputAttributeDescription,
    }
    trait ColorBlendState for VkPipelineColorBlendStateCreateInfo {
        attachments: pAttachments[attachmentCount] -> VkPipelineColorBlendAttachmentState,
    }
    trait DynamicState for VkPipelineDynamicStateCreateInfo { dynamic_states: pDynamicStates[dynamicStateCount] -> VkDynamicState }
    trait GraphicsPipelineCreateInfo for VkGraphicsPipelineCreateInfo { dynamic_state: pDynamicState -> VkPipelineDynamicStateCreateInfo }

    trait DescriptorSetLayoutCreateInfo for VkDescriptorSetLayoutCreateInfo { bindings: pBindings[bindingCount] -> VkDescriptorSetLayoutBinding }
}

// The clear unions: every member is plain data of the union's size, so any of them may be read.
pub trait ClearValue {
    fn color(&self) -> &VkClearColorValue;
    fn depth_stencil(&self) -> VkClearDepthStencilValue;
}

impl ClearValue for VkClearValue {
    fn color(&self) -> &VkClearColorValue {
        unsafe { &self.color }
    }

    fn depth_stencil(&self) -> VkClearDepthStencilValue {
        unsafe { self.depthStencil }
    }
}

pub trait ClearColorValue {
    fn float32(&self) -> [f32; 4];
    fn int32(&self) -> [i32; 4];
    fn uint32(&self) -> [u32; 4];
}

impl ClearColorValue for VkClearColorValue {
    fn float32(&self) -> [f32; 4] {
        unsafe { self.float32 }
    }

    fn int32(&self) -> [i32; 4] {
        unsafe { self.int32 }
    }

    fn uint32(&self) -> [u32; 4] {
        unsafe { self.uint32 }
    }
}
