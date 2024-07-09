// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT


pub struct Function {
    // TODO
}

pub struct PhiAllocator {
    count: u32,
}

impl PhiAllocator {
    pub fn new() -> PhiAllocator {
        PhiAllocator { count: 0 }
    }
}

pub struct Shader {
    //pub info: ShaderInfo,
    pub functions: Vec<Function>,
}

pub struct SSAValueAllocator {
    count: u32,
}

impl SSAValueAllocator {
    pub fn new() -> SSAValueAllocator {
        SSAValueAllocator { count: 0 }
    }
}
