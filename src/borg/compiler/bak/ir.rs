// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use std::cmp::max;
use std::fmt;

pub struct Function {
    // TODO
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
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

impl fmt::Display for Shader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for func in &self.functions {
            write!(f, "{}", func)?;
        }
        Ok(())
    }
}

pub struct SSAValueAllocator {
    count: u32,
}

impl SSAValueAllocator {
    pub fn new() -> SSAValueAllocator {
        SSAValueAllocator { count: 0 }
    }
}
