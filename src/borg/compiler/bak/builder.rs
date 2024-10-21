// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

use crate::ir::*;

pub trait SSABuilder {}

pub struct SSAInstrBuilder<'a> {
    alloc: &'a mut SSAValueAllocator,
}

impl<'a> SSAInstrBuilder<'a> {
    pub fn new(
        alloc: &'a mut SSAValueAllocator,
    ) ->Self {
        Self {
            alloc: alloc
        }
    }
}

impl<'a> SSABuilder for SSAInstrBuilder<'a> {}


