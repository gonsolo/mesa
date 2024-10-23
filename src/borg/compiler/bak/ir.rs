// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

extern crate bak_ir_proc;

use std::fmt;
use bak_ir_proc::*;
use compiler::as_slice::*;
use compiler::cfg::CFG;
use compiler::smallvec::SmallVec;
use std::ops::{Deref, DerefMut};

pub struct Function {
    pub ssa_alloc: SSAValueAllocator,
    pub blocks: CFG<BasicBlock>,
}

impl fmt::Display for Function {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    pub fn alloc(&mut self, file: RegFile) -> SSAValue {
        self.count += 1;
        SSAValue::new(file, self.count)
    }

    pub fn alloc_vec(&mut self, file: RegFile, comps: u8) -> SSARef {
        assert!(comps >= 1 && comps <= 4);
        let mut vec = [SSAValue::NONE; 4];
        for c in 0..comps {
            vec[usize::from(c)] = self.alloc(file);
        }
        vec[0..usize::from(comps)].try_into().unwrap()
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct SSARef {
    v: [SSAValue; 4],
}

impl SSARef {
    /// Returns a new SSA reference
    #[inline]
    fn new(comps: &[SSAValue]) -> SSARef {
        assert!(comps.len() > 0 && comps.len() <= 4);
        let mut r = SSARef {
            v: [SSAValue::NONE; 4],
        };
        for i in 0..comps.len() {
            r.v[i] = comps[i];
        }
        if comps.len() < 4 {
            r.v[3].packed = (comps.len() as u32).wrapping_neg();
        }
        r
    }

    pub fn comps(&self) -> u8 {
        if self.v[3].packed >= u32::MAX - 2 {
            self.v[3].packed.wrapping_neg() as u8
        } else {
            4
        }
    }

}

macro_rules! impl_ssa_ref_from_arr {
    ($n: expr) => {
        impl From<[SSAValue; $n]> for SSARef {
            fn from(comps: [SSAValue; $n]) -> Self {
                SSARef::new(&comps[..])
            }
        }
    };
}
impl_ssa_ref_from_arr!(1);
impl_ssa_ref_from_arr!(2);
impl_ssa_ref_from_arr!(3);
impl_ssa_ref_from_arr!(4);

impl Deref for SSARef {
    type Target = [SSAValue];

    fn deref(&self) -> &[SSAValue] {
        let comps = usize::from(self.comps());
        &self.v[..comps]
    }
}

impl DerefMut for SSARef {
    fn deref_mut(&mut self) -> &mut [SSAValue] {
        let comps = usize::from(self.comps());
        &mut self.v[..comps]
    }
}

impl TryFrom<&[SSAValue]> for SSARef {
    type Error = &'static str;

    fn try_from(comps: &[SSAValue]) -> Result<Self, Self::Error> {
        if comps.len() == 0 {
            Err("Empty vector")
        } else if comps.len() > 4 {
            Err("Too many vector components")
        } else {
            Ok(SSARef::new(comps))
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Src {
    //pub src_ref: SrcRef,
    //pub src_mod: SrcMod,
    //pub src_swizzle: SrcSwizzle,
}

impl Src {
    pub fn is_predicate(&self) -> bool {
        //self.src_ref.is_predicate()
        // TODO
        return false
    }
}

impl<T: Into<SrcRef>> From<T> for Src {
    fn from(_value: T) -> Src {
        Src {
            //src_ref: value.into(),
            //src_mod: SrcMod::None,
            //src_swizzle: SrcSwizzle::None,
        }
    }
}


#[derive(Clone, Copy)]
pub enum Dst {
    None,
    SSA(SSARef),
    // TODO Reg(RegRef),
}

impl<T: Into<SSARef>> From<T> for Dst {
    fn from(ssa: T) -> Dst {
        Dst::SSA(ssa.into())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SSAValue {
    packed: u32,
}

impl SSAValue {

    pub const NONE: Self = SSAValue { packed: 0 };

    pub fn new(file: RegFile, idx: u32) -> SSAValue {
        assert!(idx > 0 && idx < (1 << 29) - 2);
        let mut packed = idx;
        assert!(u8::from(file) < 8);
        packed |= u32::from(u8::from(file)) << 29;
        SSAValue { packed: packed }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegFile {
    /// The general-purpose register file
    ///
    /// General-purpose registers are 32 bits per SIMT channel.
    GPR = 0,

    /// The predicate reigster file
    ///
    /// Predicate registers are 1 bit per SIMT channel.
    Pred = 2,
}

#[repr(C)]
#[derive(SrcsAsSlice, DstsAsSlice)]
pub struct OpCopy {
    pub dst: Dst,
    pub src: Src,
}

pub trait SrcsAsSlice: AsSlice<Src, Attr = SrcType> {
    // TODO
}

impl<T: AsSlice<Src, Attr = SrcType>> SrcsAsSlice for T {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SrcType {
    SSA,
    GPR,
    ALU,
    F16,
    F16v2,
    F32,
    F64,
    I32,
    B32,
    Pred,
    Carry,
    Bar,
}

impl SrcType {
    const DEFAULT: SrcType = SrcType::GPR;
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DstType {
    Pred,
    GPR,
    F16,
    F16v2,
    F32,
    F64,
    Carry,
    Bar,
    Vec,
}

impl DstType {
    const DEFAULT: DstType = DstType::Vec;
}

// TODO #[derive(DisplayOp, DstsAsSlice, SrcsAsSlice, FromVariants)]
#[derive(DstsAsSlice, SrcsAsSlice, FromVariants)]
pub enum Op {
    Copy(OpCopy),
}

pub struct Instr {
    //pub pred: Pred,
    //pub op: Op,
    //pub deps: InstrDeps,
}

impl Instr {

    pub fn new(_op: impl Into<Op>) -> Instr {
        Instr {
            //op: op.into(),
            //pred: true.into(),
            //deps: InstrDeps::new(),
        }
    }

    pub fn new_boxed(op: impl Into<Op>) -> Box<Self> {
        Box::new(Instr::new(op))
    }
}

pub type MappedInstrs = SmallVec<Box<Instr>>;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum SrcRef {
    Zero,
    True,
    False,
    //Imm32(u32),
    //CBuf(CBufRef),
    //SSA(SSARef),
    //Reg(RegRef),
}

impl From<u32> for SrcRef {
    fn from(u: u32) -> SrcRef {
        if u == 0 {
            SrcRef::Zero
        } else {
            panic!("Unimplemented")
            //SrcRef::Imm32(u)
        }
    }
}

impl From<RegFile> for u8 {
    fn from(value: RegFile) -> u8 {
        value as u8
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Label {
    idx: u32
}
pub struct BasicBlock {
    pub label: Label,
    pub instrs: Vec<Box<Instr>>,
}

pub struct LabelAllocator {
    count: u32,
}

impl LabelAllocator {
    pub fn new() -> LabelAllocator {
        LabelAllocator { count: 0 }
    }

    pub fn alloc(&mut self) -> Label {
        let idx = self.count;
        self.count += 1;
        Label { idx: idx }
    }
}
