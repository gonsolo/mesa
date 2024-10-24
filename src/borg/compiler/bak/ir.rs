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

impl fmt::Display for SSARef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.comps() == 1 {
            write!(f, "{}", self[0])
        } else {
            write!(f, "{{")?;
            for (i, v) in self.iter().enumerate() {
                if i != 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", v)?;
            }
            write!(f, "}}")
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

impl fmt::Display for Src {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TODO")
    }
}


#[derive(Clone, Copy)]
pub enum Dst {
    None,
    SSA(SSARef),
    // TODO Reg(RegRef),
}

impl Dst {
    pub fn is_none(&self) -> bool {
        matches!(self, Dst::None)
    }
}

impl fmt::Display for Dst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dst::None => write!(f, "null")?,
            Dst::SSA(v) => v.fmt(f)?,
            //Dst::Reg(r) => r.fmt(f)?,
        }
        Ok(())
    }
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

    fn fmt_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO write!(f, "{}{}", self.file().fmt_prefix(), self.idx())
        write!(f, "TODO: SSAValue fmt_plain")
    }

}

impl fmt::Display for SSAValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%")?;
        self.fmt_plain(f)
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

macro_rules! impl_display_for_op {
    ($op: ident) => {
        impl fmt::Display for $op {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let s = String::new();
                // TODO write!(s, "{}", Fmt(|f| self.fmt_dsts(f)))?;
                if !s.is_empty() {
                    write!(f, "{} = ", s)?;
                }
                self.fmt_op(f)
            }
        }
    };
}

impl DisplayOp for OpCopy {
    fn fmt_op(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "copy {}", self.src)
    }
}
impl_display_for_op!(OpCopy);

fn fmt_dst_slice(f: &mut fmt::Formatter<'_>, dsts: &[Dst]) -> fmt::Result {
    if dsts.is_empty() {
        return Ok(());
    }

    // Figure out the last non-null dst
    //
    // Note: By making the top inclusive and starting at 0, we ensure that
    // at least one dst always gets printed.
    let mut last_dst = 0;
    for (i, dst) in dsts.iter().enumerate() {
        if !dst.is_none() {
            last_dst = i;
        }
    }

    for i in 0..(last_dst + 1) {
        if i != 0 {
            write!(f, " ")?;
        }
        write!(f, "{}", &dsts[i])?;
    }
    Ok(())
}

pub type SrcTypeList = AttrList<SrcType>;

pub trait SrcsAsSlice: AsSlice<Src, Attr = SrcType> {
    fn srcs_as_slice(&self) -> &[Src] {
        self.as_slice()
    }

    fn srcs_as_mut_slice(&mut self) -> &mut [Src] {
        self.as_mut_slice()
    }

    fn src_types(&self) -> SrcTypeList {
        self.attrs()
    }

    fn src_idx(&self, src: &Src) -> usize {
        let r = self.srcs_as_slice().as_ptr_range();
        assert!(r.contains(&(src as *const Src)));
        unsafe { (src as *const Src).offset_from(r.start) as usize }
    }
}

impl<T: AsSlice<Src, Attr = SrcType>> SrcsAsSlice for T {}

pub type DstTypeList = AttrList<DstType>;

pub trait DstsAsSlice: AsSlice<Dst, Attr = DstType> {
    fn dsts_as_slice(&self) -> &[Dst] {
        self.as_slice()
    }

    fn dsts_as_mut_slice(&mut self) -> &mut [Dst] {
        self.as_mut_slice()
    }

    // Currently only used by test code
    #[allow(dead_code)]
    fn dst_types(&self) -> DstTypeList {
        self.attrs()
    }

    fn dst_idx(&self, dst: &Dst) -> usize {
        let r = self.dsts_as_slice().as_ptr_range();
        assert!(r.contains(&(dst as *const Dst)));
        unsafe { (dst as *const Dst).offset_from(r.start) as usize }
    }
}

impl<T: AsSlice<Dst, Attr = DstType>> DstsAsSlice for T {}

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

pub trait DisplayOp: DstsAsSlice {

    fn fmt_dsts(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_dst_slice(f, self.dsts_as_slice())
    }

    fn fmt_op(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;

}

pub struct Fmt<F>(pub F)
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result;

        impl<F> fmt::Display for Fmt<F>
        where
            F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (self.0)(f)
    }
}


#[derive(DisplayOp, DstsAsSlice, SrcsAsSlice, FromVariants)]
pub enum Op {
    Copy(OpCopy),
}
impl_display_for_op!(Op);

pub struct Instr {
    //pub pred: Pred,
    pub op: Op,
    //pub deps: InstrDeps,
}

impl Instr {

    pub fn new(op: impl Into<Op>) -> Instr {
        Instr {
            op: op.into(),
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

