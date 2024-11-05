// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

extern crate bak_ir_proc;

use std::fmt;
use bak_ir_proc::*;
use compiler::as_slice::*;
use compiler::cfg::CFG;
use compiler::smallvec::SmallVec;
use std::ops::{Deref, DerefMut, Range};
use std::fmt::Write;

pub struct Function {
    pub ssa_alloc: SSAValueAllocator,
    pub blocks: CFG<BasicBlock>,
}

impl Function {

   pub fn map_instrs(
          &mut self,
          mut map: impl FnMut(Box<Instr>, &mut SSAValueAllocator) -> MappedInstrs,
      ) {
          let alloc = &mut self.ssa_alloc;
          for b in &mut self.blocks {
              b.map_instrs(|i| map(i, alloc));
          }
      }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.blocks {
            for i in &b.instrs {
                write!(f, "{}", i)?;
            }
        }
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

impl Shader {
   pub fn map_instrs(
          &mut self,
          mut map: impl FnMut(Box<Instr>, &mut SSAValueAllocator) -> MappedInstrs,
      ) {
          for f in &mut self.functions {
              f.map_instrs(&mut map);
          }
      }

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

    pub fn file(&self) -> Option<RegFile> {
        let comps = usize::from(self.comps());
        let file = self.v[0].file();
        for i in 1..comps {
            if self.v[i].file() != file {
                return None;
            }
        }
        Some(file)
    }

}

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

impl TryFrom<Vec<SSAValue>> for SSARef {
    type Error = &'static str;

    fn try_from(comps: Vec<SSAValue>) -> Result<Self, Self::Error> {
        SSARef::try_from(&comps[..])
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

impl From<SSAValue> for SSARef { 
    fn from(val: SSAValue) -> Self {
        [val].into()
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

pub trait HasRegFile {
    fn file(&self) -> RegFile;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegRef {
    packed: u32,
}

impl RegRef {

    pub const MAX_IDX: u32 = (1 << 26) - 1;

    fn zero_idx(file: RegFile) -> u32 {
        match file {
            RegFile::GPR => 255,
        }
    }

    pub fn new(file: RegFile, base_idx: u32, comps: u8) -> RegRef {
        assert!(base_idx <= Self::MAX_IDX);
        let mut packed = base_idx;
        assert!(comps > 0 && comps <= 8);
        packed |= u32::from(comps - 1) << 26;
        assert!(u8::from(file) < 8);
        packed |= u32::from(u8::from(file)) << 29;
        RegRef { packed: packed }
    }

    pub fn base_idx(&self) -> u32 {
        self.packed & 0x03ffffff
    }

    pub fn idx_range(&self) -> Range<u32> {
        let start = self.base_idx();
        let end = start + u32::from(self.comps());
        start..end
    }

    pub fn comps(&self) -> u8 {
        (((self.packed >> 26) & 0x7) + 1).try_into().unwrap()
    }
}

impl HasRegFile for RegRef {
    fn file(&self) -> RegFile {
        ((self.packed >> 29) & 0x7).try_into().unwrap()
    }
}

impl fmt::Display for RegRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file().fmt_prefix(), self.base_idx())?;
        if self.comps() > 1 {
            write!(f, "..{}", self.idx_range().end)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Src {
    pub src_ref: SrcRef,
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
    fn from(value: T) -> Src {
        Src {
            src_ref: value.into(),
            //src_mod: SrcMod::None,
            //src_swizzle: SrcSwizzle::None,
        }
    }
}

impl fmt::Display for Src {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.src_ref)
    }
}


#[derive(Clone, Copy)]
pub enum Dst {
    None,
    SSA(SSARef),
    Reg(RegRef),
}

impl Dst {
    pub fn is_none(&self) -> bool {
        matches!(self, Dst::None)
    }

    pub fn as_reg(&self) -> Option<&RegRef> {

        println!("as_reg: self: {}", self);
        match self {
            Dst::Reg(_r) => {
                println!("Dst::Reg");
            },
            Dst::SSA(_s) => {
                println!("Dst::SSA");
            },
            _ => {
                println!("None");
            },

        }

        match self {
            Dst::Reg(r) => Some(r),
            _ => None,
        }
    }
}

impl fmt::Display for Dst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dst::None => write!(f, "null")?,
            Dst::SSA(v) => v.fmt(f)?,
            Dst::Reg(r) => r.fmt(f)?,
        }
        Ok(())
    }
}

impl From<RegRef> for Dst {
    fn from(reg: RegRef) -> Dst {
        Dst::Reg(reg)
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

    pub fn idx(&self) -> u32 {
        self.packed & 0x1fffffff
    }

    fn fmt_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file().fmt_prefix(), self.idx())
    }
}

impl HasRegFile for SSAValue {
    fn file(&self) -> RegFile {
        RegFile::try_from(self.packed >> 29).unwrap()
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
}

impl RegFile {
   fn fmt_prefix(&self) -> &'static str {
        match self {
              RegFile::GPR => "r",
          }
      }
}

impl TryFrom<u32> for RegFile {
    type Error = &'static str;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RegFile::GPR),
            _ => Err("Invalid register file number"),
        }
    }
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
                let mut s = String::new();
                write!(s, "{}", Fmt(|f| self.fmt_dsts(f)))?;
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

#[repr(C)]
#[derive(SrcsAsSlice, DstsAsSlice)]
pub struct OpMov {
    #[dst_type(GPR)]
    pub dst: Dst,

    #[src_type(ALU)]
    pub src: Src,
}

impl DisplayOp for OpMov {
    fn fmt_op(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mov {}", self.src)
    }
}
impl_display_for_op!(OpMov);

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
    Mov(OpMov),
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

    pub fn dsts_mut(&mut self) -> &mut [Dst] {
        self.op.dsts_as_mut_slice()
    }
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " {}", self.op)
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

impl fmt::Display for SrcRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SrcRef::Zero => write!(f, "rZ"),
            SrcRef::True => write!(f, "pT"),
            SrcRef::False => write!(f, "pF"),
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

impl BasicBlock {

   pub fn map_instrs(
          &mut self,
          mut map: impl FnMut(Box<Instr>) -> MappedInstrs,
      ) {
          let mut instrs = Vec::new();
          for i in self.instrs.drain(..) {
              match map(i) {
                  MappedInstrs::None => (),
                  MappedInstrs::One(i) => {
                      instrs.push(i);
                  }
                  MappedInstrs::Many(mut v) => {
                      instrs.append(&mut v);
                  }
              }
          }
          self.instrs = instrs;
      }
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

