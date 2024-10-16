// Copyright © 2024 Andreas Wendleder
// SPDX-License-Identifier: MIT

use bak_bindings::*;

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

// From nouveau
macro_rules! offset_of {
    ($Struct:path, $field:ident) => {{
        fn offset() -> usize {
            let u = std::mem::MaybeUninit::<$Struct>::uninit();
            // Use pattern-matching to avoid accidentally going through Deref.
            let &$Struct { $field: ref f, .. } = unsafe { &*u.as_ptr() };
            let o =
                (f as *const _ as usize).wrapping_sub(&u as *const _ as usize);
            assert!((0..=std::mem::size_of_val(&u)).contains(&o));
            o
        }
        offset()
    }};
}

pub struct ExecListIter<'a, T> {
    n: &'a exec_node,
    offset: usize,
    _marker: PhantomData<T>,
}

impl<'a, T> ExecListIter<'a, T> {
    fn new(l: &'a exec_list, offset: usize) -> Self {
        Self {
            n: &l.head_sentinel,
            offset: offset,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a> Iterator for ExecListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.n = unsafe { &*self.n.next };
        if self.n.next.is_null() {
            None
        } else {
            let t: *const c_void = (self.n as *const exec_node).cast();
            Some(unsafe { &*(t.sub(self.offset).cast()) })
        }
    }
}

pub trait NirBlock {
    fn iter_instr_list(&self) -> ExecListIter<nir_instr>;
}

impl NirBlock for nir_block {
    fn iter_instr_list(&self) -> ExecListIter<nir_instr> {
        ExecListIter::new(&self.instr_list, offset_of!(nir_instr, node))
    }
}

pub trait NirCfNode {
    fn as_block(&self) -> Option<&nir_block>;
}

impl NirCfNode for nir_cf_node {
    fn as_block(&self) -> Option<&nir_block> {
        if self.type_ == nir_cf_node_block {
            Some(unsafe { &*(self as *const nir_cf_node as *const nir_block) })
        } else {
            None
        }
    }
}

pub trait NirFunction {
    fn get_impl(&self) -> Option<&nir_function_impl>;
}

impl NirFunction for nir_function {
    fn get_impl(&self) -> Option<&nir_function_impl> {
        unsafe { self.impl_.as_ref() }
    }
}

pub trait NirFunctionImpl {
    fn iter_body(&self) -> ExecListIter<nir_cf_node>;
    fn end_block(&self) -> &nir_block;
}

impl NirFunctionImpl for nir_function_impl {

    fn iter_body(&self) -> ExecListIter<nir_cf_node> {
        ExecListIter::new(&self.body, offset_of!(nir_cf_node, node))
    }

    fn end_block(&self) -> &nir_block {
        unsafe { NonNull::new(self.end_block).unwrap().as_ref() }
    }

}

pub trait NirShader {
    fn iter_functions(&self) -> ExecListIter<nir_function>;
}   

impl NirShader for nir_shader {
    fn iter_functions(&self) -> ExecListIter<nir_function> {
        ExecListIter::new(&self.functions, offset_of!(nir_function, node))
    }
}

