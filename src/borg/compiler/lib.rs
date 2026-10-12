// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(non_upper_case_globals)]

mod compute;
mod config;
mod corpus;
mod emit;
mod fixed;
mod encode;
mod epilogue;
mod isel;
mod opt;
mod regalloc;
mod select;
mod state;
mod survey;

use compiler::bindings::*;
use compiler::nir::AsDef;
use config::Flags;
use fixed::{allocate, fixed_regs};
use emit::{build_blob, encode_program, Encoded};
use isel::resolve_vm;
use opt::{dce, fuse_fmadd, schedule_for_pressure};
use state::Cx;
use survey::{survey, Survey};
use std::env;

fn packed_layout(flags: Flags) -> (u32, Vec<(usize, u32, u32)>) {
    let vfetch = flags.0;
    let kind = (vfetch >> 16) & 7;
    if (vfetch >> 19) & 3 == 3 {
        let id = kind | ((vfetch >> 21) & 7) << 3;
        let l: &[(usize, u32, u32)] = match id {
            0 => &[(0, 10, 5), (1, 5, 5), (2, 0, 5), (3, 15, 1)],    // A1R5G5B5
            1 => &[(0, 11, 5), (1, 6, 5), (2, 1, 5), (3, 0, 1)],     // R5G5B5A1
            2 => &[(2, 11, 5), (1, 6, 5), (0, 1, 5), (3, 0, 1)],     // B5G5R5A1
            3 => &[(0, 12, 4), (1, 8, 4), (2, 4, 4), (3, 0, 4)],     // R4G4B4A4
            4 => &[(2, 12, 4), (1, 8, 4), (0, 4, 4), (3, 0, 4)],     // B4G4R4A4
            5 => &[(3, 12, 4), (0, 8, 4), (1, 4, 4), (2, 0, 4)],     // A4R4G4B4
            6 => &[(3, 12, 4), (2, 8, 4), (1, 4, 4), (0, 0, 4)],     // A4B4G4R4
            7 => &[(0, 0, 10), (1, 10, 10), (2, 20, 10), (3, 30, 2)],  // A2B10G10R10
            8 => &[(2, 0, 10), (1, 10, 10), (0, 20, 10), (3, 30, 2)],  // A2R10G10B10
            9 => &[(0, 0, 5), (1, 5, 6), (2, 11, 5)],                // B5G6R5
            11 => &[(0, 11, 5), (1, 5, 6), (2, 0, 5)],               // R5G6B5
            _ => &[(0, 4, 4), (1, 0, 4)],                            // R4G4
        };
        return (0, l.to_vec());
    }
    let bits: u32 = match (vfetch >> 19) & 3 { 0 => 8, 1 => 16, _ => 32 };
    let n = ((vfetch >> 21) & 3) as usize + 1;
    let swap = (vfetch >> 23) & 1 != 0;
    (kind, (0..n).map(|k| (if swap && k < 3 { 2 - k } else { k }, k as u32 * bits, bits)).collect())
}

/// Window words a fragment shader may ask for before the pool pass sorts them out.
const FRAG_CONST_MAX: u32 = 11 + 64;
const FRAG_POOL_SLOT: u32 = 40;
thread_local! {
    static LAST_FRAG_POOL: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn set_frag_pool(pool: Vec<u32>) {
    LAST_FRAG_POOL.with(|p| *p.borrow_mut() = pool);
}

#[no_mangle]
pub unsafe extern "C" fn borgc_last_frag_pool(out: *mut u32, cap: u32) -> u32 {
    LAST_FRAG_POOL.with(|p| {
        let p = p.borrow();
        let n = p.len().min(cap as usize);
        for (i, &w) in p.iter().take(n).enumerate() {
            *out.add(i) = w;
        }
        n as u32
    })
}

pub(crate) const NO_DST: u32 = u32::MAX;

pub(crate) struct CfMark {
    pub(crate) mnem: &'static str,
    /// Condition def index, for EXPUSH only.
    pub(crate) cond: Option<u32>,
}

pub(crate) struct BorgInstr {
    pub(crate) mnem: &'static str,
    pub(crate) dst: u32,
    pub(crate) srcs: Vec<u32>,
    pub(crate) swz: Vec<u8>,
}

fn decompose_vertex_offset(
    vec_map: &std::collections::HashMap<u32, Vec<(u32, u8)>>,
    prod: &std::collections::HashMap<u32, (nir_op, Vec<(u32, u8)>)>,
    consts: &std::collections::HashMap<u32, u32>,
    descriptor_defs: &std::collections::HashSet<u32>,
    vertex_id_def: u32,
    offset_def: u32,
) -> Option<(i32, i32)> {
    fn walk(
        vec_map: &std::collections::HashMap<u32, Vec<(u32, u8)>>,
        prod: &std::collections::HashMap<u32, (nir_op, Vec<(u32, u8)>)>,
        consts: &std::collections::HashMap<u32, u32>,
        descriptor_defs: &std::collections::HashSet<u32>,
        vertex_id_def: u32,
        d: u32,
    ) -> Option<(i32, Option<i32>)> {
        if d == vertex_id_def {
            return Some((0, Some(1)));
        }
        if descriptor_defs.contains(&d) {
            return Some((0, None));
        }
        if let Some(&v) = consts.get(&d) {
            return Some((v as i32, None));
        }
        let (op, srcs) = prod.get(&d)?;
        match *op {
            nir_op_iadd if srcs.len() == 2 => {
                let (c0, s0) = walk(vec_map, prod, consts, descriptor_defs, vertex_id_def, srcs[0].0)?;
                let (c1, s1) = walk(vec_map, prod, consts, descriptor_defs, vertex_id_def, srcs[1].0)?;
                let stride = match (s0, s1) {
                    (Some(s), None) | (None, Some(s)) => Some(s),
                    (None, None) => None,
                    (Some(_), Some(_)) => return None, // two dynamic terms: not this shape
                };
                Some((c0 + c1, stride))
            }
            nir_op_ishl if srcs.len() == 2 && srcs[0].0 == vertex_id_def => {
                Some((0, Some(1i32 << *consts.get(&srcs[1].0)?)))
            }
            _ => None,
        }
    }
    let (d, _) = resolve_vm(vec_map, offset_def, 0);
    let (base, stride) = walk(vec_map, prod, consts, descriptor_defs, vertex_id_def, d)?;
    if base % 4 != 0 {
        return None;
    }
    let stride = stride.unwrap_or(0);
    if stride % 4 != 0 {
        return None;
    }
    Some((base / 4, stride / 4))
}

pub(crate) unsafe fn collect_cf_marks(
    nodes: compiler::nir::ExecListIter<'_, nir_cf_node>,
    marks: &mut std::collections::HashMap<usize, Vec<CfMark>>,
    unsupported: &mut Vec<&'static str>,
) {
    for node in nodes {
        if node.as_block().is_some() {
            continue;
        }
        if let Some(nif) = node.as_if() {
            let cond = nif.condition.as_def().index;
            let key = |b: &nir_block| b as *const nir_block as usize;
            marks.entry(key(nif.first_then_block())).or_default().push(CfMark {
                mnem: "EXPUSH",
                cond: Some(cond),
            });
            marks.entry(key(nif.first_else_block())).or_default().push(CfMark {
                mnem: "EXELSE",
                cond: None,
            });
            marks.entry(key(nif.following_block())).or_default().push(CfMark {
                mnem: "EXPOP",
                cond: None,
            });
            collect_cf_marks(nif.iter_then_list(), marks, unsupported);
            collect_cf_marks(nif.iter_else_list(), marks, unsupported);
        } else if let Some(nloop) = node.as_loop() {
            unsupported.push("loop");
            collect_cf_marks(nloop.iter_body(), marks, unsupported);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Ubo {
    Pos,
    Uniform(u8),
    Fixed(u8),
}

/// Self-test: confirms the Rust crate is linked and callable across the FFI.
#[no_mangle]
pub extern "C" fn borgc_selftest() -> u32 {
    0xB0_06
}

#[no_mangle]
pub unsafe extern "C" fn borgc_compile_nir(
    nir: *mut nir_shader,
    out_buf: *mut u8,
    buf_cap: u32,
    out_len: *mut u32,
    vfetch: u32,
) -> u32 {
    let key = if !nir.is_null() && env::var("BORGC_CORPUS_DIR").is_ok() {
        (*nir).to_string().ok().map(|s| corpus::key(&s, vfetch))
    } else {
        None
    };
    let ret = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compile_nir_inner(nir, out_buf, buf_cap, out_len, Flags(vfetch))
    }))
    .unwrap_or_else(|_| {
        if !out_len.is_null() {
            *out_len = 0;
        }
        0
    });
    if let Some(key) = key {
        let len = if out_len.is_null() { 0 } else { *out_len };
        let blob = if !out_buf.is_null() && len <= buf_cap {
            std::slice::from_raw_parts(out_buf, len as usize)
        } else {
            &[]
        };
        corpus::record(key, ret, blob);
    }
    ret
}

unsafe fn compile_nir_inner(
    nir: *mut nir_shader,
    out_buf: *mut u8,
    buf_cap: u32,
    out_len: *mut u32,
    flags: Flags,
) -> u32 {
    if !out_len.is_null() {
        *out_len = 0;
    }
    if nir.is_null() {
        return 0;
    }
    let stage = (*nir).info.stage();
    let is_vertex = stage == MESA_SHADER_VERTEX;
    let is_fragment = stage == MESA_SHADER_FRAGMENT;

    if env::var("BORGC_DUMP_NIR").is_ok() {
        if let Ok(s) = (*nir).to_string() {
            eprintln!("{s}");
        }
    }

    let entry = nir_shader_get_entrypoint(nir);
    let Survey { total, tex } = survey(entry, stage);

    let regs: &'static [u8] = if is_vertex {
        &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    } else if tex > 0 {
        &[17, 18, 19]
    } else {
        &[17, 18, 19, 23]
    };
    let mut cx = Cx::new(regs, is_vertex, is_fragment, flags);

    let frag_alpha = env::var("BORGC_FRAG_ALPHA").is_ok() || (is_fragment && flags.alpha_output());
    cx.frag_alpha = frag_alpha;
    cx.select_entry(entry);
    eprintln!("borgc: selected {} Borg instruction(s) (virtual regs)", cx.sel.prog.len());
    if env::var("BORGC_DUMP_ISA").is_ok() {
        for i in &cx.sel.prog {
            let s: Vec<String> = i.srcs.iter().map(|v| format!("v{v}")).collect();
            eprintln!("borgc:   {} v{}, {}", i.mnem, i.dst, s.join(", "));
        }
    }

    let srgb = cx.sel.prog.iter().any(|i| i.mnem == "FSRGB");
    cx.skipped.retain(|&op| !(matches!(op, "fpow" | "fge") && srgb));
    if !cx.skipped.is_empty() {
        eprintln!("borgc: ERROR unhandled NIR ops, refusing to emit a wrong shader: {}", cx.skipped.join(", "));
        return 0;
    }
    if is_vertex && flags.point_quads() {
        cx.epilogue_point_quads();
    }
    if is_fragment && flags.multi_target() {
        cx.epilogue_multi_target();
    }
    if is_fragment && flags.r8_unorm() && cx.outs.frag[0].is_some() {
        cx.epilogue_r8_unorm();
    }
    if is_fragment && flags.packed() && !flags.multi_target() && cx.outs.frag[0].is_some() {
        cx.epilogue_packed();
    }
    if is_fragment && flags.half() && cx.outs.frag[0].is_some() {
        cx.epilogue_half();
    }
    if is_fragment && flags.bytes_packed() && cx.outs.frag[3].is_some() {
        cx.epilogue_bytes_packed();
    }
    if is_fragment {
        cx.copy_frag_outputs();
    }
    let pos_out = cx.outs.pos.map(|v| v.map(|(d, _)| d));
    let frag_out = cx.outs.frag.map(|v| v.map(|(d, _)| d));
    cx.out_roots.extend(cx.outs.roots.iter().copied());
    eprintln!(
        "borgc: draw-mode I/O map — {} vertex-pulling/MVP load(s), {} SOUT(s), \
         gl_Position={}",
        cx.descriptor_defs.len(), // proxy: one load_ubo per descriptor use
        cx.outs.stores.len(),
        cx.outs.pos[0].is_some(),
    );

    let frag_z = if is_fragment { cx.interpolate_depth() } else { None };

    set_frag_pool(Vec::new());
    if is_fragment && cx.fw.count > 11 {
        cx.pool_window_consts();
    }
    dce(&mut cx.sel.prog, &cx.out_roots);
    fuse_fmadd(&mut cx.sel.prog, &cx.out_roots, &|v| {
        matches!(cx.sel.ubo.get(&v), Some(Ubo::Uniform(_)))
    });
    schedule_for_pressure(&mut cx.sel.prog, &cx.out_roots);

    if is_fragment {
        cx.log_uniformity();
    }

    // Pre-color outputs and reserve fixed registers.
    let (forced, mut extra_reserved) = fixed_regs(&cx, &pos_out, &frag_out, frag_z);
    cx.split_window_reads();

    let Some(alloc) = allocate(&cx, &forced, &mut extra_reserved) else { return 0 };

    let Encoded { mut words, n_preload, n_body, pending } = encode_program(&cx, &alloc);
    words.push(0x0000_0000);
    let blob = build_blob(&cx, &words, n_preload, n_body, pending, &frag_out, frag_z);
    if !out_len.is_null() {
        *out_len = blob.len() as u32;
    }
    if !out_buf.is_null() && (blob.len() as u32) <= buf_cap {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), out_buf, blob.len());
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frag_pool_round_trips_and_clears() {
        set_frag_pool(vec![1, 2, 3]);
        let mut out = [0u32; 2];
        let n = unsafe { borgc_last_frag_pool(out.as_mut_ptr(), 2) };
        assert_eq!((n, out), (2, [1, 2]));
        set_frag_pool(Vec::new());
        assert_eq!(unsafe { borgc_last_frag_pool(out.as_mut_ptr(), 2) }, 0);
    }
}
