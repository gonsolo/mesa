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
mod pipeline;
mod regalloc;
mod select;
mod spill;
mod state;
mod survey;

use compiler::bindings::*;
use compiler::nir::AsDef;
use borgc_flags::*;
use config::Flags;
use isel::resolve_vm;
use state::ShaderState;
use std::env;

fn packed_layout(flags: Flags) -> (u32, Vec<(usize, u32, u32)>) {
    let vfetch = flags.0;
    let kind = (vfetch >> BORGC_FS_PACKED_KIND_SHIFT) & 7;
    if (vfetch >> BORGC_FS_PACKED_SIZE_SHIFT) & 3 == 3 {
        let id = kind | ((vfetch >> BORGC_FS_PACKED_N_SHIFT) & 7) << 3;
        let l: &[(usize, u32, u32)] = match id {
            0 => &[(0, 10, 5), (1, 5, 5), (2, 0, 5), (3, 15, 1)],    // A1R5G5B5
            1 => &[(0, 11, 5), (1, 6, 5), (2, 1, 5), (3, 0, 1)],     // R5G5B5A1
            2 => &[(2, 11, 5), (1, 6, 5), (0, 1, 5), (3, 0, 1)],     // B5G5R5A1
            3 => &[(0, 12, 4), (1, 8, 4), (2, 4, 4), (3, 0, 4)],     // R4G4B4A4
            4 => &[(2, 12, 4), (1, 8, 4), (0, 4, 4), (3, 0, 4)],     // B4G4R4A4
            5 => &[(3, 12, 4), (0, 8, 4), (1, 4, 4), (2, 0, 4)],     // A4R4G4B4
            6 => &[(3, 12, 4), (2, 8, 4), (1, 4, 4), (0, 0, 4)],     // A4B4G4R4
            7 | 12 => &[(0, 0, 10), (1, 10, 10), (2, 20, 10), (3, 30, 2)],  // A2B10G10R10
            8 | 13 => &[(2, 0, 10), (1, 10, 10), (0, 20, 10), (3, 30, 2)],  // A2R10G10B10
            9 => &[(0, 0, 5), (1, 5, 6), (2, 11, 5)],                // B5G6R5
            11 => &[(0, 11, 5), (1, 5, 6), (2, 0, 5)],               // R5G6B5
            _ => &[(0, 4, 4), (1, 0, 4)],                            // R4G4
        };
        return (if id >= 12 { 4 } else { 0 }, l.to_vec());
    }
    let bits: u32 = match (vfetch >> BORGC_FS_PACKED_SIZE_SHIFT) & 3 { 0 => 8, 1 => 16, _ => 32 };
    let n = ((vfetch >> BORGC_FS_PACKED_N_SHIFT) & 3) as usize + 1;
    let swap = (vfetch >> BORGC_FS_PACKED_SWAP_SHIFT) & 1 != 0;
    (kind, (0..n).map(|k| (if swap && k < 3 { 2 - k } else { k }, k as u32 * bits, bits)).collect())
}

/// Window words a fragment shader may ask for before the pool pass sorts them out.
const FRAG_CONST_MAX: u32 = 11 + 64;
const FRAG_POOL_SLOT: u32 = 40;
/// The vertex shader's pool slot in the vertex attribute table.
pub(crate) const VERT_POOL_SLOT: u32 = 42;
thread_local! {
    static LAST_CS_POOL: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    static LAST_FRAG_POOL: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    static LAST_VERT_POOL: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Where a window shader finds its push words: the heap region's slot 33, past the 32 binding slots.
pub(crate) const PUSH_WINDOW_WORDS: u32 = 16384 * 33;
thread_local! {
    static OUTPUT_SELECT: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
}

/// A fragment shader keeps only colour output `k` (-1: all), for a target drawn on its own.
#[no_mangle]
pub extern "C" fn borgc_set_output_select(k: i32) {
    OUTPUT_SELECT.with(|c| c.set(k));
}

pub(crate) fn output_select() -> i32 {
    OUTPUT_SELECT.with(|c| c.get())
}

thread_local! {
    static LAST_PUSH_WINDOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    static LAST_SPILL_SLOTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

pub(crate) fn set_spill_slots(n: u32) {
    LAST_SPILL_SLOTS.with(|c| c.set(n));
}

/// The last fragment shader spills to this many scratch slots of the draw's buffer region.
#[no_mangle]
pub extern "C" fn borgc_last_spill_slots() -> u32 {
    LAST_SPILL_SLOTS.with(|c| c.get())
}

pub(crate) fn set_push_window(on: bool) {
    LAST_PUSH_WINDOW.with(|c| c.set(on));
}

/// The last shader read its push constants from the heap window (it also loads a buffer).
#[no_mangle]
pub extern "C" fn borgc_last_push_window() -> bool {
    LAST_PUSH_WINDOW.with(|c| c.get())
}

pub(crate) fn set_vert_pool(pool: Vec<u32>) {
    LAST_VERT_POOL.with(|p| *p.borrow_mut() = pool);
}

#[no_mangle]
pub unsafe extern "C" fn borgc_last_vert_pool(out: *mut u32, cap: u32) -> u32 {
    LAST_VERT_POOL.with(|p| {
        let p = p.borrow();
        let n = p.len().min(cap as usize);
        for (i, &w) in p.iter().take(n).enumerate() {
            *out.add(i) = w;
        }
        n as u32
    })
}

pub(crate) fn set_cs_pool(pool: Vec<u32>) {
    LAST_CS_POOL.with(|p| *p.borrow_mut() = pool);
}

/// The last compute shader's constant pool, for the job's slot 34.
#[no_mangle]
pub unsafe extern "C" fn borgc_last_cs_pool(out: *mut u32, cap: u32) -> u32 {
    LAST_CS_POOL.with(|p| {
        let p = p.borrow();
        let n = p.len().min(cap as usize);
        for (i, &w) in p.iter().take(n).enumerate() {
            *out.add(i) = w;
        }
        n as u32
    })
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
    /// The loop of a LOOP_HEAD, LOOP_END or REPUSH.
    pub(crate) lp: Option<usize>,
}

/// Compute loops: the loop each break or continue block jumps out of.
#[derive(Default)]
pub(crate) struct Loops {
    pub(crate) count: usize,
    pub(crate) jumps: std::collections::HashMap<usize, usize>,
    pub(crate) err: Option<String>,
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
        // A constant already mapped to a register by alu().
        if let Some((&k, _)) = vec_map.iter().find(|(k, c)| consts.contains_key(k) && c.first() == Some(&(d, 0))) {
            return Some((consts[&k] as i32, None));
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

/// EXPUSH/EXELSE/EXPOP at the blocks an `if` starts, switches and ends at. With `loops` a loop
/// runs while any lane is alive: LOOP_HEAD masks to the lanes going round, a jump clears them,
/// and after an `if` holding a jump REPUSH masks the rest of the list again (popped at its end).
/// Returns whether the list holds a jump and whether it needs that extra pop.
pub(crate) unsafe fn collect_cf_marks(
    nodes: compiler::nir::ExecListIter<'_, nir_cf_node>,
    marks: &mut std::collections::HashMap<usize, Vec<CfMark>>,
    unsupported: &mut Vec<&'static str>,
    mut loops: Option<&mut Loops>,
    cur: Option<usize>,
) -> (bool, bool) {
    let key = |b: &nir_block| b as *const nir_block as usize;
    let mark = |mnem, cond, lp| CfMark { mnem, cond, lp };
    let (mut jumps, mut repushed) = (false, false);
    for node in nodes {
        if let Some(b) = node.as_block() {
            if let Some(l) = loops.as_deref_mut() {
                if b.iter_instr_list().any(|i| i.as_jump().is_some()) {
                    match cur {
                        Some(c) => {
                            l.jumps.insert(key(b), c);
                            jumps = true;
                        }
                        None => l.err = Some("a jump outside a loop".into()),
                    }
                }
            }
            continue;
        }
        if let Some(nif) = node.as_if() {
            let (tj, tr) = collect_cf_marks(nif.iter_then_list(), marks, unsupported, loops.as_deref_mut(), cur);
            let (ej, er) = collect_cf_marks(nif.iter_else_list(), marks, unsupported, loops.as_deref_mut(), cur);
            marks.entry(key(nif.first_then_block())).or_default().push(mark("EXPUSH", Some(nif.condition.as_def().index), None));
            let e = marks.entry(key(nif.first_else_block())).or_default();
            if tr {
                e.push(mark("EXPOP", None, None));
            }
            e.push(mark("EXELSE", None, None));
            let f = marks.entry(key(nif.following_block())).or_default();
            if er {
                f.push(mark("EXPOP", None, None));
            }
            f.push(mark("EXPOP", None, None));
            if tj || ej {
                if repushed {
                    f.push(mark("EXPOP", None, None));
                }
                f.push(mark("REPUSH", None, cur));
                (jumps, repushed) = (true, true);
            }
        } else if let Some(nloop) = node.as_loop() {
            let Some(l) = loops.as_deref_mut() else {
                unsupported.push("loop");
                collect_cf_marks(nloop.iter_body(), marks, unsupported, None, None);
                continue;
            };
            let id = l.count;
            l.count += 1;
            let cl = &nloop.continue_list;
            if !std::ptr::eq(cl.head_sentinel.next, &cl.tail_sentinel) {
                l.err = Some("a loop with a continue construct".into());
            }
            marks.entry(key(nloop.first_block())).or_default().push(mark("LOOP_HEAD", None, Some(id)));
            let (_, br) = collect_cf_marks(nloop.iter_body(), marks, unsupported, loops.as_deref_mut(), Some(id));
            let f = marks.entry(key(nloop.following_block())).or_default();
            if br {
                f.push(mark("EXPOP", None, None));
            }
            f.push(mark("LOOP_END", None, Some(id)));
        }
    }
    (jumps, repushed)
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
    let forced = env::var("BORGC_SPILL_LIMIT").ok().and_then(|v| v.parse::<usize>().ok());
    for limit in [forced, Some(20), Some(18), Some(16), Some(14), Some(12), Some(10)] {
        let (mut cx, entry, total) = ShaderState::start(nir, flags);
        cx.spill_limit = limit;
        cx.select_entry(entry);
        if !cx.check_selected() {
            return 0;
        }
        cx.lower_outputs();
        cx.optimise();
        if cx.spill_slots > 0 {
            eprintln!("borgc: spilled {} value(s) to keep {:?} live", cx.spill_slots, limit);
        }
        let blob = match cx.assemble() {
            Some(blob) => blob,
            None if (cx.is_fragment || cx.is_vertex) && cx.alloc_failed => continue,
            None => return 0,
        };
        if !out_len.is_null() {
            *out_len = blob.len() as u32;
        }
        if !out_buf.is_null() && (blob.len() as u32) <= buf_cap {
            std::ptr::copy_nonoverlapping(blob.as_ptr(), out_buf, blob.len());
        }
        return total;
    }
    0
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
