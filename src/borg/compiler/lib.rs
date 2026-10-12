// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

#![allow(non_upper_case_globals)]

mod compute;
mod config;
mod corpus;
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
use encode::{emit_blob, encode, encode_fattr, encode_sout};
use isel::resolve_vm;
use opt::{classify_uniform, dce, fuse_fmadd, schedule_for_pressure};
use regalloc::regalloc;
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
static mut LAST_FRAG_POOL: Vec<u32> = Vec::new();

#[no_mangle]
pub unsafe extern "C" fn borgc_last_frag_pool(out: *mut u32, cap: u32) -> u32 {
    #[allow(static_mut_refs)]
    let p = &LAST_FRAG_POOL;
    let n = p.len().min(cap as usize);
    for (i, &w) in p.iter().take(n).enumerate() {
        *out.add(i) = w;
    }
    n as u32
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
    Mvp(i32),
    Pos,
    Attr,
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

    use std::collections::HashMap;

    let draw_mode = env::var("BORGC_LEGACY").is_err();
    let regs: &'static [u8] = if is_vertex {
        &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    } else if tex > 0 {
        &[17, 18, 19]
    } else {
        &[17, 18, 19, 23]
    };
    let mut cx = Cx::new(regs, draw_mode, is_vertex, is_fragment, flags);
    const VARYING_SLOT_POS: u32 = 0;

    let frag_alpha = env::var("BORGC_FRAG_ALPHA").is_ok() || (is_fragment && flags.alpha_output());
    cx.frag_alpha = frag_alpha;
    if !entry.is_null() {
        let mut cf_marks: std::collections::HashMap<usize, Vec<CfMark>> =
            std::collections::HashMap::new();
        let mut cf_unsupported: Vec<&'static str> = Vec::new();
        collect_cf_marks((*entry).iter_body(), &mut cf_marks, &mut cf_unsupported);
        if !cf_unsupported.is_empty() {
            eprintln!(
                "borgc: WARNING unsupported control flow ({}) -- the body will be \
                 emitted UNMASKED and run for every lane. See BorgCore.wireExecMask.",
                cf_unsupported.join(", ")
            );
        }

        let mut reg_vregs: HashMap<u32, Vec<u32>> = HashMap::new();
        for block in (*entry).iter_blocks() {
            if let Some(ms) = cf_marks.get(&(block as *const nir_block as usize)) {
                for m in ms {
                    cx.sel.prog.push(BorgInstr {
                        mnem: m.mnem,
                        dst: NO_DST,
                        srcs: m.cond.map(|c| vec![c]).unwrap_or_default(),
                        swz: m.cond.map(|_| vec![0u8]).unwrap_or_default(),
                    });
                }
            }
            for instr in block.iter_instr_list() {
                if let Some(intr) = instr.as_intrinsic() {
                    if intr.intrinsic == nir_intrinsic_decl_reg {
                        let n = intr.get_const_index(NIR_INTRINSIC_NUM_COMPONENTS);
                        let v: Vec<u32> = (0..n).map(|_| { let r = cx.sel.next_vreg; cx.sel.next_vreg += 1; r }).collect();
                        reg_vregs.insert(intr.def.index, v);
                        continue;
                    }
                    if intr.intrinsic == nir_intrinsic_load_reg {
                        let h = intr.get_src(0).as_def().index;
                        let comps: Vec<(u32, u8)> = reg_vregs[&h].iter().map(|&r| (r, 0u8)).collect();
                        cx.sel.vec_map.insert(intr.def.index, comps);
                        continue;
                    }
                    if intr.intrinsic == nir_intrinsic_store_reg {
                        let h = intr.get_src(1).as_def().index;
                        let src = intr.get_src(0).as_def().index;
                        for c in 0..intr.get_src(0).num_components() as usize {
                            if intr.write_mask() & (1 << c) == 0 {
                                continue;
                            }
                            let (v, comp) = resolve_vm(&cx.sel.vec_map, src, c as u8);
                            cx.sel.prog.push(BorgInstr { mnem: "FMOV", dst: reg_vregs[&h][c], srcs: vec![v], swz: vec![comp] });
                        }
                        continue;
                    }
                }
                if let Some(alu) = instr.as_alu() {
                    cx.alu(alu);
                } else if let Some(intr) = instr.as_intrinsic() {
                    cx.intrinsic(intr);
                } else if let Some(tex) = instr.as_tex() {
                    cx.tex(tex);
                } else if let Some(lc) = instr.as_load_const() {
                    cx.load_const(lc);
                }
            }
        }
    }
    eprintln!("borgc: selected {} Borg instruction(s) (virtual regs)", cx.sel.prog.len());
    if env::var("BORGC_DUMP_ISA").is_ok() {
        for i in &cx.sel.prog {
            let s: Vec<String> = i.srcs.iter().map(|v| format!("v{v}")).collect();
            eprintln!("borgc:   {} v{}, {}", i.mnem, i.dst, s.join(", "));
        }
    }

    let mut mvp_loads = 0u32;
    let mut pos_loads = 0u32;
    let mut attr_loads = 0u32;
    let mut gl_position: Option<u32> = None;
    let mut varying_outs = 0u32;
    let mut pos_out: [Option<u32>; 4] = [None; 4];
    let mut frag_out: [Option<u32>; 4] = [None; 4];
    if !draw_mode && !entry.is_null() {
        for block in (*entry).iter_blocks() {
            for instr in block.iter_instr_list() {
                if let Some(intr) = instr.as_intrinsic() {
                    match intr.intrinsic {
                        nir_intrinsic_load_ubo => {
                            let rb = intr.range_base();
                            let kind = match rb {
                                0..=63 => {
                                    mvp_loads += 1;
                                    Ubo::Mvp(rb)
                                }
                                64..=639 => {
                                    pos_loads += 1;
                                    Ubo::Pos
                                }
                                _ => {
                                    attr_loads += 1;
                                    Ubo::Attr
                                }
                            };
                            cx.sel.ubo.insert(intr.def.index, kind);
                        }
                        nir_intrinsic_store_output => {
                            let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                            let src = intr.srcs_as_slice()[0].as_def().index;
                            if is_vertex && loc == VARYING_SLOT_POS {
                                gl_position = Some(src);
                                let n = cx.sel.vec_map.get(&src).map_or(1, |v| v.len()).min(4);
                                for c in 0..n {
                                    let (d, _) = resolve_vm(&cx.sel.vec_map, src, c as u8);
                                    pos_out[c] = Some(d);
                                    cx.out_roots.push(d);
                                }
                            } else if is_fragment {
                                let ncomp = if frag_alpha { 4 } else { 3 };
                                let n = cx.sel.vec_map.get(&src).map_or(1, |v| v.len()).min(ncomp);
                                for c in 0..n {
                                    let (d, _) = resolve_vm(&cx.sel.vec_map, src, c as u8);
                                    frag_out[c] = Some(d);
                                    cx.out_roots.push(d);
                                }
                            } else {
                                varying_outs += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    let srgb = cx.sel.prog.iter().any(|i| i.mnem == "FSRGB");
    cx.skipped.retain(|&op| !(matches!(op, "fpow" | "fge") && srgb));
    if !cx.skipped.is_empty() {
        eprintln!("borgc: ERROR unhandled NIR ops, refusing to emit a wrong shader: {}", cx.skipped.join(", "));
        return 0;
    }
    if draw_mode && is_vertex && flags.point_quads() {
        cx.epilogue_point_quads();
    }
    if draw_mode && is_fragment && flags.multi_target() {
        cx.epilogue_multi_target();
    }
    if draw_mode && is_fragment && flags.r8_unorm() && cx.outs.frag[0].is_some() {
        cx.epilogue_r8_unorm();
    }
    if draw_mode && is_fragment && flags.packed() && !flags.multi_target() && cx.outs.frag[0].is_some() {
        cx.epilogue_packed();
    }
    if draw_mode && is_fragment && flags.half() && cx.outs.frag[0].is_some() {
        cx.epilogue_half();
    }
    if draw_mode && is_fragment && flags.bytes_packed() && cx.outs.frag[3].is_some() {
        cx.epilogue_bytes_packed();
    }
    if draw_mode {
        if is_fragment {
            for c in 0..4 {
                let Some((v, comp)) = cx.outs.frag[c] else { continue };
                // A TEX result block (r20-r23) or FATTR block is not the output register: copy that too.
                if cx.sel.prog.iter().any(|i| i.dst == v) && !cx.tex_dsts.contains(&v) && !cx.fattr_dsts.contains(&v) {
                    continue;
                }
                let out = cx.sel.next_vreg;
                cx.sel.next_vreg += 1;
                if cx.tex_dsts.contains(&v) || cx.fattr_dsts.contains(&v) {
                    cx.sel.prog.push(BorgInstr { mnem: "IOR", dst: out, srcs: vec![v, v], swz: vec![comp, comp] });
                } else {
                    cx.sel.prog.push(BorgInstr { mnem: "FMOV", dst: out, srcs: vec![v], swz: vec![comp] });
                }
                cx.outs.frag[c] = Some((out, 0));
                cx.out_roots.push(out);
            }
        }
        for (c, v) in cx.outs.pos.iter().enumerate() {
            pos_out[c] = v.map(|(d, _)| d);
        }
        for (c, v) in cx.outs.frag.iter().enumerate() {
            frag_out[c] = v.map(|(d, _)| d);
        }
        cx.out_roots.extend(cx.outs.roots.iter().copied());
        eprintln!(
            "borgc: draw-mode I/O map — {} vertex-pulling/MVP load(s), {} SOUT(s), \
             gl_Position={}",
            cx.descriptor_defs.len(), // proxy: one load_ubo per descriptor use
            cx.outs.stores.len(),
            cx.outs.pos[0].is_some(),
        );
    } else {
        eprintln!(
            "borgc: I/O map — MVP {mvp_loads}→u8..u23, position {pos_loads}→u0..u2, \
             attr {attr_loads}→u6/u7 (DMA, DCE'd); gl_Position=v{}→r0..r3 ({varying_outs} varying out)",
            gl_position.map_or("?".to_string(), |v| v.to_string())
        );
    }

    let mut frag_z: Option<u32> = None;
    if is_fragment {
        if let Some(w) = cx.weights {
            let uz: Vec<u32> = [30u8, 29, 28]
                .iter()
                .map(|&idx| {
                    let v = cx.sel.next_vreg;
                    cx.sel.next_vreg += 1;
                    cx.sel.ubo.insert(v, Ubo::Uniform(idx));
                    v
                })
                .collect();
            let t0 = cx.sel.next_vreg; cx.sel.next_vreg += 1;
            cx.sel.prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![w[0], uz[0]], swz: vec![0, 0] });
            let t1 = cx.sel.next_vreg; cx.sel.next_vreg += 1;
            cx.sel.prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![w[1], uz[1], t0], swz: vec![0, 0, 0] });
            let zr = cx.sel.next_vreg; cx.sel.next_vreg += 1;
            cx.sel.prog.push(BorgInstr { mnem: "FMADD", dst: zr, srcs: vec![w[2], uz[2], t1], swz: vec![0, 0, 0] });
            cx.out_roots.push(zr);
            frag_z = Some(zr);
        }
    }

    #[allow(static_mut_refs)]
    unsafe { LAST_FRAG_POOL.clear(); }
    if draw_mode && is_fragment && cx.fw.count > 11 {
        let first_pool = 20 + 9;
        // The window keeps the nine constants read most often: a window word costs no register.
        {
            let mut uses: HashMap<u32, usize> = HashMap::new();
            for i in cx.sel.prog.iter() {
                for s in i.srcs.iter() {
                    if let Some(&Ubo::Uniform(u)) = cx.sel.ubo.get(s) {
                        if u >= 20 {
                            *uses.entry(u as u32).or_insert(0) += 1;
                        }
                    }
                }
            }
            let mut by_use: Vec<(u32, u32)> = cx.fw.consts.clone();
            by_use.sort_by_key(|&(u, _)| (std::cmp::Reverse(uses.get(&u).copied().unwrap_or(0)), u));
            let renum: HashMap<u32, u32> = by_use.iter().enumerate().map(|(k, &(u, _))| (u, 20 + k as u32)).collect();
            for (_, m) in cx.sel.ubo.iter_mut() {
                if let Ubo::Uniform(u) = m {
                    if let Some(&n) = renum.get(&(*u as u32)) {
                        *u = n as u8;
                    }
                }
            }
            cx.fw.consts = by_use.iter().map(|&(u, bits)| (renum[&u], bits)).collect();
        }
        let mut pool: Vec<u32> = vec![0; (20 + cx.fw.count - first_pool) as usize];
        for &(u, bits) in cx.fw.consts.iter() {
            if u >= first_pool {
                pool[(u - first_pool) as usize] = bits;
            }
        }
        cx.fw.consts.retain(|&(u, _)| u < first_pool);
        cx.fw.consts.push((first_pool, (1 << 16) | (128 + FRAG_POOL_SLOT)));
        cx.fw.consts.push((first_pool + 1, 1));
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = cx.sel.next_vreg;
            cx.sel.next_vreg += 1;
            if let Some(u) = u {
                ubo.insert(v, u);
            }
            v
        };
        let ctl = fresh(&mut cx.sel.ubo, Some(Ubo::Uniform(first_pool as u8)));
        let one = fresh(&mut cx.sel.ubo, Some(Ubo::Uniform(first_pool as u8 + 1)));
        let f = fresh(&mut cx.sel.ubo, Some(Ubo::Fixed(30)));
        cx.per_pixel_fixed.insert(f);
        let zero = fresh(&mut cx.sel.ubo, None);
        let is_pool = |v: &u32, ubo: &HashMap<u32, Ubo>| match ubo.get(v) {
            Some(&Ubo::Uniform(u)) if u as u32 >= first_pool && *v != ctl && *v != one => Some(u as u32 - first_pool),
            _ => None,
        };
        let mut first_use: Vec<Option<usize>> = vec![None; pool.len()];
        for (at, i) in cx.sel.prog.iter().enumerate() {
            for s in i.srcs.iter() {
                if let Some(k) = is_pool(s, &cx.sel.ubo) {
                    first_use[k as usize].get_or_insert(at);
                }
            }
        }
        let mut order: Vec<usize> = (0..pool.len()).filter(|&k| first_use[k].is_some()).collect();
        order.sort_by_key(|&k| first_use[k]);
        let mut slot_of: Vec<usize> = vec![0; pool.len()];
        for (slot, &k) in order.iter().enumerate() {
            slot_of[k] = slot;
        }
        let pool: Vec<u32> = order.iter().map(|&k| pool[k]).collect();
        const GAP: usize = 16;
        let ngroups = (pool.len() + 3) / 4;
        let mut last: Vec<Option<usize>> = vec![None; ngroups];
        let mut events: Vec<(usize, usize)> = Vec::new();   // (before instruction, group)
        for (at, i) in cx.sel.prog.iter().enumerate() {
            for s in i.srcs.iter() {
                let Some(k) = is_pool(s, &cx.sel.ubo) else { continue };
                let g = slot_of[k as usize] / 4;
                if last[g].map_or(true, |l| at - l > GAP) {
                    let mut p = at;
                    loop {
                        let Some(t) = (0..p).rev().find(|&t| cx.sel.prog[t].mnem == "TEX") else { break };
                        let d = cx.sel.prog[t].dst;
                        if cx.sel.prog[p..].iter().any(|i| i.srcs.contains(&d)) { p = t; } else { break; }
                    }
                    events.push((p, g));
                }
                last[g] = Some(at);
            }
        }
        events.sort();
        let mut cur: Vec<[u32; 4]> = vec![[0; 4]; ngroups];
        let mut out: Vec<BorgInstr> = vec![BorgInstr { mnem: "ISUB", dst: zero, srcs: vec![f, f], swz: vec![0, 0] }];
        let mut e = 0;
        for (at, mut i) in std::mem::take(&mut cx.sel.prog).into_iter().enumerate() {
            while e < events.len() && events[e].0 == at {
                let g = events[e].1;
                let mut idx = zero;
                for _ in 0..g {
                    let n = fresh(&mut cx.sel.ubo, None);
                    out.push(BorgInstr { mnem: "IADD", dst: n, srcs: vec![idx, one], swz: vec![0, 0] });
                    idx = n;
                }
                let t = fresh(&mut cx.sel.ubo, None);
                out.push(BorgInstr { mnem: "TEX", dst: t, srcs: vec![idx, zero, ctl], swz: vec![0, 0, 0] });
                cx.tex_dsts.insert(t);
                for c in 0..4u8 {
                    let r = fresh(&mut cx.sel.ubo, None);
                    out.push(BorgInstr { mnem: "IOR", dst: r, srcs: vec![t, t], swz: vec![c, c] });
                    cur[g][c as usize] = r;
                }
                e += 1;
            }
            for s in i.srcs.iter_mut() {
                if let Some(k) = is_pool(s, &cx.sel.ubo) {
                    let slot = slot_of[k as usize];
                    *s = cur[slot / 4][slot % 4];
                }
            }
            out.push(i);
        }
        cx.sel.prog = out;
        eprintln!("borgc: constant pool of {} word(s)", pool.len());
        #[allow(static_mut_refs)]
        unsafe { LAST_FRAG_POOL = pool; }
    }
    dce(&mut cx.sel.prog, &cx.out_roots);
    fuse_fmadd(&mut cx.sel.prog, &cx.out_roots, &|v| {
        matches!(cx.sel.ubo.get(&v), Some(Ubo::Mvp(_)) | Some(Ubo::Uniform(_)) | Some(Ubo::Attr))
    });
    if draw_mode {
        schedule_for_pressure(&mut cx.sel.prog, &cx.out_roots);
    }

    if is_fragment {
        let uniform_roots: std::collections::HashSet<u32> = cx.sel.ubo
            .iter()
            .filter_map(|(&v, u)| match u {
                Ubo::Uniform(_) => Some(v),
                Ubo::Fixed(_) if !cx.per_pixel_fixed.contains(&v) => Some(v),
                _ => None,
            })
            .collect();
        let uniform = classify_uniform(&cx.sel.prog, &uniform_roots);
        if env::var("BORGC_DUMP_ISA").is_ok() {
            let n_uniform = cx.sel.prog.iter().filter(|i| uniform.contains(&i.dst)).count();
            eprintln!(
                "borgc: uniformity — {} of {} instr(s) are primitive-uniform (hoistable)",
                n_uniform,
                cx.sel.prog.len()
            );
            for i in &cx.sel.prog {
                if uniform.contains(&i.dst) {
                    let s: Vec<String> = i.srcs.iter().map(|v| format!("v{v}")).collect();
                    eprintln!("borgc:   [uniform] {} v{}, {}", i.mnem, i.dst, s.join(", "));
                }
            }
        }
    }

    // Pre-color outputs and reserve fixed registers.
    let mut forced: HashMap<u32, u8> = HashMap::new();
    let mut extra_reserved: Vec<u8> = Vec::new();
    if is_vertex {
        // gl_Position components → r0..r3.
        for (c, v) in pos_out.iter().enumerate() {
            if let Some(def) = v {
                forced.insert(*def, c as u8);
            }
        }
        if gl_position.is_some() {
            extra_reserved.push(4);
        }
        // A typed vertex fetch (TEX) writes rd..rd+3: one block, r20..r23, for all of them.
        for &t in &cx.tex_dsts {
            forced.insert(t, 20);
        }
        if !cx.tex_dsts.is_empty() {
            extra_reserved.extend_from_slice(&[21, 22, 23]);
        }
        extra_reserved.extend_from_slice(&cx.cp.regs[..cx.cp.count]);
    } else {
        for (c, v) in frag_out.iter().enumerate() {
            if let Some(def) = v {
                forced.insert(*def, if c == 3 { 24 } else { 26 + c as u8 });
            }
        }
        for &t in &cx.tex_dsts {
            forced.insert(t, 20);
        }
        for &f in &cx.fattr_dsts {
            forced.insert(f, 10);
        }
        if let Some(zr) = frag_z {
            forced.insert(zr, 29); // interpolated depth → r29
        }
        extra_reserved.extend_from_slice(&[4, 21, 22, 23, 25, 26, 27, 28, 29]);   // r25 is the kill flag
        if draw_mode {
            extra_reserved.extend_from_slice(&[5, 6, 7, 10, 11, 12]);
            if cx.frag_coord_w {
                extra_reserved.push(8);
            }
        } else {
            extra_reserved.extend_from_slice(&[0, 1, 2]);
        }
        extra_reserved.extend_from_slice(&cx.cp.regs[..cx.cp.count]);
        if frag_out[3].is_some() {
            extra_reserved.push(24);
        }
    }

    if draw_mode {
        let is_win = |v: &u32, ubo: &HashMap<u32, Ubo>| matches!(ubo.get(v), Some(Ubo::Uniform(_)));
        if cx.sel.prog.iter().any(|i| i.srcs.iter().filter(|v| is_win(v, &cx.sel.ubo)).count() > 1) {
            let f = cx.sel.next_vreg;
            let z = cx.sel.next_vreg + 1;
            cx.sel.next_vreg += 2;
            cx.sel.ubo.insert(f, Ubo::Fixed(30));
            let mut out = vec![BorgInstr { mnem: "ISUB", dst: z, srcs: vec![f, f], swz: vec![0, 0] }];
            for mut ins in std::mem::take(&mut cx.sel.prog) {
                let mut seen = false;
                for k in 0..ins.srcs.len() {
                    if !is_win(&ins.srcs[k], &cx.sel.ubo) {
                        continue;
                    }
                    if !seen {
                        seen = true;
                        continue;
                    }
                    let t = cx.sel.next_vreg;
                    cx.sel.next_vreg += 1;
                    out.push(BorgInstr { mnem: "IADD", dst: t, srcs: vec![ins.srcs[k], z], swz: vec![ins.swz[k], 0] });
                    ins.srcs[k] = t;
                    ins.swz[k] = 0;
                }
                out.push(ins);
            }
            cx.sel.prog = out;
        }
    }
    let soft: Vec<u8> = if is_fragment && draw_mode { vec![20, 21, 22, 23, 24, 26, 27, 28] } else { Vec::new() };
    let mut block_len: HashMap<u32, u8> = HashMap::new();
    for &t in &cx.tex_dsts {
        block_len.insert(t, 4);
    }
    extra_reserved.retain(|r| !soft.contains(r));
    let alloc = regalloc(&cx.sel.prog, &forced, &extra_reserved, &soft, &block_len);
    // A value without a register would be read and written as r0.
    if let Some(i) = cx.sel.prog.iter().find(|i| i.dst != NO_DST && !alloc.contains_key(&i.dst)) {
        let mut res: Vec<u8> = extra_reserved.clone();
        res.sort();
        res.dedup();
        eprintln!("borgc: ERROR out of registers ({} has none; reserved {:?}), refusing to emit a wrong shader", i.mnem, res);
        return 0;
    }
    if env::var("BORGC_DUMP_ALLOC").is_ok() {
        for i in &cx.sel.prog {
            let r = |v: &u32| match (alloc.get(v), cx.sel.ubo.get(v)) {
                (Some(r), _) => format!("r{r}"),
                (None, Some(Ubo::Uniform(u))) => format!("u{u}"),
                (None, Some(Ubo::Fixed(f))) => format!("R{f}"),
                _ => format!("?{v}"),
            };
            let s: Vec<String> = i.srcs.iter().zip(i.swz.iter()).map(|(v, c)| format!("{}.{}", r(v), c)).collect();
            eprintln!("borgc:A {} {} <- {}", i.mnem, r(&i.dst), s.join(", "));
        }
    }

    let mut words: Vec<u32> = Vec::new();
    let mut pos_gpr: std::collections::HashMap<(u32, u8), u8> = HashMap::new();
    let mut next_pre: u8 = 24; // r24..r29: above regalloc's low-GPR fill
    for i in &cx.sel.prog {
        if i.mnem == "mov" {
            continue;
        }
        for (s, &c) in i.srcs.iter().zip(i.swz.iter()) {
            if cx.sel.ubo.get(s) == Some(&Ubo::Pos)
                && c != 3 // pos.w is the constant 1.0 (folded below), never pre-loaded
                && !pos_gpr.contains_key(&(*s, c))
                && next_pre <= 29
            {
                let g = next_pre;
                next_pre += 1;
                pos_gpr.insert((*s, c), g);
                // g = u[c] (position component c → uniform u0..u2), funct3=1.
                if let Some(w) = encode("FADD", g, c, 30, 0, 1) {
                    words.push(w);
                }
            }
        }
    }
    let n_preload = words.len();

    let mut pending = 0u32;
    // Resolve one source operand to (physical register, reads-from-uniform).
    let resolve_op = |s: u32, c: u8| -> (u8, bool) {
        // TEX result component c → rd+c (R/G/B/A in consecutive regs).
        if cx.tex_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        // FATTR result component c → rd+c (per-vertex v0/v1/v2 in consecutive regs).
        if cx.fattr_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        match cx.sel.ubo.get(&s) {
            Some(&Ubo::Mvp(rb)) => ((8 + (rb / 16) * 4 + c as i32) as u8, true),
            Some(&Ubo::Pos) => (*pos_gpr.get(&(s, c)).unwrap_or(&0), false),
            Some(&Ubo::Uniform(idx)) => (idx, true),
            Some(&Ubo::Fixed(reg)) => (reg, false),
            Some(&Ubo::Attr) => (6 + c, true),  // u6=U, u7=V (DMA vertex descriptor)
            _ => (*alloc.get(&s).unwrap_or(&0), false),
        }
    };
    for i in &cx.sel.prog {
        if i.mnem == "mov" {
            continue;
        }
        if matches!(i.mnem, "EXPUSH" | "EXELSE" | "EXPOP") {
            let rs1 = if i.srcs.is_empty() {
                0
            } else {
                resolve_op(i.srcs[0], i.swz[0]).0
            };
            if let Some(w) = encode(i.mnem, 0, rs1, 0, 0, 0) {
                words.push(w);
            }
            continue;
        }
        if i.mnem == "SOUT" {
            let index = i.swz[0] as u16;
            let rs2 = resolve_op(i.srcs[0], 0).0;
            if let Some(w) = encode_sout(rs2, index, 0) {
                words.push(w);
            }
            continue;
        }
        if i.mnem == "FATTR" {
            let index = i.swz[0] as u16;
            let rd = *alloc.get(&i.dst).unwrap_or(&0);
            if let Some(w) = encode_fattr(rd, index) {
                words.push(w);
            }
            continue;
        }
        let rd = *alloc.get(&i.dst).unwrap_or(&0);

        let posw = i
            .srcs
            .iter()
            .zip(i.swz.iter())
            .take(2) // only the multiply operands (rs1/rs2) can be pos.w
            .position(|(s, &c)| cx.sel.ubo.get(s) == Some(&Ubo::Pos) && c == 3);
        if let Some(k) = posw {
            if i.mnem == "FMUL" || i.mnem == "FMADD" {
                let other = 1 - k; // the surviving multiply operand
                let (oreg, ouni) = resolve_op(i.srcs[other], i.swz[other]);
                // FMUL → FADD(rd, other, 0); FMADD → FADD(rd, other, acc).
                let (addend, f3) = if i.mnem == "FMADD" {
                    (resolve_op(i.srcs[2], i.swz[2]).0, if ouni { 1 } else { 0 })
                } else {
                    (30, if ouni { 1 } else { 0 }) // r30 = 0 → load `other`
                };
                if let Some(w) = encode("FADD", rd, oreg, addend, 0, f3) {
                    words.push(w);
                }
                continue;
            }
        }

        let mut r = [0u8; 3];
        let mut f3 = 0u32;
        for (k, (s, &c)) in i.srcs.iter().zip(i.swz.iter()).take(3).enumerate() {
            if cx.tex_dsts.contains(s) {
                r[k] = alloc.get(s).map_or(0, |&rr| rr + c); // TEX result rd+c
                continue;
            }
            if cx.fattr_dsts.contains(s) {
                r[k] = alloc.get(s).map_or(0, |&rr| rr + c); // FATTR result rd+c
                continue;
            }
            match cx.sel.ubo.get(s) {
                // MVP column → uniform u(8 + column*4 + component), read in place.
                Some(&Ubo::Mvp(rb)) => {
                    if f3 == 0 {
                        r[k] = (8 + (rb / 16) * 4 + c as i32) as u8;
                        f3 = k as u32 + 1; // funct3 selects this operand slot
                    } else {
                        pending += 1; // 2nd uniform in one op needs a preload
                    }
                }
                // position component → its pre-loaded GPR.
                Some(&Ubo::Pos) => r[k] = *pos_gpr.get(&(*s, c)).unwrap_or(&0),
                // fragment uniform (inv_area / per-vertex varying) → funct3 read.
                Some(&Ubo::Uniform(idx)) => {
                    if f3 == 0 {
                        r[k] = idx;
                        f3 = k as u32 + 1;
                    } else {
                        pending += 1; // 2nd uniform in one op needs a preload
                    }
                }
                // edge-function attribute in a fixed register (r0/r1/r2).
                Some(&Ubo::Fixed(reg)) => r[k] = reg,
                // attribute: u6=U, u7=V (sequencer DMA fills uniform[6,7]).
                Some(&Ubo::Attr) => {
                    if f3 == 0 {
                        r[k] = 6 + c;
                        f3 = k as u32 + 1;
                    } else {
                        pending += 1; // 2nd uniform in one op needs a preload
                    }
                }
                // ALU intermediate → allocated GPR (or pending if a stray input).
                None => match alloc.get(s) {
                    Some(&phys) => r[k] = phys,
                    None => pending += 1,
                },
            }
        }
        if let Some(w) = encode(i.mnem, rd, r[0], r[1], r[2], f3) {
            words.push(w);
        }
    }
    let n_body = words.len();

    if is_vertex && gl_position.is_some() {
        words.push(encode("FRCP", 4, 3, 0, 0, 0).unwrap()); // r4 = 1/clip_w
        words.push(encode("FMUL", 0, 0, 4, 0, 0).unwrap()); // screen_x
        words.push(encode("FMUL", 1, 1, 4, 0, 0).unwrap()); // screen_y
        words.push(encode("FMUL", 2, 2, 4, 0, 0).unwrap()); // ndc_z
        words.push(0x0000_0000); // HALT
    } else {
        words.push(0x0000_0000);
    }
    eprintln!(
        "borgc: encoded {} word(s) = {} uniform pre-load(s) + {} op(s) + {} epilogue \
         ({} operand(s) still pending)",
        words.len(),
        n_preload,
        n_body - n_preload,
        words.len() - n_body,
        pending
    );
    if !cx.cp.uniforms.is_empty() {
        let cs: Vec<String> = cx.cp.uniforms.iter().map(|(r, v)| format!("r{r}={v:#06x}")).collect();
        eprintln!("borgc: const regs (firmware-staged via MMIO): {}", cs.join(" "));
    }
    if !cx.fw.consts.is_empty() {
        let cs: Vec<String> = cx.fw.consts.iter().map(|(u, v)| format!("u{u}={v:#010x}")).collect();
        eprintln!(
            "borgc: draw-mode uniform consts (write to draw_fs_const_offset + 4*(u-20) before the draw): {}",
            cs.join(" ")
        );
    }
    if !cx.cp.vs_consts.is_empty() {
        let cs: Vec<String> = cx.cp.vs_consts.iter().map(|(u, v)| format!("u{u}={v:#010x}")).collect();
        eprintln!(
            "borgc: draw-mode vertex window consts (write to draw_vs_const_offset + 4*(u-25) before the draw): {}",
            cs.join(" ")
        );
    }
    if env::var("BORGC_DUMP_ISA").is_ok() {
        let hex: Vec<String> = words.iter().map(|w| format!("{w:#010x}")).collect();
        eprintln!("borgc:   first words: {}", hex.join(" "));
    }

    let outputs: Vec<u8> = if is_vertex && gl_position.is_some() {
        vec![0, 1, 2]
    } else if !is_vertex {
        // Fragment colour outputs r26/27/28 + r29 = interpolated depth.
        let mut o: Vec<u8> = (0..3).filter(|&c| frag_out[c].is_some()).map(|c| 26 + c as u8).collect();
        if frag_out[3].is_some() {
            o.push(24);
        }
        if frag_z.is_some() {
            o.push(29);
        }
        o
    } else {
        vec![]
    };
    let window: Vec<(u8, u32)> = if is_vertex {
        cx.cp.vs_consts.clone()
    } else {
        cx.fw.consts.iter().map(|&(u, v)| (u as u8, v)).collect()
    };
    let num_varyings: u8 = cx.sel.prog
        .iter()
        .filter(|i| i.mnem == "SOUT")
        .map(|i| i.swz[0] + 1)
        .max()
        .unwrap_or(0);
    let blob = emit_blob(&words, &outputs, &cx.cp.uniforms,
                         if draw_mode { Some((num_varyings, &window[..])) } else { None });
    eprintln!(
        "borgc: .borg blob = {} bytes ({} instr, {} output regs, {} window cx.consts, {} varyings)",
        blob.len(), words.len(), outputs.len(), window.len(), num_varyings
    );
    if let Ok(prefix) = env::var("BORGC_EMIT_BLOB") {
        let path = format!("{prefix}.{}.borg", if is_vertex { "vert" } else { "frag" });
        match std::fs::write(&path, &blob) {
            Ok(()) => eprintln!("borgc: wrote {path}"),
            Err(e) => eprintln!("borgc: failed to write {path}: {e}"),
        }
    }
    if !out_len.is_null() {
        *out_len = blob.len() as u32;
    }
    if !out_buf.is_null() && (blob.len() as u32) <= buf_cap {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), out_buf, blob.len());
    }
    total
}

