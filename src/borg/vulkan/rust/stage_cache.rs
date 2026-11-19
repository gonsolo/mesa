// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! A pipeline's compiled stages by everything their compilation reads, for the device's lifetime:
//! vkQuake builds all its pipelines again on a video restart.

use crate::device::Scratch;
use crate::entrypoint::slice;
use crate::ffi::*;
use crate::shader_cache;
use crate::types::ShaderBlob;

#[derive(Clone)]
pub struct Stages {
    blobs: Vec<(Vec<u8>, Vec<u32>, Vec<u32>)>,
    tex_pairs: Vec<[u8; 2]>,
    tex_binding: [i32; 2],
    ubo: [u128; 2],
    ubo_window: [u128; 2],
    push: [bool; 2],
    pwin: [bool; 2],
    spill: [bool; 2],
    sample_mask: bool,
    reads_pntc: bool,
}

fn put(k: &mut Vec<u8>, words: &[u32]) {
    words.iter().for_each(|w| k.extend(w.to_ne_bytes()));
}

#[cfg(borgc)]
fn layout(sc: &Scratch, k: &mut Vec<u8>) {
    use crate::compiler::{compile_layout, set_map};
    let Some(pl) = compile_layout(sc) else {
        return k.push(0);
    };
    put(k, &[pl.set_count]);
    for s in 0..pl.set_count as usize {
        match set_map(pl, s) {
            Some(m) => {
                k.push(m.n);
                put(k, &m.num);
                k.extend(m.slot);
                k.extend(m.count);
                k.extend([m.total, m.arrays as u8]);
            }
            None => k.push(0xFE),
        }
    }
}

#[cfg(not(borgc))]
fn layout(_sc: &Scratch, _k: &mut Vec<u8>) {}

/// The key: the stages' SPIR-V, entry points and specialization, the compile options, the blend
/// and vertex-pull lowering and the layout's descriptor maps. None compiles without the cache.
pub fn key(sc: &Scratch, stages: &[VkPipelineShaderStageCreateInfo], opts: &[u32], pull: Option<&borg_vertex_pull>) -> Option<Vec<u8>> {
    if !cfg!(borgc) || shader_cache::active() {
        return None;
    }
    let mut k = Vec::new();
    put(&mut k, opts);
    put(&mut k, &[sc.frag_select]);
    match unsafe { sc.blend_lower.as_ref() } {
        Some(b) => k.extend(format!("{b:?}").bytes()),
        None => k.push(0),
    }
    match pull {
        Some(p) => p.attr.iter().for_each(|a| put(&mut k, &[a.format as u32, a.offset, a.stride, a.binding, a.instance as u32])),
        None => k.push(0),
    }
    layout(sc, &mut k);
    for s in stages {
        let code = shader_cache::code(s)?;
        put(&mut k, &[code.len() as u32, s.stage, s.flags]);
        k.extend(code);
        k.extend(unsafe { std::ffi::CStr::from_ptr(s.pName) }.to_bytes_with_nul());
        if let Some(sp) = unsafe { s.pSpecializationInfo.as_ref() } {
            for e in unsafe { slice(sp.pMapEntries, sp.mapEntryCount) } {
                put(&mut k, &[e.constantID, e.offset, e.size as u32]);
            }
            if !sp.pData.is_null() {
                k.extend(unsafe { std::slice::from_raw_parts(sp.pData.cast::<u8>(), sp.dataSize) });
            }
        }
        k.push(0xFF);
    }
    Some(k)
}

fn save(sc: &Scratch) -> Stages {
    let blob = |b: &ShaderBlob| {
        let l = (b.len as usize).min(b.data.len());
        (b.data[..l].to_vec(), b.pool[..b.npool as usize].to_vec(), b.cdata[..b.ncdata as usize].to_vec())
    };
    Stages {
        blobs: sc.shader_blob.iter().map(blob).collect(),
        tex_pairs: sc.tex_pairs.clone(),
        tex_binding: [sc.vert_tex_binding, sc.frag_tex_binding],
        ubo: sc.stage_ubo,
        ubo_window: sc.stage_ubo_window,
        push: sc.stage_push,
        pwin: sc.stage_pwin,
        spill: sc.stage_spill,
        sample_mask: sc.frag_sample_mask,
        reads_pntc: sc.frag_reads_pntc,
    }
}

fn restore(sc: &mut Scratch, s: &Stages) {
    for (b, (data, pool, cdata)) in sc.shader_blob.iter_mut().zip(&s.blobs) {
        b.data[..data.len()].copy_from_slice(data);
        b.len = data.len() as u32;
        b.pool[..pool.len()].copy_from_slice(pool);
        b.npool = pool.len() as u32;
        b.cdata[..cdata.len()].copy_from_slice(cdata);
        b.ncdata = cdata.len() as u32;
    }
    sc.tex_pairs = s.tex_pairs.clone();
    [sc.vert_tex_binding, sc.frag_tex_binding] = s.tex_binding;
    sc.stage_ubo = s.ubo;
    sc.stage_ubo_window = s.ubo_window;
    sc.stage_push = s.push;
    sc.stage_pwin = s.pwin;
    sc.stage_spill = s.spill;
    sc.frag_sample_mask = s.sample_mask;
    sc.frag_reads_pntc = s.reads_pntc;
}

/// The stages under `key` if they were compiled before, else `compile` and keep them.
pub fn compiled(sc: &mut Scratch, key: Option<Vec<u8>>, compile: impl FnOnce(&mut Scratch)) {
    if let Some(s) = key.as_ref().and_then(|k| sc.compiled.get(k)).cloned() {
        restore(sc, &s);
        return;
    }
    compile(sc);
    if let Some(k) = key {
        let s = save(sc);
        sc.compiled.insert(k, s);
    }
}
