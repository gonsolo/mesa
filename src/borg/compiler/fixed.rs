use crate::regalloc::regalloc;
use crate::state::{ShaderState, Stage};
use crate::{Ubo, NO_DST};
use std::env;
use std::collections::HashMap;

pub fn fixed_regs(
    cx: &ShaderState,
    pos_out: &[Option<u32>; 4],
    frag_out: &[Option<u32>; 4],
    frag_z: Option<u32>,
) -> (HashMap<u32, u8>, Vec<u8>) {
    let is_vertex = cx.stage == Stage::Vertex;
    let mut forced: HashMap<u32, u8> = HashMap::new();
    let mut extra_reserved: Vec<u8> = Vec::new();
    if is_vertex {
        // gl_Position components → r0..r3.
        for (c, v) in pos_out.iter().enumerate() {
            if let Some(def) = v {
                forced.insert(*def, c as u8);
            }
        }
        // A typed vertex fetch (TEX) writes rd..rd+3: one block, r20..r23, for all of them.
        for &t in &cx.tex_dsts {
            forced.insert(t, 20);
        }
        if !cx.tex_dsts.is_empty() {
            extra_reserved.extend_from_slice(&[21, 22, 23]);
        }
        extra_reserved.extend_from_slice(cx.cp.preset_regs());
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
        let reads_bary = cx.sel.prog.iter().flat_map(|i| i.srcs.iter()).any(|v| matches!(cx.sel.ubo.get(v), Some(Ubo::Fixed(5..=7))));
        if reads_bary {
            extra_reserved.extend_from_slice(&[5, 6, 7]);
        }
        if !cx.fattr_dsts.is_empty() {
            extra_reserved.extend_from_slice(&[10, 11, 12]);
        }
        if cx.frag_coord_w {
            extra_reserved.push(8);
        }

        extra_reserved.extend_from_slice(cx.cp.preset_regs());
        if frag_out[3].is_some() {
            extra_reserved.push(24);
        }
    }
    (forced, extra_reserved)
}

pub fn allocate(cx: &ShaderState, forced: &HashMap<u32, u8>, extra_reserved: &mut Vec<u8>) -> Option<HashMap<u32, u8>> {
    let is_fragment = cx.stage == Stage::Fragment;
    let soft: Vec<u8> = if is_fragment { vec![20, 21, 22, 23, 24, 26, 27, 28] } else { Vec::new() };
    let mut block_len: HashMap<u32, u8> = HashMap::new();
    for &t in &cx.tex_dsts {
        block_len.insert(t, 4);
    }
    extra_reserved.retain(|r| !soft.contains(r));
    let alloc = regalloc(&cx.sel.prog, forced, extra_reserved, &soft, &block_len);
    // A value without a register would be read and written as r0.
    if let Some(i) = cx.sel.prog.iter().find(|i| i.dst != NO_DST && !alloc.contains_key(&i.dst)) {
        let mut res: Vec<u8> = extra_reserved.clone();
        res.sort();
        res.dedup();
        eprintln!("borgc: ERROR out of registers ({} has none; reserved {:?}), refusing to emit a wrong shader", i.mnem, res);
        return None;
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
    Some(alloc)
}
