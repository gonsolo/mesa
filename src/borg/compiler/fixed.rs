use crate::state::Cx;
use std::collections::HashMap;

pub fn fixed_regs(
    cx: &Cx,
    pos_out: &[Option<u32>; 4],
    frag_out: &[Option<u32>; 4],
    frag_z: Option<u32>,
) -> (HashMap<u32, u8>, Vec<u8>) {
    let is_vertex = cx.is_vertex;
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
        extra_reserved.extend_from_slice(&[5, 6, 7, 10, 11, 12]);
        if cx.frag_coord_w {
            extra_reserved.push(8);
        }

        extra_reserved.extend_from_slice(&cx.cp.regs[..cx.cp.count]);
        if frag_out[3].is_some() {
            extra_reserved.push(24);
        }
    }
    (forced, extra_reserved)
}
