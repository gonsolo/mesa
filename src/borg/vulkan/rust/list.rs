// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Render lists built in the driver (docs/B1_geometry_front_end.md, "Render lists";
//! docs/B13_device_memory.md, step 2), as `software/borg/borg_core.c` built them, with the state
//! in a struct. They are built when a batch ends, into a `Prog` the command buffer keeps, which a
//! device replays at submit.

use crate::ffi::*;
use crate::object::VkBuf;
use crate::wire::Draw;

pub trait Gpu {
    fn reg(&mut self, off: u32, v: u32);
    fn read(&mut self, off: u32) -> u32;
    fn mem(&mut self, addr: u32, v: u32);
    /// The register once `(v & mask) == val`.
    fn wait(&mut self, off: u32, mask: u32, val: u32) -> u32 {
        loop {
            let v = self.read(off);
            if v & mask == val {
                return v;
            }
        }
    }
    /// Whether window `n` of the program is already done (resumed from a checkpoint).
    fn skip(&mut self, _n: u32) -> bool {
        false
    }
    /// Windows 0..n are done.
    fn done(&mut self, _n: u32) {}
    /// Strip `k` of `n` wrote the boxes of triangles t = k (mod n) below `setup_base`: take the
    /// other strips' into this one's cache.
    fn boxes(&mut self, _setup_base: u32, _tris: u32, _k: u32, _n: u32) {}
}

macro_rules! r {
    ($f:ident) => {
        std::mem::offset_of!(borg_gpu_t, $f) as u32
    };
}

const TEXEL_BASE: u32 = 0x7A0_0000;
const TEXEL_BYTES: u32 = 0x60_0000;
const SHIFT_MIN: u32 = 8;
const SHIFT_MAX: u32 = 10;
const LIST_MAX_REGS: usize = 40;
const TEX_SLOTS: u32 = 32;
const MEM_DATA: u32 = 256;
pub const PUSH_MAX_WORDS: u32 = 32;

/// What the driver records for the list builder.
#[derive(Clone, PartialEq, Debug)]
pub enum Cmd {
    Shader(u8, Vec<u8>),
    Push(u32, Vec<u32>),
    Blend(u32, u32),
    State([u32; 5]),
    Viewport([f32; 6], [u32; 4]),
    Target(u8, [f32; 4]),
    Mem(u32, Vec<u8>),
    Vattr([u32; 6]),
    Draw(Draw),
    Att(u8, [u8; 3]),
    Pass(u8, u8),
    Place([u32; 6], u32),
    Bias(u32, u32),
    Occ(bool, u32),
    TexChunk(u32, Vec<u8>, [u32; 3], [u32; 4]),
    TexSlot(u8, u8, u32),
    TexDesc([u32; 16], [u32; 4]),
    TexLevels([u32; 12]),
}

/// Word `i` of `b`, bytes past its end as zeros.
fn word(b: &[u8], i: usize) -> u32 {
    let mut w = [0u8; 4];
    let n = b.len().saturating_sub(i).min(4);
    w[..n].copy_from_slice(&b[i..i + n]);
    u32::from_le_bytes(w)
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn float_from_uint(v: u32) -> u32 {
    (v as f32).to_bits()
}

fn f32_to_fp16(f: u32) -> u16 {
    let (sign, exp, man) = ((f >> 16) & 0x8000, (f >> 23) & 0xFF, f & 0x7F_FFFF);
    if exp == 0xFF {
        return (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16;
    }
    let e = exp as i32 - 127 + 15;
    if e >= 31 {
        return (sign | 0x7BFF) as u16;
    }
    if e <= 0 {
        if e < -10 {
            return sign as u16;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let (mut half, rem, mid) = (man >> shift, man & ((1 << shift) - 1), 1 << (shift - 1));
        if rem > mid || (rem == mid && half & 1 != 0) {
            half += 1;
        }
        return (sign | half) as u16;
    }
    let (mut half, rem) = ((e as u32) << 10 | man >> 13, man & 0x1FFF);
    if rem > 0x1000 || (rem == 0x1000 && half & 1 != 0) {
        half += 1;
    }
    (sign | half) as u16
}

/// A wire flush code: the format, and bit 3 for an sRGB RGBA8 or BGRA8 attachment.
fn wire_format(code: u8) -> Option<(u32, u32)> {
    let srgb = (code >> 3 & 1) as u32;
    let c = (code & !8) as u32;
    (c <= 7 && (srgb == 0 || c == 1 || c == 2)).then_some((c, srgb))
}

fn flush_bytes(f: u32) -> u32 {
    match f {
        5 => 1,
        4 => 2,
        6 => 8,
        7 => 16,
        0 => 2,
        _ => 4,
    }
}

fn record_shift(num_varyings: u32) -> Option<u32> {
    let bytes = (48 + 3 * num_varyings) * 4;
    (SHIFT_MIN..=SHIFT_MAX).find(|&s| bytes <= 1 << s)
}

/// A SPIR-B blob's draw extension (docs/spirb.md): the varying components a vertex shader writes
/// and its constant window, u-index and word.
struct DrawExt {
    varyings: u32,
    window: Vec<(u32, u32)>,
}

/// None without the extension or for a malformed blob.
fn draw_ext(b: &[u8]) -> Option<DrawExt> {
    let h = b.get(..6)?;
    let n = h[0] as usize | (h[5] as usize >> 1) << 8;
    if n > DRAW_VERT_SHADER_MAX_WORDS as usize || h[1..5].iter().any(|&k| k > 32) || h[5] & 1 == 0 {
        return None;
    }
    let p = 6 + 4 * n + h[1..5].iter().map(|&k| k as usize).sum::<usize>() + 4 * h[4] as usize;
    let (varyings, nw) = (*b.get(p)? as u32, *b.get(p + 1)? as usize);
    if nw > 32 {
        return None;
    }
    let (regs, vals) = (b.get(p + 2..p + 2 + nw)?, b.get(p + 2 + nw..p + 2 + 5 * nw)?);
    Some(DrawExt { varyings, window: regs.iter().zip(vals.chunks_exact(4)).map(|(&r, v)| (r as u32, le32(v))).collect() })
}

fn blob_instrs(b: &[u8]) -> u32 {
    b[0] as u32 | ((b[5] >> 1) as u32) << 8
}

#[derive(Default)]
struct Stage {
    blob: Vec<u8>,
    code: u32,
    consts: u32,
}

#[derive(Default, Clone, Copy)]
struct DrawParams {
    topology: u32,
    index_type: u32,
    restart: u32,
    load: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
    index_base: u32,
    ubo_base: u32,
    /// The draw's buffer binding table: base address and offset limit per binding (borgc).
    table: u32,
    push_stage: u32,
    /// Points: the setup ROM's point path (DRAW_CFG bit 10).
    points: u32,
    vertex_offset: i32,
}

#[derive(Default)]
struct List {
    n: u32,
    top: u32,
    state: Vec<[u32; 2]>,
    state_addr: u32,
    table: u32,
    vtab_addr: u32,
    load: u32,
    shift: u32,
    tris: u32,
}

pub struct Builder {
    fb_w: u32,
    fb_h: u32,
    place: [u32; 6],
    pitch: u32,
    cull_cfg: u32,
    bin_base: u32,
    setup_base: u32,
    half_w: u32,
    half_h: u32,
    vert: Option<DrawExt>,
    frag: Option<DrawExt>,
    vertex_count: u32,
    dp: DrawParams,
    clear: [u16; 3],
    st: [u32; 6],
    raster_valid: bool,
    bias: [u32; 2],
    vp: Option<[u32; 8]>,
    sh: [Stage; 2],
    vtab: [[u32; 4]; BORG_MAX_VATTRS as usize],
    vtab_dirty: bool,
    natt: u32,
    att_fmt: u32,
    nocolor: u32,
    ms: u32,
    srgb: u32,
    flush_format: u32,
    tex_desc: Vec<[u32; 16]>,
    tex_slot: u32,
    samp_slot: u32,
    tex_base: u32,
    tex_n: u32,
    tex_lvl: [u32; 12],
    push_used: bool,
    /// The push words, and their copy in this list (0: none yet): each draw reads its own.
    push: [u32; PUSH_MAX_WORDS as usize],
    push_at: u32,
    list: List,
    occ: (bool, u32),
    window_tiles: u32,
}

impl Builder {
    pub fn new() -> Self {
        let wt = std::env::var("BORG_WINDOW_TILES").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0 && v <= 64).unwrap_or(64);
        Builder {
            fb_w: 0,
            fb_h: 0,
            place: [0; 6],
            pitch: 0,
            cull_cfg: 2 << CULL_CFG_REG_T__CULL_MODE_bp,
            bin_base: 0,
            setup_base: 0,
            half_w: 0,
            half_h: 0,
            vert: None,
            frag: None,
            vertex_count: 0,
            dp: DrawParams { instance_count: 1, ..Default::default() },
            clear: [0; 3],
            st: [0; 6],
            raster_valid: false,
            bias: [0; 2],
            vp: None,
            sh: Default::default(),
            vtab: [[0; 4]; BORG_MAX_VATTRS as usize],
            vtab_dirty: true,
            natt: 1,
            att_fmt: 0,
            nocolor: 0,
            ms: 0,
            srgb: 0,
            flush_format: 0,
            tex_desc: vec![[0; 16]; TEX_SLOTS as usize],
            tex_slot: 0,
            samp_slot: 0,
            tex_base: 0,
            tex_n: 1,
            tex_lvl: [0; 12],
            push_used: false,
            push: [0; PUSH_MAX_WORDS as usize],
            push_at: 0,
            list: List { top: BORG_LIST_ENTRY_BYTES, ..Default::default() },
            occ: (false, 0),
            window_tiles: wt,
        }
    }

    fn sample_cfg(&self) -> u32 {
        let c = self.cull_cfg;
        (if c >> 31 != 0 { (c >> 8) & 0x1F } else { 0xF | 1 << 6 }) | ((c >> 13) & 1) << 5
    }

    fn frame_stride_words(&self) -> u32 {
        self.fb_w * self.fb_h * flush_bytes(self.flush_format) / 4 * if self.ms != 0 { 4 } else { 1 } + 1
    }

    fn apply_layout(&mut self, g: &mut Prog) {
        let stride = self.frame_stride_words();
        g.reg(r!(flush_fb_base), if self.place[0] != 0 { self.place[0] } else { DRAM_SPI_BASE + DRAM_OUT_OFFSET });
        g.reg(r!(flush_format), self.flush_format | self.nocolor << 3 | self.srgb << 4);
        g.reg(r!(flush_width), 31 - self.fb_w.max(1).leading_zeros());
        let frames = if self.place[0] != 0 || self.pitch != 0 { 0 } else if stride > 0x10_0000 { 1 } else { 2 };
        self.bin_base = DRAM_SPI_BASE + DRAM_OUT_OFFSET + frames * stride * 4;
        let (tw, th) = self.window();
        // A word per triangle below the records: the box cache (SEQ_TILE_ROWS[13:12]).
        self.setup_base = self.bin_base + tw * th * TBR_BIN_ROW_BYTES + 4 * LIST_MAX_TRIS;
    }

    /// The render window in tiles: the target's width up to `window_tiles`, then as many rows as
    /// the bin table holds, at most 128 (the 9-bit window coordinates).
    fn window(&self) -> (u32, u32) {
        let tw = (self.fb_w >> 2).min(self.window_tiles).max(1);
        let th = (self.fb_h >> 2).min(self.window_tiles * self.window_tiles / tw).min(128).max(1);
        (tw, th)
    }

    /// The target size.
    pub fn init(&mut self, g: &mut Prog, w: u32, h: u32) {
        self.flush_format = 0;
        (self.fb_w, self.fb_h, self.place, self.pitch) = (w, h, [0; 6], 0);
        (self.half_w, self.half_h) = (float_from_uint(w / 2), float_from_uint(h / 2));
        self.apply_layout(g);
    }

    fn stage_shader(&mut self, g: &mut Prog, stage: usize, blob: &[u8]) {
        let n = blob_instrs(blob);
        if n > DRAW_VERT_SHADER_MAX_WORDS {
            return;
        }
        let (u0, words) = if stage == 0 { (DRAW_VS_CONST_U0, DRAW_VS_CONST_MAX_WORDS) } else { (DRAW_FS_CONST_U0, DRAW_FS_CONST_WORDS) };
        let Some(parsed) = draw_ext(blob) else { return };
        if parsed.window.iter().any(|&(r, _)| r < u0 || r >= u0 + words) {
            return;
        }
        let bytes = 6 + n as usize * 4;
        if bytes <= blob.len() && self.sh[stage].blob[..] != blob[..bytes] {
            self.sh[stage].blob = blob[..bytes].to_vec();
            self.sh[stage].code = 0;
        }
        let addr = if stage == 0 { DRAW_VERT_SHADER_SPI } else { DRAW_FRAG_SHADER_SPI };
        for i in 0..n {
            g.mem(addr + i * 4, le32(&blob[6 + 4 * i as usize..]));
        }
        if n < DRAW_VERT_SHADER_MAX_WORDS {
            g.mem(addr + n * 4, 0);
        }
        if stage == 0 {
            g.reg(r!(seq_vert_addr), DRAW_VERT_SHADER_SPI);
            g.reg(r!(seq_vert_len), n);
            self.vert = Some(parsed);
        } else {
            g.reg(r!(seq_frag_addr), DRAW_FRAG_SHADER_SPI);
            g.reg(r!(seq_frag_len), n);
            self.frag = Some(parsed);
        }
    }

    fn tex_desc_store(&mut self, g: &mut Prog, d: &[u32; 16]) {
        let s = self.tex_slot;
        self.tex_desc[s as usize] = *d;
        for (w, &v) in d.iter().enumerate() {
            g.mem(TEX_DESC_TABLE_ADDR + s * 64 + w as u32 * 4, v);
            g.mem(TEX_DESC_TABLE_ADDR + (BORG_VTEX_SLOT + s) * 64 + w as u32 * 4, v);
        }
        self.tex_n = self.tex_n.max(s + 1);
        self.vtab_dirty = true;
    }

    fn sampler_store(&self, g: &mut Prog, samp: &[u32]) {
        for (i, &v) in samp.iter().enumerate() {
            g.mem(SAMPLER_DESC_TABLE_ADDR + self.samp_slot * 16 + i as u32 * 4, v);
        }
        g.reg(r!(desc_flush), 0);
    }

    fn root_table(g: &mut Prog, at: u32, ls: u32, tex: u32, samp: u32, table: u32) -> u32 {
        for (i, v) in [ls, tex, samp, table].into_iter().enumerate() {
            g.mem(at + 4 * i as u32, v);
        }
        at
    }

    fn ls_base(&self, stage: u32) -> u32 {
        if self.dp.push_stage >> stage & 1 != 0 {
            self.push_at
        } else if self.dp.ubo_base != 0 {
            self.dp.ubo_base
        } else if self.push_used {
            self.push_at
        } else {
            DRAW_UBO_SPI
        }
    }

    fn alloc(&mut self, bytes: u32) -> u32 {
        let a = BORG_LIST_SPI + self.list.top;
        self.list.top += (bytes + 3) & !3;
        a
    }

    fn block(&mut self, g: &mut Prog, regs: &[[u32; 2]]) -> u32 {
        let a = self.alloc(4 + regs.len() as u32 * 8);
        g.mem(a, regs.len() as u32);
        for (i, [o, v]) in regs.iter().enumerate() {
            g.mem(a + 4 + i as u32 * 8, *o);
            g.mem(a + 8 + i as u32 * 8, *v);
        }
        a
    }

    fn list_stage(&mut self, g: &mut Prog, stage: usize) {
        if self.sh[stage].code != 0 {
            return;
        }
        let window = if stage == 0 { &self.vert } else { &self.frag }.as_ref().unwrap().window.clone();
        let n = blob_instrs(&self.sh[stage].blob);
        let (words, u0) = if stage == 0 { (DRAW_VS_CONST_MAX_WORDS, DRAW_VS_CONST_U0) } else { (DRAW_FS_CONST_WORDS, DRAW_FS_CONST_U0) };
        let a = self.alloc((n + 1) * 4);
        for i in 0..n {
            g.mem(a + i * 4, le32(&self.sh[stage].blob[6 + 4 * i as usize..]));
        }
        g.mem(a + n * 4, 0);
        let c = self.alloc(words * 4);
        for i in 0..words {
            g.mem(c + i * 4, 0);
        }
        for (r, v) in window {
            g.mem(c + (r - u0) * 4, v);
        }
        (self.sh[stage].code, self.sh[stage].consts) = (a, c);
    }

    fn setup_room_tris(&self, shift: u32) -> u32 {
        (BORG_HEAP_SPI.saturating_sub(self.setup_base) >> shift).min(LIST_MAX_TRIS)
    }

    fn list_reset(&mut self) {
        self.list = List { top: BORG_LIST_ENTRY_BYTES, ..Default::default() };
        self.sh[0].code = 0;
        self.sh[1].code = 0;
        self.vtab_dirty = true;
        self.push_at = 0;
    }

    /// Queues the draw the last `Cmd::Draw` described; a list draw larger than the setup records'
    /// room goes in parts.
    pub fn draw(&mut self, g: &mut Prog) {
        let (n, dp, bias) = (self.vertex_count, self.dp, self.bias);
        let room = self.vert.as_ref().and_then(|v| record_shift(v.varyings)).map_or(0, |s| self.setup_room_tris(s) / 3 * 3);
        if dp.topology != 0 || dp.index_type != 0 || dp.instance_count > 1 || room == 0 || n <= room {
            return self.draw_part(g);
        }
        for first in (0..n).step_by(room as usize) {
            (self.vertex_count, self.dp.first_vertex, self.bias) = (room.min(n - first), dp.first_vertex + first, bias);
            self.draw_part(g);
        }
        (self.vertex_count, self.dp) = (n, dp);
    }

    fn draw_part(&mut self, g: &mut Prog) {
        let (bias_c, bias_s) = (self.bias[0], self.bias[1]);
        self.bias = [0; 2];
        let Some(shift) = self.vert.as_ref().and_then(|v| record_shift(v.varyings)) else { return };
        if self.vertex_count == 0 || self.sh[0].blob.is_empty() || self.sh[1].blob.is_empty() || self.frag.is_none() {
            return;
        }
        let tris = self.vertex_count * self.dp.instance_count.max(1);
        let l = &self.list;
        if l.n != 0 && (shift != l.shift || l.tris + tris > self.setup_room_tris(shift) || l.n >= BORG_LIST_ENTRY_BYTES / 8 - 1 || l.top + 0x4000 > BORG_LIST_BYTES) {
            self.flush(g);
        }
        if self.push_used && self.push_at == 0 {
            let a = self.alloc(PUSH_MAX_WORDS * 4);
            for (i, &w) in self.push.iter().enumerate() {
                g.mem(a + 4 * i as u32, w);
            }
            self.push_at = a;
        }
        if self.list.n == 0 {
            (self.list.load, self.list.shift) = (self.dp.load, shift);
        }
        self.list.tris += tris;
        self.list_stage(g, 0);
        self.list_stage(g, 1);
        if self.vtab_dirty || self.list.vtab_addr == 0 {
            let a = self.alloc((BORG_MAX_VATTRS + self.tex_n) * 64);
            for s in 0..BORG_MAX_VATTRS {
                for w in 0..16 {
                    g.mem(a + s * 64 + w * 4, if w < 4 { self.vtab[s as usize][w as usize] } else { 0 });
                }
            }
            for k in 0..self.tex_n {
                for w in 0..16 {
                    g.mem(a + (BORG_MAX_VATTRS + k) * 64 + w * 4, self.tex_desc[k as usize][w as usize]);
                }
            }
            (self.list.vtab_addr, self.vtab_dirty) = (a, false);
        }
        let blen = |b: &[u8]| blob_instrs(b);
        let mut st: Vec<[u32; 2]> = Vec::with_capacity(LIST_MAX_REGS);
        st.push([r!(seq_vert_addr), self.sh[0].code]);
        st.push([r!(seq_vert_len), blen(&self.sh[0].blob)]);
        st.push([r!(seq_frag_addr), self.sh[1].code]);
        st.push([r!(seq_frag_len), blen(&self.sh[1].blob).min(63)]);
        st.push([r!(draw_vs_const), self.sh[0].consts]);
        st.push([r!(draw_fs_const), self.sh[1].consts]);
        st.push([r!(desc_flush), 0]);
        let root_i = st.len();
        st.push([r!(root), self.ls_base(1)]);
        st.push([r!(blend_cfg), self.st[0]]);
        st.push([r!(blend_const), self.st[1]]);
        if self.raster_valid {
            st.push([r!(stencil_cfg), self.st[2]]);
            st.push([r!(stencil_front), self.st[3]]);
            st.push([r!(stencil_back), self.st[4]]);
            st.push([r!(depth_cfg), self.st[5]]);
        }
        st.push([r!(cull_cfg), self.cull_cfg & 7]);
        st.push([r!(depth_bias_const), bias_c]);
        st.push([r!(depth_bias_slope), bias_s]);
        match self.vp {
            Some(v) => {
                for (f, val) in [r!(viewport_sx), r!(viewport_sy), r!(viewport_ox), r!(viewport_oy), r!(depth_scale), r!(depth_offset), r!(scissor_x), r!(scissor_y)].into_iter().zip(v) {
                    st.push([f, val]);
                }
            }
            None => {
                st.push([r!(viewport_sx), self.half_w]);
                st.push([r!(viewport_sy), self.half_h]);
                st.push([r!(viewport_ox), self.half_w]);
                st.push([r!(viewport_oy), self.half_h]);
                st.push([r!(depth_scale), 1.0f32.to_bits()]);
                st.push([r!(depth_offset), 0]);
                st.push([r!(scissor_y), 0]);
            }
        }
        let dp = self.dp;
        st.push([r!(draw_cfg), 1 | dp.topology << 1 | dp.index_type << 3 | dp.restart << 5 | shift << 6 | dp.points << 10]);
        if self.list.state_addr == 0 || self.list.state != st || self.list.table != dp.table {
            (self.list.state, self.list.table) = (st.clone(), dp.table);
            let at = self.alloc(16);
            st[root_i][1] = Self::root_table(g, at, self.ls_base(1), TEX_DESC_TABLE_ADDR, SAMPLER_DESC_TABLE_ADDR, dp.table);
            self.list.state_addr = self.block(g, &st);
        }
        let at = self.alloc(16);
        let root = Self::root_table(g, at, self.ls_base(0), self.list.vtab_addr - BORG_VATTR_SLOT0 * 64, SAMPLER_DESC_TABLE_ADDR, dp.table);
        let pr = [
            [r!(draw_vertex_count), self.vertex_count],
            [r!(draw_instance_count), dp.instance_count],
            [r!(draw_first_vertex), dp.first_vertex],
            [r!(draw_first_instance), dp.first_instance],
            [r!(draw_vertex_offset), dp.vertex_offset as u32],
            [r!(draw_index_base), dp.index_base],
            [r!(desc_flush), 0],
            [r!(root), root],
        ];
        let params = self.block(g, &pr);
        g.mem(BORG_LIST_SPI + self.list.n * 8, self.list.state_addr);
        g.mem(BORG_LIST_SPI + self.list.n * 8 + 4, params);
        self.list.n += 1;
    }

    fn trigger(&mut self, g: &mut Prog, list: u32, load: u32) {
        let cc_lo = (self.clear[2] as u32) << 16 | FP16_MAX_DEPTH;
        let cc_hi = (self.clear[0] as u32) << 16 | self.clear[1] as u32;
        g.reg(r!(seq_fb_base), if self.place[0] != 0 { self.place[0] } else { DRAM_SPI_BASE + DRAM_OUT_OFFSET });
        let fb_tiles = self.fb_w >> 2;
        let (tw, th) = self.window();
        g.reg(r!(seq_tiles_per_row), tw);
        g.reg(r!(seq_tile_rows), th);
        g.reg(r!(fb_pitch), if self.pitch != 0 { self.pitch } else { fb_tiles });
        g.reg(r!(seq_clear_lo), cc_lo);
        g.reg(r!(seq_clear_hi), cc_hi);
        g.reg(r!(seq_bin_base), self.bin_base);
        g.reg(r!(seq_bin_row_bytes), TBR_BIN_ROW_BYTES);
        g.reg(r!(seq_setup_base), self.setup_base);
        g.reg(r!(tile_bz), cc_lo);
        g.reg(r!(frag_pc), BORG_IMEM_FRAG_OFFSET);
        g.reg(r!(sample_mask_cfg), self.sample_cfg());
        g.reg(r!(tile_load), load);
        g.reg(r!(draw_cfg), 1 | self.list.shift << 6);
        g.reg(r!(render_list), list);
        g.reg(r!(att_cfg), if self.natt > 1 { (self.natt - 1) | 7 << 2 } else { 0 });
        if self.natt > 1 {
            g.reg(r!(att_format), self.att_fmt);
            for (k, reg) in [r!(att_base1), r!(att_base2), r!(att_base3)].into_iter().enumerate() {
                let a = if self.place[3 + k] != 0 { self.place[3 + k] } else { BORG_ATT_SPI + k as u32 * BORG_ATT_STRIDE };
                g.reg(reg, a);
            }
        }
        g.op(&[TILES, fb_tiles, self.fb_h >> 2, tw, th, self.list.tris, self.setup_base]);
        g.reg(r!(render_list), 0);
    }

    /// Renders what is queued.
    pub fn flush(&mut self, g: &mut Prog) {
        if self.list.n == 0 {
            return;
        }
        g.mem(BORG_LIST_SPI + self.list.n * 8, 0);
        g.mem(BORG_LIST_SPI + self.list.n * 8 + 4, 0);
        self.trigger(g, BORG_LIST_SPI, self.list.load);
        self.list_reset();
        if self.occ.0 {
            g.op(&[OCC, self.occ.1]);
        }
    }

    /// One command; false when it is malformed.
    pub fn apply(&mut self, g: &mut Prog, c: &Cmd) -> bool {
        if matches!(c, Cmd::TexChunk(..) | Cmd::Target(..) | Cmd::Pass(..) | Cmd::Att(..) | Cmd::Occ(..) | Cmd::Place(..) | Cmd::TexDesc(..)) {
            self.flush(g);
        }
        match c {
            Cmd::Shader(stage, blob) => {
                if *stage > 1 || blob.len() < 6 {
                    return false;
                }
                self.stage_shader(g, *stage as usize, blob);
            }
            Cmd::Push(off, words) => {
                let (off, nw) = (*off, words.len() as u32);
                if nw < 1 || nw > PUSH_MAX_WORDS || off >= PUSH_MAX_WORDS || nw > PUSH_MAX_WORDS - off {
                    return false;
                }
                let nw = nw.min(BORG_PUSH_CONST_MAX_WORDS - off);
                self.push[off as usize..(off + nw) as usize].copy_from_slice(&words[..nw as usize]);
                (self.push_used, self.push_at) = (true, 0);
            }
            &Cmd::Blend(cfg, k) => {
                g.reg(r!(blend_cfg), cfg);
                g.reg(r!(blend_const), k);
                (self.st[0], self.st[1]) = (cfg, k);
            }
            Cmd::State(s) => {
                g.reg(r!(stencil_cfg), s[0]);
                g.reg(r!(stencil_front), s[1]);
                g.reg(r!(stencil_back), s[2]);
                g.reg(r!(depth_cfg), s[3]);
                self.cull_cfg = s[4];
                self.st[2..6].copy_from_slice(&s[..4]);
                self.raster_valid = true;
            }
            Cmd::Viewport(f, sc) => {
                let v = [f[2] / 2.0, f[3] / 2.0, f[0] + f[2] / 2.0, f[1] + f[3] / 2.0, f[5] - f[4], f[4]];
                let [x0, y0, x1, y1] = *sc;
                if x0 > x1 || y0 > y1 || x1 > 4096 || y1 > 4096 {
                    return false;
                }
                let b = v.map(f32::to_bits);
                self.vp = Some([b[0], b[1], b[2], b[3], b[4], b[5], x0 | x1 << 13, y0 | y1 << 13 | 1 << 26]);
            }
            &Cmd::Occ(on, off) => {
                if off & 3 != 0 || off >= BORG_HEAP_BYTES {
                    return false;
                }
                self.occ = (on, BORG_HEAP_SPI + off);
                g.op(&[OCC_BEGIN]);
                if on {
                    g.mem(self.occ.1, 0);
                }
                g.reg(r!(occ_ctrl), if on { 3 } else { 0 });
            }
            &Cmd::Bias(k, s) => self.bias = [k, s],
            Cmd::Target(f, clear) => {
                let Some((fmt, srgb)) = wire_format(*f) else { return false };
                if fmt != self.flush_format || srgb != self.srgb {
                    (self.flush_format, self.srgb) = (fmt, srgb);
                    self.apply_layout(g);
                }
                self.clear = [0, 1, 2].map(|k| f32_to_fp16(clear[k].to_bits()));
            }
            Cmd::Att(n, f) => {
                if *n < 1 || *n > 4 {
                    return false;
                }
                let mut att = 0;
                for k in 0..3 {
                    let Some((fmt, srgb)) = wire_format(f[k]) else { return false };
                    att |= fmt << (3 * k) | srgb << (9 + k);
                }
                (self.natt, self.att_fmt) = (*n as u32, att);
            }
            Cmd::Place(at, pitch) => {
                self.place = *at;
                self.pitch = *pitch;
                if self.pitch > 4095 {
                    return false;
                }
                self.apply_layout(g);
                if self.place[1] != 0 {
                    g.reg(r!(flush_zb_base), self.place[1]);
                }
                if self.place[2] != 0 {
                    g.reg(r!(flush_sb_base), self.place[2]);
                }
            }
            &Cmd::Pass(f, flags) => {
                let Some((fmt, srgb)) = wire_format(f) else { return false };
                if flags > 63 {
                    return false;
                }
                let (nocolor, ms) = ((flags >> 3 & 1) as u32, (flags >> 4 & 1) as u32);
                if fmt != self.flush_format || srgb != self.srgb || nocolor != self.nocolor || ms != self.ms {
                    (self.flush_format, self.srgb, self.nocolor, self.ms) = (fmt, srgb, nocolor, ms);
                    self.apply_layout(g);
                }
                let pl = |k: usize, fixed: u32, b: &Self| if b.place[k] != 0 { b.place[k] } else { fixed };
                g.reg(r!(flush_zb_base), if flags & 1 != 0 { pl(1, BORG_ZB_SPI, self) } else { 0 });
                g.reg(r!(flush_sb_base), if flags & 2 != 0 { pl(2, BORG_SB_SPI, self) } else { 0 });
                g.reg(r!(depth_format), (flags >> 2 & 1) as u32 | ((flags >> 5 & 1) as u32) << 1);
                g.reg(r!(attach_ms), self.ms);
            }
            Cmd::Mem(off, data) => {
                let (off, nb) = (*off, data.len() as u32);
                if nb > MEM_DATA || off & 3 != 0 || off + nb > BORG_HEAP_BYTES {
                    return false;
                }
                for i in (0..nb).step_by(4) {
                    g.mem(BORG_HEAP_SPI + off + i, word(data, i as usize));
                }
            }
            &Cmd::Vattr([slot, fmt, base, count, stride, swz]) => {
                let base = BORG_HEAP_SPI.wrapping_add(base);
                if slot >= BORG_MAX_VATTRS || fmt == 0 || fmt > 51 || count < 1 || count > 4096 * 4096 || stride == 0 || stride > 16 {
                    return false;
                }
                let rows = count.div_ceil(4096);
                let w1 = 4095 | (rows - 1) << 16 | 1 << 28 | 1 << 30;
                let d = TEX_DESC_TABLE_ADDR + (BORG_VATTR_SLOT0 + slot) * 64;
                let words = [base, w1, fmt << 14 | (swz & 0xFFF0_0000), stride << 12];
                self.vtab[slot as usize] = words;
                self.vtab_dirty = true;
                for (i, &v) in words.iter().enumerate() {
                    g.mem(d + 4 * i as u32, v);
                }
                for i in 4..16 {
                    g.mem(d + i * 4, 0);
                }
                g.reg(r!(desc_flush), 0);
            }
            Cmd::Draw(d) => {
                if d.topology > 2 || d.index_type > 2 || d.restart > 255 {
                    return false;
                }
                let (p3, w) = (d.restart, d.words);
                let load = p3 >> 1 & 15;
                self.dp = DrawParams {
                    topology: d.topology,
                    index_type: d.index_type,
                    restart: p3 & 1,
                    load: if load == 1 { 9 } else { load },
                    push_stage: p3 >> 5 & 3,
                    points: p3 >> 7 & 1,
                    instance_count: w[1],
                    first_vertex: w[2],
                    first_instance: w[3],
                    vertex_offset: w[4] as i32,
                    index_base: if d.index_type != 0 { w[5] } else { 0 },
                    ubo_base: if w[6] != 0 { BORG_HEAP_SPI + w[6] - 1 } else { 0 },
                    table: if w[7] != 0 { BORG_HEAP_SPI + w[7] - 1 } else { 0 },
                };
                self.vertex_count = w[0];
                (self.tex_slot, self.samp_slot, self.tex_base) = (0, 0, 0);
            }
            Cmd::TexLevels(l) => self.tex_lvl = *l,
            &Cmd::TexSlot(t, s, base) => {
                if t as u32 >= TEX_SLOTS || s as u32 >= TEX_SLOTS || base & 255 != 0 || base >= TEXEL_BYTES {
                    return false;
                }
                (self.tex_slot, self.samp_slot, self.tex_base) = (t as u32, s as u32, base);
            }
            Cmd::TexChunk(off, data, d3, samp) => {
                let (off, nb) = (*off, data.len() as u32);
                if nb > BORGVK_TEXG_CHUNK || off & 3 != 0 || self.tex_base + off + nb > TEXEL_BYTES {
                    return false;
                }
                if (d3[1] >> 10) & 15 == 0 {
                    self.tex_lvl = [0; 12];
                }
                let mut d = [0u32; 16];
                d[0] = TEXEL_BASE + self.tex_base;
                d[1..4].copy_from_slice(d3);
                d[4..].copy_from_slice(&self.tex_lvl);
                self.tex_desc_store(g, &d);
                self.sampler_store(g, samp);
                for i in (0..nb).step_by(4) {
                    g.mem(TEXEL_BASE + self.tex_base + off + i, word(data, i as usize));
                }
            }
            Cmd::TexDesc(d, samp) => {
                self.tex_desc_store(g, d);
                self.sampler_store(g, samp);
            }
        }
        true
    }
}

const MEM: u32 = 0;
const REG: u32 = 1;
const TILES: u32 = 2;
/// The most triangles a list holds.
const LIST_MAX_TRIS: u32 = 60000;
const OCC_BEGIN: u32 = 3;
const OCC: u32 = 4;

/// What the builder writes, kept with the command buffer: memory, registers, the renders of the
/// tile windows and the occlusion counts. A device replays it as one strip of tile rows.
pub struct Prog {
    buf: VkBuf,
    at: u32,
    run: Vec<u32>,
    pub ok: bool,
}

// Shared only to be read, by the threads of the strips.
unsafe impl Sync for Prog {}

impl Prog {
    fn put(&mut self, w: &[u32]) {
        self.ok &= self.buf.extend(unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), w.len() * 4) });
    }

    fn end_mem(&mut self) {
        if !self.run.is_empty() {
            let run = std::mem::take(&mut self.run);
            self.put(&[MEM, self.at, run.len() as u32]);
            self.put(&run);
        }
    }

    fn mem(&mut self, addr: u32, v: u32) {
        if !self.run.is_empty() && addr != self.at + 4 * self.run.len() as u32 {
            self.end_mem();
        }
        if self.run.is_empty() {
            self.at = addr;
        }
        self.run.push(v);
    }

    fn reg(&mut self, off: u32, v: u32) {
        self.op(&[REG, off, v]);
    }

    fn op(&mut self, w: &[u32]) {
        self.end_mem();
        self.put(w);
    }

    /// Replays on `g` as strip `k` of `n`, `per` tile rows each.
    /// The program's words, for the pass cache's fingerprint.
    pub fn words(&self) -> (&[u8], &[u32]) {
        (self.buf.as_slice(), &self.run)
    }

    pub fn run(&self, g: &mut dyn Gpu, (k, n, per): (u32, u32, u32)) {
        let b = self.buf.as_slice();
        let w = |i: usize| le32(&b[4 * i..]);
        let (mut i, mut occ) = (0, 0u32);
        // Checkpoints stop at the first query: its count lives across windows.
        let (mut win, mut queries) = (0u32, false);
        while 4 * i < b.len() {
            match w(i) {
                MEM => {
                    let (a, m) = (w(i + 1), w(i + 2) as usize);
                    for j in 0..m {
                        g.mem(a + 4 * j as u32, w(i + 3 + j));
                    }
                    i += 3 + m;
                }
                REG => {
                    g.reg(w(i + 1), w(i + 2));
                    i += 3;
                }
                TILES => {
                    let (fb_tiles, fb_rows, tw, th) = (w(i + 1), w(i + 2), w(i + 3), w(i + 4));
                    let (tris, setup_base) = (w(i + 5), w(i + 6));
                    let (y0, rows) = if n > 1 { (k * per, per.min(fb_rows.saturating_sub(k * per))) } else { (0, fb_rows) };
                    // Strips share the boxes: each runs the vertex shaders of every n-th triangle
                    // (mode 3, one tile row, nothing binned), then every window skips what misses it.
                    let shared = n > 1 && tris > 0;
                    if shared {
                        g.reg(r!(seq_tile_rows), 1 | 3 << 12 | k << 16 | n << 21);
                        g.reg(r!(fb_origin), y0 << 16);
                        g.reg(r!(seq_trigger), 1);
                        g.wait(r!(status), STATUS_REG_T__SEQ_BUSY_bm, 0);
                        g.wait(r!(status), STATUS_REG_T__IDLE_bm, STATUS_REG_T__IDLE_bm);
                        g.boxes(setup_base, tris, k, n);
                    }
                    // More than one window: the first writes the box cache, the others skip what misses them.
                    let many = rows.div_ceil(th.max(1)) * fb_tiles.div_ceil(tw.max(1)) > 1;
                    let mut box_mode = if shared { 2 } else if many { 1 } else { 0 };
                    for wy in (y0..y0 + rows).step_by(th.max(1) as usize) {
                        for wx in (0..fb_tiles).step_by(tw.max(1) as usize) {
                            let mode = box_mode;
                            box_mode = if shared || many { 2 } else { 0 };
                            win += 1;
                            if g.skip(win - 1) {
                                continue;
                            }
                            g.reg(r!(seq_tile_rows), (y0 + rows - wy).min(th) | mode << 12);
                            g.reg(r!(fb_origin), wx | wy << 16);
                            let t0 = std::time::Instant::now();
                            g.reg(r!(seq_trigger), 1);
                            g.wait(r!(status), STATUS_REG_T__SEQ_BUSY_bm, 0);
                            if crate::hw::env_set("BORGVK_PERF") {
                                // total, frag, flush, stall, dma (32 bits: the time covers a wrap)
                                let p: Vec<u32> = (0..5).map(|j| g.read(r!(perf_total) + 4 * j)).collect();
                                eprintln!("[perf] win {win} at {wx},{wy} mode {mode} {:?} ms {}", p, t0.elapsed().as_millis());
                            }
                            if !queries {
                                g.done(win);
                            }
                        }
                    }
                    g.wait(r!(status), STATUS_REG_T__IDLE_bm, STATUS_REG_T__IDLE_bm);
                    i += 7;
                }
                OCC_BEGIN => {
                    (occ, queries) = (0, true);
                    i += 1;
                }
                _ => {
                    occ = occ.wrapping_add(g.read(r!(occ_count)));
                    g.mem(w(i + 1), occ);
                    g.reg(r!(occ_ctrl), 3);
                    i += 2;
                }
            }
        }
    }
}

/// The open batch: its commands not yet built, the last draw state sent, its lists and its corpus
/// key.
pub struct Batch {
    lists: Option<(Builder, Prog)>,
    pub pending: Vec<Cmd>,
    pub sent: Vec<Cmd>,
    key: u64,
}

impl Batch {
    pub fn new() -> Box<Self> {
        Box::new(Batch { lists: None, pending: Vec::new(), sent: Vec::new(), key: BORGVK_CORPUS_SEED as u64 })
    }

    /// The pending commands into the lists of a `w` x `h` pass, in a buffer from `alloc`; without
    /// a size the pass is not rendered.
    pub fn build(&mut self, size: Option<(u32, u32)>, alloc: *const VkAllocationCallbacks) {
        if crate::corpus::borgvk_corpus_on() {
            self.key = self.pending.iter().fold(self.key, |h, c| crate::corpus::hash(h, format!("{c:?}").as_bytes()));
        }
        if let (None, Some((w, h))) = (&self.lists, size) {
            let (mut b, mut g) = (Builder::new(), Prog { buf: unsafe { VkBuf::new(alloc) }, at: 0, run: Vec::new(), ok: true });
            b.init(&mut g, w, h);
            self.lists = Some((b, g));
        }
        if let Some((b, g)) = self.lists.as_mut() {
            for c in &self.pending {
                if b.apply(g, c) && matches!(c, Cmd::Draw(_)) {
                    b.draw(g);
                }
            }
        }
        self.pending.clear();
    }

    pub fn end(self) -> (Option<Prog>, u64) {
        let prog = self.lists.map(|(mut b, mut g)| {
            b.flush(&mut g);
            g.end_mem();
            g
        });
        (prog, self.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_rounds_like_the_firmware() {
        assert_eq!(f32_to_fp16(1.0f32.to_bits()), 0x3C00);
        assert_eq!(f32_to_fp16(65504.0f32.to_bits()), 0x7BFF);
        assert_eq!(f32_to_fp16(1e9f32.to_bits()), 0x7BFF);
        assert_eq!(f32_to_fp16(6e-8f32.to_bits()), 1);
    }

    #[test]
    fn draw_extension() {
        // Two instructions, one uniform, no constants; 3 varyings and a window of u25 = 7.
        let mut b = vec![2, 1, 0, 0, 0, 1];
        b.extend([0; 8]);
        b.extend([4, 3, 1, 25, 7, 0, 0, 0]);
        let e = draw_ext(&b).unwrap();
        assert_eq!((e.varyings, e.window), (3, vec![(25, 7)]));
        assert!(draw_ext(&b[..b.len() - 1]).is_none());
        b[5] = 0;
        assert!(draw_ext(&b).is_none());
    }

    #[test]
    fn words_pad_with_zeros() {
        assert_eq!(word(&[1, 2, 3, 4, 5], 0), 0x0403_0201);
        assert_eq!(word(&[1, 2, 3, 4, 5], 4), 5);
    }

    #[test]
    fn record_shifts() {
        assert_eq!(record_shift(0), Some(8));
        assert_eq!(record_shift(16), Some(9));
        assert_eq!(record_shift(200), None);
    }
}
