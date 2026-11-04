// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Render lists built in the driver (docs/B1_geometry_front_end.md, "Render lists";
//! docs/B13_device_memory.md, step 2), as `software/borg/borg_core.c` built them, with the state
//! in a struct. The GPU is three primitives, as there.

use crate::ffi::*;

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

/// A packet's length from its marker, 0 for an unknown one.
pub fn packet_len(marker: u8) -> usize {
    (match marker {
        0xB0 => BC_PKT_LEN_SHADER,
        0xC1 => BC_PKT_LEN_SHADER_BIG,
        0xC3 => BC_PKT_LEN_SHADER_HUGE,
        0xB2 => BC_PKT_LEN_PUSH,
        0xB3 => BC_PKT_LEN_BLEND,
        0xB4 => BC_PKT_LEN_STATE,
        0xB5 => BC_PKT_LEN_TEXG,
        0xC2 => BC_PKT_LEN_VP,
        0xB6 => BC_PKT_LEN_TARGET,
        0xB8 => BC_PKT_LEN_MEM,
        0xBB => BC_PKT_LEN_PASS,
        0xC0 => BC_PKT_LEN_TEXL,
        0xC4 => BC_PKT_LEN_TEXS,
        0xC5 => BC_PKT_LEN_BIAS,
        0xC6 => BC_PKT_LEN_OCC,
        0xBF => BC_PKT_LEN_ATT,
        0xC7 => BC_PKT_LEN_PLACE,
        0xC8 => BC_PKT_LEN_TEXD,
        0xB9 => BC_PKT_LEN_VATTR,
        0xBA => BC_PKT_LEN_DRAW,
        _ => 0,
    }) as usize
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
    push_stage: u32,
    vertex_offset: i32,
}

#[derive(Default)]
struct List {
    n: u32,
    top: u32,
    state: Vec<[u32; 2]>,
    state_addr: u32,
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
    vert: Option<spirb_shader_t>,
    frag: Option<spirb_shader_t>,
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
    list: List,
    strip: (u32, u32, u32),
    occ: (bool, u32, u32),
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
            tex_desc: vec![[0; 16]; BC_TEX_SLOTS as usize],
            tex_slot: 0,
            samp_slot: 0,
            tex_base: 0,
            tex_n: 1,
            tex_lvl: [0; 12],
            push_used: false,
            list: List { top: BORG_LIST_ENTRY_BYTES, ..Default::default() },
            strip: (0, 1, 0),
            occ: (false, 0, 0),
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

    fn apply_layout(&mut self, g: &mut dyn Gpu) {
        let stride = self.frame_stride_words();
        g.reg(r!(flush_fb_base), if self.place[0] != 0 { self.place[0] } else { DRAM_SPI_BASE + DRAM_OUT_OFFSET });
        g.reg(r!(flush_format), self.flush_format | self.nocolor << 3 | self.srgb << 4);
        g.reg(r!(flush_width), 31 - self.fb_w.max(1).leading_zeros());
        let frames = if self.place[0] != 0 || self.pitch != 0 { 0 } else if stride > 0x10_0000 { 1 } else { 2 };
        self.bin_base = DRAM_SPI_BASE + DRAM_OUT_OFFSET + frames * stride * 4;
        let (ww, wh) = (self.fb_w.min(256), self.fb_h.min(256));
        self.setup_base = self.bin_base + (ww >> 2) * (wh >> 2) * TBR_BIN_ROW_BYTES;
    }

    /// The target size, as the 0xBC request sets it.
    pub fn init(&mut self, g: &mut dyn Gpu, w: u32, h: u32) {
        self.flush_format = 0;
        (self.fb_w, self.fb_h, self.place, self.pitch) = (w, h, [0; 6], 0);
        (self.half_w, self.half_h) = (float_from_uint(w / 2), float_from_uint(h / 2));
        self.apply_layout(g);
    }

    /// The clear colour (FP16 RGB) of a render whose stream sets no target.
    pub fn set_clear(&mut self, c: [u16; 3]) {
        self.clear = c;
    }

    pub fn flush_format(&self) -> u32 {
        self.flush_format
    }

    pub fn set_strip(&mut self, k: u32, n: u32, rows: u32) {
        self.strip = (k, n.max(1), rows);
    }

    fn stage_shader(&mut self, g: &mut dyn Gpu, stage: usize, blob: &[u8]) {
        let n = blob_instrs(blob);
        if n > DRAW_VERT_SHADER_MAX_WORDS {
            return;
        }
        let mut parsed: spirb_shader_t = unsafe { std::mem::zeroed() };
        let (u0, words) = if stage == 0 { (DRAW_VS_CONST_U0, DRAW_VS_CONST_MAX_WORDS) } else { (DRAW_FS_CONST_U0, DRAW_FS_CONST_WORDS) };
        let pr = unsafe { spirb_parse(blob.as_ptr(), &mut parsed) };
        if pr < 0 || parsed.has_draw_ext == 0 {
            return;
        }
        if parsed.window_regs[..parsed.num_window as usize].iter().any(|&r| (r as u32) < u0 || r as u32 >= u0 + words) {
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

    fn tex_desc_store(&mut self, g: &mut dyn Gpu, d: &[u32; 16]) {
        let s = self.tex_slot;
        self.tex_desc[s as usize] = *d;
        for (w, &v) in d.iter().enumerate() {
            g.mem(TEX_DESC_TABLE_ADDR + s * 64 + w as u32 * 4, v);
            g.mem(TEX_DESC_TABLE_ADDR + (BORG_VTEX_SLOT + s) * 64 + w as u32 * 4, v);
        }
        self.tex_n = self.tex_n.max(s + 1);
        self.vtab_dirty = true;
    }

    fn sampler_store(&self, g: &mut dyn Gpu, samp: &[u32]) {
        for (i, &v) in samp.iter().enumerate() {
            g.mem(SAMPLER_DESC_TABLE_ADDR + self.samp_slot * 16 + i as u32 * 4, v);
        }
        g.reg(r!(desc_flush), 0);
    }

    fn root_table(g: &mut dyn Gpu, at: u32, ls: u32, tex: u32, samp: u32) -> u32 {
        for (i, v) in [ls, tex, samp, 0].into_iter().enumerate() {
            g.mem(at + 4 * i as u32, v);
        }
        at
    }

    fn ls_base(&self, stage: u32) -> u32 {
        if self.dp.push_stage >> stage & 1 != 0 {
            BORG_PUSH_CONST_SPI
        } else if self.dp.ubo_base != 0 {
            self.dp.ubo_base
        } else if self.push_used {
            BORG_PUSH_CONST_SPI
        } else {
            DRAW_UBO_SPI
        }
    }

    fn alloc(&mut self, bytes: u32) -> u32 {
        let a = BORG_LIST_SPI + self.list.top;
        self.list.top += (bytes + 3) & !3;
        a
    }

    fn block(&mut self, g: &mut dyn Gpu, regs: &[[u32; 2]]) -> u32 {
        let a = self.alloc(4 + regs.len() as u32 * 8);
        g.mem(a, regs.len() as u32);
        for (i, [o, v]) in regs.iter().enumerate() {
            g.mem(a + 4 + i as u32 * 8, *o);
            g.mem(a + 8 + i as u32 * 8, *v);
        }
        a
    }

    fn list_stage(&mut self, g: &mut dyn Gpu, stage: usize) {
        if self.sh[stage].code != 0 {
            return;
        }
        let sh = if stage == 0 { self.vert } else { self.frag }.unwrap();
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
        for i in 0..sh.num_window as usize {
            g.mem(c + (sh.window_regs[i] as u32 - u0) * 4, sh.window_vals[i]);
        }
        (self.sh[stage].code, self.sh[stage].consts) = (a, c);
    }

    fn setup_room_tris(&self) -> u32 {
        ((BORG_HEAP_SPI - self.setup_base) / TBR_SETUP_ENTRY_BYTES).min(60000)
    }

    fn list_reset(&mut self) {
        self.list = List { top: BORG_LIST_ENTRY_BYTES, ..Default::default() };
        self.sh[0].code = 0;
        self.sh[1].code = 0;
        self.vtab_dirty = true;
    }

    /// Queues the draw the last 0xBA described.
    pub fn draw(&mut self, g: &mut dyn Gpu) {
        let (bias_c, bias_s) = (self.bias[0], self.bias[1]);
        self.bias = [0; 2];
        let Some(vert) = self.vert else { return };
        let Some(shift) = record_shift(vert.num_varyings as u32) else { return };
        if self.vertex_count == 0 || self.sh[0].blob.is_empty() || self.sh[1].blob.is_empty() || self.frag.is_none() {
            return;
        }
        let tris = self.vertex_count * self.dp.instance_count.max(1);
        let l = &self.list;
        if l.n != 0 && (shift != l.shift || l.tris + tris > self.setup_room_tris() || l.n >= BORG_LIST_ENTRY_BYTES / 8 - 1 || l.top + 0x4000 > BORG_LIST_BYTES) {
            self.flush(g);
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
        st.push([r!(draw_cfg), 1 | dp.topology << 1 | dp.index_type << 3 | dp.restart << 5 | shift << 6]);
        if self.list.state_addr == 0 || self.list.state != st {
            self.list.state = st.clone();
            let at = self.alloc(16);
            st[root_i][1] = Self::root_table(g, at, self.ls_base(1), TEX_DESC_TABLE_ADDR, SAMPLER_DESC_TABLE_ADDR);
            self.list.state_addr = self.block(g, &st);
        }
        let at = self.alloc(16);
        let root = Self::root_table(g, at, self.ls_base(0), self.list.vtab_addr - BORG_VATTR_SLOT0 * 64, SAMPLER_DESC_TABLE_ADDR);
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

    fn wait(g: &mut dyn Gpu, bit: u32, set: bool) {
        g.wait(r!(status), bit, if set { bit } else { 0 });
    }

    fn trigger(&mut self, g: &mut dyn Gpu, list: u32, load: u32) {
        let cc_lo = (self.clear[2] as u32) << 16 | FP16_MAX_DEPTH;
        let cc_hi = (self.clear[0] as u32) << 16 | self.clear[1] as u32;
        g.reg(r!(seq_fb_base), if self.place[0] != 0 { self.place[0] } else { DRAM_SPI_BASE + DRAM_OUT_OFFSET });
        let fb_tiles = self.fb_w >> 2;
        let win = fb_tiles.min(self.window_tiles);
        g.reg(r!(seq_tiles_per_row), win);
        g.reg(r!(seq_tile_rows), win);
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
                let a = if self.place[3 + k] != 0 { self.place[3 + k] } else { 0x4A0_0000 + k as u32 * 0x100_0000 };
                g.reg(reg, a);
            }
        }
        let fb_rows = self.fb_h >> 2;
        let (k, n, per) = self.strip;
        let (y0, rows) = if n > 1 { (k * per, per.min(fb_rows.saturating_sub(k * per))) } else { (0, fb_rows) };
        let mut wy = y0;
        while wy < y0 + rows {
            let mut wx = 0;
            while wx < fb_tiles {
                g.reg(r!(seq_tile_rows), (y0 + rows - wy).min(win));
                g.reg(r!(fb_origin), wx | wy << 16);
                g.reg(r!(seq_trigger), 1);
                Self::wait(g, STATUS_REG_T__SEQ_BUSY_bm, false);
                wx += win;
            }
            wy += win;
        }
        Self::wait(g, STATUS_REG_T__IDLE_bm, true);
        g.reg(r!(render_list), 0);
    }

    /// Renders what is queued.
    pub fn flush(&mut self, g: &mut dyn Gpu) {
        if self.list.n == 0 {
            return;
        }
        g.mem(BORG_LIST_SPI + self.list.n * 8, 0);
        g.mem(BORG_LIST_SPI + self.list.n * 8 + 4, 0);
        self.trigger(g, BORG_LIST_SPI, self.list.load);
        self.list_reset();
        if self.occ.0 {
            self.occ.2 = self.occ.2.wrapping_add(g.read(r!(occ_count)));
            g.mem(self.occ.1, self.occ.2);
            g.reg(r!(occ_ctrl), 3);
        }
    }

    /// One packet of the stream; false when it is malformed.
    pub fn packet(&mut self, g: &mut dyn Gpu, p: &[u8]) -> bool {
        if matches!(p[0], 0xB2 | 0xB5 | 0xB6 | 0xBB | 0xBF | 0xC6 | 0xC7 | 0xC8) {
            self.flush(g);
        }
        let n = packet_len(p[0]);
        if n == 0 || p.len() < n || p[1..n - 1].iter().fold(0, |c, b| c ^ b) != p[n - 1] {
            return false;
        }
        let w = |i: usize| le32(&p[i..]);
        match p[0] {
            0xB0 | 0xC1 | 0xC3 => {
                let (stage, blen) = (p[1] as usize, p[2] as usize | (p[3] as usize) << 8);
                if stage > 1 || blen < 6 || 4 + blen > n {
                    return false;
                }
                self.stage_shader(g, stage, &p[4..4 + blen]);
            }
            0xB2 => {
                let (off, nw) = (p[1] as u32, p[2] as u32);
                if nw < 1 || nw > BC_PUSH_MAX_WORDS || off >= BC_PUSH_MAX_WORDS || nw > BC_PUSH_MAX_WORDS - off {
                    return false;
                }
                let nw = nw.min(BORG_PUSH_CONST_MAX_WORDS - off);
                for i in 0..nw {
                    g.mem(BORG_PUSH_CONST_SPI + (off + i) * 4, w(3 + 4 * i as usize));
                }
                let root = Self::root_table(g, DRAW_ROOT_SPI, BORG_PUSH_CONST_SPI, TEX_DESC_TABLE_ADDR, SAMPLER_DESC_TABLE_ADDR);
                g.reg(r!(root), root);
                self.push_used = true;
            }
            0xB3 => {
                g.reg(r!(blend_cfg), w(1));
                g.reg(r!(blend_const), w(5));
                (self.st[0], self.st[1]) = (w(1), w(5));
            }
            0xB4 => {
                g.reg(r!(stencil_cfg), w(1));
                g.reg(r!(stencil_front), w(5));
                g.reg(r!(stencil_back), w(9));
                g.reg(r!(depth_cfg), w(13));
                self.cull_cfg = w(17);
                self.st[2..6].copy_from_slice(&[w(1), w(5), w(9), w(13)]);
                self.raster_valid = true;
            }
            0xC2 => {
                let f: [f32; 6] = std::array::from_fn(|k| f32::from_bits(w(1 + 4 * k)));
                let v = [f[2] / 2.0, f[3] / 2.0, f[0] + f[2] / 2.0, f[1] + f[3] / 2.0, f[5] - f[4], f[4]];
                let (x0, y0, x1, y1) = (w(25), w(29), w(33), w(37));
                if x0 > x1 || y0 > y1 || x1 > 4096 || y1 > 4096 {
                    return false;
                }
                let b = v.map(f32::to_bits);
                self.vp = Some([b[0], b[1], b[2], b[3], b[4], b[5], x0 | x1 << 13, y0 | y1 << 13 | 1 << 26]);
            }
            0xC6 => {
                let off = w(2);
                if p[1] > 1 || off & 3 != 0 || off >= BORG_HEAP_BYTES {
                    return false;
                }
                self.occ = (p[1] != 0, BORG_HEAP_SPI + off, 0);
                if self.occ.0 {
                    g.mem(self.occ.1, 0);
                }
                g.reg(r!(occ_ctrl), if p[1] != 0 { 3 } else { 0 });
            }
            0xC5 => self.bias = [w(1), w(5)],
            0xB6 => {
                let Some((fmt, srgb)) = wire_format(p[1]) else { return false };
                if fmt != self.flush_format || srgb != self.srgb {
                    (self.flush_format, self.srgb) = (fmt, srgb);
                    self.apply_layout(g);
                }
                self.clear = [f32_to_fp16(w(2)), f32_to_fp16(w(6)), f32_to_fp16(w(10))];
            }
            0xBF => {
                if p[1] < 1 || p[1] > 4 {
                    return false;
                }
                let mut att = 0;
                for k in 0..3 {
                    let Some((fmt, srgb)) = wire_format(p[2 + k]) else { return false };
                    att |= fmt << (3 * k) | srgb << (9 + k);
                }
                (self.natt, self.att_fmt) = (p[1] as u32, att);
            }
            0xC7 => {
                for k in 0..6 {
                    self.place[k] = w(1 + 4 * k);
                }
                self.pitch = w(25);
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
            0xBB => {
                let Some((fmt, srgb)) = wire_format(p[1]) else { return false };
                if p[2] > 63 {
                    return false;
                }
                let (nocolor, ms) = ((p[2] >> 3 & 1) as u32, (p[2] >> 4 & 1) as u32);
                if fmt != self.flush_format || srgb != self.srgb || nocolor != self.nocolor || ms != self.ms {
                    (self.flush_format, self.srgb, self.nocolor, self.ms) = (fmt, srgb, nocolor, ms);
                    self.apply_layout(g);
                }
                let pl = |k: usize, fixed: u32, b: &Self| if b.place[k] != 0 { b.place[k] } else { fixed };
                g.reg(r!(flush_zb_base), if p[2] & 1 != 0 { pl(1, BORG_ZB_SPI, self) } else { 0 });
                g.reg(r!(flush_sb_base), if p[2] & 2 != 0 { pl(2, BORG_SB_SPI, self) } else { 0 });
                g.reg(r!(depth_format), (p[2] >> 2 & 1) as u32 | ((p[2] >> 5 & 1) as u32) << 1);
                g.reg(r!(attach_ms), self.ms);
            }
            0xB8 => {
                let (off, nb) = (w(1), p[5] as u32 | (p[6] as u32) << 8);
                if nb > BC_MEM_DATA || off & 3 != 0 || off + nb > BORG_HEAP_BYTES {
                    return false;
                }
                for i in (0..nb).step_by(4) {
                    g.mem(BORG_HEAP_SPI + off + i, w(7 + i as usize));
                }
            }
            0xB9 => {
                let (slot, fmt) = (p[1] as u32, p[2] as u32);
                let (base, count, stride, swz) = (BORG_HEAP_SPI.wrapping_add(w(3)), w(7), w(11), w(15));
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
            0xBA => {
                if p[1] > 2 || p[2] > 2 || p[3] > 127 {
                    return false;
                }
                let load = (p[3] >> 1 & 15) as u32;
                self.dp = DrawParams {
                    topology: p[1] as u32,
                    index_type: p[2] as u32,
                    restart: (p[3] & 1) as u32,
                    load: if load == 1 { 9 } else { load },
                    push_stage: (p[3] >> 5 & 3) as u32,
                    instance_count: w(8),
                    first_vertex: w(12),
                    first_instance: w(16),
                    vertex_offset: w(20) as i32,
                    index_base: if p[2] != 0 { BORG_HEAP_SPI + w(24) } else { 0 },
                    ubo_base: if w(28) != 0 { BORG_HEAP_SPI + w(28) - 1 } else { 0 },
                };
                self.vertex_count = w(4);
                (self.tex_slot, self.samp_slot, self.tex_base) = (0, 0, 0);
            }
            0xC0 => self.tex_lvl = std::array::from_fn(|i| w(1 + 4 * i)),
            0xC4 => {
                if p[1] as u32 >= BC_TEX_SLOTS || p[2] as u32 >= BC_TEX_SLOTS || w(3) & 255 != 0 || w(3) >= TEXEL_BYTES {
                    return false;
                }
                (self.tex_slot, self.samp_slot, self.tex_base) = (p[1] as u32, p[2] as u32, w(3));
            }
            0xB5 => {
                let (off, nb) = (w(1), p[5] as u32 | (p[6] as u32) << 8);
                if nb > BC_TEXG_DATA || off & 3 != 0 || self.tex_base + off + nb > TEXEL_BYTES {
                    return false;
                }
                let d3 = [w(7), w(11), w(15)];
                if (d3[1] >> 10) & 15 == 0 {
                    self.tex_lvl = [0; 12];
                }
                let mut d = [0u32; 16];
                d[0] = TEXEL_BASE + self.tex_base;
                d[1..4].copy_from_slice(&d3);
                d[4..].copy_from_slice(&self.tex_lvl);
                self.tex_desc_store(g, &d);
                let samp = [w(19), w(23), w(27), w(31)];
                self.sampler_store(g, &samp);
                for i in (0..nb).step_by(4) {
                    g.mem(TEXEL_BASE + self.tex_base + off + i, w(35 + i as usize));
                }
            }
            0xC8 => {
                let d: [u32; 16] = std::array::from_fn(|i| w(1 + 4 * i));
                self.tex_desc_store(g, &d);
                let samp = [w(65), w(69), w(73), w(77)];
                self.sampler_store(g, &samp);
            }
            _ => return false,
        }
        true
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
    fn packet_lengths() {
        assert_eq!(packet_len(0xBA), 1 + 3 + 7 * 4 + 1);
        assert_eq!(packet_len(0xB6), 1 + 1 + 16 + 1);
        assert_eq!(packet_len(0xAD), 0);
    }

    #[test]
    fn record_shifts() {
        assert_eq!(record_shift(0), Some(8));
        assert_eq!(record_shift(16), Some(9));
        assert_eq!(record_shift(200), None);
    }
}
