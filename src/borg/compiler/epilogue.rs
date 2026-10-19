use crate::state::{Selector, ShaderState, DRAW_VS_CONST_U0, DRAW_VS_CONST_WORDS};
use std::collections::HashMap;
use crate::{FRAG_POOL_SLOT, packed_layout, BorgInstr, Ubo, NO_DST};

impl ShaderState {
    /// Vertex epilogue for points, which are drawn as quads of six vertices. Fetches the corner's
    /// clip-space offset and point coordinate by vertex index, scales the offset by gl_PointSize
    /// (clamped to 1..64) and w, and adds it to the position. The point coordinate also goes out
    /// as two varyings when the fragment shader reads gl_PointCoord.
    pub unsafe fn epilogue_point_quads(&mut self) {
        let ctl_bits = (1u32 << 16) | (128 + 16);
        assert!(self.cp.vs_consts.len() < DRAW_VS_CONST_WORDS,
            "borgc: vertex shader needs more than {} window constants", DRAW_VS_CONST_WORDS);
        let u = DRAW_VS_CONST_U0 + self.cp.vs_consts.len() as u8;
        self.cp.vs_consts.push((u, ctl_bits));
        let ctl = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.ubo.insert(ctl, Ubo::Uniform(u));
        let idx = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.ubo.insert(idx, Ubo::Fixed(30));
        let twelve = self.cp.vs_const(&mut self.sel, 12, false);   // shared with the attribute fetches
        let ty = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "ISRL", dst: ty, srcs: vec![idx, twelve], swz: vec![0, 0] });
        let hi = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "ISHL", dst: hi, srcs: vec![ty, twelve], swz: vec![0, 0] });
        let tx = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "ISUB", dst: tx, srcs: vec![idx, hi], swz: vec![0, 0] });
        let tex_v = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "TEX", dst: tex_v, srcs: vec![tx, ty, ctl], swz: vec![0, 0, 0] });
        self.tex_dsts.insert(tex_v);
        let mut d = [0u32; 4];
        for k in 0..4 {
            let m = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "IOR", dst: m, srcs: vec![tex_v, tex_v], swz: vec![k as u8, k as u8] });
            d[k] = m;
        }
        // gl_PointCoord rides as two more varyings, after the shader's own.
        if self.flags.pntc_varyings() {
            let base = self.sel.prog.iter().filter(|i| i.mnem == "SOUT").map(|i| i.swz[0] as u32 + 1).max().unwrap_or(0);
            assert!(base + 2 <= 255, "borgc: no varying slot left for the point coordinate");
            for k in 0..2u32 {
                self.sel.prog.push(BorgInstr { mnem: "SOUT", dst: NO_DST, srcs: vec![d[2 + k as usize]], swz: vec![(base + k) as u8] });
                self.outs.roots.push(d[2 + k as usize]);
            }
        }
        if let (Some(px), Some(py), Some(pw)) = (self.outs.pos[0], self.outs.pos[1], self.outs.pos[3]) {
            let w = self.sel.reg(pw);
            let size = self.outs.psize.map(|ps| {
                // Clamp to the advertised range [1, 64]: x - c*(x - 64), c = step(x - 64).
                let x = self.sel.reg(ps).0;
                // upper: x -= step(x - 64) * (x - 64)
                let neg_hi = self.cp.vs_win(&mut self.sel, (-64.0f32).to_bits());
                let t = self.sel.op0("FADD", vec![x, neg_hi]);
                let c = self.sel.op0("FSTEP", vec![t]);
                let ct = self.sel.op0("FMUL", vec![c, t]);
                let nct = self.sel.op0("FNEG", vec![ct]);
                let x = self.sel.op0("FADD", vec![x, nct]);
                // lower: x += step(1 - x) * (1 - x); 1.0 = step(dx * dx), as the window has no word left
                let dd = self.sel.op0("FMUL", vec![d[0], d[0]]);
                let one = self.sel.op0("FSTEP", vec![dd]);
                let nx = self.sel.op0("FNEG", vec![x]);
                let t2 = self.sel.op0("FADD", vec![nx, one]);
                let c2 = self.sel.op0("FSTEP", vec![t2]);
                let m2 = self.sel.op0("FMUL", vec![c2, t2]);
                let x = self.sel.op0("FADD", vec![x, m2]);
                (x, 0u8)
            });
            for (k, pc) in [px, py].into_iter().enumerate() {
                let base = self.sel.reg(pc);
                let mut off = d[k];
                if let Some(sz) = size {
                    let o = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: o, srcs: vec![off, sz.0], swz: vec![0, sz.1] });
                    off = o;
                }
                let o = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: o, srcs: vec![off, w.0, base.0],
                                      swz: vec![0, w.1, base.1] });
                self.outs.pos[k] = Some((o, 0));
                self.outs.roots.push(o);
            }
        }
    }
    /// Multiple render targets: converts each target's value to its class (unorm8, snorm8 or
    /// raw bits), then selects the one for the attachment being rendered (ATTIDX). A target the
    /// shader does not write keeps its destination value (TLD).
    pub unsafe fn epilogue_multi_target(&mut self) {
        let n = self.flags.target_count();
        let class = |k: usize| self.flags.target_class(k);
        let cff = self.fw.win(&mut self.sel, 0xFF);
        let idx = self.sel.op0("ATTIDX", vec![]);
        let zero = self.sel.op0("ISUB", vec![idx, idx]);
        let mut vals: Vec<u32> = Vec::new();
        for k in 0..n {
            let w = match self.outs.mrt[k] {
                None => self.sel.op0("TLD", vec![]),
                Some((x, xc)) => {
                    let x = if matches!(self.sel.ubo.get(&x), Some(Ubo::Uniform(_))) {
                        self.sel.op0("IADD", vec![x, zero])
                    } else if xc != 0 {
                        let o = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "FMOV", dst: o, srcs: vec![x], swz: vec![xc] });
                        o
                    } else { x };
                    match class(k) {
                        1 => {
                            let c255 = self.fw.win(&mut self.sel, 255.0f32.to_bits());
                            let c05 = self.fw.win(&mut self.sel, 0.5f32.to_bits());
                            let m = self.sel.op0("FMUL", vec![x, c255]);
                            let h = self.sel.op0("FADD", vec![m, c05]);
                            let i = self.sel.op0("F2I", vec![h]);
                            self.sel.op0("IAND", vec![i, cff])
                        }
                        2 => {
                            let c127 = self.fw.win(&mut self.sel, 127.0f32.to_bits());
                            let c05 = self.fw.win(&mut self.sel, 0.5f32.to_bits());
                            let t = self.sel.op0("FMUL", vec![x, c127]);
                            let s = self.sel.op0("FSTEP", vec![t]);
                            let nh = self.sel.op0("FNEG", vec![c05]);
                            let hh = self.sel.op0("FADD", vec![s, nh]);
                            let u = self.sel.op0("FADD", vec![t, hh]);
                            let i = self.sel.op0("F2I", vec![u]);
                            self.sel.op0("IAND", vec![i, cff])
                        }
                        _ => self.sel.op0("IAND", vec![x, cff]),
                    }
                }
            };
            vals.push(w);
        }
        let mut acc = vals[0];
        if n > 1 {
            let one_u = self.fw.win(&mut self.sel, 1);
            let one = self.sel.op0("IADD", vec![one_u, zero]);
            let mut kreg = one;
            for k in 1..n {
                if k == 2 { kreg = self.sel.op0("IADD", vec![one, one]); }
                if k == 3 { kreg = self.sel.op0("IADD", vec![kreg, one]); }
                let hit = self.sel.op0("ISEQ", vec![idx, kreg]);
                let diff = self.sel.op0("ISUB", vec![vals[k], acc]);
                let term = self.sel.op0("IMUL", vec![hit, diff]);
                acc = self.sel.op0("IADD", vec![acc, term]);
            }
        }
        self.outs.frag = [Some((acc, 0)), None, None, None];
        self.out_roots.push(acc);
    }
    /// R8_UNORM output (RAW8): the red value as round(r * 255) & 0xFF.
    pub unsafe fn epilogue_r8_unorm(&mut self) {
        // R8_UNORM (RAW8): r26 = round(r * 255) & 0xFF.
        let c255 = self.fw.win(&mut self.sel, 255.0f32.to_bits());
        let c05 = self.fw.win(&mut self.sel, 0.5f32.to_bits());
        let cff = self.fw.win(&mut self.sel, 0xFF);
        let (x, xc) = self.outs.frag[0].unwrap();
        let x = if matches!(self.sel.ubo.get(&x), Some(Ubo::Uniform(_))) {
            let o = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "FMOV", dst: o, srcs: vec![x], swz: vec![xc] });
            o
        } else { x };
        let xc = if x == self.outs.frag[0].unwrap().0 { xc } else { 0 };
        let m = self.sel.op("FMUL", vec![x, c255], vec![xc, 0]);
        let h = self.sel.op("FADD", vec![m, c05], vec![0, 0]);
        let i = self.sel.op("F2I", vec![h], vec![0]);
        let w = self.sel.op("IAND", vec![i, cff], vec![0, 0]);
        self.outs.frag = [Some((w, 0)), None, None, None];
        self.out_roots.push(w);
    }
    /// Packed output formats (565, 4444, 5551, 10:10:10:2, 8/16/32-bit channels; unorm, snorm,
    /// sRGB or raw): quantizes each channel, shifts it to its bit position and ORs the channels
    /// into up to four words. RAW128 renders two words per pass, picked by the slice index.
    pub unsafe fn epilogue_packed(&mut self) {
        let (kind, layout) = packed_layout(self.flags);
        let total_bits = layout.iter().map(|&(_, sh, b)| sh + b).max().unwrap_or(0);
        let mut words: [Option<u32>; 4] = [None; 4];
        for &(comp, bitpos, bits) in layout.iter() {
            let Some((x, xc)) = self.outs.frag[comp] else { continue };
            let mask = if bits < 32 && (kind == 1 || kind == 3) { Some(self.fw.win(&mut self.sel, (1u32 << bits) - 1)) } else { None };
            let (x, xc) = if matches!(self.sel.ubo.get(&x), Some(Ubo::Uniform(_))) {
                (self.sel.op("FMOV", vec![x], vec![xc]), 0)
            } else { (x, xc) };
            let (x, xc) = if kind == 2 && comp != 3 {
                (self.sel.op("FSRGB", vec![x], vec![xc]), 0)
            } else { (x, xc) };
            let q = match kind {
                0 | 2 => {
                    let cmax = self.fw.win(&mut self.sel, ((1u32 << bits) as f32 - 1.0).to_bits());
                    let c05 = self.fw.win(&mut self.sel, 0.5f32.to_bits());
                    let m = self.sel.op("FMUL", vec![x, cmax], vec![xc, 0]);
                    let h = self.sel.op("FADD", vec![m, c05], vec![0, 0]);
                    self.sel.op("F2I", vec![h], vec![0])
                }
                1 => {
                    let cmax = self.fw.win(&mut self.sel, (((1u32 << (bits - 1)) - 1) as f32).to_bits());
                    let c05 = self.fw.win(&mut self.sel, 0.5f32.to_bits());
                    let t = self.sel.op("FMUL", vec![x, cmax], vec![xc, 0]);
                    let st = self.sel.op("FSTEP", vec![t], vec![0]);
                    let nh = self.sel.op("FNEG", vec![c05], vec![0]);
                    let hh = self.sel.op("FADD", vec![st, nh], vec![0, 0]);
                    let u = self.sel.op("FADD", vec![t, hh], vec![0, 0]);
                    let i = self.sel.op("F2I", vec![u], vec![0]);
                    self.sel.op("IAND", vec![i, mask.unwrap()], vec![0, 0])
                }
                4 => {
                    let cmax = self.fw.win(&mut self.sel, (1u32 << bits) - 1);
                    let lt = self.sel.op("ISLTU", vec![x, cmax], vec![xc, 0]);
                    let d = self.sel.op("ISUB", vec![x, cmax], vec![xc, 0]);
                    let m = self.sel.op("IMUL", vec![lt, d], vec![0, 0]);
                    self.sel.op("IADD", vec![cmax, m], vec![0, 0])
                }
                _ => match mask {
                    Some(m) => self.sel.op("IAND", vec![x, m], vec![xc, 0]),
                    None => self.sel.op("IOR", vec![x, x], vec![xc, xc]),
                },
            };
            let (wi, sh) = ((bitpos / 32) as usize, bitpos % 32);
            let q = if sh == 0 { q } else {
                let shv = self.fw.win(&mut self.sel, sh);
                self.sel.op("ISHL", vec![q, shv], vec![0, 0])
            };
            words[wi] = Some(match words[wi] {
                None => q,
                Some(a) => self.sel.op("IOR", vec![a, q], vec![0, 0]),
            });
        }
        let nwords = (total_bits + 31) / 32;
        let mut out: [Option<(u32, u8)>; 4] = [None; 4];
        if nwords <= 2 {
            for w in 0..nwords as usize {
                out[w] = words[w].map(|v| (v, 0u8));
            }
        } else {
            // RAW128: the slice this pass renders (ATTIDX / 4) picks words 0-1 or 2-3.
            let idx = self.sel.op0("ATTIDX", vec![]);
            let zero = self.sel.op0("ISUB", vec![idx, idx]);
            let four_w = self.fw.win(&mut self.sel, 4);
            let four = self.sel.op0("IADD", vec![four_w, zero]);
            let hit = self.sel.op0("ISEQ", vec![idx, four]);
            for w in 0..2 {
                let lo = words[w].unwrap_or(zero);
                let hi = words[w + 2].unwrap_or(zero);
                let d = self.sel.op0("ISUB", vec![hi, lo]);
                let m = self.sel.op0("IMUL", vec![hit, d]);
                out[w] = Some((self.sel.op0("IADD", vec![lo, m]), 0u8));
            }
        }
        for o in out.iter().flatten() {
            self.out_roots.push(o.0);
        }
        self.outs.frag = out;
    }
    /// Half-float output: converts each f32 channel to f16 (round to nearest even, denormals,
    /// overflow to infinity, NaN) and packs two channels per word.
    pub unsafe fn epilogue_half(&mut self) {
        let nch = self.flags.half_channels();
        let (x0, x0c) = self.outs.frag[0].unwrap();
        let mut pw: [Option<u32>; 2] = [None; 2];
        for n in 0..nch {
            let src = if n == 0 { Some((x0, x0c)) } else { self.outs.frag[n] };
            let Some(x) = src else { continue };
            let h = self.sel.opc("F2H", &[x]);
            let h = if n & 1 == 0 { h } else {
                let c = self.fw.win(&mut self.sel, 16);
                self.sel.opc("ISHL", &[(h, 0), (c, 0)])
            };
            pw[n / 2] = Some(match pw[n / 2] { None => h, Some(a) => self.sel.opc("IOR", &[(a, 0), (h, 0)]) });
        }
        self.outs.frag = [pw[0].map(|p| (p, 0)), pw[1].map(|p| (p, 0)), None, None];
        for p in pw.iter().flatten() {
            self.out_roots.push(*p);
        }
    }
    /// Packs four 8-bit channel values into one word, channel 3 in the top byte.
    pub unsafe fn epilogue_bytes_packed(&mut self) {
        let (mut p, mut pc) = self.outs.frag[3].unwrap();
        let eight = self.fw.slot(8);
        let sh = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        self.sel.ubo.insert(sh, Ubo::Uniform(eight as u8));
        for c in (0..3).rev() {
            {
                let d = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "ISHL", dst: d, srcs: vec![p, sh], swz: vec![pc, 0] });
                p = d;
                pc = 0;
            }
            if let Some((v, comp)) = self.outs.frag[c] {
                let d = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "IOR", dst: d, srcs: vec![p, v], swz: vec![pc, comp] });
                p = d;
                pc = 0;
            }
        }
        self.outs.frag = [Some((p, pc)), None, None, None];
        self.out_roots.push(p);
    }
    pub unsafe fn pool_window_consts(&mut self) {
        let first_pool = 20 + 9;
        // The window keeps the nine constants read most often: a window word costs no register.
        {
            let mut uses: HashMap<u32, usize> = HashMap::new();
            for i in self.sel.prog.iter() {
                for s in i.srcs.iter() {
                    if let Some(&Ubo::Uniform(u)) = self.sel.ubo.get(s) {
                        if u >= 20 {
                            *uses.entry(u as u32).or_insert(0) += 1;
                        }
                    }
                }
            }
            let mut by_use: Vec<(u32, u32)> = self.fw.consts.clone();
            by_use.sort_by_key(|&(u, _)| (std::cmp::Reverse(uses.get(&u).copied().unwrap_or(0)), u));
            let renum: HashMap<u32, u32> = by_use.iter().enumerate().map(|(k, &(u, _))| (u, 20 + k as u32)).collect();
            for (_, m) in self.sel.ubo.iter_mut() {
                if let Ubo::Uniform(u) = m {
                    if let Some(&n) = renum.get(&(*u as u32)) {
                        *u = n as u8;
                    }
                }
            }
            self.fw.consts = by_use.iter().map(|&(u, bits)| (renum[&u], bits)).collect();
        }
        let mut pool: Vec<u32> = vec![0; (20 + self.fw.count - first_pool) as usize];
        for &(u, bits) in self.fw.consts.iter() {
            if u >= first_pool {
                pool[(u - first_pool) as usize] = bits;
            }
        }
        self.fw.consts.retain(|&(u, _)| u < first_pool);
        self.fw.consts.push((first_pool, (1 << 16) | (128 + FRAG_POOL_SLOT)));
        self.fw.consts.push((first_pool + 1, 1));
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            if let Some(u) = u {
                ubo.insert(v, u);
            }
            v
        };
        let ctl = fresh(&mut self.sel.ubo, Some(Ubo::Uniform(first_pool as u8)));
        let one = fresh(&mut self.sel.ubo, Some(Ubo::Uniform(first_pool as u8 + 1)));
        let f = fresh(&mut self.sel.ubo, Some(Ubo::Fixed(30)));
        self.per_pixel_fixed.insert(f);
        let zero = fresh(&mut self.sel.ubo, None);
        let is_pool = |v: &u32, ubo: &HashMap<u32, Ubo>| match ubo.get(v) {
            Some(&Ubo::Uniform(u)) if u as u32 >= first_pool && *v != ctl && *v != one => Some(u as u32 - first_pool),
            _ => None,
        };
        let pool = self.fetch_pool(&pool, is_pool, ctl, one, f, zero);
        eprintln!("borgc: constant pool of {} word(s)", pool.len());
        #[allow(static_mut_refs)]
        crate::set_frag_pool(pool);
    }
    /// A vertex shader with more constants than its window holds: the most used stay in the window, the rest
    /// are fetched from a pool in memory, four words per TEX (the way a fragment shader's are).
    pub unsafe fn pool_vs_consts(&mut self) {
        const POOL_TAG: u32 = 100;
        let first = DRAW_VS_CONST_U0 as u32;
        let mut uses: HashMap<u32, usize> = HashMap::new();
        for i in self.sel.prog.iter() {
            for s in i.srcs.iter() {
                if let Some(&Ubo::Uniform(u)) = self.sel.ubo.get(s) {
                    if u as u32 >= first {
                        *uses.entry(u as u32).or_insert(0) += 1;
                    }
                }
            }
        }
        let consts = self.cp.vs_consts.clone();
        let one_old = consts.iter().find(|c| c.1 == 1).map(|c| c.0 as u32);
        let mut others: Vec<(u32, u32)> = consts.iter().filter(|c| Some(c.0 as u32) != one_old).map(|&(u, b)| (u as u32, b)).collect();
        others.sort_by_key(|&(u, _)| (std::cmp::Reverse(uses.get(&u).copied().unwrap_or(0)), u));
        let cap = DRAW_VS_CONST_WORDS - 2;   // the pool's control word and the constant 1
        let (keep, pooled) = others.split_at(cap.min(others.len()));
        let mut keep: Vec<(u32, u32)> = keep.to_vec();
        keep.sort();
        let mut window: Vec<(u32, u32)> = keep.clone();   // (old id, bits)
        if let Some(o) = one_old { window.push((o, 1)); } else { window.push((u32::MAX, 1)); }
        window.push((u32::MAX - 1, (1u32 << 16) | (128 + crate::VERT_POOL_SLOT)));
        let renum: HashMap<u32, u32> = window.iter().enumerate().map(|(k, &(u, _))| (u, first + k as u32)).collect();
        let pool_tag: HashMap<u32, u32> = pooled.iter().enumerate().map(|(k, &(u, _))| (u, POOL_TAG + k as u32)).collect();
        let pool_bits: Vec<u32> = pooled.iter().map(|&(_, b)| b).collect();
        for (_, m) in self.sel.ubo.iter_mut() {
            if let Ubo::Uniform(u) = m {
                if let Some(&n) = renum.get(&(*u as u32)) { *u = n as u8; }
                else if let Some(&t) = pool_tag.get(&(*u as u32)) { *u = t as u8; }
            }
        }
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            if let Some(u) = u { ubo.insert(v, u); }
            v
        };
        let n = window.len() as u32;
        let ctl = fresh(&mut self.sel.ubo, Some(Ubo::Uniform((first + n - 1) as u8)));
        let one = fresh(&mut self.sel.ubo, Some(Ubo::Uniform(renum[&window[window.len() - 2].0] as u8)));
        let f = fresh(&mut self.sel.ubo, Some(Ubo::Fixed(30)));
        let zero = fresh(&mut self.sel.ubo, None);
        let is_pool = |v: &u32, ubo: &HashMap<u32, Ubo>| match ubo.get(v) {
            Some(&Ubo::Uniform(u)) if u as u32 >= POOL_TAG => Some(u as u32 - POOL_TAG),
            _ => None,
        };
        let pool = self.fetch_pool(&pool_bits, is_pool, ctl, one, f, zero);
        assert!(pool.len() <= 64, "borgc: vertex shader needs a constant pool of {} words (64 fit)", pool.len());
        self.cp.vs_consts = window.iter().map(|&(u, bits)| (renum[&u] as u8, bits)).collect();
        eprintln!("borgc: vertex constant pool of {} word(s)", pool.len());
        crate::set_vert_pool(pool);
    }
    /// Fetches the pooled words from memory four per TEX, again when a group goes unused for a while,
    /// and hoisted above the TEX results its first use waits for. Returns the pool in fetch order.
    fn fetch_pool(&mut self, words: &[u32], is_pool: impl Fn(&u32, &HashMap<u32, Ubo>) -> Option<u32>,
                  ctl: u32, one: u32, f: u32, zero: u32) -> Vec<u32> {
        let mut fresh = |ubo: &mut HashMap<u32, Ubo>, u: Option<Ubo>| {
            let v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            if let Some(u) = u {
                ubo.insert(v, u);
            }
            v
        };
        let mut first_use: Vec<Option<usize>> = vec![None; words.len()];
        for (at, i) in self.sel.prog.iter().enumerate() {
            for s in i.srcs.iter() {
                if let Some(k) = is_pool(s, &self.sel.ubo) {
                    first_use[k as usize].get_or_insert(at);
                }
            }
        }
        let mut order: Vec<usize> = (0..words.len()).filter(|&k| first_use[k].is_some()).collect();
        order.sort_by_key(|&k| first_use[k]);
        let mut slot_of: Vec<usize> = vec![0; words.len()];
        for (slot, &k) in order.iter().enumerate() {
            slot_of[k] = slot;
        }
        let pool: Vec<u32> = order.iter().map(|&k| words[k]).collect();
        const GAP: usize = 16;
        let ngroups = (pool.len() + 3) / 4;
        let mut last: Vec<Option<usize>> = vec![None; ngroups];
        let mut events: Vec<(usize, usize)> = Vec::new();   // (before instruction, group)
        for (at, i) in self.sel.prog.iter().enumerate() {
            for s in i.srcs.iter() {
                let Some(k) = is_pool(s, &self.sel.ubo) else { continue };
                let g = slot_of[k as usize] / 4;
                if last[g].map_or(true, |l| at - l > GAP) {
                    let mut p = at;
                    loop {
                        let Some(t) = (0..p).rev().find(|&t| self.sel.prog[t].mnem == "TEX") else { break };
                        let d = self.sel.prog[t].dst;
                        if self.sel.prog[p..].iter().any(|i| i.srcs.contains(&d)) { p = t; } else { break; }
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
        for (at, mut i) in std::mem::take(&mut self.sel.prog).into_iter().enumerate() {
            while e < events.len() && events[e].0 == at {
                let g = events[e].1;
                let mut idx = zero;
                for _ in 0..g {
                    let n = fresh(&mut self.sel.ubo, None);
                    out.push(BorgInstr { mnem: "IADD", dst: n, srcs: vec![idx, one], swz: vec![0, 0] });
                    idx = n;
                }
                let t = fresh(&mut self.sel.ubo, None);
                out.push(BorgInstr { mnem: "TEX", dst: t, srcs: vec![idx, zero, ctl], swz: vec![0, 0, 0] });
                self.tex_dsts.insert(t);
                for c in 0..4u8 {
                    let r = fresh(&mut self.sel.ubo, None);
                    out.push(BorgInstr { mnem: "IOR", dst: r, srcs: vec![t, t], swz: vec![c, c] });
                    cur[g][c as usize] = r;
                }
                e += 1;
            }
            for s in i.srcs.iter_mut() {
                if let Some(k) = is_pool(s, &self.sel.ubo) {
                    let slot = slot_of[k as usize];
                    *s = cur[slot / 4][slot % 4];
                }
            }
            out.push(i);
        }
        self.sel.prog = out;
        pool
    }
    pub fn split_window_reads(&mut self) {
        let is_win = |v: &u32, ubo: &HashMap<u32, Ubo>| matches!(ubo.get(v), Some(Ubo::Uniform(_)));
        if self.sel.prog.iter().any(|i| i.srcs.iter().filter(|v| is_win(v, &self.sel.ubo)).count() > 1) {
            let f = self.sel.next_vreg;
            let z = self.sel.next_vreg + 1;
            self.sel.next_vreg += 2;
            self.sel.ubo.insert(f, Ubo::Fixed(30));
            let mut out = vec![BorgInstr { mnem: "ISUB", dst: z, srcs: vec![f, f], swz: vec![0, 0] }];
            for mut ins in std::mem::take(&mut self.sel.prog) {
                let mut seen = false;
                for k in 0..ins.srcs.len() {
                    if !is_win(&ins.srcs[k], &self.sel.ubo) {
                        continue;
                    }
                    if !seen {
                        seen = true;
                        continue;
                    }
                    let t = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    out.push(BorgInstr { mnem: "IADD", dst: t, srcs: vec![ins.srcs[k], z], swz: vec![ins.swz[k], 0] });
                    ins.srcs[k] = t;
                    ins.swz[k] = 0;
                }
                out.push(ins);
            }
            self.sel.prog = out;
        }
    }
}

