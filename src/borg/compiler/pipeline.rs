use crate::block::block_starts;
use crate::emit::{build_blob, encode_program, Encoded};
use crate::fixed::{allocate, fixed_regs};
use crate::opt::{dce, fuse_fmadd, schedule_for_latency, schedule_for_pressure};
use crate::state::{ShaderState, Stage};
use crate::survey::{survey, Survey};
use crate::opcodes::{CSR_BARY0, CSR_FRAG_X, CSR_GROUP_ID_X, CSR_LOCAL_INDEX, CSR_ROOT, CSR_VERTEX_INDEX};
use crate::{BorgInstr, Flags, Loc, Mnem, NO_DST};
use std::collections::HashMap;
use compiler::bindings::*;
use crate::opcodes::Op;

impl ShaderState {
    pub unsafe fn start(nir: *mut nir_shader, flags: Flags, out_select: i32) -> (ShaderState, *mut nir_function_impl, u32) {
        let mesa_stage = (*nir).info.stage();
        let stage = Stage::of(mesa_stage);
        if crate::config::env().dump_nir {
            if let Ok(s) = (*nir).to_string() {
                eprintln!("{s}");
            }
        }
        let entry = nir_shader_get_entrypoint(nir);
        let Survey { total, tex, push_window } = survey(entry, mesa_stage);
        let regs: &'static [u8] = match stage {
            Stage::Vertex => &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            // Preset registers, from the top; r30/r31 are the invocation IDs, r12-r14 the grid base.
            Stage::Compute => &[29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15],
            _ if tex > 0 => &[17, 18, 19],
            _ => &[17, 18, 19, 23],
        };
        let mut st = ShaderState::new(regs, stage, flags);
        st.frag.push_window = push_window && stage == Stage::Fragment;
        st.frag.out_select = out_select;
        if stage == Stage::Fragment {
            for var in (*nir).iter_variables() {
                let d = &var.data;
                if d.mode() == nir_var_shader_in && d.centroid() != 0 && d.location >= VARYING_SLOT_VAR0 as i32 {
                    let first = d.location as u32 - VARYING_SLOT_VAR0;
                    for k in 0..glsl_count_attribute_slots(var.type_, false) {
                        st.frag.centroid |= 1u64.checked_shl(first + k).unwrap_or(0);
                    }
                }
            }
            if st.frag.centroid != 0 {
                st.frag.centroid_bary = st.centroid_barycentrics();
            }
        }
        (st, entry, total)
    }

    /// After instruction selection: logs the result, and refuses the shader when a NIR op had no
    /// Borg instruction, so no wrong shader is emitted.
    pub fn check_selected(&mut self) -> Result<(), String> {
        self.refusal()?;
        debug!("borgc: selected {} Borg instruction(s) (virtual regs)", self.sel.prog.len());
        if crate::config::env().dump_isa {
            for i in &self.sel.prog {
                let s: Vec<String> = i.srcs.iter().map(|v| format!("v{v}")).collect();
                eprintln!("borgc:   {} v{}, {}", i.mnem, i.dst, s.join(", "));
            }
        }
        if !self.skipped.is_empty() {
            return Err(format!("unsupported: {}", self.skipped.join(", ")));
        }
        Ok(())
    }

    /// Final pass over the outputs: runs the epilogues the option word asks for (point quads,
    /// multiple targets, packed, half or byte formats), records the output registers and, for
    /// fragments, interpolates depth.
    pub unsafe fn lower_outputs(&mut self) {
        let (is_vertex, is_fragment, flags) = (self.stage == Stage::Vertex, self.stage == Stage::Fragment, self.flags);
        if is_vertex && flags.point_quads() {
            self.epilogue_point_quads();
        }
        // Alpha to coverage reads r24 whatever the target's format packs.
        let alpha = self.outs.frag[3].filter(|_| is_fragment && self.frag.alpha && !flags.bytes_packed() && !flags.multi_target());
        if is_fragment && flags.multi_target() {
            self.epilogue_multi_target();
        }
        if is_fragment && flags.r8_unorm() && self.outs.frag[0].is_some() {
            self.epilogue_r8_unorm();
        }
        if is_fragment && flags.packed() && !flags.multi_target() && self.outs.frag[0].is_some() {
            self.epilogue_packed();
        }
        if is_fragment && flags.half() && self.outs.frag[0].is_some() {
            self.epilogue_half();
        }
        if is_fragment && flags.bytes_packed() && self.outs.frag[3].is_some() {
            self.epilogue_bytes_packed();
        }
        if let Some(a) = alpha.filter(|_| self.outs.frag[3].is_none()) {
            self.outs.frag[3] = Some(a);
            self.out_roots.push(a.0);
        }
        if is_fragment {
            self.copy_frag_outputs();
            self.csr_frag_outputs();
        } else if is_vertex {
            self.csr_vert_outputs();
        }
        self.frag.out = self.outs.frag.map(|v| v.map(|(d, _)| d));
        self.out_roots.extend(self.outs.roots.iter().copied());
        debug!(
            "borgc: draw-mode I/O map — {} vertex-pulling/MVP load(s), {} SOUT(s), gl_Position={}",
            self.vert.descriptor_defs.len(),
            self.outs.stores.len(),
            self.outs.pos[0].is_some(),
        );
    }

    pub unsafe fn optimise(&mut self) {
        self.place_const_operands();
        self.sys_reads();
        // Once before the window pool too: without it 24 of 137 replayed draw shaders grow by 13%.
        self.tex_addresses();
        if matches!(self.stage, Stage::Vertex | Stage::Fragment) {
            self.root_accesses();
        }
        self.const_immediates();
        self.pool_window();
        dce(&mut self.sel.prog, &self.out_roots);
        if matches!(self.stage, Stage::Vertex | Stage::Fragment) {
            self.root_accesses();
        }
        self.tex_addresses();
        dce(&mut self.sel.prog, &self.out_roots);
        self.const_immediates();
        let loc = &self.sel.loc;
        fuse_fmadd(&mut self.sel.prog, &self.out_roots, &|v| matches!(loc.get(&v), Some(Loc::Uniform(_))));
        schedule_for_pressure(&mut self.sel.prog, &self.out_roots);
        self.iseq_forms();
        self.fstep_forms();
        self.legalize();
        if !self.spill.in_order {
            schedule_for_latency(&mut self.sel.prog, &self.out_roots);
        }
        if self.stage == Stage::Fragment {
            self.log_uniformity();
        }
    }

    /// ISA v2 (B8 step 15): ISEQ as XOR (XORI for a small constant) and SLTIU 1; SEQZ against 0.
    fn iseq_forms(&mut self) {
        if !self.sel.prog.iter().any(|i| i.mnem == Op::Iseq) {
            return;
        }
        let small = |b: Option<u32>| b.filter(|&b| (b as i32) >= -2048 && (b as i32) < 2048);
        let mut out = Vec::with_capacity(self.sel.prog.len());
        for mut i in std::mem::take(&mut self.sel.prog) {
            if i.mnem != Op::Iseq {
                out.push(i);
                continue;
            }
            let (a, b) = ((i.srcs[0], i.swz[0]), (i.srcs[1], i.swz[1]));
            let (ka, kb) = (small(self.bits_of(a.0)), small(self.bits_of(b.0)));
            let (x, k) = match (ka, kb) {
                (_, Some(k)) => (a, k),
                (Some(k), None) => (b, k),
                (None, None) => {
                    let t = self.sel.fresh();
                    out.push(BorgInstr { mnem: Op::Ixor.into(), dst: t, srcs: vec![a.0, b.0], swz: vec![a.1, b.1], imm: 0 });
                    ((t, 0), 0)
                }
            };
            let (code, x) = self.window_copy(x);
            out.extend(code);
            let x = if k == 0 { x } else {
                let t = self.sel.fresh();
                out.push(BorgInstr { mnem: Op::Xori.into(), dst: t, srcs: vec![x.0], swz: vec![x.1], imm: k & 0xFFF });
                (t, 0)
            };
            (i.mnem, i.srcs, i.swz, i.imm) = (Op::Sltiu.into(), vec![x.0], vec![x.1], 1);
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// FSTEP as SLT from x0 and FCVT.S.W: 1.0 when the raw bits are a positive signed word.
    fn fstep_forms(&mut self) {
        if !self.sel.prog.iter().any(|i| i.mnem == Op::Fstep) {
            return;
        }
        let x0 = self.x0().0;
        let mut out = Vec::with_capacity(self.sel.prog.len());
        for mut i in std::mem::take(&mut self.sel.prog) {
            if i.mnem != Op::Fstep {
                out.push(i);
                continue;
            }
            let (code, x) = self.window_copy((i.srcs[0], i.swz[0]));
            out.extend(code);
            let t = self.sel.fresh();
            out.push(BorgInstr { mnem: Op::Slt.into(), dst: t, srcs: vec![x0, x.0], swz: vec![0, x.1], imm: 0 });
            (i.mnem, i.srcs, i.swz) = (Op::I2f.into(), vec![t], vec![0]);
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// `bits` in a register: ADDI, LUI, or LUI and ADDI.
    pub(crate) fn materialize(&mut self, out: &mut Vec<BorgInstr>, bits: u32) -> u32 {
        let (hi, lo) = (bits.wrapping_add(0x800) >> 12 & 0xF_FFFF, bits & 0xFFF);
        let u = if hi == 0 { self.x0().0 } else {
            let u = self.sel.fresh();
            out.push(BorgInstr { mnem: Op::Lui.into(), dst: u, srcs: vec![], swz: vec![], imm: hi });
            u
        };
        if lo == 0 && hi != 0 {
            return u;
        }
        let d = self.sel.fresh();
        out.push(BorgInstr { mnem: Op::Addi.into(), dst: d, srcs: vec![u], swz: vec![0], imm: lo });
        d
    }

    /// ISA v2 (B8 step 9): a draw shader's constant operands by immediates, built near each use and
    /// reused for `const_gap` instructions within a block (compute: pool_consts). A TEX's control
    /// word stays for tex_addresses.
    pub(crate) fn const_immediates(&mut self) {
        if self.stage == Stage::Compute {
            return;
        }
        let (mut near, mut built): (HashMap<u32, (u32, usize)>, usize) = (HashMap::new(), 0);
        let mut out = Vec::with_capacity(self.sel.prog.len());
        for (n, (start, mut i)) in block_starts(std::mem::take(&mut self.sel.prog)).enumerate() {
            if start {
                near.clear();
            }
            for k in 0..i.srcs.len() {
                if k == 2 && i.mnem == Op::Tex {
                    continue;
                }
                let Some(bits) = self.bits_of(i.srcs[k]) else { continue };
                let v = match near.get(&bits) {
                    Some(&(v, last)) if n - last <= self.const_gap => v,
                    _ => {
                        let v = self.materialize(&mut out, bits);
                        self.spill.remat.insert(v, bits);
                        v
                    }
                };
                near.insert(bits, (v, n));
                (i.srcs[k], i.swz[k]) = (v, 0);
                built += 1;
            }
            out.push(i);
        }
        debug!("borgc: {built} constant operand(s) by immediates");
        self.sel.prog = out;
        self.prune_window();
    }

    /// `rd = rs + k` without a new constant: ADDI, or LUI and ADDI and ADD.
    fn add_const(&mut self, out: &mut Vec<BorgInstr>, rs: u32, k: u32) -> u32 {
        if k == 0 {
            return rs;
        }
        let d = self.sel.fresh();
        if k < 2048 {
            out.push(BorgInstr { mnem: Op::Addi.into(), dst: d, srcs: vec![rs], swz: vec![0], imm: k });
            return d;
        }
        let (hi, lo) = (k.wrapping_add(0x800) >> 12, k & 0xFFF);
        let (u, v) = (self.sel.fresh(), self.sel.fresh());
        out.push(BorgInstr { mnem: Op::Lui.into(), dst: u, srcs: vec![], swz: vec![], imm: hi & 0xF_FFFF });
        out.push(BorgInstr { mnem: Op::Addi.into(), dst: v, srcs: vec![u], swz: vec![0], imm: lo });
        out.push(BorgInstr { mnem: Op::Add.into(), dst: d, srcs: vec![rs, v], swz: vec![0, 0], imm: 0 });
        d
    }

    /// ISA v2 (B8 step 11): TEX and TEXA by descriptor address. v1's control word names the
    /// texture (bits 7:0) and the sampler (15:8) by index; their descriptors lie in the tables
    /// the root table holds (words 1 and 2), 64 and 16 bytes apiece. TEXS takes the sampler's
    /// address and the rest of the word, TEX the texture's address. Per block: compute loses its
    /// registers at a barrier.
    fn tex_addresses(&mut self) {
        if !self.sel.prog.iter().any(|i| matches!(i.mnem, Mnem::Op(Op::Tex | Op::Texa))) {
            return;
        }
        let mut cache: HashMap<u32, u32> = HashMap::new();   // word of the root table -> its value
        // A constant control word's (texture address, sampler address, rest); the last TEXS's operands.
        let mut known: HashMap<u32, (u32, u32, u32)> = HashMap::new();
        let mut texs: Option<(u32, u32)> = None;
        let mut out = Vec::new();
        for (start, mut i) in block_starts(std::mem::take(&mut self.sel.prog)) {
            if start {
                cache.clear();
                known.clear();
                texs = None;
            }
            match i.mnem {
                // An earlier run's TEXS.
                Mnem::Op(Op::Texs) => texs = None,
                Mnem::Op(Op::Texa) => {
                    for k in 0..3 {
                        let (code, (v, c)) = self.window_copy((i.srcs[k], i.swz[k]));
                        out.extend(code);
                        (i.srcs[k], i.swz[k]) = (v, c);
                    }
                    i.mnem = Op::Texad.into();
                }
                Mnem::Op(Op::Tex) => {
                    let mut word = |s: &mut Self, out: &mut Vec<BorgInstr>, w: u32| -> u32 {
                        if let Some(&v) = cache.get(&w) {
                            return v;
                        }
                        let t = match cache.get(&0) {
                            Some(&t) => t,
                            None => {
                                let t = s.sel.fresh();
                                out.push(BorgInstr { mnem: Op::Csrrs.into(), dst: t, srcs: vec![], swz: vec![], imm: CSR_ROOT });
                                cache.insert(0, t);
                                t
                            }
                        };
                        let v = s.sel.fresh();
                        out.push(BorgInstr { mnem: Op::Lw.into(), dst: v, srcs: vec![t], swz: vec![0], imm: 4 * w });
                        cache.insert(w, v);
                        v
                    };
                    let bits = self.bits_of(i.srcs[2]);
                    let (ta, sa, rest) = match bits {
                        Some(b) if known.contains_key(&b) => known[&b],
                        Some(bits) => {
                            let (tb, sb) = (word(self, &mut out, 1), word(self, &mut out, 2));
                            let ta = self.add_const(&mut out, tb, 64 * (bits & 0xFF));
                            let sa = self.add_const(&mut out, sb, 16 * ((bits >> 8) & 0xFF));
                            // TEXS is no RISC-V form: r0 is a register there.
                            let rest = if bits >> 16 == 0 {
                                self.materialize(&mut out, 0)
                            } else {
                                let r = self.sel.fresh();
                                out.push(BorgInstr { mnem: Op::Lui.into(), dst: r, srcs: vec![], swz: vec![], imm: bits >> 12 & 0xF_FFF0 });
                                r
                            };
                            known.insert(bits, (ta, sa, rest));
                            (ta, sa, rest)
                        }
                        None => {
                            let (tb, sb) = (word(self, &mut out, 1), word(self, &mut out, 2));
                            let (code, c) = self.window_copy((i.srcs[2], i.swz[2]));
                            out.extend(code);
                            let mut op = |s: &mut Self, m: Op, srcs: Vec<(u32, u8)>, imm: u32| {
                                let d = s.sel.fresh();
                                let (srcs, swz) = srcs.into_iter().unzip();
                                out.push(BorgInstr { mnem: m.into(), dst: d, srcs, swz, imm });
                                (d, 0u8)
                            };
                            let t1 = op(self, Op::Andi, vec![c], 0xFF);
                            let t2 = op(self, Op::Slli, vec![t1], 6);
                            let ta = op(self, Op::Add, vec![(tb, 0), t2], 0);
                            let s1 = op(self, Op::Srli, vec![c], 8);
                            let s2 = op(self, Op::Andi, vec![s1], 0xFF);
                            let s3 = op(self, Op::Slli, vec![s2], 4);
                            let sa = op(self, Op::Add, vec![(sb, 0), s3], 0);
                            let r1 = op(self, Op::Srli, vec![c], 16);
                            let rest = op(self, Op::Slli, vec![r1], 16);
                            (ta.0, sa.0, rest.0)
                        }
                    };
                    if texs != Some((sa, rest)) {
                        out.push(BorgInstr { mnem: Op::Texs.into(), dst: NO_DST, srcs: vec![sa, rest], swz: vec![0, 0], imm: 0 });
                        texs = Some((sa, rest));
                    }
                    for k in 0..2 {
                        let (code, (v, c)) = self.window_copy((i.srcs[k], i.swz[k]));
                        out.extend(code);
                        (i.srcs[k], i.swz[k]) = (v, c);
                    }
                    (i.srcs[2], i.swz[2]) = (ta, 0);
                    i.mnem = Op::Texd.into();
                }
                _ => {}
            }
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// The CSR that holds what v1 keeps in a role register or window word, for this stage.
    fn sys_csr(&self, v: u32) -> Option<u32> {
        match (self.stage, self.sel.loc.get(&v)) {
            (Stage::Fragment, Some(&Loc::Fixed(r @ (30 | 31)))) => Some(CSR_FRAG_X + r as u32 - 30),
            (Stage::Fragment, Some(&Loc::Fixed(29))) => Some(CSR_FRAG_X + 2),
            (Stage::Fragment, Some(&Loc::Fixed(8))) => Some(CSR_FRAG_X + 3),
            (Stage::Fragment, Some(&Loc::Fixed(r @ 5..=7))) => Some(CSR_BARY0 + r as u32 - 5),
            (Stage::Vertex, Some(&Loc::Fixed(r @ (30 | 31)))) => Some(CSR_VERTEX_INDEX + r as u32 - 30),
            (Stage::Compute, Some(&Loc::Fixed(r @ (30 | 31)))) => Some(CSR_LOCAL_INDEX + r as u32 - 30),
            (Stage::Compute, Some(&Loc::Uniform(u @ 29..=31))) => Some(CSR_GROUP_ID_X + u as u32 - 29),
            _ => None,
        }
    }

    /// ISA v2 (B8 step 7): system values are read with CSRRS, once per block; the role registers
    /// and window words they live in today are no operands any more.
    pub(crate) fn sys_reads(&mut self) {
        let mut cache: HashMap<u32, u32> = HashMap::new();
        let mut out = Vec::new();
        for (start, mut i) in block_starts(std::mem::take(&mut self.sel.prog)) {
            if start {
                cache.clear();
            }
            for k in 0..i.srcs.len() {
                let Some(csr) = self.sys_csr(i.srcs[k]) else { continue };
                let v = match cache.get(&csr) {
                    Some(&v) => v,
                    None => {
                        let v = self.sel.fresh();
                        out.push(BorgInstr { mnem: Op::Csrrs.into(), dst: v, srcs: vec![], swz: vec![], imm: csr });
                        cache.insert(csr, v);
                        v
                    }
                };
                i.srcs[k] = v;
                i.swz[k] = 0;
            }
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// ISA v2 (B8 step 10): a vertex or fragment shader's LOAD and STORE of word `a` from LS_BASE
    /// become LW and SW at base + 4a, the base the root table's first word. A constant word, or a constant
    /// step from an address already formed in this block, goes into the immediate.
    fn root_accesses(&mut self) {
        if !self.sel.prog.iter().any(|i| matches!(i.mnem, Mnem::Op(Op::Load | Op::Store))) {
            return;
        }
        let mut out = Vec::new();
        let table = self.sel.prog.iter().find(|i| i.mnem == Op::Csrrs && i.imm == CSR_ROOT).map(|i| i.dst);
        let loaded = table.and_then(|t| self.sel.prog.iter().find(|i| i.mnem == Op::Lw && i.srcs == [t] && i.imm == 0)).map(|i| i.dst);
        let root = match loaded {
            Some(base) => base,
            None => {
                let (table, base) = (self.sel.fresh(), self.sel.fresh());
                out.push(BorgInstr { mnem: Op::Csrrs.into(), dst: table, srcs: vec![], swz: vec![], imm: CSR_ROOT });
                out.push(BorgInstr { mnem: Op::Lw.into(), dst: base, srcs: vec![table], swz: vec![0], imm: 0 });
                base
            }
        };
        let mut byte: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut plus: HashMap<u32, (u32, u32)> = HashMap::new();
        for (start, mut i) in block_starts(std::mem::take(&mut self.sel.prog)) {
            if start {
                byte.clear();
                plus.clear();
            }
            // Not every value is written once: what was formed from a rewritten one is stale.
            let d = i.dst;
            byte.retain(|&a, &mut (b, _)| a != d && b != d);
            plus.retain(|&v, &mut (x, _)| v != d && x != d);
            match i.mnem {
                Mnem::Op(Op::Iadd) if i.swz == [0, 0] && i.srcs[0] != d => {
                    if let Some(k) = self.bits_of(i.srcs[1]) {
                        plus.insert(i.dst, (i.srcs[0], k));
                    }
                }
                Mnem::Op(op @ (Op::Load | Op::Store)) => {
                    let (a, c) = (i.srcs[0], i.swz[0]);
                    let near = |b: (u32, u32), k: u32| Some((b.0, b.1.wrapping_add(k.wrapping_mul(4)))).filter(|&(_, o)| o < 2048);
                    let at = match (self.bits_of(a), plus.get(&a).and_then(|&(x, k)| byte.get(&x).map(|&b| (b, k)))) {
                        (Some(k), _) if c == 0 => near((root, 0), k),
                        (None, Some((b, k))) if c == 0 => near(b, k),
                        _ => None,
                    };
                    // Else the byte address of x for x + k (spills: lane key + slot), or of a itself.
                    let (x, k, xc) = match plus.get(&a) {
                        Some(&(x, k)) if c == 0 && k.wrapping_mul(4) < 2048 => (x, k, 0),
                        _ => (a, 0, c),
                    };
                    let (b, o) = at.unwrap_or_else(|| {
                        let (t, u) = (self.sel.fresh(), self.sel.fresh());
                        // A window word reaches a RISC-V instruction through a register.
                        let (code, (x, xc)) = self.window_copy((x, xc));
                        out.extend(code);
                        out.push(BorgInstr { mnem: Op::Slli.into(), dst: t, srcs: vec![x], swz: vec![xc], imm: 2 });
                        out.push(BorgInstr { mnem: Op::Add.into(), dst: u, srcs: vec![t, root], swz: vec![0, 0], imm: 0 });
                        if xc == 0 {
                            byte.insert(x, (u, 0));
                        }
                        (u, k.wrapping_mul(4))
                    });
                    i.srcs[0] = b;
                    i.swz[0] = 0;
                    i.imm = o;
                    // A spill slot's store runs in helper lanes too.
                    i.mnem = match op {
                        Op::Load => Op::Lw.into(),
                        _ if self.spill.addrs.contains(&a) => Op::Sws.into(),
                        _ => Op::Sw.into(),
                    };
                }
                _ => {}
            }
            out.push(i);
        }
        self.sel.prog = out;
    }

    pub fn assemble(&mut self) -> Option<Vec<u8>> {
        let (forced, mut extra_reserved) = fixed_regs(self);
        let mut alloc = allocate(self, &forced, &mut extra_reserved);
        // The fit counts live values; the allocator may still need a register or two more.
        let late = self.spill.late && matches!(self.stage, Stage::Fragment | Stage::Vertex);
        for margin in (0..4).take_while(|_| late) {
            if alloc.is_some() {
                break;
            }
            if self.spill_to_fit(&forced, &mut extra_reserved, &[20, 21, 22, 23], margin).is_err() {
                break;
            }
            let (forced, mut extra_reserved) = fixed_regs(self);
            alloc = allocate(self, &forced, &mut extra_reserved);
        }
        let Some(alloc) = alloc else {
            self.spill.alloc_failed = true;
            return None;
        };
        if let Err(e) = crate::legal::check(self, &alloc) {
            self.refuse(e);
            return None;
        }
        let Encoded { mut words, n_body, bad } = encode_program(self, &alloc);
        if bad > 0 {
            self.refuse(format!("{bad} instruction(s) without an encoding"));
            return None;
        }
        words.push(0x0000_0000);
        Some(build_blob(self, &words, n_body, &self.frag.out))
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Flags;
    use crate::opcodes::Op;
    use crate::state::{ShaderState, Stage};
    use crate::{BorgInstr, Mnem, NO_DST};

    fn op(op: Op, dst: u32, srcs: Vec<u32>) -> BorgInstr {
        let swz = vec![0; srcs.len()];
        BorgInstr { mnem: op.into(), dst, srcs, swz, imm: 0 }
    }

    /// The register the last LW's address is shifted from, and its offset (the first LW reads the root table).
    fn lw_base(c: &ShaderState) -> (u32, u32) {
        let lw = c.sel.prog.iter().rposition(|i| i.mnem == Op::Lw).unwrap();
        let slli = c.sel.prog[..lw].iter().rev().find(|i| i.mnem == Op::Slli).unwrap();
        (slli.srcs[0], c.sel.prog[lw].imm)
    }

    #[test]
    fn a_load_in_a_loop_uses_its_own_address() {
        let mut c = ShaderState::new(&[17, 18, 19], Stage::Vertex, Flags(0));
        let (x, v, y) = (c.sel.fresh(), c.sel.fresh(), c.sel.fresh());
        let (k, one) = (c.konst(3), c.konst(1));
        c.sel.prog = vec![
            op(Op::Iadd, v, vec![x, k]),
            BorgInstr { mnem: Mnem::Label, dst: NO_DST, srcs: vec![], swz: vec![], imm: 1 },
            op(Op::Load, y, vec![v]),
            op(Op::Iadd, x, vec![x, one]),
            BorgInstr { mnem: Op::Brnz.into(), dst: NO_DST, srcs: vec![x], swz: vec![0], imm: 1 },
        ];
        c.root_accesses();
        assert_eq!(lw_base(&c), (v, 0));
    }

    #[test]
    fn an_address_incremented_in_place_is_not_its_own_base() {
        let mut c = ShaderState::new(&[17, 18, 19], Stage::Vertex, Flags(0));
        let (x, y) = (c.sel.fresh(), c.sel.fresh());
        let k = c.konst(3);
        c.sel.prog = vec![op(Op::Iadd, x, vec![x, k]), op(Op::Load, y, vec![x])];
        c.root_accesses();
        assert_eq!(lw_base(&c), (x, 0));
    }
}
