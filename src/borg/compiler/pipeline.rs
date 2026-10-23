use crate::emit::{build_blob, encode_program, Encoded};
use crate::fixed::{allocate, fixed_regs};
use crate::opt::{dce, fuse_fmadd, schedule_for_pressure};
use crate::state::{ShaderState, Stage};
use crate::survey::{survey, Survey};
use crate::encode::CSR_ROOT;
use crate::{BorgInstr, Flags, Loc, Mnem};
use std::collections::{HashMap, HashSet};
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
        let srgb = self.sel.prog.iter().any(|i| i.mnem == Op::Fsrgb);
        self.skipped.retain(|&op| !(matches!(op, "fpow" | "fge") && srgb));
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
        }
        self.vert.pos_out = self.outs.pos.map(|v| v.map(|(d, _)| d));
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
        // Before spilling, so its pressure counts; again below for the pool's and the spills' loads.
        if matches!(self.stage, Stage::Vertex | Stage::Fragment) {
            self.root_accesses();
        }
        if let Some(limit) = self.spill.limit {
            dce(&mut self.sel.prog, &self.out_roots);
            self.spill_to_pressure(limit);
        }
        self.pool_window();
        dce(&mut self.sel.prog, &self.out_roots);
        if matches!(self.stage, Stage::Vertex | Stage::Fragment) {
            self.root_accesses();
        }
        let loc = &self.sel.loc;
        fuse_fmadd(&mut self.sel.prog, &self.out_roots, &|v| matches!(loc.get(&v), Some(Loc::Uniform(_))));
        schedule_for_pressure(&mut self.sel.prog, &self.out_roots);
        self.copy_tex_results();
        if self.stage == Stage::Fragment {
            self.log_uniformity();
        }
    }

    /// ISA v2 (B8 step 10): a vertex or fragment shader's LOAD and STORE of word `a` from LS_BASE
    /// become LW and SW at root + 4a; ROOT holds the same base. A constant word, or a constant
    /// step from an address already formed in this block, goes into the immediate.
    fn root_accesses(&mut self) {
        if !self.sel.prog.iter().any(|i| matches!(i.mnem, Mnem::Op(Op::Load | Op::Store))) {
            return;
        }
        let mut out = Vec::new();
        let root = match self.sel.prog.first() {
            Some(i) if i.mnem == Op::Csrrs && i.imm == CSR_ROOT => i.dst,
            _ => {
                let root = self.sel.fresh();
                out.push(BorgInstr { mnem: Op::Csrrs.into(), dst: root, srcs: vec![], swz: vec![], imm: CSR_ROOT });
                root
            }
        };
        let mut byte: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut plus: HashMap<u32, (u32, u32)> = HashMap::new();
        for mut i in std::mem::take(&mut self.sel.prog) {
            // Not every value is written once: what was formed from a rewritten one is stale.
            let d = i.dst;
            byte.retain(|&a, &mut (b, _)| a != d && b != d);
            plus.retain(|&v, &mut (x, _)| v != d && x != d);
            match i.mnem {
                Mnem::Op(Op::Expush | Op::Exelse | Op::Expop | Op::Brz | Op::Brnz | Op::Jmp) | Mnem::Label => byte.clear(),
                Mnem::Op(Op::Iadd) if i.swz == [0, 0] => {
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
                        let (x, xc) = if matches!(self.sel.loc.get(&x), Some(Loc::Uniform(_))) {
                            let w = self.sel.fresh();
                            out.push(BorgInstr { mnem: Op::Fmov.into(), dst: w, srcs: vec![x], swz: vec![xc], imm: 0 });
                            (w, 0)
                        } else {
                            (x, xc)
                        };
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
                    i.mnem = if op == Op::Load { Op::Lw.into() } else { Op::Sw.into() };
                }
                _ => {}
            }
            out.push(i);
        }
        self.sel.prog = out;
    }

    /// Every TEX writes the same register block: a result still read after the next TEX is copied
    /// out right after its own.
    fn copy_tex_results(&mut self) {
        let is_tex = |i: &BorgInstr, d: &HashSet<u32>| i.mnem == Op::Tex && d.contains(&i.dst);
        let texs: Vec<usize> = (0..self.sel.prog.len()).filter(|&i| is_tex(&self.sel.prog[i], &self.tex_dsts)).collect();
        for w in texs.windows(2).rev() {
            let (p, next) = (w[0], w[1]);
            let t = self.sel.prog[p].dst;
            let mut comps: Vec<u8> = self.sel.prog[next + 1..]
                .iter()
                .flat_map(|i| i.srcs.iter().zip(&i.swz).filter(|(s, _)| **s == t).map(|(_, &c)| c))
                .collect();
            comps.sort();
            comps.dedup();
            let copies: Vec<(u8, u32)> = comps.iter().map(|&c| (c, self.sel.fresh())).collect();
            for i in &mut self.sel.prog[next + 1..] {
                for (s, c) in i.srcs.iter_mut().zip(i.swz.iter_mut()) {
                    if let Some(&(_, v)) = copies.iter().find(|(k, _)| *s == t && k == c) {
                        (*s, *c) = (v, 0);
                    }
                }
            }
            for (k, &(c, v)) in copies.iter().enumerate() {
                self.sel.prog.insert(p + 1 + k, BorgInstr { mnem: Op::Ior.into(), dst: v, srcs: vec![t, t], swz: vec![c, c], imm: 0 });
            }
        }
    }

    pub fn assemble(&mut self) -> Option<Vec<u8>> {
        let (forced, mut extra_reserved) = fixed_regs(self, &self.vert.pos_out);
        self.split_window_reads();
        let Some(alloc) = allocate(self, &forced, &mut extra_reserved) else {
            self.spill.alloc_failed = true;
            return None;
        };
        let Encoded { mut words, n_preload, n_body, pending } = encode_program(self, &alloc);
        if pending > 0 {
            self.refuse(format!("{pending} operand(s) without a register"));
            return None;
        }
        words.push(0x0000_0000);
        Some(build_blob(self, &words, n_preload, n_body, pending, &self.frag.out))
    }
}
