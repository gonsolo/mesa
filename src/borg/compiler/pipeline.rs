use crate::emit::{build_blob, encode_program, Encoded};
use crate::fixed::{allocate, fixed_regs};
use crate::opt::{dce, fuse_fmadd, schedule_for_pressure};
use crate::state::{ShaderState, Stage};
use crate::survey::{survey, Survey};
use crate::{Flags, Loc};
use compiler::bindings::*;
use std::env;
use crate::opcodes::Op;

impl ShaderState {
    pub unsafe fn start(nir: *mut nir_shader, flags: Flags, out_select: i32) -> (ShaderState, *mut nir_function_impl, u32) {
        let mesa_stage = (*nir).info.stage();
        let stage = Stage::of(mesa_stage);
        if env::var("BORGC_DUMP_NIR").is_ok() {
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
        (st, entry, total)
    }

    /// After instruction selection: logs the result, and refuses the shader when a NIR op had no
    /// Borg instruction, so no wrong shader is emitted.
    pub fn check_selected(&mut self) -> Result<(), String> {
        self.refusal()?;
        debug!("borgc: selected {} Borg instruction(s) (virtual regs)", self.sel.prog.len());
        if env::var("BORGC_DUMP_ISA").is_ok() {
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
        if is_fragment {
            self.copy_frag_outputs();
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
        if let Some(limit) = self.spill.limit {
            dce(&mut self.sel.prog, &self.out_roots);
            self.spill_to_pressure(limit);
        }
        self.pool_window();
        dce(&mut self.sel.prog, &self.out_roots);
        let loc = &self.sel.loc;
        fuse_fmadd(&mut self.sel.prog, &self.out_roots, &|v| matches!(loc.get(&v), Some(Loc::Uniform(_))));
        schedule_for_pressure(&mut self.sel.prog, &self.out_roots);
        if self.stage == Stage::Fragment {
            self.log_uniformity();
        }
    }

    pub fn assemble(&mut self) -> Option<Vec<u8>> {
        let (forced, mut extra_reserved) = fixed_regs(self, &self.vert.pos_out, &self.frag.out);
        self.split_window_reads();
        let Some(alloc) = allocate(self, &forced, &mut extra_reserved) else {
            self.spill.alloc_failed = true;
            return None;
        };
        let Encoded { mut words, n_preload, n_body, pending } = encode_program(self, &alloc);
        words.push(0x0000_0000);
        Some(build_blob(self, &words, n_preload, n_body, pending, &self.frag.out))
    }
}
