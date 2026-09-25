// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// borgc — the Borg GPU shader compiler (Rust), called from the borgvk Vulkan
// driver. It lowers Mesa NIR to Borg ISA (FADD/FMUL/FMADD/FNEG/FSTEP/FRCP/TEX
// and friends, on the FP32 shader datapath) and emits the .borg blob the
// firmware loads. Modeled on
// Mesa's NAK (src/nouveau/compiler/nak).
//
// Status: SPIR-V→NIR→borgc is live. This pass classifies every NIR instruction
// against the Borg ISA — a coverage report that drives instruction selection.
// The arithmetic core (fmul/ffma → FMUL/FMADD) selects directly; the open work
// is lowering the UBO/I-O intrinsics to the firmware's uniform model and real
// register allocation + .borg emission.

#![allow(non_upper_case_globals)]

mod encode;
mod isel;
mod opt;
mod regalloc;

use compiler::bindings::*;
use compiler::nir::AsDef;
use encode::{emit_blob, encode, encode_fattr, encode_sout};
use isel::{borg_isel, resolve_vm};
use opt::{classify_uniform, dce, fuse_fmadd};
use regalloc::regalloc;
use std::env;

/// One selected Borg instruction in virtual-register form (pre-register-alloc).
/// `dst`/`srcs` are NIR SSA indices used directly as virtual registers; `swz` is
/// the scalar component each source reads from its (possibly vec4) def — needed to
/// pin a uniform/attribute load to the right column/component.
/// `BorgInstr::dst` for an instruction that writes no register.
///
/// The execution-mask ops and STORE have no destination -- their rd field
/// either is unused or carries something else entirely. Without a sentinel
/// they would look like definitions of vreg 0 to the register allocator,
/// which would then keep a register alive for a value nothing produces.
pub(crate) const NO_DST: u32 = u32::MAX;

/// A control-flow instruction to splice in before a given NIR block.
///
/// NIR's `if` maps onto the mask ops with no branching at all, and every
/// marker happens to land *before* some block: EXPUSH before the first `then`
/// block, EXELSE before the first `else` block, EXPOP before the block
/// following the `if`. That symmetry is why this is a single "before" map
/// rather than a pair of before/after lists.
struct CfMark {
    mnem: &'static str,
    /// Condition def index, for EXPUSH only.
    cond: Option<u32>,
}

pub(crate) struct BorgInstr {
    pub(crate) mnem: &'static str,
    pub(crate) dst: u32,
    pub(crate) srcs: Vec<u32>,
    pub(crate) swz: Vec<u8>,
}

/// Walk the NIR control-flow tree and record where mask instructions belong.
///
/// The instruction-selection walk below iterates blocks FLAT, in layout order,
/// which is exactly the order this tree linearizes to -- so the two can be
/// matched up by block pointer without restructuring selection at all. That
/// matters: selection is the part with all the operand and swizzle handling,
/// and threading control flow through it would have meant rewriting it.
///
/// Note what the flat walk does WITHOUT this: it emits the `then` and `else`
/// bodies one after the other with nothing to distinguish them, so both
/// execute unconditionally. Any shader whose control flow survived NIR's
/// flattening passes was already being miscompiled; this is what fixes it.
///
/// Loops are deliberately not handled -- see the caller, which refuses them
/// rather than emitting a body that runs exactly once.
/// Draw front end only (docs/B1_geometry_front_end.md): decompose a
/// dynamically-indexed load_ubo's byte offset into `(word-aligned base,
/// per-vertex word stride)`, recognizing `const + (gl_VertexIndex << shift)`
/// -- what NIR's own optimizer canonicalizes a UBO array indexed by
/// gl_VertexIndex to (verified against real cube.vert output, BORGC_DUMP_NIR)
/// -- through `resolve_vm` (so any mov/vecN wrapping already selected earlier
/// in the walk is transparent) and `prod`/`consts` (populated by that same
/// generic ALU selection, not a second walk of raw NIR).
///
/// A `load_vulkan_descriptor` result is treated as contributing 0 wherever it
/// appears in the offset, by construction rather than by proving it from the
/// IR: Borg has one binding per resource, no descriptor arrays, no
/// VK_DESCRIPTOR_TYPE_*_DYNAMIC (borgvk_descriptor_set.c never reads
/// pDynamicOffsets), so it never actually varies. (An earlier version of this
/// made that true at the NIR level, constant-folding the descriptor to a real
/// zero before this function ever ran -- which spuriously turned OTHER
/// multi-component load_consts the folding touched into "shader constants"
/// this compiler pins to GPRs, exhausting the const-register budget on a
/// shader that uses none. Recognizing the descriptor here instead has no such
/// side effect.)
///
/// Anything else is reported, not guessed at: the caller drops the shader
/// rather than emit an address for an offset shape this does not recognize.
fn decompose_vertex_offset(
    vec_map: &std::collections::HashMap<u32, Vec<(u32, u8)>>,
    prod: &std::collections::HashMap<u32, (nir_op, Vec<(u32, u8)>)>,
    consts: &std::collections::HashMap<u32, u32>,
    descriptor_defs: &std::collections::HashSet<u32>,
    vertex_id_def: u32,
    offset_def: u32,
) -> Option<(i32, i32)> {
    // Sum of `d`'s additive terms as (constant, dynamic stride if one leaf was
    // gl_VertexIndex, optionally shifted). nir_lower_explicit_io's
    // vec2_index_32bit_offset format adds the descriptor's own dynamic-offset
    // component into EVERY load_ubo address, not just a dynamically-indexed
    // one -- range_base=0's own offset in cube.vert (MVP column 0) is exactly
    // `descriptor.z` alone, no visible "+0" left for nir_opt_algebraic to have
    // simplified away -- so this has to walk the WHOLE additive tree, not
    // assume a single flat `const + dynamic` split.
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

unsafe fn collect_cf_marks(
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
            // A divergent loop needs the mask PLUS a way to ask "is any lane
            // still active" to decide the backward branch, and that
            // instruction does not exist yet. Emitting the body unmasked
            // would run every iteration for every lane.
            unsupported.push("loop");
            collect_cf_marks(nloop.iter_body(), marks, unsupported);
        }
    }
}

/// How a load_ubo def maps onto the firmware's uniform register convention
/// (cube.c `vktexcube_vs_uniform` layout — see the I/O map below). Carried per
/// def index so the encoder can pin each operand that reads it.
#[derive(Clone, Copy, PartialEq)]
enum Ubo {
    /// MVP column at this byte range_base (column = range_base/16). Component `c`
    /// → uniform u(8 + column*4 + c). Read directly as a funct3 uniform operand.
    Mvp(i32),
    /// position[gl_VertexIndex] vec4. Component `c` → uniform u(c) (firmware
    /// pre-fetches the current vertex into u0..u2). Pre-loaded into a GPR.
    Pos,
    /// Vertex attribute (texcoord). Component `c` → uniform u(6+c): the
    /// sequencer DMA pre-loads the per-vertex descriptor words [x,y,z,r,g,b,u,v]
    /// into uniform[0..7] before running the vertex shader, so u6=U and u7=V.
    /// In practice, vertex shader `store_output(VAR0)` is DCE'd (the sequencer
    /// snoops UV from the DMA stream directly, bypassing the vertex shader), so
    /// Attr values are eliminated before encoding and this path is latent.
    Attr,
    /// A fragment uniform read directly at this physical index via funct3
    /// (inv_area, per-vertex varyings staged by the sequencer at u12..u30).
    Uniform(u8),
    /// A value that lives in a fixed physical register, not allocated (the
    /// fragment edge-function attributes e0/e1/e2 in r0/r1/r2).
    Fixed(u8),
}

/// Self-test: confirms the Rust crate is linked and callable across the FFI.
#[no_mangle]
pub extern "C" fn borgc_selftest() -> u32 {
    0xB0_06
}

/// Compile the app's shader (NIR) to a Borg-ISA `.borg` blob and return it to the
/// caller. The blob (firmware `spirb_parse` format) is written into `out_buf` (up
/// to `buf_cap` bytes); `*out_len` is set to the blob's true size so the caller can
/// detect an undersized buffer (`*out_len > buf_cap` ⇒ nothing copied). Any of
/// `out_buf`/`out_len` may be null to skip blob return (diagnostics-only, as before).
/// Returns the instruction count (0 on null/empty shader).
///
/// # Safety
/// `nir` must be a valid `nir_shader` pointer from the Mesa runtime (or null).
/// `out_buf` must point to at least `buf_cap` writable bytes (or be null).
#[no_mangle]
pub unsafe extern "C" fn borgc_compile_nir(
    nir: *mut nir_shader,
    out_buf: *mut u8,
    buf_cap: u32,
    out_len: *mut u32,
) -> u32 {
    if !out_len.is_null() {
        *out_len = 0;
    }
    if nir.is_null() {
        return 0;
    }
    let stage = (*nir).info.stage();

    if env::var("BORGC_DUMP_NIR").is_ok() {
        if let Ok(s) = (*nir).to_string() {
            eprintln!("{s}");
        }
    }

    let mut total = 0u32;
    let mut alu_ok = 0u32;
    let mut alu_todo = 0u32;
    let mut tex = 0u32;
    let mut consts = 0u32;
    let mut intrinsics = 0u32;
    let mut other = 0u32;
    let mut todo: Vec<&'static str> = Vec::new();
    let mut note = |op: &'static str| {
        if !todo.contains(&op) {
            todo.push(op);
        }
    };

    let entry = nir_shader_get_entrypoint(nir);
    if !entry.is_null() {
        for block in (*entry).iter_blocks() {
            for instr in block.iter_instr_list() {
                total += 1;
                if let Some(alu) = instr.as_alu() {
                    if borg_isel(alu.op).is_some() {
                        alu_ok += 1;
                    } else {
                        alu_todo += 1;
                        note(alu.info().name());
                    }
                } else if instr.as_tex().is_some() {
                    tex += 1; // → TEX
                } else if let Some(i) = instr.as_intrinsic() {
                    intrinsics += 1;
                    note(i.info().name());
                } else if instr.as_load_const().is_some() {
                    consts += 1;
                } else {
                    other += 1;
                }
            }
        }
    }

    eprintln!(
        "borgc: stage={stage} instrs={total} | alu_ok={alu_ok} alu_todo={alu_todo} \
         tex={tex} const={consts} intrinsic={intrinsics} other={other}"
    );
    if !todo.is_empty() {
        eprintln!("borgc:   needs lowering/selection: {}", todo.join(", "));
    }

    use std::collections::HashMap;

    // Vector-construction / move resolution. mov and vecN don't become Borg ops;
    // instead they define, per result component, which (scalar def, component) it
    // really reads. cube.c's matrix multiply threads its accumulator through vec4
    // builds and swizzle reads (e.g. `%67 = fadd(%66, %65.x)` where %65 = vec4(...)),
    // so an operand must be resolved through these to its true scalar producer —
    // otherwise the data flow is severed and accumulators read garbage.
    // Single in-order selection walk. NIR is in SSA + topological order, so when we
    // reach an instruction all its sources are already in vec_map/prog. Each
    // value-producing op either (a) registers a vec_map entry (mov/vecN and the
    // vector-wide intrinsics, which resolve away component-wise) or (b) emits scalar
    // Borg ops. Operands are resolved through vec_map to their true scalar producer.
    // Synthetic results from expansions (ddx/ddy/fmax/tex/interp) use fresh vregs
    // above any NIR ssa index.
    let mut vec_map: HashMap<u32, Vec<(u32, u8)>> = HashMap::new();
    let mut prog: Vec<BorgInstr> = Vec::new();
    // Phi renames, collected during the selection walk and applied after it
    // (see the comment at the collection site).
    let mut phi_alias_final: HashMap<u32, u32> = HashMap::new();
    let mut next_vreg: u32 = 1_000_000;
    // Operand classification (populated here for fragment uniforms/attrs, and in the
    // I/O pass below for the vertex's load_ubo).
    let mut ubo: HashMap<u32, Ubo> = HashMap::new();
    // Barycentric weights w_i = e_i·inv_area, emitted once on the first load_input.
    let mut weights: Option<[u32; 3]> = None;
    // e0/e1/e2 (edge-function attrs r0/r1/r2) are Ubo::Fixed too, but unlike the
    // *other* Ubo::Fixed use (true shader constants pinned to r17+, e.g. lightDir),
    // they vary per pixel — exclude them from the uniformity classification's
    // "is this a per-triangle-constant root" test below.
    let mut per_pixel_fixed: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // Scalar load_const f32 bits (for the sRGB idiom match) and a producer map
    // (alu def → its op + resolved scalar srcs) for recognising the bcsel tree.
    let mut consts: HashMap<u32, u32> = HashMap::new();
    let mut prod: HashMap<u32, (nir_op, Vec<(u32, u8)>)> = HashMap::new();
    // TEX results occupy 4 consecutive regs (rd..rd+3 = R, G, B, A).
    let mut tex_dsts: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // TEX's control word (docs/B2_texture_unit.md), created on first use. Word 0
    // is texture 0, sampler 0, a plain sample with the LOD taken from the quad,
    // no offsets -- all a shader with one sampler2D needs. It is built in the
    // shader rather than pinned: FSTEP of -r30 is exactly +0 (r30 >= 0 in every
    // stage), so it costs two instructions and no constant register.
    let mut zero_ctl: Option<u32> = None;
    // Distinct texture derefs seen, to say so when a shader samples more than
    // one: every sample uses descriptor 0 until borgc maps bindings to indices.
    let mut tex_derefs: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // Shader constants (cube.frag's lightDir, push-constant word indices) live in
    // GPRs the firmware writes ONCE before the autonomous render, so they must
    // survive every stage that runs before the fragment on every triangle and
    // pixel: the vertex shaders (hand + borgc: r0-9, r24-26), the setup shader
    // (r0-16), the rasterizer ROM (r0-4), and the fragment's own fixed blocks
    // (r0-2 edges, r20-23 TEX, r24 alpha, r25 kill, r26-29 outputs, r30/31
    // pixel centre). That leaves r17-19, plus r23 in a shader that never
    // samples: TEX writes rd..rd+3, so with rd = 20 it overwrites r23. Handing
    // out r20 -- the next register after r19 -- once put fge's 1.0 where the
    // texel lands. Uniforms are no alternative: rast(u0-11) + frag
    // varyings(u12-30) fill 31 of 32. Records (reg, value), value a datapath
    // float or a raw integer.
    // Vertex shaders have none of the fragment-only reservations below (edge
    // attrs r0-2, TEX r20-23, alpha r24, kill r25) -- only r0-4 (gl_Position
    // output + the perspective-divide epilogue's scratch, both via `forced`/
    // regalloc's own reserved set, not this list) and r30/31 (VertexIndex,
    // never in the general pool). r5-16 is a generous, exclusively
    // vertex-stage pool, wide enough for a draw-mode vertex shader's LOAD
    // address constants (MVP base, the shared +1 increment, each
    // vertex-pulled array's base and stride) -- more of them than a legacy
    // vertex shader has ever needed, since none pinned any before.
    let const_regs: &[u8] = if stage == 0 {
        &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    } else if tex > 0 {
        &[17, 18, 19]
    } else {
        &[17, 18, 19, 23]
    };
    let mut const_reg_count: usize = 0;
    let alloc_const_reg = |count: &mut usize| -> u8 {
        assert!(*count < const_regs.len(),
            "borgc: shader needs more than {} constant registers ({:?})",
            const_regs.len(), const_regs);
        let reg = const_regs[*count];
        *count += 1;
        reg
    };
    // Pin a compile-time-constant integer to a GPR, reusing the same
    // register for every later reference to the same value (draw-mode LOAD
    // word indices and the literal 1 an address sequence increments by).
    let pin_const_int = |v: i32,
                          const_int_reg: &mut HashMap<i32, u8>,
                          const_reg_count: &mut usize,
                          const_uniforms: &mut Vec<(u8, u32)>|
     -> u8 {
        if let Some(&r) = const_int_reg.get(&v) {
            return r;
        }
        let r = alloc_const_reg(const_reg_count);
        const_uniforms.push((r, v as u32));
        const_int_reg.insert(v, r);
        r
    };
    // 1.0 for the inverting comparisons (fge/feq), created on first use. In a
    // fragment shader it is FSTEP(r30): r30 reads the pixel centre (>= 0.5)
    // during the fragment pass, so the result is exactly 1.0 and costs no
    // constant register. Vertex shaders see r30 = 0 and fall back to a const.
    let mut one_vreg: Option<u32> = None;
    // Word-index constants for push-constant LOADs, pinned to const GPRs on
    // first use and shared across every load_push_constant reaching the same
    // word (a vec4 field reads 4 consecutive words; a scalar read elsewhere
    // in the shader at the same offset reuses the same pin rather than
    // burning a second scarce const register on an identical value).
    let mut push_const_reg: HashMap<u32, u32> = HashMap::new();
    let mut const_uniforms: Vec<(u8, u32)> = Vec::new();
    // gl_varying_slot: VAR0=texcoord, VAR1=frag_pos (Mesa enum: VAR0 = 32).
    const VARYING_SLOT_VAR0: u32 = 32;
    const VARYING_SLOT_POS: u32 = 0;

    // --- Draw front end (docs/B1_geometry_front_end.md), gated behind an env
    // var so the legacy path -- and every shader compiled against it, incl.
    // the checked-in shader_blobs.h -- is untouched until borgvk moves over
    // (docs/B1's own "Coexistence" section). Everything below this point that
    // is draw_mode-specific is a parallel, independent I/O path: it shares
    // the generic ALU/tex/ddx selection above and nothing of the legacy
    // MVP/Pos/Attr/load_input scheme, which stays exactly as it was.
    let draw_mode = env::var("BORGC_DRAW_MODE").is_ok();
    // Moved up from the legacy I/O pass below so the draw-mode store_output
    // handling (in the main walk, ahead of that pass) can read it too; same
    // value, same meaning, in both modes.
    let frag_alpha = env::var("BORGC_FRAG_ALPHA").is_ok();
    // gl_VertexIndex's SSA def, once seen -- the index every vertex-pulling
    // load_ubo (position[], attr[]) is computed from. Always seen before any
    // load_ubo that depends on it: NIR/SPIR-V evaluates gl_VertexIndex where
    // the source first reads it, and a dynamic array index has to read it
    // before it can compute an offset from it.
    let mut vertex_id_def: Option<u32> = None;
    // A compile-time-constant integer, pinned to a GPR on first use and
    // reused for every later reference to the SAME value -- word indices for
    // draw-mode LOADs (MVP columns, vertex-pulling base/stride) and the
    // literal 1 a multi-word LOAD sequence increments its address by.
    // Deliberately separate from push_const_reg: that cache is word indices
    // into the push-constant staging buffer, a different LS_BASE-relative
    // address space than the vertex/uniform buffer draw-mode LOADs read.
    let mut const_int_reg: HashMap<i32, u8> = HashMap::new();
    // Vertex stage: resolved scalar producer per output location, built
    // in program order as store_output is seen, so a load_output reading
    // an earlier store in the SAME shader (`frag_pos = gl_Position.xyz`)
    // resolves -- SOUT has no matching load, so that is the only way such a
    // read can ever be answered. Also doubles as the SOUT-index source: a
    // location's SOUT indices are 4*(location - VARYING_SLOT_VAR0) upward
    // (VARYING_SLOT_POS's own components are never SOUT, so it does not
    // compete with this), a fixed mapping the fragment compile computes
    // identically without any shared state between the two compiles.
    let mut draw_output_stores: HashMap<u32, Vec<(u32, u8)>> = HashMap::new();
    let mut draw_pos_out: [Option<(u32, u8)>; 4] = [None; 4];
    let mut draw_frag_out: [Option<(u32, u8)>; 4] = [None; 4];
    let mut draw_out_roots: Vec<u32> = Vec::new();
    // load_vulkan_descriptor results seen, for decompose_vertex_offset.
    let mut descriptor_defs: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // Running (next expected word address, its address vreg) for consecutive
    // constant-address loads (the MVP columns: 4 separate load_ubo calls, 16
    // words back to back). Chaining them through ONE running IADD-by-1
    // sequence, continued across calls when a later one picks up exactly
    // where the previous left off, is what keeps this under the ~4-word
    // constant-register budget -- pinning each of the 16 word indices on its
    // own does not fit (see git history).
    let mut const_addr_chain: Option<(i32, u32)> = None;
    if !entry.is_null() {
        // Control-flow markers, collected from the CF tree so this flat block
        // walk can splice them in at the right seams. Must be here rather than
        // in the I/O pass below: the marks are instructions and have to
        // interleave with selection's output, not land after all of it.
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

        // Phi elimination, predication style.
        //
        // A phi at an `if`'s merge point selects between the two arms' values.
        // Under an execution mask there is nothing to select: BOTH arms
        // execute and write, and the mask decides whose write lands. So the
        // phi collapses to "every arm writes the same register" -- which is
        // achieved by renaming each arm's result to the phi's own def.
        //
        // Without this the phi's def has no producer borgc understands, the
        // store_output reading it is unreachable from any DCE root, and BOTH
        // ARMS ARE DELETED -- a 93-instruction shader emitting 7. That is how
        // this was found.
        //
        // The rename is safe for the SSA a structured `if` produces, where each
        // arm's value feeds only the phi. A source that is also live elsewhere
        // would need a real copy instead; borgc warns rather than silently
        // renaming one of those out from under its other users.
        for block in (*entry).iter_blocks() {
            // Emitted before the block's own instructions, which is what makes
            // EXPUSH gate the `then` body and EXPOP land after the `if` rather
            // than inside it.
            if let Some(ms) = cf_marks.get(&(block as *const nir_block as usize)) {
                for m in ms {
                    prog.push(BorgInstr {
                        mnem: m.mnem,
                        dst: NO_DST,
                        srcs: m.cond.map(|c| vec![c]).unwrap_or_default(),
                        swz: m.cond.map(|_| vec![0u8]).unwrap_or_default(),
                    });
                }
            }
            for instr in block.iter_instr_list() {
                // Phi -> predicated register sharing.
                //
                // Componentwise, because a phi is often a vector and its
                // sources are vec_map entries built by nir_op_vec*, not
                // instructions with a dst -- renaming whole defs reaches
                // nothing. For each component, every arm's producer is renamed
                // onto the first arm's vreg, and the phi resolves there too.
                // Both arms then write one register and the execution mask
                // decides whose write survives, which is what a phi means
                // under predication.
                //
                // Safe for the SSA a structured `if` produces, where an arm's
                // value feeds only the phi. A value also live elsewhere would
                // need a real copy.
                if let Some(phi) = instr.as_phi() {
                    let n = phi.def.num_components as usize;
                    let srcs: Vec<u32> =
                        phi.iter_srcs().map(|s| s.src.as_def().index).collect();
                    if let Some(&first) = srcs.first() {
                        let mut comps: Vec<(u32, u8)> = Vec::with_capacity(n);
                        for c in 0..n {
                            let (d0, c0) = resolve_vm(&vec_map, first, c as u8);
                            for &other in &srcs[1..] {
                                let (d1, _) = resolve_vm(&vec_map, other, c as u8);
                                if d1 != d0 {
                                    phi_alias_final.insert(d1, d0);
                                }
                            }
                            comps.push((d0, c0));
                        }
                        vec_map.insert(phi.def.index, comps);
                    }
                    continue;
                }
                if let Some(alu) = instr.as_alu() {
                    match alu.op {
                        nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => {
                            let c: Vec<(u32, u8)> = alu
                                .srcs_as_slice()
                                .iter()
                                .map(|s| (s.src.as_def().index, s.swizzle[0]))
                                .collect();
                            vec_map.insert(alu.def.index, c);
                        }
                        nir_op_mov | nir_op_i2i16 | nir_op_i2i32 | nir_op_u2u16
                        | nir_op_u2u32 => {
                            let s = &alu.srcs_as_slice()[0];
                            let n = alu.def.num_components as usize;
                            let c: Vec<(u32, u8)> =
                                (0..n).map(|k| (s.src.as_def().index, s.swizzle[k])).collect();
                            vec_map.insert(alu.def.index, c);
                        }
                        // Float comparisons, built from FSTEP.
                        //
                        // FSTEP(a) is 1.0 for a strictly positive and 0.0
                        // otherwise (BorgLane.computeFstep), so `a < b` is
                        // FSTEP(b - a) and the rest follow by swapping or
                        // inverting. Subtraction is FADD(a, FNEG(b)) -- the
                        // ISA has no subtract.
                        //
                        // These exist for the execution mask: without them a
                        // condition like `if (x < y)` is a NIR op borgc
                        // cannot select, so EXPUSH would read a register no
                        // instruction ever wrote. The result is 1.0 or 0.0,
                        // and the mask tests raw bits against zero, so
                        // both encodings land correctly without a bool
                        // conversion.
                        nir_op_flt | nir_op_fge | nir_op_fneu | nir_op_feq => {
                            let sl = alu.srcs_as_slice();
                            let a = resolve_vm(&vec_map, sl[0].src.as_def().index, sl[0].swizzle[0]);
                            let b = resolve_vm(&vec_map, sl[1].src.as_def().index, sl[1].swizzle[0]);

                            // diff(hi, lo) = hi - lo, as a fresh vreg.
                            let mut diff = |hi: (u32, u8), lo: (u32, u8), nv: &mut u32, prog: &mut Vec<BorgInstr>| {
                                let neg = *nv; *nv += 1;
                                let d = *nv; *nv += 1;
                                prog.push(BorgInstr { mnem: "FNEG", dst: neg, srcs: vec![lo.0], swz: vec![lo.1] });
                                prog.push(BorgInstr { mnem: "FADD", dst: d, srcs: vec![hi.0, neg], swz: vec![hi.1, 0] });
                                d
                            };

                            match alu.op {
                                // a < b  ->  FSTEP(b - a)
                                nir_op_flt => {
                                    let d = diff(b, a, &mut next_vreg, &mut prog);
                                    prog.push(BorgInstr { mnem: "FSTEP", dst: alu.def.index, srcs: vec![d], swz: vec![0] });
                                }
                                // a >= b  ->  1 - FSTEP(b - a), as
                                // FADD(1.0, FNEG(step)). The 1.0 comes from a
                                // pinned constant register, same mechanism as
                                // the existing const_uniforms.
                                nir_op_fge => {
                                    let d = diff(b, a, &mut next_vreg, &mut prog);
                                    let st = next_vreg; next_vreg += 1;
                                    let nst = next_vreg; next_vreg += 1;
                                    prog.push(BorgInstr { mnem: "FSTEP", dst: st, srcs: vec![d], swz: vec![0] });
                                    prog.push(BorgInstr { mnem: "FNEG", dst: nst, srcs: vec![st], swz: vec![0] });
                                    let one = match one_vreg {
                                        Some(v) => v,
                                        None => {
                                            let v = next_vreg;
                                            next_vreg += 1;
                                            if stage == 4 {
                                                let c = next_vreg;
                                                next_vreg += 1;
                                                ubo.insert(c, Ubo::Fixed(30));
                                                per_pixel_fixed.insert(c);
                                                prog.push(BorgInstr { mnem: "FSTEP", dst: v, srcs: vec![c], swz: vec![0] });
                                            } else {
                                                let reg = alloc_const_reg(&mut const_reg_count);
                                                const_uniforms.push((reg, 1.0f32.to_bits()));
                                                ubo.insert(v, Ubo::Fixed(reg));
                                            }
                                            one_vreg = Some(v);
                                            v
                                        }
                                    };
                                    prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![one, nst], swz: vec![0, 0] });
                                }
                                // a != b  ->  FSTEP(a-b) + FSTEP(b-a). At most
                                // one term is 1.0, so the sum is a clean 0/1
                                // and needs no clamp.
                                nir_op_fneu => {
                                    let dab = diff(a, b, &mut next_vreg, &mut prog);
                                    let dba = diff(b, a, &mut next_vreg, &mut prog);
                                    let s1 = next_vreg; next_vreg += 1;
                                    let s2 = next_vreg; next_vreg += 1;
                                    prog.push(BorgInstr { mnem: "FSTEP", dst: s1, srcs: vec![dab], swz: vec![0] });
                                    prog.push(BorgInstr { mnem: "FSTEP", dst: s2, srcs: vec![dba], swz: vec![0] });
                                    prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![s1, s2], swz: vec![0, 0] });
                                }
                                // a == b  ->  1 - (a != b)
                                _ => {
                                    let dab = diff(a, b, &mut next_vreg, &mut prog);
                                    let dba = diff(b, a, &mut next_vreg, &mut prog);
                                    let s1 = next_vreg; next_vreg += 1;
                                    let s2 = next_vreg; next_vreg += 1;
                                    let ne = next_vreg; next_vreg += 1;
                                    let nne = next_vreg; next_vreg += 1;
                                    prog.push(BorgInstr { mnem: "FSTEP", dst: s1, srcs: vec![dab], swz: vec![0] });
                                    prog.push(BorgInstr { mnem: "FSTEP", dst: s2, srcs: vec![dba], swz: vec![0] });
                                    prog.push(BorgInstr { mnem: "FADD", dst: ne, srcs: vec![s1, s2], swz: vec![0, 0] });
                                    prog.push(BorgInstr { mnem: "FNEG", dst: nne, srcs: vec![ne], swz: vec![0] });
                                    let one = match one_vreg {
                                        Some(v) => v,
                                        None => {
                                            let v = next_vreg;
                                            next_vreg += 1;
                                            if stage == 4 {
                                                let c = next_vreg;
                                                next_vreg += 1;
                                                ubo.insert(c, Ubo::Fixed(30));
                                                per_pixel_fixed.insert(c);
                                                prog.push(BorgInstr { mnem: "FSTEP", dst: v, srcs: vec![c], swz: vec![0] });
                                            } else {
                                                let reg = alloc_const_reg(&mut const_reg_count);
                                                const_uniforms.push((reg, 1.0f32.to_bits()));
                                                ubo.insert(v, Ubo::Fixed(reg));
                                            }
                                            one_vreg = Some(v);
                                            v
                                        }
                                    };
                                    prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![one, nne], swz: vec![0, 0] });
                                }
                            }
                        }
                        // fmax(0, x) → x · FSTEP(x)  (FSTEP(x)=1 if x>0 else 0).
                        nir_op_fmax => {
                            let s = alu.srcs_as_slice();
                            let a_zero = s[0].comp_as_uint(0) == Some(0);
                            let xs = if a_zero { &s[1] } else { &s[0] };
                            let x = resolve_vm(&vec_map, xs.src.as_def().index, xs.swizzle[0]);
                            let vstep = next_vreg;
                            next_vreg += 1;
                            prog.push(BorgInstr { mnem: "FSTEP", dst: vstep, srcs: vec![x.0], swz: vec![x.1] });
                            prog.push(BorgInstr { mnem: "FMUL", dst: alu.def.index, srcs: vec![x.0, vstep], swz: vec![x.1, 0] });
                        }
                        // linearToSrgb idiom → FSRGB: bcsel(fge(knee,x), x·12.92,
                        // 1.055·pow(x,1/2.4)-0.055). Recognised by the linear branch
                        // fmul(x, 12.92); x is the value to sRGB-encode.
                        nir_op_bcsel => {
                            let s = alu.srcs_as_slice();
                            let then_def = s[1].src.as_def().index;
                            let mut x: Option<(u32, u8)> = None;
                            if let Some((op, tsrcs)) = prod.get(&then_def) {
                                if *op == nir_op_fmul && tsrcs.len() == 2 {
                                    let is12 = |d: u32| {
                                        consts.get(&d).map_or(false, |&b| {
                                            (f32::from_bits(b) - 12.92).abs() < 0.1
                                        })
                                    };
                                    x = if is12(tsrcs[0].0) {
                                        Some(tsrcs[1])
                                    } else if is12(tsrcs[1].0) {
                                        Some(tsrcs[0])
                                    } else {
                                        None
                                    };
                                }
                            }
                            if let Some(x) = x {
                                let v = next_vreg;
                                next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FSRGB", dst: v, srcs: vec![x.0], swz: vec![x.1] });
                                vec_map.insert(alu.def.index, vec![(v, 0)]);
                            } else {
                                // non-sRGB bcsel: pass through the then-value.
                                let s0 = resolve_vm(&vec_map, s[1].src.as_def().index, s[1].swizzle[0]);
                                vec_map.insert(alu.def.index, vec![s0]);
                            }
                        }
                        _ => {
                            if let Some(mnem) = borg_isel(alu.op) {
                                if mnem != "mov" {
                                    let mut srcs = Vec::new();
                                    let mut swz = Vec::new();
                                    for s in alu.srcs_as_slice() {
                                        let (d, c) =
                                            resolve_vm(&vec_map, s.src.as_def().index, s.swizzle[0]);
                                        srcs.push(d);
                                        swz.push(c);
                                    }
                                    let pairs: Vec<(u32, u8)> =
                                        srcs.iter().cloned().zip(swz.iter().cloned()).collect();
                                    prod.insert(alu.def.index, (alu.op, pairs));
                                    prog.push(BorgInstr { mnem, dst: alu.def.index, srcs, swz });
                                }
                            }
                            // else: unhandled alu (fpow/fge feed the sRGB idiom) — skip.
                        }
                    }
                } else if let Some(intr) = instr.as_intrinsic() {
                    // Vector-wide quad derivatives → per-component scalar DDX/DDY,
                    // registered in vec_map so reads of the result resolve correctly.
                    let deriv = match intr.intrinsic {
                        nir_intrinsic_ddx | nir_intrinsic_ddx_coarse | nir_intrinsic_ddx_fine => Some("DDX"),
                        nir_intrinsic_ddy | nir_intrinsic_ddy_coarse | nir_intrinsic_ddy_fine => Some("DDY"),
                        _ => None,
                    };
                    if let Some(mnem) = deriv {
                        let src = intr.srcs_as_slice()[0].as_def().index;
                        let n = intr.def.num_components as usize;
                        let raw: Vec<u32> = (0..n)
                            .map(|c| {
                                let (sd, sc) = resolve_vm(&vec_map, src, c as u8);
                                let v = next_vreg;
                                next_vreg += 1;
                                prog.push(BorgInstr { mnem, dst: v, srcs: vec![sd], swz: vec![sc] });
                                v
                            })
                            .collect();
                        // (The FP16 datapath needed DDX scaled by 32 here, because
                        // |cross(ddx, ddy)|^2 underflowed below FP16's minimum
                        // normal at 128x128. FP32 has no such floor.)
                        let comps: Vec<(u32, u8)> = raw.into_iter().map(|v| (v, 0u8)).collect();
                        vec_map.insert(intr.def.index, comps);
                    } else if intr.intrinsic == nir_intrinsic_load_input && draw_mode {
                        // Draw front end: FATTR + the perspective-correct barycentrics
                        // already in r5-r7 at fragment start (docs/B1_geometry_front_end.md),
                        // one component at a time -- FATTR loads a component's three
                        // per-vertex values into rd..rd+2, which the next FATTR overwrites,
                        // so each component's FMUL/FMADD chain must consume its own before
                        // requesting the next.
                        let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                        let base_index = 4 * loc.wrapping_sub(VARYING_SLOT_VAR0);
                        let bary = [5u8, 6, 7].map(|r| {
                            let v = next_vreg; next_vreg += 1;
                            ubo.insert(v, Ubo::Fixed(r));
                            per_pixel_fixed.insert(v);
                            v
                        });
                        let n = intr.def.num_components as usize;
                        let comps: Vec<(u32, u8)> = (0..n)
                            .map(|c| {
                                let fattr_rd = next_vreg; next_vreg += 3; // rd, rd+1, rd+2
                                prog.push(BorgInstr {
                                    mnem: "FATTR", dst: fattr_rd, srcs: vec![],
                                    swz: vec![(base_index + c as u32) as u8],
                                });
                                let t0 = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![bary[0], fattr_rd], swz: vec![0, 0] });
                                let t1 = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![bary[1], fattr_rd + 1, t0], swz: vec![0, 0, 0] });
                                let res = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMADD", dst: res, srcs: vec![bary[2], fattr_rd + 2, t1], swz: vec![0, 0, 0] });
                                (res, 0u8)
                            })
                            .collect();
                        vec_map.insert(intr.def.index, comps);
                    } else if intr.intrinsic == nir_intrinsic_load_input {
                        // Interpolated varying. Borg has no fixed-function interpolation:
                        // the shader computes it barycentrically. Emit the weights once
                        // (w_i = e_i·inv_area; e0/1/2 = attrs r0/1/2, inv_area = u12), then
                        // per component: w0·v0 + w1·v1 + w2·v2 over the per-vertex uniforms.
                        if weights.is_none() {
                            let mut w = [0u32; 3];
                            for (i, wi) in w.iter_mut().enumerate() {
                                let e = next_vreg; next_vreg += 1;
                                ubo.insert(e, Ubo::Fixed(i as u8));        // e_i in r0/r1/r2
                                per_pixel_fixed.insert(e); // NOT uniform: varies per pixel
                                let inv = next_vreg; next_vreg += 1;
                                ubo.insert(inv, Ubo::Uniform(12));         // inv_area = u12
                                let v = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMUL", dst: v, srcs: vec![e, inv], swz: vec![0, 0] });
                                *wi = v;
                            }
                            weights = Some(w);
                        }
                        let w = weights.unwrap();
                        let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                        // Per-vertex uniform base per varying (sequencer convention,
                        // each group stored (v2,v1,v0)): VAR0 texcoord → u13.., VAR1
                        // frag_pos → u19.., depth → u28...
                        let var = loc.wrapping_sub(VARYING_SLOT_VAR0); // 0 = texcoord, 1 = frag_pos
                        let n = intr.def.num_components as usize;
                        let comps: Vec<(u32, u8)> = (0..n)
                            .map(|c| {
                                let base = ((if var == 0 { 13 } else { 19 }) + (c as u32) * 3) as u8;
                                // uniforms for (v0, v1, v2) = (base+2, base+1, base+0)
                                let uv: Vec<u32> = [2u8, 1, 0]
                                    .iter()
                                    .map(|&o| {
                                        let v = next_vreg; next_vreg += 1;
                                        ubo.insert(v, Ubo::Uniform(base + o));
                                        v
                                    })
                                    .collect();
                                let t0 = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![w[0], uv[0]], swz: vec![0, 0] });
                                let t1 = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![w[1], uv[1], t0], swz: vec![0, 0, 0] });
                                let res = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "FMADD", dst: res, srcs: vec![w[2], uv[2], t1], swz: vec![0, 0, 0] });
                                (res, 0u8)
                            })
                            .collect();
                        vec_map.insert(intr.def.index, comps);
                    } else if intr.intrinsic == nir_intrinsic_load_push_constant {
                        // layout(push_constant) reads. Reached here only after
                        // borg_nir_passes.c's nir_lower_explicit_io turns the
                        // pointer-deref access into this intrinsic -- WITHOUT
                        // that pass this never arrives; it arrives as
                        // load_deref instead, which is unselectable and DCEs
                        // the whole shader to nothing (found empirically, not
                        // assumed, the first time this was tried).
                        //
                        // `base` is the field's static byte offset; `src(0)`
                        // is a dynamic offset for array-indexed push constants,
                        // which this does not support yet -- only a
                        // compile-time-constant src(0) is handled, mirroring
                        // how divergent LOOPS are refused rather than silently
                        // mis-lowered. A non-constant offset is warned about
                        // and the intrinsic is left unselected (DCE removes
                        // whatever depended on it, same failure shape as an
                        // unhandled op, not a wrong answer).
                        //
                        // LOAD addresses whole 32-bit DRAM words
                        // (LS_BASE + rs1<<2), so each 4-byte-aligned field
                        // maps onto exactly one word -- push-constant data is
                        // naturally word-granular already, no repacking
                        // needed on the hardware side.
                        let dyn_off = intr.get_src(0).comp_as_uint(0);
                        if let Some(dyn_off) = dyn_off {
                            let base = intr.base() + dyn_off as i32;
                            let n = intr.def.num_components as usize;
                            let comps: Vec<(u32, u8)> = (0..n)
                                .map(|c| {
                                    let byte_off = base + 4 * (c as i32);
                                    assert_eq!(byte_off % 4, 0,
                                        "push constant field at byte {byte_off} is not word-aligned");
                                    let word_idx = (byte_off / 4) as u32;
                                    let idx_reg = *push_const_reg.entry(word_idx).or_insert_with(|| {
                                        let reg = alloc_const_reg(&mut const_reg_count);
                                        // RAW integer, not a float -- this
                                        // becomes rs1 for LOAD, a word INDEX,
                                        // not a shader-visible float value.
                                        const_uniforms.push((reg, word_idx));
                                        let v = next_vreg;
                                        next_vreg += 1;
                                        ubo.insert(v, Ubo::Fixed(reg));
                                        v
                                    });
                                    let dst = next_vreg;
                                    next_vreg += 1;
                                    prog.push(BorgInstr {
                                        mnem: "LOAD", dst, srcs: vec![idx_reg], swz: vec![0],
                                    });
                                    (dst, 0u8)
                                })
                                .collect();
                            vec_map.insert(intr.def.index, comps);
                        } else {
                            eprintln!(
                                "borgc: WARNING push constant at base={} read with a \
                                 non-constant dynamic offset -- not supported, dropped",
                                intr.base()
                            );
                        }
                    }
                    // load_ubo/store_output handled in the I/O pass below
                    // (legacy) or inline here (draw mode).
                    else if intr.intrinsic == nir_intrinsic_load_vulkan_descriptor {
                        descriptor_defs.insert(intr.def.index);
                    } else if draw_mode && intr.intrinsic == nir_intrinsic_load_vertex_id {
                        // r30 = VertexIndex at vertex-shader start
                        // (docs/B1_geometry_front_end.md); referenced directly,
                        // like the fragment stage's e0-e2/barycentrics, never
                        // through vec_map/regalloc (see resolve_op's Ubo::Fixed
                        // arm -- checked before the alloc fallback for every
                        // source, whether or not that source has a producing
                        // BorgInstr of its own, which load_vertex_id does not).
                        ubo.insert(intr.def.index, Ubo::Fixed(30));
                        vertex_id_def = Some(intr.def.index);
                    } else if draw_mode && intr.intrinsic == nir_intrinsic_load_ubo {
                        let offset_def = intr.get_src(1).as_def().index;
                        let n = intr.def.num_components as usize;
                        // vertex_id_def is always Some by now if the shader reads
                        // gl_VertexIndex anywhere, which every load_ubo here does
                        // transitively via the descriptor's own address term
                        // (see decompose_vertex_offset's doc) even when THIS
                        // particular load's own offset has no vid-scaled term
                        // (the MVP columns); u32::MAX is a sentinel no real NIR
                        // index can equal, for the degenerate shader that never
                        // reads gl_VertexIndex at all.
                        let (base_words, stride_words) = match decompose_vertex_offset(
                            &vec_map, &prod, &consts, &descriptor_defs,
                            vertex_id_def.unwrap_or(u32::MAX), offset_def,
                        ) {
                            Some(bs) => bs,
                            None => {
                                eprintln!(
                                    "borgc: WARNING load_ubo offset (def {offset_def}) is \
                                     not a recognized draw-mode address shape -- dropped"
                                );
                                (-1, 0)
                            }
                        };
                        if base_words >= 0 {
                            // A physical register pinned by pin_const_int is not
                            // itself a valid `prog` source: `srcs` holds SSA/vreg
                            // indices, resolved to a physical register via `ubo`
                            // at encode time, and a small integer like a GPR
                            // number can easily collide with a real NIR SSA
                            // index. Every reference below therefore mints its
                            // OWN fresh vreg (next_vreg, far above any real NIR
                            // index) wrapping the pinned register, exactly like
                            // e0-e2/lightDir elsewhere in this file -- never the
                            // register number directly.
                            let mut fixed = |reg: u8, next_vreg: &mut u32, ubo: &mut HashMap<u32, Ubo>| -> u32 {
                                let v = *next_vreg; *next_vreg += 1;
                                ubo.insert(v, Ubo::Fixed(reg));
                                v
                            };
                            // The running address register this load's words come
                            // from. A constant address (stride_words == 0, the MVP
                            // columns) continues an earlier load_ubo's chain when it
                            // left off exactly here (const_addr_chain), instead of
                            // pinning every one of the (here) 16 word indices to its
                            // own constant register -- which does not fit the ~4-word
                            // budget (see const_addr_chain's doc).
                            let addr = match (stride_words, const_addr_chain) {
                                (0, Some((next, a))) if next == base_words => a,
                                (0, _) => {
                                    let base_reg = pin_const_int(base_words, &mut const_int_reg, &mut const_reg_count, &mut const_uniforms);
                                    fixed(base_reg, &mut next_vreg, &mut ubo)
                                }
                                _ => {
                                    let vid = vertex_id_def.unwrap();
                                    let stride_reg =
                                        pin_const_int(stride_words, &mut const_int_reg, &mut const_reg_count, &mut const_uniforms);
                                    let base_reg =
                                        pin_const_int(base_words, &mut const_int_reg, &mut const_reg_count, &mut const_uniforms);
                                    let stride_v = fixed(stride_reg, &mut next_vreg, &mut ubo);
                                    let tmp = next_vreg; next_vreg += 1;
                                    prog.push(BorgInstr { mnem: "IMUL", dst: tmp, srcs: vec![vid, stride_v], swz: vec![0, 0] });
                                    let base_v = fixed(base_reg, &mut next_vreg, &mut ubo);
                                    let a = next_vreg; next_vreg += 1;
                                    prog.push(BorgInstr { mnem: "IADD", dst: a, srcs: vec![tmp, base_v], swz: vec![0, 0] });
                                    a
                                }
                            };
                            let one_reg = pin_const_int(1, &mut const_int_reg, &mut const_reg_count, &mut const_uniforms);
                            let mut cur = addr;
                            let comps: Vec<(u32, u8)> = (0..n)
                                .map(|c| {
                                    let dst = next_vreg; next_vreg += 1;
                                    prog.push(BorgInstr { mnem: "LOAD", dst, srcs: vec![cur], swz: vec![0] });
                                    if c + 1 < n {
                                        let one_v = fixed(one_reg, &mut next_vreg, &mut ubo);
                                        let next_a = next_vreg; next_vreg += 1;
                                        prog.push(BorgInstr {
                                            mnem: "IADD", dst: next_a, srcs: vec![cur, one_v], swz: vec![0, 0],
                                        });
                                        cur = next_a;
                                    }
                                    (dst, 0u8)
                                })
                                .collect();
                            // Leave the chain one past the last word THIS load
                            // read, for a later constant-address load_ubo to pick
                            // up (a stray unconsumed IADD if none does, which dce
                            // then drops like any other dead instruction).
                            if stride_words == 0 {
                                let one_v = fixed(one_reg, &mut next_vreg, &mut ubo);
                                let past = next_vreg; next_vreg += 1;
                                prog.push(BorgInstr { mnem: "IADD", dst: past, srcs: vec![cur, one_v], swz: vec![0, 0] });
                                const_addr_chain = Some((base_words + n as i32, past));
                            }
                            vec_map.insert(intr.def.index, comps);
                        }
                    } else if draw_mode && intr.intrinsic == nir_intrinsic_store_output {
                        let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                        let src = intr.get_src(0).as_def().index;
                        let ncomp = intr.get_src(0).num_components() as usize;
                        let comps: Vec<(u32, u8)> = (0..ncomp).map(|c| resolve_vm(&vec_map, src, c as u8)).collect();
                        if stage == 0 && loc == VARYING_SLOT_POS {
                            for (c, &v) in comps.iter().enumerate().take(4) {
                                draw_pos_out[c] = Some(v);
                                draw_out_roots.push(v.0);
                            }
                        } else if stage == 4 {
                            let ncomp2 = if frag_alpha { 4 } else { 3 };
                            for (c, &v) in comps.iter().enumerate().take(ncomp2) {
                                draw_frag_out[c] = Some(v);
                                draw_out_roots.push(v.0);
                            }
                        } else {
                            let base_index = 4 * loc.wrapping_sub(VARYING_SLOT_VAR0);
                            for (c, &(vd, vc)) in comps.iter().enumerate() {
                                let index = base_index + c as u32;
                                assert!(index < 256, "borgc: SOUT index {index} does not fit a byte");
                                prog.push(BorgInstr {
                                    mnem: "SOUT", dst: NO_DST, srcs: vec![vd], swz: vec![index as u8],
                                });
                                draw_out_roots.push(vd);
                                let _ = vc; // see decompose_vertex_offset's doc: draw-mode producers are scalar (c=0) by construction
                            }
                        }
                        draw_output_stores.insert(loc, comps);
                    } else if draw_mode && intr.intrinsic == nir_intrinsic_load_output {
                        let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                        match draw_output_stores.get(&loc) {
                            Some(v) => { vec_map.insert(intr.def.index, v.clone()); }
                            None => eprintln!(
                                "borgc: WARNING load_output of location {loc}, never stored earlier \
                                 in program order -- dropped (draw mode resolves it from the store, \
                                 not a real load; see decompose_vertex_offset's neighbor)"
                            ),
                        }
                    }
                } else if let Some(tex) = instr.as_tex() {
                    // Texture sample → TEX rd, u, v, ctl: R, G, B, A to rd..rd+3.
                    // Only a plain implicit-LOD sample of a 2D texture so far;
                    // anything else is reported rather than sampled wrongly.
                    let srcs = tex.srcs_as_slice();
                    if tex.op != nir_texop_tex {
                        eprintln!("borgc: WARNING texture op {} not supported yet, dropped", tex.op);
                        continue;
                    }
                    if let Some(t) = srcs.iter().find(|s| s.src_type == nir_tex_src_texture_deref) {
                        tex_derefs.insert(t.src.as_def().index);
                        if tex_derefs.len() == 2 {
                            eprintln!("borgc: WARNING more than one texture: all sample descriptor 0");
                        }
                    }
                    let coord = srcs
                        .iter()
                        .find(|s| s.src_type == nir_tex_src_coord)
                        .map(|s| s.src.as_def().index);
                    if let Some(cd) = coord {
                        let (ud, uc) = resolve_vm(&vec_map, cd, 0);
                        let (vd, vc) = resolve_vm(&vec_map, cd, 1);
                        let ctl = match zero_ctl {
                            Some(v) => v,
                            None => {
                                let c = next_vreg;
                                let neg = next_vreg + 1;
                                let z = next_vreg + 2;
                                next_vreg += 3;
                                ubo.insert(c, Ubo::Fixed(30));
                                per_pixel_fixed.insert(c);
                                prog.push(BorgInstr { mnem: "FNEG", dst: neg, srcs: vec![c], swz: vec![0] });
                                prog.push(BorgInstr { mnem: "FSTEP", dst: z, srcs: vec![neg], swz: vec![0] });
                                zero_ctl = Some(z);
                                z
                            }
                        };
                        let tex_v = next_vreg;
                        next_vreg += 1;
                        prog.push(BorgInstr {
                            mnem: "TEX", dst: tex_v, srcs: vec![ud, vd, ctl], swz: vec![uc, vc, 0],
                        });
                        tex_dsts.insert(tex_v);
                        vec_map.insert(
                            tex.def.index,
                            vec![(tex_v, 0), (tex_v, 1), (tex_v, 2), (tex_v, 3)],
                        );
                    }
                } else if let Some(lc) = instr.as_load_const() {
                    let n = lc.def.num_components as usize;
                    if n == 1 {
                        consts.insert(lc.def.index, unsafe { lc.values()[0].u32_ });
                    } else {
                        // Vector constant (lightDir) → pin components to constant
                        // GPRs (firmware writes them once via MMIO).
                        let comps: Vec<(u32, u8)> = (0..n)
                            .map(|c| {
                                let reg = alloc_const_reg(&mut const_reg_count);
                                let bits = unsafe { lc.values()[c].u32_ };
                                const_uniforms.push((reg, bits));
                                let v = next_vreg;
                                next_vreg += 1;
                                ubo.insert(v, Ubo::Fixed(reg));
                                (v, 0u8)
                            })
                            .collect();
                        vec_map.insert(lc.def.index, comps);
                    }
                }
            }
        }
    }
    eprintln!("borgc: selected {} Borg instruction(s) (virtual regs)", prog.len());
    if env::var("BORGC_DUMP_ISA").is_ok() {
        for i in &prog {
            let s: Vec<String> = i.srcs.iter().map(|v| format!("v{v}")).collect();
            eprintln!("borgc:   {} v{}, {}", i.mnem, i.dst, s.join(", "));
        }
    }

    // I/O map: pin the shader interface to the firmware register convention.
    //   load_ubo range_base (cube.c vktexcube_vs_uniform layout):
    //     0/16/32/48 → MVP columns    → uniforms u8..u23 (col-major; u = 8 + rb/4)
    //     64..639    → position[idx]  → uniforms u0..u2 (firmware pre-fetches the
    //                                    current vertex; the index math is dead)
    //     640..      → attr[idx]      → texcoord varying (firmware-handled)
    //   store_output io_semantics.location (low 7 bits of IO_SEMANTICS):
    //     VARYING_SLOT_POS(0) → gl_Position → output regs r0..r3 (sequencer-snooped)
    //     VAR0/VAR1           → texcoord/frag_pos varyings (firmware-handled)
    let mut mvp_loads = 0u32;
    let mut pos_loads = 0u32;
    let mut attr_loads = 0u32;
    let mut gl_position: Option<u32> = None;
    let mut varying_outs = 0u32;
    // (`ubo` is declared above the selection walk so the fragment path can populate it.)
    // Output roots for DCE: every value a store_output consumes is live.
    let mut out_roots: Vec<u32> = Vec::new();
    // gl_Position component c (x/y/z/w) → the SSA value that produces it. These
    // clip coords are pinned to r0..r3 so the perspective-divide epilogue and the
    // sequencer's clipReg snoop find them where the firmware convention expects.
    let mut pos_out: [Option<u32>; 4] = [None; 4];
    // Fragment colour uFragColor → output regs: RGB to r26/r27/r28, A to r24.
    // The hardware ABI block is r24=A, r25=Kill, r26..r28=RGB, r29=Z.
    //
    // Alpha is opt-in and off by default. It is only meaningful to a
    // BorgConfig.hasBlend build -- without a blend stage the dispatcher
    // ignores r24 entirely -- and emitting it unconditionally would change
    // the codegen of every existing fragment shader: the alpha component
    // becomes a live output root, so the instructions computing it stop
    // being dead code and the shader grows. Off by default keeps today's
    // shaders byte-identical; the driver sets this when it binds a pipeline
    // with blendEnable.
    let mut frag_out: [Option<u32>; 4] = [None; 4];
    // Legacy-only: draw mode built pos_out/frag_out/out_roots inline in the
    // main walk above (draw_pos_out/draw_frag_out/draw_out_roots), in program
    // order, so a load_output reading back an earlier store_output resolves
    // -- SOUT has no matching load, so that is the only way such a read can
    // ever be answered, which rules out a second, later pass like this one.
    // gl_position deliberately stays None in draw mode: it is what gates the
    // perspective-divide epilogue below, and draw mode wants the RAW clip
    // coordinates in r0-r3 (BorgSetupRom does its own homogeneous divide) --
    // running that epilogue would corrupt them.
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
                            ubo.insert(intr.def.index, kind);
                        }
                        nir_intrinsic_store_output => {
                            let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                            let src = intr.srcs_as_slice()[0].as_def().index;
                            if stage == 0 && loc == VARYING_SLOT_POS {
                                // gl_Position is a vec4 stored whole (wrmask=xyzw);
                                // resolve each component to its scalar producer and
                                // pin it to r0..r3 (also the DCE roots). Varyings
                                // (texcoord/frag_pos) are firmware-handled — not rooted.
                                gl_position = Some(src);
                                let n = vec_map.get(&src).map_or(1, |v| v.len()).min(4);
                                for c in 0..n {
                                    let (d, _) = resolve_vm(&vec_map, src, c as u8);
                                    pos_out[c] = Some(d);
                                    out_roots.push(d);
                                }
                            } else if stage == 4 {
                                // Fragment colour uFragColor (vec4) → r26/27/28 rgb,
                                // plus r24 alpha when the blend path is in use.
                                let ncomp = if frag_alpha { 4 } else { 3 };
                                let n = vec_map.get(&src).map_or(1, |v| v.len()).min(ncomp);
                                for c in 0..n {
                                    let (d, _) = resolve_vm(&vec_map, src, c as u8);
                                    frag_out[c] = Some(d);
                                    out_roots.push(d);
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
    if draw_mode {
        for (c, v) in draw_pos_out.iter().enumerate() {
            pos_out[c] = v.map(|(d, _)| d);
        }
        for (c, v) in draw_frag_out.iter().enumerate() {
            frag_out[c] = v.map(|(d, _)| d);
        }
        out_roots.extend(draw_out_roots.iter().copied());
        eprintln!(
            "borgc: draw-mode I/O map — {} vertex-pulling/MVP load(s), {} SOUT(s), \
             gl_Position={}",
            descriptor_defs.len(), // proxy: one load_ubo per descriptor use
            draw_output_stores.len(),
            draw_pos_out[0].is_some(),
        );
    } else {
        eprintln!(
            "borgc: I/O map — MVP {mvp_loads}→u8..u23, position {pos_loads}→u0..u2, \
             attr {attr_loads}→u6/u7 (DMA, DCE'd); gl_Position=v{}→r0..r3 ({varying_outs} varying out)",
            gl_position.map_or("?".to_string(), |v| v.to_string())
        );
    }

    // Fragment depth output (target boilerplate): cube.frag writes no depth, but
    // the tile-buffer depth test needs r29 = interpolated z. z is staged per-vertex
    // at u28-30 ((v2,v1,v0) → v0=u30); emit w0·u30 + w1·u29 + w2·u28 → r29.
    // Draw mode: r29 is FragCoord.z already, from the raster ROM
    // (docs/B1_geometry_front_end.md) -- weights stays None (only the legacy
    // load_input path populates it), so this never fires there, correctly:
    // the fragment shader not writing r29 IS draw mode's depth output.
    let mut frag_z: Option<u32> = None;
    if stage == 4 {
        if let Some(w) = weights {
            let uz: Vec<u32> = [30u8, 29, 28]
                .iter()
                .map(|&idx| {
                    let v = next_vreg;
                    next_vreg += 1;
                    ubo.insert(v, Ubo::Uniform(idx));
                    v
                })
                .collect();
            let t0 = next_vreg; next_vreg += 1;
            prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![w[0], uz[0]], swz: vec![0, 0] });
            let t1 = next_vreg; next_vreg += 1;
            prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![w[1], uz[1], t0], swz: vec![0, 0, 0] });
            let zr = next_vreg; next_vreg += 1;
            prog.push(BorgInstr { mnem: "FMADD", dst: zr, srcs: vec![w[2], uz[2], t1], swz: vec![0, 0, 0] });
            out_roots.push(zr);
            frag_z = Some(zr);
        }
    }

    // Apply the phi renames BEFORE dce. Each arm's producer now writes the
    // phi's register directly, so the mask -- not a select -- picks the winner.
    // Order matters: dce roots at the shader outputs and walks back, and until
    // the rename happens the arms are unreachable from those roots, so running
    // it after would mean renaming instructions dce had already deleted.
    if !phi_alias_final.is_empty() {
        for i in prog.iter_mut() {
            if let Some(&d) = phi_alias_final.get(&i.dst) {
                i.dst = d;
            }
        }
    }

    dce(&mut prog, &out_roots);
    fuse_fmadd(&mut prog, &out_roots);

    // Uniformity classification (fragment shaders only — vertex/setup shaders
    // have no per-pixel divergence to hoist away from). Roots: any read of a
    // per-vertex-staged uniform (Ubo::Uniform — constant across the triangle,
    // e.g. inv_area, UV/frag_pos/z of v0..v2) or a true shader constant
    // (Ubo::Fixed, excluding the per-pixel edge-attr use of the same variant).
    // See opt::classify_uniform for the propagation rule and why DDX/DDY are
    // unconditional roots on this architecture.
    if stage == 4 {
        let uniform_roots: std::collections::HashSet<u32> = ubo
            .iter()
            .filter_map(|(&v, u)| match u {
                Ubo::Uniform(_) => Some(v),
                Ubo::Fixed(_) if !per_pixel_fixed.contains(&v) => Some(v),
                _ => None,
            })
            .collect();
        let uniform = classify_uniform(&prog, &uniform_roots);
        if env::var("BORGC_DUMP_ISA").is_ok() {
            let n_uniform = prog.iter().filter(|i| uniform.contains(&i.dst)).count();
            eprintln!(
                "borgc: uniformity — {} of {} instr(s) are primitive-uniform (hoistable)",
                n_uniform,
                prog.len()
            );
            for i in &prog {
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
    let is_vertex = stage == 0; // MESA_SHADER_VERTEX
    if is_vertex {
        // gl_Position components → r0..r3.
        for (c, v) in pos_out.iter().enumerate() {
            if let Some(def) = v {
                forced.insert(*def, c as u8);
            }
        }
        // r4: the perspective-divide epilogue's scratch, reserved only when
        // that epilogue actually runs (gl_position.is_some(), gated the same
        // way the epilogue itself is emitted below) -- draw mode has none,
        // wanting r0..r3 left as the RAW clip coordinates BorgSetupRom does
        // its own homogeneous divide from, so reserving r4 there would only
        // shrink its register pool for nothing.
        if gl_position.is_some() {
            extra_reserved.push(4);
        }
        // Only the const_regs prefix actually pinned (const_reg_count) --
        // reserving the whole 12-register pool regardless of use starved the
        // general ALU pool for a shader that only pins a few (a 4x4 matmul
        // alone needs more than 30-17 live registers at once, spilling to
        // r0/0.0 with no real spill support -- see regalloc's own doc).
        // Without reserving even that prefix, regalloc's general pool could
        // hand one of them to an ordinary ALU temporary, clobbering a pinned
        // LOAD-address constant -- latent since no vertex shader pinned any
        // before draw mode's LOADs.
        extra_reserved.extend_from_slice(&const_regs[..const_reg_count]);
    } else {
        // Fragment: colour → r26/27/28; edge-function attrs occupy r0/r1/r2; the
        // TEX result occupies a fixed 4-reg block r20..r23.
        for (c, v) in frag_out.iter().enumerate() {
            if let Some(def) = v {
                // Alpha is r24, not r29 -- r29 is the interpolated depth
                // output, so the naive 26+c would collide with it.
                forced.insert(*def, if c == 3 { 24 } else { 26 + c as u8 });
            }
        }
        for &t in &tex_dsts {
            forced.insert(t, 20);
        }
        if let Some(zr) = frag_z {
            forced.insert(zr, 29); // interpolated depth → r29
        }
        // r0-2 attrs, r4 (unused by a fragment shader, but reserved here too
        // -- matching every existing fragment compile's register assignment
        // exactly, rather than only freeing r4 where it is actually unused,
        // which reassigns everything after the first spot that would have
        // taken it and changes the checked-in shader_blobs.h), r17-19 (+ r23)
        // constants, r20-23 TEX, r26-29 outputs.
        extra_reserved.extend_from_slice(&[0, 1, 2, 4, 17, 18, 19, 21, 22, 23, 26, 27, 28, 29]);
        // r24 (alpha) only when it is actually an output. Reserving it
        // unconditionally would shrink the allocator's pool for every
        // existing shader to no purpose.
        if frag_out[3].is_some() {
            extra_reserved.push(24);
        }
    }

    let alloc = regalloc(&prog, &forced, &extra_reserved);

    // Encode the selected+allocated instructions into Borg machine words, pinning
    // shader inputs to the firmware's uniform convention. The Borg core has ONE
    // uniform read port: funct3 is a selector (1=rs1, 2=rs2, 3=rs3 reads the
    // uniform indexed by that register field), so each op reads at most one
    // uniform. cube.c's transform is position×MVP, so positions are pre-loaded
    // into high GPRs (firmware idiom `FADD g, u<comp>, r30, f3=1` → g = u[comp])
    // and the MVP column is read in place as the single uniform operand. `mov`
    // ops are register copies handled by coalescing (skipped).
    let mut words: Vec<u32> = Vec::new();
    let mut pos_gpr: std::collections::HashMap<(u32, u8), u8> = HashMap::new();
    let mut next_pre: u8 = 24; // r24..r29: above regalloc's low-GPR fill
    for i in &prog {
        if i.mnem == "mov" {
            continue;
        }
        for (s, &c) in i.srcs.iter().zip(i.swz.iter()) {
            if ubo.get(s) == Some(&Ubo::Pos)
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
        if tex_dsts.contains(&s) {
            return (alloc.get(&s).map_or(0, |&r| r + c), false);
        }
        match ubo.get(&s) {
            Some(&Ubo::Mvp(rb)) => ((8 + (rb / 16) * 4 + c as i32) as u8, true),
            Some(&Ubo::Pos) => (*pos_gpr.get(&(s, c)).unwrap_or(&0), false),
            Some(&Ubo::Uniform(idx)) => (idx, true),
            Some(&Ubo::Fixed(reg)) => (reg, false),
            Some(&Ubo::Attr) => (6 + c, true),  // u6=U, u7=V (DMA vertex descriptor)
            _ => (*alloc.get(&s).unwrap_or(&0), false),
        }
    };
    for i in &prog {
        if i.mnem == "mov" {
            continue;
        }
        // Execution-mask ops: no destination, and EXPUSH's single source is a
        // condition rather than an arithmetic operand, so they bypass the
        // operand folding below entirely.
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
        // SOUT/FATTR (docs/B1_geometry_front_end.md): the 10-bit index is not
        // a register operand, so it does not fit encode()'s rd/rs1/rs2/rs3
        // shape -- swz[0] carries it instead (see the draw-mode load_input
        // and store_output codegen). u8 covers every index this compiler can
        // produce today (record_shift=8, five varying components); a future
        // shader with more needs swz widened, not a silent wraparound, hence
        // the assert rather than an `as u16` truncation.
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

        // pos.w fold: under the 3-component-position ABI pos.w is the constant 1.0,
        // so a multiply by it is the identity. col3·pos.w (+acc) collapses to a load
        // of col3 (added to the accumulator) — exactly the hand-written shader's
        // direct col3 bias, and avoids reading pos.w from u3 (which holds color.r).
        let posw = i
            .srcs
            .iter()
            .zip(i.swz.iter())
            .take(2) // only the multiply operands (rs1/rs2) can be pos.w
            .position(|(s, &c)| ubo.get(s) == Some(&Ubo::Pos) && c == 3);
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
            if tex_dsts.contains(s) {
                r[k] = alloc.get(s).map_or(0, |&rr| rr + c); // TEX result rd+c
                continue;
            }
            match ubo.get(s) {
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

    // Fixed-function epilogue (vertex only): perspective divide + viewport. The
    // body left clip coords in r0..r3 (uniforms u8..u23 are viewport-baked by the
    // firmware's cache_ts_mvp, so screen = clip'/w covers the full viewport map).
    // r4 = 1/clip_w; screen_x/y and ndc_z = clip'/w → r0/r1/r2, snooped by the
    // sequencer. Matches seq_vert_shader's tail; HALT terminates.
    if is_vertex && gl_position.is_some() {
        words.push(encode("FRCP", 4, 3, 0, 0, 0).unwrap()); // r4 = 1/clip_w
        words.push(encode("FMUL", 0, 0, 4, 0, 0).unwrap()); // screen_x
        words.push(encode("FMUL", 1, 1, 4, 0, 0).unwrap()); // screen_y
        words.push(encode("FMUL", 2, 2, 4, 0, 0).unwrap()); // ndc_z
        words.push(0x0000_0000); // HALT
    } else if !is_vertex {
        words.push(0x0000_0000); // fragment: HALT terminates (no epilogue)
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
    if !const_uniforms.is_empty() {
        let cs: Vec<String> = const_uniforms.iter().map(|(r, v)| format!("r{r}={v:#06x}")).collect();
        eprintln!("borgc: const regs (firmware-staged via MMIO): {}", cs.join(" "));
    }
    if env::var("BORGC_DUMP_ISA").is_ok() {
        let hex: Vec<String> = words.iter().map(|w| format!("{w:#010x}")).collect();
        eprintln!("borgc:   first words: {}", hex.join(" "));
    }

    // Serialize the .borg blob (software/borg/borg_spirb.c format). For the
    // autonomous TBR sequencer only instrs[] matter — they're uploaded straight to
    // SEQ_VERT_SHADER_ADDR with the conventions baked in; the output_regs list is
    // descriptive (the snooped screen regs r0..r2). The blob is the artifact the
    // firmware loads to replace the hand-written seq_vert_shader.
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
    let blob = emit_blob(&words, &outputs, &const_uniforms);
    eprintln!("borgc: .borg blob = {} bytes ({} instr, {} output regs)", blob.len(), words.len(), outputs.len());
    if let Ok(prefix) = env::var("BORGC_EMIT_BLOB") {
        let path = format!("{prefix}.{}.borg", if is_vertex { "vert" } else { "frag" });
        match std::fs::write(&path, &blob) {
            Ok(()) => eprintln!("borgc: wrote {path}"),
            Err(e) => eprintln!("borgc: failed to write {path}: {e}"),
        }
    }
    // Return the blob to the C caller (borgvk_pipeline.c stores it per stage and
    // ships it to the firmware over serial). out_len always reports the true size;
    // copy only if it fits so the caller can grow its buffer on mismatch.
    if !out_len.is_null() {
        *out_len = blob.len() as u32;
    }
    if !out_buf.is_null() && (blob.len() as u32) <= buf_cap {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), out_buf, blob.len());
    }
    total
}

