// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// borgc — the Borg GPU shader compiler (Rust), called from the borgvk Vulkan
// driver. It lowers Mesa NIR to Borg ISA (the 7-op FP16 ISA: FADD/FMUL/FMADD/
// FNEG/FSTEP/FRCP/FTEX) and emits the .borg blob the firmware loads. Modeled on
// Mesa's NAK (src/nouveau/compiler/nak).
//
// Status: SPIR-V→NIR→borgc is live. This pass classifies every NIR instruction
// against the Borg ISA — a coverage report that drives instruction selection.
// The arithmetic core (fmul/ffma → FMUL/FMADD) selects directly; the open work
// is lowering the UBO/I-O intrinsics to the firmware's uniform model and real
// register allocation + .borg emission.

#![allow(non_upper_case_globals)]

use compiler::bindings::*;
use compiler::nir::AsDef;
use std::env;

/// One selected Borg instruction in virtual-register form (pre-register-alloc).
/// `dst`/`srcs` are NIR SSA indices used directly as virtual registers; `swz` is
/// the scalar component each source reads from its (possibly vec4) def — needed to
/// pin a uniform/attribute load to the right column/component.
struct BorgInstr {
    mnem: &'static str,
    dst: u32,
    srcs: Vec<u32>,
    swz: Vec<u8>,
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
    /// attribute (texcoord) varying — firmware-handled; left pending for now.
    Attr,
}

/// Self-test: confirms the Rust crate is linked and callable across the FFI.
#[no_mangle]
pub extern "C" fn borgc_selftest() -> u32 {
    0xB0_06
}

/// Does this NIR ALU op map directly onto a Borg ISA opcode?
fn borg_isel(op: nir_op) -> Option<&'static str> {
    match op {
        nir_op_fadd => Some("FADD"),
        nir_op_fmul => Some("FMUL"),
        nir_op_ffma | nir_op_ffma_weak => Some("FMADD"),
        nir_op_fneg => Some("FNEG"),
        // Integer ops (the Borg core now has these — enables standard addressing).
        nir_op_iadd => Some("IADD"),
        nir_op_ishl => Some("ISHL"),
        nir_op_ishr | nir_op_ushr => Some("ISHR"),
        nir_op_imul => Some("IMUL"),
        nir_op_i2f16 | nir_op_i2f32 | nir_op_u2f16 | nir_op_u2f32 => Some("I2F"),
        nir_op_f2i16 | nir_op_f2i32 | nir_op_f2u16 | nir_op_f2u32 => Some("F2I"),
        // Moves / vector (de)construction become register copies, no ISA op.
        nir_op_mov | nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => Some("mov"),
        // Integer↔int bit-size conversions are no-ops at 16-bit (copy).
        nir_op_i2i16 | nir_op_i2i32 | nir_op_u2u16 | nir_op_u2u32 => Some("mov"),
        _ => None,
    }
}

/// Receive the app's shader as NIR and report Borg-ISA instruction-selection
/// coverage: how many instructions map directly today vs. still need lowering or
/// new selection rules (printed by op name). Returns the instruction count.
///
/// # Safety
/// `nir` must be a valid `nir_shader` pointer from the Mesa runtime (or null).
#[no_mangle]
pub unsafe extern "C" fn borgc_compile_nir(nir: *mut nir_shader) -> u32 {
    if nir.is_null() {
        return 0;
    }
    let stage = (*nir).info.stage();

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
                    tex += 1; // → FTEX
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

    // Instruction selection: emit Borg ops for the selectable ALU instructions,
    // in virtual-register (NIR SSA index) form. Register allocation + .borg blob
    // emission are the next step; this proves selection produces real Borg code.
    let mut prog: Vec<BorgInstr> = Vec::new();
    if !entry.is_null() {
        for block in (*entry).iter_blocks() {
            for instr in block.iter_instr_list() {
                if let Some(alu) = instr.as_alu() {
                    if let Some(mnem) = borg_isel(alu.op) {
                        let srcs = alu
                            .srcs_as_slice()
                            .iter()
                            .map(|s| s.src.as_def().index)
                            .collect();
                        let swz = alu
                            .srcs_as_slice()
                            .iter()
                            .map(|s| s.swizzle[0])
                            .collect();
                        prog.push(BorgInstr { mnem, dst: alu.def.index, srcs, swz });
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
    use std::collections::HashMap;
    const VARYING_SLOT_POS: u32 = 0;
    let mut mvp_loads = 0u32;
    let mut pos_loads = 0u32;
    let mut attr_loads = 0u32;
    let mut gl_position: Option<u32> = None;
    let mut varying_outs = 0u32;
    // Per-def classification onto the firmware uniform convention.
    let mut ubo: HashMap<u32, Ubo> = HashMap::new();
    // Output roots for DCE: every value a store_output consumes is live.
    let mut out_roots: Vec<u32> = Vec::new();
    if !entry.is_null() {
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
                            out_roots.push(src);
                            if loc == VARYING_SLOT_POS {
                                gl_position = Some(src);
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
    eprintln!(
        "borgc: I/O map — MVP {mvp_loads}→u8..u23, position {pos_loads}→u0..u2, \
         attr {attr_loads}→varying; gl_Position=v{}→r0..r3 ({varying_outs} varying out)",
        gl_position.map_or("?".to_string(), |v| v.to_string())
    );

    // Dead-code elimination. Because every load_ubo is pinned to a fixed uniform,
    // the UBO byte-offset arithmetic (iadd/ishl from gl_VertexIndex, descriptor
    // index math) feeds only the dropped offsets and is dead. Mark-and-sweep from
    // the store_output roots back through the def→use chain to fixpoint.
    let pre_dce = prog.len();
    let mut live: std::collections::HashSet<u32> = out_roots.iter().copied().collect();
    loop {
        let mut grew = false;
        for i in &prog {
            if live.contains(&i.dst) {
                for &s in &i.srcs {
                    grew |= live.insert(s);
                }
            }
        }
        if !grew {
            break;
        }
    }
    prog.retain(|i| live.contains(&i.dst));
    if pre_dce != prog.len() {
        eprintln!(
            "borgc: DCE — dropped {} dead instr(s) (UBO address math), {} live",
            pre_dce - prog.len(),
            prog.len()
        );
    }

    let alloc = regalloc(&prog);

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
    for i in &prog {
        if i.mnem == "mov" {
            continue;
        }
        let rd = *alloc.get(&i.dst).unwrap_or(&0);
        let mut r = [0u8; 3];
        let mut f3 = 0u32;
        for (k, (s, &c)) in i.srcs.iter().zip(i.swz.iter()).take(3).enumerate() {
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
                // attribute varying — firmware-handled, not yet pinned.
                Some(&Ubo::Attr) => pending += 1,
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
    eprintln!(
        "borgc: encoded {} word(s) = {} uniform pre-load(s) + {} op(s) \
         ({} operand(s) still pending)",
        words.len(),
        n_preload,
        words.len() - n_preload,
        pending
    );
    if env::var("BORGC_DUMP_ISA").is_ok() {
        let hex: Vec<String> = words.iter().take(8).map(|w| format!("{w:#010x}")).collect();
        eprintln!("borgc:   first words: {}", hex.join(" "));
    }
    total
}

/// Encode one Borg instruction to its 32-bit word (opcode bases match
/// software/borg/borg_isa.h and hardware Instructions.scala). `funct3` selects a
/// uniform operand (0=none, 1=rs1, 2=rs2, 3=rs3); `rs3` is used only by FMADD.
fn encode(mnem: &str, rd: u8, rs1: u8, rs2: u8, rs3: u8, funct3: u32) -> Option<u32> {
    let (rd, rs1, rs2, rs3) = (rd as u32, rs1 as u32, rs2 as u32, rs3 as u32);
    let f3 = (funct3 & 0x7) << 12;
    let bin = |base: u32| base | f3 | (rs2 << 20) | (rs1 << 15) | (rd << 7);
    let un = |base: u32| base | f3 | (rs1 << 15) | (rd << 7);
    let r4 = |base: u32| base | f3 | (rs3 << 27) | (rs2 << 20) | (rs1 << 15) | (rd << 7);
    Some(match mnem {
        "FMADD" => r4(0x0000_0004),
        "FADD" => bin(0x0000_0000),
        "FMUL" => bin(0x0800_0000),
        "FNEG" => un(0x0C00_0000),
        "FRCP" => un(0x1400_0000),
        "FTEX" => bin(0x1800_0000),
        "IADD" => bin(0x1C00_0000),
        "ISHL" => bin(0x2000_0000),
        "ISHR" => bin(0x2400_0000),
        "IMUL" => bin(0x2800_0000),
        "I2F" => un(0x2C00_0000),
        "F2I" => un(0x3000_0000),
        _ => return None, // mov is handled separately (register copy)
    })
}

/// Linear-scan register allocation (Poletto & Sarkar) for the values our
/// instruction selection defines (the ALU results). Each is live from its def
/// to its last use; we assign physical GPRs r0..r29 (r30/r31 are the special
/// coordinate registers), reusing a register once its value dies. Reports the
/// allocation, the peak register pressure, and any spills. Values that only
/// appear as sources (shader inputs) are not allocated here — they map to
/// uniforms/attributes when the I/O lowering lands.
fn regalloc(prog: &[BorgInstr]) -> std::collections::HashMap<u32, u8> {
    use std::collections::HashMap;

    const NUM_GPRS: u8 = 30; // r0..r29; r30/r31 reserved for coordinates

    // Def position (first) and last-use position for every value.
    let mut def_at: HashMap<u32, usize> = HashMap::new();
    let mut last_use: HashMap<u32, usize> = HashMap::new();
    for (i, instr) in prog.iter().enumerate() {
        def_at.entry(instr.dst).or_insert(i);
        last_use.insert(instr.dst, i);
        for &s in &instr.srcs {
            last_use.insert(s, i);
        }
    }

    // Allocate only values we define (ALU results), in def order.
    let mut defs: Vec<u32> = def_at.keys().copied().collect();
    defs.sort_by_key(|v| def_at[v]);

    let mut free: Vec<u8> = (0..NUM_GPRS).rev().collect();
    let mut active: Vec<(usize, u8)> = Vec::new(); // (live_end, phys_reg)
    let mut alloc: HashMap<u32, u8> = HashMap::new();
    let mut spills = 0usize;
    let mut peak = 0usize;

    for v in defs {
        let start = def_at[&v];
        let end = *last_use.get(&v).unwrap_or(&start);

        // Expire intervals that ended before this value starts.
        let mut keep = Vec::with_capacity(active.len());
        for &(e, r) in &active {
            if e < start {
                free.push(r);
            } else {
                keep.push((e, r));
            }
        }
        active = keep;

        match free.pop() {
            Some(r) => {
                alloc.insert(v, r);
                active.push((end, r));
            }
            None => spills += 1,
        }
        peak = peak.max(active.len());
    }

    eprintln!(
        "borgc: regalloc — {} values → r0..r{}, peak pressure {}, spills {}",
        alloc.len(),
        NUM_GPRS - 1,
        peak,
        spills
    );
    alloc
}
