// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// Instruction encoding and blob serialization: turns a fully register-allocated
// Vec<BorgInstr> word list into the .borg blob bytes the firmware loads.

/// Serialize a Borg shader to the .borg blob format parsed by spirb_parse()
/// (software/borg/borg_spirb.c): a 6-byte header (num_instrs, num_uniforms,
/// num_attributes, num_outputs, num_consts, reserved), then num_instrs LE u32
/// instruction words, then the uint8 uniform/attribute/output/const register
/// lists, then num_consts LE u32 constant values. We emit instructions, the
/// output-register list, and the const-register list. The consts (e.g.
/// cube.frag's lightDir in r17-19) are written once to the GPRs by the firmware
/// (spirb_parse → BORG_GPU->gpr[const_regs[i]] = const_vals[i]) and persist
/// across the autonomous render. Uniforms are read inline via funct3, so there
/// is no host uniform/attribute interface list.
///
/// const_vals are LE u32: one datapath float (IEEE binary32 bits) or a raw
/// integer such as a push-constant word index, written to the GPR unchanged.
pub(crate) fn emit_blob(words: &[u32], outputs: &[u8], consts: &[(u8, u32)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.push(words.len() as u8); // num_instrs
    b.push(0); // num_uniforms (inline funct3 reads, no host interface)
    b.push(0); // num_attributes
    b.push(outputs.len() as u8); // num_outputs
    b.push(consts.len() as u8); // num_consts
    b.push(0); // reserved
    for &w in words {
        b.extend_from_slice(&w.to_le_bytes());
    }
    b.extend_from_slice(outputs); // output_regs
    for &(reg, _) in consts {
        b.push(reg); // const_regs[]
    }
    for &(_, val) in consts {
        b.extend_from_slice(&val.to_le_bytes()); // const_vals[] (LE u32)
    }
    b
}

/// Encode one Borg instruction to its 32-bit word (opcode bases match
/// software/borg/borg_isa.h and hardware Instructions.scala). `funct3` selects a
/// uniform operand (0=none, 1=rs1, 2=rs2, 3=rs3); `rs3` is used by the R4-type
/// ops, FMADD and TEX.
pub(crate) fn encode(mnem: &str, rd: u8, rs1: u8, rs2: u8, rs3: u8, funct3: u32) -> Option<u32> {
    let (rd, rs1, rs2, rs3) = (rd as u32, rs1 as u32, rs2 as u32, rs3 as u32);
    let f3 = (funct3 & 0x7) << 12;
    let bin = |base: u32| base | f3 | (rs2 << 20) | (rs1 << 15) | (rd << 7);
    let un = |base: u32| base | f3 | (rs1 << 15) | (rd << 7);
    let r4 = |base: u32| base | f3 | (rs3 << 27) | (rs2 << 20) | (rs1 << 15) | (rd << 7);
    Some(match mnem {
        "FMADD" => r4(0x0000_0004),
        // TEX rd, u, v, ctl: R4-type funct2 = 2 (docs/B2_texture_unit.md).
        // funct2 = 1 was FTEX, retired with the legacy texture unit.
        "TEX" => r4(0x0400_0004),
        "FADD" => bin(0x0000_0000),
        "FMUL" => bin(0x0800_0000),
        "FNEG" => un(0x0C00_0000),
        "FRCP" => un(0x1400_0000),
        "IADD" => bin(0x1C00_0000),
        "ISHL" => bin(0x2000_0000),
        "ISHR" => bin(0x2400_0000),
        "IMUL" => bin(0x2800_0000),
        "I2F" => un(0x2C00_0000),
        "F2I" => un(0x3000_0000),
        "FRSQ" => un(0x3400_0000),
        "FSRGB" => un(0x3800_0000),
        "DDX" => un(0x3C00_0000),
        "DDY" => un(0x4000_0000),
        "FSTEP" => un(0x1000_0000),
        // --- Extended ISA (hardware Instructions.scala, same funct7 << 25) ---
        // LOAD/STORE address LS_BASE + (rs1 << 2): the operand is a word
        // INDEX, not a byte address (words are the natural unit, and the
        // scheme predates FP32, when a register held only 16 bits). STORE has no destination.
        "LOAD" => un(0x4400_0000),
        "STORE" => bin(0x4800_0000) & !(0x1F << 7), // rd field is unused
        // Execution mask. None of these has a destination; EXPUSH reads its
        // condition from rs1.
        "EXPUSH" => un(0x5400_0000) & !(0x1F << 7),
        "EXELSE" => 0x5800_0000,
        "EXPOP" => 0x5C00_0000,
        _ => return None, // mov is handled separately (register copy)
    })
}

/// Encode SOUT (docs/B1_geometry_front_end.md): store `rs2` as the current
/// invocation's corner's output component `index`. Separate from [`encode`]
/// for the same reason as [`encode_branch`] -- `index` is packed into the
/// otherwise-unused rs1/rd fields as `(index >> 5)`/`(index & 31)`, not a
/// register operand.
pub(crate) fn encode_sout(rs2: u8, index: u16, funct3: u32) -> Option<u32> {
    if index >= 1024 {
        return None; // 10 bits
    }
    let t = index as u32;
    Some(0x8400_0000u32
        | ((funct3 & 0x7) << 12)
        | ((rs2 as u32) << 20)
        | (((t >> 5) & 0x1F) << 15)
        | ((t & 0x1F) << 7))
}

/// Encode FATTR (docs/B1_geometry_front_end.md): load component `index`'s
/// three per-vertex values into `rd..rd+2`. `index` is packed into rs2/rs1
/// the same way [`encode_sout`]'s is.
pub(crate) fn encode_fattr(rd: u8, index: u16) -> Option<u32> {
    if index >= 1024 {
        return None; // 10 bits
    }
    let t = index as u32;
    Some(0x8800_0000u32 | (((t >> 5) & 0x1F) << 20) | ((t & 0x1F) << 15) | ((rd as u32) << 7))
}

/// Encode a conditional branch. Separate from [`encode`] because the target is
/// packed into the otherwise-unused rs2 and rd fields as `(target >> 5)` and
/// `(target & 31)` -- it is not a register operand, and passing it through the
/// register-shaped signature above would invite treating it as one.
pub(crate) fn encode_branch(mnem: &str, rs1: u8, target: u16) -> Option<u32> {
    let base = match mnem {
        "BRZ" => 0x4C00_0000u32,
        "BRNZ" => 0x5000_0000u32,
        _ => return None,
    };
    if target >= 1024 {
        return None; // 10 bits; far past any IMEM Borg builds
    }
    let t = target as u32;
    Some(base | (((t >> 5) & 0x1F) << 20) | ((rs1 as u32) << 15) | ((t & 0x1F) << 7))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encodings are a mirror of the hardware's Instructions.scala. A
    /// divergence here is silent: the shader would execute a different opcode
    /// rather than fail to load, so the funct7 bases are pinned explicitly.
    #[test]
    fn blob_constants_are_raw_little_endian_words() {
        // A constant reaches its GPR unchanged, so the blob must carry the
        // FP32 bits themselves: 1.0f = 0x3F800000, not an FP16 0x3C00.
        let b = emit_blob(&[0], &[], &[(17, 1.0f32.to_bits()), (23, 9)]);
        assert_eq!(&b[..6], &[1, 0, 0, 0, 2, 0], "header: 1 instr, 2 consts");
        assert_eq!(&b[10..12], &[17, 23], "const registers");
        assert_eq!(&b[12..16], &[0x00, 0x00, 0x80, 0x3F], "1.0f as LE binary32");
        assert_eq!(&b[16..20], &[9, 0, 0, 0], "raw integer (a word index)");
    }

    #[test]
    fn extended_isa_bases_match_hardware() {
        // funct7 << 25, from hardware/borg/src/Instructions.scala.
        assert_eq!(encode("LOAD", 3, 5, 0, 0, 0).unwrap() & 0xFE00_0000, 0x22 << 25);
        assert_eq!(encode("STORE", 0, 5, 7, 0, 0).unwrap() & 0xFE00_0000, 0x24 << 25);
        assert_eq!(encode_branch("BRZ", 5, 0).unwrap() & 0xFE00_0000, 0x26 << 25);
        assert_eq!(encode_branch("BRNZ", 5, 0).unwrap() & 0xFE00_0000, 0x28 << 25);
        assert_eq!(encode("EXPUSH", 0, 5, 0, 0, 0).unwrap() & 0xFE00_0000, 0x2A << 25);
        assert_eq!(encode("EXELSE", 0, 0, 0, 0, 0).unwrap() & 0xFE00_0000, 0x2C << 25);
        assert_eq!(encode("EXPOP", 0, 0, 0, 0, 0).unwrap() & 0xFE00_0000, 0x2E << 25);
        assert_eq!(encode_sout(0, 0, 0).unwrap() & 0xFE00_0000, 0x42 << 25);
        assert_eq!(encode_fattr(0, 0).unwrap() & 0xFE00_0000, 0x44 << 25);
    }

    #[test]
    fn sout_and_fattr_split_the_index_over_two_5_bit_fields() {
        // index = 515 = 0b10000_00011: high 5 bits 0b10000 = 16 in
        // rs1(SOUT)/rs2(FATTR), low 5 bits 0b00011 = 3 in rd(SOUT)/rs1(FATTR)
        // -- the same split as a BRZ/BRNZ branch target (see
        // branch_target_splits_across_rs2_and_rd).
        let s = encode_sout(9, 515, 2).unwrap();
        assert_eq!((s >> 20) & 0x1F, 9, "rs2 = the stored value");
        assert_eq!((s >> 15) & 0x1F, 16, "index high 5 bits");
        assert_eq!((s >> 7) & 0x1F, 3, "index low 5 bits");
        assert_eq!((s >> 12) & 0x7, 2, "funct3");

        let f = encode_fattr(20, 515).unwrap();
        assert_eq!((f >> 20) & 0x1F, 16, "index high 5 bits");
        assert_eq!((f >> 15) & 0x1F, 3, "index low 5 bits");
        assert_eq!((f >> 7) & 0x1F, 20, "rd");

        assert!(encode_sout(0, 1024, 0).is_none(), "index out of range must not encode");
        assert!(encode_fattr(0, 1024).is_none(), "index out of range must not encode");
    }

    #[test]
    fn tex_is_r4_type_funct2_2_with_the_control_word_in_rs3() {
        // Instructions.encodeR4Type(rs3, FUNCT2_TEX = 2, rs2, rs1, rd): the
        // opcode's FMA bit, funct2 at 26:25, rs3 at 31:27.
        let w = encode("TEX", 20, 15, 7, 9, 0).unwrap();
        assert_eq!(w & 0x7F, 0x04, "R4-type opcode");
        assert_eq!((w >> 25) & 0x3, 2, "funct2 = TEX");
        assert_eq!((w >> 27, (w >> 20) & 0x1F, (w >> 15) & 0x1F, (w >> 7) & 0x1F),
                   (9, 7, 15, 20), "rs3 ctl, rs2 v, rs1 u, rd");
        assert!(encode("FTEX", 20, 15, 7, 0, 0).is_none(), "FTEX is retired");
    }

    #[test]
    fn load_places_operands_where_the_hardware_reads_them() {
        let w = encode("LOAD", 7, 5, 0, 0, 0).unwrap();
        assert_eq!((w >> 15) & 0x1F, 5, "rs1 = address index");
        assert_eq!((w >> 7) & 0x1F, 7, "rd = destination");
    }

    #[test]
    fn store_and_mask_ops_leave_no_destination() {
        // rd carries no meaning for these; a stray value there would name a
        // register the hardware does not write but a reader would believe.
        assert_eq!((encode("STORE", 31, 5, 7, 0, 0).unwrap() >> 7) & 0x1F, 0);
        assert_eq!((encode("EXPUSH", 31, 5, 0, 0, 0).unwrap() >> 7) & 0x1F, 0);
        // STORE's data operand is rs2.
        assert_eq!((encode("STORE", 0, 5, 7, 0, 0).unwrap() >> 20) & 0x1F, 7);
    }

    #[test]
    fn branch_target_splits_across_rs2_and_rd() {
        // Target 33 = 0b1_00001 -> high bits 1 in rs2, low bits 1 in rd.
        let w = encode_branch("BRZ", 2, 33).unwrap();
        assert_eq!((w >> 20) & 0x1F, 1, "target high 5 bits");
        assert_eq!((w >> 7) & 0x1F, 1, "target low 5 bits");
        assert_eq!((w >> 15) & 0x1F, 2, "rs1 = condition");
        // Round-trip every representable target.
        for t in 0..1024u16 {
            let w = encode_branch("BRNZ", 0, t).unwrap();
            let back = (((w >> 20) & 0x1F) << 5) | ((w >> 7) & 0x1F);
            assert_eq!(back, t as u32, "target {t} did not round-trip");
        }
        assert!(encode_branch("BRZ", 0, 1024).is_none(), "out of range must not encode");
    }
}
