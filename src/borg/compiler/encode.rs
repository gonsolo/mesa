// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

pub(crate) fn emit_blob(words: &[u32], outputs: &[u8], consts: &[(u8, u32)],
                        draw: Option<(u8, &[(u8, u32)])>) -> Vec<u8> {
    let mut b = Vec::new();
    b.push(words.len() as u8); // num_instrs, low byte; the high bits ride in the extensions byte
    b.push(0); // num_uniforms (inline funct3 reads, no host interface)
    b.push(0); // num_attributes
    b.push(outputs.len() as u8); // num_outputs
    b.push(consts.len() as u8); // num_consts
    b.push((if draw.is_some() { 1 } else { 0 }) | ((words.len() >> 8) as u8) << 1); // extensions: bit 0 = draw, bits 7:1 = num_instrs >> 8
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
    if let Some((num_varyings, window)) = draw {
        b.push(num_varyings);
        b.push(window.len() as u8); // num_window
        for &(u, _) in window {
            b.push(u); // window_regs[]
        }
        for &(_, val) in window {
            b.extend_from_slice(&val.to_le_bytes()); // window_vals[] (LE u32)
        }
    }
    b
}

pub(crate) fn encode(mnem: &str, rd: u8, rs1: u8, rs2: u8, rs3: u8, funct3: u32) -> Option<u32> {
    let (rd, rs1, rs2, rs3) = (rd as u32, rs1 as u32, rs2 as u32, rs3 as u32);
    let f3 = (funct3 & 0x7) << 12;
    let bin = |base: u32| base | f3 | (rs2 << 20) | (rs1 << 15) | (rd << 7);
    let un = |base: u32| base | f3 | (rs1 << 15) | (rd << 7);
    let r4 = |base: u32| base | f3 | (rs3 << 27) | (rs2 << 20) | (rs1 << 15) | (rd << 7);
    Some(match mnem {
        "FMADD" => r4(0x0000_0004),
        "TEX" => r4(0x0400_0004),
        // TEXA w, lod, dref: this lane's extra arguments for the TEX after it (funct2 = 3).
        "TEXA" => r4(0x0600_0004),
        "FADD" => bin(0x0000_0000),
        "FMUL" => bin(0x0800_0000),
        "FNEG" => un(0x0C00_0000),
        "FMOV" => un(0xA400_0000),
        "FRCP" => un(0x1400_0000),
        "IADD" => bin(0x1C00_0000),
        "ISHL" => bin(0x2000_0000),
        "ISHR" => bin(0x2400_0000),
        "IMUL" => bin(0x2800_0000),
        "ISUB" => bin(0x6000_0000),
        "IAND" => bin(0x6400_0000),
        "IOR" => bin(0x6800_0000),
        "IXOR" => bin(0x6C00_0000),
        "ISLT" => bin(0x7000_0000),
        "ISEQ" => bin(0x7400_0000),
        "ISRL" => bin(0x9400_0000),
        "ISLTU" => bin(0x9800_0000),
        "I2F" => un(0x2C00_0000),
        "F2I" => un(0x3000_0000),
        "F2H" => un(0xA800_0000),
        "H2F" => un(0xAC00_0000),
        "FRSQ" => un(0x3400_0000),
        "FSRGB" => un(0x3800_0000),
        "DDX" => un(0x3C00_0000),
        "DDY" => un(0x4000_0000),
        "FSTEP" => un(0x1000_0000),
        "ATTIDX" => un(0x9000_0000),
        "TLD" => un(0x9C00_0000),
        "TLD1" => un(0x9C00_0000) | (1 << 12),   // word 1 of a RAW64 texel
        "LOAD" => un(0x4400_0000),
        "STORE" => bin(0x4800_0000) & !(0x1F << 7), // rd field is unused
        "EXPUSH" => un(0x5400_0000) & !(0x1F << 7),
        "EXELSE" => 0x5800_0000,
        "EXPOP" => 0x5C00_0000,
        "BARRIER" => 0x7800_0000,
        "EXANY" => un(0x7C00_0000),
        _ => return None, // mov is handled separately (register copy)
    })
}

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

pub(crate) fn encode_fattr(rd: u8, index: u16) -> Option<u32> {
    if index >= 1024 {
        return None; // 10 bits
    }
    let t = index as u32;
    Some(0x8800_0000u32 | (((t >> 5) & 0x1F) << 20) | ((t & 0x1F) << 15) | ((rd as u32) << 7))
}

/// JMP: absolute, 18 bits packed as funct3:rs2:rs1:rd.
pub(crate) fn encode_jmp(target: u32) -> u32 {
    let t = target;
    0xA000_0000 | ((t >> 10) & 0x1F) << 20 | ((t >> 5) & 0x1F) << 15 | ((t >> 15) & 7) << 12 | (t & 0x1F) << 7
}

pub(crate) fn encode_branch(mnem: &str, rs1: u8, target: u16) -> Option<u32> {
    let base = match mnem {
        "BRZ" => 0x4C00_0000u32,
        "BRNZ" => 0x5000_0000u32,
        _ => return None,
    };
    if target >= 1024 {
        return None; // 10 bits: the low bits of a target in the branch's own page
    }
    let t = target as u32;
    Some(base | (((t >> 5) & 0x1F) << 20) | ((rs1 as u32) << 15) | ((t & 0x1F) << 7))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_constants_are_raw_little_endian_words() {
        let b = emit_blob(&[0], &[], &[(17, 1.0f32.to_bits()), (23, 9)], None);
        assert_eq!(&b[..6], &[1, 0, 0, 0, 2, 0], "header: 1 instr, 2 consts");
        assert_eq!(&b[10..12], &[17, 23], "const registers");
        assert_eq!(&b[12..16], &[0x00, 0x00, 0x80, 0x3F], "1.0f as LE binary32");
        assert_eq!(&b[16..20], &[9, 0, 0, 0], "raw integer (a word index)");
        assert_eq!(b.len(), 20, "a legacy blob ends at const_vals");
    }

    #[test]
    fn blob_counts_more_than_255_instructions() {
        let b = emit_blob(&[0; 300], &[], &[], None);
        assert_eq!(&b[..6], &[44, 0, 0, 0, 0, 2], "300 = 1 * 256 + 44");
        assert_eq!(b.len(), 6 + 300 * 4);
    }

    #[test]
    fn blob_carries_the_draw_extension() {
        let b = emit_blob(&[0], &[], &[], Some((7, &[(25, 4), (26, 160)])));
        assert_eq!(&b[..6], &[1, 0, 0, 0, 0, 1], "header: extensions bit 0");
        assert_eq!(&b[10..12], &[7, 2], "7 varyings, 2 window words");
        assert_eq!(&b[12..14], &[25, 26], "window u-indices");
        assert_eq!(&b[14..18], &[4, 0, 0, 0], "u25 = 4");
        assert_eq!(&b[18..22], &[160, 0, 0, 0], "u26 = 160");
        assert_eq!(b.len(), 22);
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
        assert_eq!(encode("BARRIER", 0, 0, 0, 0, 0).unwrap(), 0x3C << 25);
        assert_eq!(encode("EXANY", 3, 0, 0, 0, 0).unwrap(), 0x3E << 25 | 3 << 7);
        // Instructions.JMP(0x2_5ABC): rs2 = 22, rs1 = 21, rd = 28, funct3 = 4.
        assert_eq!(encode_jmp(0x2_5ABC), 0x50 << 25 | 22 << 20 | 21 << 15 | 4 << 12 | 28 << 7);
        assert_eq!(encode_sout(0, 0, 0).unwrap() & 0xFE00_0000, 0x42 << 25);
        assert_eq!(encode_fattr(0, 0).unwrap() & 0xFE00_0000, 0x44 << 25);
    }

    #[test]
    fn sout_and_fattr_split_the_index_over_two_5_bit_fields() {
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
        let w = encode("TEX", 20, 15, 7, 9, 0).unwrap();
        assert_eq!(w & 0x7F, 0x04, "R4-type opcode");
        assert_eq!((w >> 25) & 0x3, 2, "funct2 = TEX");
        assert_eq!((w >> 27, (w >> 20) & 0x1F, (w >> 15) & 0x1F, (w >> 7) & 0x1F),
                   (9, 7, 15, 20), "rs3 ctl, rs2 v, rs1 u, rd");
        assert!(encode("FTEX", 20, 15, 7, 0, 0).is_none(), "FTEX is retired");
    }

    #[test]
    fn half_conversions_are_funct7_0x54_and_0x56() {
        for (m, f7) in [("F2H", 0x54u32), ("H2F", 0x56)] {
            let w = encode(m, 7, 5, 0, 0, 0).unwrap();
            assert_eq!((w >> 25, (w >> 15) & 0x1F, (w >> 7) & 0x1F), (f7, 5, 7), "{m}: funct7, rs1, rd");
        }
    }

    #[test]
    fn load_places_operands_where_the_hardware_reads_them() {
        let w = encode("LOAD", 7, 5, 0, 0, 0).unwrap();
        assert_eq!((w >> 15) & 0x1F, 5, "rs1 = address index");
        assert_eq!((w >> 7) & 0x1F, 7, "rd = destination");
    }

    #[test]
    fn store_and_mask_ops_leave_no_destination() {
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
