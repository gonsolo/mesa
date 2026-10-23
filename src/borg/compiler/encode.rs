// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::opcodes::{Fields, Op};
use crate::Mnem;

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

pub(crate) fn encode(mnem: Mnem, rd: u8, rs1: u8, rs2: u8, rs3: u8, funct3: u32) -> Option<u32> {
    let (op, f3) = match mnem {
        Mnem::Op(op) => (op, funct3),
        Mnem::Tld1 => (Op::Tld, funct3 | 1),
        _ => return None,
    };
    let (rd, rs1, rs2, rs3) = (rd as u32, rs1 as u32, rs2 as u32, rs3 as u32);
    // ISA v2: register-only integer ops in RISC-V form, unless one names r0, which is x0 there.
    let rv = match op {
        Op::Iadd => Some(Op::Add), Op::Isub => Some(Op::Sub), Op::Ishl => Some(Op::Sll),
        Op::Islt => Some(Op::Slt), Op::Isltu => Some(Op::Sltu), Op::Ixor => Some(Op::Xor),
        Op::Isrl => Some(Op::Srl), Op::Ishr => Some(Op::Sra), Op::Ior => Some(Op::Or),
        Op::Iand => Some(Op::And), _ => None,
    };
    let op = match rv {
        Some(r) if f3 == 0 && rd != 0 && rs1 != 0 && rs2 != 0 => r,
        _ => op,
    };
    Some(op.encode(&Fields { rd, rs1, rs2, rs3, f3, ..Default::default() }))
}

/// The fragment output CSRs (docs/B8_isa_v2.md); out_c1..3 follow out_c0.
pub(crate) const CSR_OUT_C0: u32 = 0x800;

/// CSRRW x0, csr, rs1.
pub(crate) fn encode_csrw(rs1: u8, csr: u32) -> Option<u32> {
    Some(Op::Csrrw.encode(&Fields { rs1: rs1 as u32, csr, ..Default::default() }))
}

pub(crate) fn encode_sout(rs2: u8, index: u16, funct3: u32) -> Option<u32> {
    if index >= 1024 {
        return None; // 10 bits
    }
    Some(Op::Sout.encode(&Fields { rs2: rs2 as u32, index: index as u32, f3: funct3, ..Default::default() }))
}

pub(crate) fn encode_fattr(rd: u8, index: u16) -> Option<u32> {
    if index >= 1024 {
        return None; // 10 bits
    }
    Some(Op::Fattr.encode(&Fields { rd: rd as u32, aindex: index as u32, ..Default::default() }))
}

/// JMP by `offset` words from the jump (18 bits, signed).
pub(crate) fn encode_jmp(offset: i32) -> Option<u32> {
    if !(-(1 << 17)..1 << 17).contains(&offset) {
        return None;
    }
    Some(Op::Jmp.encode(&Fields { joffset: offset as u32, ..Default::default() }))
}

/// BRZ or BRNZ by `offset` words from the branch (10 bits, signed).
pub(crate) fn encode_branch(op: Op, rs1: u8, offset: i32) -> Option<u32> {
    if !(-512..512).contains(&offset) {
        return None;
    }
    Some(op.encode(&Fields { rs1: rs1 as u32, offset: offset as u32, ..Default::default() }))
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
        // funct7 << 25, from hardware/borg/isa/opcodes.
        assert_eq!(encode(Op::Load.into(), 3, 5, 0, 0, 0).unwrap() & 0xFE00_0000, 0x22 << 25);
        assert_eq!(encode(Op::Store.into(), 0, 5, 7, 0, 0).unwrap() & 0xFE00_0000, 0x24 << 25);
        assert_eq!(encode_branch(Op::Brz, 5, 0).unwrap() & 0xFE00_0000, 0x26 << 25);
        assert_eq!(encode_branch(Op::Brnz, 5, 0).unwrap() & 0xFE00_0000, 0x28 << 25);
        assert_eq!(encode(Op::Expush.into(), 0, 5, 0, 0, 0).unwrap() & 0xFE00_0000, 0x2A << 25);
        assert_eq!(encode(Op::Exelse.into(), 0, 0, 0, 0, 0).unwrap() & 0xFE00_0000, 0x2C << 25);
        assert_eq!(encode(Op::Expop.into(), 0, 0, 0, 0, 0).unwrap() & 0xFE00_0000, 0x2E << 25);
        assert_eq!(encode(Op::Barrier.into(), 0, 0, 0, 0, 0).unwrap(), 0x3C << 25);
        assert_eq!(encode(Op::Exany.into(), 3, 0, 0, 0, 0).unwrap(), 0x3E << 25 | 3 << 7);
        // JMP by 0x1_5ABC: funct3 = 2, rs2 = 22, rs1 = 21, rd = 28; by -1: every offset bit set.
        assert_eq!(encode_jmp(0x1_5ABC), Some(0x50 << 25 | 22 << 20 | 21 << 15 | 2 << 12 | 28 << 7));
        assert_eq!(encode_jmp(-1), Some(0x50 << 25 | 31 << 20 | 31 << 15 | 7 << 12 | 31 << 7));
        assert_eq!(encode_jmp(1 << 17), None);
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
        let w = encode(Op::Tex.into(), 20, 15, 7, 9, 0).unwrap();
        assert_eq!(w & 0x7F, 0x04, "R4-type opcode");
        assert_eq!((w >> 25) & 0x3, 2, "funct2 = TEX");
        assert_eq!((w >> 27, (w >> 20) & 0x1F, (w >> 15) & 0x1F, (w >> 7) & 0x1F),
                   (9, 7, 15, 20), "rs3 ctl, rs2 v, rs1 u, rd");
    }

    #[test]
    fn half_conversions_are_funct7_0x54_and_0x56() {
        for (m, f7) in [(Op::F2h, 0x54u32), (Op::H2f, 0x56)] {
            let w = encode(m.into(), 7, 5, 0, 0, 0).unwrap();
            assert_eq!((w >> 25, (w >> 15) & 0x1F, (w >> 7) & 0x1F), (f7, 5, 7), "{m:?}: funct7, rs1, rd");
        }
    }

    #[test]
    fn load_places_operands_where_the_hardware_reads_them() {
        let w = encode(Op::Load.into(), 7, 5, 0, 0, 0).unwrap();
        assert_eq!((w >> 15) & 0x1F, 5, "rs1 = address index");
        assert_eq!((w >> 7) & 0x1F, 7, "rd = destination");
    }

    #[test]
    fn store_and_mask_ops_leave_no_destination() {
        assert_eq!((encode(Op::Store.into(), 31, 5, 7, 0, 0).unwrap() >> 7) & 0x1F, 0);
        assert_eq!((encode(Op::Expush.into(), 31, 5, 0, 0, 0).unwrap() >> 7) & 0x1F, 0);
        // STORE's data operand is rs2.
        assert_eq!((encode(Op::Store.into(), 0, 5, 7, 0, 0).unwrap() >> 20) & 0x1F, 7);
    }

    #[test]
    fn branch_offset_splits_across_rs2_and_rd() {
        // Offset 33 = 0b1_00001 -> high bits 1 in rs2, low bits 1 in rd.
        let w = encode_branch(Op::Brz, 2, 33).unwrap();
        assert_eq!(((w >> 20) & 0x1F, (w >> 7) & 0x1F, (w >> 15) & 0x1F), (1, 1, 2), "offset high, low, rs1");
        // Every offset round-trips as a signed 10-bit value.
        for t in -512..512i32 {
            let w = encode_branch(Op::Brnz, 0, t).unwrap();
            let back = ((((w >> 20) & 0x1F) << 5 | ((w >> 7) & 0x1F)) as i32) << 22 >> 22;
            assert_eq!(back, t, "offset {t} did not round-trip");
        }
        assert!(encode_branch(Op::Brz, 0, 512).is_none() && encode_branch(Op::Brz, 0, -513).is_none());
    }
}
