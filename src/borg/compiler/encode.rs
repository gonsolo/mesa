// Copyright © 2026 Borg GPU project
// SPDX-License-Identifier: MIT
//
// Instruction encoding and blob serialization: turns a fully register-allocated
// Vec<BorgInstr> word list into the .borg blob bytes the firmware loads.

/// Convert f32 bits to fp16 bits (round toward zero; adequate for the small
/// normal constants we pin, e.g. lightDir).
pub(crate) fn f32_to_fp16(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x7F_FFFF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

/// Serialize a Borg shader to the .borg blob format parsed by spirb_parse()
/// (software/borg/borg_spirb.c): a 6-byte header (num_instrs, num_uniforms,
/// num_attributes, num_outputs, num_consts, reserved), then num_instrs LE u32
/// instruction words, then the uint8 uniform/attribute/output/const register
/// lists, then num_consts LE u16 constant values. We emit instructions, the
/// output-register list, and the const-register list. The consts (e.g.
/// cube.frag's lightDir in r23-25) are written once to the GPRs by the firmware
/// (spirb_parse → BORG_GPU->gpr[const_regs[i]] = const_vals[i]) and persist
/// across the autonomous render. Uniforms are read inline via funct3, so there
/// is no host uniform/attribute interface list.
pub(crate) fn emit_blob(words: &[u32], outputs: &[u8], consts: &[(u8, u16)]) -> Vec<u8> {
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
        b.extend_from_slice(&val.to_le_bytes()); // const_vals[] (LE u16)
    }
    b
}

/// Encode one Borg instruction to its 32-bit word (opcode bases match
/// software/borg/borg_isa.h and hardware Instructions.scala). `funct3` selects a
/// uniform operand (0=none, 1=rs1, 2=rs2, 3=rs3); `rs3` is used only by FMADD.
pub(crate) fn encode(mnem: &str, rd: u8, rs1: u8, rs2: u8, rs3: u8, funct3: u32) -> Option<u32> {
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
        "FRSQ" => un(0x3400_0000),
        "FSRGB" => un(0x3800_0000),
        "DDX" => un(0x3C00_0000),
        "DDY" => un(0x4000_0000),
        "FSTEP" => un(0x1000_0000),
        _ => return None, // mov is handled separately (register copy)
    })
}
