// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! LiteDRAM's DDR3 calibration from the host, for a Nexys Video whose memory controller is
//! LiteDRAM (Borg's fpga/nexysvideo/LiteDram.yml): LiteX's `sdram_init` for the A7DDRPHY (read
//! leveling, no write leveling) through the board's register window at offsets 0x200-0x3FF. The
//! register indices follow that core's csr.csv.

use crate::dev::Dev;
use crate::list::Gpu;

const ID: u32 = 0x4c44_5231; // "LDR1" at window word 31
const MODULES: u32 = 2;
const DELAYS: u32 = 32;
const BITSLIPS: u32 = 8;
const PHASES: u32 = 4;
const RDPHASE: u32 = 3;
const WRPHASE: u32 = 3;

const fn ctrl(i: u32) -> u32 {
    0x200 + 4 * i
}
const fn phy(i: u32) -> u32 {
    0x200 + 4 * (32 + i)
}
const fn dfii(i: u32) -> u32 {
    0x200 + 4 * (64 + i)
}

const INIT_DONE: u32 = ctrl(0);
const INIT_ERROR: u32 = ctrl(1);
const WINDOW_ID: u32 = ctrl(31);
const PHY_RST: u32 = phy(0);
const DLY_SEL: u32 = phy(1);
const RDLY_RST: u32 = phy(5);
const RDLY_INC: u32 = phy(6);
const BITSLIP_RST: u32 = phy(7);
const BITSLIP_INC: u32 = phy(8);
const RDPHASE_REG: u32 = phy(11);
const WRPHASE_REG: u32 = phy(12);
const CONTROL: u32 = dfii(0);
const fn pi_command(p: u32) -> u32 {
    dfii(1 + 6 * p)
}
const fn pi_issue(p: u32) -> u32 {
    dfii(2 + 6 * p)
}
const fn pi_address(p: u32) -> u32 {
    dfii(3 + 6 * p)
}
const fn pi_baddress(p: u32) -> u32 {
    dfii(4 + 6 * p)
}
const fn pi_wrdata(p: u32) -> u32 {
    dfii(5 + 6 * p)
}
const fn pi_rddata(p: u32) -> u32 {
    dfii(6 + 6 * p)
}

const CTL_SEL: u32 = 0x01;
const CTL_CKE: u32 = 0x02;
const CTL_ODT: u32 = 0x04;
const CTL_RESET_N: u32 = 0x08;
const CMD_CS: u32 = 0x01;
const CMD_WE: u32 = 0x02;
const CMD_CAS: u32 = 0x04;
const CMD_RAS: u32 = 0x08;
const CMD_WRDATA: u32 = 0x10;
const CMD_RDDATA: u32 = 0x20;

/// Waits `ms` after the writes so far have reached the board.
fn settle(d: &mut Dev, ms: u64) {
    d.read(WINDOW_ID);
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

fn command(d: &mut Dev, p: u32, addr: u32, bank: u32, cmd: u32) {
    d.reg(pi_address(p), addr);
    d.reg(pi_baddress(p), bank);
    d.reg(pi_command(p), cmd);
    d.reg(pi_issue(p), 1);
}

/// One action on a byte group's PHY delay or bitslip.
fn action(d: &mut Dev, module: u32, reg: u32) {
    d.reg(DLY_SEL, 1 << module);
    d.reg(reg, 1);
    d.reg(DLY_SEL, 0);
}

/// JEDEC DDR3 power-up: reset, CKE, MR2 (CWL 5), MR3, MR1 (RZQ/7 drive, RZQ/6 nominal ODT), MR0 (CL 6,
/// BL 8, DLL reset), ZQ calibration.
fn init_sequence(d: &mut Dev) {
    d.reg(CONTROL, CTL_ODT | CTL_RESET_N);
    settle(d, 1);
    d.reg(CONTROL, CTL_CKE | CTL_ODT | CTL_RESET_N);
    settle(d, 1);
    let mrs = CMD_RAS | CMD_CAS | CMD_WE | CMD_CS;
    command(d, 0, 0x0, 2, mrs);
    command(d, 0, 0x0, 3, mrs);
    command(d, 0, 0x46, 1, mrs);
    command(d, 0, 0x920, 0, mrs);
    settle(d, 1);
    command(d, 0, 0x400, 0, CMD_WE | CMD_CS);
    settle(d, 1);
}

/// Bit errors of byte group `module` on one burst of row 0, bank 0, written and read back through
/// the DFII: bits 8m..8m+7 of each phase's rising and falling beat.
fn errors(d: &mut Dev, module: u32, seed: u32) -> u32 {
    let mut x = seed;
    let pattern: Vec<u32> = (0..PHASES)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        })
        .collect();
    command(d, 0, 0, 0, CMD_RAS | CMD_CS);
    for (p, v) in pattern.iter().enumerate() {
        d.reg(pi_wrdata(p as u32), *v);
    }
    command(d, WRPHASE, 0, 0, CMD_CAS | CMD_WE | CMD_CS | CMD_WRDATA);
    command(d, RDPHASE, 0, 0, CMD_CAS | CMD_CS | CMD_RDDATA);
    command(d, 0, 0, 0, CMD_RAS | CMD_WE | CMD_CS);
    let mask = 0xff << (8 * module) | 0xff << (16 + 8 * module);
    (0..PHASES).map(|p| ((d.read(pi_rddata(p)) ^ pattern[p as usize]) & mask).count_ones()).sum()
}

fn works(d: &mut Dev, module: u32) -> bool {
    [0x2a2a_2a2a, 0x5470_c0de].iter().all(|&s| errors(d, module, s) == 0)
}

/// The longest run of working delays at the current bitslip: (start, length).
fn window(d: &mut Dev, module: u32) -> (u32, u32) {
    action(d, module, RDLY_RST);
    let (mut best, mut start, mut run) = ((0, 0), 0, 0);
    for delay in 0..DELAYS {
        if works(d, module) {
            if run == 0 {
                start = delay;
            }
            run += 1;
            if run > best.1 {
                best = (start, run);
            }
        } else {
            run = 0;
        }
        action(d, module, RDLY_INC);
    }
    best
}

/// Read leveling of one byte group: the bitslip with the widest delay window, the delay at its centre.
fn level(d: &mut Dev, module: u32) -> Option<(u32, u32)> {
    let mut best = (0, (0, 0));
    action(d, module, BITSLIP_RST);
    for bitslip in 0..BITSLIPS {
        let w = window(d, module);
        if w.1 > best.1 .1 {
            best = (bitslip, w);
        }
        action(d, module, BITSLIP_INC);
    }
    let (bitslip, (start, len)) = best;
    if len == 0 {
        return None;
    }
    action(d, module, BITSLIP_RST);
    (0..bitslip).for_each(|_| action(d, module, BITSLIP_INC));
    let delay = start + len / 2;
    action(d, module, RDLY_RST);
    (0..delay).for_each(|_| action(d, module, RDLY_INC));
    Some((bitslip, delay))
}

/// Calibrates the board's DDR3 if it is LiteDRAM's and not calibrated yet. True when memory is usable.
pub fn calibrate(d: &mut Dev) -> bool {
    if d.read(WINDOW_ID) != ID || d.read(INIT_DONE) == 1 {
        return true;
    }
    d.reg(INIT_ERROR, 0);
    d.reg(RDPHASE_REG, RDPHASE);
    d.reg(WRPHASE_REG, WRPHASE);
    d.reg(CONTROL, CTL_CKE | CTL_ODT | CTL_RESET_N);
    d.reg(PHY_RST, 1);
    settle(d, 1);
    d.reg(PHY_RST, 0);
    settle(d, 1);
    init_sequence(d);
    let mut ok = true;
    for m in 0..MODULES {
        match level(d, m) {
            Some((b, delay)) => eprintln!("borgvk: DDR3 byte {m}: bitslip {b}, delay {delay}"),
            None => {
                eprintln!("borgvk: DDR3 byte {m}: no working read delay");
                ok = false;
            }
        }
    }
    d.reg(CONTROL, CTL_SEL);
    d.reg(INIT_ERROR, !ok as u32);
    d.reg(INIT_DONE, 1);
    settle(d, 0);
    ok && memtest(d)
}

/// Words at the start and the end of the 512 MB written and read back through the controller.
fn memtest(d: &mut Dev) -> bool {
    let mut x = 0x1234_5678u32;
    for base in [0u32, 0x1ff0_0000] {
        let words: Vec<u32> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x
            })
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        d.mem_write(base, &bytes);
        let mut back = vec![0u8; bytes.len()];
        d.mem_read(base, &mut back);
        let bad = back.chunks(4).zip(&words).filter(|(b, w)| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != **w).count();
        if bad != 0 {
            eprintln!("borgvk: DDR3 memory test at {base:#x}: {bad} of {} words wrong", words.len());
            return false;
        }
    }
    true
}
