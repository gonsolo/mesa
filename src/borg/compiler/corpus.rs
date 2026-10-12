use std::env;

pub fn key(nir_text: &str, vfetch: u32) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in nir_text.bytes().chain(vfetch.to_le_bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

pub fn record(key: u64, ret: u32, blob: &[u8]) {
    let Ok(dir) = env::var("BORGC_CORPUS_DIR") else { return };
    let mut out = ret.to_le_bytes().to_vec();
    out.extend_from_slice(blob);
    let _ = std::fs::write(format!("{dir}/{key:016x}.out"), out);
}
