use crate::isel::borg_isel;
use compiler::bindings::*;
use compiler::nir::AsDef;

/// Instruction counts of a shader; `tex` picks the fragment register set.
pub struct Survey {
    pub total: u32,
    pub tex: u32,
    pub push_window: bool,
}

pub unsafe fn survey(entry: *mut nir_function_impl, stage: mesa_shader_stage) -> Survey {
    let mut total = 0u32;
    let mut alu_ok = 0u32;
    let mut alu_todo = 0u32;
    let mut tex = 0u32;
    let mut consts = 0u32;
    let mut intrinsics = 0u32;
    let mut other = 0u32;
    let (mut push, mut mem) = (false, false);
    let mut todo: Vec<&'static str> = Vec::new();
    let mut note = |op: &'static str| {
        if !todo.contains(&op) {
            todo.push(op);
        }
    };

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
                    tex += (i.intrinsic == nir_intrinsic_load_constant && i.get_src(0).comp_as_uint(0).is_none()) as u32; // → TEX
                    push |= i.intrinsic == nir_intrinsic_load_push_constant;
                    mem |= i.intrinsic == nir_intrinsic_load_ubo || i.intrinsic == nir_intrinsic_load_ssbo
                        || i.intrinsic == nir_intrinsic_store_ssbo;
                    note(i.info().name());
                } else if instr.as_load_const().is_some() {
                    consts += 1;
                } else {
                    other += 1;
                }
            }
        }
    }

    debug!(
        "borgc: stage={stage} instrs={total} | alu_ok={alu_ok} alu_todo={alu_todo} \
         tex={tex} const={consts} intrinsic={intrinsics} other={other}"
    );
    if !todo.is_empty() {
        debug!("borgc:   needs lowering/selection: {}", todo.join(", "));
    }
    Survey { total, tex, push_window: push && mem }
}
