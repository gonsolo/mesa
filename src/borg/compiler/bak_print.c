#include "bak_private.h"

void bak_print_instr(const nir_instr *instr)
{
        nir_print_instr(instr, stdout);
        puts("\n");
        fflush(stdout);
}

void bak_print_defer_instr(const nir_instr *instr)
{
        nir_deref_instr *di = nir_instr_as_deref(instr);
        nir_print_deref(di, stdout);
        puts("\n");
        fflush(stdout);
}
