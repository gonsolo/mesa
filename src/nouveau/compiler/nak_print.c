#include "nak_private.h"

void nak_print_instr(const nir_instr *instr)
{
        nir_print_instr(instr, stdout);
        puts("\n");
        fflush(stdout);
}

