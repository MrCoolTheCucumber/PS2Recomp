#ifndef PS2RECOMP_EE_OBSERVATION_MODE_H
#define PS2RECOMP_EE_OBSERVATION_MODE_H

#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

namespace ps2recomp
{
    // Only writes to the breakpoint register bank or performance-counter
    // register bank can change the generated EE observation specialization.
    // The selected sub-register may leave the mode unchanged; generated code
    // performs the runtime comparison only at these statically rare sites.
    inline bool mayChangeEeArchitecturalObservationMode(
        const Instruction &instruction) noexcept
    {
        return instruction.opcode == OPCODE_COP0 &&
               instruction.rs == COP0_MT &&
               (instruction.rd == COP0_REG_DEBUG ||
                instruction.rd == COP0_REG_PERF);
    }
}

#endif // PS2RECOMP_EE_OBSERVATION_MODE_H
