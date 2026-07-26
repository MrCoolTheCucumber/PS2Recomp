#ifndef PS2RECOMP_EE_CYCLE_MODEL_H
#define PS2RECOMP_EE_CYCLE_MODEL_H

#include <cstdint>

namespace ps2recomp
{
    struct Instruction;

    // Approximate R5900 issue time in three-bit fixed-point ticks
    // (eight ticks per EE cycle while dual issue is enabled).
    uint32_t eeInstructionCycleTicks(const Instruction &inst);
}

#endif // PS2RECOMP_EE_CYCLE_MODEL_H
