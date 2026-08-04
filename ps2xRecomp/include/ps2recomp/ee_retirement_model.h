#ifndef PS2RECOMP_EE_RETIREMENT_MODEL_H
#define PS2RECOMP_EE_RETIREMENT_MODEL_H

namespace ps2recomp
{
    struct Instruction;

    // True only when the translated instruction cannot fault, transfer
    // control, call an observing runtime helper, or otherwise publish the EE
    // context. Such instructions may defer insn_count/Random retirement until
    // the end of a straight-line generated run in Fast mode.
    [[nodiscard]] bool
    eeInstructionCanBatchRetirement(const Instruction &instruction);
}

#endif
