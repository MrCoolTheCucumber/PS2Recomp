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

    // True when Fast generated code can account for an instruction before
    // its effect without immediately materializing COP0 Random. The runtime
    // access policy must materialize before any slow path which can fault or
    // publish the context. This is deliberately narrower than the ordinary
    // batchable set: these instructions are pre-retired rather than appended
    // to a successful straight-line run.
    [[nodiscard]] bool
    eeInstructionCanDeferRetirementMaterialization(
        const Instruction &instruction);
}

#endif
