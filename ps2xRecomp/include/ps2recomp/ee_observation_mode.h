#ifndef PS2RECOMP_EE_OBSERVATION_MODE_H
#define PS2RECOMP_EE_OBSERVATION_MODE_H

#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <string>
#include <string_view>

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

    inline std::string eeFastFunctionName(std::string_view baseName)
    {
        return std::string(baseName) + "__ee_fast";
    }

    inline std::string eePreciseFunctionName(std::string_view baseName)
    {
        return std::string(baseName) + "__ee_precise";
    }

    inline std::string eeTemplateBodyName(std::string_view baseName)
    {
        return std::string(baseName) + "__ee_body";
    }

    inline std::string eeObservationDescriptorName(
        std::string_view baseName)
    {
        return std::string(baseName) + "__ee_observation";
    }
}

#endif // PS2RECOMP_EE_OBSERVATION_MODE_H
