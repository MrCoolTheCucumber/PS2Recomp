#ifndef PS2RECOMP_EE_PERFORMANCE_MODEL_H
#define PS2RECOMP_EE_PERFORMANCE_MODEL_H

#include <cstdint>
#include <string>

namespace ps2recomp
{
    struct Instruction;

    struct EeInstructionPerformanceInfo
    {
        uint32_t issuePipeMask = 0u;
        uint32_t issueFlags = 0u;
        uint32_t completionEvents = 0u;
        bool completionBeforeEffect = false;
    };

    [[nodiscard]] EeInstructionPerformanceInfo
    eeInstructionPerformanceInfo(const Instruction &inst);
    [[nodiscard]] std::string
    eeInstructionIssueCall(const Instruction &inst);
    [[nodiscard]] std::string
    eeInstructionCompletionCall(const Instruction &inst);
}

#endif
