#pragma once

#include <cstdint>

namespace ps2x::performance
{
    inline constexpr uint32_t kIssuePipe0 = 1u << 0u;
    inline constexpr uint32_t kIssuePipe1 = 1u << 1u;
    inline constexpr uint32_t kIssueBranch = 1u << 0u;
    inline constexpr uint32_t kIssueEret = 1u << 1u;
    inline constexpr uint32_t kIssueConservativeBarrier = 1u << 2u;

    inline constexpr uint32_t kInstructionCompleted = 1u << 0u;
    inline constexpr uint32_t kNonBdsInstructionCompleted = 1u << 1u;
    inline constexpr uint32_t kCop2InstructionCompleted = 1u << 2u;
    inline constexpr uint32_t kCop1InstructionCompleted = 1u << 3u;
    inline constexpr uint32_t kLoadCompleted = 1u << 4u;
    inline constexpr uint32_t kStoreCompleted = 1u << 5u;

    struct EeInstructionIssueProfile
    {
        uint32_t pipeMask = 0u;
        uint32_t gprReadMask = 0u;
        uint32_t gprWriteMask = 0u;
        uint32_t flags = 0u;
    };
}
