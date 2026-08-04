#include "ee_observation_policy_prototype.h"

#include "ps2_runtime_macros.h"
#include "runtime/ee_performance.h"

#include <cstddef>
#include <cstdint>

namespace ps2x::test::observation
{
    using namespace ps2x::performance;

#if defined(_MSC_VER)
#define PS2_OBSERVATION_COLD __declspec(noinline)
#define PS2_OBSERVATION_COLD_DATA
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, noipa, cold,                               \
                   section(".text.unlikely.ee_observation")))
#define PS2_OBSERVATION_COLD_DATA                                        \
    __attribute__((used, section(".rodata.unlikely.ee_observation")))
#else
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, cold,                                      \
                   section(".text.unlikely.ee_observation")))
#define PS2_OBSERVATION_COLD_DATA                                        \
    __attribute__((used, section(".rodata.unlikely.ee_observation")))
#endif

    enum class CompactOperation : uint8_t
    {
        AddImmediate,
        Load32,
        Store32,
        ConditionalBranch,
    };

    struct CompactInstructionDescriptor
    {
        uint32_t pc = 0u;
        EeInstructionIssueProfile issue{};
        uint8_t completionEvents = 0u;
        CompactOperation operation = CompactOperation::AddImmediate;
        uint16_t reserved = 0u;
    };

    static_assert(sizeof(CompactInstructionDescriptor) == 24u);

    constexpr uint8_t kNormalCompletion =
        kInstructionCompleted |
        kNonBdsInstructionCompleted;

    extern "C"
    {
        PS2_OBSERVATION_COLD_DATA
        extern const CompactInstructionDescriptor
            EeObservationCompactDescriptors[kInstructionsPerBlock] = {
                {kCodeBase,
                 {kIssuePipe0, 1u << 2u, 1u << 2u, 0u},
                 kNormalCompletion,
                 CompactOperation::AddImmediate,
                 0u},
                {kCodeBase + 4u,
                 {kIssuePipe1, 1u << 2u, 1u << 4u, 0u},
                 kNormalCompletion | kLoadCompleted,
                 CompactOperation::Load32,
                 0u},
                {kCodeBase + 8u,
                 {kIssuePipe1,
                  (1u << 2u) | (1u << 3u) | (1u << 4u),
                  0u,
                  0u},
                 kNormalCompletion | kStoreCompleted,
                 CompactOperation::Store32,
                 0u},
                {kCodeBase + 12u,
                 {kIssuePipe0 | kIssuePipe1,
                  (1u << 2u) | (1u << 3u),
                  1u << 3u,
                  kIssueBranch},
                 kInstructionCompleted,
                 CompactOperation::ConditionalBranch,
                 0u},
            };
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationCompactExecutor(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        const CompactInstructionDescriptor *instructions,
        size_t instructionCount)
    {
        for (size_t index = 0u;
             index < instructionCount;
             ++index)
        {
            const CompactInstructionDescriptor &instruction =
                instructions[index];
            ctx->pc = instruction.pc;
            runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
            if ((ctx->cop0_perf & 0x80000000u) != 0u)
            {
                runtime->recordEeInstructionIssue(
                    ctx, instruction.issue);
            }
            ctx->beginEeInstruction();
            ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);

            switch (instruction.operation)
            {
            case CompactOperation::AddImmediate:
                SET_GPR_U32(
                    ctx,
                    2u,
                    ADD32(GPR_U32(ctx, 2u), 0x9e3779b9u));
                break;
            case CompactOperation::Load32:
            {
                const uint32_t address =
                    kDirectBase +
                    (GPR_U32(ctx, 2u) &
                     (kAddressSlots - 1u)) *
                        sizeof(uint32_t);
                const bool checkValue =
                    runtime->CheckEeDataAddressBreakpoint(
                        ctx, address, false);
                const uint32_t value = READ32(address);
                if (checkValue)
                {
                    runtime->CheckEeDataValueBreakpoint(
                        ctx, address, value, false);
                }
                SET_GPR_U32(ctx, 4u, value);
                break;
            }
            case CompactOperation::Store32:
            {
                const uint32_t address =
                    kDirectBase +
                    (GPR_U32(ctx, 2u) &
                     (kAddressSlots - 1u)) *
                        sizeof(uint32_t);
                const uint32_t value =
                    GPR_U32(ctx, 2u) ^
                    GPR_U32(ctx, 3u) ^
                    GPR_U32(ctx, 4u);
                if (runtime->CheckEeDataAddressBreakpoint(
                        ctx, address, true))
                {
                    runtime->CheckEeDataValueBreakpoint(
                        ctx, address, value, true);
                }
                WRITE32(address, value);
                break;
            }
            case CompactOperation::ConditionalBranch:
            {
                const bool taken =
                    (GPR_U32(ctx, 2u) & 1u) == 0u;
                if ((ctx->cop0_perf & 0x80000000u) != 0u)
                {
                    runtime->recordEeConditionalBranch(
                        ctx, instruction.pc, taken);
                }
                SET_GPR_U32(
                    ctx,
                    3u,
                    taken
                        ? ADD32(
                              GPR_U32(ctx, 3u),
                              GPR_U32(ctx, 2u))
                        : XOR32(
                              GPR_U32(ctx, 3u),
                              GPR_U32(ctx, 2u)));
                break;
            }
            }

            if ((ctx->cop0_perf & 0x80000000u) != 0u)
            {
                runtime->recordEeInstructionCompletion(
                    ctx, instruction.completionEvents);
            }
        }
        ctx->pc = kCodeBase + 16u;
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationCompactPrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        EeObservationCompactExecutor(
            rdram,
            ctx,
            runtime,
            EeObservationCompactDescriptors,
            kInstructionsPerBlock);
    }
}
