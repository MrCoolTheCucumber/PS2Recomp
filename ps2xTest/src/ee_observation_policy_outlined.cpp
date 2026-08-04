#include "ee_observation_policy_prototype.h"

#include "ps2_runtime_macros.h"
#include "runtime/ee_performance.h"

#include <cstdint>

namespace ps2x::test::observation
{
    using namespace ps2x::performance;

#if defined(_MSC_VER)
#define PS2_OBSERVATION_COLD __declspec(noinline)
#define PS2_OBSERVATION_COLD_DATA
#define PS2_OBSERVATION_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, noipa, cold,                               \
                   section(".text.unlikely.ee_observation")))
#define PS2_OBSERVATION_COLD_DATA                                        \
    __attribute__((used, section(".rodata.unlikely.ee_observation")))
#define PS2_OBSERVATION_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, cold,                                      \
                   section(".text.unlikely.ee_observation")))
#define PS2_OBSERVATION_COLD_DATA                                        \
    __attribute__((used, section(".rodata.unlikely.ee_observation")))
#define PS2_OBSERVATION_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

    struct OutlinedInstructionDescriptor
    {
        uint32_t pc = 0u;
        EeInstructionIssueProfile issue{};
        uint32_t completionEvents = 0u;
    };

    static_assert(sizeof(OutlinedInstructionDescriptor) == 24u);

    constexpr uint32_t kNormalCompletion =
        kInstructionCompleted |
        kNonBdsInstructionCompleted;

    extern "C"
    {
        PS2_OBSERVATION_COLD_DATA
        extern const OutlinedInstructionDescriptor
            EeObservationOutlinedDescriptors[kInstructionsPerBlock] = {
                {kCodeBase,
                 {kIssuePipe0, 1u << 2u, 1u << 2u, 0u},
                 kNormalCompletion},
                {kCodeBase + 4u,
                 {kIssuePipe1, 1u << 2u, 1u << 4u, 0u},
                 kNormalCompletion | kLoadCompleted},
                {kCodeBase + 8u,
                 {kIssuePipe1,
                  (1u << 2u) | (1u << 3u) | (1u << 4u),
                  0u,
                  0u},
                 kNormalCompletion | kStoreCompleted},
                {kCodeBase + 12u,
                 {kIssuePipe0 | kIssuePipe1,
                  (1u << 2u) | (1u << 3u),
                  1u << 3u,
                  kIssueBranch},
                 kInstructionCompleted},
            };
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationOutlinedBegin(
        R5900Context *ctx,
        PS2Runtime *runtime,
        const OutlinedInstructionDescriptor *instruction)
    {
        runtime->CheckEeInstructionBreakpoint(ctx, instruction->pc);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionIssue(
                ctx, instruction->issue);
        }
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationOutlinedComplete(
        R5900Context *ctx,
        PS2Runtime *runtime,
        const OutlinedInstructionDescriptor *instruction)
    {
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionCompletion(
                ctx, instruction->completionEvents);
        }
    }

    extern "C" PS2_OBSERVATION_COLD
    bool EeObservationOutlinedDataAddress(
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address,
        bool write)
    {
        return runtime->CheckEeDataAddressBreakpoint(
            ctx, address, write);
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationOutlinedDataValue(
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address,
        uint32_t value,
        bool write)
    {
        runtime->CheckEeDataValueBreakpoint(
            ctx, address, value, write);
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationOutlinedBranch(
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t pc,
        bool taken)
    {
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeConditionalBranch(ctx, pc, taken);
        }
    }

    PS2_OBSERVATION_ALWAYS_INLINE void beginInstruction(
        R5900Context *ctx,
        PS2Runtime *runtime,
        const OutlinedInstructionDescriptor &instruction)
    {
        ctx->pc = instruction.pc;
        EeObservationOutlinedBegin(
            ctx, runtime, &instruction);
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
    }

    PS2_OBSERVATION_ALWAYS_INLINE void completeInstruction(
        R5900Context *ctx,
        PS2Runtime *runtime,
        const OutlinedInstructionDescriptor &instruction)
    {
        EeObservationOutlinedComplete(
            ctx, runtime, &instruction);
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationOutlinedPrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        const OutlinedInstructionDescriptor *const instructions =
            EeObservationOutlinedDescriptors;

        beginInstruction(ctx, runtime, instructions[0u]);
        SET_GPR_U32(
            ctx,
            2u,
            ADD32(GPR_U32(ctx, 2u), 0x9e3779b9u));
        completeInstruction(ctx, runtime, instructions[0u]);

        const uint32_t address =
            kDirectBase +
            (GPR_U32(ctx, 2u) & (kAddressSlots - 1u)) *
                sizeof(uint32_t);
        beginInstruction(ctx, runtime, instructions[1u]);
        const bool checkLoadValue =
            EeObservationOutlinedDataAddress(
                ctx, runtime, address, false);
        const uint32_t loaded = READ32(address);
        if (checkLoadValue)
        {
            EeObservationOutlinedDataValue(
                ctx, runtime, address, loaded, false);
        }
        SET_GPR_U32(ctx, 4u, loaded);
        completeInstruction(ctx, runtime, instructions[1u]);

        beginInstruction(ctx, runtime, instructions[2u]);
        const uint32_t stored =
            GPR_U32(ctx, 2u) ^
            GPR_U32(ctx, 3u) ^
            GPR_U32(ctx, 4u);
        if (EeObservationOutlinedDataAddress(
                ctx, runtime, address, true))
        {
            EeObservationOutlinedDataValue(
                ctx, runtime, address, stored, true);
        }
        WRITE32(address, stored);
        completeInstruction(ctx, runtime, instructions[2u]);

        beginInstruction(ctx, runtime, instructions[3u]);
        const bool taken = (GPR_U32(ctx, 2u) & 1u) == 0u;
        EeObservationOutlinedBranch(
            ctx, runtime, instructions[3u].pc, taken);
        SET_GPR_U32(
            ctx,
            3u,
            taken
                ? ADD32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u))
                : XOR32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u)));
        completeInstruction(ctx, runtime, instructions[3u]);
        ctx->pc = kCodeBase + 16u;
    }
}
