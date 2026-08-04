#include "ee_observation_policy_prototype.h"

#include "ps2_runtime_macros.h"
#include "runtime/ee_performance.h"

#include <cstdint>

namespace ps2x::test::observation
{
    using namespace ps2x::performance;

#if defined(_MSC_VER)
#define PS2_OBSERVATION_NOINLINE __declspec(noinline)
#define PS2_OBSERVATION_COLD __declspec(noinline)
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_OBSERVATION_NOINLINE __attribute__((noinline, noipa))
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, noipa, cold,                               \
                   section(".text.unlikely.ee_observation")))
#else
#define PS2_OBSERVATION_NOINLINE __attribute__((noinline))
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, cold,                                      \
                   section(".text.unlikely.ee_observation")))
#endif

    extern "C" PS2_OBSERVATION_NOINLINE
    void EeObservationExplicitFast(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        static_cast<void>(runtime);

        ctx->pc = kCodeBase;
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        SET_GPR_U32(
            ctx,
            2u,
            ADD32(GPR_U32(ctx, 2u), 0x9e3779b9u));

        const uint32_t address =
            kDirectBase +
            (GPR_U32(ctx, 2u) & (kAddressSlots - 1u)) *
                sizeof(uint32_t);
        ctx->pc = kCodeBase + 4u;
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        SET_GPR_U32(ctx, 4u, READ32(address));

        ctx->pc = kCodeBase + 8u;
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        WRITE32(
            address,
            GPR_U32(ctx, 2u) ^
                GPR_U32(ctx, 3u) ^
                GPR_U32(ctx, 4u));

        ctx->pc = kCodeBase + 12u;
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        const bool taken = (GPR_U32(ctx, 2u) & 1u) == 0u;
        SET_GPR_U32(
            ctx,
            3u,
            taken
                ? ADD32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u))
                : XOR32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u)));
        ctx->pc = kCodeBase + 16u;
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationExplicitPrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        constexpr uint32_t normalCompletion =
            kInstructionCompleted |
            kNonBdsInstructionCompleted;

        ctx->pc = kCodeBase;
        runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionIssue(
                ctx,
                {kIssuePipe0, 1u << 2u, 1u << 2u, 0u});
        }
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        SET_GPR_U32(
            ctx,
            2u,
            ADD32(GPR_U32(ctx, 2u), 0x9e3779b9u));
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionCompletion(
                ctx, normalCompletion);
        }

        const uint32_t address =
            kDirectBase +
            (GPR_U32(ctx, 2u) & (kAddressSlots - 1u)) *
                sizeof(uint32_t);
        ctx->pc = kCodeBase + 4u;
        runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionIssue(
                ctx,
                {kIssuePipe1, 1u << 2u, 1u << 4u, 0u});
        }
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        const bool checkLoadValue =
            runtime->CheckEeDataAddressBreakpoint(
                ctx, address, false);
        const uint32_t loaded = READ32(address);
        if (checkLoadValue)
        {
            runtime->CheckEeDataValueBreakpoint(
                ctx, address, loaded, false);
        }
        SET_GPR_U32(ctx, 4u, loaded);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionCompletion(
                ctx,
                normalCompletion | kLoadCompleted);
        }

        ctx->pc = kCodeBase + 8u;
        runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionIssue(
                ctx,
                {kIssuePipe1,
                 (1u << 2u) | (1u << 3u) | (1u << 4u),
                 0u,
                 0u});
        }
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        const uint32_t stored =
            GPR_U32(ctx, 2u) ^
            GPR_U32(ctx, 3u) ^
            GPR_U32(ctx, 4u);
        if (runtime->CheckEeDataAddressBreakpoint(
                ctx, address, true))
        {
            runtime->CheckEeDataValueBreakpoint(
                ctx, address, stored, true);
        }
        WRITE32(address, stored);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionCompletion(
                ctx,
                normalCompletion | kStoreCompleted);
        }

        ctx->pc = kCodeBase + 12u;
        runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionIssue(
                ctx,
                {kIssuePipe0 | kIssuePipe1,
                 (1u << 2u) | (1u << 3u),
                 1u << 3u,
                 kIssueBranch});
        }
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
        const bool taken = (GPR_U32(ctx, 2u) & 1u) == 0u;
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeConditionalBranch(
                ctx, kCodeBase + 12u, taken);
        }
        SET_GPR_U32(
            ctx,
            3u,
            taken
                ? ADD32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u))
                : XOR32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u)));
        if ((ctx->cop0_perf & 0x80000000u) != 0u)
        {
            runtime->recordEeInstructionCompletion(
                ctx, kInstructionCompleted);
        }
        ctx->pc = kCodeBase + 16u;
    }
}
