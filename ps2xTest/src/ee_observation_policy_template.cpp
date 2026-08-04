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
#define PS2_OBSERVATION_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_OBSERVATION_NOINLINE __attribute__((noinline, noipa))
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, noipa, cold,                               \
                   section(".text.unlikely.ee_observation")))
#define PS2_OBSERVATION_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define PS2_OBSERVATION_NOINLINE __attribute__((noinline))
#define PS2_OBSERVATION_COLD                                             \
    __attribute__((noinline, cold,                                      \
                   section(".text.unlikely.ee_observation")))
#define PS2_OBSERVATION_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

    template <EeArchitecturalObservationMode Mode>
    PS2_OBSERVATION_ALWAYS_INLINE void beginInstruction(
        PS2Runtime *runtime,
        R5900Context *ctx,
        uint32_t pc,
        EeInstructionIssueProfile profile)
    {
        ctx->pc = pc;
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
            if ((ctx->cop0_perf & 0x80000000u) != 0u)
            {
                runtime->recordEeInstructionIssue(ctx, profile);
            }
        }
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
    }

    template <EeArchitecturalObservationMode Mode>
    PS2_OBSERVATION_ALWAYS_INLINE void completeInstruction(
        PS2Runtime *runtime,
        R5900Context *ctx,
        uint32_t events)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            if ((ctx->cop0_perf & 0x80000000u) != 0u)
            {
                runtime->recordEeInstructionCompletion(ctx, events);
            }
        }
    }

    template <EeArchitecturalObservationMode Mode>
    PS2_OBSERVATION_ALWAYS_INLINE uint32_t load32(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            const bool checkValue =
                runtime->CheckEeDataAddressBreakpoint(
                    ctx, address, false);
            const uint32_t value = READ32(address);
            if (checkValue)
            {
                runtime->CheckEeDataValueBreakpoint(
                    ctx, address, value, false);
            }
            return value;
        }
        else
        {
            return READ32(address);
        }
    }

    template <EeArchitecturalObservationMode Mode>
    PS2_OBSERVATION_ALWAYS_INLINE void store32(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address,
        uint32_t value)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            if (runtime->CheckEeDataAddressBreakpoint(
                    ctx, address, true))
            {
                runtime->CheckEeDataValueBreakpoint(
                    ctx, address, value, true);
            }
        }
        WRITE32(address, value);
    }

    template <EeArchitecturalObservationMode Mode>
    PS2_OBSERVATION_ALWAYS_INLINE void recordBranch(
        PS2Runtime *runtime,
        R5900Context *ctx,
        bool taken)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            if ((ctx->cop0_perf & 0x80000000u) != 0u)
            {
                runtime->recordEeConditionalBranch(
                    ctx, kCodeBase + 12u, taken);
            }
        }
    }

    template <EeArchitecturalObservationMode Mode>
    PS2_OBSERVATION_ALWAYS_INLINE void generatedBody(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        constexpr uint32_t normalCompletion =
            kInstructionCompleted |
            kNonBdsInstructionCompleted;

        beginInstruction<Mode>(
            runtime,
            ctx,
            kCodeBase,
            {kIssuePipe0, 1u << 2u, 1u << 2u, 0u});
        SET_GPR_U32(
            ctx,
            2u,
            ADD32(GPR_U32(ctx, 2u), 0x9e3779b9u));
        completeInstruction<Mode>(
            runtime, ctx, normalCompletion);

        const uint32_t address =
            kDirectBase +
            (GPR_U32(ctx, 2u) & (kAddressSlots - 1u)) *
                sizeof(uint32_t);
        beginInstruction<Mode>(
            runtime,
            ctx,
            kCodeBase + 4u,
            {kIssuePipe1, 1u << 2u, 1u << 4u, 0u});
        SET_GPR_U32(
            ctx,
            4u,
            load32<Mode>(rdram, ctx, runtime, address));
        completeInstruction<Mode>(
            runtime,
            ctx,
            normalCompletion | kLoadCompleted);

        beginInstruction<Mode>(
            runtime,
            ctx,
            kCodeBase + 8u,
            {kIssuePipe1,
             (1u << 2u) | (1u << 3u) | (1u << 4u),
             0u,
             0u});
        store32<Mode>(
            rdram,
            ctx,
            runtime,
            address,
            GPR_U32(ctx, 2u) ^
                GPR_U32(ctx, 3u) ^
                GPR_U32(ctx, 4u));
        completeInstruction<Mode>(
            runtime,
            ctx,
            normalCompletion | kStoreCompleted);

        beginInstruction<Mode>(
            runtime,
            ctx,
            kCodeBase + 12u,
            {kIssuePipe0 | kIssuePipe1,
             (1u << 2u) | (1u << 3u),
             1u << 3u,
             kIssueBranch});
        const bool taken = (GPR_U32(ctx, 2u) & 1u) == 0u;
        recordBranch<Mode>(runtime, ctx, taken);
        SET_GPR_U32(
            ctx,
            3u,
            taken
                ? ADD32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u))
                : XOR32(GPR_U32(ctx, 3u), GPR_U32(ctx, 2u)));
        completeInstruction<Mode>(
            runtime, ctx, kInstructionCompleted);
        ctx->pc = kCodeBase + 16u;
    }

    extern "C" PS2_OBSERVATION_NOINLINE
    void EeObservationTemplateFast(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        generatedBody<EeArchitecturalObservationMode::Fast>(
            rdram, ctx, runtime);
    }

    extern "C" PS2_OBSERVATION_COLD
    void EeObservationTemplatePrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        generatedBody<EeArchitecturalObservationMode::Precise>(
            rdram, ctx, runtime);
    }
}
