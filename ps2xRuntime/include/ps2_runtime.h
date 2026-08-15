#ifndef PS2_RUNTIME_H
#define PS2_RUNTIME_H

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX instructions
#include <smmintrin.h> // For SSE4.1 instructions
#endif
#include <atomic>
#include <array>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include "runtime/ps2_build_config.h"
#include "ps2_log.h"
#include "runtime/cop0_timing.h"
#include "runtime/ee_cache.h"
#include "runtime/ee_performance.h"
#include "runtime/ee_execution_backend.h"
#include "runtime/ee_event_scheduler.h"
#include "runtime/ee_runtime_executor.h"
#include "runtime/ps2_address.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_gs_command_stream.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_vu_clip.h"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_command_stream.h"
#include "runtime/ps2_audio.h"
#include "runtime/ps2_pad.h"
#include "ps2x/iop/iop_types.h"

namespace ps2x::iop
{
    class IopSubsystem;
}

class PS2IopHostAdapter;
class PS2IopTransport;
class PS2DebugServer;
struct HostPresentationUploadState;
struct ThreadInfo;
struct EeThreadRuntimeState;
struct EeAlarmRuntimeState;
struct EeSyncRuntimeState;
struct EeInterruptRuntimeState;
struct EeKernelRuntimeState;
struct EeRpcRuntimeState;
struct EeFileRuntimeState;
struct Deci2RuntimeState;
namespace ps2_stubs
{
    struct AudioRuntimeState;
    struct CdRuntimeState;
    struct DmaRuntimeState;
    struct LibCRuntimeState;
    struct MemoryCardRuntimeState;
    struct PadRuntimeState;
    class MpegRuntimeState;
    struct SifRuntimeState;
}

struct PS2RuntimeConfiguration
{
    // Desktop monitor index used to place the host window after creation.
    // An unset value leaves placement to the window manager.
    std::optional<int> hostMonitorIndex;
    GsExecutionMode gsExecutionMode = GsExecutionMode::Inline;
    size_t gsCommandQueueCapacity = 64u;
    uint64_t gsCommandPayloadCapacityBytes =
        64u * 1024u * 1024u;
    uint32_t gsMaximumFieldLead = 1u;
    // Deterministic owner hooks are intended for focused scheduling tests.
    std::function<void(const GsCommand &, uint64_t)>
        gsBeforeProcess;
    std::function<void(const GsCommandResult &, uint64_t)>
        gsBeforePublish;
    Vu1ExecutionMode vu1ExecutionMode = Vu1ExecutionMode::Inline;
    size_t vu1CommandQueueCapacity = 64u;
    uint64_t vu1CommandPayloadCapacityBytes =
        8u * 1024u * 1024u;
    // Phase A remains available as a differential/debugger fallback. The
    // production runner explicitly enables bounded Phase B checkpoint
    // epochs for threaded-async execution.
    bool vu1CheckpointEpochs = false;
    size_t vu1CheckpointCapacity = 4u;
    // The coarse mode is a separately selected timing experiment. It owns a
    // whole activation on the VU thread and publishes at a deterministic
    // guest-cycle estimate; none of these bounds affect exact modes.
    uint32_t vu1CoarseInitialEstimateCycles = 16u;
    uint32_t vu1CoarseMinimumEstimateCycles = 4u;
    uint32_t vu1CoarseMaximumLeadCycles = 65536u;
    uint32_t vu1CoarseMaximumInvocationCycles = 65536u;
    uint64_t vu1CoarseMaximumPath1Bytes =
        64u * 1024u * 1024u;
    size_t vu1CoarseEstimatorHistoryCapacity = 4u;
    size_t vu1CoarseEstimatorIdentityCapacity = 64u;
    std::function<void(const Vu1Command &, uint64_t)>
        vu1BeforeProcess;
    std::function<void(const Vu1CommandResult &, uint64_t)>
        vu1BeforePublish;
    EeExecutionBackendKind eeExecutionBackend =
        EeExecutionBackendKind::LegacyHostThread;
    bool useEeExecutionBackendEnvironment = true;
    VuBackendKind vu0Backend = VuBackendKind::Auto;
    VuBackendKind vu1Backend = VuBackendKind::Auto;
    bool vu0NativeInstrumentation = false;
    bool vu1NativeInstrumentation = false;
    bool captureVu1CommandDigests = false;
    bool captureVu1ArchitecturalStateHashes = false;
    bool useVuBackendEnvironment = true;
    bool eeThreadDiagnostics = false;
    bool useEeThreadDiagnosticsEnvironment = true;
    bool eeTlbTranslationCacheDiagnostics = false;
    bool useEeTlbTranslationCacheDiagnosticsEnvironment = true;
};

struct GsAsyncRuntimeStatistics
{
    bool enabled = false;
    size_t pendingCompletions = 0u;
    size_t pendingHighWater = 0u;
    uint64_t fieldsSubmitted = 0u;
    uint64_t fieldsCompleted = 0u;
    uint64_t lastSubmittedFieldSequence = 0u;
    uint64_t lastCompletedFieldSequence = 0u;
    uint64_t fieldLeadHighWater = 0u;
    uint64_t fieldLeadBlockCount = 0u;
    uint64_t fieldLeadBlockedNanoseconds = 0u;
    ThreadedGsExecutorStatistics owner{};
};

struct Vu1AsyncRuntimeStatistics
{
    bool enabled = false;
    bool pendingSlice = false;
    bool deferredSlice = false;
    uint64_t slicesSubmitted = 0u;
    uint64_t slicesPublished = 0u;
    uint64_t resultsReadyAtEvent = 0u;
    uint64_t resultsLateAtEvent = 0u;
    uint64_t eventWaitCount = 0u;
    uint64_t eventWaitNanoseconds = 0u;
    uint64_t maximumEventWaitNanoseconds = 0u;
    uint64_t budgetFallbackCount = 0u;
    uint64_t budgetFallbackCycleDelta = 0u;
    uint64_t budgetFallbackWaitNanoseconds = 0u;
    uint64_t budgetExtensionCount = 0u;
    uint64_t budgetExtensionCycleDelta = 0u;
    uint64_t budgetExtensionExecutedCycles = 0u;
    uint64_t budgetExtensionWaitNanoseconds = 0u;
    uint64_t hazardBarrierCount = 0u;
    uint64_t hazardWaitNanoseconds = 0u;
    uint64_t hazardRequeueCount = 0u;
    uint64_t deferredSliceCount = 0u;
    uint64_t deferredSliceArmCount = 0u;
    uint64_t forcedDeferredSliceArmCount = 0u;
    uint64_t sliceSubmitCount = 0u;
    uint64_t sliceSubmitNanoseconds = 0u;
    uint64_t maximumSliceSubmitNanoseconds = 0u;
    ThreadedVu1ExecutorStatistics owner{};
};

struct Vu1CoarseRuntimeStatistics
{
    bool enabled = false;
    bool pendingInvocation = false;
    uint32_t initialEstimateCycles = 0u;
    uint32_t minimumEstimateCycles = 0u;
    uint32_t maximumLeadCycles = 0u;
    uint32_t maximumInvocationCycles = 0u;
    uint64_t maximumPath1Bytes = 0u;
    size_t estimatorHistoryCapacity = 0u;
    size_t estimatorHistorySize = 0u;
    size_t estimatorHistoryHighWater = 0u;
    size_t estimatorIdentityCapacity = 0u;
    size_t estimatorIdentitySize = 0u;
    size_t estimatorIdentityHighWater = 0u;
    uint64_t estimatorIdentityHitCount = 0u;
    uint64_t estimatorIdentityMissCount = 0u;
    size_t pendingInvocationHighWater = 0u;
    uint64_t invocationsSubmitted = 0u;
    uint64_t invocationsCompleted = 0u;
    uint64_t invocationsPublished = 0u;
    uint64_t invocationsDiscardedAtFailure = 0u;
    uint64_t invocationsDiscardedAtShutdown = 0u;
    uint64_t resultsReadyAtEvent = 0u;
    uint64_t resultsLateAtEvent = 0u;
    uint64_t earlyEstimateCount = 0u;
    uint64_t exactEstimateCount = 0u;
    uint64_t lateEstimateCount = 0u;
    uint64_t estimatedCycleSum = 0u;
    uint64_t actualCycleSum = 0u;
    uint64_t absoluteErrorCycleSum = 0u;
    uint64_t maximumAbsoluteErrorCycles = 0u;
    uint64_t causalDeadlineDeferralCount = 0u;
    uint64_t causalDeadlineDeferredCycles = 0u;
    uint64_t maximumCausalDeadlineDeferredCycles = 0u;
    uint64_t forcedBarrierCount = 0u;
    std::array<uint64_t, kVu1CommandTypeCount>
        forcedBarriersByCommandType{};
    uint64_t nonPublishingDiagnosticSnapshotCount = 0u;
    uint64_t nonPublishingDiagnosticsUpdateCount = 0u;
    uint64_t nonPublishingBackendChangeCount = 0u;
    uint64_t nonPublishingVifStateUpdateCount = 0u;
    uint64_t nonPublishingDiagnosticSnapshotWaitCount = 0u;
    uint64_t nonPublishingDiagnosticSnapshotWaitNanoseconds = 0u;
    uint64_t maximumNonPublishingDiagnosticSnapshotWaitNanoseconds = 0u;
    uint64_t forcedWaitCount = 0u;
    uint64_t forcedWaitNanoseconds = 0u;
    uint64_t maximumForcedWaitNanoseconds = 0u;
    uint64_t path1ReservationsStarted = 0u;
    uint64_t path1ReservationsReleased = 0u;
    uint64_t path1ReservationsInvalidated = 0u;
    uint64_t path1ReservationsDiscardedAtFailure = 0u;
    uint64_t path1ReservationsDiscardedAtShutdown = 0u;
    uint64_t path1BytesPublished = 0u;
    uint64_t maximumInvocationPath1Bytes = 0u;
    uint32_t lastEstimatedCycles = 0u;
    uint32_t lastActualCycles = 0u;
    int64_t lastEstimateErrorCycles = 0;
    uint64_t estimatorInputDigest = 0u;
    uint64_t estimatorOutputDigest = 0u;
    uint64_t estimatorResultDigest = 0u;
    ThreadedVu1ExecutorStatistics owner{};
};

[[nodiscard]] PS2RuntimeConfiguration
defaultPs2RuntimeConfiguration() noexcept;

enum PS2Exception
{
    EXCEPTION_INTERRUPT = 0x00,
    EXCEPTION_TLB_MODIFIED = 0x01,
    EXCEPTION_TLB_REFILL_LOAD = 0x02,
    EXCEPTION_TLB_REFILL = EXCEPTION_TLB_REFILL_LOAD,
    EXCEPTION_TLB_REFILL_STORE = 0x03,
    EXCEPTION_ADDRESS_ERROR_LOAD = 0x04,  // Address error on load
    EXCEPTION_ADDRESS_ERROR_STORE = 0x05, // Address error on store
    EXCEPTION_BUS_ERROR_INSTRUCTION = 0x06,
    EXCEPTION_BUS_ERROR_DATA = 0x07,
    EXCEPTION_SYSCALL = 0x08,             // SYSCALL instruction
    EXCEPTION_BREAKPOINT = 0x09,          // BREAK instruction
    EXCEPTION_RESERVED_INSTRUCTION = 0x0A,
    EXCEPTION_COPROCESSOR_UNUSABLE = 0x0B,
    EXCEPTION_INTEGER_OVERFLOW = 0x0C, // From MIPS spec
    EXCEPTION_TRAP = 0x0D,             // Trap instruction condition met
};

enum class PS2BusErrorAccess : uint8_t
{
    InstructionFetch,
    DataLoad,
    DataStore,
};

// Internal control-flow signal used to leave generated guest code immediately
// after the runtime has entered an EE exception.
struct PS2GuestException final
{
};

enum class PS2GuestCheckpointResult : uint8_t
{
    Continue,
    ExitToDispatcher,
};

enum class EeArchitecturalObservationMode : uint8_t
{
    Fast,
    Precise,
};

struct EeObservationInstructionDescriptor
{
    uint32_t pc = 0u;
    ps2x::performance::EeInstructionIssueProfile issue{};
    uint32_t completionEvents = 0u;
};

static_assert(sizeof(EeObservationInstructionDescriptor) == 24u);

#if defined(__GNUC__) || defined(__clang__)
#define PS2X_EE_OBSERVATION_COLD_DATA                                  \
    __attribute__((used, section(".rodata.unlikely.ee_observation")))
#else
#define PS2X_EE_OBSERVATION_COLD_DATA
#endif

#if defined(_MSC_VER)
#define PS2X_EE_OBSERVATION_POLICY_INLINE __forceinline
#define PS2X_EE_RUNTIME_NOINLINE __declspec(noinline)
#define PS2X_EE_MEMORY_POLICY_NOINLINE PS2X_EE_RUNTIME_NOINLINE
#define PS2X_EE_OBSERVATION_PRECISE_BODY __declspec(noinline)
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2X_EE_OBSERVATION_POLICY_INLINE                              \
    inline __attribute__((always_inline))
#define PS2X_EE_RUNTIME_NOINLINE                                       \
    __attribute__((noinline, noipa))
#define PS2X_EE_MEMORY_POLICY_NOINLINE PS2X_EE_RUNTIME_NOINLINE
#define PS2X_EE_OBSERVATION_PRECISE_BODY                               \
    __attribute__((noinline, noipa, cold, optimize("Oz"),             \
                   section(".text.unlikely.ee_observation")))
#elif defined(__clang__)
#define PS2X_EE_OBSERVATION_POLICY_INLINE                              \
    inline __attribute__((always_inline))
#define PS2X_EE_RUNTIME_NOINLINE __attribute__((noinline))
#define PS2X_EE_MEMORY_POLICY_NOINLINE PS2X_EE_RUNTIME_NOINLINE
#define PS2X_EE_OBSERVATION_PRECISE_BODY                               \
    __attribute__((noinline, cold, minsize,                            \
                   section(".text.unlikely.ee_observation")))
#else
#define PS2X_EE_OBSERVATION_POLICY_INLINE inline
#define PS2X_EE_RUNTIME_NOINLINE
#define PS2X_EE_MEMORY_POLICY_NOINLINE
#define PS2X_EE_OBSERVATION_PRECISE_BODY
#endif

inline constexpr uint32_t PS2_RECOMPILED_FUNCTION_PAIR_ABI_VERSION = 2u;

struct PS2GuestFunctionSymbol
{
    uint32_t start;
    uint32_t end;
    const char *name;
};

// PS2 CPU context (R5900)
struct alignas(16) R5900Context
{
    enum class InitializationProfile : uint8_t
    {
        Deterministic,
        PostBiosElf,
    };

    static constexpr uint32_t kPostBiosElfStatus = 0x70020C11u;

    // General Purpose Registers (128-bit)
    __m128i r[32]; // Main registers

    // Control registers
    uint32_t pc;         // Program counter
    uint64_t insn_count; // Retired instruction counter
    uint32_t ee_block_cycle_ticks;
    bool ee_block_cycle_active;
    // Instructions whose architectural retirement is already reflected in
    // insn_count, but whose identical COP0 Random decrements have not yet
    // been materialized. Generated straight-line runs may accumulate here;
    // every architectural observation boundary drains the count first.
    uint64_t cop0_random_pending_retirements;
    bool ee_performance_issue_pending;
    ps2x::performance::EeInstructionIssueProfile
        ee_performance_pending_issue;
    uint64_t hi, lo;     // Lower 64-bit lanes of the 128-bit HI/LO registers
    uint64_t hi1, lo1;   // Upper lanes, also used by scalar MULT1/DIV1 operations
    uint32_t sa;         // Shift amount register

    // VU0 registers (when used in macro mode)
    __m128 vu0_vf[32];        // VU0 vector float registers
    uint16_t vi[16];          // VU0 vector integer registers
    float vu0_q;              // VU0 Q register (quotient)
    float vu0_p;              // VU0 P register (EFU result)
    float vu0_i;              // VU0 I register (integer value)
    __m128 vu0_r;             // VU0 R register
    __m128 vu0_acc;           // VU0 ACC accumulator register
    uint16_t vu0_status;      // VU0 status register
    uint32_t vu0_mac_flags;   // VU0 MAC flags
    uint32_t vu0_clip_flags;  // VU0 clipping flags
    uint32_t vu0_clip_flags2; // VU0 clipping flags
    uint32_t vu0_cmsar0;      // VU0 microprogram start address
    uint32_t vu0_cmsar1;      // VU0 microprogram start address
    uint32_t vu0_cmsar2;      // VU0 microprogram start address
    uint32_t vu0_cmsar3;      // VU0 microprogram start address
    uint32_t vu0_vpu_stat;
    uint32_t vu0_vpu_stat2; // extra VPU status (used by CR_VPU_STAT2)
    uint32_t vu0_vpu_stat3; // extra VPU status 3
    uint32_t vu0_vpu_stat4; // extra VPU status 4
    uint32_t vu0_tpc;       // TPC (VU0 PC)
    uint32_t vu0_tpc2;      // second TPC
    uint32_t vu0_fbrst;     // VIF/VU reset register
    uint32_t vu0_fbrst2;    // FBRST2
    uint32_t vu0_fbrst3;    // FBRST3
    uint32_t vu0_fbrst4;    // FBRST4
    uint32_t vu0_itop;
    uint32_t vu0_top;
    uint32_t vu0_info;
    uint32_t vu0_xitop; // VU0 XITOP - input ITOP for VIF/VU sync
    uint32_t vu0_pc;

    float vu0_cf[4]; // VU0 FMAC control floating-point registers

    // COP0 System control registers
    uint32_t cop0_index;
    uint32_t cop0_random;
    uint32_t cop0_entrylo0;
    uint32_t cop0_entrylo1;
    uint32_t cop0_context;
    uint32_t cop0_pagemask;
    uint32_t cop0_wired;
    uint32_t cop0_badvaddr;
    uint32_t cop0_count;
    uint32_t cop0_entryhi;
    uint32_t cop0_compare;
    uint32_t cop0_status;
    uint32_t cop0_cause;
    uint32_t cop0_epc;
    uint32_t cop0_prid;
    uint32_t cop0_config;
    uint32_t cop0_badpaddr;
    uint32_t cop0_bpc;
    uint32_t cop0_iab;
    uint32_t cop0_iabm;
    uint32_t cop0_dab;
    uint32_t cop0_dabm;
    uint32_t cop0_dvb;
    uint32_t cop0_dvbm;
    uint32_t cop0_perf;
    uint32_t cop0_pcr0;
    uint32_t cop0_pcr1;
    uint32_t cop0_taglo;
    uint32_t cop0_taghi;
    uint32_t cop0_errorepc;

    // Delay slot state tracking
    bool in_delay_slot;
    uint32_t branch_pc;

    // COP2 control registers (VU0 integer + control)
    uint32_t cop2_ccr[32];

    // FPU registers (COP1)
    float f[32];
    float f_acc;    // FPU accumulator
    uint32_t fcr31; // Control/status register

    void enforceVu0RegisterInvariants()
    {
        // VF0 and VI0 are architectural constants. Instructions may name them
        // as destinations, but those writes must be discarded.
        vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        vi[0] = 0;
    }

    void advanceCop0Random(uint64_t retirementCount) noexcept
    {
        if (retirementCount == 0u)
        {
            return;
        }

        constexpr uint32_t upperBound = 47u;
        const uint32_t lowerBound =
            std::min(cop0_wired & 0x3fu, upperBound);
        const uint32_t interval =
            upperBound - lowerBound + 1u;
        uint32_t random =
            cop0_random & 0x3fu;

        // Out-of-range state follows the scalar EE rule: the first
        // retirement selects entry 47. Thereafter Random is a cyclic
        // decrement over the closed [Wired, 47] interval.
        if (random < lowerBound || random > upperBound)
        {
            random = upperBound;
            --retirementCount;
        }
        if (retirementCount != 0u)
        {
            const uint32_t offset = random - lowerBound;
            if (retirementCount <= offset)
            {
                random -= static_cast<uint32_t>(retirementCount);
            }
            else
            {
                // Consuming offset + 1 retirements wraps Random from the
                // lower bound to 47. Generated basic blocks normally end
                // within the following interval, so handle that common case
                // without either of the dynamic divisions used by the
                // general modulo expression.
                retirementCount -=
                    static_cast<uint64_t>(offset) + 1u;
                if (retirementCount >= interval)
                {
                    retirementCount -= interval;
                    if (retirementCount >= interval)
                    {
                        retirementCount %= interval;
                    }
                }
                random = upperBound -
                         static_cast<uint32_t>(retirementCount);
            }
        }
        cop0_random = random;
    }

    void finishEeInstruction() noexcept
    {
        advanceCop0Random(
            cop0_random_pending_retirements);
        cop0_random_pending_retirements = 0u;
    }

    void retireEeInstructions(uint64_t retirementCount) noexcept
    {
        if (retirementCount == 0u)
        {
            return;
        }
        if (cop0_random_pending_retirements >
            std::numeric_limits<uint64_t>::max() - retirementCount)
        {
            finishEeInstruction();
        }
        insn_count += retirementCount;
        cop0_random_pending_retirements += retirementCount;
    }

    void beginEeInstruction() noexcept
    {
        finishEeInstruction();
        ++insn_count;
        cop0_random_pending_retirements = 1u;
    }

    void advanceEeCycleTicks(uint32_t dualIssueTicks)
    {
        const uint32_t issueScale = 2u - ((cop0_config >> 18u) & 1u);
        const uint32_t ticks = dualIssueTicks * issueScale;
        ee_block_cycle_ticks += ticks;
        ee_block_cycle_active = true;
    }

    [[nodiscard]] ps2x::timing::EeTickDelta commitEeBlockCycles()
    {
        if (!ee_block_cycle_active)
        {
            return {};
        }

        const uint64_t committedCycles =
            std::max<uint64_t>(
                ee_block_cycle_ticks /
                    ps2x::timing::kEeTicksPerCycle,
                1u);
        ee_block_cycle_ticks %=
            static_cast<uint32_t>(ps2x::timing::kEeTicksPerCycle);
        return ps2x::timing::eeCyclesToTicks(committedCycles);
    }

    [[nodiscard]] ps2x::timing::EeTickDelta finishEeBasicBlock()
    {
        // PCSX2's EE recompiler accumulates three-bit fixed-point issue time
        // inside a block. Its block commit is at least one cycle, and then the
        // fractional remainder is discarded when the block ends.
        if (!ee_block_cycle_active)
        {
            return {};
        }

        const uint64_t committedCycles =
            std::max<uint64_t>(
                ee_block_cycle_ticks /
                    ps2x::timing::kEeTicksPerCycle,
                1u);
        ee_block_cycle_ticks = 0u;
        ee_block_cycle_active = false;
        return ps2x::timing::eeCyclesToTicks(committedCycles);
    }

    void resetEeBlockTiming()
    {
        // Reset and state-copy paths are architectural publication
        // boundaries. Never discard completed Random retirements merely
        // because the local block timing accumulator is being cleared.
        finishEeInstruction();
        ee_block_cycle_ticks = 0u;
        ee_block_cycle_active = false;
        ee_performance_issue_pending = false;
        ee_performance_pending_issue = {};
    }

    void resetPostBiosTlbRegisters() noexcept
    {
        cop0_index = 38u;
        cop0_entrylo0 = 0x0006003fu;
        cop0_entrylo1 = 0x0007003fu;
        cop0_pagemask = 0x007fe000u;
        cop0_wired = 31u;
        cop0_entryhi = 0x31800000u;
    }

    R5900Context()
        : R5900Context(InitializationProfile::Deterministic)
    {
    }

    explicit R5900Context(InitializationProfile profile)
    {
        std::memset(this, 0, sizeof(*this));

        // Q is architecturally indeterminate. Preserve the established
        // deterministic synthetic-context value, while direct ELF launch
        // uses the zero observed after the standard BIOS loader handoff.
        vu0_q =
            profile == InitializationProfile::Deterministic ?
                1.0f :
                0.0f;
        enforceVu0RegisterInvariants();

        // Standalone contexts use the architectural Random reset value. The
        // BIOS loader has already populated its wired TLB entries by direct
        // ELF entry, where the standard handoff exposes Random as zero.
        cop0_random =
            profile == InitializationProfile::Deterministic ? 47u : 0u;
        // Direct ELF execution starts after EELOAD has enabled COP0, COP1,
        // and COP2, selected user mode, and enabled the ordinary INTC/DMAC
        // interrupt masks. Synthetic contexts retain the legacy neutral
        // Status value so focused instruction tests can select their mode.
        cop0_status =
            profile == InitializationProfile::PostBiosElf ?
                kPostBiosElfStatus :
                0u;
        if (profile == InitializationProfile::PostBiosElf)
        {
            resetPostBiosTlbRegisters();
        }
        cop0_prid = 0x00002e20; // CPU ID for R5900
        cop0_config = 0x00073443; // Normal post-BIOS EE configuration

        in_delay_slot = false;
        branch_pc = 0;
    }

    void dump() const
    {
        std::ios_base::fmtflags flags = std::cout.flags();
        std::cout << std::hex << std::setfill('0');
        std::cout << "--- R5900 Context Dump ---\n";
        std::cout << "PC: 0x" << std::setw(8) << pc << "\n";
        std::cout << "HI: 0x" << std::setw(8) << hi << " LO: 0x" << std::setw(8) << lo << "\n";
        std::cout << "HI1:0x" << std::setw(8) << hi1 << " LO1:0x" << std::setw(8) << lo1 << "\n";
        std::cout << "SA: 0x" << std::setw(8) << sa << "\n";
        for (int i = 0; i < 32; ++i)
        {
            std::cout << "R" << std::setw(2) << std::dec << i << ": 0x" << std::hex
                      << std::setw(8) << static_cast<uint32_t>(_mm_extract_epi32(r[i], 3))
                      << std::setw(8) << static_cast<uint32_t>(_mm_extract_epi32(r[i], 2)) << "_"
                      << std::setw(8) << static_cast<uint32_t>(_mm_extract_epi32(r[i], 1))
                      << std::setw(8) << static_cast<uint32_t>(_mm_extract_epi32(r[i], 0)) << "\n";
        }
        std::cout << "Status: 0x" << std::setw(8) << cop0_status
                  << " Cause: 0x" << std::setw(8) << cop0_cause
                  << " EPC: 0x" << std::setw(8) << cop0_epc << "\n";
        std::cout << "--- End Context Dump ---\n";
        std::cout.flags(flags); // Restore format flags
    }

    ~R5900Context() = default;
};

inline uint32_t getRegU32(const R5900Context *ctx, int reg)
{
    // Check if reg is valid (0-31)
    if (reg < 0 || reg > 31)
        return 0;
    if (reg == 0)
        return 0;
    return static_cast<uint32_t>(_mm_extract_epi32(ctx->r[reg], 0));
}

inline void setReturnU32(R5900Context *ctx, uint32_t value)
{
    // R5900 sign-extends 32-bit results into 64-bit GPR, even for unsigned values.
    ctx->r[2] = _mm_set_epi64x(0, static_cast<int64_t>(static_cast<int32_t>(value))); // $v0
}

inline void setReturnS32(R5900Context *ctx, int32_t value)
{
    // Signed 32-bit return should be sign-extended when observed as 64-bit.
    ctx->r[2] = _mm_set_epi64x(0, static_cast<int64_t>(value)); // $v0
}

inline void setReturnU64(R5900Context *ctx, uint64_t value)
{
    // Keep both conventions: full 64-bit value in $v0 and high 32-bit in $v1.
    ctx->r[2] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    ctx->r[3] = _mm_set_epi64x(0, static_cast<int64_t>(static_cast<uint32_t>(value >> 32)));
}

inline constexpr uint32_t PS2_PATH_WATCH_ADDR = 0x01EFFFA0u;
inline constexpr uint32_t PS2_PATH_WATCH_BYTES = 0x200u;

extern std::atomic<bool> g_ps2GsDmaWriteTraceEnabled;
void ps2RecordGsDmaWriteTrace(uint32_t guestAddr,
                              uint32_t size,
                              uint32_t writerPc,
                              uint32_t returnAddress);
extern std::atomic<bool> g_ps2GuestWriteTraceEnabled;
void ps2RecordGuestWriteTrace(uint32_t guestAddr,
                              uint32_t size,
                              uint64_t valueLo,
                              uint64_t valueHi,
                              const char *op,
                              uint32_t writerPc,
                              uint32_t returnAddress);

inline uint32_t ps2PathWatchPhysAddr()
{
    return PS2_PATH_WATCH_ADDR & PS2_RAM_MASK;
}

inline uint8_t ps2PathWatchExtractByteFromWrite(uint32_t writeAddr, uint32_t watchAddr, uint64_t valueLo, uint64_t valueHi)
{
    const uint32_t byteIndex = watchAddr - writeAddr;
    if (byteIndex < 8u)
    {
        return static_cast<uint8_t>((valueLo >> (byteIndex * 8u)) & 0xFFu);
    }
    return static_cast<uint8_t>((valueHi >> ((byteIndex - 8u) * 8u)) & 0xFFu);
}

inline void ps2TraceGuestWrite(uint8_t *rdram,
                               uint32_t guestAddr,
                               uint32_t size,
                               uint64_t valueLo,
                               uint64_t valueHi,
                               const char *op,
                               const R5900Context *ctx)
{
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    (void)rdram;
    if (g_ps2GsDmaWriteTraceEnabled.load(std::memory_order_relaxed))
    {
        ps2RecordGsDmaWriteTrace(
            guestAddr, size, ctx ? ctx->pc : 0u, ctx ? getRegU32(ctx, 31) : 0u);
    }
    if (g_ps2GuestWriteTraceEnabled.load(std::memory_order_relaxed))
    {
        ps2RecordGuestWriteTrace(
            guestAddr, size, valueLo, valueHi, op,
            ctx ? ctx->pc : 0u, ctx ? getRegU32(ctx, 31) : 0u);
    }
#else
    (void)rdram;
    (void)guestAddr;
    (void)size;
    (void)valueLo;
    (void)valueHi;
    (void)op;
    (void)ctx;
#endif
}

inline void ps2TraceGuestRangeWrite(uint8_t *rdram,
                                    uint32_t guestAddr,
                                    uint32_t size,
                                    const char *op,
                                    const R5900Context *ctx)
{
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    (void)rdram;
    if (g_ps2GsDmaWriteTraceEnabled.load(std::memory_order_relaxed))
    {
        ps2RecordGsDmaWriteTrace(
            guestAddr, size, ctx ? ctx->pc : 0u, ctx ? getRegU32(ctx, 31) : 0u);
    }
    if (g_ps2GuestWriteTraceEnabled.load(std::memory_order_relaxed))
    {
        ps2RecordGuestWriteTrace(
            guestAddr, size, 0u, 0u, op,
            ctx ? ctx->pc : 0u, ctx ? getRegU32(ctx, 31) : 0u);
    }
#else
    (void)rdram;
    (void)guestAddr;
    (void)size;
    (void)op;
    (void)ctx;
#endif
}

class PS2Runtime
    : private ps2x::ee::IEeSchedulerExecutorHooks
{
public:
    struct IoPaths
    {
        std::filesystem::path elfPath;
        std::filesystem::path elfDirectory;
        std::filesystem::path hostRoot;
        std::filesystem::path cdRoot;
        std::filesystem::path mcRoot;
        std::filesystem::path cdImage;
    };

    enum class HleSifDmaDestination : uint8_t
    {
        IopRam = 0u,
        SyntheticGuest,
    };

    struct HleSifDmaTransfer
    {
        uint32_t src = 0u;
        uint32_t dest = 0u;
        uint32_t size = 0u;
        uint32_t iopOffset = 0u;
        HleSifDmaDestination destination =
            HleSifDmaDestination::IopRam;
    };

    static constexpr size_t kMaximumHleSifDmaTransfers = 32u;
    static constexpr size_t kHleSifDmaQueueCapacity = 32u;

    struct HleSifDmaSubmission
    {
        std::array<
            HleSifDmaTransfer,
            kMaximumHleSifDmaTransfers>
            transfers{};
        uint32_t transferId = 0u;
        uint32_t transferCount = 0u;
    };

    PS2Runtime();
    explicit PS2Runtime(PS2RuntimeConfiguration configuration);
    ~PS2Runtime();

    bool initialize(const char *title = "PS2 Game");
    bool syncCoreSubsystems();
    [[nodiscard]] GsCommandResult submitGsCommand(
        GsCommandPayload payload,
        uint64_t publicationToken = 0u);
    // Guest execution is the sole hot GS producer. This entry records the
    // authoritative EE timeline; control/UI callers use submitGsCommand and
    // inherit the last safely published guest tick.
    [[nodiscard]] GsCommandResult submitEeGsCommand(
        GsCommandPayload payload,
        uint64_t publicationToken = 0u);
    [[nodiscard]] bool usesAsyncGsExecution() const noexcept
    {
        return m_gsAsyncEnabled;
    }
    [[nodiscard]] uint64_t
    lastCompletedGsFieldSequence() const noexcept;
    [[nodiscard]] GsAsyncRuntimeStatistics
    gsAsyncStatistics() const;
    [[nodiscard]] Vu1ExecutionMode
    vu1ExecutionMode() const noexcept
    {
        return m_vu1ExecutionMode;
    }
    [[nodiscard]] ThreadedVu1ExecutorStatistics
    vu1OwnerStatistics() const;
    [[nodiscard]] Vu1AsyncRuntimeStatistics
    vu1AsyncStatistics() const;
    [[nodiscard]] Vu1CoarseRuntimeStatistics
    vu1CoarseStatistics() const;
    [[nodiscard]] std::shared_ptr<const Vu1Snapshot>
    snapshotVu1Owner();
    // Debugger observation must not change guest-visible VU timing. In coarse
    // mode this may inspect owner state after privately completed work while
    // leaving the scheduled publication, Busy state, VIF wait, and PATH1
    // effects untouched. Use snapshotVu1Owner for an architectural barrier.
    [[nodiscard]] std::shared_ptr<const Vu1Snapshot>
    debugSnapshotVu1Owner();
    [[nodiscard]] VuProgressSnapshot
    vu1ProgressSnapshot() const;
    void setVu1ProgressTrackingEnabled(bool enabled);
    [[nodiscard]] ps2x::timing::EeTick currentEeTick() const noexcept;
    void resetEeTiming(R5900Context *context = nullptr);
    void setRealtimeVSyncPacingEnabled(bool enabled);
    bool loadELF(const std::string &elfPath);
    void run();

    void setIopPluginSearchPaths(std::vector<std::filesystem::path> paths);
    [[nodiscard]] ps2x::iop::DebugSnapshot iopDebugSnapshot() const;

    using DebugUiCallback = void (*)(PS2Runtime &runtime, void *userData);
    void setDebugUiCallbacks(DebugUiCallback initCallback,
                             DebugUiCallback drawCallback,
                             DebugUiCallback shutdownCallback,
                             void *userData);

    using RecompiledFunction = void (*)(uint8_t *, R5900Context *, PS2Runtime *);

    // Dense dispatch entries are either empty or contain both policies.
    // Runtime installation rejects partial pairs so a mode transition can
    // never turn an existing guest target into a missing function.
    struct RecompiledFunctionPair
    {
        RecompiledFunction fast = nullptr;
        RecompiledFunction precise = nullptr;

        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return fast == nullptr && precise == nullptr;
        }

        [[nodiscard]] constexpr bool complete() const noexcept
        {
            return fast != nullptr && precise != nullptr;
        }

        [[nodiscard]] constexpr RecompiledFunction select(
            EeArchitecturalObservationMode mode) const noexcept
        {
            return mode == EeArchitecturalObservationMode::Fast
                       ? fast
                       : precise;
        }

        [[nodiscard]] static constexpr RecompiledFunctionPair identical(
            RecompiledFunction function) noexcept
        {
            return {function, function};
        }
    };

    enum class GuestBranchKind
    {
        DirectJump,
        DirectCall,
        IndirectJump,
        IndirectCall,
        Return,
    };

    enum class MissingFunctionPolicy : uint32_t
    {
        // Strict mode for tests/CI: log the bad target and request the runtime to stop.
        Stop = 0,

        // Debug mode: log once, leave ctx->pc on the bad target, and let the caller unwind.
        ContinueToTarget = 1,

        // Debug mode: same as ContinueToTarget, but triggers a debugger break once on MSVC.
        BreakOnce = 2,

        // Escape hatch only: skip missing calls by returning to fallthrough (it can hide guest bugs)
        SkipCallDebug = 3,
    };

    enum class DebugMemoryAccess : uint8_t
    {
        Read = 1u,
        Write = 2u,
        ReadWrite = 3u,
    };

    struct DebugStopInfo
    {
        bool completed = false;
        std::string reason;
        uint32_t pc = 0u;
        uint64_t sequence = 0u;
        std::string pauseSource;
        std::string pauseWriter;
        uint64_t controlGeneration = 0u;
        uint64_t observedControlGeneration = 0u;
        uint64_t staleBoundaryStops = 0u;
    };

    struct DebugWatchpoint
    {
        uint64_t id = 0u;
        uint32_t start = 0u;
        uint32_t size = 0u;
        DebugMemoryAccess access = DebugMemoryAccess::Write;
    };

    struct DebugRuntimeProgress
    {
        uint64_t dispatches = 0u;
        uint64_t eeInstructions = 0u;
        uint64_t vsyncFields = 0u;
        uint64_t mpegPicturesServed = 0u;
        uint64_t mpegUniquePicturesServed = 0u;
        uint64_t mpegRepeatedPicturesServed = 0u;
        uint32_t pc = 0u;
        uint32_t ra = 0u;
        uint32_t sp = 0u;
        uint32_t gp = 0u;
        uint32_t guestExecutionWaiters = 0u;
        uint64_t guestExecutionHandoffTimeouts = 0u;
    };

    static constexpr size_t
        kEeThreadDiagnosticPriorityCount = 128u;
    static constexpr size_t
        kEeThreadDiagnosticThreadCount = 256u;
    static constexpr uint64_t
        kEeThreadDiagnosticRotationSequenceHashSeed =
            14695981039346656037ull;

    struct DebugEeThreadDiagnostics
    {
        bool enabled = false;
        uint64_t guestLockRequests = 0u;
        uint64_t guestLockAcquisitions = 0u;
        uint64_t guestLockContentions = 0u;
        uint64_t outerGuestExecutionAcquisitions = 0u;
        uint64_t guestContextChanges = 0u;
        uint64_t handoffNotifications = 0u;
        uint64_t handoffWaitRequests = 0u;
        uint64_t handoffWaitFastPaths = 0u;
        uint64_t handoffCvWaits = 0u;
        uint64_t handoffCompletions = 0u;
        uint64_t handoffTimeouts = 0u;
        uint64_t yieldRequests = 0u;
        uint64_t deferredYields = 0u;
        uint64_t hostThreadYields = 0u;
        uint64_t requestedGuestSwitches = 0u;
        uint64_t guestSwitchCvWaits = 0u;
        uint64_t completedGuestSwitches = 0u;
        uint64_t guestSwitchTimeouts = 0u;
        uint64_t rotationRequests = 0u;
        uint64_t acceptedRotationRequests = 0u;
        uint64_t rejectedRotationRequests = 0u;
        uint64_t priorityZeroRotationRequests = 0u;
        uint64_t untrackedThreadRotationRequests = 0u;
        uint64_t acceptedRotationSequenceHash =
            kEeThreadDiagnosticRotationSequenceHashSeed;
        std::array<
            uint64_t,
            kEeThreadDiagnosticPriorityCount>
            acceptedRotationsByPriority{};
        std::array<
            uint64_t,
            kEeThreadDiagnosticThreadCount>
            acceptedRotationsByThread{};
    };

    struct DebugEeTiming
    {
        uint64_t currentTick = 0u;
        uint64_t currentCycle = 0u;
        uint32_t localBlockTicks = 0u;
        bool localBlockActive = false;
        bool contextBound = false;
    };

    struct DebugVu1Timing
    {
        uint64_t currentTick = 0u;
        uint64_t generation = 0u;
        uint64_t lastAdvancedTick = 0u;
        uint64_t totalAdvancedCycles = 0u;
        uint64_t eventGeneration = 0u;
        uint64_t eventDeadlineTick = 0u;
        uint32_t pc = 0u;
        bool active = false;
        bool vifWaitingForVu = false;
        bool eventPending = false;
    };

    struct DebugVuProgramCacheDiagnostics
    {
        uint64_t hits = 0u;
        uint64_t misses = 0u;
        uint64_t compilations = 0u;
        uint64_t invalidations = 0u;
        uint64_t invalidatedPrograms = 0u;
        uint64_t generationRetentions = 0u;
        uint64_t retainedPrograms = 0u;
        uint64_t crossGenerationHits = 0u;
        uint64_t evictionFlushes = 0u;
        uint64_t selectiveEvictions = 0u;
        uint64_t evictedPrograms = 0u;
        uint64_t manualFlushes = 0u;
        uint64_t rejectedPrograms = 0u;
        uint64_t linkResolutions = 0u;
        uint64_t linkResolutionFailures = 0u;
        uint64_t linkInvalidations = 0u;
        uint64_t generatedBytes = 0u;
        uint64_t compilationNanoseconds = 0u;
        uint64_t residentPrograms = 0u;
        uint64_t residentExecutableBytes = 0u;
        uint64_t highWaterPrograms = 0u;
        uint64_t highWaterExecutableBytes = 0u;
        uint64_t maximumPrograms = 0u;
        uint64_t maximumExecutableBytes = 0u;
    };

    struct DebugVuRecompilerDiagnostics
    {
        struct BlockJitRegistration
        {
            uint64_t generation = 0u;
            uint64_t codeIndex = 0u;
        };

        struct BlockOpcodeCount
        {
            std::string name;
            uint64_t operations = 0u;
        };

        struct BlockProfile
        {
            uint64_t compilationIdentity = 0u;
            uint64_t codeContentIdentity = 0u;
            uint64_t compilationGeneration = 0u;
            uint64_t nativeAddress = 0u;
            uint64_t nativeBytes = 0u;
            uint32_t entryPc = 0u;
            uint32_t firstPc = 0u;
            uint32_t lastPc = 0u;
            uint32_t blockPairs = 0u;
            uint32_t fixedCycles = 0u;
            std::string compilationMode;
            std::string blockForm;
            std::string blockExit;
            bool resident = false;
            uint64_t executions = 0u;
            uint64_t guestPairs = 0u;
            uint64_t fullBudgetEntries = 0u;
            uint64_t boundedEntries = 0u;
            uint64_t linkedEdges = 0u;
            uint64_t helperBarriers = 0u;
            uint32_t residentVfRegisters = 0u;
            uint32_t residentVfDirtyRegisters = 0u;
            uint32_t residentVfDirtyLanes = 0u;
            uint32_t maximumLiveVfRegisters = 0u;
            uint32_t vfAccesses = 0u;
            uint32_t allocatedVfAccesses = 0u;
            uint64_t vfRegisterLoads = 0u;
            uint64_t vfRegisterStores = 0u;
            uint64_t vfRegisterSpills = 0u;
            uint64_t vfRegisterReloads = 0u;
            std::array<uint64_t, 4u>
                registerMaterializations{};
            std::array<uint64_t, 10u> exitReasons{};
            std::vector<BlockOpcodeCount> opcodeCounts;
            std::vector<BlockJitRegistration>
                jitRegistrations;
        };

        bool blockLinkingEnabled = false;
        bool blockBudgetGuardsEnabled = false;
        bool blockLocalVfRegistersEnabled = false;
        bool blockLocalVfRegistersAutomatic = false;
        bool inlineXgkickEnabled = false;
        uint64_t nativeEntries = 0u;
        uint64_t nativeBlocks = 0u;
        uint64_t nativePairs = 0u;
        uint64_t instrumentedNativeEntries = 0u;
        uint64_t instrumentedNativeBlocks = 0u;
        uint64_t instrumentedNativePairs = 0u;
        uint64_t inlinePairs = 0u;
        uint64_t helperPairs = 0u;
        uint64_t linkedEdges = 0u;
        uint64_t slowLinkExits = 0u;
        uint64_t resolvedLinks = 0u;
        uint64_t abandonedLinkResolutions = 0u;
        uint64_t incompatibleLinkExits = 0u;
        uint64_t fullBlockGuards = 0u;
        uint64_t preciseTailEntries = 0u;
        uint64_t fullGuardPairs = 0u;
        uint64_t preciseTailPairs = 0u;
        uint64_t budgetGuardComparisons = 0u;
        uint64_t nativeBudgetExits = 0u;
        uint64_t blockCompletes = 0u;
        uint64_t cycleBudgetExits = 0u;
        uint64_t xgkickExits = 0u;
        uint64_t xgkickAdvanceHelperCalls = 0u;
        uint64_t unsupportedExits = 0u;
        uint64_t codeInvalidationExits = 0u;
        uint64_t faultExits = 0u;
        std::array<uint64_t, 10u> nativeExitReasons{};
        uint64_t interpreterInstrumentationFallbacks = 0u;
        uint64_t interpreterFallbackPairs = 0u;
        uint64_t codeImageIdentities = 0u;
        uint64_t codeImageReuses = 0u;
        uint64_t codeImageCatalogEvictions = 0u;
        uint64_t jitDumpRegistrations = 0u;
        uint64_t jitDumpFailures = 0u;
        std::string lastJitDiagnostic;
        bool blockProfileEnabled = false;
        uint64_t blockProfileMaximumRecords = 0u;
        uint64_t blockProfileDroppedRecords = 0u;
        std::vector<BlockProfile> blockProfiles;
    };

    struct DebugVuVerifyDiagnostics
    {
        uint64_t runs = 0u;
        uint64_t comparedPairs = 0u;
        uint64_t publishedPairs = 0u;
        uint64_t publishedPath1Packets = 0u;
        uint64_t mismatches = 0u;
        std::string lastMismatch;
    };

    struct DebugVuBackendDiagnostics
    {
        uint64_t snapshotSequence = 0u;
        uint64_t issuedCycles = 0u;
        uint32_t pc = 0u;
        VuExitReason lastExitReason = VuExitReason::Inactive;
        bool captured = false;
        bool cacheCreated = false;
        bool recompilerCreated = false;
        DebugVuProgramCacheDiagnostics cache{};
        DebugVuRecompilerDiagnostics recompiler{};
        DebugVuVerifyDiagnostics verify{};
    };

    enum class DebugEeEventDeviceKind : uint8_t
    {
        None = 0u,
        Cop0Performance,
        Cop0Timer,
        VSync,
        EeCounters,
        GifDma,
        HleSifDma,
        Vif0Dma,
        Vif1Dma,
        Vu0,
        Vu1,
        ScratchpadDma,
        FromIpuDma,
        ToIpuDma,
    };

    [[nodiscard]] static constexpr const char *
    debugEeEventDeviceKindName(
        DebugEeEventDeviceKind kind) noexcept
    {
        switch (kind)
        {
        case DebugEeEventDeviceKind::Cop0Performance:
            return "cop0_performance";
        case DebugEeEventDeviceKind::Cop0Timer:
            return "cop0_timer";
        case DebugEeEventDeviceKind::VSync:
            return "vsync";
        case DebugEeEventDeviceKind::EeCounters:
            return "ee_counters";
        case DebugEeEventDeviceKind::GifDma:
            return "gif_dma";
        case DebugEeEventDeviceKind::HleSifDma:
            return "hle_sif_dma";
        case DebugEeEventDeviceKind::Vif0Dma:
            return "vif0_dma";
        case DebugEeEventDeviceKind::Vif1Dma:
            return "vif1_dma";
        case DebugEeEventDeviceKind::Vu0:
            return "vu0";
        case DebugEeEventDeviceKind::Vu1:
            return "vu1";
        case DebugEeEventDeviceKind::ScratchpadDma:
            return "scratchpad_dma";
        case DebugEeEventDeviceKind::FromIpuDma:
            return "from_ipu_dma";
        case DebugEeEventDeviceKind::ToIpuDma:
            return "to_ipu_dma";
        case DebugEeEventDeviceKind::None:
            return "none";
        }
        return "unknown";
    }

    struct DebugEeEventDeviceState
    {
        DebugEeEventDeviceKind kind =
            DebugEeEventDeviceKind::None;
        uint64_t operationGeneration = 0u;
        uint64_t lastAdvancedTick = 0u;
        uint64_t totalAdvancedCycles = 0u;
        uint32_t phase = 0u;
        uint32_t stall = 0u;
        uint32_t tadr = 0u;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        uint32_t tagsProcessed = 0u;
        uint32_t pc = 0u;
        bool active = false;
    };

    struct DebugEeEventSlot
    {
        ps2x::timing::EeEventSource source =
            ps2x::timing::EeEventSource::Cop0Performance;
        uint64_t deadlineTick = 0u;
        uint64_t generation = 0u;
        uint64_t sequence = 0u;
        DebugEeEventDeviceState device{};
        bool pending = false;
    };

    struct DebugEeScheduler
    {
        uint64_t currentTick = 0u;
        uint64_t nextDeadlineTick = 0u;
        bool hasNextDeadline = false;
        std::array<DebugEeEventSlot,
                   ps2x::timing::kEeEventSourceCount>
            slots{};
        ps2x::timing::EeEventSchedulerStatistics statistics{};
    };

    struct DebugEeEventEntry
    {
        uint64_t sequence = 0u;
        ps2x::timing::EeEventSource source =
            ps2x::timing::EeEventSource::Cop0Performance;
        uint64_t eventSequence = 0u;
        uint64_t generation = 0u;
        uint64_t scheduledTick = 0u;
        uint64_t serviceTick = 0u;
        uint64_t latenessTicks = 0u;
        uint64_t followupTick = 0u;
        uint32_t eePc = 0u;
        DebugEeEventDeviceState deviceBefore{};
        DebugEeEventDeviceState deviceAfter{};
        bool hasFollowup = false;
        bool rescheduled = false;
    };

    struct DebugEeEventTrace
    {
        bool enabled = false;
        bool triggered = false;
        bool stopOnFull = false;
        std::optional<uint32_t> triggerEePc;
        uint64_t totalEntries = 0u;
        uint64_t droppedEntries = 0u;
        std::vector<DebugEeEventEntry> entries;
    };

    enum class DebugVu1TimingEventKind : uint8_t
    {
        InvocationStart = 0u,
        InvocationComplete,
        Cop2ConditionObservation,
        VifBusyObservation,
        ForcedBarrier,
        Path1Packet,
        VifResume,
    };

    [[nodiscard]] static constexpr const char *
    debugVu1TimingEventKindName(
        DebugVu1TimingEventKind kind) noexcept
    {
        switch (kind)
        {
        case DebugVu1TimingEventKind::InvocationStart:
            return "invocation_start";
        case DebugVu1TimingEventKind::InvocationComplete:
            return "invocation_complete";
        case DebugVu1TimingEventKind::Cop2ConditionObservation:
            return "cop2_condition_observation";
        case DebugVu1TimingEventKind::VifBusyObservation:
            return "vif_busy_observation";
        case DebugVu1TimingEventKind::ForcedBarrier:
            return "forced_barrier";
        case DebugVu1TimingEventKind::Path1Packet:
            return "path1_packet";
        case DebugVu1TimingEventKind::VifResume:
            return "vif_resume";
        }
        return "unknown";
    }

    struct DebugVu1TimingEntry
    {
        uint64_t sequence = 0u;
        DebugVu1TimingEventKind kind =
            DebugVu1TimingEventKind::InvocationStart;
        Vu1ExecutionMode mode = Vu1ExecutionMode::Inline;
        Vu1InvocationKind invocationKind =
            Vu1InvocationKind::Mscal;
        uint64_t invocation = 0u;
        uint64_t eeTick = 0u;
        uint64_t eeInstructions = 0u;
        uint64_t vsyncField = 0u;
        uint64_t executionGeneration = 0u;
        uint64_t eventGeneration = 0u;
        uint64_t startTick = 0u;
        uint64_t deadlineTick = 0u;
        uint64_t totalAdvancedCycles = 0u;
        uint64_t dmaStarts = 0u;
        uint64_t gsCsr = 0u;
        uint64_t path1Packet = 0u;
        uint64_t path1CycleOffset = 0u;
        uint64_t path1Digest = 0u;
        uint64_t architecturalStateHash = 0u;
        uint64_t inputExecutionStateHash = 0u;
        uint64_t inputMicroMemoryHash = 0u;
        uint64_t inputDataMemoryHash = 0u;
        uint64_t inputVifStateHash = 0u;
        uint64_t executionStateHash = 0u;
        uint64_t microMemoryHash = 0u;
        uint64_t dataMemoryHash = 0u;
        uint64_t vifStateHash = 0u;
        uint32_t estimatedCycles = 0u;
        uint32_t executedCycles = 0u;
        uint32_t startPc = 0u;
        uint32_t top = 0u;
        uint32_t itop = 0u;
        uint32_t eePc = 0u;
        uint32_t vuPc = 0u;
        uint32_t path1Bytes = 0u;
        uint32_t vifStat = 0u;
        uint32_t vifCode = 0u;
        uint32_t intcStatus = 0u;
        uint32_t intcMask = 0u;
        VuExitReason exitReason = VuExitReason::Inactive;
        DebugEeEventDeviceState vif1Dma{};
        bool hasExitReason = false;
        bool active = false;
        bool busy = false;
        bool observedValue = false;
        bool schedulerEvent = false;
        bool ownerReady = false;
        bool hasBarrierCommandType = false;
        Vu1CommandType barrierCommandType =
            Vu1CommandType::Barrier;
        bool vifWaitingBefore = false;
        bool vifWaitingAfter = false;
    };

    struct DebugVu1TimingTrace
    {
        bool enabled = false;
        bool stopOnFull = false;
        uint64_t totalEntries = 0u;
        uint64_t droppedEntries = 0u;
        std::vector<DebugVu1TimingEntry> entries;
    };

    struct DebugBranchEntry
    {
        uint64_t sequence = 0u;
        uint32_t pc = 0u;
    };

    struct DebugFaultInfo
    {
        bool active = false;
        uint64_t sequence = 0u;
        std::string type;
        std::string operation;
        GuestBranchKind branchKind = GuestBranchKind::IndirectJump;
        uint32_t sourcePc = 0u;
        uint32_t targetPc = 0u;
        uint32_t pc = 0u;
        uint32_t ra = 0u;
        uint32_t sp = 0u;
        uint32_t gp = 0u;
        uint32_t a0 = 0u;
        uint32_t a1 = 0u;
        uint32_t v0 = 0u;
        uint32_t v1 = 0u;
        bool codeRegion = false;
        MissingFunctionPolicy policy = MissingFunctionPolicy::ContinueToTarget;
    };

    struct DebugVu0SyncEntry
    {
        uint64_t sequence = 0u;
        uint64_t invocation = 0u;
        uint64_t invocationInstruction = 0u;
        uint64_t eeCycleTicks = 0u;
        uint64_t vsyncTick = 0u;
        uint64_t vuCycleTicks = 0u;
        uint64_t nextEventCycleTicks = 0u;
        uint32_t eePc = 0u;
        uint32_t cycleBudget = 0u;
        uint32_t vuPcBefore = 0u;
        uint32_t vuPcAfter = 0u;
        uint32_t pendingVfMask = 0u;
        uint16_t pendingViMask = 0u;
        uint16_t contextVi1 = 0u;
        uint16_t contextVi2 = 0u;
        uint16_t vuVi1Before = 0u;
        uint16_t vuVi2Before = 0u;
        uint16_t vuVi1After = 0u;
        uint16_t vuVi2After = 0u;
        bool interlocked = false;
        bool blockBoundary = false;
        bool eventDue = false;
        bool activeBefore = false;
        bool activeAfter = false;
    };

    struct DebugVu0SyncTrace
    {
        bool enabled = false;
        bool triggered = false;
        bool stopOnFull = false;
        std::optional<uint32_t> triggerEePc;
        std::optional<uint64_t> triggerInvocation;
        bool hasTriggerSchedulerSnapshot = false;
        uint64_t triggerVsyncTick = 0u;
        DebugEeScheduler triggerScheduler{};
        uint64_t totalEntries = 0u;
        uint64_t droppedEntries = 0u;
        std::vector<DebugVu0SyncEntry> entries;
    };

    struct DebugVu0InstructionEntry
    {
        uint64_t sequence = 0u;
        uint64_t invocation = 0u;
        uint64_t invocationInstruction = 0u;
        uint64_t codeHashFnv1a64 = 0u;
        uint64_t dataHashFnv1a64 = 0u;
        uint32_t pc = 0u;
        uint32_t lower = 0u;
        uint32_t upper = 0u;
        std::array<uint16_t, 16> vi{};
        std::array<uint32_t, 32 * 4> vf{};
        std::array<uint32_t, 4> acc{};
        uint32_t q = 0u;
        uint32_t p = 0u;
        uint32_t i = 0u;
        uint32_t status = 0u;
        uint32_t mac = 0u;
        uint32_t clip = 0u;
        uint32_t top = 0u;
        uint32_t itop = 0u;
        uint32_t branchTarget = 0u;
        uint32_t branchDelay = 0u;
        uint16_t viBackupValue = 0u;
        uint8_t viBackupCycles = 0u;
        uint8_t viBackupRegister = 0u;
        bool branchPending = false;
        bool invocationStart = false;
    };

    struct DebugVu0InstructionTrace
    {
        bool enabled = false;
        bool triggered = false;
        bool stopOnFull = false;
        std::optional<uint32_t> triggerEePc;
        std::optional<uint64_t> triggerInvocation;
        uint64_t totalEntries = 0u;
        uint64_t droppedEntries = 0u;
        std::vector<DebugVu0InstructionEntry> entries;
    };

    class AsyncCallbackInvocationScope;

    class GuestExecutionScope
    {
    public:
        explicit GuestExecutionScope(
            PS2Runtime *runtime,
            R5900Context *context = nullptr,
            int selectionHint = -1) noexcept;
        ~GuestExecutionScope();

        GuestExecutionScope(const GuestExecutionScope &) = delete;
        GuestExecutionScope &operator=(const GuestExecutionScope &) = delete;

    private:
        friend class PS2Runtime;
        friend class AsyncCallbackInvocationScope;

        PS2Runtime *m_runtime = nullptr;
        R5900Context *m_context = nullptr;
        R5900Context *m_previousContext = nullptr;
        int m_previousThreadId = 1;
    };

    // An asynchronous EE callback executes from a separate register context,
    // but it still belongs to the continuation interrupted at the current
    // boundary. This scope initializes the private callback context from a
    // complete owned snapshot and retains the continuation's identity for the
    // duration of the callback. The callback executor installs its entry PC
    // and ABI argument registers after constructing the scope. Active and
    // explicit synchronous callbacks descend from their owned
    // interrupted/caller stack, while a truly idle delivery uses the BIOS
    // interrupt stack.
    class AsyncCallbackInvocationScope
    {
    public:
        AsyncCallbackInvocationScope(
            PS2Runtime *runtime,
            R5900Context *callbackContext);
        AsyncCallbackInvocationScope(
            PS2Runtime *runtime,
            R5900Context *callbackContext,
            R5900Context *continuationContext,
            int continuationThreadId);
        ~AsyncCallbackInvocationScope();

        [[nodiscard]] uint32_t stackTop() const noexcept
        {
            return m_stackTop;
        }

        [[nodiscard]] int ownerThreadId() const noexcept
        {
            return m_ownerThreadId;
        }

        AsyncCallbackInvocationScope(
            const AsyncCallbackInvocationScope &) = delete;
        AsyncCallbackInvocationScope &operator=(
            const AsyncCallbackInvocationScope &) = delete;

    private:
        friend class PS2Runtime;

        PS2Runtime *m_runtime = nullptr;
        GuestExecutionScope m_guestExecution;
        R5900Context m_interruptedContext{};
        R5900Context *m_continuationContext = nullptr;
        std::shared_ptr<ThreadInfo> m_continuationOwner;
        uint32_t m_continuationGeneration = 0u;
        uint32_t m_stackTop = 0u;
        uint32_t m_depth = 0u;
        int m_ownerThreadId = 0;
        bool m_idle = true;
        bool m_active = false;
    };

    class GuestExecutionReleaseScope
    {
    public:
        explicit GuestExecutionReleaseScope(PS2Runtime *runtime) noexcept;
        ~GuestExecutionReleaseScope();

        GuestExecutionReleaseScope(const GuestExecutionReleaseScope &) = delete;
        GuestExecutionReleaseScope &operator=(const GuestExecutionReleaseScope &) = delete;

    private:
        PS2Runtime *m_runtime = nullptr;
        R5900Context *m_context = nullptr;
        int m_threadId = 1;
        uint32_t m_depth = 0u;
    };

    // Transitional worker paths must never block runtime shutdown while
    // waiting to enter legacy guest execution. Normal guest dispatch uses
    // GuestExecutionScope; this bounded form lets a worker re-check its
    // runtime-owned cancellation state between acquisition attempts.
    class GuestExecutionTryScope
    {
    public:
        GuestExecutionTryScope(
            PS2Runtime *runtime,
            R5900Context *context,
            std::chrono::milliseconds timeout) noexcept;
        ~GuestExecutionTryScope();

        GuestExecutionTryScope(
            const GuestExecutionTryScope &) = delete;
        GuestExecutionTryScope &operator=(
            const GuestExecutionTryScope &) = delete;

        [[nodiscard]] bool acquired() const noexcept
        {
            return m_acquired;
        }

    private:
        PS2Runtime *m_runtime = nullptr;
        R5900Context *m_context = nullptr;
        R5900Context *m_previousContext = nullptr;
        int m_previousThreadId = 1;
        bool m_acquired = false;
    };

    class DeferredGuestYieldScope
    {
    public:
        explicit DeferredGuestYieldScope(bool &pendingOut) noexcept;
        ~DeferredGuestYieldScope();

        DeferredGuestYieldScope(const DeferredGuestYieldScope &) = delete;
        DeferredGuestYieldScope &operator=(const DeferredGuestYieldScope &) = delete;

    private:
        bool &m_pendingOut;
    };

    bool replaceFunction(
        uint32_t address,
        RecompiledFunctionPair functions);
    bool replaceFunction(uint32_t address, RecompiledFunction func);
    // TODO remove this later need to update all tests
    bool registerFunction(
        uint32_t address,
        RecompiledFunctionPair functions);
    bool registerFunction(uint32_t address, RecompiledFunction func);
    [[nodiscard]] static EeArchitecturalObservationMode
    architecturalObservationMode(
        const R5900Context *ctx) noexcept;
    // Generated code captures the derived mode immediately before a
    // statically known BPC/PCCR write, then calls this only after the changing
    // instruction has completed. On a real mode change, finish the resolved
    // boundary and advance the same epoch used by stackless checkpoints so
    // nested generated calls unwind.
    [[nodiscard]] bool completeEeObservationModeTransition(
        uint8_t *rdram,
        R5900Context *ctx,
        EeArchitecturalObservationMode previousMode);
    [[nodiscard]] bool completeEeObservationModeTransitionAtGuestBranch(
        uint8_t *rdram,
        R5900Context *ctx,
        EeArchitecturalObservationMode previousMode,
        GuestBranchKind kind);
    // Compatibility lookup is deliberately precise. Runtime-owned execution
    // paths pass their context and select the pair once at the transfer.
    RecompiledFunction lookupFunction(uint32_t address);
    RecompiledFunction lookupFunction(
        uint32_t address,
        const R5900Context *ctx);
    bool hasFunction(uint32_t address) const;
    static std::string formatGuestPc(uint32_t address);
    bool dispatchGuestBranch(uint8_t *rdram,
                             R5900Context *ctx,
                             uint32_t targetPc,
                             uint32_t sourcePc,
                             uint32_t fallthroughPc,
                             GuestBranchKind kind,
                             const char *debugName);
    // Generated code has already validated the resolved target with its
    // compile-time observation policy before servicing the block boundary.
    // Preserve that policy through the transfer without repeating the fetch
    // breakpoint check or deriving the mode again inside the dispatcher.
    bool dispatchValidatedGuestBranch(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t targetPc,
        uint32_t sourcePc,
        uint32_t fallthroughPc,
        GuestBranchKind kind,
        const char *debugName,
        EeArchitecturalObservationMode mode);
    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    bool dispatchValidatedGuestBranchPolicy(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t targetPc,
        uint32_t sourcePc,
        uint32_t fallthroughPc,
        GuestBranchKind kind,
        const char *debugName)
    {
        return dispatchValidatedGuestBranch(
            rdram,
            ctx,
            targetPc,
            sourcePc,
            fallthroughPc,
            kind,
            debugName,
            Mode);
    }
    [[nodiscard]] bool ValidateGuestBranchTarget(
        R5900Context *ctx,
        uint32_t targetPc,
        GuestBranchKind kind);
    [[nodiscard]] bool ValidateGuestBranchTargetWithoutObservation(
        R5900Context *ctx,
        uint32_t targetPc,
        GuestBranchKind kind);
    void reportMissingFunction(uint8_t *rdram,
                               R5900Context *ctx,
                               uint32_t targetPc,
                               uint32_t sourcePc,
                               GuestBranchKind kind,
                               const char *debugName);
    void setMissingFunctionPolicy(MissingFunctionPolicy policy);
    MissingFunctionPolicy missingFunctionPolicy() const;
    void resetMissingFunctionReportOnce();

    [[nodiscard]] IoPaths ioPaths() const;
    void configureIoPaths(const IoPaths &paths);
    void configureIoPathsFromElf(const std::string &elfPath);

    [[noreturn]] void SignalException(R5900Context *ctx, PS2Exception exception);
    void RequireCoprocessorUsable(R5900Context *ctx, uint32_t coprocessor);
    [[noreturn]] void SignalMemoryException(R5900Context *ctx,
                                            PS2Exception exception,
                                            uint32_t badVAddr);
    void ValidateInstructionFetch(
        R5900Context *ctx,
        uint32_t virtualAddress);
    PS2X_EE_MEMORY_POLICY_NOINLINE
    void ValidateInstructionFetchWithoutObservation(
        R5900Context *ctx,
        uint32_t virtualAddress);
    void CheckEeInstructionBreakpoint(
        R5900Context *ctx,
        uint32_t virtualAddress);
    [[nodiscard]] bool EeDataBreakpointEnabled(
        const R5900Context *ctx,
        bool write) const noexcept;
    [[nodiscard]] bool CheckEeDataAddressBreakpoint(
        R5900Context *ctx,
        uint32_t virtualAddress,
        bool write);
    void CheckEeDataValueBreakpoint(
        R5900Context *ctx,
        uint32_t virtualAddress,
        uint32_t value,
        bool write);
    void SignalBusError(R5900Context *ctx,
                        PS2BusErrorAccess access,
                        uint32_t physicalAddress);
    void recordEeInstructionIssue(
        R5900Context *ctx,
        ps2x::performance::EeInstructionIssueProfile profile);
    void recordEeInstructionCompletion(
        R5900Context *ctx,
        uint32_t completionEvents);
    void recordEePredictedBranch(
        R5900Context *ctx,
        uint32_t branchPc);
    void recordEeConditionalBranch(
        R5900Context *ctx,
        uint32_t branchPc,
        bool taken);
    // Cold precise-mode entry points used by generated policy templates. The
    // fast instantiation does not reference these functions or its descriptor
    // data at all.
    void observeEeInstructionBeginPrecise(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction,
        bool breakpointAlreadyChecked);
    void observeEeInstructionCompletePrecise(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction);
    [[nodiscard]] bool observeEeDataAddressPrecise(
        R5900Context *ctx,
        uint32_t virtualAddress,
        bool write);
    void observeEeDataValuePrecise(
        R5900Context *ctx,
        uint32_t virtualAddress,
        uint32_t value,
        bool write);
    void observeEePredictedBranchPrecise(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction);
    void observeEeConditionalBranchPrecise(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction,
        bool taken);

    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void validateEeInstructionFetchPolicy(
        R5900Context *ctx,
        uint32_t virtualAddress)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            ValidateInstructionFetch(ctx, virtualAddress);
        }
        else
        {
            ValidateInstructionFetchWithoutObservation(
                ctx, virtualAddress);
        }
    }

    template <EeArchitecturalObservationMode Mode>
    [[nodiscard]] PS2X_EE_OBSERVATION_POLICY_INLINE
    bool validateEeGuestBranchTargetPolicy(
        R5900Context *ctx,
        uint32_t target,
        GuestBranchKind kind)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            return ValidateGuestBranchTarget(
                ctx, target, kind);
        }
        return ValidateGuestBranchTargetWithoutObservation(
            ctx, target, kind);
    }

    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void observeEeInstructionBeginPolicy(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction,
        bool breakpointAlreadyChecked)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            observeEeInstructionBeginPrecise(
                ctx, instruction, breakpointAlreadyChecked);
        }
    }

    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void observeEeInstructionCompletePolicy(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            observeEeInstructionCompletePrecise(
                ctx, instruction);
        }
    }

    template <EeArchitecturalObservationMode Mode>
    [[nodiscard]] PS2X_EE_OBSERVATION_POLICY_INLINE
    bool observeEeDataAddressPolicy(
        R5900Context *ctx,
        uint32_t virtualAddress,
        bool write)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            return observeEeDataAddressPrecise(
                ctx, virtualAddress, write);
        }
        return false;
    }

    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void observeEeDataValuePolicy(
        R5900Context *ctx,
        uint32_t virtualAddress,
        uint32_t value,
        bool write)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            observeEeDataValuePrecise(
                ctx, virtualAddress, value, write);
        }
    }

    template <EeArchitecturalObservationMode Mode>
    [[nodiscard]] PS2X_EE_OBSERVATION_POLICY_INLINE
    bool eeDataBreakpointEnabledPolicy(
        const R5900Context *ctx,
        bool write) const noexcept
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            return EeDataBreakpointEnabled(ctx, write);
        }
        return false;
    }

    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void observeEePredictedBranchPolicy(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            observeEePredictedBranchPrecise(
                ctx, instruction);
        }
    }

    template <EeArchitecturalObservationMode Mode>
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void observeEeConditionalBranchPolicy(
        R5900Context *ctx,
        const EeObservationInstructionDescriptor *instruction,
        bool taken)
    {
        if constexpr (Mode ==
                      EeArchitecturalObservationMode::Precise)
        {
            observeEeConditionalBranchPrecise(
                ctx, instruction, taken);
        }
    }

    void executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address);
    void vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address);
    void synchronizeVU0Microprogram(uint8_t *rdram,
                                    R5900Context *ctx,
                                    bool interlocked);
    void publishVU0VectorRegisterWrite(
        const R5900Context *ctx, uint32_t index) noexcept;
    void publishVU0ControlRegisterWrite(
        const R5900Context *ctx, uint32_t index) noexcept;
    [[nodiscard]] bool readCop2Condition(
        const R5900Context *ctx) const noexcept;
    void serviceEeEventsAtBlockBoundary(uint8_t *rdram,
                                        R5900Context *ctx);
    void serviceEeEventsAtGeneratedBlockBoundary(
        uint8_t *rdram,
        R5900Context *ctx);
    void configureEeVSyncVideoMode(
        uint32_t videoMode,
        bool interlaced);
    void ensureEeVSyncScheduled();
    uint64_t waitForNextScheduledVSync(
        uint8_t *rdram,
        R5900Context *ctx,
        uint64_t observedVSyncTick);

public:
    void handleSyscall(uint8_t *rdram, R5900Context *ctx);
    void handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId);
    void handleBreak(uint8_t *rdram, R5900Context *ctx);

    void handleTrap(uint8_t *rdram, R5900Context *ctx);
    void handleTLBR(uint8_t *rdram, R5900Context *ctx);
    void handleTLBWI(uint8_t *rdram, R5900Context *ctx);
    void handleTLBWR(uint8_t *rdram, R5900Context *ctx);
    void handleTLBP(uint8_t *rdram, R5900Context *ctx);
    [[nodiscard]] bool readCop0Condition0();
    [[nodiscard]] uint32_t readCop0Count(
        uint8_t *rdram,
        R5900Context *ctx);
    void writeCop0Count(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t value);
    [[nodiscard]] uint32_t readCop0Compare(
        R5900Context *ctx);
    void writeCop0Compare(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t value);
    [[nodiscard]] uint32_t readCop0Performance(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t selector);
    void writeCop0Breakpoint(
        R5900Context *ctx,
        uint32_t selector,
        uint32_t value);
    void writeCop0Performance(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t selector,
        uint32_t value);
    void writeCop0Status(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t value);
    void handleCop0Eret(
        uint8_t *rdram,
        R5900Context *ctx);
    void handleCop0Ei(
        uint8_t *rdram,
        R5900Context *ctx);
    void handleCop0Di(
        uint8_t *rdram,
        R5900Context *ctx);
    void handleEeCacheOperation(
        uint8_t *rdram,
        R5900Context *ctx,
        uint32_t operation,
        uint32_t virtualAddress);
    void configureGuestHeap(uint32_t guestBase, uint32_t guestLimit = PS2_RAM_SIZE);
    uint32_t guestMalloc(uint32_t size, uint32_t alignment = 16u);
    uint32_t guestCalloc(uint32_t count, uint32_t size, uint32_t alignment = 16u);
    uint32_t guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment = 16u);
    void guestFree(uint32_t guestAddr);
    uint32_t guestHeapBase() const;
    uint32_t guestHeapEnd() const;
    uint32_t guestHeapLimit() const;
    void executeGuestStep(uint8_t *rdram, R5900Context *ctx, RecompiledFunction function);
    void dispatchLoop(uint8_t *rdram, R5900Context *ctx);

    void drainCompletedDmacHandlers(uint8_t *rdram);
    void requestDmacCompletion(DmacChannel channel);
    bool submitHleSifDma(
        uint8_t *rdram,
        R5900Context *ctx,
        const HleSifDmaSubmission &submission);
    [[nodiscard]] int32_t hleSifDmaStatus(
        uint32_t transferId) const noexcept;

    void requestGuestPreemption();
    [[nodiscard]] PS2GuestCheckpointResult
    checkpointGuestExecution(R5900Context *ctx);
    void yieldGuestExecutionAtBoundary();
    void yieldGuestExecutionAfterWake();
    void waitForGuestExecutionHandoff();
    void waitForGuestExecutionHandoff(uint64_t baselineEpoch);
    // A raw i* wake publishes READY synchronously but cannot release its
    // legacy host waiter until the interrupted owner reaches a boundary.
    // This preserves the same non-rescheduling policy as the EE scheduler.
    [[nodiscard]] bool
    shouldDeferLegacyEeWakeNotification() const noexcept;
    void deferLegacyEeWakeNotification(
        const std::shared_ptr<ThreadInfo> &thread);
    void flushDeferredLegacyEeWakeNotifications();
    uint64_t guestExecutionHandoffEpochSnapshot() const
    {
        return m_guestExecutionHandoffEpoch.load(std::memory_order_acquire);
    }

    void recordEeThreadQueueRotation(
        int threadId,
        int requestedPriority,
        int effectivePriority,
        bool accepted) noexcept;
    void recordMpegPictureServed(R5900Context *ctx, bool repeated);

    void requestStop();
    bool isStopRequested() const;

    // Debug-control operations are intentionally backend-neutral. They are used
    // by the local ps2dbg server and by focused runtime tests.
    bool debugPause(
        std::chrono::milliseconds timeout = std::chrono::seconds(30),
        const char *source = "external");
    void debugResume();
    bool debugIsPaused() const;
    bool debugSetVuBackend(
        VuUnitId unit, VuBackendKind backend,
        std::string *diagnostic = nullptr,
        std::chrono::milliseconds timeout =
            std::chrono::seconds(30));
    DebugStopInfo debugRunUntilPc(uint32_t pc, std::chrono::milliseconds timeout);
    DebugStopInfo debugRunUntilMpegUniquePictures(
        uint64_t target, std::chrono::milliseconds timeout);
    DebugStopInfo debugRunUntilBeforeNextMpegUniquePicture(
        uint64_t current, std::chrono::milliseconds timeout);
    DebugStopInfo debugRunForVSyncFields(
        uint64_t count, std::chrono::milliseconds timeout);
    DebugStopInfo debugStepDispatches(uint64_t count, std::chrono::milliseconds timeout);
    R5900Context debugCpuSnapshot();
    VuExecutionState debugVu0ArchitecturalSnapshot();
    DebugEeTiming debugEeTimingSnapshot();
    DebugEeScheduler debugEeSchedulerSnapshot();
    bool debugReadRdram(uint32_t address, uint32_t size, std::vector<uint8_t> &output);
    bool debugReadMemory(uint32_t address, uint32_t size, std::vector<uint8_t> &output);
    bool debugCopyGsVram(std::vector<uint8_t> &output);
    DebugRuntimeProgress debugRuntimeProgress() const;
    DebugEeThreadDiagnostics debugEeThreadDiagnosticsSnapshot() const;
    std::optional<ps2x::ee::EeRuntimeExecutorStatistics>
    debugEeRuntimeExecutorStatisticsSnapshot() const;
    DebugVu1Timing debugVu1TimingSnapshot();
    std::array<DebugVuBackendDiagnostics, 2u>
    debugVuBackendDiagnosticsSnapshot() const;
    std::vector<DebugBranchEntry> debugBranchHistory(
        size_t maximumEntries = 256u) const;
    DebugFaultInfo debugFaultSnapshot() const;
    void debugClearFault();
    void debugStartVu0SyncTrace(
        size_t maximumEntries,
        std::optional<uint32_t> triggerEePc = std::nullopt,
        bool stopOnFull = false,
        std::optional<uint64_t> triggerInvocation = std::nullopt);
    DebugVu0SyncTrace debugVu0SyncTraceSnapshot(bool stop);
    void debugStartVu0InstructionTrace(
        size_t maximumEntries,
        std::optional<uint32_t> triggerEePc = std::nullopt,
        bool stopOnFull = false,
        std::optional<uint64_t> triggerInvocation = std::nullopt);
    DebugVu0InstructionTrace debugVu0InstructionTraceSnapshot(bool stop);
    void debugStartEeEventTrace(
        size_t maximumEntries,
        std::optional<uint32_t> triggerEePc = std::nullopt,
        bool stopOnFull = false);
    DebugEeEventTrace debugEeEventTraceSnapshot(bool stop);
    void debugStartVu1TimingTrace(
        size_t maximumEntries,
        bool stopOnFull = false);
    DebugVu1TimingTrace debugVu1TimingTraceSnapshot(bool stop);

    std::vector<uint32_t> debugBreakpoints() const;
    void debugAddBreakpoint(uint32_t address);
    void debugRemoveBreakpoint(uint32_t address);

    std::vector<DebugWatchpoint> debugWatchpoints() const;
    uint64_t debugAddWatchpoint(uint32_t start, uint32_t size, DebugMemoryAccess access);
    bool debugRemoveWatchpoint(uint64_t id);
    bool debugWatchpointsEnabled() const;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    void debugObserveMemoryAccess(uint32_t address,
                                  uint32_t size,
                                  DebugMemoryAccess access,
                                  const R5900Context *ctx);
#else
    void debugObserveMemoryAccess(
        uint32_t address,
        uint32_t size,
        DebugMemoryAccess access,
        const R5900Context *ctx)
    {
        (void)address;
        (void)size;
        (void)access;
        (void)ctx;
    }
#endif

    uint32_t guestExecutionWaiterCountForTesting() const
    {
        return m_guestExecutionWaiters.load(std::memory_order_acquire);
    }
    EeThreadRuntimeState &eeThreadRuntimeState();
    const EeThreadRuntimeState &eeThreadRuntimeState() const;
    IEeExecutionBackend &eeExecutionBackend();
    const IEeExecutionBackend &eeExecutionBackend() const;
    [[nodiscard]] std::string_view
    eeExecutionBackendName() const noexcept;
    [[nodiscard]] size_t
    managedEeExecutionThreadCountForTesting() const;
    [[nodiscard]] bool
    eeExecutionQuiescentForTesting() const noexcept;
    using EeThreadContextImageForTesting = std::pair<
        int,
        std::array<uint8_t, sizeof(R5900Context)>>;
    [[nodiscard]] std::vector<
        EeThreadContextImageForTesting>
    eeThreadContextsForTesting();
    [[nodiscard]] bool
    usesDedicatedEeExecutor() const noexcept;
    // Headless lifecycle seams used by focused runtime fixtures. The generic
    // pair starts whichever production backend was selected; the dedicated
    // pair retains the stricter executor-only contract used by fiber-specific
    // tests.
    void startEeExecutionForTesting();
    void stopEeExecutionForTesting();
    void startDedicatedEeExecutionForTesting();
    void stopDedicatedEeExecutionForTesting();
    // Complete a kernel reset after guest thread state has been released.
    // Unlike requestStop(), this retires native EE continuations without
    // publishing runtime shutdown, so a fresh execution epoch may start on
    // the same runtime.
    void completeEeExecutionKernelReset();
    [[nodiscard]] bool publishEeExecutorUpdate(
        std::function<void(
            ps2x::ee::EeThreadScheduler &,
            IEeExecutionBackend &)> update,
        ps2x::ee::EeSchedulerReschedulePolicy policy =
            ps2x::ee::EeSchedulerReschedulePolicy::
                HigherPriorityOnly);
    [[nodiscard]] bool publishEeSchedulerUpdate(
        std::function<void(
            ps2x::ee::EeThreadScheduler &)> update,
        ps2x::ee::EeSchedulerReschedulePolicy policy =
            ps2x::ee::EeSchedulerReschedulePolicy::
                HigherPriorityOnly);
    [[nodiscard]] bool publishEeSchedulerUpdateAt(
        std::chrono::steady_clock::time_point deadline,
        std::function<void(
            ps2x::ee::EeThreadScheduler &)> update,
        ps2x::ee::EeSchedulerReschedulePolicy policy =
            ps2x::ee::EeSchedulerReschedulePolicy::
                HigherPriorityOnly);
    void invokeEeSchedulerUpdateAtBoundary(
        std::function<void(
            ps2x::ee::EeThreadScheduler &)> update,
        ps2x::ee::EeSchedulerReschedulePolicy policy =
            ps2x::ee::EeSchedulerReschedulePolicy::
                HigherPriorityOnly);
    void yieldEeExecutorCurrent(
        ps2x::ee::EeSchedulerExitReason reason,
        ps2x::ee::EeSchedulerWaitKey wait = {});
    [[nodiscard]] const void *
    hostPresentationUploadStateIdentityForTesting() const noexcept;
    [[nodiscard]] size_t
    pendingAlarmCallbackCountForTesting();
    EeAlarmRuntimeState &eeAlarmRuntimeState();
    const EeAlarmRuntimeState &eeAlarmRuntimeState() const;
    EeSyncRuntimeState &eeSyncRuntimeState();
    const EeSyncRuntimeState &eeSyncRuntimeState() const;
    EeInterruptRuntimeState &eeInterruptRuntimeState();
    const EeInterruptRuntimeState &eeInterruptRuntimeState() const;
    EeKernelRuntimeState &eeKernelRuntimeState();
    const EeKernelRuntimeState &eeKernelRuntimeState() const;
    EeRpcRuntimeState &eeRpcRuntimeState();
    const EeRpcRuntimeState &eeRpcRuntimeState() const;
    EeFileRuntimeState &eeFileRuntimeState();
    const EeFileRuntimeState &eeFileRuntimeState() const;
    Deci2RuntimeState &deci2RuntimeState();
    const Deci2RuntimeState &deci2RuntimeState() const;
    ps2_stubs::AudioRuntimeState &audioRuntimeState();
    const ps2_stubs::AudioRuntimeState &
    audioRuntimeState() const;
    ps2_stubs::CdRuntimeState &cdRuntimeState();
    const ps2_stubs::CdRuntimeState &cdRuntimeState() const;
    ps2_stubs::DmaRuntimeState &dmaRuntimeState();
    const ps2_stubs::DmaRuntimeState &dmaRuntimeState() const;
    ps2_stubs::LibCRuntimeState &libcRuntimeState();
    const ps2_stubs::LibCRuntimeState &libcRuntimeState() const;
    ps2_stubs::MemoryCardRuntimeState &
    memoryCardRuntimeState();
    const ps2_stubs::MemoryCardRuntimeState &
    memoryCardRuntimeState() const;
    ps2_stubs::PadRuntimeState &padRuntimeState();
    const ps2_stubs::PadRuntimeState &padRuntimeState() const;
    ps2_stubs::SifRuntimeState &sifRuntimeState();
    const ps2_stubs::SifRuntimeState &
    sifRuntimeState() const;
    ps2_stubs::MpegRuntimeState &mpegRuntimeState();
    const ps2_stubs::MpegRuntimeState &mpegRuntimeState() const;
    int activeEeHostThreadCount() const;
    int currentEeThreadId() noexcept;
    uint64_t eeThreadLegacyAdapterMismatchCount() const noexcept;
    bool guestPreemptionRequestedForTesting() const
    {
        const uint64_t requestedGeneration =
            m_guestExecutionPreemptionGeneration.load(
                std::memory_order_acquire);
        return requestedGeneration !=
               m_guestExecutionConsumedPreemptionGeneration.load(
                   std::memory_order_acquire);
    }
    uint64_t dmacInterruptDrainPassesForTesting() const
    {
        return m_dmacInterruptDrainPasses.load(
            std::memory_order_acquire);
    }
    size_t pendingDmacInterruptCountForTesting() const
    {
        return m_pendingDmacInterrupts.size();
    }

    uint64_t guestExecutionHandoffTimeouts() const
    {
        return m_guestExecutionHandoffTimeouts.load(std::memory_order_relaxed);
    }

    uint8_t Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint16_t Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint32_t Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint64_t Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    __m128i Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);

    void Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value);
    void Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value);
    void Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value);
    void Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value);
    void StoreMasked32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value, uint8_t byteEnable);
    void StoreMasked64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value, uint8_t byteEnable);
    void Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value);
    void kickGifDmaChainFromMMIO(uint8_t *rdram,
                                 R5900Context *ctx,
                                 uint32_t dPcrValue,
                                 uint32_t dStatValue,
                                 uint32_t tadr,
                                 uint32_t chcr);

    static inline bool isSpecialAddress(uint32_t addr)
    {
        return Ps2IsSpecialAddress(addr);
    }

public:
    inline R5900Context &cpu() { return m_cpuContext; }
    inline const R5900Context &cpu() const { return m_cpuContext; }

    inline PS2Memory &memory() { return m_memory; }
    inline const PS2Memory &memory() const { return m_memory; }

    inline GS &gs() { return m_gs; }
    inline const GS &gs() const { return m_gs; }
    inline GifArbiter &gifArbiter() { return m_gifArbiter; }
    inline const GifArbiter &gifArbiter() const { return m_gifArbiter; }
    inline VuUnit &vu0() { return m_vu0; }
    inline const VuUnit &vu0() const { return m_vu0; }
    // Compatibility/test access to the semantic unit is valid only while
    // the caller itself is the inline owner. Threaded modes must use ordered
    // commands or immutable snapshots.
    inline VuUnit &vu1()
    {
        if (m_vu1ExecutionMode != Vu1ExecutionMode::Inline)
        {
            throw std::logic_error(
                "direct VU1 access is unavailable in threaded mode");
        }
        return m_vu1;
    }
    inline const VuUnit &vu1() const
    {
        if (m_vu1ExecutionMode != Vu1ExecutionMode::Inline)
        {
            throw std::logic_error(
                "direct VU1 access is unavailable in threaded mode");
        }
        return m_vu1;
    }
    inline Vu1CommandProcessor &vu1CommandProcessor()
    {
        if (m_vu1ExecutionMode != Vu1ExecutionMode::Inline)
        {
            throw std::logic_error(
                "direct VU1 processor access is unavailable in threaded mode");
        }
        return m_vu1CommandProcessor;
    }
    inline const Vu1CommandProcessor &vu1CommandProcessor() const
    {
        if (m_vu1ExecutionMode != Vu1ExecutionMode::Inline)
        {
            throw std::logic_error(
                "direct VU1 processor access is unavailable in threaded mode");
        }
        return m_vu1CommandProcessor;
    }

    inline PS2AudioBackend &audioBackend() { return m_audioBackend; }
    inline const PS2AudioBackend &audioBackend() const { return m_audioBackend; }
    inline PSPadBackend &padBackend() { return m_padBackend; }
    inline const PSPadBackend &padBackend() const { return m_padBackend; }

private:
    void acknowledgeGuestPreemptionRequests() noexcept;
    [[nodiscard]] bool consumeGuestPreemptionRequest() noexcept;
    void startDedicatedEeExecution();
    void joinDedicatedEeExecution();
    void runMainEeContinuation();
    [[nodiscard]] ps2x::timing::EeTickDelta
    eeExecutorElapsedSinceSelection() const noexcept;

    void commitPriorContext(
        std::optional<int> priorThreadId,
        ps2x::timing::EeTickDelta elapsed,
        ps2x::timing::EeTick now) override;
    void publishSelectedContext(
        std::optional<int> selectedThreadId,
        ps2x::timing::EeTick now) override;
    void publishWaitCompletion(
        int threadId,
        ps2x::ee::EeSchedulerCompletedWait completion,
        ps2x::timing::EeTick now) override;
    [[nodiscard]] bool hasImmediateConsequence(
        ps2x::ee::EeSchedulerConsequenceStage stage,
        ps2x::timing::EeTick now) const override;
    [[nodiscard]] ps2x::ee::
        EeSchedulerReschedulePolicy
    applyNextConsequence(
        ps2x::ee::EeSchedulerConsequenceStage stage,
        ps2x::timing::EeTick now,
        ps2x::ee::EeThreadScheduler &scheduler) override;

    struct EeVSyncDurations;
    enum class EeVSyncVideoModeClass : uint8_t;
    struct PendingVSyncDelivery;

    struct EeInterruptedContinuationFrame
    {
        R5900Context context{};
        R5900Context *continuationContext = nullptr;
        std::shared_ptr<ThreadInfo> continuationOwner;
        uint32_t generation = 0u;
        int threadId = 0;
        bool idle = true;
    };

    struct GuestHeapBlock
    {
        uint32_t addr = 0;
        uint32_t size = 0;
        bool free = true;
    };

    static uint32_t alignGuestHeapValue(uint32_t value, uint32_t alignment);
    static bool isGuestHeapAlignmentValid(uint32_t alignment);
    static uint32_t normalizeGuestHeapAlignment(uint32_t alignment);
    uint32_t clampGuestHeapBase(uint32_t guestBase) const;
    uint32_t clampGuestHeapLimit(uint32_t guestLimit) const;
    void resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit);
    void ensureGuestHeapInitializedLocked();
    int32_t findGuestHeapBlockIndexLocked(uint32_t guestAddr) const;
    uint32_t allocateGuestBlockLocked(uint32_t size, uint32_t alignment);
    void freeGuestBlockLocked(uint32_t guestAddr);
    void coalesceGuestHeapLocked();
    [[nodiscard]] EeInterruptedContinuationFrame
    makeEeInterruptedContinuation(
        R5900Context *context,
        int threadId,
        bool retainDormantContext = false);
    void captureEeInterruptContinuation(
        R5900Context *context,
        int threadId);
    void captureEeInterruptContinuation(
        std::optional<int> threadId);
    void clearEeInterruptContinuation() noexcept;
    void beginAsyncCallbackInvocation(
        AsyncCallbackInvocationScope &scope,
        R5900Context *continuationContext = nullptr,
        int continuationThreadId = -1);
    void endAsyncCallbackInvocation(
        AsyncCallbackInvocationScope &scope) noexcept;
    R5900Context *enterGuestExecution(
        R5900Context *context,
        int *previousThreadId = nullptr,
        int selectionHint = -1);
    bool tryEnterGuestExecution(
        R5900Context *context,
        std::chrono::milliseconds timeout,
        R5900Context *&previousContext,
        int &previousThreadId,
        int selectionHint = -1);
    void lockGuestExecutionMutex(bool mayContend);
    void recordOuterGuestExecutionAcquisition(
        R5900Context *context) noexcept;
    void leaveGuestExecution(
        R5900Context *context,
        R5900Context *previousContext,
        int previousThreadId);
    uint32_t releaseGuestExecution(
        R5900Context *&context,
        int &threadId);
    void reacquireGuestExecution(
        uint32_t depth,
        R5900Context *context,
        int threadId);
    void markGuestExecutionAcquired();
    [[nodiscard]] bool
    invokeEeExecutorTaskAtBoundary(
        std::function<void()> task);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    void debugBeforeGuestStep(R5900Context *ctx);
    void debugAfterGuestStep(R5900Context *ctx);
#else
    void debugBeforeGuestStep(R5900Context *ctx)
    {
        (void)ctx;
    }
    void debugAfterGuestStep(R5900Context *ctx)
    {
        (void)ctx;
    }
#endif
    bool debugBlockGuestAtBoundary(
        R5900Context *ctx,
        const char *reason,
        std::optional<uint64_t> observedControlGeneration =
            std::nullopt);
    void debugWaitUntilResumed();
    void debugRecordStopLocked(
        const char *reason,
        uint32_t pc,
        uint64_t observedControlGeneration = 0u);
    void debugSetPauseRequestedLocked(
        bool requested,
        const char *writer);
    void debugAdvanceControlGenerationLocked(
        const char *writer);
    void debugRefreshControlActiveLocked();
    void debugPublishVuBackendDiagnostics();
    void debugRecordBranch(uint32_t pc);
    void debugRecordFault(const DebugFaultInfo &fault);
    void debugRecordVu0Sync(DebugVu0SyncEntry entry);
    void debugRecordVu0Instruction(DebugVu0InstructionEntry entry);
    void debugRecordEeEvent(DebugEeEventEntry entry);
    void debugBeginVu1TimingInvocation(
        Vu1InvocationKind kind,
        uint32_t startPc,
        uint32_t top,
        uint32_t itop,
        ps2x::timing::EeTick startTick) noexcept;
    void debugPublishVu1TimingInvocationStart(
        ps2x::timing::EeTick deadline,
        uint32_t estimatedCycles);
    void debugRecordVu1Timing(
        DebugVu1TimingEntry entry) const;
    [[nodiscard]] DebugVu1TimingEntry
    debugVu1TimingEntry(
        DebugVu1TimingEventKind kind,
        const R5900Context *ctx = nullptr) const noexcept;
    void debugRecordVu1Path1Packet(
        const Vu1Path1Packet &packet,
        uint32_t defaultCycleOffset);
    bool resumeVif1AfterVuWithTimingTrace();
    [[nodiscard]] DebugEeEventDeviceState
    debugEeEventDeviceState(
        ps2x::timing::EeEventSource source) const noexcept;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    PS2X_EE_OBSERVATION_POLICY_INLINE
    void debugArmVu0Traces(const R5900Context *ctx)
    {
        // PC-triggered traces are unusual, while this hook runs at every
        // generated EE block boundary.  Keep the large snapshot-capable
        // implementation out of that dormant path.  This is deliberately a
        // one-way latch: clearing it would require coordinating independent
        // debug-server trace reconfiguration and is unnecessary for
        // correctness.
        if (!m_debugEePcTraceTriggerEverConfigured.load(
                std::memory_order_acquire)) [[likely]]
        {
            return;
        }
        debugArmVu0TracesSlow(ctx);
    }
    PS2X_EE_RUNTIME_NOINLINE
    void debugArmVu0TracesSlow(const R5900Context *ctx);
    void beginVu0Invocation();
#else
    void debugArmVu0Traces(const R5900Context *ctx)
    {
        (void)ctx;
    }
    void beginVu0Invocation()
    {
    }
#endif

    [[noreturn]] void HandleIntegerOverflow(R5900Context *ctx);

    [[nodiscard]] ps2x::iop::RpcAbi selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const;
    [[nodiscard]] ps2x::iop::RpcResult handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request);
    void notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer);
    void resetIop();

    friend class GuestExecutionScope;
    friend class GuestExecutionReleaseScope;
    friend class GuestExecutionTryScope;
    friend class PS2IopTransport;
    friend class PS2DebugServer;

private:
    void synchronizeVU0MicroprogramAtTick(
        uint8_t *rdram,
        R5900Context *ctx,
        bool interlocked,
        bool blockBoundary,
        bool eventDue,
        ps2x::timing::EeTick eeCycleTick,
        ps2x::timing::EeTick nextEventTick);
    void serviceEeEventsAtTick(
        uint8_t *rdram,
        R5900Context *ctx,
        ps2x::timing::EeTick eeCycleTick);
    PS2X_EE_RUNTIME_NOINLINE
    void serviceEeEventsAtGeneratedPendingIssueBoundary(
        uint8_t *rdram,
        R5900Context *ctx);
    PS2X_EE_RUNTIME_NOINLINE
    void serviceEeEventsAtGeneratedDeadline(
        uint8_t *rdram,
        R5900Context *ctx,
        ps2x::timing::EeTick eeCycleTick);
    void dispatchEeEvent(
        uint8_t *rdram,
        R5900Context *ctx,
        const ps2x::timing::EeEventService &service);
    void startVU0FromVif(
        uint32_t startPC, uint32_t itop);
    void resumeVU0FromVif(uint32_t itop);
    void serviceVU0AtVifFinishEvent(
        uint8_t *rdram,
        R5900Context *ctx,
        const ps2x::timing::EeEventService &service);
    void scheduleVif0VuFinishEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void cancelVif0VuFinishEvent() noexcept;
    void startVU1FromVif(
        uint32_t startPC, uint32_t top, uint32_t itop,
        bool flushBeforeStart);
    void resumeVU1FromVif(uint32_t top, uint32_t itop);
    void serviceVU1AtEvent(
        R5900Context *ctx,
        const ps2x::timing::EeEventService &service);
    void scheduleVU1Event(
        ps2x::timing::EeTick deadline) noexcept;
    void cancelVU1Execution(bool resetInterpreter);
    struct Vu1PendingSlicePlan
    {
        uint32_t cycleBudget = 0u;
        ps2x::timing::EeTick deadline{};
        uint64_t eventGeneration = 0u;
        uint64_t executionGeneration = 0u;
    };
    struct Vu1SynchronousCommandBarrier
    {
        Vu1SpeculationResolution resolution =
            Vu1SpeculationResolution::None;
        std::optional<Vu1PendingSlicePlan> requeue;
        bool coarseInvocationCompleted = false;
    };
    enum class Vu1SpeculationPublicationState : uint8_t
    {
        None,
        Unpublished,
        Published,
    };
    [[nodiscard]] Vu1SynchronousCommandBarrier
    beginVu1SynchronousCommandBarrier(
        Vu1CommandType commandType);
    void recordVu1CoarseForcedBarrier(
        Vu1CommandType commandType);
    void finishVu1SynchronousCommandBarrier(
        const Vu1SynchronousCommandBarrier &barrier);
    void beginVu1SynchronousCommandBatch();
    void finishVu1SynchronousCommandBatch(bool requeue);
    [[nodiscard]] bool isCurrentVu1PendingSlicePlan(
        const Vu1PendingSlicePlan &plan) const noexcept;
    [[nodiscard]] bool hasPrecedingVif1Event(
        const Vu1PendingSlicePlan &plan) const noexcept;
    void clearDeferredVu1SpeculativeSlice() noexcept;
    void requeueOrDeferVu1SpeculativeSlice(
        const Vu1PendingSlicePlan &plan);
    void armDeferredVu1SpeculativeSlice(
        uint64_t requiredEventGeneration = 0u);
    void enqueueVu1SpeculativeSlice(
        const Vu1PendingSlicePlan &plan,
        Vu1SpeculationResolution previousResolution);
    [[nodiscard]] Vu1CommandResult
    consumeVu1SpeculativeSliceAtEvent(
        const ps2x::timing::EeEventService &service,
        uint32_t cycleBudget);
    [[nodiscard]] Vu1CommandResult
    consumeVu1ExecutionEpochAtEvent(
        const ps2x::timing::EeEventService &service,
        uint32_t cycleBudget);
    void resolveVu1SpeculationForShutdown() noexcept;
    struct Vu1CoarseEstimatorIdentity
    {
        Vu1InvocationKind kind = Vu1InvocationKind::Mscal;
        uint32_t startPc = 0u;
        uint32_t top = 0u;
        uint32_t itop = 0u;
        uint64_t codeGeneration = 0u;

        bool operator==(
            const Vu1CoarseEstimatorIdentity &) const = default;
    };
    struct Vu1CoarseEstimatorEntry
    {
        Vu1CoarseEstimatorIdentity identity{};
        std::deque<uint32_t> cycleHistory;
    };
    [[nodiscard]] Vu1CoarseEstimatorIdentity
    vu1CoarseEstimatorIdentity(
        const Vu1InvocationCommand &command) const;
    [[nodiscard]] uint32_t estimateVu1CoarseCycles(
        const Vu1CoarseEstimatorIdentity &identity);
    void submitVu1CoarseInvocation(
        Vu1InvocationCommand command,
        ps2x::timing::EeTick startTick);
    [[nodiscard]] bool completeVu1CoarseInvocation(
        ps2x::timing::EeTick publicationTick,
        bool schedulerEvent);
    void discardVu1CoarseInvocationAfterFailure() noexcept;
    void resolveVu1CoarseInvocationForShutdown() noexcept;
    [[nodiscard]] Vu1CommandResult submitVu1Command(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandResult submitVu1AdvanceSlice(
        Vu1AdvanceSliceCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandResult submitVu1DecodedUnpack(
        Vu1DecodedUnpackCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandResult submitVu1VifStateUpdate(
        Vu1VifStateUpdateCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandResult
    submitVu1VifStateUpdateImmediate(
        Vu1VifStateUpdateCommand command,
        uint64_t guestTick,
        uint64_t publicationToken);
    [[nodiscard]] std::shared_ptr<
        const Vu1DecodedUnpackBatch>
    takeBatchedVu1DecodedUnpacks();
    void flushBatchedVu1DecodedUnpacks();
    void flushBatchedVu1VifStateUpdate();
    void refreshVu1DiagnosticsConfiguration(
        uint64_t guestTick,
        uint64_t publicationToken);
    void publishVu1CommandResult(
        const Vu1CommandResult &result);
    void synchronizeVu1MemoryMirror();
    [[nodiscard]] std::shared_ptr<const Vu1Snapshot>
    captureVu1OwnerSnapshot(
        bool includeBackendDiagnostics = false);
    [[nodiscard]] std::shared_ptr<const Vu1Snapshot>
    captureVu1OwnerDiagnosticSnapshot(
        bool includeBackendDiagnostics = false);
    void publishVu1SliceEffects(
        const Vu1SliceResult &slice);
    bool scheduleVif1DmaFromMemory(
        uint32_t delayEeCycles);
    void cancelVif1DmaEvent() noexcept;
    void scheduleVif1DmaEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void serviceVif1DmaAtEvent(
        const ps2x::timing::EeEventService &service);
    bool scheduleVif0DmaFromMemory(
        uint32_t delayEeCycles);
    void cancelVif0DmaEvent() noexcept;
    void scheduleVif0DmaEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void serviceVif0DmaAtEvent(
        const ps2x::timing::EeEventService &service);
    void scheduleVif0DmaFollowup(
        ps2x::timing::EeTick serviceTick,
        uint32_t delayEeCycles,
        bool allowImmediateRescan) noexcept;
    bool scheduleGifDmaFromMemory(
        uint32_t delayEeCycles);
    void cancelGifDmaEvent() noexcept;
    void scheduleGifDmaEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void serviceGifDmaAtEvent(
        const ps2x::timing::EeEventService &service);
    bool scheduleFromIpuDmaFromMemory(
        uint32_t delayEeCycles);
    void cancelFromIpuDmaEvent() noexcept;
    void scheduleFromIpuDmaEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void serviceFromIpuDmaAtEvent(
        const ps2x::timing::EeEventService &service);
    bool scheduleToIpuDmaFromMemory(
        uint32_t delayEeCycles);
    void cancelToIpuDmaEvent() noexcept;
    void scheduleToIpuDmaEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void serviceToIpuDmaAtEvent(
        const ps2x::timing::EeEventService &service);
    [[nodiscard]] static uint64_t hleSifDmaIopCycles(
        const HleSifDmaSubmission &submission) noexcept;
    [[nodiscard]] bool executeHleSifDmaOperation(
        uint8_t *rdram,
        const HleSifDmaSubmission &submission);
    void scheduleHleSifDmaEvent(
        ps2x::timing::EeTick deadline) noexcept;
    void cancelHleSifDmaEvent() noexcept;
    void serviceHleSifDmaAtEvent(
        uint8_t *rdram,
        const ps2x::timing::EeEventService &service);
    bool scheduleScratchpadDmaFromMemory(
        DmacChannel channel,
        uint32_t delayEeCycles);
    void cancelScratchpadDmaEvent(
        DmacChannel channel) noexcept;
    void scheduleScratchpadDmaEvent(
        DmacChannel channel,
        ps2x::timing::EeTick deadline) noexcept;
    void serviceScratchpadDmaAtEvent(
        DmacChannel channel,
        const ps2x::timing::EeEventService &service);
    void serviceEeVSyncAtEvent(
        uint8_t *rdram,
        const ps2x::timing::EeEventService &service);
    void onEeCounterScheduleChanged(
        std::optional<uint64_t> deadlineCycle) noexcept;
    void onEeCounterInterruptStateChanged(
        uint32_t status,
        uint32_t mask,
        uint32_t newlyRaised) noexcept;
    void serviceEeCountersAtEvent(
        const ps2x::timing::EeEventService &service);
    void serviceCop0TimerAtEvent(
        R5900Context *ctx,
        const ps2x::timing::EeEventService &service);
    void serviceCop0PerformanceAtEvent(
        R5900Context *ctx,
        const ps2x::timing::EeEventService &service);
    void cancelEeCounterEvent() noexcept;
    void synchronizeEeCounterMmio(
        R5900Context *ctx,
        uint32_t address,
        uint32_t size);
    void drainPendingEeCounterHandlers(uint8_t *rdram);
    void drainPendingAlarmCallbacks();
    [[nodiscard]] static bool overlapsEeCounterRegister(
        uint32_t address, uint32_t size) noexcept;
    void scheduleEeVSyncEvent(
        ps2x::timing::EeTick deadline) noexcept;
    [[nodiscard]] static EeVSyncDurations
    eeVSyncDurationsForVideoMode(
        uint32_t videoMode,
        bool interlaced) noexcept;
    [[nodiscard]] static EeVSyncVideoModeClass
    eeVSyncVideoModeClassForMode(
        uint32_t videoMode) noexcept;
    void publishEeVSyncField();
    void resetEeVSyncStateUnlocked() noexcept;
    void resetEeVSyncPacingUnlocked() noexcept;
    [[nodiscard]] GsPrivilegedRegisterSnapshot
    captureGsPrivilegedRegisters() const noexcept;
    void applyGsPrivilegedSideEffects(
        const std::vector<GsPrivilegedSideEffect> &effects);
    void consumePendingEeGsResult(GsCommandResult result);
    bool consumeFrontEeGsSubmission(bool wait);
    void reapEeGsSubmissions(bool waitAll);
    void submitEeGsDrainBatch(GifArbiterDrainBatch batch);
    void submitEeGsFieldMarker(uint64_t fieldSequence);
    void updateGsPendingHighWater(size_t pending) noexcept;
    void paceEeVSyncStart(
        ps2x::timing::EeTick scheduledTick);
    [[nodiscard]] static std::chrono::steady_clock::duration
    hostDurationForEeTicks(
        ps2x::timing::EeTickDelta ticks) noexcept;
    [[nodiscard]] bool queuePendingVSyncDelivery(
        PendingVSyncDelivery delivery) noexcept;
    void drainPendingVSyncHandlers(uint8_t *rdram);
    [[nodiscard]] static ps2x::timing::EeEventSource
    scratchpadDmaEventSource(
        DmacChannel channel) noexcept;
    void setVU1BusyFlag(
        R5900Context *context, bool busy) noexcept;
    void onDmacCompletionReady();
    size_t publishReadyDmacCompletions(
        uint64_t eventSequence);
    void mergePendingDmacInterrupts(
        std::vector<DmacPendingInterrupt> pending);
    void collectPendingDmacInterrupts();
    void onDmacInterruptStateChanged();
    [[nodiscard]] ps2x::timing::EeTick
    prepareCop0Access(
        uint8_t *rdram,
        R5900Context *ctx);
    void synchronizeCop0Timing(
        R5900Context *ctx,
        uint64_t currentCycle,
        bool synchronizeCount = true,
        bool synchronizePerformance = true) noexcept;
    void publishCop0TimingMirrors(
        R5900Context *context = nullptr) noexcept;
    void refreshCop0EventSchedules(
        R5900Context *ctx,
        uint64_t currentCycle) noexcept;
    void deliverPendingCop0Exceptions(
        R5900Context *ctx);
    void recordEePerformanceEvents(
        R5900Context *ctx,
        const ps2x::timing::Cop0PerformanceEvents &events) noexcept;
    void flushEeInstructionIssue(
        R5900Context *ctx) noexcept;
    [[nodiscard]] bool cop0TimerInterruptEnabled(
        const R5900Context *ctx) const noexcept;
    [[noreturn]] void raiseCop0Level1Exception(
        R5900Context *ctx,
        uint32_t exceptionCode,
        bool commitContextProgress = true,
        bool useTlbRefillVector = true);
    [[noreturn]] void raiseTlbFault(
        R5900Context *ctx,
        const PS2TlbFaultException &fault,
        bool storeAccess);
    PS2X_EE_MEMORY_POLICY_NOINLINE
    void validateInstructionFetchPageMissWithoutObservation(
        R5900Context *ctx,
        uint32_t virtualAddress);
    [[noreturn]] void raiseCop0PerformanceException(
        R5900Context *ctx);
    [[noreturn]] void raiseCop0DebugException(
        R5900Context *ctx,
        bool dataBreakpoint);
    [[nodiscard]] ps2x::timing::EeTick publishEeElapsed(
        ps2x::timing::EeTickDelta elapsed) noexcept;
    [[nodiscard]] ps2x::timing::EeTick commitEeContextProgress(
        R5900Context *context) noexcept;
    [[nodiscard]] ps2x::timing::EeTick finishEeContextBlock(
        R5900Context *context) noexcept;
    int threadIdForEeContext(
        const R5900Context *context) noexcept;
    void publishEeProcessorGlobalState(
        R5900Context *context) noexcept;
    void restoreEeProcessorGlobalState(
        R5900Context *context) noexcept;
    void selectEeThread(int threadId) noexcept;
    void switchGuestExecutionContext(
        R5900Context *context,
        int selectionHint = -1) noexcept;
    void restoreGuestExecutionContext(
        R5900Context *context,
        R5900Context *previousContext,
        int previousThreadId) noexcept;
    void resetEeTimingUnlocked(R5900Context *context) noexcept;

    PS2Memory m_memory;
    EeCache m_eeCache;
    GifArbiter m_gifArbiter;
    GS m_gs;
    GsCommandProcessor m_gsCommandProcessor;
    std::unique_ptr<GsCommandExecutor> m_gsExecutor;
    ThreadedGsExecutor *m_threadedGsExecutor = nullptr;
    bool m_gsAsyncEnabled = false;
    uint32_t m_gsMaximumFieldLead = 1u;
    size_t m_gsAsyncPendingLimit = 1u;
    mutable std::mutex m_gsEeJournalMutex;
    std::deque<GsCommandSubmission>
        m_pendingEeGsSubmissions;
    std::atomic<size_t> m_gsPendingCompletions{0u};
    std::atomic<size_t> m_gsPendingHighWater{0u};
    std::atomic<uint64_t> m_gsFieldsSubmitted{0u};
    std::atomic<uint64_t> m_gsFieldsCompleted{0u};
    std::atomic<uint64_t> m_gsLastSubmittedFieldSequence{0u};
    std::atomic<uint64_t> m_gsLastCompletedFieldSequence{0u};
    std::atomic<uint64_t> m_gsFieldLeadHighWater{0u};
    std::atomic<uint64_t> m_gsFieldLeadBlockCount{0u};
    std::atomic<uint64_t> m_gsFieldLeadBlockedNanoseconds{0u};
    std::atomic<uint64_t> m_publishedGsGuestTick{0u};
    std::unique_ptr<HostPresentationUploadState>
        m_hostPresentationUploadState;
    std::unique_ptr<PS2IopHostAdapter> m_iopHost;
    std::unique_ptr<ps2x::iop::IopSubsystem> m_iopSubsystem;
    PS2AudioBackend m_audioBackend;
    PSPadBackend m_padBackend;
    mutable std::mutex m_ioPathsMutex;
    IoPaths m_ioPaths;
    VuUnit m_vu0;
    VuUnit m_vu1;
    Vu1CommandProcessor m_vu1CommandProcessor;
    std::unique_ptr<Vu1CommandExecutor> m_vu1Executor;
    ThreadedVu1Executor *m_threadedVu1Executor = nullptr;
    Vu1ExecutionMode m_vu1ExecutionMode =
        Vu1ExecutionMode::Inline;
    bool m_vu1CheckpointEpochs = false;
    size_t m_vu1CheckpointCapacity = 4u;
    uint64_t m_vu1CommandPayloadCapacityBytes =
        8u * 1024u * 1024u;
    uint32_t m_vu1CoarseInitialEstimateCycles = 16u;
    uint32_t m_vu1CoarseMinimumEstimateCycles = 4u;
    uint32_t m_vu1CoarseMaximumLeadCycles = 65536u;
    uint32_t m_vu1CoarseMaximumInvocationCycles = 65536u;
    uint64_t m_vu1CoarseMaximumPath1Bytes =
        64u * 1024u * 1024u;
    size_t m_vu1CoarseEstimatorHistoryCapacity = 4u;
    size_t m_vu1CoarseEstimatorIdentityCapacity = 64u;
    std::atomic<bool> m_publishedVu1Active{false};
    std::atomic<uint32_t> m_publishedVu1ProgramCounter{0u};
    std::atomic<uint64_t> m_publishedVu1CodeGeneration{0u};
    std::atomic<bool> m_publishedVu1ProgressEnabled{false};
    std::atomic<bool> m_publishedVu1ProgressActive{false};
    std::atomic<uint64_t> m_publishedVu1ProgressInvocations{0u};
    std::atomic<uint64_t> m_publishedVu1ProgressCycles{0u};
    std::atomic<uint32_t> m_publishedVu1ProgressPc{0u};
    bool m_publishedVu1TraceEnabled = false;
    bool m_publishedVu1WorkloadProfileEnabled = false;
    bool m_vu1MemoryMirrorDirty = false;
    bool m_vu1MemoryMirrorSynchronizationActive = false;
    struct Vu1PendingSlice
    {
        Vu1CommandSubmission submission;
        Vu1PendingSlicePlan plan{};
    };
    std::optional<Vu1PendingSlice> m_pendingVu1Slice;
    struct Vu1PendingExecutionEpoch
    {
        Vu1ExecutionEpoch epoch;
        Vu1PendingSlicePlan plan{};
    };
    std::optional<Vu1PendingExecutionEpoch>
        m_pendingVu1ExecutionEpoch;
    struct Vu1PendingCoarseInvocation
    {
        Vu1CommandSubmission submission;
        std::optional<Vu1CommandResult> completedResult;
        Vu1CoarseEstimatorIdentity estimatorIdentity{};
        uint32_t estimatedCycles = 0u;
        uint64_t estimatorInputHash = 0u;
        bool ownerReadyAtResolution = false;
        GifArbiter::Path1ReservationToken
            path1Reservation = 0u;
        ps2x::timing::EeTick startTick{};
        ps2x::timing::EeTick deadline{};
        uint64_t eventGeneration = 0u;
        uint64_t executionGeneration = 0u;
    };
    std::optional<Vu1PendingCoarseInvocation>
        m_pendingVu1CoarseInvocation;
    std::deque<uint32_t> m_vu1CoarseCycleHistory;
    std::deque<Vu1CoarseEstimatorEntry>
        m_vu1CoarseEstimatorEntries;
    std::optional<Vu1PendingSlicePlan>
        m_deferredVu1SlicePlan;
    Vu1SpeculationPublicationState
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
    Vu1SynchronousCommandBarrier
        m_batchedVu1SynchronousCommandBarrier{};
    std::optional<Vu1VifState> m_batchedVu1VifState;
    std::optional<Vu1VifState> m_knownVu1VifState;
    std::vector<Vu1DecodedUnpackCommand>
        m_batchedVu1DecodedUnpacks;
    std::optional<Vu1VifState>
        m_batchedVu1DecodedUnpackFinalVifState;
    uint64_t m_batchedVu1DecodedUnpackPayloadBytes = 0u;
    bool m_vu1SynchronousCommandBatchActive = false;
    std::atomic<bool> m_vu1AsyncPendingSlice{false};
    std::atomic<bool> m_vu1AsyncDeferredSlice{false};
    std::atomic<uint64_t> m_vu1AsyncSlicesSubmitted{0u};
    std::atomic<uint64_t> m_vu1AsyncSlicesPublished{0u};
    std::atomic<uint64_t> m_vu1AsyncResultsReadyAtEvent{0u};
    std::atomic<uint64_t> m_vu1AsyncResultsLateAtEvent{0u};
    std::atomic<uint64_t> m_vu1AsyncEventWaitCount{0u};
    std::atomic<uint64_t> m_vu1AsyncEventWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1AsyncMaximumEventWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetFallbackCount{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetFallbackCycleDelta{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetFallbackWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetExtensionCount{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetExtensionCycleDelta{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetExtensionExecutedCycles{0u};
    std::atomic<uint64_t> m_vu1AsyncBudgetExtensionWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1AsyncHazardBarrierCount{0u};
    std::atomic<uint64_t> m_vu1AsyncHazardWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1AsyncHazardRequeueCount{0u};
    std::atomic<uint64_t> m_vu1AsyncDeferredSliceCount{0u};
    std::atomic<uint64_t> m_vu1AsyncDeferredSliceArmCount{0u};
    std::atomic<uint64_t>
        m_vu1AsyncForcedDeferredSliceArmCount{0u};
    std::atomic<uint64_t> m_vu1AsyncSliceSubmitCount{0u};
    std::atomic<uint64_t> m_vu1AsyncSliceSubmitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1AsyncMaximumSliceSubmitNanoseconds{0u};
    std::atomic<size_t> m_vu1CoarseEstimatorHistorySize{0u};
    std::atomic<bool> m_vu1CoarsePendingInvocation{false};
    std::atomic<size_t> m_vu1CoarseEstimatorHistoryHighWater{0u};
    std::atomic<size_t> m_vu1CoarseEstimatorIdentitySize{0u};
    std::atomic<size_t> m_vu1CoarseEstimatorIdentityHighWater{0u};
    std::atomic<uint64_t> m_vu1CoarseEstimatorIdentityHitCount{0u};
    std::atomic<uint64_t> m_vu1CoarseEstimatorIdentityMissCount{0u};
    std::atomic<size_t> m_vu1CoarsePendingInvocationHighWater{0u};
    std::atomic<uint64_t> m_vu1CoarseInvocationsSubmitted{0u};
    std::atomic<uint64_t> m_vu1CoarseInvocationsCompleted{0u};
    std::atomic<uint64_t> m_vu1CoarseInvocationsPublished{0u};
    std::atomic<uint64_t>
        m_vu1CoarseInvocationsDiscardedAtFailure{0u};
    std::atomic<uint64_t>
        m_vu1CoarseInvocationsDiscardedAtShutdown{0u};
    std::atomic<uint64_t> m_vu1CoarseResultsReadyAtEvent{0u};
    std::atomic<uint64_t> m_vu1CoarseResultsLateAtEvent{0u};
    std::atomic<uint64_t> m_vu1CoarseEarlyEstimateCount{0u};
    std::atomic<uint64_t> m_vu1CoarseExactEstimateCount{0u};
    std::atomic<uint64_t> m_vu1CoarseLateEstimateCount{0u};
    std::atomic<uint64_t> m_vu1CoarseEstimatedCycleSum{0u};
    std::atomic<uint64_t> m_vu1CoarseActualCycleSum{0u};
    std::atomic<uint64_t> m_vu1CoarseAbsoluteErrorCycleSum{0u};
    std::atomic<uint64_t> m_vu1CoarseMaximumAbsoluteErrorCycles{0u};
    std::atomic<uint64_t> m_vu1CoarseCausalDeadlineDeferralCount{0u};
    std::atomic<uint64_t> m_vu1CoarseCausalDeadlineDeferredCycles{0u};
    std::atomic<uint64_t>
        m_vu1CoarseMaximumCausalDeadlineDeferredCycles{0u};
    std::atomic<uint64_t> m_vu1CoarseForcedBarrierCount{0u};
    std::array<std::atomic<uint64_t>, kVu1CommandTypeCount>
        m_vu1CoarseForcedBarriersByCommandType{};
    std::atomic<uint64_t>
        m_vu1CoarseNonPublishingDiagnosticSnapshotCount{0u};
    std::atomic<uint64_t>
        m_vu1CoarseNonPublishingDiagnosticsUpdateCount{0u};
    std::atomic<uint64_t>
        m_vu1CoarseNonPublishingBackendChangeCount{0u};
    std::atomic<uint64_t>
        m_vu1CoarseNonPublishingVifStateUpdateCount{0u};
    std::atomic<uint64_t>
        m_vu1CoarseNonPublishingDiagnosticSnapshotWaitCount{0u};
    std::atomic<uint64_t>
        m_vu1CoarseNonPublishingDiagnosticSnapshotWaitNanoseconds{0u};
    std::atomic<uint64_t>
        m_vu1CoarseMaximumNonPublishingDiagnosticSnapshotWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1CoarseForcedWaitCount{0u};
    std::atomic<uint64_t> m_vu1CoarseForcedWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1CoarseMaximumForcedWaitNanoseconds{0u};
    std::atomic<uint64_t> m_vu1CoarsePath1ReservationsStarted{0u};
    std::atomic<uint64_t> m_vu1CoarsePath1ReservationsReleased{0u};
    std::atomic<uint64_t> m_vu1CoarsePath1ReservationsInvalidated{0u};
    std::atomic<uint64_t>
        m_vu1CoarsePath1ReservationsDiscardedAtFailure{0u};
    std::atomic<uint64_t>
        m_vu1CoarsePath1ReservationsDiscardedAtShutdown{0u};
    std::atomic<uint64_t> m_vu1CoarsePath1BytesPublished{0u};
    std::atomic<uint64_t> m_vu1CoarseMaximumInvocationPath1Bytes{0u};
    std::atomic<uint32_t> m_vu1CoarseLastEstimatedCycles{0u};
    std::atomic<uint32_t> m_vu1CoarseLastActualCycles{0u};
    std::atomic<int64_t> m_vu1CoarseLastEstimateErrorCycles{0};
    std::atomic<uint64_t> m_vu1CoarseEstimatorInputDigest{0u};
    std::atomic<uint64_t> m_vu1CoarseEstimatorOutputDigest{0u};
    std::atomic<uint64_t> m_vu1CoarseEstimatorResultDigest{0u};
    ps2x::timing::Cop0Timing m_cop0Timing;
    static constexpr size_t kEeBranchHistoryEntries = 4096u;
    std::array<uint32_t, kEeBranchHistoryEntries>
        m_eeBranchHistoryTags{};
    std::array<uint8_t, kEeBranchHistoryEntries>
        m_eeBranchHistoryStates{};
    ps2x::timing::EeTimeline m_eeTimeline;
    ps2x::timing::EeEventScheduler m_eeEventScheduler;
    struct Vu1ExecutionTiming
    {
        uint64_t generation = 0u;
        ps2x::timing::EeTick lastAdvancedTick{};
        uint64_t totalAdvancedCycles = 0u;
        ps2x::timing::EeEventToken eventToken{
            ps2x::timing::EeEventSource::VifVu1Finish, 0u};
    };
    Vu1ExecutionTiming m_vu1ExecutionTiming{};
    ps2x::timing::EeEventToken
        m_vif0VuFinishEventToken{
            ps2x::timing::EeEventSource::VifVu0Finish,
            0u};
    ps2x::timing::EeEventToken m_vif0DmaEventToken{
        ps2x::timing::EeEventSource::DmacVif0, 0u};
    ps2x::timing::EeEventToken m_vif1DmaEventToken{
        ps2x::timing::EeEventSource::DmacVif1, 0u};
    ps2x::timing::EeEventToken m_gifDmaEventToken{
        ps2x::timing::EeEventSource::DmacGif, 0u};
    ps2x::timing::EeEventToken
        m_fromIpuDmaEventToken{
            ps2x::timing::EeEventSource::DmacFromIpu,
            0u};
    ps2x::timing::EeEventToken
        m_toIpuDmaEventToken{
            ps2x::timing::EeEventSource::DmacToIpu,
            0u};
    struct HleSifDmaOperation
    {
        HleSifDmaSubmission submission{};
        uint64_t operationGeneration = 0u;
        uint64_t iopCycles = 0u;
    };
    std::array<
        HleSifDmaOperation,
        kHleSifDmaQueueCapacity>
        m_hleSifDmaQueue{};
    size_t m_hleSifDmaQueueHead = 0u;
    size_t m_hleSifDmaQueueCount = 0u;
    uint64_t m_hleSifDmaOperationGeneration = 0u;
    ps2x::timing::EeEventToken m_hleSifDmaEventToken{
        ps2x::timing::EeEventSource::HleSif1, 0u};
    ps2x::timing::EeEventToken
        m_fromScratchpadDmaEventToken{
            ps2x::timing::EeEventSource::
                DmacFromScratchpad,
            0u};
    ps2x::timing::EeEventToken
        m_toScratchpadDmaEventToken{
            ps2x::timing::EeEventSource::
                DmacToScratchpad,
            0u};
    enum class EeVSyncPhase : uint8_t
    {
        Start = 0u,
        GsBlank,
        End,
    };
    enum class EeVSyncVideoModeClass : uint8_t
    {
        Uninitialized = 0u,
        Unknown,
        Ntsc,
        Pal,
        Vesa,
        Sdtv480P,
        Sdtv576P,
        Hdtv720P,
        Hdtv1080I,
        Hdtv1080P,
        DvdNtsc,
        DvdPal,
    };
    struct EeVSyncDurations
    {
        // Before SetGsCrt, PCSX2 uses a temporary 60 Hz interlaced NTSC
        // counter shape. A later mode change preserves the current phase
        // deadline and changes only the durations used by following phases.
        uint64_t periodCycles = 4'915'200u;
        uint64_t renderCycles = 4'493'898u;
        uint64_t blankCycles = 421'302u;
        uint64_t gsBlankCycles = 65'535u;
        uint32_t hRenderCycles = 15'669u;
        uint32_t hBlankCycles = 3'055u;
        uint32_t hSyncErrorCycles = 149u;
    };
    struct EeVSyncTiming
    {
        EeVSyncPhase phase = EeVSyncPhase::Start;
        ps2x::timing::EeTick startTick{};
        ps2x::timing::EeEventToken eventToken{
            ps2x::timing::EeEventSource::VSync, 0u};
        EeVSyncDurations durations{};
        uint64_t currentVSyncTick = 0u;
        uint32_t videoMode = 0u;
        EeVSyncVideoModeClass videoModeClass =
            EeVSyncVideoModeClass::Uninitialized;
        bool videoModeConfigured = false;
        bool interlaced = true;
        bool enabled = false;
    };
    struct EeVSyncPacing
    {
        std::chrono::steady_clock::time_point hostDeadline{};
        ps2x::timing::EeTick eeDeadlineTick{};
        bool enabled = false;
        bool anchored = false;
    };
    enum class PendingVSyncKind : uint8_t
    {
        Start = 0u,
        End,
    };
    struct PendingVSyncDelivery
    {
        PendingVSyncKind kind = PendingVSyncKind::Start;
        uint64_t tick = 0u;
        uint64_t eventSequence = 0u;
    };
    static constexpr size_t
        kMaximumPendingVSyncDeliveries =
            ps2x::timing::EeEventScheduler::
                kMaximumServicesPerBoundary;
    EeVSyncTiming m_eeVSyncTiming{};
    EeVSyncPacing m_eeVSyncPacing{};
    std::atomic<uint64_t> m_debugVSyncPeriodCycles{4'915'200u};
    std::atomic<EeVSyncVideoModeClass>
        m_hostPresentationVideoModeClass{
            EeVSyncVideoModeClass::Ntsc};
    ps2x::timing::EeEventToken m_eeCounterEventToken{
        ps2x::timing::EeEventSource::EeCounters, 0u};
    ps2x::timing::EeEventToken m_cop0TimerEventToken{
        ps2x::timing::EeEventSource::Cop0Timer, 0u};
    ps2x::timing::EeEventToken
        m_cop0PerformanceEventToken{
            ps2x::timing::EeEventSource::
                Cop0Performance,
            0u};
    uint32_t m_pendingEeCounterInterrupts = 0u;
    std::array<
        PendingVSyncDelivery,
        kMaximumPendingVSyncDeliveries>
        m_pendingVSyncDeliveries{};
    size_t m_pendingVSyncDeliveryCount = 0u;
    // Executor-only callback publication state. A suspended guest fiber can
    // retain its outer GuestExecutionScope across many scheduler boundaries,
    // so the legacy scope destructor is not a boundary for the dedicated
    // backend. Drain published interrupt work once from each explicit
    // executor boundary instead.
    bool m_eeExecutorInterruptPassComplete = false;
    ps2x::timing::EeTick m_vu0CycleTick{};
    std::atomic<uint64_t> m_vu0InvocationSequence{0u};
    std::atomic<uint64_t> m_vu0CurrentInvocation{0u};
    std::atomic<uint64_t> m_vu0CurrentInvocationInstruction{0u};
    R5900Context m_cpuContext;
    std::unique_ptr<EeThreadRuntimeState> m_eeThreadRuntimeState;
    std::unique_ptr<IEeExecutionBackend>
        m_eeExecutionBackend;
    std::unique_ptr<ps2x::ee::EeRuntimeExecutor>
        m_eeRuntimeExecutor;
    mutable std::mutex
        m_headlessEeExecutionFailureMutex;
    std::exception_ptr
        m_headlessEeExecutionFailure;
    ps2x::timing::EeTick
        m_eeExecutorDispatchStartTick{};
    R5900Context *m_boundEeContext = nullptr;
    int m_boundEeThreadId = 1;
    std::optional<EeInterruptedContinuationFrame>
        m_eeInterruptedContinuation;
    uint32_t m_asyncCallbackInvocationDepth = 0u;
    mutable std::recursive_timed_mutex m_guestExecutionMutex;
    mutable std::atomic<uint32_t> m_guestExecutionWaiters{0u};
    mutable std::mutex m_legacyDeferredEeWakeMutex;
    std::vector<std::weak_ptr<ThreadInfo>>
        m_legacyDeferredEeWakeNotifications;
    mutable std::mutex m_guestExecutionHandoffMutex;
    mutable std::condition_variable m_guestExecutionHandoffCv;
    mutable std::mutex m_eeIdleAdvanceMutex;
    std::atomic<uint64_t> m_guestExecutionHandoffEpoch{0u};
    std::atomic<uint64_t> m_guestExecutionHandoffTimeouts{0u};
    std::atomic<uint64_t>
        m_guestExecutionDispatcherExitEpoch{0u};
    std::atomic<uint64_t> m_guestExecutionPreemptionGeneration{0u};
    std::atomic<uint64_t>
        m_guestExecutionConsumedPreemptionGeneration{0u};
    bool m_eeThreadDiagnosticsEnabled = false;
    bool m_emitTlbTranslationCacheDiagnosticsOnDestruction = false;
    bool m_eeThreadDiagnosticsHasLastOuterContext = false;
    R5900Context *m_eeThreadDiagnosticsLastOuterContext = nullptr;
    std::atomic<uint64_t> m_eeThreadDiagnosticGuestLockRequests{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticGuestLockAcquisitions{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticGuestLockContentions{0u};
    std::atomic<uint64_t>
        m_eeThreadDiagnosticOuterGuestExecutionAcquisitions{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticGuestContextChanges{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHandoffNotifications{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHandoffWaitRequests{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHandoffWaitFastPaths{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHandoffCvWaits{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHandoffCompletions{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHandoffTimeouts{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticYieldRequests{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticDeferredYields{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticHostThreadYields{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticRequestedGuestSwitches{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticGuestSwitchCvWaits{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticCompletedGuestSwitches{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticGuestSwitchTimeouts{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticRotationRequests{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticAcceptedRotationRequests{0u};
    std::atomic<uint64_t> m_eeThreadDiagnosticRejectedRotationRequests{0u};
    std::atomic<uint64_t>
        m_eeThreadDiagnosticPriorityZeroRotationRequests{0u};
    std::atomic<uint64_t>
        m_eeThreadDiagnosticUntrackedThreadRotationRequests{0u};
    std::atomic<uint64_t>
        m_eeThreadDiagnosticAcceptedRotationSequenceHash{
            kEeThreadDiagnosticRotationSequenceHashSeed};
    std::array<
        std::atomic<uint64_t>,
        kEeThreadDiagnosticPriorityCount>
        m_eeThreadDiagnosticAcceptedRotationsByPriority{};
    std::array<
        std::atomic<uint64_t>,
        kEeThreadDiagnosticThreadCount>
        m_eeThreadDiagnosticAcceptedRotationsByThread{};
    std::atomic<uint64_t> m_debugVSyncFields{0u};
    std::atomic<uint64_t> m_debugMpegPicturesServed{0u};
    std::atomic<uint64_t> m_debugMpegUniquePicturesServed{0u};
    std::atomic<uint64_t> m_debugMpegRepeatedPicturesServed{0u};
    std::atomic<bool> m_dmacCompletionReady{false};
    std::atomic<bool> m_dmacInterruptDeliveryDirty{false};
    std::atomic<uint64_t> m_dmacInterruptDrainPasses{0u};
    std::vector<DmacPendingInterrupt> m_pendingDmacInterrupts;
    mutable std::mutex m_guestHeapMutex;
    std::vector<GuestHeapBlock> m_guestHeapBlocks;
    uint32_t m_guestHeapBase = 0x00100000u;
    uint32_t m_guestHeapEnd = 0x00100000u;
    uint32_t m_guestHeapLimit = PS2_RAM_SIZE;
    uint32_t m_guestHeapSuggestedBase = 0x00100000u;
    bool m_guestHeapConfigured = false;

    std::atomic<uint32_t> m_missingFunctionPolicy{static_cast<uint32_t>(MissingFunctionPolicy::ContinueToTarget)};
    std::atomic<bool> m_missingFunctionReported{false};
    std::atomic<bool> m_stopRequested{false};
    std::atomic<bool> m_debugControlActive{false};
    std::atomic<bool> m_debugPauseRequested{false};
    std::atomic<bool> m_debugWatchpointsActive{false};
    std::atomic<uint64_t> m_debugDispatches{0u};
    std::atomic<uint64_t> m_debugEeInstructions{0u};
    mutable std::mutex m_debugVuBackendDiagnosticsMutex;
    std::array<DebugVuBackendDiagnostics, 2u>
        m_debugVuBackendDiagnostics{};
    uint64_t m_debugVuBackendDiagnosticsSequence = 0u;
    static constexpr size_t kDebugBranchHistoryCapacity = 256u;
    struct DebugBranchSlot
    {
        std::atomic<uint64_t> sequence{0u};
        std::atomic<uint32_t> pc{0u};
    };
    std::array<DebugBranchSlot, kDebugBranchHistoryCapacity>
        m_debugBranchHistory{};
    std::atomic<uint64_t> m_debugBranchSequence{0u};
    mutable std::mutex m_debugFaultMutex;
    DebugFaultInfo m_debugFault{};
    std::atomic<uint64_t> m_debugFaultSequence{0u};
    std::atomic<bool>
        m_debugEePcTraceTriggerEverConfigured{false};
    std::atomic<bool> m_debugVu0SyncTraceEnabled{false};
    std::atomic<bool> m_debugVu0SyncTraceTriggered{false};
    std::atomic<bool> m_debugVu0SyncTraceHasTrigger{false};
    std::atomic<uint32_t> m_debugVu0SyncTraceTriggerEePc{0u};
    std::atomic<bool>
        m_debugVu0SyncTraceHasInvocationTrigger{false};
    std::atomic<uint64_t>
        m_debugVu0SyncTraceTriggerInvocation{0u};
    mutable std::mutex m_debugVu0SyncTraceMutex;
    std::vector<DebugVu0SyncEntry> m_debugVu0SyncTrace;
    size_t m_debugVu0SyncTraceCapacity = 0u;
    size_t m_debugVu0SyncTraceNext = 0u;
    uint64_t m_debugVu0SyncTraceTotal = 0u;
    bool m_debugVu0SyncTraceStopOnFull = false;
    bool m_debugVu0SyncTraceHasTriggerSchedulerSnapshot = false;
    uint64_t m_debugVu0SyncTraceTriggerVsyncTick = 0u;
    DebugEeScheduler m_debugVu0SyncTraceTriggerScheduler{};
    std::atomic<bool> m_debugVu0InstructionTraceEnabled{false};
    std::atomic<bool> m_debugVu0InstructionTraceTriggered{false};
    std::atomic<bool> m_debugVu0InstructionTraceHasTrigger{false};
    std::atomic<uint32_t> m_debugVu0InstructionTraceTriggerEePc{0u};
    std::atomic<bool>
        m_debugVu0InstructionTraceHasInvocationTrigger{false};
    std::atomic<uint64_t>
        m_debugVu0InstructionTraceTriggerInvocation{0u};
    mutable std::mutex m_debugVu0InstructionTraceMutex;
    std::vector<DebugVu0InstructionEntry> m_debugVu0InstructionTrace;
    size_t m_debugVu0InstructionTraceCapacity = 0u;
    size_t m_debugVu0InstructionTraceNext = 0u;
    uint64_t m_debugVu0InstructionTraceTotal = 0u;
    bool m_debugVu0InstructionTraceStopOnFull = false;
    std::atomic<bool> m_debugEeEventTraceEnabled{false};
    std::atomic<bool> m_debugEeEventTraceTriggered{false};
    std::atomic<bool> m_debugEeEventTraceHasTrigger{false};
    std::atomic<uint32_t> m_debugEeEventTraceTriggerEePc{0u};
    mutable std::mutex m_debugEeEventTraceMutex;
    std::vector<DebugEeEventEntry> m_debugEeEventTrace;
    size_t m_debugEeEventTraceCapacity = 0u;
    size_t m_debugEeEventTraceNext = 0u;
    uint64_t m_debugEeEventTraceTotal = 0u;
    bool m_debugEeEventTraceStopOnFull = false;
    struct DebugVu1TimingInvocation
    {
        uint64_t sequence = 0u;
        Vu1InvocationKind kind = Vu1InvocationKind::Mscal;
        uint32_t startPc = 0u;
        uint32_t top = 0u;
        uint32_t itop = 0u;
        uint32_t estimatedCycles = 0u;
        ps2x::timing::EeTick startTick{};
        ps2x::timing::EeTick deadline{};
        uint64_t eventGeneration = 0u;
        Vu1ArchitecturalStateHashes inputStateHashes{};
    };
    DebugVu1TimingInvocation m_debugVu1TimingInvocation{};
    uint64_t m_debugVu1TimingNextInvocation = 0u;
    uint64_t m_debugVu1TimingNextPath1Packet = 0u;
    mutable std::atomic<bool>
        m_debugVu1TimingTraceEnabled{false};
    mutable std::mutex m_debugVu1TimingTraceMutex;
    mutable std::vector<DebugVu1TimingEntry>
        m_debugVu1TimingTrace;
    mutable size_t m_debugVu1TimingTraceCapacity = 0u;
    mutable size_t m_debugVu1TimingTraceNext = 0u;
    mutable uint64_t m_debugVu1TimingTraceTotal = 0u;
    mutable bool m_debugVu1TimingTraceStopOnFull = false;
    mutable std::mutex m_debugControlMutex;
    mutable std::condition_variable m_debugControlCv;
    uint64_t m_debugStopSequence = 0u;
    DebugStopInfo m_debugLastStop{};
    std::string m_debugPauseSource;
    std::string m_debugPauseWriter{"initial"};
    uint64_t m_debugControlGeneration = 0u;
    uint64_t m_debugStaleBoundaryStops = 0u;
    bool m_debugRunUntilActive = false;
    uint32_t m_debugRunUntilPc = 0u;
    bool m_debugRunUntilMpegUniquePicturesActive = false;
    uint64_t m_debugRunUntilMpegUniquePicturesTarget = 0u;
    bool m_debugRunBeforeNextMpegUniquePictureActive = false;
    uint64_t m_debugRunBeforeNextMpegUniquePictureCurrent = 0u;
    bool m_debugRunForVSyncFieldsActive = false;
    uint64_t m_debugRunForVSyncFieldsTarget = 0u;
    bool m_debugStepActive = false;
    uint64_t m_debugStepRemaining = 0u;
    bool m_debugSkipBreakpoint = false;
    uint32_t m_debugSkipBreakpointPc = 0u;
    std::vector<uint32_t> m_debugBreakpoints;
    std::vector<DebugWatchpoint> m_debugWatchpoints;
    uint64_t m_debugNextWatchpointId = 1u;
    DebugUiCallback m_debugUiInitCallback = nullptr;
    DebugUiCallback m_debugUiDrawCallback = nullptr;
    DebugUiCallback m_debugUiShutdownCallback = nullptr;
    void *m_debugUiUserData = nullptr;
    std::string m_windowTitle{"PS2 Game"};
    std::optional<int> m_hostMonitorIndex;

    bool m_debugUiInitialized = false;

public:
    std::atomic<uint32_t> m_debugPc{0};
    std::atomic<uint32_t> m_debugRa{0};
    std::atomic<uint32_t> m_debugSp{0};
    std::atomic<uint32_t> m_debugGp{0};

private:
    struct LoadedModule
    {
        std::string name;
        uint32_t baseAddress;
        size_t size;
        bool active;
    };

    std::vector<LoadedModule> m_loadedModules;
    uint8_t *m_boundRdram = nullptr;
    uint8_t *m_boundGSVram = nullptr;
#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
    std::unique_ptr<PS2DebugServer> m_debugServer;
#endif
    std::unique_ptr<ps2_stubs::MpegRuntimeState>
        m_mpegRuntimeState;
    std::unique_ptr<EeSyncRuntimeState> m_eeSyncRuntimeState;
    std::unique_ptr<EeInterruptRuntimeState>
        m_eeInterruptRuntimeState;
    std::unique_ptr<EeKernelRuntimeState>
        m_eeKernelRuntimeState;
    std::unique_ptr<EeRpcRuntimeState>
        m_eeRpcRuntimeState;
    std::unique_ptr<EeFileRuntimeState>
        m_eeFileRuntimeState;
    std::unique_ptr<Deci2RuntimeState>
        m_deci2RuntimeState;
    std::unique_ptr<ps2_stubs::AudioRuntimeState>
        m_audioRuntimeState;
    std::unique_ptr<ps2_stubs::CdRuntimeState>
        m_cdRuntimeState;
    std::unique_ptr<ps2_stubs::DmaRuntimeState>
        m_dmaRuntimeState;
    std::unique_ptr<ps2_stubs::LibCRuntimeState>
        m_libcRuntimeState;
    std::unique_ptr<ps2_stubs::MemoryCardRuntimeState>
        m_memoryCardRuntimeState;
    std::unique_ptr<ps2_stubs::PadRuntimeState>
        m_padRuntimeState;
    std::unique_ptr<ps2_stubs::SifRuntimeState>
        m_sifRuntimeState;
    // Declared last so the alarm worker's fallback join runs before other
    // runtime members are destroyed if normal stop cleanup was interrupted.
    std::unique_ptr<EeAlarmRuntimeState> m_eeAlarmRuntimeState;
};

static_assert(
    sizeof(PS2Runtime::RecompiledFunctionPair) ==
        2u * sizeof(PS2Runtime::RecompiledFunction),
    "dense EE dispatch slots must contain only fast and precise pointers");

// A fixed-address executable module produced separately from the primary ELF.
// Generated modules are selected only while their immutable match bytes are
// present in guest RDRAM, allowing overlays at the same virtual addresses to
// coexist in one native runner.
struct PS2RecompiledModuleFunction
{
    uint32_t address;
    PS2Runtime::RecompiledFunctionPair functions;
};

struct PS2RecompiledModuleDescriptor
{
    uint32_t abiVersion;
    const char *name;
    uint32_t matchAddress;
    const uint8_t *matchBytes;
    uint32_t matchSize;
    const PS2RecompiledModuleFunction *functions;
    uint32_t functionCount;
    const PS2GuestFunctionSymbol *symbols;
    uint32_t symbolCount;
};

// Intended for generated static initializers. Descriptors and every referenced
// array must remain alive for the process lifetime.
bool ps2RegisterRecompiledModule(
    const PS2RecompiledModuleDescriptor *descriptor) noexcept;

// Generated by ps2xRecomp in ps2xRuntime/src/runner/register_functions.cpp.
extern const uint32_t g_ps2RecompiledFunctionPairAbiVersion;
extern const uint32_t g_ps2RecompiledFunctionTableBase;
extern const uint32_t g_ps2RecompiledFunctionTableEnd;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount;
extern PS2Runtime::RecompiledFunctionPair g_ps2RecompiledFunctionTable[];
extern const uint32_t g_ps2GuestFunctionSymbolCount;
extern const PS2GuestFunctionSymbol g_ps2GuestFunctionSymbols[];

#endif // PS2_RUNTIME_H
