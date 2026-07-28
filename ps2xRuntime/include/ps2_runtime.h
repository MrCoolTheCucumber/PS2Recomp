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
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <memory>
#include <optional>

#include "ps2_log.h"
#include "runtime/cop0_timing.h"
#include "runtime/ee_event_scheduler.h"
#include "runtime/ps2_address.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_vu_clip.h"
#include "runtime/ps2_vu1.h"
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

enum PS2Exception
{
    EXCEPTION_INTERRUPT = 0x00,
    EXCEPTION_TLB_MODIFIED = 0x01,
    EXCEPTION_TLB_REFILL_LOAD = 0x02,
    EXCEPTION_TLB_REFILL = EXCEPTION_TLB_REFILL_LOAD,
    EXCEPTION_TLB_REFILL_STORE = 0x03,
    EXCEPTION_ADDRESS_ERROR_LOAD = 0x04,  // Address error on load
    EXCEPTION_ADDRESS_ERROR_STORE = 0x05, // Address error on store
    EXCEPTION_SYSCALL = 0x08,             // SYSCALL instruction
    EXCEPTION_BREAKPOINT = 0x09,          // BREAK instruction
    EXCEPTION_RESERVED_INSTRUCTION = 0x0A,
    EXCEPTION_INTEGER_OVERFLOW = 0x0C, // From MIPS spec
    EXCEPTION_TRAP = 0x0D,             // Trap instruction condition met
};

// Internal control-flow signal used to leave generated guest code immediately
// after the runtime has entered an EE exception.
struct PS2GuestException final
{
};

struct PS2GuestFunctionSymbol
{
    uint32_t start;
    uint32_t end;
    const char *name;
};

// PS2 CPU context (R5900)
struct alignas(16) R5900Context
{
    // General Purpose Registers (128-bit)
    __m128i r[32]; // Main registers

    // Control registers
    uint32_t pc;         // Program counter
    uint64_t insn_count; // Retired instruction counter
    uint32_t ee_block_cycle_ticks;
    bool ee_block_cycle_active;
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
    uint32_t cop0_debug;
    uint32_t cop0_perf;
    uint32_t cop0_pcr0;
    uint32_t cop0_pcr1;
    uint32_t cop0_taglo;
    uint32_t cop0_taghi;
    uint32_t cop0_errorepc;

    // LL/SC reservation state (not part of COP0 Status bits).
    uint32_t llbit;
    uint32_t lladdr;

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
        ee_block_cycle_ticks = 0u;
        ee_block_cycle_active = false;
    }

    R5900Context()
    {
        std::memset(this, 0, sizeof(*this));

        // Initialize VU0 registers
        vu0_q = 1.0f; // Q register usually initialized to 1.0
        enforceVu0RegisterInvariants();

        // Reset COP0 registers
        cop0_random = 47; // Start at maximum value
        // cop0_status = 0x400000; // BEV set, ERL clear, kernel mode
        // 0x00400000 = BEV (Boot Exception Vectors).
        // 0x00000000 = Normal mode (after BIOS handoff).
        cop0_status = 0x00000000;
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
    (void)rdram;
    (void)valueLo;
    (void)valueHi;
    (void)op;
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
}

inline void ps2TraceGuestRangeWrite(uint8_t *rdram,
                                    uint32_t guestAddr,
                                    uint32_t size,
                                    const char *op,
                                    const R5900Context *ctx)
{
    (void)rdram;
    (void)op;
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
}

class PS2Runtime
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
    ~PS2Runtime();

    bool initialize(const char *title = "PS2 Game");
    bool syncCoreSubsystems();
    [[nodiscard]] ps2x::timing::EeTick currentEeTick() const noexcept;
    void resetEeTiming(R5900Context *context = nullptr);
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
        uint32_t pc = 0u;
        uint32_t ra = 0u;
        uint32_t sp = 0u;
        uint32_t gp = 0u;
        uint32_t guestExecutionWaiters = 0u;
        uint64_t guestExecutionHandoffTimeouts = 0u;
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
    };

    struct DebugVu0InstructionTrace
    {
        bool enabled = false;
        bool triggered = false;
        bool stopOnFull = false;
        std::optional<uint32_t> triggerEePc;
        uint64_t totalEntries = 0u;
        uint64_t droppedEntries = 0u;
        std::vector<DebugVu0InstructionEntry> entries;
    };

    class GuestExecutionScope
    {
    public:
        explicit GuestExecutionScope(
            PS2Runtime *runtime,
            R5900Context *context = nullptr) noexcept;
        ~GuestExecutionScope();

        GuestExecutionScope(const GuestExecutionScope &) = delete;
        GuestExecutionScope &operator=(const GuestExecutionScope &) = delete;

    private:
        PS2Runtime *m_runtime = nullptr;
        R5900Context *m_context = nullptr;
        R5900Context *m_previousContext = nullptr;
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
        uint32_t m_depth = 0u;
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

    bool replaceFunction(uint32_t address, RecompiledFunction func);
    // TODO remove this later need to update all tests
    bool registerFunction(uint32_t address, RecompiledFunction func);
    RecompiledFunction lookupFunction(uint32_t address);
    bool hasFunction(uint32_t address) const;
    static std::string formatGuestPc(uint32_t address);
    bool dispatchGuestBranch(uint8_t *rdram,
                             R5900Context *ctx,
                             uint32_t targetPc,
                             uint32_t sourcePc,
                             uint32_t fallthroughPc,
                             GuestBranchKind kind,
                             const char *debugName);
    void reportMissingFunction(uint8_t *rdram,
                               R5900Context *ctx,
                               uint32_t targetPc,
                               uint32_t sourcePc,
                               GuestBranchKind kind,
                               const char *debugName);
    void setMissingFunctionPolicy(MissingFunctionPolicy policy);
    MissingFunctionPolicy missingFunctionPolicy() const;
    void resetMissingFunctionReportOnce();

    static const IoPaths &getIoPaths();
    static void setIoPaths(const IoPaths &paths);
    static void configureIoPathsFromElf(const std::string &elfPath);

    [[noreturn]] void SignalException(R5900Context *ctx, PS2Exception exception);
    [[noreturn]] void SignalMemoryException(R5900Context *ctx,
                                            PS2Exception exception,
                                            uint32_t badVAddr);

    void executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address);
    void vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address);
    void synchronizeVU0Microprogram(uint8_t *rdram,
                                    R5900Context *ctx,
                                    bool interlocked);
    void serviceEeEventsAtBlockBoundary(uint8_t *rdram,
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
    void clearLLBit(R5900Context *ctx);
    void configureGuestHeap(uint32_t guestBase, uint32_t guestLimit = PS2_RAM_SIZE);
    uint32_t guestMalloc(uint32_t size, uint32_t alignment = 16u);
    uint32_t guestCalloc(uint32_t count, uint32_t size, uint32_t alignment = 16u);
    uint32_t guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment = 16u);
    void guestFree(uint32_t guestAddr);
    uint32_t guestHeapBase() const;
    uint32_t guestHeapEnd() const;
    uint32_t guestHeapLimit() const;
    uint32_t reserveAsyncCallbackStack(uint32_t size, uint32_t alignment = 16u);

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
    bool shouldPreemptGuestExecution();
    void yieldGuestExecutionAfterWake();
    void waitForGuestExecutionHandoff();
    void waitForGuestExecutionHandoff(uint64_t baselineEpoch);
    uint64_t guestExecutionHandoffEpochSnapshot() const
    {
        return m_guestExecutionHandoffEpoch.load(std::memory_order_acquire);
    }

    void requestStop();
    bool isStopRequested() const;

    // Debug-control operations are intentionally backend-neutral. They are used
    // by the local ps2dbg server and by focused runtime tests.
    bool debugPause(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    void debugResume();
    bool debugIsPaused() const;
    DebugStopInfo debugRunUntilPc(uint32_t pc, std::chrono::milliseconds timeout);
    DebugStopInfo debugStepDispatches(uint64_t count, std::chrono::milliseconds timeout);
    R5900Context debugCpuSnapshot();
    DebugEeTiming debugEeTimingSnapshot();
    DebugEeScheduler debugEeSchedulerSnapshot();
    bool debugReadRdram(uint32_t address, uint32_t size, std::vector<uint8_t> &output);
    bool debugReadMemory(uint32_t address, uint32_t size, std::vector<uint8_t> &output);
    bool debugCopyGsVram(std::vector<uint8_t> &output);
    DebugRuntimeProgress debugRuntimeProgress() const;
    DebugVu1Timing debugVu1TimingSnapshot();
    std::vector<DebugBranchEntry> debugBranchHistory(
        size_t maximumEntries = 256u) const;
    DebugFaultInfo debugFaultSnapshot() const;
    void debugClearFault();
    void debugStartVu0SyncTrace(
        size_t maximumEntries,
        std::optional<uint32_t> triggerEePc = std::nullopt,
        bool stopOnFull = false);
    DebugVu0SyncTrace debugVu0SyncTraceSnapshot(bool stop);
    void debugStartVu0InstructionTrace(
        size_t maximumEntries,
        std::optional<uint32_t> triggerEePc = std::nullopt,
        bool stopOnFull = false);
    DebugVu0InstructionTrace debugVu0InstructionTraceSnapshot(bool stop);
    void debugStartEeEventTrace(
        size_t maximumEntries,
        std::optional<uint32_t> triggerEePc = std::nullopt,
        bool stopOnFull = false);
    DebugEeEventTrace debugEeEventTraceSnapshot(bool stop);

    std::vector<uint32_t> debugBreakpoints() const;
    void debugAddBreakpoint(uint32_t address);
    void debugRemoveBreakpoint(uint32_t address);

    std::vector<DebugWatchpoint> debugWatchpoints() const;
    uint64_t debugAddWatchpoint(uint32_t start, uint32_t size, DebugMemoryAccess access);
    bool debugRemoveWatchpoint(uint64_t id);
    bool debugWatchpointsEnabled() const;
    void debugObserveMemoryAccess(uint32_t address,
                                  uint32_t size,
                                  DebugMemoryAccess access,
                                  const R5900Context *ctx);

    uint32_t guestExecutionWaiterCountForTesting() const
    {
        return m_guestExecutionWaiters.load(std::memory_order_acquire);
    }
    bool guestPreemptionRequestedForTesting() const
    {
        return m_guestExecutionPreemptionRequested.load(
            std::memory_order_acquire);
    }
    uint64_t dmacInterruptDrainPassesForTesting() const
    {
        return m_dmacInterruptDrainPasses.load(
            std::memory_order_acquire);
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
    inline VU1Interpreter &vu0() { return m_vu0; }
    inline const VU1Interpreter &vu0() const { return m_vu0; }
    inline VU1Interpreter &vu1() { return m_vu1; }
    inline const VU1Interpreter &vu1() const { return m_vu1; }

    inline PS2AudioBackend &audioBackend() { return m_audioBackend; }
    inline const PS2AudioBackend &audioBackend() const { return m_audioBackend; }
    inline PSPadBackend &padBackend() { return m_padBackend; }
    inline const PSPadBackend &padBackend() const { return m_padBackend; }

private:
    struct EeVSyncDurations;
    enum class EeVSyncVideoModeClass : uint8_t;
    struct PendingVSyncDelivery;

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
    R5900Context *enterGuestExecution(R5900Context *context);
    void leaveGuestExecution(
        R5900Context *context,
        R5900Context *previousContext);
    uint32_t releaseGuestExecution(R5900Context *&context);
    void reacquireGuestExecution(
        uint32_t depth,
        R5900Context *context);
    void markGuestExecutionAcquired();
    void debugBeforeGuestStep(R5900Context *ctx);
    void debugAfterGuestStep(R5900Context *ctx);
    void debugBlockGuestAtBoundary(R5900Context *ctx, const char *reason);
    void debugWaitUntilResumed();
    void debugRecordStopLocked(const char *reason, uint32_t pc);
    void debugRefreshControlActiveLocked();
    void debugRecordBranch(uint32_t pc);
    void debugRecordFault(const DebugFaultInfo &fault);
    void debugRecordVu0Sync(DebugVu0SyncEntry entry);
    void debugRecordVu0Instruction(DebugVu0InstructionEntry entry);
    void debugRecordEeEvent(DebugEeEventEntry entry);
    [[nodiscard]] DebugEeEventDeviceState
    debugEeEventDeviceState(
        ps2x::timing::EeEventSource source) const noexcept;
    void debugArmVu0Traces(const R5900Context *ctx);
    void beginVu0Invocation();

    [[noreturn]] void HandleIntegerOverflow(R5900Context *ctx);

    [[nodiscard]] ps2x::iop::RpcAbi selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const;
    [[nodiscard]] ps2x::iop::RpcResult handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request);
    void notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer);
    void resetIop();

    friend class GuestExecutionScope;
    friend class GuestExecutionReleaseScope;
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
        uint32_t startPC, uint32_t top, uint32_t itop);
    void resumeVU1FromVif(uint32_t top, uint32_t itop);
    void serviceVU1AtEvent(
        R5900Context *ctx,
        const ps2x::timing::EeEventService &service);
    void scheduleVU1Event(
        ps2x::timing::EeTick deadline) noexcept;
    void cancelVU1Execution(bool resetInterpreter);
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
    void publishEeVSyncField() noexcept;
    void resetEeVSyncStateUnlocked() noexcept;
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
    [[nodiscard]] bool cop0TimerInterruptEnabled(
        const R5900Context *ctx) const noexcept;
    [[noreturn]] void raiseCop0Level1Exception(
        R5900Context *ctx,
        uint32_t exceptionCode,
        bool commitContextProgress = true);
    [[noreturn]] void raiseCop0PerformanceException(
        R5900Context *ctx);
    [[nodiscard]] ps2x::timing::EeTick publishEeElapsed(
        ps2x::timing::EeTickDelta elapsed) noexcept;
    [[nodiscard]] ps2x::timing::EeTick commitEeContextProgress(
        R5900Context *context) noexcept;
    [[nodiscard]] ps2x::timing::EeTick finishEeContextBlock(
        R5900Context *context) noexcept;
    void switchGuestExecutionContext(R5900Context *context) noexcept;
    void restoreGuestExecutionContext(
        R5900Context *context,
        R5900Context *previousContext) noexcept;
    void resetEeTimingUnlocked(R5900Context *context) noexcept;

    PS2Memory m_memory;
    GifArbiter m_gifArbiter;
    GS m_gs;
    std::unique_ptr<PS2IopHostAdapter> m_iopHost;
    std::unique_ptr<ps2x::iop::IopSubsystem> m_iopSubsystem;
    PS2AudioBackend m_audioBackend;
    PSPadBackend m_padBackend;
    VU1Interpreter m_vu0;
    VU1Interpreter m_vu1;
    ps2x::timing::Cop0Timing m_cop0Timing;
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
    ps2x::timing::EeTick m_vu0CycleTick{};
    std::atomic<uint64_t> m_vu0InvocationSequence{0u};
    std::atomic<uint64_t> m_vu0CurrentInvocation{0u};
    std::atomic<uint64_t> m_vu0CurrentInvocationInstruction{0u};
    R5900Context m_cpuContext;
    R5900Context *m_boundEeContext = nullptr;
    mutable std::recursive_timed_mutex m_guestExecutionMutex;
    mutable std::atomic<uint32_t> m_guestExecutionWaiters{0u};
    mutable std::mutex m_guestExecutionHandoffMutex;
    mutable std::condition_variable m_guestExecutionHandoffCv;
    mutable std::mutex m_eeIdleAdvanceMutex;
    std::atomic<uint64_t> m_guestExecutionHandoffEpoch{0u};
    std::atomic<uint64_t> m_guestExecutionHandoffTimeouts{0u};
    std::atomic<uint64_t> m_guestExecutionPreemptionEpoch{0u};
    std::atomic<bool> m_guestExecutionPreemptionRequested{false};
    std::atomic<bool> m_dmacCompletionReady{false};
    std::atomic<bool> m_dmacInterruptDeliveryDirty{false};
    std::atomic<uint64_t> m_dmacInterruptDrainPasses{0u};
    std::vector<DmacPendingInterrupt> m_pendingDmacInterrupts;
    mutable std::mutex m_guestHeapMutex;
    mutable std::mutex m_asyncCallbackStackMutex;
    std::vector<GuestHeapBlock> m_guestHeapBlocks;
    uint32_t m_guestHeapBase = 0x00100000u;
    uint32_t m_guestHeapEnd = 0x00100000u;
    uint32_t m_guestHeapLimit = PS2_RAM_SIZE;
    uint32_t m_guestHeapSuggestedBase = 0x00100000u;
    bool m_guestHeapConfigured = false;
    uint32_t m_asyncCallbackStackFloor = 0x01F00000u;
    uint32_t m_asyncCallbackStackTop = PS2_RAM_SIZE;

    std::atomic<uint32_t> m_missingFunctionPolicy{static_cast<uint32_t>(MissingFunctionPolicy::ContinueToTarget)};
    std::atomic<bool> m_missingFunctionReported{false};
    std::atomic<bool> m_stopRequested{false};
    std::atomic<bool> m_debugControlActive{false};
    std::atomic<bool> m_debugPauseRequested{false};
    std::atomic<bool> m_debugWatchpointsActive{false};
    std::atomic<uint64_t> m_debugDispatches{0u};
    std::atomic<uint64_t> m_debugEeInstructions{0u};
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
    std::atomic<bool> m_debugVu0SyncTraceEnabled{false};
    std::atomic<bool> m_debugVu0SyncTraceTriggered{false};
    std::atomic<bool> m_debugVu0SyncTraceHasTrigger{false};
    std::atomic<uint32_t> m_debugVu0SyncTraceTriggerEePc{0u};
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
    mutable std::mutex m_debugControlMutex;
    mutable std::condition_variable m_debugControlCv;
    uint64_t m_debugStopSequence = 0u;
    DebugStopInfo m_debugLastStop{};
    bool m_debugRunUntilActive = false;
    uint32_t m_debugRunUntilPc = 0u;
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
};

// Generated by ps2xRecomp in ps2xRuntime/src/runner/register_functions.cpp.
extern const uint32_t g_ps2RecompiledFunctionTableBase;
extern const uint32_t g_ps2RecompiledFunctionTableEnd;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount;
extern PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[];
extern const uint32_t g_ps2GuestFunctionSymbolCount;
extern const PS2GuestFunctionSymbol g_ps2GuestFunctionSymbols[];

#endif // PS2_RUNTIME_H
