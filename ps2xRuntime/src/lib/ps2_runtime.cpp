#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ee_guest_exceptions.h"
#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu_recompiler.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/CD.h"
#include "Kernel/Stubs/DMA.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/LibC.h"
#include "Kernel/Stubs/MemoryCard.h"
#include "Kernel/Stubs/MPEG.h"
#include "Kernel/Stubs/Pad.h"
#include "Kernel/Stubs/Helpers/AudioRuntimeState.h"
#include "Kernel/Stubs/Helpers/CdRuntimeState.h"
#include "Kernel/Stubs/Helpers/DmaRuntimeState.h"
#include "Kernel/Stubs/Helpers/LibCRuntimeState.h"
#include "Kernel/Stubs/Helpers/MemoryCardRuntimeState.h"
#include "Kernel/Stubs/Helpers/PadRuntimeState.h"
#include "Kernel/Stubs/Helpers/SifRuntimeState.h"
#include "Kernel/Syscalls/Helpers/AlarmRuntimeState.h"
#include "Kernel/Syscalls/Helpers/Deci2RuntimeState.h"
#include "Kernel/Syscalls/Helpers/FileRuntimeState.h"
#include "Kernel/Syscalls/Helpers/InterruptRuntimeState.h"
#include "Kernel/Syscalls/Helpers/KernelRuntimeState.h"
#include "Kernel/Syscalls/Helpers/RpcRuntimeState.h"
#include "Kernel/Syscalls/Helpers/SyncRuntimeState.h"
#include "Kernel/Syscalls/Helpers/ThreadRuntimeState.h"
#include "Kernel/Syscalls/FileIO.h"
#include "Kernel/Syscalls/System.h"
#include "ps2_host_backend.h"
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"
#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
#include "runtime/ps2_debug_server.h"
#endif

#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <sstream>

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint64_t EE_CYCLES_PER_SECOND = 294'912'000u;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
static constexpr const char *GS_HISTORY_DUMP_ENV = "PS2X_GS_HISTORY_DUMP";
static std::atomic<uint64_t> g_nextEeThreadRuntimeGeneration{1u};

static uint64_t debugFnv1a64(const uint8_t *data, size_t size) noexcept
{
    constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffsetBasis;
    for (size_t index = 0u; index < size; ++index)
    {
        hash ^= data[index];
        hash *= kPrime;
    }
    return hash;
}

#if !defined(PS2X_DEFAULT_EE_EXECUTION_BACKEND_CPP_FIBER)
#define PS2X_DEFAULT_EE_EXECUTION_BACKEND_CPP_FIBER 0
#endif

#if defined(_MSC_VER)
#define PS2X_EE_OBSERVATION_COLD __declspec(noinline)
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2X_EE_OBSERVATION_COLD                                      \
    __attribute__((noinline, noipa, cold,                             \
                   section(".text.unlikely.ee_observation")))
#else
#define PS2X_EE_OBSERVATION_COLD                                      \
    __attribute__((noinline, cold,                                    \
                   section(".text.unlikely.ee_observation")))
#endif

PS2RuntimeConfiguration
defaultPs2RuntimeConfiguration() noexcept
{
    PS2RuntimeConfiguration configuration{};
#if PS2X_DEFAULT_EE_EXECUTION_BACKEND_CPP_FIBER
    configuration.eeExecutionBackend =
        EeExecutionBackendKind::LegacyCppFiber;
#endif
    return configuration;
}

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool Ps2ResolveFastGuestRdramAccess(
    PS2Runtime *runtime,
    const R5900Context *ctx,
    uint32_t address,
    uint32_t bytes,
    bool writeAccess,
    uint32_t &physicalOffset)
{
    if (runtime == nullptr)
    {
        return false;
    }
    const EeAddressTranslationContext translation =
        ctx != nullptr
            ? EeAddressTranslationContext::fromCop0Status(
                  ctx->cop0_status,
                  static_cast<uint8_t>(ctx->cop0_entryhi))
            : EeAddressTranslationContext::unchecked();
    return Ps2ResolveFastGuestRdramOffset(
        runtime->memory(),
        translation,
        address,
        bytes,
        writeAccess,
        physicalOffset);
}

static void emitEeTlbTranslationCacheDiagnostics(
    PS2Memory &memory,
    bool &pending)
{
    if (!pending)
    {
        return;
    }
    pending = false;

    const EeTlbTranslationCacheStats stats =
        memory.tlbTranslationCacheStats();
    std::cerr
        << "{\"schema_version\":1,"
           "\"event\":\"ee-tlb-cache-stats\","
           "\"sets\":"
        << PS2Memory::kTlbTranslationCacheSetCount
        << ",\"ways\":"
        << PS2Memory::kTlbTranslationCacheWayCount
        << ",\"fast_sets\":"
        << PS2Memory::kTlbFastCacheSetCount
        << ",\"page_size\":"
        << PS2Memory::kTlbTranslationCachePageSize
        << ",\"generation\":"
        << memory.tlbMappingGeneration()
        << ",\"hits\":" << stats.hits
        << ",\"misses\":" << stats.misses
        << ",\"permission_misses\":"
        << stats.permissionMisses
        << ",\"generation_misses\":"
        << stats.generationMisses
        << ",\"interval_crossings\":"
        << stats.intervalCrossings
        << ",\"slow_path_faults\":"
        << stats.slowPathFaults
        << ",\"fills\":" << stats.fills
        << ",\"replacements\":"
        << stats.replacements
        << ",\"mapping_changes\":"
        << stats.mappingChanges
        << ",\"fast_hits\":"
        << stats.fastHits
        << ",\"fast_misses\":"
        << stats.fastMisses
        << ",\"backing_hits\":"
        << stats.backingHits
        << ",\"backing_misses\":"
        << stats.backingMisses
        << ",\"fast_fills\":"
        << stats.fastFills
        << ",\"fast_replacements\":"
        << stats.fastReplacements
        << "}" << std::endl;
}

struct HostPresentationUploadState
{
    uint64_t lastPresentationTick =
        std::numeric_limits<uint64_t>::max();
    bool hasLatchedInitialFrame = false;
    uint32_t lastDisplayFbp =
        std::numeric_limits<uint32_t>::max();
    uint32_t lastSourceFbp =
        std::numeric_limits<uint32_t>::max();
    bool lastPreferred = false;
    uint32_t lastWidth = 0u;
    uint32_t lastHeight = 0u;
    bool hasUploadedFrame = false;
    std::vector<uint8_t> scratch;
    uint32_t uploadDebugCount = 0u;

    void reset()
    {
        lastPresentationTick =
            std::numeric_limits<uint64_t>::max();
        hasLatchedInitialFrame = false;
        lastDisplayFbp =
            std::numeric_limits<uint32_t>::max();
        lastSourceFbp =
            std::numeric_limits<uint32_t>::max();
        lastPreferred = false;
        lastWidth = 0u;
        lastHeight = 0u;
        hasUploadedFrame = false;
        scratch.clear();
        uploadDebugCount = 0u;
    }
};

static uint64_t allocateEeThreadRuntimeGeneration() noexcept
{
    uint64_t generation = 0u;
    do
    {
        generation = g_nextEeThreadRuntimeGeneration.fetch_add(
            1u, std::memory_order_relaxed);
    } while (generation == 0u);
    return generation;
}

#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

struct SectionHeader
{
    uint32_t name;
    uint32_t type;
    uint32_t flags;
    uint32_t addr;
    uint32_t offset;
    uint32_t size;
    uint32_t link;
    uint32_t info;
    uint32_t addralign;
    uint32_t entsize;
};

static_assert(sizeof(ElfHeader) == 52u);
static_assert(sizeof(ProgramHeader) == 32u);
static_assert(sizeof(SectionHeader) == 40u);

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;
    // RAC1's BIOS VSync entry captured in PCSX2 uses this handler stack.
    // The idle PC is the BIOS EENULL continuation used when no guest thread
    // owns the interrupted boundary.
    constexpr uint32_t kEeBiosInterruptHandlerStack = 0x01FFFC20u;
    constexpr uint32_t kEeBiosIdleThreadPc = 0x00081FC0u;
    constexpr uint32_t kElfSectionAlloc = 0x2u;
    constexpr uint32_t kElfSectionExecInstr = 0x4u;
    constexpr uint32_t kVu0SyncThresholdCycles = 4u;
    constexpr uint32_t kVu0MinimumRunCycles = 16u;
    constexpr uint32_t kVu0VifFinishStartupCycles = 16u;
    constexpr uint32_t kVu0VifFinishFollowupCycles = 128u;
    // PCSX2 starts non-interlocked VU work in a minimum 16-cycle batch and
    // revisits an active VU1 from VIF_VU1_FINISH after 128 cycles. The
    // scheduler places that startup batch at a future EE boundary so
    // submission itself performs no VU work.
    constexpr uint32_t kVu1StartupEventCycles = 16u;
    constexpr uint32_t kVu1FollowupEventCycles = 128u;
    constexpr uint32_t kVu1MaximumAdvanceCycles = 65536u;
    constexpr uint32_t kVu1BusyMask = 1u << 8u;

    [[nodiscard]] constexpr bool usesBufferedVu1OwnerStream(
        Vu1ExecutionMode mode) noexcept
    {
        return mode == Vu1ExecutionMode::ThreadedAsync ||
               mode == Vu1ExecutionMode::ThreadedCoarse;
    }

    [[nodiscard]] constexpr bool usesExactVu1Scheduling(
        Vu1ExecutionMode mode) noexcept
    {
        return mode == Vu1ExecutionMode::ThreadedAsync;
    }

    uint64_t appendVu1CoarseDigest(
        uint64_t hash, uint64_t value) noexcept
    {
        constexpr uint64_t kOffsetBasis =
            14695981039346656037ull;
        constexpr uint64_t kPrime = 1099511628211ull;
        if (hash == 0u)
            hash = kOffsetBasis;
        for (uint32_t byte = 0u; byte < 8u; ++byte)
        {
            hash ^= static_cast<uint8_t>(value >> (byte * 8u));
            hash *= kPrime;
        }
        return hash;
    }

    uint64_t debugVu1TimingPacketDigest(
        const uint8_t *data, size_t size) noexcept
    {
        constexpr uint64_t kOffsetBasis =
            14695981039346656037ull;
        constexpr uint64_t kPrime = 1099511628211ull;
        uint64_t hash = kOffsetBasis;
        for (size_t index = 0u; index < size; ++index)
        {
            hash ^= data[index];
            hash *= kPrime;
        }
        return hash;
    }

    template <typename T>
    void updateAtomicMaximum(
        std::atomic<T> &destination,
        T candidate) noexcept
    {
        T current = destination.load(
            std::memory_order_relaxed);
        while (current < candidate &&
               !destination.compare_exchange_weak(
                   current, candidate,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed))
        {
        }
    }
    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_IP7 = 0x00008000u;
    constexpr uint32_t COP0_CAUSE_EXC2_MASK = 0x00070000u;
    constexpr uint32_t COP0_CAUSE_EXC2_PERFORMANCE = 0x00020000u;
    constexpr uint32_t COP0_CAUSE_EXC2_DEBUG = 0x00030000u;
    constexpr uint32_t COP0_CAUSE_BD2 = 0x40000000u;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_CAUSE_CE_MASK = 0x30000000u;
    constexpr uint32_t COP0_STATUS_IE = 0x00000001u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_ERL = 0x00000004u;
    constexpr uint32_t COP0_STATUS_KSU_MASK = 0x00000018u;
    constexpr uint32_t COP0_STATUS_BEM = 0x00001000u;
    constexpr uint32_t COP0_STATUS_IM7 = 0x00008000u;
    constexpr uint32_t COP0_STATUS_EIE = 0x00010000u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t COP0_STATUS_DEV = 0x00800000u;
    constexpr uint32_t COP0_STATUS_WRITABLE_MASK = 0xF0C79C1Fu;
    constexpr uint32_t EXCEPTION_VECTOR_NORMAL_BASE = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT_BASE = 0xBFC00200u;
    constexpr uint32_t EXCEPTION_VECTOR_COMMON_OFFSET = 0x180u;
    constexpr uint32_t EXCEPTION_VECTOR_INTERRUPT_OFFSET = 0x200u;
    constexpr uint32_t COP0_BPC_IAE = 1u << 31u;
    constexpr uint32_t COP0_BPC_DRE = 1u << 30u;
    constexpr uint32_t COP0_BPC_DWE = 1u << 29u;
    constexpr uint32_t COP0_BPC_DVE = 1u << 28u;
    constexpr uint32_t COP0_BPC_IUE = 1u << 26u;
    constexpr uint32_t COP0_BPC_ISE = 1u << 25u;
    constexpr uint32_t COP0_BPC_IKE = 1u << 24u;
    constexpr uint32_t COP0_BPC_IXE = 1u << 23u;
    constexpr uint32_t COP0_BPC_DUE = 1u << 21u;
    constexpr uint32_t COP0_BPC_DSE = 1u << 20u;
    constexpr uint32_t COP0_BPC_DKE = 1u << 19u;
    constexpr uint32_t COP0_BPC_DXE = 1u << 18u;
    constexpr uint32_t COP0_BPC_BED = 1u << 15u;
    constexpr uint32_t COP0_BPC_DWB = 1u << 2u;
    constexpr uint32_t COP0_BPC_DRB = 1u << 1u;
    constexpr uint32_t COP0_BPC_IAB = 1u << 0u;

    bool eeBreakpointModeEnabled(
        uint32_t status,
        uint32_t bpc,
        bool instruction) noexcept
    {
        if ((status & COP0_STATUS_ERL) != 0u)
        {
            return false;
        }

        uint32_t enable = 0u;
        if ((status & COP0_STATUS_EXL) != 0u)
        {
            enable = instruction
                         ? COP0_BPC_IXE
                         : COP0_BPC_DXE;
        }
        else
        {
            switch (status & COP0_STATUS_KSU_MASK)
            {
            case 0u:
                enable = instruction
                             ? COP0_BPC_IKE
                             : COP0_BPC_DKE;
                break;
            case 1u << 3u:
                enable = instruction
                             ? COP0_BPC_ISE
                             : COP0_BPC_DSE;
                break;
            case 2u << 3u:
                enable = instruction
                             ? COP0_BPC_IUE
                             : COP0_BPC_DUE;
                break;
            default:
                return false;
            }
        }
        return (bpc & enable) != 0u;
    }

    EeAddressTranslationContext guestAddressTranslation(
        const R5900Context *ctx) noexcept
    {
        return ctx
                   ? EeAddressTranslationContext::fromCop0Status(
                         ctx->cop0_status,
                         static_cast<uint8_t>(
                             ctx->cop0_entryhi))
                   : EeAddressTranslationContext::unchecked();
    }

    uint32_t eeInstructionFetchPageKey(
        const R5900Context *ctx,
        uint32_t virtualAddress) noexcept
    {
        constexpr uint32_t kPageMask = 0xfffff000u;
        constexpr uint32_t kStatusAddressModeMask =
            0x0000001eu;
        constexpr uint32_t kStatusAddressModeShift = 7u;
        // The page base occupies bits 12..31, leaving bits 8..11 for the
        // four relevant Status bits and bits 0..7 for EntryHi.ASID.
        return (virtualAddress & kPageMask) |
               ((ctx->cop0_status &
                 kStatusAddressModeMask)
                << kStatusAddressModeShift) |
               static_cast<uint8_t>(ctx->cop0_entryhi);
    }

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;
    thread_local std::unordered_map<PS2Runtime *, uint32_t> g_guestExecutionDepths;
    thread_local uint32_t g_deferredGuestYieldDepth = 0u;
    thread_local bool g_deferredGuestYieldPending = false;

    struct EeExecutorConsequenceContext
    {
        PS2Runtime *runtime = nullptr;
        ps2x::ee::EeThreadScheduler *scheduler =
            nullptr;
        IEeExecutionBackend *backend = nullptr;
        ps2x::ee::EeSchedulerReschedulePolicy
            reschedulePolicy =
                ps2x::ee::
                    EeSchedulerReschedulePolicy::None;
    };

    thread_local EeExecutorConsequenceContext
        g_eeExecutorConsequenceContext;

    [[nodiscard]] constexpr
    ps2x::ee::EeSchedulerReschedulePolicy
    combineEeReschedulePolicies(
        ps2x::ee::EeSchedulerReschedulePolicy left,
        ps2x::ee::EeSchedulerReschedulePolicy right)
        noexcept
    {
        using Policy =
            ps2x::ee::EeSchedulerReschedulePolicy;
        if (left == Policy::EqualOrHigherPriority ||
            right == Policy::EqualOrHigherPriority)
        {
            return Policy::EqualOrHigherPriority;
        }
        if (left == Policy::HigherPriorityOnly ||
            right == Policy::HigherPriorityOnly)
        {
            return Policy::HigherPriorityOnly;
        }
        return Policy::None;
    }

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    struct Vif0MmioTrace
    {
        std::FILE *output = nullptr;
        uint64_t sequence = 0u;

        ~Vif0MmioTrace()
        {
            if (output)
                std::fclose(output);
        }
    };

    Vif0MmioTrace &getVif0MmioTrace()
    {
        static Vif0MmioTrace trace = []
        {
            Vif0MmioTrace result;
            if (const char *path = std::getenv("PS2X_VIF0_MMIO_TRACE");
                path && *path)
            {
                result.output = std::fopen(path, "wb");
                if (!result.output)
                {
                    std::fprintf(stderr,
                                 "VIF0 MMIO trace: cannot open '%s': %s\n",
                                 path, std::strerror(errno));
                }
            }
            return result;
        }();
        return trace;
    }

    void traceVif0MmioWrite(PS2Memory &memory,
                            const R5900Context *ctx,
                            uint32_t address,
                            uint32_t size,
                            uint64_t valueLo,
                            uint64_t valueHi)
    {
        Vif0MmioTrace &trace = getVif0MmioTrace();
        if (!trace.output)
            return;

        constexpr uint32_t VIF0_CHANNEL = 0x10008000u;
        constexpr uint32_t VIF0_CHANNEL_END = VIF0_CHANNEL + 0x60u;
        const uint32_t physicalAddress = address & 0x1FFFFFFFu;
        const uint64_t writeEnd = static_cast<uint64_t>(physicalAddress) + size;
        if (physicalAddress >= VIF0_CHANNEL_END || writeEnd <= VIF0_CHANNEL)
            return;

        std::fprintf(
            trace.output,
            "{\"schema_version\":1,\"event\":\"vif0-mmio-write\","
            "\"sequence\":%" PRIu64 ",\"pc\":\"0x%08" PRIx32 "\","
            "\"ra\":\"0x%08" PRIx32 "\",\"address\":\"0x%08" PRIx32 "\","
            "\"size\":%" PRIu32 ",\"value_lo\":\"0x%016" PRIx64 "\","
            "\"value_hi\":\"0x%016" PRIx64 "\"",
            trace.sequence++,
            ctx ? ctx->pc : 0u,
            ctx ? getRegU32(ctx, 31) : 0u,
            physicalAddress, size, valueLo, valueHi);

        const uint32_t chcrOffset = VIF0_CHANNEL - physicalAddress;
        if (physicalAddress <= VIF0_CHANNEL &&
            static_cast<uint64_t>(chcrOffset) + sizeof(uint32_t) <= size)
        {
            uint32_t chcr = 0u;
            if (chcrOffset < 8u)
                chcr = static_cast<uint32_t>(valueLo >> (chcrOffset * 8u));
            else
                chcr = static_cast<uint32_t>(valueHi >> ((chcrOffset - 8u) * 8u));

            if ((chcr & 0x100u) != 0u)
            {
                const uint32_t madr = memory.readIORegister(VIF0_CHANNEL + 0x10u);
                const uint32_t qwc = memory.readIORegister(VIF0_CHANNEL + 0x20u);
                const uint32_t tadr = memory.readIORegister(VIF0_CHANNEL + 0x30u);
                const uint32_t asr0 = memory.readIORegister(VIF0_CHANNEL + 0x40u);
                const uint32_t asr1 = memory.readIORegister(VIF0_CHANNEL + 0x50u);
                uint32_t tagWords[4]{};
                bool tagValid = false;
                try
                {
                    for (uint32_t index = 0u; index < 4u; ++index)
                        tagWords[index] = memory.read32(tadr + index * 4u);
                    tagValid = true;
                }
                catch (...)
                {
                }

                std::fprintf(
                    trace.output,
                    ",\"start\":{\"chcr\":\"0x%08" PRIx32 "\","
                    "\"madr\":\"0x%08" PRIx32 "\",\"qwc\":\"0x%08" PRIx32 "\","
                    "\"tadr\":\"0x%08" PRIx32 "\",\"asr0\":\"0x%08" PRIx32 "\","
                    "\"asr1\":\"0x%08" PRIx32 "\",\"tag_valid\":%s,"
                    "\"tag\":[\"0x%08" PRIx32 "\",\"0x%08" PRIx32 "\","
                    "\"0x%08" PRIx32 "\",\"0x%08" PRIx32 "\"]}",
                    chcr, madr, qwc, tadr, asr0, asr1,
                    tagValid ? "true" : "false",
                    tagWords[0], tagWords[1], tagWords[2], tagWords[3]);
            }
        }

        std::fputs("}\n", trace.output);
        std::fflush(trace.output);
    }
#else
    inline void traceVif0MmioWrite(
        PS2Memory &memory,
        const R5900Context *ctx,
        uint32_t address,
        uint32_t size,
        uint64_t valueLo,
        uint64_t valueHi)
    {
        (void)memory;
        (void)ctx;
        (void)address;
        (void)size;
        (void)valueLo;
        (void)valueHi;
    }
#endif

    struct ElfAddressRange
    {
        uint32_t start = 0u;
        uint32_t end = 0u;
    };

    bool containsRange(const ElfAddressRange &outer, const ElfAddressRange &inner)
    {
        return inner.start >= outer.start && inner.end <= outer.end;
    }

    void mergeAddressRanges(std::vector<ElfAddressRange> &ranges)
    {
        std::sort(ranges.begin(), ranges.end(), [](const ElfAddressRange &lhs, const ElfAddressRange &rhs)
        {
            return lhs.start < rhs.start || (lhs.start == rhs.start && lhs.end < rhs.end);
        });

        std::vector<ElfAddressRange> merged;
        merged.reserve(ranges.size());
        for (const ElfAddressRange &range : ranges)
        {
            if (range.end <= range.start)
            {
                continue;
            }

            if (!merged.empty() && range.start <= merged.back().end)
            {
                merged.back().end = std::max(merged.back().end, range.end);
                continue;
            }

            merged.push_back(range);
        }
        ranges = std::move(merged);
    }

    bool readExecutableSectionRanges(std::ifstream &file,
                                     uint64_t fileSize,
                                     const ElfHeader &header,
                                     const std::vector<ElfAddressRange> &executableLoadRanges,
                                     std::vector<ElfAddressRange> &sectionRanges)
    {
        sectionRanges.clear();
        if (header.shoff == 0u || header.shnum == 0u ||
            header.shentsize < sizeof(SectionHeader))
        {
            return false;
        }

        const uint64_t tableEnd =
            static_cast<uint64_t>(header.shoff) +
            static_cast<uint64_t>(header.shnum) * static_cast<uint64_t>(header.shentsize);
        if (tableEnd > fileSize)
        {
            return false;
        }

        for (uint16_t i = 0; i < header.shnum; ++i)
        {
            const uint64_t sectionOffset =
                static_cast<uint64_t>(header.shoff) +
                static_cast<uint64_t>(i) * static_cast<uint64_t>(header.shentsize);

            SectionHeader section{};
            file.clear();
            file.seekg(static_cast<std::streamoff>(sectionOffset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(&section), sizeof(section)))
            {
                sectionRanges.clear();
                return false;
            }

            constexpr uint32_t requiredFlags = kElfSectionAlloc | kElfSectionExecInstr;
            if ((section.flags & requiredFlags) != requiredFlags || section.size == 0u)
            {
                continue;
            }

            const uint64_t sectionEnd =
                static_cast<uint64_t>(section.addr) + static_cast<uint64_t>(section.size);
            if (sectionEnd > std::numeric_limits<uint32_t>::max())
            {
                continue;
            }

            const ElfAddressRange candidate{
                section.addr,
                static_cast<uint32_t>(sectionEnd),
            };
            const bool isFileBackedExecutableRange =
                std::any_of(executableLoadRanges.begin(), executableLoadRanges.end(),
                            [&candidate](const ElfAddressRange &loadRange)
                            {
                                return containsRange(loadRange, candidate);
                            });
            if (isFileBackedExecutableRange)
            {
                sectionRanges.push_back(candidate);
            }
        }

        if (sectionRanges.empty())
        {
            return false;
        }

        mergeAddressRanges(sectionRanges);
        return true;
    }

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << PS2Runtime::formatGuestPc(h.pcs[idx]);
        }
        return oss.str();
    }

    std::string escapeJsonString(std::string_view text)
    {
        std::ostringstream output;
        for (const unsigned char ch : text)
        {
            switch (ch)
            {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (ch < 0x20u)
                {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<uint32_t>(ch)
                           << std::dec << std::setfill(' ');
                }
                else
                {
                    output << static_cast<char>(ch);
                }
                break;
            }
        }
        return output.str();
    }

    std::string rawGuestAddress(uint32_t address)
    {
        std::ostringstream output;
        output << "0x" << std::hex << std::setw(8) << std::setfill('0')
               << address;
        return output.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx,
                                   uint32_t exceptionCode,
                                   bool alreadyInException,
                                   bool useTlbRefillVector)
    {
        uint32_t offset = EXCEPTION_VECTOR_COMMON_OFFSET;
        if (!alreadyInException)
        {
            if (useTlbRefillVector &&
                (exceptionCode == EXCEPTION_TLB_REFILL_LOAD ||
                 exceptionCode == EXCEPTION_TLB_REFILL_STORE))
            {
                offset = 0u;
            }
            else if (exceptionCode == EXCEPTION_INTERRUPT)
            {
                offset = EXCEPTION_VECTOR_INTERRUPT_OFFSET;
            }
        }

        const uint32_t base = (ctx->cop0_status & COP0_STATUS_BEV)
                                  ? EXCEPTION_VECTOR_BOOT_BASE
                                  : EXCEPTION_VECTOR_NORMAL_BASE;
        return base + offset;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextRegistersToState(
        const R5900Context *ctx, VuExecutionState &state)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.top = ctx->vu0_top;
        state.itop = ctx->vu0_itop;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VuExecutionState &state)
    {
        state = {};
        copyVu0ContextRegistersToState(ctx, state);
        state.pc = ctx->vu0_pc;
    }

    void copyVu0StateToContext(const VuExecutionState &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->enforceVu0RegisterInvariants();
    }

    void copyEeCop2ProcessorGlobalState(
        R5900Context &destination,
        const R5900Context &source) noexcept
    {
        std::copy_n(
            source.vu0_vf,
            std::size(source.vu0_vf),
            destination.vu0_vf);
        std::copy_n(
            source.vi,
            std::size(source.vi),
            destination.vi);
        destination.vu0_q = source.vu0_q;
        destination.vu0_p = source.vu0_p;
        destination.vu0_i = source.vu0_i;
        destination.vu0_r = source.vu0_r;
        destination.vu0_acc = source.vu0_acc;
        destination.vu0_status = source.vu0_status;
        destination.vu0_mac_flags = source.vu0_mac_flags;
        destination.vu0_clip_flags = source.vu0_clip_flags;
        destination.vu0_clip_flags2 = source.vu0_clip_flags2;
        destination.vu0_cmsar0 = source.vu0_cmsar0;
        destination.vu0_cmsar1 = source.vu0_cmsar1;
        destination.vu0_cmsar2 = source.vu0_cmsar2;
        destination.vu0_cmsar3 = source.vu0_cmsar3;
        destination.vu0_vpu_stat = source.vu0_vpu_stat;
        destination.vu0_vpu_stat2 = source.vu0_vpu_stat2;
        destination.vu0_vpu_stat3 = source.vu0_vpu_stat3;
        destination.vu0_vpu_stat4 = source.vu0_vpu_stat4;
        destination.vu0_tpc = source.vu0_tpc;
        destination.vu0_tpc2 = source.vu0_tpc2;
        destination.vu0_fbrst = source.vu0_fbrst;
        destination.vu0_fbrst2 = source.vu0_fbrst2;
        destination.vu0_fbrst3 = source.vu0_fbrst3;
        destination.vu0_fbrst4 = source.vu0_fbrst4;
        destination.vu0_itop = source.vu0_itop;
        destination.vu0_top = source.vu0_top;
        destination.vu0_info = source.vu0_info;
        destination.vu0_xitop = source.vu0_xitop;
        destination.vu0_pc = source.vu0_pc;
        std::copy_n(
            source.vu0_cf,
            std::size(source.vu0_cf),
            destination.vu0_cf);
        std::copy_n(
            source.cop2_ccr,
            std::size(source.cop2_ccr),
            destination.cop2_ccr);
        destination.enforceVu0RegisterInvariants();
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths defaultRuntimeIoPaths()
    {
        PS2Runtime::IoPaths defaults;
        std::error_code ec;
        const std::filesystem::path cwd =
            std::filesystem::current_path(ec);
        defaults.elfDirectory =
            ec ? std::filesystem::path(".")
               : cwd.lexically_normal();
        defaults.hostRoot = defaults.elfDirectory;
        defaults.cdRoot = defaults.elfDirectory;
        defaults.mcRoot =
            defaults.elfDirectory / "mc0";
        return defaults;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

PS2Runtime::GuestExecutionScope::GuestExecutionScope(
    PS2Runtime *runtime,
    R5900Context *context,
    int selectionHint) noexcept
    : m_runtime(runtime),
      m_context(context)
{
    if (m_runtime)
    {
        m_previousContext =
            m_runtime->enterGuestExecution(
                m_context,
                &m_previousThreadId,
                selectionHint);
    }
}

PS2Runtime::AsyncCallbackInvocationScope::
    AsyncCallbackInvocationScope(
        PS2Runtime *runtime,
        R5900Context *callbackContext)
    : AsyncCallbackInvocationScope(
          runtime,
          callbackContext,
          nullptr,
          -1)
{
}

PS2Runtime::AsyncCallbackInvocationScope::
    AsyncCallbackInvocationScope(
        PS2Runtime *runtime,
        R5900Context *callbackContext,
        R5900Context *continuationContext,
        int continuationThreadId)
    : m_runtime(runtime),
      m_guestExecution(runtime, callbackContext)
{
    if (m_runtime)
    {
        m_runtime->beginAsyncCallbackInvocation(
            *this,
            continuationContext,
            continuationThreadId);
    }
}

PS2Runtime::AsyncCallbackInvocationScope::
    ~AsyncCallbackInvocationScope()
{
    if (m_runtime)
    {
        m_runtime->endAsyncCallbackInvocation(*this);
    }
}

PS2Runtime::GuestExecutionScope::~GuestExecutionScope()
{
    if (m_runtime)
    {
        m_runtime->leaveGuestExecution(
            m_context,
            m_previousContext,
            m_previousThreadId);
    }
}

PS2Runtime::GuestExecutionReleaseScope::GuestExecutionReleaseScope(PS2Runtime *runtime) noexcept
    : m_runtime(runtime)
{
    if (m_runtime)
    {
        m_depth =
            m_runtime->releaseGuestExecution(
                m_context, m_threadId);
    }
}

PS2Runtime::GuestExecutionReleaseScope::~GuestExecutionReleaseScope()
{
    if (m_runtime && m_depth != 0u)
    {
        m_runtime->reacquireGuestExecution(
            m_depth, m_context, m_threadId);
    }
}

PS2Runtime::GuestExecutionTryScope::GuestExecutionTryScope(
    PS2Runtime *runtime,
    R5900Context *context,
    std::chrono::milliseconds timeout) noexcept
    : m_runtime(runtime),
      m_context(context)
{
    if (m_runtime)
    {
        m_acquired =
            m_runtime->tryEnterGuestExecution(
                m_context,
                timeout,
                m_previousContext,
                m_previousThreadId);
    }
}

PS2Runtime::GuestExecutionTryScope::~GuestExecutionTryScope()
{
    if (m_runtime && m_acquired)
    {
        m_runtime->leaveGuestExecution(
            m_context,
            m_previousContext,
            m_previousThreadId);
    }
}

static bool UploadPresentationTexture(
    Texture2D &texture,
    const uint8_t *pixels,
    uint32_t width,
    uint32_t height)
{
    if (!pixels || width == 0u || height == 0u)
    {
        return false;
    }

    if (texture.width == static_cast<int>(width) &&
        texture.height == static_cast<int>(height))
    {
        UpdateTexture(texture, pixels);
        return true;
    }

    Image image{};
    image.data = const_cast<uint8_t *>(pixels);
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    Texture2D replacement = LoadTextureFromImage(image);
    if (replacement.id == 0u)
    {
        return false;
    }

    UnloadTexture(texture);
    texture = replacement;
    return true;
}

static void UploadFrame(
    Texture2D &tex,
    PS2Runtime *rt,
    HostPresentationUploadState &state,
    uint32_t &outWidth,
    uint32_t &outHeight)
{
    const bool asynchronous =
        rt->usesAsyncGsExecution();
    const uint64_t currentTick = asynchronous
        ? rt->lastCompletedGsFieldSequence()
        : ps2_syscalls::GetCurrentVSyncTick(rt);
    const bool needsLatch =
        !state.hasLatchedInitialFrame ||
        currentTick != state.lastPresentationTick;
    if (needsLatch)
    {
        if (!asynchronous)
        {
            (void)rt->submitGsCommand(
                GsLatchPresentationCommand{});
        }
        state.lastPresentationTick = currentTick;
        state.hasLatchedInitialFrame = true;
    }
    else if (state.hasUploadedFrame)
    {
        outWidth =
            (state.lastWidth != 0u)
                ? state.lastWidth
                : FB_WIDTH;
        outHeight =
            (state.lastHeight != 0u)
                ? state.lastHeight
                : DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    GsPresentationResult frame =
        takeGsCommandResult<GsPresentationResult>(
            rt->submitGsCommand(
                GsCopyPresentationCommand{}));
    if (asynchronous)
    {
        state.lastPresentationTick =
            frame.completedFieldSequence;
    }
    state.scratch = std::move(frame.pixels);
    const uint32_t width = frame.width;
    const uint32_t height = frame.height;
    const uint32_t displayFbp = frame.displayFbp;
    const uint32_t sourceFbp = frame.sourceFbp;
    const bool usedPreferredDisplaySource =
        frame.usedPreferred;
    if (!frame.available)
    {
        Image blank = GenImageColor(
            FB_WIDTH, DEFAULT_DISPLAY_HEIGHT, MAGENTA);
        const bool uploaded = UploadPresentationTexture(
            tex,
            static_cast<const uint8_t *>(blank.data),
            FB_WIDTH,
            DEFAULT_DISPLAY_HEIGHT);
        UnloadImage(blank);
        if (!uploaded)
        {
            return;
        }

        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        state.lastWidth = outWidth;
        state.lastHeight = outHeight;
        state.hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        if (state.uploadDebugCount < 128u ||
            displayFbp != state.lastDisplayFbp ||
            sourceFbp != state.lastSourceFbp ||
            usedPreferredDisplaySource != state.lastPreferred ||
            width != state.lastWidth ||
            height != state.lastHeight)
        {
            std::cout << "[frame:upload] idx="
                      << state.uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++state.uploadDebugCount;
    });

    const size_t expectedBytes =
        static_cast<size_t>(width) * height * 4u;
    if (state.scratch.size() < expectedBytes ||
        !UploadPresentationTexture(
            tex, state.scratch.data(), width, height))
    {
        return;
    }

    state.lastDisplayFbp = displayFbp;
    state.lastSourceFbp = sourceFbp;
    state.lastPreferred = usedPreferredDisplaySource;
    state.lastWidth = width;
    state.lastHeight = height;
    outWidth = width;
    outHeight = height;
    state.hasUploadedFrame = true;
}

PS2Runtime::PS2Runtime()
    : PS2Runtime(defaultPs2RuntimeConfiguration())
{
}

PS2Runtime::PS2Runtime(PS2RuntimeConfiguration configuration)
    : m_gsCommandProcessor(m_gs),
      m_vu0(VuUnitId::Vu0),
      m_vu1(VuUnitId::Vu1),
      m_vu1CommandProcessor(
          m_vu1,
          {
              .captureCommandDigests =
                  configuration.captureVu1CommandDigests,
              .captureArchitecturalStateHashes =
                  configuration.
                      captureVu1ArchitecturalStateHashes,
          }),
      m_cpuContext(
          R5900Context::InitializationProfile::PostBiosElf)
{
    m_hostMonitorIndex = configuration.hostMonitorIndex;
    if (m_hostMonitorIndex.has_value() &&
        *m_hostMonitorIndex < 0)
    {
        throw std::invalid_argument(
            "the host monitor index must be nonnegative");
    }
    m_vu1CommandPayloadCapacityBytes =
        configuration.vu1CommandPayloadCapacityBytes;
    if (g_ps2RecompiledFunctionPairAbiVersion !=
        PS2_RECOMPILED_FUNCTION_PAIR_ABI_VERSION)
    {
        throw std::runtime_error(
            "generated EE function table uses an incompatible fast/precise ABI");
    }

    if (configuration.gsMaximumFieldLead > 1u)
    {
        throw std::invalid_argument(
            "the asynchronous GS field lead must be zero or one");
    }
    if (configuration.gsCommandQueueCapacity >
        std::numeric_limits<size_t>::max() - 2u)
    {
        throw std::invalid_argument(
            "the GS command queue capacity is too large");
    }
    m_gsMaximumFieldLead =
        configuration.gsMaximumFieldLead;
    m_gsAsyncPendingLimit = std::max<size_t>(
        1u, configuration.gsCommandQueueCapacity + 2u);

    switch (configuration.gsExecutionMode)
    {
    case GsExecutionMode::Inline:
        m_gsExecutor = std::make_unique<InlineGsExecutor>(
            m_gsCommandProcessor);
        break;
    case GsExecutionMode::ThreadedSynchronous:
    case GsExecutionMode::ThreadedAsync:
    {
        auto executor =
            std::make_unique<ThreadedGsExecutor>(
                m_gsCommandProcessor,
                ThreadedGsExecutorOptions{
                    .queueCapacity =
                        configuration.gsCommandQueueCapacity,
                    .payloadCapacityBytes =
                        configuration.
                            gsCommandPayloadCapacityBytes,
                    .beforeProcess =
                        std::move(configuration.gsBeforeProcess),
                    .beforePublish =
                        std::move(configuration.gsBeforePublish),
                });
        m_threadedGsExecutor = executor.get();
        m_gsExecutor = std::move(executor);
        m_gsAsyncEnabled =
            configuration.gsExecutionMode ==
            GsExecutionMode::ThreadedAsync;
        break;
    }
    default:
        throw std::invalid_argument(
            "unsupported GS execution mode");
    }

    m_vu1ExecutionMode = configuration.vu1ExecutionMode;
    m_vu1CheckpointEpochs =
        configuration.vu1CheckpointEpochs &&
        m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedAsync;
    m_vu1CheckpointCapacity =
        configuration.vu1CheckpointCapacity;
    m_vu1CoarseInitialEstimateCycles =
        configuration.vu1CoarseInitialEstimateCycles;
    m_vu1CoarseMinimumEstimateCycles =
        configuration.vu1CoarseMinimumEstimateCycles;
    m_vu1CoarseMaximumLeadCycles =
        configuration.vu1CoarseMaximumLeadCycles;
    m_vu1CoarseMaximumInvocationCycles =
        configuration.vu1CoarseMaximumInvocationCycles;
    m_vu1CoarseMaximumPath1Bytes =
        configuration.vu1CoarseMaximumPath1Bytes;
    m_vu1CoarseEstimatorHistoryCapacity =
        configuration.vu1CoarseEstimatorHistoryCapacity;
    m_vu1CoarseEstimatorIdentityCapacity =
        configuration.vu1CoarseEstimatorIdentityCapacity;
    if (m_vu1CheckpointEpochs &&
        (m_vu1CheckpointCapacity == 0u ||
         m_vu1CheckpointCapacity > 64u))
    {
        throw std::invalid_argument(
            "VU1 checkpoint capacity must be between one and 64");
    }
    if (m_vu1ExecutionMode == Vu1ExecutionMode::ThreadedCoarse &&
        (m_vu1CoarseInitialEstimateCycles == 0u ||
         m_vu1CoarseMinimumEstimateCycles == 0u ||
         m_vu1CoarseMaximumLeadCycles == 0u ||
         m_vu1CoarseMaximumInvocationCycles == 0u ||
         m_vu1CoarseMaximumPath1Bytes == 0u ||
         m_vu1CoarseEstimatorHistoryCapacity == 0u ||
         m_vu1CoarseEstimatorHistoryCapacity > 64u ||
         m_vu1CoarseEstimatorIdentityCapacity == 0u ||
         m_vu1CoarseEstimatorIdentityCapacity > 4096u ||
         m_vu1CoarseMinimumEstimateCycles >
             m_vu1CoarseInitialEstimateCycles ||
         m_vu1CoarseInitialEstimateCycles >
             m_vu1CoarseMaximumLeadCycles ||
         m_vu1CoarseMaximumLeadCycles >
             m_vu1CoarseMaximumInvocationCycles))
    {
        throw std::invalid_argument(
            "invalid bounded coarse VU1 configuration");
    }

    m_memory.installPostBiosTlbState();
    m_hostPresentationUploadState =
        std::make_unique<
            HostPresentationUploadState>();
    EeExecutionBackendKind eeExecutionBackend =
        configuration.eeExecutionBackend;
    if (configuration
            .useEeExecutionBackendEnvironment)
    {
        if (const char *const value =
                std::getenv(
                    "PS2X_EE_EXECUTION_BACKEND"))
        {
            if (!parseEeExecutionBackendKind(
                    value, eeExecutionBackend))
            {
                throw std::invalid_argument(
                    "PS2X_EE_EXECUTION_BACKEND must be "
                    "legacy-host-thread or "
                    "legacy-cpp-fiber");
            }
        }
    }
    m_eeExecutionBackend =
        createEeExecutionBackend(
            eeExecutionBackend);
    if (m_eeExecutionBackend->executorResumable())
    {
        m_eeRuntimeExecutor =
            std::make_unique<
                ps2x::ee::EeRuntimeExecutor>(
                *m_eeExecutionBackend,
                static_cast<
                    ps2x::ee::
                        IEeSchedulerExecutorHooks *>(
                    this));
    }
    m_ioPaths = defaultRuntimeIoPaths();
    m_mpegRuntimeState =
        std::make_unique<ps2_stubs::MpegRuntimeState>();
    m_eeAlarmRuntimeState =
        std::make_unique<EeAlarmRuntimeState>();
    m_eeSyncRuntimeState =
        std::make_unique<EeSyncRuntimeState>();
    m_eeInterruptRuntimeState =
        std::make_unique<EeInterruptRuntimeState>();
    m_eeKernelRuntimeState =
        std::make_unique<EeKernelRuntimeState>();
    m_eeRpcRuntimeState =
        std::make_unique<EeRpcRuntimeState>();
    m_eeFileRuntimeState =
        std::make_unique<EeFileRuntimeState>();
    m_deci2RuntimeState =
        std::make_unique<Deci2RuntimeState>();
    m_audioRuntimeState =
        std::make_unique<ps2_stubs::AudioRuntimeState>();
    m_cdRuntimeState =
        std::make_unique<ps2_stubs::CdRuntimeState>();
    m_dmaRuntimeState =
        std::make_unique<ps2_stubs::DmaRuntimeState>();
    m_libcRuntimeState =
        std::make_unique<ps2_stubs::LibCRuntimeState>();
    m_memoryCardRuntimeState =
        std::make_unique<ps2_stubs::MemoryCardRuntimeState>();
    m_padRuntimeState =
        std::make_unique<ps2_stubs::PadRuntimeState>();
    m_sifRuntimeState =
        std::make_unique<ps2_stubs::SifRuntimeState>();
    m_eeThreadRuntimeState =
        std::make_unique<EeThreadRuntimeState>(
            allocateEeThreadRuntimeGeneration());
    m_eeThreadRuntimeState->contextThreadIds.emplace(
        &m_cpuContext, 1);
    selectEeThread(1);
    const auto configureVuBackend =
        [&](VuUnit &unit, VuBackendKind requested,
            const char *environmentName)
    {
        if (configuration.useVuBackendEnvironment)
        {
            if (const char *const value = std::getenv(environmentName))
            {
                if (!parseVuBackendKind(value, requested))
                {
                    throw std::invalid_argument(
                        std::string(environmentName) +
                        " must be auto, interpreter, recompiler, or verify");
                }
            }
        }

        std::string diagnostic;
        if (!unit.setBackend(requested, &diagnostic))
        {
            throw std::runtime_error(
                std::string(environmentName) + ": " + diagnostic);
        }
    };
    configureVuBackend(
        m_vu0, configuration.vu0Backend, "PS2X_VU0_BACKEND");
    configureVuBackend(
        m_vu1, configuration.vu1Backend, "PS2X_VU1_BACKEND");
    const auto configureNativeInstrumentation =
        [&](VuUnit &unit, bool enabled,
            const char *environmentName)
    {
        if (configuration.useVuBackendEnvironment)
        {
            if (const char *const value =
                    std::getenv(environmentName))
            {
                const std::string_view text(value);
                if (text == "1" || text == "true" ||
                    text == "on")
                {
                    enabled = true;
                }
                else if (text == "0" || text == "false" ||
                         text == "off")
                {
                    enabled = false;
                }
                else
                {
                    throw std::invalid_argument(
                        std::string(environmentName) +
                        " must be 0, 1, false, true, off, or on");
                }
            }
        }
        unit.setNativeInstrumentationEnabled(enabled);
    };
    configureNativeInstrumentation(
        m_vu0, configuration.vu0NativeInstrumentation,
        "PS2X_VU0_NATIVE_INSTRUMENTATION");
    configureNativeInstrumentation(
        m_vu1, configuration.vu1NativeInstrumentation,
        "PS2X_VU1_NATIVE_INSTRUMENTATION");

    // Finish configuring VU1 before a threaded executor claims ownership.
    // Once the worker exists, all architectural and backend mutation enters
    // the ordered command stream.
    switch (m_vu1ExecutionMode)
    {
    case Vu1ExecutionMode::Inline:
        m_vu1Executor =
            std::make_unique<InlineVu1Executor>(
                m_vu1CommandProcessor);
        break;
    case Vu1ExecutionMode::ThreadedSynchronous:
    case Vu1ExecutionMode::ThreadedAsync:
    case Vu1ExecutionMode::ThreadedCoarse:
    {
        auto executor =
            std::make_unique<ThreadedVu1Executor>(
                m_vu1CommandProcessor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity =
                        configuration.vu1CommandQueueCapacity,
                    .payloadCapacityBytes =
                        configuration.
                            vu1CommandPayloadCapacityBytes,
                    .beforeProcess =
                        std::move(configuration.vu1BeforeProcess),
                    .beforePublish =
                        std::move(configuration.vu1BeforePublish),
                });
        m_threadedVu1Executor = executor.get();
        m_vu1Executor = std::move(executor);
        break;
    }
    default:
        throw std::invalid_argument(
            "unsupported VU1 execution mode");
    }

    m_eeThreadDiagnosticsEnabled =
        configuration.eeThreadDiagnostics;
    if (configuration.useEeThreadDiagnosticsEnvironment)
    {
        if (const char *const value =
                std::getenv("PS2X_EE_THREAD_DIAGNOSTICS"))
        {
            const std::string_view text(value);
            if (text == "1" || text == "true" ||
                text == "on")
            {
                m_eeThreadDiagnosticsEnabled = true;
            }
            else if (text == "0" || text == "false" ||
                     text == "off")
            {
                m_eeThreadDiagnosticsEnabled = false;
            }
            else
            {
                throw std::invalid_argument(
                    "PS2X_EE_THREAD_DIAGNOSTICS must be "
                    "0, 1, false, true, off, or on");
            }
        }
    }

    bool tlbTranslationCacheDiagnostics =
        configuration.eeTlbTranslationCacheDiagnostics;
    if (configuration
            .useEeTlbTranslationCacheDiagnosticsEnvironment)
    {
        if (const char *const value =
                std::getenv(
                    "PS2X_EE_TLB_CACHE_DIAGNOSTICS"))
        {
            const std::string_view text(value);
            if (text == "1" || text == "true" ||
                text == "on")
            {
                tlbTranslationCacheDiagnostics = true;
            }
            else if (text == "0" || text == "false" ||
                     text == "off")
            {
                tlbTranslationCacheDiagnostics = false;
            }
            else
            {
                throw std::invalid_argument(
                    "PS2X_EE_TLB_CACHE_DIAGNOSTICS must be "
                    "0, 1, false, true, off, or on");
            }
        }
    }
    m_memory.setTlbTranslationCacheDiagnosticsEnabled(
        tlbTranslationCacheDiagnostics);
    m_emitTlbTranslationCacheDiagnosticsOnDestruction =
        tlbTranslationCacheDiagnostics;

    m_memory.setEeCounterCycleCallback(
        [this]()
        {
            return ps2x::timing::eeTickToCyclesFloor(
                currentEeTick());
        });
    m_memory.setEeCounterScheduleCallback(
        [this](std::optional<uint64_t> deadlineCycle)
        {
            onEeCounterScheduleChanged(
                deadlineCycle);
        });
    m_memory.setEeCounterInterruptStateCallback(
        [this](
            uint32_t status,
            uint32_t mask,
            uint32_t newlyRaised)
        {
            onEeCounterInterruptStateChanged(
                status, mask, newlyRaised);
        });
    m_memory.setDmacCompletionReadyCallback(
        [this]()
        {
            onDmacCompletionReady();
        });
    m_memory.setDmacInterruptStateCallback(
        [this]()
        {
            onDmacInterruptStateChanged();
        });
    m_memory.setVif1DmaScheduleCallback(
        [this](uint32_t delayEeCycles)
        {
            return scheduleVif1DmaFromMemory(
                delayEeCycles);
        });
    m_memory.setVif1DmaCancelCallback(
        [this]()
        {
            cancelVif1DmaEvent();
        });
    m_memory.setVif0DmaScheduleCallback(
        [this](uint32_t delayEeCycles)
        {
            return scheduleVif0DmaFromMemory(
                delayEeCycles);
        });
    m_memory.setVif0DmaCancelCallback(
        [this]()
        {
            cancelVif0DmaEvent();
            cancelVif0VuFinishEvent();
        });
    m_memory.setScratchpadDmaScheduleCallback(
        [this](
            DmacChannel channel,
            uint32_t delayEeCycles)
        {
            return scheduleScratchpadDmaFromMemory(
                channel, delayEeCycles);
        });
    m_memory.setScratchpadDmaCancelCallback(
        [this](DmacChannel channel)
        {
            cancelScratchpadDmaEvent(channel);
        });
    m_memory.setGifDmaScheduleCallback(
        [this](uint32_t delayEeCycles)
        {
            return scheduleGifDmaFromMemory(
                delayEeCycles);
        });
    m_memory.setGifDmaCancelCallback(
        [this]()
        {
            cancelGifDmaEvent();
        });
    m_memory.setFromIpuDmaScheduleCallback(
        [this](uint32_t delayEeCycles)
        {
            return scheduleFromIpuDmaFromMemory(
                delayEeCycles);
        });
    m_memory.setFromIpuDmaCancelCallback(
        [this]()
        {
            cancelFromIpuDmaEvent();
        });
    m_memory.setToIpuDmaScheduleCallback(
        [this](uint32_t delayEeCycles)
        {
            return scheduleToIpuDmaFromMemory(
                delayEeCycles);
        });
    m_memory.setToIpuDmaCancelCallback(
        [this]()
        {
            cancelToIpuDmaEvent();
        });

    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    m_vu0.setInstructionObserver(
        [this](uint64_t, uint32_t pc, uint32_t lower, uint32_t upper,
               const VuExecutionState &state)
        {
            const bool recordInstruction =
                m_debugVu0InstructionTraceEnabled.load(
                    std::memory_order_relaxed) &&
                m_debugVu0InstructionTraceTriggered.load(
                    std::memory_order_acquire);
            const bool countForSync =
                m_debugVu0SyncTraceEnabled.load(
                    std::memory_order_relaxed) &&
                m_debugVu0SyncTraceTriggered.load(
                    std::memory_order_acquire);
            if (!recordInstruction && !countForSync)
            {
                return;
            }

            const uint64_t invocationInstruction =
                m_vu0CurrentInvocationInstruction.fetch_add(
                    1u, std::memory_order_relaxed);
            if (!recordInstruction)
            {
                return;
            }

            DebugVu0InstructionEntry entry{};
            entry.invocation =
                m_vu0CurrentInvocation.load(std::memory_order_relaxed);
            entry.invocationInstruction =
                invocationInstruction;
            if (invocationInstruction == 0u)
            {
                entry.invocationStart = true;
                entry.codeHashFnv1a64 = debugFnv1a64(
                    m_memory.getVU0Code(), PS2_VU0_CODE_SIZE);
                entry.dataHashFnv1a64 = debugFnv1a64(
                    m_memory.getVU0Data(), PS2_VU0_DATA_SIZE);
            }
            entry.pc = pc;
            entry.lower = lower;
            entry.upper = upper;
            for (size_t index = 0u; index < entry.vi.size(); ++index)
            {
                entry.vi[index] =
                    static_cast<uint16_t>(state.vi[index]);
            }
            std::memcpy(
                entry.vf.data(), state.vf, sizeof(state.vf));
            std::memcpy(
                entry.acc.data(), state.acc, sizeof(state.acc));
            std::memcpy(&entry.q, &state.q, sizeof(entry.q));
            std::memcpy(&entry.p, &state.p, sizeof(entry.p));
            std::memcpy(&entry.i, &state.i, sizeof(entry.i));
            entry.status = state.status;
            entry.mac = state.mac;
            entry.clip = state.clip;
            entry.top = state.top;
            entry.itop = state.itop;
            entry.branchPending = state.branchPending;
            entry.branchTarget = state.branchTarget;
            entry.branchDelay = state.branchDelay;
            entry.viBackupCycles = state.viBackupCycles;
            entry.viBackupRegister = state.viBackupRegister;
            entry.viBackupValue =
                static_cast<uint16_t>(state.viBackupValue);
            debugRecordVu0Instruction(std::move(entry));
        });
#endif
    m_vu0.setInstructionObserverEnabled(false);
#if defined(PS2X_IOP_ENABLE_PLUGINS) && PS2X_IOP_ENABLE_PLUGINS && \
    !defined(PLATFORM_VITA) && (defined(_WIN32) || defined(__linux__))
    if (const char *applicationDirectory = GetApplicationDirectory();
        applicationDirectory && applicationDirectory[0] != '\0')
    {
        m_iopSubsystem->setPluginSearchPaths({std::filesystem::path(applicationDirectory) / "iop_plugins"});
    }
#endif

    m_cop0Timing.reset();
    publishCop0TimingMirrors(&m_cpuContext);

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        if (m_eeRuntimeExecutor)
        {
            m_eeRuntimeExecutor->join();
        }
        emitEeTlbTranslationCacheDiagnostics(
            m_memory,
            m_emitTlbTranslationCacheDiagnosticsOnDestruction);
#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
        if (m_debugServer)
        {
            m_debugServer->stop();
            m_debugServer.reset();
        }
#endif
        // Worker contexts and registries are owned by this runtime. Stop
        // notification releases every kernel wait, so workers must be joined
        // before those runtime-owned objects are destroyed.
        ps2_syscalls::joinAllGuestHostThreads(this);
        if (m_gsAsyncEnabled)
        {
            std::lock_guard<std::mutex> lock(
                m_gsEeJournalMutex);
            reapEeGsSubmissions(true);
        }
        if (m_threadedGsExecutor)
        {
            m_threadedGsExecutor->shutdown(
                GsExecutorShutdownMode::Drain);
        }
        if (m_threadedVu1Executor)
        {
            resolveVu1SpeculationForShutdown();
            m_threadedVu1Executor->shutdown(
                Vu1ExecutorShutdownMode::Drain);
        }
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

EeThreadRuntimeState &PS2Runtime::eeThreadRuntimeState()
{
    return *m_eeThreadRuntimeState;
}

const EeThreadRuntimeState &PS2Runtime::eeThreadRuntimeState() const
{
    return *m_eeThreadRuntimeState;
}

IEeExecutionBackend &PS2Runtime::eeExecutionBackend()
{
    return *m_eeExecutionBackend;
}

const IEeExecutionBackend &
PS2Runtime::eeExecutionBackend() const
{
    return *m_eeExecutionBackend;
}

std::string_view
PS2Runtime::eeExecutionBackendName() const noexcept
{
    return m_eeExecutionBackend
               ? m_eeExecutionBackend->name()
               : std::string_view{"uninitialized"};
}

size_t
PS2Runtime::managedEeExecutionThreadCountForTesting() const
{
    return m_eeExecutionBackend
               ? m_eeExecutionBackend
                     ->managedThreadCount()
               : 0u;
}

bool PS2Runtime::eeExecutionQuiescentForTesting()
    const noexcept
{
    if (usesDedicatedEeExecutor())
    {
        return managedEeExecutionThreadCountForTesting() ==
               0u;
    }
    return m_eeThreadRuntimeState
               ->activeHostThreads.load(
                   std::memory_order_acquire) == 0;
}

std::vector<PS2Runtime::EeThreadContextImageForTesting>
PS2Runtime::eeThreadContextsForTesting()
{
    std::vector<EeThreadContextImageForTesting> result;
    const auto capture =
        [this, &result]()
        {
            EeThreadRuntimeState &state =
                *m_eeThreadRuntimeState;
            std::vector<
                std::pair<
                    int,
                    std::shared_ptr<ThreadInfo>>>
                threads;
            {
                std::lock_guard<std::mutex> lock(
                    state.threadMapMutex);
                threads.reserve(state.threads.size());
                for (const auto &[id, info] :
                     state.threads)
                {
                    threads.emplace_back(id, info);
                }
            }
            std::sort(
                threads.begin(),
                threads.end(),
                [](const auto &lhs, const auto &rhs)
                {
                    return lhs.first < rhs.first;
                });

            result.clear();
            result.reserve(threads.size());
            for (const auto &[id, info] : threads)
            {
                if (!info)
                {
                    continue;
                }
                EeThreadContextImageForTesting image{};
                image.first = id;
                {
                    std::lock_guard<std::mutex> lock(
                        info->m);
                    const R5900Context *const context =
                        info->boundContext
                            ? info->boundContext
                            : &info->context;
                    std::memcpy(
                        image.second.data(),
                        context,
                        image.second.size());
                }
                result.push_back(std::move(image));
            }
        };

    if (m_eeRuntimeExecutor &&
        m_eeRuntimeExecutor->running())
    {
        if (!invokeEeExecutorTaskAtBoundary(capture))
        {
            throw std::runtime_error(
                "could not capture EE thread contexts "
                "at an executor boundary");
        }
        return result;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

bool PS2Runtime::usesDedicatedEeExecutor() const noexcept
{
    return m_eeRuntimeExecutor != nullptr;
}

bool PS2Runtime::publishEeExecutorUpdate(
    std::function<void(
        ps2x::ee::EeThreadScheduler &,
        IEeExecutionBackend &)> update,
    ps2x::ee::EeSchedulerReschedulePolicy policy)
{
    if (!m_eeRuntimeExecutor || !update)
    {
        return false;
    }

    EeExecutorConsequenceContext &consequence =
        g_eeExecutorConsequenceContext;
    if (consequence.runtime == this)
    {
        if (!consequence.scheduler ||
            !consequence.backend)
        {
            throw std::logic_error(
                "EE callback consequence has no "
                "scheduler/backend context");
        }
        update(
            *consequence.scheduler,
            *consequence.backend);
        consequence.reschedulePolicy =
            combineEeReschedulePolicies(
                consequence.reschedulePolicy,
                policy);
        return true;
    }

    return m_eeRuntimeExecutor->publish(
        std::move(update), policy);
}

bool PS2Runtime::publishEeSchedulerUpdate(
    std::function<void(
        ps2x::ee::EeThreadScheduler &)> update,
    ps2x::ee::EeSchedulerReschedulePolicy policy)
{
    if (!update)
    {
        return false;
    }

    return publishEeExecutorUpdate(
        [update = std::move(update)](
            ps2x::ee::EeThreadScheduler &scheduler,
            IEeExecutionBackend &)
        {
            update(scheduler);
        },
        policy);
}

bool PS2Runtime::publishEeSchedulerUpdateAt(
    std::chrono::steady_clock::time_point deadline,
    std::function<void(
        ps2x::ee::EeThreadScheduler &)> update,
    ps2x::ee::EeSchedulerReschedulePolicy policy)
{
    if (!m_eeRuntimeExecutor || !update)
    {
        return false;
    }

    return m_eeRuntimeExecutor->publishAt(
        deadline,
        [update = std::move(update)](
            ps2x::ee::EeThreadScheduler &scheduler,
            IEeExecutionBackend &)
        {
            update(scheduler);
        },
        policy);
}

void PS2Runtime::invokeEeSchedulerUpdateAtBoundary(
    std::function<void(
        ps2x::ee::EeThreadScheduler &)> update,
    ps2x::ee::EeSchedulerReschedulePolicy policy)
{
    if (!m_eeRuntimeExecutor || !update)
    {
        throw std::logic_error(
            "EE scheduler boundary invocation requires "
            "a dedicated executor and a command");
    }

    m_eeRuntimeExecutor->invokeAtBoundary(
        [update = std::move(update)](
            ps2x::ee::EeThreadScheduler &scheduler,
            IEeExecutionBackend &)
        {
            update(scheduler);
        },
        policy);
}

bool PS2Runtime::invokeEeExecutorTaskAtBoundary(
    std::function<void()> task)
{
    if (!m_eeRuntimeExecutor ||
        !m_eeRuntimeExecutor->running() ||
        m_eeRuntimeExecutor->ownsCurrentThread())
    {
        return false;
    }
    if (!task)
    {
        throw std::invalid_argument(
            "EE executor boundary task is empty");
    }

    // The executor publication wakes an idle or paused owner. Ask a running
    // continuation to reach its backend-neutral checkpoint as well, then
    // execute the task before that or any other continuation resumes.
    requestGuestPreemption();
    m_eeRuntimeExecutor->invokeAtBoundary(
        [task = std::move(task)](
            ps2x::ee::EeThreadScheduler &,
            IEeExecutionBackend &)
        {
            task();
        },
        ps2x::ee::EeSchedulerReschedulePolicy::None);
    return true;
}

ps2x::timing::EeTickDelta
PS2Runtime::eeExecutorElapsedSinceSelection()
    const noexcept
{
    return ps2x::timing::elapsedEeTicks(
        m_eeExecutorDispatchStartTick,
        currentEeTick());
}

void PS2Runtime::yieldEeExecutorCurrent(
    ps2x::ee::EeSchedulerExitReason reason,
    ps2x::ee::EeSchedulerWaitKey wait)
{
    if (!m_eeRuntimeExecutor)
    {
        throw std::logic_error(
            "EE executor yield requested without a "
            "dedicated executor");
    }

    if (g_eeExecutorConsequenceContext.runtime ==
        this)
    {
        if (reason ==
                ps2x::ee::EeSchedulerExitReason::
                    Blocked ||
            wait.valid())
        {
            throw std::logic_error(
                "an EE interrupt/callback consequence "
                "cannot block the executor");
        }
        // Non-blocking interrupt-context syscalls have already applied
        // their scheduler mutation inline above. The containing consequence
        // returns the strongest accumulated reschedule policy after all
        // callback code finishes; there is no guest fiber to suspend here.
        return;
    }

    R5900Context *releasedContext = nullptr;
    int releasedThreadId = 1;
    const uint32_t releasedDepth =
        releaseGuestExecution(
            releasedContext,
            releasedThreadId);
    const ps2x::timing::EeTickDelta elapsed =
        eeExecutorElapsedSinceSelection();

    try
    {
        m_eeExecutionBackend->yieldCurrent(
            {
                reason,
                elapsed.raw(),
                wait,
                {},
            });
    }
    catch (...)
    {
        if (releasedDepth != 0u)
        {
            reacquireGuestExecution(
                releasedDepth,
                releasedContext,
                releasedThreadId);
        }
        throw;
    }

    if (releasedDepth != 0u)
    {
        reacquireGuestExecution(
            releasedDepth,
            releasedContext,
            releasedThreadId);
    }
}

PS2Runtime::EeInterruptedContinuationFrame
PS2Runtime::makeEeInterruptedContinuation(
    R5900Context *context,
    int threadId,
    bool retainDormantContext)
{
    EeInterruptedContinuationFrame frame{};
    std::shared_ptr<ThreadInfo> owner;

    if (threadId <= 0 && context)
    {
        threadId = threadIdForEeContext(context);
    }
    if (threadId > 0)
    {
        EeThreadRuntimeState &state =
            *m_eeThreadRuntimeState;
        std::lock_guard<std::mutex> lock(
            state.threadMapMutex);
        const auto entry = state.threads.find(threadId);
        if (entry != state.threads.end())
        {
            owner = entry->second;
            if (!context)
            {
                context = owner->boundContext;
            }
        }
    }

    bool dormant = false;
    uint32_t generation = 0u;
    if (owner)
    {
        std::lock_guard<std::mutex> lock(owner->m);
        dormant = owner->guestState.isDormant();
        generation = owner->generation;
    }

    const bool hasContinuation =
        context &&
        ((threadId > 0 &&
          (!dormant || retainDormantContext)) ||
         (retainDormantContext && threadId == 0));
    if (hasContinuation)
    {
        frame.context = *context;
        frame.continuationContext = context;
        frame.continuationOwner = std::move(owner);
        frame.generation = generation;
        frame.threadId = threadId;
        frame.idle = threadId == 0;
        return frame;
    }

    frame.context = {};
    frame.context.pc = kEeBiosIdleThreadPc;
    SET_GPR_U32(
        &frame.context,
        29,
        kEeBiosInterruptHandlerStack);
    frame.threadId = 0;
    frame.idle = true;
    return frame;
}

void PS2Runtime::captureEeInterruptContinuation(
    R5900Context *context,
    int threadId)
{
    m_eeInterruptedContinuation =
        makeEeInterruptedContinuation(
            context, threadId);
}

void PS2Runtime::captureEeInterruptContinuation(
    std::optional<int> threadId)
{
    m_eeInterruptedContinuation =
        makeEeInterruptedContinuation(
            nullptr,
            threadId.value_or(0));
}

void PS2Runtime::clearEeInterruptContinuation() noexcept
{
    m_eeInterruptedContinuation.reset();
}

void PS2Runtime::beginAsyncCallbackInvocation(
    AsyncCallbackInvocationScope &scope,
    R5900Context *continuationContext,
    int continuationThreadId)
{
    EeInterruptedContinuationFrame interrupted{};
    const bool nested =
        m_asyncCallbackInvocationDepth != 0u;
    const bool explicitContinuation =
        continuationContext != nullptr;
    if (explicitContinuation)
    {
        // Synchronous guest callbacks can intentionally release the guest
        // execution binding around their host-side dispatcher. Preserve the
        // caller supplied at that release boundary instead of mistaking the
        // temporarily unbound runtime for the BIOS idle thread.
        interrupted = makeEeInterruptedContinuation(
            continuationContext,
            continuationThreadId,
            nested);
    }
    else if (scope.m_guestExecution.m_previousContext)
    {
        interrupted = makeEeInterruptedContinuation(
            scope.m_guestExecution.m_previousContext,
            scope.m_guestExecution.m_previousThreadId,
            nested);
    }
    else if (m_eeInterruptedContinuation)
    {
        interrupted = *m_eeInterruptedContinuation;
    }
    else
    {
        interrupted = makeEeInterruptedContinuation(
            nullptr, 0);
    }

    scope.m_interruptedContext = interrupted.context;
    if (scope.m_guestExecution.m_context)
    {
        // Host-invoked guest callbacks use a private register context so
        // their writes cannot leak into the interrupted continuation. Begin
        // that context from the complete owned snapshot; callback executors
        // install their entry PC and ABI argument registers after the scope
        // has been constructed.
        *scope.m_guestExecution.m_context =
            scope.m_interruptedContext;
    }
    scope.m_continuationContext =
        interrupted.continuationContext;
    scope.m_continuationOwner =
        std::move(interrupted.continuationOwner);
    scope.m_continuationGeneration =
        interrupted.generation;
    scope.m_ownerThreadId = interrupted.threadId;
    scope.m_idle = interrupted.idle;
    scope.m_depth = m_asyncCallbackInvocationDepth++;
    scope.m_active = true;

    scope.m_stackTop = kEeBiosInterruptHandlerStack;
    const uint32_t interruptedSp =
        static_cast<uint32_t>(
            _mm_cvtsi128_si32(
                scope.m_interruptedContext.r[29]));
    const uint32_t alignedSp =
        interruptedSp & ~0xFu;
    if (!scope.m_idle &&
        alignedSp >= 0x10u &&
        alignedSp <= PS2_RAM_SIZE)
    {
        scope.m_stackTop = alignedSp;
    }

    // The callback's temporary register context is untracked by the thread
    // map. Bind it explicitly to the interrupted continuation (or thread 0
    // for the BIOS idle frame); never infer ownership from handler
    // registration state.
    selectEeThread(scope.m_ownerThreadId);
}

void PS2Runtime::endAsyncCallbackInvocation(
    AsyncCallbackInvocationScope &scope) noexcept
{
    if (!scope.m_active)
    {
        return;
    }
    if (m_asyncCallbackInvocationDepth != 0u)
    {
        --m_asyncCallbackInvocationDepth;
    }

    // The callback's GPR, COP0, and COP1 snapshots are private interrupt
    // state, but COP2 belongs to the processor and may keep executing while
    // the callback is active. Publish that shared state before the private
    // callback context is unbound so restoring the interrupted context cannot
    // roll VU0 back to its pre-callback snapshot.
    R5900Context *const callbackContext =
        scope.m_guestExecution.m_context;
    if (callbackContext)
    {
        const auto publishCop2 =
            [callbackContext](R5900Context *destination)
            {
                if (!destination || destination == callbackContext)
                {
                    return;
                }
                copyEeCop2ProcessorGlobalState(
                    *destination, *callbackContext);
            };

        publishCop2(&m_cpuContext);
        publishCop2(scope.m_guestExecution.m_previousContext);
        if (scope.m_continuationContext !=
            scope.m_guestExecution.m_previousContext)
        {
            publishCop2(scope.m_continuationContext);
        }
    }
    scope.m_active = false;
}

void PS2Runtime::commitPriorContext(
    std::optional<int> priorThreadId,
    ps2x::timing::EeTickDelta,
    ps2x::timing::EeTick)
{
    // Generated code and runtime helpers already advance the canonical EE
    // timeline. The executor's relative tick is a consistency/accounting
    // view and must not advance that timeline a second time.
    captureEeInterruptContinuation(priorThreadId);
    m_eeExecutorInterruptPassComplete = false;
}

void PS2Runtime::publishSelectedContext(
    std::optional<int> selectedThreadId,
    ps2x::timing::EeTick)
{
    m_eeExecutorDispatchStartTick = currentEeTick();
    if (selectedThreadId.has_value())
    {
        selectEeThread(*selectedThreadId);
        return;
    }

    m_eeThreadRuntimeState->currentThreadId.store(
        0, std::memory_order_release);
    m_boundEeThreadId = 0;
    g_currentThreadRuntime = this;
    g_currentThreadId = 1;
}

void PS2Runtime::publishWaitCompletion(
    int,
    ps2x::ee::EeSchedulerCompletedWait,
    ps2x::timing::EeTick)
{
    // Live syscall adapters mirror completion while applying the scheduler
    // transition. The executor consumes the scheduler's retained result here
    // so a later wait cannot observe the prior completion a second time.
}

bool PS2Runtime::hasImmediateConsequence(
    ps2x::ee::EeSchedulerConsequenceStage stage,
    ps2x::timing::EeTick) const
{
    if (stage !=
            ps2x::ee::EeSchedulerConsequenceStage::
                InterruptCause ||
        m_eeExecutorInterruptPassComplete)
    {
        return false;
    }

    return m_eeAlarmRuntimeState
                   ->pendingCallbackPublished.load(
                       std::memory_order_acquire) ||
           m_pendingVSyncDeliveryCount != 0u ||
           m_pendingEeCounterInterrupts != 0u ||
           m_dmacInterruptDeliveryDirty.load(
               std::memory_order_acquire);
}

ps2x::ee::EeSchedulerReschedulePolicy
PS2Runtime::applyNextConsequence(
    ps2x::ee::EeSchedulerConsequenceStage stage,
    ps2x::timing::EeTick,
    ps2x::ee::EeThreadScheduler &scheduler)
{
    if (stage !=
            ps2x::ee::EeSchedulerConsequenceStage::
                InterruptCause ||
        m_eeExecutorInterruptPassComplete)
    {
        throw std::logic_error(
            "PS2 runtime has no immediate EE scheduler "
            "consequence at this stage");
    }

    // These adapters may execute generated interrupt/callback code. They run
    // only here on the EE executor, after the prior guest continuation has
    // yielded and before the next continuation is selected. Mark the pass
    // complete first so a masked retained cause cannot spin the consequence
    // loop; a later executor boundary will reconsider retained work.
    m_eeExecutorInterruptPassComplete = true;
    EeExecutorConsequenceContext &consequence =
        g_eeExecutorConsequenceContext;
    if (consequence.runtime)
    {
        throw std::logic_error(
            "nested EE executor callback consequence");
    }
    consequence.runtime = this;
    consequence.scheduler = &scheduler;
    consequence.backend =
        m_eeExecutionBackend.get();
    consequence.reschedulePolicy =
        ps2x::ee::
            EeSchedulerReschedulePolicy::None;
    try
    {
        drainPendingAlarmCallbacks();
        drainPendingVSyncHandlers(
            m_memory.getRDRAM());
        drainPendingEeCounterHandlers(
            m_memory.getRDRAM());
        drainCompletedDmacHandlers(
            m_memory.getRDRAM());
    }
    catch (...)
    {
        consequence = {};
        clearEeInterruptContinuation();
        throw;
    }
    const ps2x::ee::EeSchedulerReschedulePolicy
        reschedulePolicy =
            consequence.reschedulePolicy;
    consequence = {};
    clearEeInterruptContinuation();
    acknowledgeGuestPreemptionRequests();
    return reschedulePolicy;
}

void PS2Runtime::runMainEeContinuation()
{
    try
    {
        dispatchLoop(
            m_memory.getRDRAM(),
            &m_cpuContext);
    }
    catch (const ThreadExitException &)
    {
    }

    std::shared_ptr<ThreadInfo> mainInfo;
    {
        EeThreadRuntimeState &state =
            *m_eeThreadRuntimeState;
        std::lock_guard<std::mutex> lock(
            state.threadMapMutex);
        const auto current = state.threads.find(1);
        if (current != state.threads.end())
        {
            mainInfo = current->second;
        }
    }
    if (mainInfo)
    {
        // Started guest workers already make their ThreadInfo dormant when
        // their entry returns. Mirror that lifecycle for the bootstrap
        // continuation before any surviving worker can target thread 1.
        {
            std::lock_guard<std::mutex> lock(
                mainInfo->m);
            mainInfo->guestState.makeDormant();
        }
        mainInfo->cv.notify_all();
    }

    const uint32_t pc = m_cpuContext.pc;
    RUNTIME_LOG(
        "Game thread returned. PC=0x"
        << std::hex << pc
        << " RA=0x"
        << static_cast<uint32_t>(
               _mm_extract_epi32(
                   m_cpuContext.r[31], 0))
        << std::dec << std::endl);
}

void PS2Runtime::startDedicatedEeExecution()
{
    if (!m_eeRuntimeExecutor)
    {
        throw std::logic_error(
            "selected EE backend has no dedicated "
            "executor");
    }

    std::shared_ptr<ThreadInfo> mainInfo;
    {
        EeThreadRuntimeState &state =
            *m_eeThreadRuntimeState;
        std::lock_guard<std::mutex> lock(
            state.threadMapMutex);
        const auto current = state.threads.find(1);
        if (current != state.threads.end())
        {
            mainInfo = current->second;
        }
        else
        {
            mainInfo =
                std::make_shared<ThreadInfo>();
            mainInfo->generation =
                state.allocateThreadGenerationLocked();
            mainInfo->priority = 0u;
            mainInfo->guestState.initializeRunning(0);
            mainInfo->entry = m_cpuContext.pc;
            mainInfo->stack =
                static_cast<uint32_t>(
                    _mm_extract_epi32(
                        m_cpuContext.r[29], 0));
            mainInfo->gp =
                static_cast<uint32_t>(
                    _mm_extract_epi32(
                        m_cpuContext.r[28], 0));
            state.threads.emplace(1, mainInfo);
        }
        mainInfo->boundContext = &m_cpuContext;
        state.contextThreadIds[&m_cpuContext] = 1;
    }

    int mainPriority = 0;
    {
        std::lock_guard<std::mutex> lock(mainInfo->m);
        const EeThreadGuestStateSnapshot guest =
            mainInfo->guestState.snapshot();
        mainPriority = guest.currentPriority;
        if (!guest.started)
        {
            mainInfo->guestState.initializeRunning(
                mainPriority);
        }
    }

    const uint32_t generation =
        mainInfo->generation;
    m_eeRuntimeExecutor->start(
        [this, generation, mainPriority](
            ps2x::ee::EeThreadScheduler &scheduler,
            IEeExecutionBackend &backend)
        {
            ThreadNaming::SetCurrentThreadName(
                "EeExecutor");
            backend.create(
                1,
                [this]()
                {
                    runMainEeContinuation();
                });
            if (!scheduler.addRunningThread(
                    1,
                    generation,
                    mainPriority))
            {
                throw std::logic_error(
                    "failed to seed the main EE "
                    "scheduler record");
            }
        });
}

void PS2Runtime::joinDedicatedEeExecution()
{
    if (!m_eeRuntimeExecutor)
    {
        return;
    }
    m_eeRuntimeExecutor->join();
    m_eeRuntimeExecutor->rethrowFailure();
}

void PS2Runtime::
    startDedicatedEeExecutionForTesting()
{
    m_stopRequested.store(
        false, std::memory_order_relaxed);
    startDedicatedEeExecution();
}

void PS2Runtime::
    stopDedicatedEeExecutionForTesting()
{
    requestStop();
    joinDedicatedEeExecution();
}

void PS2Runtime::startEeExecutionForTesting()
{
    if (usesDedicatedEeExecutor())
    {
        startDedicatedEeExecutionForTesting();
        return;
    }

    if (m_eeThreadRuntimeState->activeHostThreads.load(
            std::memory_order_acquire) != 0 ||
        m_eeExecutionBackend->managedThreadCount() !=
            0u)
    {
        throw std::logic_error(
            "headless EE execution is already active");
    }

    {
        std::lock_guard<std::mutex> lock(
            m_headlessEeExecutionFailureMutex);
        m_headlessEeExecutionFailure = nullptr;
    }
    m_stopRequested.store(
        false, std::memory_order_relaxed);
    m_eeThreadRuntimeState->activeHostThreads.store(
        1, std::memory_order_release);
    try
    {
        m_eeExecutionBackend->create(
            1,
            [this]()
            {
                struct ActiveThreadCompletion
                {
                    EeThreadRuntimeState &state;

                    ~ActiveThreadCompletion()
                    {
                        state.activeHostThreads.fetch_sub(
                            1, std::memory_order_release);
                    }
                } completion{*m_eeThreadRuntimeState};

                ThreadNaming::SetCurrentThreadName(
                    "GameThread");
                try
                {
                    runMainEeContinuation();
                }
                catch (...)
                {
                    {
                        std::lock_guard<std::mutex> lock(
                            m_headlessEeExecutionFailureMutex);
                        m_headlessEeExecutionFailure =
                            std::current_exception();
                    }
                    requestStop();
                }
            });
    }
    catch (...)
    {
        m_eeThreadRuntimeState->activeHostThreads.fetch_sub(
            1, std::memory_order_release);
        throw;
    }
}

void PS2Runtime::stopEeExecutionForTesting()
{
    if (usesDedicatedEeExecutor())
    {
        stopDedicatedEeExecutionForTesting();
        return;
    }

    requestStop();
    ps2_syscalls::joinAllGuestHostThreads(this);
    if (m_eeThreadRuntimeState->activeHostThreads.load(
            std::memory_order_acquire) != 0)
    {
        throw std::logic_error(
            "headless EE host threads remained active "
            "after join");
    }

    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(
            m_headlessEeExecutionFailureMutex);
        failure = std::exchange(
            m_headlessEeExecutionFailure, nullptr);
    }
    if (failure)
    {
        std::rethrow_exception(failure);
    }
}

void PS2Runtime::completeEeExecutionKernelReset()
{
    if (isStopRequested())
    {
        return;
    }

    if (usesDedicatedEeExecutor())
    {
        if (m_eeRuntimeExecutor->ownsCurrentThread())
        {
            throw std::logic_error(
                "EE kernel reset cannot join its own "
                "executor thread");
        }
        m_eeRuntimeExecutor->requestStop();
        joinDedicatedEeExecution();
        return;
    }

    m_eeExecutionBackend->joinAll();
    if (m_eeThreadRuntimeState->activeHostThreads.load(
            std::memory_order_acquire) != 0 ||
        m_eeExecutionBackend->managedThreadCount() !=
            0u)
    {
        throw std::logic_error(
            "EE kernel reset left host continuations "
            "active");
    }
}

const void *
PS2Runtime::hostPresentationUploadStateIdentityForTesting()
    const noexcept
{
    return m_hostPresentationUploadState.get();
}

size_t
PS2Runtime::pendingAlarmCallbackCountForTesting()
{
    EeAlarmRuntimeState &state =
        eeAlarmRuntimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.pendingCallbacks.size();
}

EeAlarmRuntimeState &PS2Runtime::eeAlarmRuntimeState()
{
    return *m_eeAlarmRuntimeState;
}

const EeAlarmRuntimeState &PS2Runtime::eeAlarmRuntimeState() const
{
    return *m_eeAlarmRuntimeState;
}

EeSyncRuntimeState &PS2Runtime::eeSyncRuntimeState()
{
    return *m_eeSyncRuntimeState;
}

const EeSyncRuntimeState &PS2Runtime::eeSyncRuntimeState() const
{
    return *m_eeSyncRuntimeState;
}

EeInterruptRuntimeState &PS2Runtime::eeInterruptRuntimeState()
{
    return *m_eeInterruptRuntimeState;
}

const EeInterruptRuntimeState &PS2Runtime::eeInterruptRuntimeState() const
{
    return *m_eeInterruptRuntimeState;
}

EeKernelRuntimeState &PS2Runtime::eeKernelRuntimeState()
{
    return *m_eeKernelRuntimeState;
}

const EeKernelRuntimeState &PS2Runtime::eeKernelRuntimeState() const
{
    return *m_eeKernelRuntimeState;
}

EeRpcRuntimeState &PS2Runtime::eeRpcRuntimeState()
{
    return *m_eeRpcRuntimeState;
}

const EeRpcRuntimeState &PS2Runtime::eeRpcRuntimeState() const
{
    return *m_eeRpcRuntimeState;
}

EeFileRuntimeState &PS2Runtime::eeFileRuntimeState()
{
    return *m_eeFileRuntimeState;
}

const EeFileRuntimeState &
PS2Runtime::eeFileRuntimeState() const
{
    return *m_eeFileRuntimeState;
}

Deci2RuntimeState &PS2Runtime::deci2RuntimeState()
{
    return *m_deci2RuntimeState;
}

const Deci2RuntimeState &
PS2Runtime::deci2RuntimeState() const
{
    return *m_deci2RuntimeState;
}

ps2_stubs::AudioRuntimeState &
PS2Runtime::audioRuntimeState()
{
    return *m_audioRuntimeState;
}

const ps2_stubs::AudioRuntimeState &
PS2Runtime::audioRuntimeState() const
{
    return *m_audioRuntimeState;
}

ps2_stubs::CdRuntimeState &PS2Runtime::cdRuntimeState()
{
    return *m_cdRuntimeState;
}

const ps2_stubs::CdRuntimeState &
PS2Runtime::cdRuntimeState() const
{
    return *m_cdRuntimeState;
}

ps2_stubs::DmaRuntimeState &PS2Runtime::dmaRuntimeState()
{
    return *m_dmaRuntimeState;
}

const ps2_stubs::DmaRuntimeState &
PS2Runtime::dmaRuntimeState() const
{
    return *m_dmaRuntimeState;
}

ps2_stubs::LibCRuntimeState &PS2Runtime::libcRuntimeState()
{
    return *m_libcRuntimeState;
}

const ps2_stubs::LibCRuntimeState &
PS2Runtime::libcRuntimeState() const
{
    return *m_libcRuntimeState;
}

ps2_stubs::MemoryCardRuntimeState &
PS2Runtime::memoryCardRuntimeState()
{
    return *m_memoryCardRuntimeState;
}

const ps2_stubs::MemoryCardRuntimeState &
PS2Runtime::memoryCardRuntimeState() const
{
    return *m_memoryCardRuntimeState;
}

ps2_stubs::PadRuntimeState &PS2Runtime::padRuntimeState()
{
    return *m_padRuntimeState;
}

const ps2_stubs::PadRuntimeState &
PS2Runtime::padRuntimeState() const
{
    return *m_padRuntimeState;
}

ps2_stubs::SifRuntimeState &PS2Runtime::sifRuntimeState()
{
    return *m_sifRuntimeState;
}

const ps2_stubs::SifRuntimeState &
PS2Runtime::sifRuntimeState() const
{
    return *m_sifRuntimeState;
}

ps2_stubs::MpegRuntimeState &PS2Runtime::mpegRuntimeState()
{
    return *m_mpegRuntimeState;
}

const ps2_stubs::MpegRuntimeState &PS2Runtime::mpegRuntimeState() const
{
    return *m_mpegRuntimeState;
}

int PS2Runtime::activeEeHostThreadCount() const
{
    return m_eeThreadRuntimeState
        ? m_eeThreadRuntimeState->activeHostThreads.load(
              std::memory_order_acquire)
        : 0;
}

int PS2Runtime::currentEeThreadId() noexcept
{
    const auto depthIt =
        g_guestExecutionDepths.find(this);
    const bool ownsGuestExecution =
        depthIt != g_guestExecutionDepths.end() &&
        depthIt->second != 0u;
    if (!ownsGuestExecution)
    {
        // Direct syscall tests and legacy host callbacks can still enter
        // without a GuestExecutionScope. Their TLS slot is only a
        // compatibility identity; it may never override a bound guest.
        if (g_currentThreadRuntime != this)
        {
            g_currentThreadRuntime = this;
            g_currentThreadId = 1;
        }
        return g_currentThreadId;
    }

    const int selected = m_boundEeThreadId;
    if (g_currentThreadRuntime != this ||
        g_currentThreadId != selected)
    {
        m_eeThreadRuntimeState->legacyAdapterMismatches.fetch_add(
            1u, std::memory_order_relaxed);
        g_currentThreadRuntime = this;
        g_currentThreadId = selected;
    }
    return selected;
}

uint64_t
PS2Runtime::eeThreadLegacyAdapterMismatchCount() const noexcept
{
    return m_eeThreadRuntimeState->legacyAdapterMismatches.load(
        std::memory_order_relaxed);
}

void PS2Runtime::setIopPluginSearchPaths(std::vector<std::filesystem::path> paths)
{
    m_iopSubsystem->setPluginSearchPaths(std::move(paths));
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::timing::EeTick PS2Runtime::currentEeTick() const noexcept
{
    return m_eeTimeline.now();
}

GsCommandResult PS2Runtime::submitGsCommand(
    GsCommandPayload payload,
    uint64_t publicationToken)
{
    if (m_gsAsyncEnabled &&
        (std::holds_alternative<GsDrainBatchCommand>(payload) ||
         std::holds_alternative<GsWriteRegisterCommand>(payload) ||
         std::holds_alternative<GsNativePackedGifCommand>(payload) ||
         std::holds_alternative<GsNativeImageUploadCommand>(payload) ||
         std::holds_alternative<GsClearFramebufferCommand>(payload) ||
         std::holds_alternative<GsFieldMarkerCommand>(payload)))
    {
        throw std::logic_error(
            "asynchronous GS hot commands must be submitted by the EE path");
    }
    if (auto *const latch =
            std::get_if<GsLatchPresentationCommand>(&payload);
        latch && !latch->registers.valid)
    {
        if (m_gsAsyncEnabled)
        {
            throw std::logic_error(
                "asynchronous GS presentation latches require an EE-owned privileged-register snapshot");
        }
        latch->registers =
            captureGsPrivilegedRegisters();
    }
    GsCommandResult result = m_gsExecutor->submit(
        std::move(payload),
        m_publishedGsGuestTick.load(
            std::memory_order_acquire),
        publicationToken,
        ps2_syscalls::GetCurrentVSyncTick(this));
    applyGsPrivilegedSideEffects(
        result.privilegedEffects);
    return result;
}

GsCommandResult PS2Runtime::submitEeGsCommand(
    GsCommandPayload payload,
    uint64_t publicationToken)
{
    std::unique_lock<std::mutex> journalLock;
    if (m_gsAsyncEnabled)
    {
        journalLock = std::unique_lock<std::mutex>(
            m_gsEeJournalMutex);
        reapEeGsSubmissions(true);
    }
    if (auto *const latch =
            std::get_if<GsLatchPresentationCommand>(&payload);
        latch && !latch->registers.valid)
    {
        latch->registers =
            captureGsPrivilegedRegisters();
    }
    const uint64_t guestTick = currentEeTick().raw();
    m_publishedGsGuestTick.store(
        guestTick, std::memory_order_release);
    GsCommandResult result = m_gsExecutor->submit(
        std::move(payload), guestTick,
        publicationToken,
        ps2_syscalls::GetCurrentVSyncTick(this));
    applyGsPrivilegedSideEffects(
        result.privilegedEffects);
    return result;
}

GsPrivilegedRegisterSnapshot
PS2Runtime::captureGsPrivilegedRegisters() const noexcept
{
    const GSRegisters &registers = m_memory.gs();
    return {
        .valid = true,
        .pmode = registers.pmode,
        .smode2 = registers.smode2,
        .dispfb1 = registers.dispfb1,
        .display1 = registers.display1,
        .dispfb2 = registers.dispfb2,
        .display2 = registers.display2,
        .bgcolor = registers.bgcolor,
    };
}

void PS2Runtime::applyGsPrivilegedSideEffects(
    const std::vector<GsPrivilegedSideEffect> &effects)
{
    publishGsPrivilegedSideEffects(
        m_memory.gs(), effects);
}

void PS2Runtime::consumePendingEeGsResult(
    GsCommandResult result)
{
    if (!result.completed())
    {
        throw std::logic_error(
            std::string("asynchronous GS command rejected: ") +
            gsCommandTypeName(result.digest.type));
    }
    applyGsPrivilegedSideEffects(
        result.privilegedEffects);
    if (auto *const batch =
            std::get_if<GsDrainBatchResult>(&result.payload))
    {
        m_gifArbiter.recycleDrainBatch(
            std::move(batch->batch));
    }
    else if (const auto *const field =
                 std::get_if<GsFieldMarkerResult>(&result.payload))
    {
        m_gsFieldsCompleted.fetch_add(
            1u, std::memory_order_relaxed);
        m_gsLastCompletedFieldSequence.store(
            field->fieldSequence,
            std::memory_order_release);
    }
}

bool PS2Runtime::consumeFrontEeGsSubmission(bool wait)
{
    if (m_pendingEeGsSubmissions.empty() ||
        (!wait &&
         !m_pendingEeGsSubmissions.front().ready()))
    {
        return false;
    }
    GsCommandSubmission submission =
        std::move(m_pendingEeGsSubmissions.front());
    m_pendingEeGsSubmissions.pop_front();
    m_gsPendingCompletions.store(
        m_pendingEeGsSubmissions.size(),
        std::memory_order_release);
    consumePendingEeGsResult(submission.wait());
    return true;
}

void PS2Runtime::reapEeGsSubmissions(bool waitAll)
{
    while (consumeFrontEeGsSubmission(waitAll))
    {
    }
}

void PS2Runtime::updateGsPendingHighWater(
    size_t pending) noexcept
{
    size_t current = m_gsPendingHighWater.load(
        std::memory_order_relaxed);
    while (current < pending &&
           !m_gsPendingHighWater.compare_exchange_weak(
               current, pending,
               std::memory_order_relaxed,
               std::memory_order_relaxed))
    {
    }
}

void PS2Runtime::submitEeGsDrainBatch(
    GifArbiterDrainBatch batch)
{
    if (!m_gsAsyncEnabled)
    {
        GsDrainBatchCommand command{};
        command.batch = std::move(batch);
        GsCommandResult result =
            submitEeGsCommand(std::move(command));
        m_gifArbiter.recycleDrainBatch(
            std::move(takeGsCommandResult<GsDrainBatchResult>(
                std::move(result))
                .batch));
        return;
    }

    std::lock_guard<std::mutex> lock(
        m_gsEeJournalMutex);
    reapEeGsSubmissions(false);
    while (m_pendingEeGsSubmissions.size() >=
           m_gsAsyncPendingLimit)
    {
        (void)consumeFrontEeGsSubmission(true);
    }

    GsDrainBatchCommand command{};
    command.batch = std::move(batch);
    const uint64_t guestTick = currentEeTick().raw();
    m_publishedGsGuestTick.store(
        guestTick, std::memory_order_release);
    m_pendingEeGsSubmissions.push_back(
        m_threadedGsExecutor->submitAsync(
            std::move(command), guestTick, 0u,
            ps2_syscalls::GetCurrentVSyncTick(this)));
    m_gsPendingCompletions.store(
        m_pendingEeGsSubmissions.size(),
        std::memory_order_release);
    updateGsPendingHighWater(
        m_pendingEeGsSubmissions.size());
    reapEeGsSubmissions(false);
}

void PS2Runtime::submitEeGsFieldMarker(
    uint64_t fieldSequence)
{
    if (!m_gsAsyncEnabled)
        return;

    std::lock_guard<std::mutex> lock(
        m_gsEeJournalMutex);
    reapEeGsSubmissions(false);
    const auto outstandingFields = [this]()
    {
        return m_gsFieldsSubmitted.load(
                   std::memory_order_acquire) -
               m_gsFieldsCompleted.load(
                   std::memory_order_acquire);
    };
    if (m_gsMaximumFieldLead != 0u &&
        outstandingFields() >= m_gsMaximumFieldLead)
    {
        const auto blockedAt =
            std::chrono::steady_clock::now();
        m_gsFieldLeadBlockCount.fetch_add(
            1u, std::memory_order_relaxed);
        while (outstandingFields() >=
               m_gsMaximumFieldLead)
        {
            if (!consumeFrontEeGsSubmission(true))
            {
                throw std::logic_error(
                    "GS field lead has no pending completion");
            }
        }
        m_gsFieldLeadBlockedNanoseconds.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - blockedAt)
                    .count()),
            std::memory_order_relaxed);
    }
    while (m_pendingEeGsSubmissions.size() >=
           m_gsAsyncPendingLimit)
    {
        (void)consumeFrontEeGsSubmission(true);
    }

    const uint64_t guestTick = currentEeTick().raw();
    m_publishedGsGuestTick.store(
        guestTick, std::memory_order_release);
    m_pendingEeGsSubmissions.push_back(
        m_threadedGsExecutor->submitAsync(
            GsFieldMarkerCommand{
                .fieldSequence = fieldSequence,
                .registers = captureGsPrivilegedRegisters(),
            },
            guestTick, 0u,
            ps2_syscalls::GetCurrentVSyncTick(this)));
    m_gsFieldsSubmitted.fetch_add(
        1u, std::memory_order_relaxed);
    m_gsLastSubmittedFieldSequence.store(
        fieldSequence, std::memory_order_release);
    const uint64_t lead = outstandingFields();
    uint64_t highWater = m_gsFieldLeadHighWater.load(
        std::memory_order_relaxed);
    while (highWater < lead &&
           !m_gsFieldLeadHighWater.compare_exchange_weak(
               highWater, lead,
               std::memory_order_relaxed,
               std::memory_order_relaxed))
    {
    }
    m_gsPendingCompletions.store(
        m_pendingEeGsSubmissions.size(),
        std::memory_order_release);
    updateGsPendingHighWater(
        m_pendingEeGsSubmissions.size());

    if (m_gsMaximumFieldLead == 0u)
    {
        while (m_gsLastCompletedFieldSequence.load(
                   std::memory_order_acquire) < fieldSequence)
        {
            if (!consumeFrontEeGsSubmission(true))
            {
                throw std::logic_error(
                    "GS field marker lost its completion");
            }
        }
    }
    else
    {
        reapEeGsSubmissions(false);
    }
}

uint64_t PS2Runtime::lastCompletedGsFieldSequence() const noexcept
{
    if (!m_gsAsyncEnabled || !m_threadedGsExecutor)
        return 0u;
    return m_threadedGsExecutor
        ->lastCompletedFieldSequence();
}

GsAsyncRuntimeStatistics
PS2Runtime::gsAsyncStatistics() const
{
    GsAsyncRuntimeStatistics statistics{
        .enabled = m_gsAsyncEnabled,
        .pendingCompletions =
            m_gsPendingCompletions.load(
                std::memory_order_acquire),
        .pendingHighWater =
            m_gsPendingHighWater.load(
                std::memory_order_acquire),
        .fieldsSubmitted =
            m_gsFieldsSubmitted.load(
                std::memory_order_acquire),
        .fieldsCompleted =
            m_gsFieldsCompleted.load(
                std::memory_order_acquire),
        .lastSubmittedFieldSequence =
            m_gsLastSubmittedFieldSequence.load(
                std::memory_order_acquire),
        .lastCompletedFieldSequence =
            m_gsLastCompletedFieldSequence.load(
                std::memory_order_acquire),
        .fieldLeadHighWater =
            m_gsFieldLeadHighWater.load(
                std::memory_order_acquire),
        .fieldLeadBlockCount =
            m_gsFieldLeadBlockCount.load(
                std::memory_order_acquire),
        .fieldLeadBlockedNanoseconds =
            m_gsFieldLeadBlockedNanoseconds.load(
                std::memory_order_acquire),
    };
    if (m_threadedGsExecutor)
        statistics.owner =
            m_threadedGsExecutor->statistics();
    return statistics;
}

ThreadedVu1ExecutorStatistics
PS2Runtime::vu1OwnerStatistics() const
{
    return m_threadedVu1Executor
        ? m_threadedVu1Executor->statistics()
        : ThreadedVu1ExecutorStatistics{};
}

Vu1AsyncRuntimeStatistics
PS2Runtime::vu1AsyncStatistics() const
{
    Vu1AsyncRuntimeStatistics statistics{
        .enabled =
            m_vu1ExecutionMode ==
                Vu1ExecutionMode::ThreadedAsync,
        .pendingSlice = m_vu1AsyncPendingSlice.load(
            std::memory_order_acquire),
        .deferredSlice = m_vu1AsyncDeferredSlice.load(
            std::memory_order_acquire),
        .slicesSubmitted = m_vu1AsyncSlicesSubmitted.load(
            std::memory_order_acquire),
        .slicesPublished = m_vu1AsyncSlicesPublished.load(
            std::memory_order_acquire),
        .resultsReadyAtEvent =
            m_vu1AsyncResultsReadyAtEvent.load(
                std::memory_order_acquire),
        .resultsLateAtEvent =
            m_vu1AsyncResultsLateAtEvent.load(
                std::memory_order_acquire),
        .eventWaitCount = m_vu1AsyncEventWaitCount.load(
            std::memory_order_acquire),
        .eventWaitNanoseconds =
            m_vu1AsyncEventWaitNanoseconds.load(
                std::memory_order_acquire),
        .maximumEventWaitNanoseconds =
            m_vu1AsyncMaximumEventWaitNanoseconds.load(
                std::memory_order_acquire),
        .budgetFallbackCount =
            m_vu1AsyncBudgetFallbackCount.load(
                std::memory_order_acquire),
        .budgetFallbackCycleDelta =
            m_vu1AsyncBudgetFallbackCycleDelta.load(
                std::memory_order_acquire),
        .budgetFallbackWaitNanoseconds =
            m_vu1AsyncBudgetFallbackWaitNanoseconds.load(
                std::memory_order_acquire),
        .budgetExtensionCount =
            m_vu1AsyncBudgetExtensionCount.load(
                std::memory_order_acquire),
        .budgetExtensionCycleDelta =
            m_vu1AsyncBudgetExtensionCycleDelta.load(
                std::memory_order_acquire),
        .budgetExtensionExecutedCycles =
            m_vu1AsyncBudgetExtensionExecutedCycles.load(
                std::memory_order_acquire),
        .budgetExtensionWaitNanoseconds =
            m_vu1AsyncBudgetExtensionWaitNanoseconds.load(
                std::memory_order_acquire),
        .hazardBarrierCount =
            m_vu1AsyncHazardBarrierCount.load(
                std::memory_order_acquire),
        .hazardWaitNanoseconds =
            m_vu1AsyncHazardWaitNanoseconds.load(
                std::memory_order_acquire),
        .hazardRequeueCount =
            m_vu1AsyncHazardRequeueCount.load(
                std::memory_order_acquire),
        .deferredSliceCount =
            m_vu1AsyncDeferredSliceCount.load(
                std::memory_order_acquire),
        .deferredSliceArmCount =
            m_vu1AsyncDeferredSliceArmCount.load(
                std::memory_order_acquire),
        .forcedDeferredSliceArmCount =
            m_vu1AsyncForcedDeferredSliceArmCount.load(
                std::memory_order_acquire),
        .sliceSubmitCount =
            m_vu1AsyncSliceSubmitCount.load(
                std::memory_order_acquire),
        .sliceSubmitNanoseconds =
            m_vu1AsyncSliceSubmitNanoseconds.load(
                std::memory_order_acquire),
        .maximumSliceSubmitNanoseconds =
            m_vu1AsyncMaximumSliceSubmitNanoseconds.load(
                std::memory_order_acquire),
    };
    if (m_threadedVu1Executor)
        statistics.owner =
            m_threadedVu1Executor->statistics();
    return statistics;
}

Vu1CoarseRuntimeStatistics
PS2Runtime::vu1CoarseStatistics() const
{
    Vu1CoarseRuntimeStatistics statistics{
        .enabled = m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse,
        .pendingInvocation =
            m_vu1CoarsePendingInvocation.load(
                std::memory_order_acquire),
        .initialEstimateCycles =
            m_vu1CoarseInitialEstimateCycles,
        .minimumEstimateCycles =
            m_vu1CoarseMinimumEstimateCycles,
        .maximumLeadCycles =
            m_vu1CoarseMaximumLeadCycles,
        .maximumInvocationCycles =
            m_vu1CoarseMaximumInvocationCycles,
        .maximumPath1Bytes =
            m_vu1CoarseMaximumPath1Bytes,
        .estimatorHistoryCapacity =
            m_vu1CoarseEstimatorHistoryCapacity,
        .estimatorHistorySize =
            m_vu1CoarseEstimatorHistorySize.load(
                std::memory_order_acquire),
        .estimatorHistoryHighWater =
            m_vu1CoarseEstimatorHistoryHighWater.load(
                std::memory_order_acquire),
        .estimatorIdentityCapacity =
            m_vu1CoarseEstimatorIdentityCapacity,
        .estimatorIdentitySize =
            m_vu1CoarseEstimatorIdentitySize.load(
                std::memory_order_acquire),
        .estimatorIdentityHighWater =
            m_vu1CoarseEstimatorIdentityHighWater.load(
                std::memory_order_acquire),
        .estimatorIdentityHitCount =
            m_vu1CoarseEstimatorIdentityHitCount.load(
                std::memory_order_acquire),
        .estimatorIdentityMissCount =
            m_vu1CoarseEstimatorIdentityMissCount.load(
                std::memory_order_acquire),
        .pendingInvocationHighWater =
            m_vu1CoarsePendingInvocationHighWater.load(
                std::memory_order_acquire),
        .invocationsSubmitted =
            m_vu1CoarseInvocationsSubmitted.load(
                std::memory_order_acquire),
        .invocationsCompleted =
            m_vu1CoarseInvocationsCompleted.load(
                std::memory_order_acquire),
        .invocationsPublished =
            m_vu1CoarseInvocationsPublished.load(
                std::memory_order_acquire),
        .invocationsDiscardedAtFailure =
            m_vu1CoarseInvocationsDiscardedAtFailure.load(
                std::memory_order_acquire),
        .invocationsDiscardedAtShutdown =
            m_vu1CoarseInvocationsDiscardedAtShutdown.load(
                std::memory_order_acquire),
        .resultsReadyAtEvent =
            m_vu1CoarseResultsReadyAtEvent.load(
                std::memory_order_acquire),
        .resultsLateAtEvent =
            m_vu1CoarseResultsLateAtEvent.load(
                std::memory_order_acquire),
        .earlyEstimateCount =
            m_vu1CoarseEarlyEstimateCount.load(
                std::memory_order_acquire),
        .exactEstimateCount =
            m_vu1CoarseExactEstimateCount.load(
                std::memory_order_acquire),
        .lateEstimateCount =
            m_vu1CoarseLateEstimateCount.load(
                std::memory_order_acquire),
        .estimatedCycleSum =
            m_vu1CoarseEstimatedCycleSum.load(
                std::memory_order_acquire),
        .actualCycleSum =
            m_vu1CoarseActualCycleSum.load(
                std::memory_order_acquire),
        .absoluteErrorCycleSum =
            m_vu1CoarseAbsoluteErrorCycleSum.load(
                std::memory_order_acquire),
        .maximumAbsoluteErrorCycles =
            m_vu1CoarseMaximumAbsoluteErrorCycles.load(
                std::memory_order_acquire),
        .causalDeadlineDeferralCount =
            m_vu1CoarseCausalDeadlineDeferralCount.load(
                std::memory_order_acquire),
        .causalDeadlineDeferredCycles =
            m_vu1CoarseCausalDeadlineDeferredCycles.load(
                std::memory_order_acquire),
        .maximumCausalDeadlineDeferredCycles =
            m_vu1CoarseMaximumCausalDeadlineDeferredCycles.load(
                std::memory_order_acquire),
        .forcedBarrierCount =
            m_vu1CoarseForcedBarrierCount.load(
                std::memory_order_acquire),
        .nonPublishingDiagnosticSnapshotCount =
            m_vu1CoarseNonPublishingDiagnosticSnapshotCount.load(
                std::memory_order_acquire),
        .nonPublishingDiagnosticsUpdateCount =
            m_vu1CoarseNonPublishingDiagnosticsUpdateCount.load(
                std::memory_order_acquire),
        .nonPublishingBackendChangeCount =
            m_vu1CoarseNonPublishingBackendChangeCount.load(
                std::memory_order_acquire),
        .nonPublishingVifStateUpdateCount =
            m_vu1CoarseNonPublishingVifStateUpdateCount.load(
                std::memory_order_acquire),
        .nonPublishingDiagnosticSnapshotWaitCount =
            m_vu1CoarseNonPublishingDiagnosticSnapshotWaitCount.load(
                std::memory_order_acquire),
        .nonPublishingDiagnosticSnapshotWaitNanoseconds =
            m_vu1CoarseNonPublishingDiagnosticSnapshotWaitNanoseconds.load(
                std::memory_order_acquire),
        .maximumNonPublishingDiagnosticSnapshotWaitNanoseconds =
            m_vu1CoarseMaximumNonPublishingDiagnosticSnapshotWaitNanoseconds.load(
                std::memory_order_acquire),
        .forcedWaitCount =
            m_vu1CoarseForcedWaitCount.load(
                std::memory_order_acquire),
        .forcedWaitNanoseconds =
            m_vu1CoarseForcedWaitNanoseconds.load(
                std::memory_order_acquire),
        .maximumForcedWaitNanoseconds =
            m_vu1CoarseMaximumForcedWaitNanoseconds.load(
                std::memory_order_acquire),
        .path1ReservationsStarted =
            m_vu1CoarsePath1ReservationsStarted.load(
                std::memory_order_acquire),
        .path1ReservationsReleased =
            m_vu1CoarsePath1ReservationsReleased.load(
                std::memory_order_acquire),
        .path1ReservationsInvalidated =
            m_vu1CoarsePath1ReservationsInvalidated.load(
                std::memory_order_acquire),
        .path1ReservationsDiscardedAtFailure =
            m_vu1CoarsePath1ReservationsDiscardedAtFailure.load(
                std::memory_order_acquire),
        .path1ReservationsDiscardedAtShutdown =
            m_vu1CoarsePath1ReservationsDiscardedAtShutdown.load(
                std::memory_order_acquire),
        .path1BytesPublished =
            m_vu1CoarsePath1BytesPublished.load(
                std::memory_order_acquire),
        .maximumInvocationPath1Bytes =
            m_vu1CoarseMaximumInvocationPath1Bytes.load(
                std::memory_order_acquire),
        .lastEstimatedCycles =
            m_vu1CoarseLastEstimatedCycles.load(
                std::memory_order_acquire),
        .lastActualCycles =
            m_vu1CoarseLastActualCycles.load(
                std::memory_order_acquire),
        .lastEstimateErrorCycles =
            m_vu1CoarseLastEstimateErrorCycles.load(
                std::memory_order_acquire),
        .estimatorInputDigest =
            m_vu1CoarseEstimatorInputDigest.load(
                std::memory_order_acquire),
        .estimatorOutputDigest =
            m_vu1CoarseEstimatorOutputDigest.load(
                std::memory_order_acquire),
        .estimatorResultDigest =
            m_vu1CoarseEstimatorResultDigest.load(
                std::memory_order_acquire),
    };
    for (size_t index = 0u;
         index < kVu1CommandTypeCount; ++index)
    {
        statistics.forcedBarriersByCommandType[index] =
            m_vu1CoarseForcedBarriersByCommandType[index].load(
                std::memory_order_acquire);
    }
    if (m_threadedVu1Executor)
        statistics.owner =
            m_threadedVu1Executor->statistics();
    return statistics;
}

VuProgressSnapshot PS2Runtime::vu1ProgressSnapshot() const
{
    return VuProgressSnapshot{
        .enabled = m_publishedVu1ProgressEnabled.load(
            std::memory_order_acquire),
        .active = m_publishedVu1ProgressActive.load(
            std::memory_order_acquire),
        .invocations =
            m_publishedVu1ProgressInvocations.load(
                std::memory_order_acquire),
        .cycles = m_publishedVu1ProgressCycles.load(
            std::memory_order_acquire),
        .pc = m_publishedVu1ProgressPc.load(
            std::memory_order_acquire),
    };
}

void PS2Runtime::setVu1ProgressTrackingEnabled(bool enabled)
{
    m_vu1.setProgressTrackingEnabled(enabled);
    m_publishedVu1ProgressEnabled.store(
        enabled, std::memory_order_release);
    if (!enabled)
    {
        m_publishedVu1ProgressActive.store(
            false, std::memory_order_release);
    }
}

std::shared_ptr<const Vu1Snapshot>
PS2Runtime::snapshotVu1Owner()
{
    std::shared_ptr<const Vu1Snapshot> result;
    const auto capture =
        [this, &result]()
        {
            result = captureVu1OwnerSnapshot();
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
        return result;

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

std::shared_ptr<const Vu1Snapshot>
PS2Runtime::debugSnapshotVu1Owner()
{
    std::shared_ptr<const Vu1Snapshot> result;
    const auto capture =
        [this, &result]()
        {
            result = captureVu1OwnerDiagnosticSnapshot();
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
        return result;

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

void PS2Runtime::cancelEeCounterEvent() noexcept
{
    if (m_eeCounterEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_eeCounterEventToken);
    }
    else
    {
        (void)m_eeEventScheduler.cancel(
            ps2x::timing::EeEventSource::
                EeCounters);
    }
    m_eeCounterEventToken = {
        ps2x::timing::EeEventSource::EeCounters,
        0u};
}

void PS2Runtime::onEeCounterScheduleChanged(
    std::optional<uint64_t> deadlineCycle) noexcept
{
    if (!deadlineCycle.has_value())
    {
        cancelEeCounterEvent();
        return;
    }

    const ps2x::timing::EeTick deadline =
        ps2x::timing::eeTickFromRaw(
            ps2x::timing::eeCyclesToTicks(
                *deadlineCycle)
                .raw());
    const auto scheduled =
        m_eeEventScheduler.event(
            ps2x::timing::EeEventSource::
                EeCounters);
    if (scheduled.has_value() &&
        scheduled->deadline == deadline)
    {
        m_eeCounterEventToken =
            scheduled->token;
        return;
    }

    m_eeCounterEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                EeCounters,
            deadline);
}

void PS2Runtime::onEeCounterInterruptStateChanged(
    uint32_t status,
    uint32_t mask,
    uint32_t newlyRaised) noexcept
{
    constexpr uint32_t kTimerCauseMask =
        (1u << 9u) | (1u << 10u) |
        (1u << 11u) | (1u << 12u);
    m_pendingEeCounterInterrupts &=
        status & kTimerCauseMask;
    m_pendingEeCounterInterrupts |=
        newlyRaised & kTimerCauseMask;

    for (uint32_t cause = 9u;
         cause <= 12u;
         ++cause)
    {
        if ((m_pendingEeCounterInterrupts &
             (1u << cause)) != 0u &&
            (status & mask &
             (1u << cause)) != 0u &&
            ps2_syscalls::isIntcCauseEnabled(
                this, cause))
        {
            requestGuestPreemption();
            return;
        }
    }
}

void PS2Runtime::serviceEeCountersAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_eeCounterEventToken.generation == 0u ||
        service.generation !=
            m_eeCounterEventToken.generation)
    {
        return;
    }
    m_eeCounterEventToken = {
        ps2x::timing::EeEventSource::EeCounters,
        0u};
    (void)m_memory.synchronizeEeCounters(
        ps2x::timing::eeTickToCyclesFloor(
            service.serviceTick));
}

void PS2Runtime::serviceCop0TimerAtEvent(
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    if (m_cop0TimerEventToken.generation == 0u ||
        service.generation !=
            m_cop0TimerEventToken.generation)
    {
        return;
    }
    m_cop0TimerEventToken = {
        ps2x::timing::EeEventSource::Cop0Timer,
        0u};
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(
            service.serviceTick);
    (void)m_cop0Timing.synchronizeCount(cycle);
    publishCop0TimingMirrors(ctx);
    if (!m_cop0Timing.snapshot().timerPending)
    {
        refreshCop0EventSchedules(ctx, cycle);
    }
}

void PS2Runtime::serviceCop0PerformanceAtEvent(
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    if (m_cop0PerformanceEventToken.generation ==
            0u ||
        service.generation !=
            m_cop0PerformanceEventToken.generation)
    {
        return;
    }
    m_cop0PerformanceEventToken = {
        ps2x::timing::EeEventSource::
            Cop0Performance,
        0u};
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(
            service.serviceTick);
    (void)m_cop0Timing.synchronizePerformance(
        cycle,
        ctx ? ctx->cop0_status : 0u);
    publishCop0TimingMirrors(ctx);
    if (!m_cop0Timing.performanceExceptionPending(
            ctx ? ctx->cop0_status : 0u))
    {
        refreshCop0EventSchedules(ctx, cycle);
    }
}

bool PS2Runtime::overlapsEeCounterRegister(
    uint32_t address,
    uint32_t size) noexcept
{
    if (size == 0u)
    {
        return false;
    }
    // The EE scratchpad lives at 0x70000000, which would alias the counter
    // MMIO page if it were normalized with the ordinary KSEG physical mask.
    // Classify this dedicated segment before resolving cached/uncached MMIO
    // aliases.
    if (ps2IsScratchpadAddress(address))
    {
        return false;
    }
    const uint32_t physical = address & 0x1fffffffu;
    const uint64_t end =
        static_cast<uint64_t>(physical) + size;
    constexpr std::array<uint32_t, 4u> offsets{
        0x00u, 0x10u, 0x20u, 0x30u};
    for (size_t counter = 0u;
         counter < ps2x::timing::kEeCounterCount;
         ++counter)
    {
        for (const uint32_t offset : offsets)
        {
            if (counter >= 2u && offset == 0x30u)
            {
                continue;
            }
            const uint64_t reg =
                ps2x::timing::kEeCounterBaseAddress +
                counter *
                    ps2x::timing::kEeCounterStride +
                offset;
            if (static_cast<uint64_t>(physical) <
                    reg + sizeof(uint32_t) &&
                end > reg)
            {
                return true;
            }
        }
    }
    return false;
}

void PS2Runtime::synchronizeEeCounterMmio(
    R5900Context *ctx,
    uint32_t address,
    uint32_t size)
{
    if (!overlapsEeCounterRegister(
            address, size))
    {
        return;
    }
    const ps2x::timing::EeTick now =
        commitEeContextProgress(ctx);
    (void)m_memory.synchronizeEeCounters(
        ps2x::timing::eeTickToCyclesFloor(now));
}

void PS2Runtime::scheduleEeVSyncEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    if (!m_eeVSyncTiming.enabled)
    {
        m_eeVSyncTiming.eventToken = {
            ps2x::timing::EeEventSource::VSync, 0u};
        return;
    }

    m_eeVSyncTiming.eventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::VSync,
            deadline);
}

PS2Runtime::EeVSyncDurations
PS2Runtime::eeVSyncDurationsForVideoMode(
    uint32_t videoMode,
    bool interlaced) noexcept
{
    // These values reproduce PCSX2 vSyncInfoCalc() at revision f22c4b47d
    // using the default 294.912 MHz EE clock and default refresh rates.
    // SetGsCrt treats both BIOS (0/1) and libgs (2/3) encodings as
    // NTSC/PAL. DVD modes use the corresponding standard.
    switch (videoMode & 0xFFu)
    {
    case 0x01u:
    case 0x03u:
    case 0x73u:
    case 0x83u:
        if (interlaced)
        {
            return {
                .periodCycles = 5'898'240u,
                .renderCycles = 5'435'818u,
                .blankCycles = 462'422u,
                .gsBlankCycles = 56'623u,
                .hRenderCycles = 15'795u,
                .hBlankCycles = 3'079u,
                .hSyncErrorCycles = 114u,
            };
        }
        return {
            .periodCycles = 5'926'688u,
            .renderCycles = 5'435'944u,
            .blankCycles = 490'744u,
            .gsBlankCycles = 84'936u,
            .hRenderCycles = 7'898u,
            .hBlankCycles = 1'539u,
            .hSyncErrorCycles = 252u,
        };
    case 0x51u:
    case 0x54u:
        if (interlaced)
        {
            return {
                .periodCycles = 4'915'200u,
                .renderCycles = 4'718'592u,
                .blankCycles = 196'608u,
                .gsBlankCycles = 30'583u,
                .hRenderCycles = 7'313u,
                .hBlankCycles = 1'425u,
                .hSyncErrorCycles = 74u,
            };
        }
        return {
            .periodCycles = 4'915'200u,
            .renderCycles = 4'714'223u,
            .blankCycles = 200'977u,
            .gsBlankCycles = 34'952u,
            .hRenderCycles = 3'657u,
            .hBlankCycles = 712u,
            .hSyncErrorCycles = 74u,
        };
    case 0x50u:
        // PCSX2 treats SDTV 480p as a distinct 59.94 Hz mode with the
        // ordinary 525-line geometry. It therefore must not share the
        // otherwise similar 60 Hz VESA/720p timing below.
        if (interlaced)
        {
            return {
                .periodCycles = 4'920'120u,
                .renderCycles = 4'498'396u,
                .blankCycles = 421'724u,
                .gsBlankCycles = 65'601u,
                .hRenderCycles = 15'685u,
                .hBlankCycles = 3'058u,
                .hSyncErrorCycles = 82u,
            };
        }
        return {
            .periodCycles = 4'920'120u,
            .renderCycles = 4'489'024u,
            .blankCycles = 431'096u,
            .gsBlankCycles = 74'973u,
            .hRenderCycles = 7'842u,
            .hBlankCycles = 1'529u,
            .hSyncErrorCycles = 345u,
        };
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x3Bu:
    case 0x3Cu:
    case 0x3Du:
    case 0x3Eu:
    case 0x4Au:
    case 0x4Bu:
    case 0x52u:
    case 0x53u:
        if (interlaced)
        {
            return {
                .periodCycles = 4'915'200u,
                .renderCycles = 4'493'898u,
                .blankCycles = 421'302u,
                .gsBlankCycles = 65'535u,
                .hRenderCycles = 15'669u,
                .hBlankCycles = 3'055u,
                .hSyncErrorCycles = 149u,
            };
        }
        return {
            .periodCycles = 4'915'200u,
            .renderCycles = 4'484'535u,
            .blankCycles = 430'665u,
            .gsBlankCycles = 74'898u,
            .hRenderCycles = 7'835u,
            .hBlankCycles = 1'527u,
            .hSyncErrorCycles = 149u,
        };
    case 0x00u:
    case 0x02u:
    case 0x72u:
    case 0x82u:
    default:
        if (interlaced)
        {
            return {
                .periodCycles = 4'920'120u,
                .renderCycles = 4'498'396u,
                .blankCycles = 421'724u,
                .gsBlankCycles = 65'601u,
                .hRenderCycles = 15'685u,
                .hBlankCycles = 3'058u,
                .hSyncErrorCycles = 82u,
            };
        }
        return {
            .periodCycles = 4'929'165u,
            .renderCycles = 4'498'098u,
            .blankCycles = 431'067u,
            .gsBlankCycles = 74'968u,
            .hRenderCycles = 7'842u,
            .hBlankCycles = 1'529u,
            .hSyncErrorCycles = 19u,
        };
    }
}

PS2Runtime::EeVSyncVideoModeClass
PS2Runtime::eeVSyncVideoModeClassForMode(
    uint32_t videoMode) noexcept
{
    switch (videoMode & 0xFFu)
    {
    case 0x00u:
    case 0x02u:
        return EeVSyncVideoModeClass::Ntsc;
    case 0x01u:
    case 0x03u:
        return EeVSyncVideoModeClass::Pal;
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x3Bu:
    case 0x3Cu:
    case 0x3Du:
    case 0x3Eu:
    case 0x4Au:
    case 0x4Bu:
        return EeVSyncVideoModeClass::Vesa;
    case 0x50u:
        return EeVSyncVideoModeClass::Sdtv480P;
    case 0x53u:
        return EeVSyncVideoModeClass::Sdtv576P;
    case 0x52u:
        return EeVSyncVideoModeClass::Hdtv720P;
    case 0x51u:
        return EeVSyncVideoModeClass::Hdtv1080I;
    case 0x54u:
        return EeVSyncVideoModeClass::Hdtv1080P;
    case 0x72u:
    case 0x82u:
        return EeVSyncVideoModeClass::DvdNtsc;
    case 0x73u:
    case 0x83u:
        return EeVSyncVideoModeClass::DvdPal;
    default:
        return EeVSyncVideoModeClass::Unknown;
    }
}

void PS2Runtime::configureEeVSyncVideoMode(
    uint32_t videoMode,
    bool interlaced)
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    const EeVSyncVideoModeClass videoModeClass =
        eeVSyncVideoModeClassForMode(videoMode);
    const bool modeClassChanged =
        m_eeVSyncTiming.videoModeClass !=
        videoModeClass;
    m_eeVSyncTiming.durations =
        eeVSyncDurationsForVideoMode(
            videoMode, interlaced);
    m_debugVSyncPeriodCycles.store(
        m_eeVSyncTiming.durations.periodCycles,
        std::memory_order_relaxed);
    m_hostPresentationVideoModeClass.store(
        videoModeClass,
        std::memory_order_relaxed);
    m_eeVSyncTiming.videoMode = videoMode & 0xFFu;
    m_eeVSyncTiming.videoModeClass = videoModeClass;
    m_eeVSyncTiming.videoModeConfigured = true;
    m_eeVSyncTiming.interlaced = interlaced;
    m_memory.configureEeCounterVideoTiming(
        ps2x::timing::EeCounterVideoTiming{
            .renderCycles =
                m_eeVSyncTiming.durations.renderCycles,
            .blankCycles =
                m_eeVSyncTiming.durations.blankCycles,
            .hRenderCycles =
                m_eeVSyncTiming.durations.hRenderCycles,
            .hBlankCycles =
                m_eeVSyncTiming.durations.hBlankCycles,
            .hSyncErrorCycles =
                m_eeVSyncTiming.durations.hSyncErrorCycles,
        },
        ps2x::timing::eeTickToCyclesFloor(
            currentEeTick()));

    if (modeClassChanged)
    {
        // PCSX2 initializes CSR.FIELD high when SetGsCrt changes its
        // normalized GS video-mode class. BIOS/libgs aliases for the same
        // class do not constitute a change. The later GS-blank phase swaps
        // it for interlaced modes.
        constexpr uint64_t kGsCsrFieldMask = 0x2000ull;
        m_memory.gs().csr.fetch_or(
            kGsCsrFieldMask,
            std::memory_order_relaxed);
    }

    // PCSX2's UpdateVSyncRate() adjusts the phase origin by the difference
    // between old and new durations. That leaves the already-pending phase
    // endpoint unchanged; the new timing takes effect when this phase
    // services and schedules its successor.
}

void PS2Runtime::resetEeVSyncStateUnlocked() noexcept
{
    const bool enabled = m_eeVSyncTiming.enabled;
    const EeVSyncDurations durations =
        m_eeVSyncTiming.durations;
    const uint32_t videoMode =
        m_eeVSyncTiming.videoMode;
    const EeVSyncVideoModeClass videoModeClass =
        m_eeVSyncTiming.videoModeClass;
    const bool videoModeConfigured =
        m_eeVSyncTiming.videoModeConfigured;
    const bool interlaced =
        m_eeVSyncTiming.interlaced;
    m_eeVSyncTiming = {};
    m_eeVSyncTiming.enabled = enabled;
    m_eeVSyncTiming.durations = durations;
    m_eeVSyncTiming.videoMode = videoMode;
    m_eeVSyncTiming.videoModeClass = videoModeClass;
    m_eeVSyncTiming.videoModeConfigured =
        videoModeConfigured;
    m_eeVSyncTiming.interlaced = interlaced;
    m_pendingVSyncDeliveryCount = 0u;

    if (enabled)
    {
        scheduleEeVSyncEvent(
            ps2x::timing::saturatingAdd(
                currentEeTick(),
                ps2x::timing::eeCyclesToTicks(
                    m_eeVSyncTiming
                        .durations.renderCycles)));
    }
}

std::chrono::steady_clock::duration
PS2Runtime::hostDurationForEeTicks(
    ps2x::timing::EeTickDelta ticks) noexcept
{
    // The default Emotion Engine clock is 294.912 MHz. This conversion only
    // rate-limits already-scheduled work; its result never feeds back into
    // the emulated timeline.
    constexpr uint64_t kEeTicksPerSecond =
        EE_CYCLES_PER_SECOND *
        ps2x::timing::kEeTicksPerCycle;
    constexpr uint64_t kNanosecondsPerSecond =
        1'000'000'000u;

    const uint64_t rawTicks = ticks.raw();
    const uint64_t wholeSeconds =
        rawTicks / kEeTicksPerSecond;
    const uint64_t remainingTicks =
        rawTicks % kEeTicksPerSecond;
    const uint64_t remainingNanoseconds =
        (remainingTicks * kNanosecondsPerSecond) /
        kEeTicksPerSecond;

    return std::chrono::duration_cast<
        std::chrono::steady_clock::duration>(
            std::chrono::seconds(wholeSeconds) +
            std::chrono::nanoseconds(
                remainingNanoseconds));
}

void PS2Runtime::resetEeVSyncPacingUnlocked() noexcept
{
    m_eeVSyncPacing.anchored =
        m_eeVSyncPacing.enabled;
    m_eeVSyncPacing.eeDeadlineTick =
        currentEeTick();
    m_eeVSyncPacing.hostDeadline =
        std::chrono::steady_clock::now();
}

void PS2Runtime::setRealtimeVSyncPacingEnabled(
    bool enabled)
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    m_eeVSyncPacing.enabled = enabled;
    resetEeVSyncPacingUnlocked();
}

void PS2Runtime::paceEeVSyncStart(
    ps2x::timing::EeTick scheduledTick)
{
    if (!m_eeVSyncPacing.enabled)
    {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const Clock::time_point now = Clock::now();
    if (!m_eeVSyncPacing.anchored ||
        scheduledTick <=
            m_eeVSyncPacing.eeDeadlineTick)
    {
        m_eeVSyncPacing.anchored = true;
        m_eeVSyncPacing.eeDeadlineTick =
            scheduledTick;
        m_eeVSyncPacing.hostDeadline = now;
        return;
    }

    const auto elapsedHostTime =
        hostDurationForEeTicks(
            ps2x::timing::elapsedEeTicks(
                m_eeVSyncPacing.eeDeadlineTick,
                scheduledTick));
    Clock::time_point deadline =
        m_eeVSyncPacing.hostDeadline +
        elapsedHostTime;

    // Match the former worker's bounded catch-up policy. A slow frame or
    // debugger pause must not turn later guest VSync waits into an unbounded
    // burst, while a few late fields may still recover without added delay.
    constexpr uint64_t kMaximumCatchUpFields = 4u;
    const auto maximumLag =
        hostDurationForEeTicks(
            ps2x::timing::eeCyclesToTicks(
                m_eeVSyncTiming
                    .durations.periodCycles *
                kMaximumCatchUpFields));
    if (now > deadline &&
        now - deadline > maximumLag)
    {
        deadline = now - maximumLag;
    }

    if (deadline > now)
    {
        std::this_thread::sleep_until(deadline);
    }

    m_eeVSyncPacing.eeDeadlineTick =
        scheduledTick;
    m_eeVSyncPacing.hostDeadline = deadline;
}

void PS2Runtime::publishEeVSyncField()
{
    constexpr uint64_t kGsCsrFieldMask = 0x2000ull;
    std::atomic<uint64_t> &csr = m_memory.gs().csr;
    if (m_eeVSyncTiming.interlaced)
    {
        csr.fetch_xor(
            kGsCsrFieldMask,
            std::memory_order_relaxed);
    }
    else
    {
        csr.fetch_or(
            kGsCsrFieldMask,
            std::memory_order_relaxed);
    }
    const uint64_t fieldSequence =
        m_debugVSyncFields.fetch_add(
            1u, std::memory_order_relaxed) + 1u;
    submitEeGsFieldMarker(fieldSequence);
}

bool PS2Runtime::queuePendingVSyncDelivery(
    PendingVSyncDelivery delivery) noexcept
{
    if (m_pendingVSyncDeliveryCount >=
        m_pendingVSyncDeliveries.size())
    {
        return false;
    }

    m_pendingVSyncDeliveries[
        m_pendingVSyncDeliveryCount++] = delivery;
    return true;
}

void PS2Runtime::ensureEeVSyncScheduled()
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);

    m_eeVSyncTiming.enabled = true;
    if (m_eeEventScheduler.pending(
            ps2x::timing::EeEventSource::VSync))
    {
        return;
    }

    m_eeVSyncTiming.phase = EeVSyncPhase::Start;
    m_eeVSyncTiming.startTick = {};
    scheduleEeVSyncEvent(
        ps2x::timing::saturatingAdd(
            currentEeTick(),
            ps2x::timing::eeCyclesToTicks(
                m_eeVSyncTiming
                    .durations.renderCycles)));
}

uint64_t PS2Runtime::waitForNextScheduledVSync(
    uint8_t *rdram,
    R5900Context *ctx,
    uint64_t observedVSyncTick)
{
    // Only one blocked VSync waiter may arbitrate idle-time advancement.
    // Other VSync waiters remain outside the guest mutex and will observe the
    // same published tick. This also makes m_guestExecutionWaiters below mean
    // that an ordinary runnable guest is queued and must receive the token
    // before emulated time can be skipped to the next device deadline.
    GuestExecutionReleaseScope releaseGuestExecution(this);
    std::unique_lock<std::mutex> idleAdvanceLock(
        m_eeIdleAdvanceMutex);
    R5900Context *const serviceContext =
        ctx ? ctx : &m_cpuContext;

    while (!isStopRequested())
    {
        const uint64_t currentVSyncTick =
            ps2_syscalls::GetCurrentVSyncTick(this);
        if (currentVSyncTick > observedVSyncTick)
        {
            return currentVSyncTick;
        }

        bool runnableGuestQueued = false;
        uint64_t handoffBaseline = 0u;
        {
            GuestExecutionScope guestExecution(
                this, serviceContext);

            if (ps2_syscalls::GetCurrentVSyncTick(
                    this) >
                observedVSyncTick)
            {
                continue;
            }

            if (m_guestExecutionWaiters.load(
                    std::memory_order_acquire) != 0u)
            {
                runnableGuestQueued = true;
                handoffBaseline =
                    guestExecutionHandoffEpochSnapshot();
            }
            else
            {
                if (!m_eeEventScheduler.pending(
                        ps2x::timing::EeEventSource::
                            VSync))
                {
                    m_eeVSyncTiming.enabled = true;
                    m_eeVSyncTiming.phase =
                        EeVSyncPhase::Start;
                    m_eeVSyncTiming.startTick = {};
                    scheduleEeVSyncEvent(
                        ps2x::timing::saturatingAdd(
                            currentEeTick(),
                            ps2x::timing::eeCyclesToTicks(
                                m_eeVSyncTiming
                                    .durations.renderCycles)));
                }

                const std::optional<ps2x::timing::EeTick>
                    nextDeadline =
                        m_eeEventScheduler.nextDeadline();
                if (!nextDeadline.has_value())
                {
                    requestStop();
                    continue;
                }

                const ps2x::timing::EeTick now =
                    currentEeTick();
                if (*nextDeadline > now)
                {
                    (void)publishEeElapsed(
                        ps2x::timing::elapsedEeTicks(
                            now, *nextDeadline));
                }
                serviceEeEventsAtTick(
                    rdram, serviceContext,
                    currentEeTick());
            }
        }

        if (runnableGuestQueued)
        {
            waitForGuestExecutionHandoff(
                handoffBaseline);
        }
    }

    return ps2_syscalls::GetCurrentVSyncTick(this);
}

ps2x::timing::EeTick PS2Runtime::publishEeElapsed(
    ps2x::timing::EeTickDelta elapsed) noexcept
{
    return m_eeTimeline.advance(elapsed);
}

void PS2Runtime::publishCop0TimingMirrors(
    R5900Context *context) noexcept
{
    const ps2x::timing::Cop0TimingSnapshot state =
        m_cop0Timing.snapshot();
    const auto publish =
        [&state](R5900Context *target)
        {
            if (!target)
            {
                return;
            }
            target->cop0_count = state.count;
            target->cop0_compare = state.compare;
            target->cop0_perf = state.pccr;
            target->cop0_pcr0 = state.pcr[0];
            target->cop0_pcr1 = state.pcr[1];
            if (state.timerPending)
            {
                target->cop0_cause |= COP0_CAUSE_IP7;
            }
            else
            {
                target->cop0_cause &= ~COP0_CAUSE_IP7;
            }
        };

    publish(&m_cpuContext);
    if (m_boundEeContext != &m_cpuContext)
    {
        publish(m_boundEeContext);
    }
    if (context != &m_cpuContext &&
        context != m_boundEeContext)
    {
        publish(context);
    }
}

void PS2Runtime::synchronizeCop0Timing(
    R5900Context *ctx,
    uint64_t currentCycle,
    bool synchronizeCount,
    bool synchronizePerformance) noexcept
{
    R5900Context *const active =
        ctx ? ctx
            : (m_boundEeContext
                   ? m_boundEeContext
                   : &m_cpuContext);
    if (synchronizeCount)
    {
        (void)m_cop0Timing.synchronizeCount(
            currentCycle);
    }
    if (synchronizePerformance)
    {
        (void)m_cop0Timing.synchronizePerformance(
            currentCycle,
            active ? active->cop0_status : 0u);
    }
    publishCop0TimingMirrors(active);
}

void PS2Runtime::refreshCop0EventSchedules(
    R5900Context *ctx,
    uint64_t currentCycle) noexcept
{
    const auto update =
        [this](
            ps2x::timing::EeEventSource source,
            std::optional<uint64_t> deadlineCycle,
            ps2x::timing::EeEventToken &token)
        {
            if (!deadlineCycle.has_value())
            {
                (void)m_eeEventScheduler.cancel(
                    source);
                token = {source, 0u};
                return;
            }

            const ps2x::timing::EeTick deadline =
                ps2x::timing::eeTickFromRaw(
                    ps2x::timing::eeCyclesToTicks(
                        *deadlineCycle)
                        .raw());
            const auto scheduled =
                m_eeEventScheduler.event(source);
            if (scheduled.has_value() &&
                scheduled->deadline == deadline)
            {
                token = scheduled->token;
                return;
            }
            token =
                m_eeEventScheduler.scheduleAbsolute(
                    source, deadline);
        };

    update(
        ps2x::timing::EeEventSource::
            Cop0Timer,
        m_cop0Timing.nextTimerCycle(),
        m_cop0TimerEventToken);

    const R5900Context *const active =
        ctx ? ctx
            : (m_boundEeContext
                   ? m_boundEeContext
                   : &m_cpuContext);
    update(
        ps2x::timing::EeEventSource::
            Cop0Performance,
        m_cop0Timing.nextPerformanceEventCycle(
            currentCycle,
            active ? active->cop0_status : 0u),
        m_cop0PerformanceEventToken);
}

ps2x::timing::EeTick PS2Runtime::prepareCop0Access(
    uint8_t *rdram,
    R5900Context *ctx)
{
    flushEeInstructionIssue(ctx);
    const ps2x::timing::EeTick now =
        commitEeContextProgress(ctx);
    if (m_eeEventScheduler.deadlineDue(now))
    {
        serviceEeEventsAtTick(rdram, ctx, now);
    }
    return now;
}

bool PS2Runtime::cop0TimerInterruptEnabled(
    const R5900Context *ctx) const noexcept
{
    if (!ctx ||
        !m_cop0Timing.snapshot().timerPending)
    {
        return false;
    }
    constexpr uint32_t kGlobalInterruptMask =
        COP0_STATUS_EIE |
        COP0_STATUS_ERL |
        COP0_STATUS_EXL |
        COP0_STATUS_IE;
    constexpr uint32_t kGlobalInterruptEnabled =
        COP0_STATUS_EIE |
        COP0_STATUS_IE;
    return (ctx->cop0_status &
            kGlobalInterruptMask) ==
               kGlobalInterruptEnabled &&
           (ctx->cop0_status &
            COP0_STATUS_IM7) != 0u;
}

void PS2Runtime::deliverPendingCop0Exceptions(
    R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }
    if (m_cop0Timing.performanceExceptionPending(
            ctx->cop0_status))
    {
        raiseCop0PerformanceException(ctx);
    }
    if (cop0TimerInterruptEnabled(ctx))
    {
        raiseCop0Level1Exception(
            ctx, EXCEPTION_INTERRUPT, false);
    }
}

bool PS2Runtime::readCop0Condition0()
{
    constexpr uint32_t kDStat = 0x1000E010u;
    constexpr uint32_t kDPcr = 0x1000E020u;
    constexpr uint32_t kChannelMask = 0x3FFu;
    const uint32_t channelStatus =
        m_memory.readIORegister(kDStat);
    const uint32_t selectedChannels =
        m_memory.readIORegister(kDPcr);
    return ((channelStatus | ~selectedChannels) &
            kChannelMask) == kChannelMask;
}

uint32_t PS2Runtime::readCop0Count(
    uint8_t *rdram,
    R5900Context *ctx)
{
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    const uint32_t value =
        m_cop0Timing.readCount(cycle);
    publishCop0TimingMirrors(ctx);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
    return value;
}

void PS2Runtime::writeCop0Count(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t value)
{
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    m_cop0Timing.writeCount(value, cycle);
    publishCop0TimingMirrors(ctx);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

uint32_t PS2Runtime::readCop0Compare(
    R5900Context *ctx)
{
    publishCop0TimingMirrors(ctx);
    return m_cop0Timing.snapshot().compare;
}

void PS2Runtime::writeCop0Compare(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t value)
{
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    m_cop0Timing.writeCompare(value, cycle);
    publishCop0TimingMirrors(ctx);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

uint32_t PS2Runtime::readCop0Performance(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t selector)
{
    selector &= 0x3fu;
    if ((selector & 1u) == 0u)
    {
        publishCop0TimingMirrors(ctx);
        return m_cop0Timing.snapshot().pccr;
    }

    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    const uint32_t value =
        m_cop0Timing.readPerformance(
            selector, cycle,
            ctx ? ctx->cop0_status : 0u);
    publishCop0TimingMirrors(ctx);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
    return value;
}

void PS2Runtime::writeCop0Breakpoint(
    R5900Context *ctx,
    uint32_t selector,
    uint32_t value)
{
    if (!ctx)
    {
        return;
    }

    // The EE encodes the seven register-24 breakpoint registers in the low
    // eleven instruction bits. Unknown selectors retain the previous no-op
    // behavior while all supported writes now pass one runtime seam.
    switch (selector & 0x7ffu)
    {
    case 0x000u:
        ctx->cop0_bpc = value;
        break;
    case 0x002u:
        ctx->cop0_iab = value;
        break;
    case 0x003u:
        ctx->cop0_iabm = value;
        break;
    case 0x004u:
        ctx->cop0_dab = value;
        break;
    case 0x005u:
        ctx->cop0_dabm = value;
        break;
    case 0x006u:
        ctx->cop0_dvb = value;
        break;
    case 0x007u:
        ctx->cop0_dvbm = value;
        break;
    default:
        break;
    }
}

void PS2Runtime::writeCop0Performance(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t selector,
    uint32_t value)
{
    selector &= 0x3fu;
    if ((selector & 1u) == 0u &&
        selector != 0u)
    {
        return;
    }

    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    (void)m_cop0Timing.writePerformance(
        selector, value, cycle,
        ctx ? ctx->cop0_status : 0u);
    publishCop0TimingMirrors(ctx);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

void PS2Runtime::writeCop0Status(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t value)
{
    if (!ctx)
    {
        return;
    }
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    (void)m_cop0Timing.synchronizeCount(cycle);
    (void)m_cop0Timing.synchronizePerformance(
        cycle, ctx->cop0_status, true);
    ctx->cop0_status =
        value & COP0_STATUS_WRITABLE_MASK;
    publishCop0TimingMirrors(ctx);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

void PS2Runtime::handleCop0Eret(
    uint8_t *rdram,
    R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    synchronizeCop0Timing(ctx, cycle);
    if ((ctx->cop0_status &
         COP0_STATUS_ERL) != 0u)
    {
        ctx->pc = ctx->cop0_errorepc;
        ctx->cop0_status &= ~COP0_STATUS_ERL;
    }
    else
    {
        ctx->pc = ctx->cop0_epc;
        ctx->cop0_status &= ~COP0_STATUS_EXL;
    }
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

void PS2Runtime::handleCop0Ei(
    uint8_t *rdram,
    R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const bool permitted =
        (ctx->cop0_status & 0x00020000u) != 0u ||
        (ctx->cop0_status &
         (COP0_STATUS_EXL |
          COP0_STATUS_ERL)) != 0u ||
        (ctx->cop0_status & 0x18u) == 0u;
    if (permitted)
    {
        ctx->cop0_status |= COP0_STATUS_EIE;
    }
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

void PS2Runtime::handleCop0Di(
    uint8_t *rdram,
    R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }
    const ps2x::timing::EeTick now =
        prepareCop0Access(rdram, ctx);
    const bool permitted =
        (ctx->cop0_status & 0x00020000u) != 0u ||
        (ctx->cop0_status &
         (COP0_STATUS_EXL |
          COP0_STATUS_ERL)) != 0u ||
        (ctx->cop0_status & 0x18u) == 0u;
    if (permitted)
    {
        ctx->cop0_status &= ~COP0_STATUS_EIE;
    }
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    refreshCop0EventSchedules(ctx, cycle);
    deliverPendingCop0Exceptions(ctx);
}

void PS2Runtime::handleEeCacheOperation(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t operation,
    uint32_t virtualAddress)
{
    (void)rdram;
    if (!ctx)
    {
        return;
    }
    try
    {
        if (!m_eeCache.execute(
                m_memory,
                operation,
                virtualAddress,
                ctx->cop0_taglo,
                ctx->cop0_taghi,
                ctx->pc,
                guestAddressTranslation(ctx)))
        {
            SignalException(
                ctx,
                EXCEPTION_RESERVED_INSTRUCTION);
        }
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, false);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(
            ctx,
            PS2BusErrorAccess::DataLoad,
            fault.physicalAddress());
    }
}

ps2x::timing::EeTick PS2Runtime::commitEeContextProgress(
    R5900Context *context) noexcept
{
    if (!context)
    {
        return currentEeTick();
    }
    context->finishEeInstruction();
    return publishEeElapsed(context->commitEeBlockCycles());
}

ps2x::timing::EeTick PS2Runtime::finishEeContextBlock(
    R5900Context *context) noexcept
{
    if (!context)
    {
        return currentEeTick();
    }
    context->finishEeInstruction();
    flushEeInstructionIssue(context);
    return publishEeElapsed(context->finishEeBasicBlock());
}

void PS2Runtime::resetEeTimingUnlocked(
    R5900Context *context) noexcept
{
    R5900Context *const cop0Context =
        context ? context : &m_cpuContext;
    synchronizeCop0Timing(
        cop0Context,
        ps2x::timing::eeTickToCyclesFloor(
            currentEeTick()));
    const ps2x::timing::Cop0TimingSnapshot
        cop0State = m_cop0Timing.snapshot();

    cancelEeCounterEvent();
    cancelVif0DmaEvent();
    cancelVif0VuFinishEvent();
    if (m_memory.vif0DmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::Vif0);
    }
    m_memory.cancelVIF0VuWait();
    cancelVif1DmaEvent();
    if (m_memory.vif1DmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::Vif1);
    }
    cancelGifDmaEvent();
    if (m_memory.gifDmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::Gif);
    }
    cancelFromIpuDmaEvent();
    if (m_memory.fromIpuDmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::FromIpu);
    }
    cancelToIpuDmaEvent();
    if (m_memory.toIpuDmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::ToIpu);
    }
    cancelHleSifDmaEvent();
    for (const DmacChannel channel :
         {DmacChannel::FromScratchpad,
          DmacChannel::ToScratchpad})
    {
        cancelScratchpadDmaEvent(channel);
        if (m_memory
                .scratchpadDmaSnapshot(channel)
                .active)
        {
            (void)m_memory.cancelDmacTransfer(
                channel);
        }
    }
    m_eeTimeline.reset();
    m_publishedGsGuestTick.store(
        0u, std::memory_order_release);
    m_eeEventScheduler.reset();
    m_eeBranchHistoryTags = {};
    m_eeBranchHistoryStates = {};
    cop0Context->ee_performance_issue_pending = false;
    cop0Context->ee_performance_pending_issue = {};
    resetEeVSyncPacingUnlocked();
    m_cop0Timing.reset(
        0u,
        cop0State.count,
        cop0State.compare,
        cop0State.pccr,
        cop0State.pcr[0],
        cop0State.pcr[1]);
    publishCop0TimingMirrors(cop0Context);
    m_pendingEeCounterInterrupts = 0u;
    m_memory.resetEeCounters(0u);
    resetEeVSyncStateUnlocked();
    cancelVU1Execution(true);
    m_vu0CycleTick = {};
    acknowledgeGuestPreemptionRequests();
    if (m_dmacCompletionReady.load(
            std::memory_order_acquire))
    {
        (void)m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                DmacCompletion,
            currentEeTick());
    }
    refreshCop0EventSchedules(
        cop0Context, 0u);

    m_cpuContext.resetEeBlockTiming();
    if (m_boundEeContext && m_boundEeContext != &m_cpuContext)
    {
        m_boundEeContext->resetEeBlockTiming();
    }
    if (context &&
        context != &m_cpuContext &&
        context != m_boundEeContext)
    {
        context->resetEeBlockTiming();
    }
}

void PS2Runtime::resetEeTiming(R5900Context *context)
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    resetEeTimingUnlocked(context);
}

void PS2Runtime::onDmacCompletionReady()
{
    m_dmacCompletionReady.store(
        true, std::memory_order_release);
    (void)m_eeEventScheduler.scheduleAbsolute(
        ps2x::timing::EeEventSource::
            DmacCompletion,
        currentEeTick());
}

void PS2Runtime::requestDmacCompletion(
    DmacChannel channel)
{
    const DmacTransferToken transfer =
        m_memory.beginDmacTransfer(channel);
    (void)m_memory.requestDmacCompletion(transfer);
}

void PS2Runtime::mergePendingDmacInterrupts(
    std::vector<DmacPendingInterrupt> pending)
{
    for (DmacPendingInterrupt &interrupt : pending)
    {
        const auto existing = std::find_if(
            m_pendingDmacInterrupts.begin(),
            m_pendingDmacInterrupts.end(),
            [&interrupt](
                const DmacPendingInterrupt &candidate)
            {
                return candidate.source ==
                           interrupt.source &&
                       candidate.cause ==
                           interrupt.cause;
            });
        if (existing == m_pendingDmacInterrupts.end())
        {
            m_pendingDmacInterrupts.push_back(
                std::move(interrupt));
            continue;
        }

        // D_STAT exposes one level-latched status bit per channel. Repeated
        // completions while that level remains asserted do not create
        // additional interrupt deliveries, but keep the newest publication
        // metadata for diagnostics and a later unmask.
        *existing = std::move(interrupt);
    }
}

void PS2Runtime::collectPendingDmacInterrupts()
{
    std::vector<DmacPendingInterrupt> pending =
        m_memory.consumePendingDmacInterrupts();
    if (pending.empty())
    {
        return;
    }
    mergePendingDmacInterrupts(
        std::move(pending));
    m_dmacInterruptDeliveryDirty.store(
        true, std::memory_order_release);
}

void PS2Runtime::onDmacInterruptStateChanged()
{
    m_dmacInterruptDeliveryDirty.store(
        true, std::memory_order_release);
    collectPendingDmacInterrupts();
    for (const DmacPendingInterrupt &interrupt :
         m_pendingDmacInterrupts)
    {
        if (m_memory.dmacInterruptStatusLatched(
                interrupt.source) &&
            m_memory.dmacInterruptUnmasked(
                interrupt.source) &&
            ps2_syscalls::isDmacCauseEnabled(
                this, interrupt.cause))
        {
            requestGuestPreemption();
            return;
        }
    }
}

size_t PS2Runtime::publishReadyDmacCompletions(
    uint64_t eventSequence)
{
    const size_t published =
        m_memory.publishReadyDmacCompletions(
            eventSequence);
    m_dmacCompletionReady.store(
        m_memory.hasReadyDmacCompletions(),
        std::memory_order_release);
    collectPendingDmacInterrupts();

    if (published != 0u)
    {
        onDmacInterruptStateChanged();
    }
    return published;
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram &&
        m_publishedVu1CodeGeneration.load(
            std::memory_order_acquire) ==
            m_memory.getVU1CodeGeneration())
    {
        return true;
    }

    m_gs.init(
        gsVram,
        static_cast<uint32_t>(PS2_GS_VRAM_SIZE),
        m_gsAsyncEnabled ? nullptr : &m_memory.gs());
    if (m_vu1ExecutionMode == Vu1ExecutionMode::Inline)
    {
        m_vu1CommandProcessor.claimOwnerThread(
            std::this_thread::get_id());
        m_vu1CommandProcessor.bindInlineDiagnosticsObserver(
            &m_memory);
    }
    const uint8_t *const vu1Code = m_memory.getVU1Code();
    const uint8_t *const vu1Data = m_memory.getVU1Data();
    if (!vu1Code || !vu1Data)
        return false;
    (void)submitVu1Command(
        Vu1BindMemoryCommand{
            .microMemory = std::vector<uint8_t>(
                vu1Code, vu1Code + PS2_VU1_CODE_SIZE),
            .dataMemory = std::vector<uint8_t>(
                vu1Data, vu1Data + PS2_VU1_DATA_SIZE),
            .codeGeneration =
                m_memory.getVU1CodeGeneration(),
            .deferredDiagnostics =
                m_vu1ExecutionMode != Vu1ExecutionMode::Inline,
        });
    m_vu1MemoryMirrorDirty = false;
    m_gifArbiter.reset();
    m_gifArbiter.setProcessBatchFn(
        [this](GifArbiterDrainBatch batch)
        {
            submitEeGsDrainBatch(std::move(batch));
        });
    m_memory.setGsPrivilegedObservationCallback(
        m_gsAsyncEnabled
            ? PS2Memory::GsPrivilegedObservationCallback(
                  [this](uint32_t address, bool)
                  {
                      const uint32_t registerOffset =
                          (address - PS2_GS_PRIV_REG_BASE) &
                          ~0x7u;
                      // GIF-authored effects can change CSR/SIGLBLID and
                      // affect interrupt/FIFO control. Display registers are
                      // EE-owned inputs copied into ordered field markers;
                      // synchronizing them would collapse ordinary overlap.
                      if (registerOffset < 0x1000u)
                          return;
                      std::lock_guard<std::mutex> lock(
                          m_gsEeJournalMutex);
                      reapEeGsSubmissions(true);
                  })
            : PS2Memory::GsPrivilegedObservationCallback{});
    m_memory.setGifArbiter(&m_gifArbiter);
    m_memory.setVif1GifPathAvailableCallback(
        [this](bool directHl)
        {
            return m_gifArbiter.canAcceptPath2(
                directHl);
        });
    m_memory.setVu0MscalCallback(
        [this](uint32_t startPC, uint32_t itop)
        {
            startVU0FromVif(startPC, itop);
        });
    m_memory.setVu0MscntCallback(
        [this](uint32_t itop)
        {
            resumeVU0FromVif(itop);
        });
    m_memory.setVu0BusyCallback(
        [this]()
        {
            return m_vu0.isActive();
        });
    m_memory.setVif0ResetCallback(
        [this]()
        {
            cancelVif0VuFinishEvent();
        });
    m_memory.setVu1CommandCallback(
        [this](Vu1CommandPayload payload)
        {
            return submitVu1Command(
                std::move(payload), currentEeTick().raw(),
                m_vu1ExecutionTiming.generation);
        });
    m_memory.setVu1DecodedUnpackCallback(
        [this](Vu1DecodedUnpackCommand command)
        {
            return submitVu1DecodedUnpack(
                std::move(command), currentEeTick().raw(),
                m_vu1ExecutionTiming.generation);
        });
    m_memory.setVu1VifStateUpdateCallback(
        [this](Vu1VifStateUpdateCommand command)
        {
            return submitVu1VifStateUpdate(
                command, currentEeTick().raw(),
                m_vu1ExecutionTiming.generation);
        });
    m_memory.setVu1MscalCallback(
        [this](uint32_t startPC, uint32_t top,
               uint32_t itop, bool flushBeforeStart)
        {
            startVU1FromVif(
                startPC, top, itop, flushBeforeStart);
        });
    m_memory.setVu1MscntCallback(
        [this](uint32_t top, uint32_t itop)
        {
            resumeVU1FromVif(top, itop);
        });
    m_memory.setVu1BusyCallback(
        [this]()
        {
            const bool busy =
                m_publishedVu1Active.load(
                std::memory_order_acquire);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
            if (m_debugVu1TimingTraceEnabled.load(
                    std::memory_order_relaxed))
            {
                DebugVu1TimingEntry entry =
                    debugVu1TimingEntry(
                        DebugVu1TimingEventKind::
                            VifBusyObservation,
                        m_boundEeContext);
                entry.observedValue = busy;
                debugRecordVu1Timing(std::move(entry));
            }
#endif
            return busy;
        });
    m_memory.setVu1MemoryObservationCallback(
        [this]()
        {
            synchronizeVu1MemoryMirror();
        });
    m_memory.setVif1ResetCallback(
        [this]()
        {
            cancelVU1Execution(true);
        });
    resetIop();
    m_vu0.reset();
    (void)submitVu1Command(Vu1ResetCommand{});
    resetEeTimingUnlocked(&m_cpuContext);
    m_vu0InvocationSequence.store(0u, std::memory_order_relaxed);
    m_vu0CurrentInvocation.store(0u, std::memory_order_relaxed);
    m_vu0CurrentInvocationInstruction.store(
        0u, std::memory_order_relaxed);

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

void PS2Runtime::setVU1BusyFlag(
    R5900Context *context, bool busy) noexcept
{
    const auto update =
        [busy](R5900Context *target)
        {
            if (!target)
                return;
            if (busy)
                target->vu0_vpu_stat |= kVu1BusyMask;
            else
                target->vu0_vpu_stat &= ~kVu1BusyMask;
        };

    update(&m_cpuContext);
    if (m_boundEeContext != &m_cpuContext)
        update(m_boundEeContext);
    if (context != &m_cpuContext &&
        context != m_boundEeContext)
    {
        update(context);
    }
}

bool PS2Runtime::readCop2Condition(
    const R5900Context *ctx) const noexcept
{
    const bool busy = ctx != nullptr &&
        (ctx->vu0_vpu_stat & kVu1BusyMask) != 0u;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        DebugVu1TimingEntry entry = debugVu1TimingEntry(
            DebugVu1TimingEventKind::
                Cop2ConditionObservation,
            ctx);
        entry.observedValue = busy;
        debugRecordVu1Timing(std::move(entry));
    }
#endif
    return busy;
}

void PS2Runtime::startVU0FromVif(
    uint32_t startPC, uint32_t itop)
{
    R5900Context *const context =
        m_boundEeContext
            ? m_boundEeContext
            : &m_cpuContext;
    const ps2x::timing::EeTick startTick =
        commitEeContextProgress(context);
    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    if (!vu0Code || !vu0Data ||
        startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(context);
        m_vu0.reset();
        m_vu0CycleTick = startTick;
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(context, m_vu0.state());
    beginVu0Invocation();
    m_vu0.execute(
        vu0Code, PS2_VU0_CODE_SIZE,
        vu0Data, PS2_VU0_DATA_SIZE,
        m_gs, &m_memory, startPC, 0u, itop,
        kVu0MinimumRunCycles);
    copyVu0StateToContext(m_vu0.state(), context);
    context->vu0_vpu_stat =
        (context->vu0_vpu_stat & ~1u) |
        (m_vu0.isActive() ? 1u : 0u);
    m_vu0CycleTick = startTick;
    if (m_vu0.isActive())
    {
        m_vu0CycleTick =
            ps2x::timing::saturatingAdd(
                startTick,
                ps2x::timing::vuCyclesToEeTicks(
                    kVu0MinimumRunCycles));
    }
}

void PS2Runtime::resumeVU0FromVif(uint32_t itop)
{
    R5900Context *const context =
        m_boundEeContext
            ? m_boundEeContext
            : &m_cpuContext;
    const ps2x::timing::EeTick startTick =
        commitEeContextProgress(context);
    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    if (!vu0Code || !vu0Data)
    {
        seedVu0IdleSuccess(context);
        m_vu0.reset();
        m_vu0CycleTick = startTick;
        return;
    }

    copyVu0ContextRegistersToState(
        context, m_vu0.state());
    m_vu0.resumeState(0u, itop, &m_memory);
    (void)m_vu0.advance(
        vu0Code, PS2_VU0_CODE_SIZE,
        vu0Data, PS2_VU0_DATA_SIZE,
        m_gs, &m_memory, kVu0MinimumRunCycles);
    copyVu0StateToContext(m_vu0.state(), context);
    context->vu0_vpu_stat =
        (context->vu0_vpu_stat & ~1u) |
        (m_vu0.isActive() ? 1u : 0u);
    m_vu0CycleTick = startTick;
    if (m_vu0.isActive())
    {
        m_vu0CycleTick =
            ps2x::timing::saturatingAdd(
                startTick,
                ps2x::timing::vuCyclesToEeTicks(
                    kVu0MinimumRunCycles));
    }
}

void PS2Runtime::scheduleVif0VuFinishEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vif0VuFinishEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                VifVu0Finish,
            deadline);
}

void PS2Runtime::cancelVif0VuFinishEvent() noexcept
{
    if (m_vif0VuFinishEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vif0VuFinishEventToken);
    }
    m_vif0VuFinishEventToken = {
        ps2x::timing::EeEventSource::VifVu0Finish,
        0u};
}

bool PS2Runtime::scheduleVif0DmaFromMemory(
    uint32_t delayEeCycles)
{
    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleVif0DmaEvent(
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    return m_vif0DmaEventToken.generation != 0u;
}

void PS2Runtime::scheduleVif0DmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vif0DmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::DmacVif0,
            deadline);
}

void PS2Runtime::cancelVif0DmaEvent() noexcept
{
    if (m_vif0DmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vif0DmaEventToken);
    }
    m_vif0DmaEventToken = {
        ps2x::timing::EeEventSource::DmacVif0,
        0u};
}

void PS2Runtime::scheduleVif0DmaFollowup(
    ps2x::timing::EeTick serviceTick,
    uint32_t delayEeCycles,
    bool allowImmediateRescan) noexcept
{
    const uint32_t effectiveDelay =
        allowImmediateRescan && delayEeCycles < 4u
            ? 0u
            : delayEeCycles;
    scheduleVif0DmaEvent(
        ps2x::timing::saturatingAdd(
            serviceTick,
            ps2x::timing::eeCyclesToTicks(
                effectiveDelay)));
}

void PS2Runtime::serviceVif0DmaAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_vif0DmaEventToken.generation == 0u ||
        service.generation !=
            m_vif0DmaEventToken.generation)
    {
        return;
    }

    m_vif0DmaEventToken = {
        ps2x::timing::EeEventSource::DmacVif0,
        0u};
    const Vif0DmaAdvanceResult advance =
        m_memory.advanceVif0Dma();
    if (!advance.active || advance.completed)
        return;

    if (advance.progressed)
    {
        scheduleVif0DmaFollowup(
            service.serviceTick,
            advance.delayEeCycles, true);
        return;
    }

    if (advance.stall ==
        Vif0DmaStallReason::WaitingForVu)
    {
        scheduleVif0VuFinishEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::eeCyclesToTicks(
                    kVu0VifFinishStartupCycles)));
        return;
    }

    if (advance.stall ==
        Vif0DmaStallReason::None)
    {
        scheduleVif0DmaFollowup(
            service.serviceTick,
            advance.delayEeCycles, true);
    }
}

void PS2Runtime::serviceVU0AtVifFinishEvent(
    uint8_t *rdram,
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    if (m_vif0VuFinishEventToken.generation == 0u ||
        service.generation !=
            m_vif0VuFinishEventToken.generation)
    {
        return;
    }
    m_vif0VuFinishEventToken = {
        ps2x::timing::EeEventSource::VifVu0Finish,
        0u};

    R5900Context *const context =
        ctx ? ctx
            : (m_boundEeContext
                   ? m_boundEeContext
                   : &m_cpuContext);
    if (m_vu0.isActive())
    {
        ps2x::timing::EeTick nextEventTick{};
        const auto nextDeadline =
            m_eeEventScheduler.nextDeadline();
        if (nextDeadline.has_value())
            nextEventTick = *nextDeadline;
        synchronizeVU0MicroprogramAtTick(
            rdram, context, false, false, true,
            service.serviceTick, nextEventTick);
    }

    if (m_vu0.isActive())
    {
        scheduleVif0VuFinishEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::vuCyclesToEeTicks(
                    kVu0VifFinishFollowupCycles)));
        return;
    }

    context->vu0_vpu_stat &= ~1u;
    if (!m_memory.resumeVIF0AfterVu())
        return;

    const Vif0DmaAdvanceResult advance =
        m_memory.advanceVif0Dma();
    if (!advance.active || advance.completed)
        return;
    if (advance.progressed)
    {
        // PCSX2 resumes VIF directly from VIF_VU0_FINISH. The resulting
        // sub-four-cycle DMA tail is a real later callback, not an active
        // DMAC interrupt-scan rescan.
        scheduleVif0DmaFollowup(
            service.serviceTick,
            advance.delayEeCycles, false);
        return;
    }
    if (advance.stall ==
        Vif0DmaStallReason::WaitingForVu)
    {
        scheduleVif0VuFinishEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::eeCyclesToTicks(
                    kVu0VifFinishStartupCycles)));
        return;
    }
    if (advance.stall ==
        Vif0DmaStallReason::None)
    {
        scheduleVif0DmaFollowup(
            service.serviceTick,
            advance.delayEeCycles, false);
    }
}

void PS2Runtime::scheduleVU1Event(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vu1ExecutionTiming.eventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::VifVu1Finish,
            deadline);
}

bool PS2Runtime::scheduleVif1DmaFromMemory(
    uint32_t delayEeCycles)
{
    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleVif1DmaEvent(
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    return m_vif1DmaEventToken.generation != 0u;
}

void PS2Runtime::scheduleVif1DmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vif1DmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::DmacVif1,
            deadline);
}

void PS2Runtime::cancelVif1DmaEvent() noexcept
{
    if (m_vif1DmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vif1DmaEventToken);
    }
    m_vif1DmaEventToken = {
        ps2x::timing::EeEventSource::DmacVif1,
        0u};
}

void PS2Runtime::serviceVif1DmaAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_vif1DmaEventToken.generation == 0u ||
        service.generation !=
            m_vif1DmaEventToken.generation)
    {
        return;
    }

    m_vif1DmaEventToken = {
        ps2x::timing::EeEventSource::DmacVif1,
        0u};
    Vif1DmaAdvanceResult advance{};
    beginVu1SynchronousCommandBatch();
    try
    {
        advance = m_memory.advanceVif1Dma();
    }
    catch (...)
    {
        finishVu1SynchronousCommandBatch(false);
        throw;
    }
    const bool scheduleFollowup =
        advance.active && !advance.completed &&
        (advance.stall == Vif1DmaStallReason::None ||
         advance.stall == Vif1DmaStallReason::GifPath);
    if (scheduleFollowup)
    {
        // Publish the known callback before resolving the VU1 hazard. A VIF1
        // callback at or before the VU1 deadline would otherwise invalidate
        // the immediately requeued speculative slice.
        scheduleVif1DmaEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::eeCyclesToTicks(
                    advance.delayEeCycles)));
    }
    try
    {
        finishVu1SynchronousCommandBatch(true);
    }
    catch (...)
    {
        if (scheduleFollowup)
            cancelVif1DmaEvent();
        throw;
    }
}

bool PS2Runtime::scheduleGifDmaFromMemory(
    uint32_t delayEeCycles)
{
    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleGifDmaEvent(
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    return m_gifDmaEventToken.generation != 0u;
}

void PS2Runtime::scheduleGifDmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_gifDmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::DmacGif,
            deadline);
}

void PS2Runtime::cancelGifDmaEvent() noexcept
{
    if (m_gifDmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_gifDmaEventToken);
    }
    m_gifDmaEventToken = {
        ps2x::timing::EeEventSource::DmacGif,
        0u};
}

void PS2Runtime::serviceGifDmaAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_gifDmaEventToken.generation == 0u ||
        service.generation !=
            m_gifDmaEventToken.generation)
    {
        return;
    }

    m_gifDmaEventToken = {
        ps2x::timing::EeEventSource::DmacGif,
        0u};
    const GifDmaAdvanceResult advance =
        m_memory.advanceGifDma();
    if (!advance.active || advance.completed)
        return;

    if (advance.stall ==
            GifDmaStallReason::None ||
        advance.stall ==
            GifDmaStallReason::GifPaused)
    {
        scheduleGifDmaEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::eeCyclesToTicks(
                    advance.delayEeCycles)));
    }
}

bool PS2Runtime::scheduleFromIpuDmaFromMemory(
    uint32_t delayEeCycles)
{
    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleFromIpuDmaEvent(
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    return m_fromIpuDmaEventToken.generation != 0u;
}

void PS2Runtime::scheduleFromIpuDmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_fromIpuDmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                DmacFromIpu,
            deadline);
    m_memory.setFromIpuDmaEventPending(
        m_fromIpuDmaEventToken.generation != 0u);
}

void PS2Runtime::cancelFromIpuDmaEvent() noexcept
{
    if (m_fromIpuDmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_fromIpuDmaEventToken);
    }
    m_fromIpuDmaEventToken = {
        ps2x::timing::EeEventSource::DmacFromIpu,
        0u};
    m_memory.setFromIpuDmaEventPending(false);
}

void PS2Runtime::serviceFromIpuDmaAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_fromIpuDmaEventToken.generation == 0u ||
        service.generation !=
            m_fromIpuDmaEventToken.generation)
    {
        return;
    }

    m_fromIpuDmaEventToken = {
        ps2x::timing::EeEventSource::DmacFromIpu,
        0u};
    m_memory.setFromIpuDmaEventPending(false);
    const FromIpuDmaAdvanceResult advance =
        m_memory.advanceFromIpuDma();
    if (!advance.active || advance.completed)
        return;

    if (advance.stall ==
        FromIpuDmaStallReason::None)
    {
        scheduleFromIpuDmaEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::eeCyclesToTicks(
                    advance.delayEeCycles)));
    }
}

bool PS2Runtime::scheduleToIpuDmaFromMemory(
    uint32_t delayEeCycles)
{
    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleToIpuDmaEvent(
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    return m_toIpuDmaEventToken.generation != 0u;
}

void PS2Runtime::scheduleToIpuDmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_toIpuDmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                DmacToIpu,
            deadline);
    m_memory.setToIpuDmaEventPending(
        m_toIpuDmaEventToken.generation != 0u);
}

void PS2Runtime::cancelToIpuDmaEvent() noexcept
{
    if (m_toIpuDmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_toIpuDmaEventToken);
    }
    m_toIpuDmaEventToken = {
        ps2x::timing::EeEventSource::DmacToIpu,
        0u};
    m_memory.setToIpuDmaEventPending(false);
}

void PS2Runtime::serviceToIpuDmaAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_toIpuDmaEventToken.generation == 0u ||
        service.generation !=
            m_toIpuDmaEventToken.generation)
    {
        return;
    }

    m_toIpuDmaEventToken = {
        ps2x::timing::EeEventSource::DmacToIpu,
        0u};
    m_memory.setToIpuDmaEventPending(false);
    const ToIpuDmaAdvanceResult advance =
        m_memory.advanceToIpuDma();
    if (!advance.active || advance.completed)
        return;

    if (advance.stall ==
        ToIpuDmaStallReason::None)
    {
        scheduleToIpuDmaEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::eeCyclesToTicks(
                    advance.delayEeCycles)));
    }
}

uint64_t PS2Runtime::hleSifDmaIopCycles(
    const HleSifDmaSubmission &submission) noexcept
{
    uint64_t cycles = 0u;
    const uint32_t transferCount = std::min<uint32_t>(
        submission.transferCount,
        static_cast<uint32_t>(
            kMaximumHleSifDmaTransfers));
    for (uint32_t index = 0u;
         index < transferCount; ++index)
    {
        const uint64_t size =
            submission.transfers[index].size;
        cycles += (size + 15u) / 16u;
    }
    return std::max<uint64_t>(cycles, 1u);
}

bool PS2Runtime::executeHleSifDmaOperation(
    uint8_t *rdram,
    const HleSifDmaSubmission &submission)
{
    if (!rdram ||
        submission.transferCount >
            kMaximumHleSifDmaTransfers)
    {
        return false;
    }

    uint8_t *const iopRam = m_memory.getIOPRAM();
    for (uint32_t index = 0u;
         index < submission.transferCount; ++index)
    {
        const HleSifDmaTransfer &transfer =
            submission.transfers[index];
        notifyIopSifTransfer(
            rdram,
            {
                ps2x::iop::SifTransferKind::SetDma,
                ps2x::iop::SifTransferPhase::BeforeCopy,
                transfer.src,
                transfer.dest,
                transfer.size,
            });

        bool copied = true;
        if (transfer.destination ==
            HleSifDmaDestination::IopRam)
        {
            copied =
                iopRam &&
                transfer.iopOffset <= PS2_IOP_RAM_SIZE &&
                transfer.size <=
                    PS2_IOP_RAM_SIZE -
                        transfer.iopOffset;
            for (uint32_t byte = 0u;
                 copied && byte < transfer.size;
                 ++byte)
            {
                const uint8_t *const source =
                    getConstMemPtr(
                        rdram, transfer.src + byte);
                if (!source)
                {
                    copied = false;
                    break;
                }
                iopRam[transfer.iopOffset + byte] =
                    *source;
            }
        }
        else
        {
            ps2TraceGuestRangeWrite(
                rdram, transfer.dest, transfer.size,
                "hleSifDmaCopy", nullptr);
            const uint64_t sourceBegin = transfer.src;
            const uint64_t sourceEnd =
                sourceBegin + transfer.size;
            const uint64_t destinationBegin =
                transfer.dest;
            const bool copyBackward =
                destinationBegin > sourceBegin &&
                destinationBegin < sourceEnd;
            if (copyBackward)
            {
                for (uint32_t remaining = transfer.size;
                     remaining > 0u; --remaining)
                {
                    const uint32_t byte = remaining - 1u;
                    const uint8_t *const source =
                        getConstMemPtr(
                            rdram, transfer.src + byte);
                    uint8_t *const destination =
                        getMemPtr(
                            rdram, transfer.dest + byte);
                    if (!source || !destination)
                    {
                        copied = false;
                        break;
                    }
                    *destination = *source;
                }
            }
            else
            {
                for (uint32_t byte = 0u;
                     byte < transfer.size; ++byte)
                {
                    const uint8_t *const source =
                        getConstMemPtr(
                            rdram, transfer.src + byte);
                    uint8_t *const destination =
                        getMemPtr(
                            rdram, transfer.dest + byte);
                    if (!source || !destination)
                    {
                        copied = false;
                        break;
                    }
                    *destination = *source;
                }
            }
        }

        if (!copied)
        {
            return false;
        }

        notifyIopSifTransfer(
            rdram,
            {
                ps2x::iop::SifTransferKind::SetDma,
                ps2x::iop::SifTransferPhase::AfterCopy,
                transfer.src,
                transfer.dest,
                transfer.size,
            });
    }
    return true;
}

void PS2Runtime::scheduleHleSifDmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_hleSifDmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::HleSif1,
            deadline);
}

void PS2Runtime::cancelHleSifDmaEvent() noexcept
{
    if (m_hleSifDmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_hleSifDmaEventToken);
    }
    m_hleSifDmaEventToken = {
        ps2x::timing::EeEventSource::HleSif1, 0u};
    m_hleSifDmaQueueHead = 0u;
    m_hleSifDmaQueueCount = 0u;
    m_hleSifDmaQueue.fill({});
}

bool PS2Runtime::submitHleSifDma(
    uint8_t *rdram,
    R5900Context *ctx,
    const HleSifDmaSubmission &submission)
{
    if (!rdram || submission.transferId == 0u ||
        submission.transferCount >
            kMaximumHleSifDmaTransfers)
    {
        return false;
    }

    if (m_hleSifDmaQueueCount >=
        kHleSifDmaQueueCapacity)
    {
        return false;
    }

    const ps2x::timing::EeTick now =
        commitEeContextProgress(ctx);
    const size_t insertion =
        (m_hleSifDmaQueueHead +
         m_hleSifDmaQueueCount) %
        kHleSifDmaQueueCapacity;
    HleSifDmaOperation &operation =
        m_hleSifDmaQueue[insertion];
    operation = {};
    operation.submission = submission;
    operation.operationGeneration =
        ++m_hleSifDmaOperationGeneration;
    if (operation.operationGeneration == 0u)
    {
        operation.operationGeneration =
            ++m_hleSifDmaOperationGeneration;
    }
    operation.iopCycles =
        hleSifDmaIopCycles(submission);
    ++m_hleSifDmaQueueCount;

    if (m_hleSifDmaQueueCount == 1u)
    {
        scheduleHleSifDmaEvent(
            ps2x::timing::saturatingAdd(
                now,
                ps2x::timing::iopCyclesToEeTicks(
                    operation.iopCycles)));
    }
    return m_hleSifDmaEventToken.generation != 0u;
}

int32_t PS2Runtime::hleSifDmaStatus(
    uint32_t transferId) const noexcept
{
    if (transferId == 0u)
    {
        return -1;
    }
    for (size_t offset = 0u;
         offset < m_hleSifDmaQueueCount; ++offset)
    {
        const size_t index =
            (m_hleSifDmaQueueHead + offset) %
            kHleSifDmaQueueCapacity;
        if (m_hleSifDmaQueue[index]
                .submission.transferId == transferId)
        {
            return offset == 0u ? 0 : 1;
        }
    }
    return -1;
}

void PS2Runtime::serviceHleSifDmaAtEvent(
    uint8_t *rdram,
    const ps2x::timing::EeEventService &service)
{
    if (m_hleSifDmaEventToken.generation == 0u ||
        service.generation !=
            m_hleSifDmaEventToken.generation ||
        m_hleSifDmaQueueCount == 0u)
    {
        return;
    }

    m_hleSifDmaEventToken = {
        ps2x::timing::EeEventSource::HleSif1, 0u};
    HleSifDmaOperation &operation =
        m_hleSifDmaQueue[m_hleSifDmaQueueHead];
    if (!executeHleSifDmaOperation(
            rdram, operation.submission))
    {
        std::cerr
            << "HLE SIF DMA failed after validated submission"
            << std::endl;
        m_hleSifDmaQueueHead = 0u;
        m_hleSifDmaQueueCount = 0u;
        m_hleSifDmaQueue.fill({});
        requestStop();
        return;
    }

    operation = {};
    m_hleSifDmaQueueHead =
        (m_hleSifDmaQueueHead + 1u) %
        kHleSifDmaQueueCapacity;
    --m_hleSifDmaQueueCount;
    requestDmacCompletion(DmacChannel::Sif0);

    if (m_hleSifDmaQueueCount != 0u)
    {
        const HleSifDmaOperation &next =
            m_hleSifDmaQueue[
                m_hleSifDmaQueueHead];
        scheduleHleSifDmaEvent(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::iopCyclesToEeTicks(
                    next.iopCycles)));
    }
}

ps2x::timing::EeEventSource
PS2Runtime::scratchpadDmaEventSource(
    DmacChannel channel) noexcept
{
    switch (channel)
    {
    case DmacChannel::FromScratchpad:
        return ps2x::timing::EeEventSource::
            DmacFromScratchpad;
    case DmacChannel::ToScratchpad:
        return ps2x::timing::EeEventSource::
            DmacToScratchpad;
    default:
        return ps2x::timing::EeEventSource::Count;
    }
}

bool PS2Runtime::scheduleScratchpadDmaFromMemory(
    DmacChannel channel,
    uint32_t delayEeCycles)
{
    if (scratchpadDmaEventSource(channel) ==
            ps2x::timing::EeEventSource::Count)
    {
        return false;
    }

    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleScratchpadDmaEvent(
        channel,
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    const ps2x::timing::EeEventToken &token =
        channel == DmacChannel::FromScratchpad
            ? m_fromScratchpadDmaEventToken
            : m_toScratchpadDmaEventToken;
    return token.generation != 0u;
}

void PS2Runtime::scheduleScratchpadDmaEvent(
    DmacChannel channel,
    ps2x::timing::EeTick deadline) noexcept
{
    const ps2x::timing::EeEventSource source =
        scratchpadDmaEventSource(channel);
    if (source == ps2x::timing::EeEventSource::Count)
        return;

    const ps2x::timing::EeEventToken token =
        m_eeEventScheduler.scheduleAbsolute(
            source, deadline);
    if (channel == DmacChannel::FromScratchpad)
        m_fromScratchpadDmaEventToken = token;
    else
        m_toScratchpadDmaEventToken = token;
}

void PS2Runtime::cancelScratchpadDmaEvent(
    DmacChannel channel) noexcept
{
    ps2x::timing::EeEventToken *token = nullptr;
    if (channel == DmacChannel::FromScratchpad)
        token = &m_fromScratchpadDmaEventToken;
    else if (channel == DmacChannel::ToScratchpad)
        token = &m_toScratchpadDmaEventToken;
    if (!token)
        return;

    if (token->generation != 0u)
        (void)m_eeEventScheduler.cancel(*token);
    *token = {
        scratchpadDmaEventSource(channel), 0u};
}

void PS2Runtime::serviceScratchpadDmaAtEvent(
    DmacChannel channel,
    const ps2x::timing::EeEventService &service)
{
    ps2x::timing::EeEventToken *token =
        channel == DmacChannel::FromScratchpad
            ? &m_fromScratchpadDmaEventToken
            : &m_toScratchpadDmaEventToken;
    if (service.source !=
            scratchpadDmaEventSource(channel) ||
        token->generation == 0u ||
        service.generation != token->generation)
    {
        return;
    }

    *token = {
        scratchpadDmaEventSource(channel), 0u};
    const ScratchpadDmaAdvanceResult advance =
        m_memory.advanceScratchpadDma(channel);
    if (!advance.active || advance.completed ||
        advance.phase == ScratchpadDmaPhase::Fault)
    {
        return;
    }

    scheduleScratchpadDmaEvent(
        channel,
        ps2x::timing::saturatingAdd(
            service.serviceTick,
            ps2x::timing::eeCyclesToTicks(
                advance.delayEeCycles)));
}

PS2Runtime::Vu1SynchronousCommandBarrier
PS2Runtime::beginVu1SynchronousCommandBarrier(
    Vu1CommandType commandType)
{
    Vu1SynchronousCommandBarrier barrier{};
    if (m_vu1SynchronousCommandBatchActive &&
        (m_batchedVu1SynchronousCommandBarrier.resolution !=
             Vu1SpeculationResolution::None ||
         m_batchedVu1SynchronousCommandBarrier.requeue ||
         m_batchedVu1SynchronousCommandBarrier.
             coarseInvocationCompleted))
    {
        return barrier;
    }
    const auto prepareForBatch =
        [this](Vu1SynchronousCommandBarrier prepared)
        {
            if (!m_vu1SynchronousCommandBatchActive ||
                (prepared.resolution ==
                     Vu1SpeculationResolution::None &&
                 !prepared.requeue &&
                 !prepared.coarseInvocationCompleted))
            {
                return prepared;
            }
            if (m_batchedVu1SynchronousCommandBarrier.resolution !=
                    Vu1SpeculationResolution::None ||
                m_batchedVu1SynchronousCommandBarrier.requeue ||
                m_batchedVu1SynchronousCommandBarrier.
                    coarseInvocationCompleted)
            {
                throw std::logic_error(
                    "VU1 synchronous command batch already owns a hazard");
            }
            m_batchedVu1SynchronousCommandBarrier = prepared;
            prepared.requeue.reset();
            prepared.coarseInvocationCompleted = false;
            return prepared;
        };
    if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        m_threadedVu1Executor)
    {
        if (m_pendingVu1CoarseInvocation)
        {
            recordVu1CoarseForcedBarrier(commandType);
            (void)completeVu1CoarseInvocation(
                currentEeTick(), false);
            barrier.coarseInvocationCompleted = true;
        }
        return prepareForBatch(std::move(barrier));
    }
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedAsync ||
        !m_threadedVu1Executor)
    {
        return prepareForBatch(std::move(barrier));
    }

    if (m_vu1CheckpointEpochs)
    {
        if (!m_pendingVu1ExecutionEpoch)
            return prepareForBatch(std::move(barrier));
        Vu1PendingExecutionEpoch pending =
            std::move(*m_pendingVu1ExecutionEpoch);
        m_pendingVu1ExecutionEpoch.reset();
        m_vu1AsyncPendingSlice.store(
            false, std::memory_order_release);
        m_threadedVu1Executor->stopExecutionEpoch(
            pending.epoch);
        barrier.requeue = pending.plan;
        m_vu1AsyncHazardBarrierCount.fetch_add(
            1u, std::memory_order_relaxed);
        return prepareForBatch(std::move(barrier));
    }

    if (m_vu1SpeculationPublicationState ==
        Vu1SpeculationPublicationState::None)
    {
        return prepareForBatch(std::move(barrier));
    }
    if (m_vu1SpeculationPublicationState ==
        Vu1SpeculationPublicationState::Published)
    {
        barrier.resolution =
            Vu1SpeculationResolution::Commit;
        return prepareForBatch(std::move(barrier));
    }
    if (!m_pendingVu1Slice)
    {
        throw std::logic_error(
            "unpublished VU1 speculation has no pending slice");
    }

    Vu1PendingSlice pending =
        std::move(*m_pendingVu1Slice);
    m_pendingVu1Slice.reset();
    m_vu1AsyncPendingSlice.store(
        false, std::memory_order_release);
    barrier.resolution =
        Vu1SpeculationResolution::Rollback;
    barrier.requeue = pending.plan;

    const auto waitAt = std::chrono::steady_clock::now();
    Vu1CommandResult discarded{};
    try
    {
        discarded = m_threadedVu1Executor->wait(
            std::move(pending.submission));
    }
    catch (...)
    {
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
        throw;
    }
    const uint64_t waitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitAt)
                .count());
    m_vu1AsyncHazardBarrierCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1AsyncHazardWaitNanoseconds.fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    if (discarded.digest.type !=
            Vu1CommandType::AdvanceSlice ||
        !std::holds_alternative<Vu1SliceResult>(
            discarded.payload))
    {
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
        throw std::logic_error(
            "VU1 hazard discarded a non-slice result");
    }
    return prepareForBatch(std::move(barrier));
}

void PS2Runtime::recordVu1CoarseForcedBarrier(
    Vu1CommandType commandType)
{
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedCoarse ||
        !m_pendingVu1CoarseInvocation)
    {
        throw std::logic_error(
            "coarse VU1 barrier attribution has no pending invocation");
    }
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        DebugVu1TimingEntry entry =
            debugVu1TimingEntry(
                DebugVu1TimingEventKind::ForcedBarrier,
                m_boundEeContext);
        entry.observedValue = true;
        entry.ownerReady =
            m_pendingVu1CoarseInvocation->
                completedResult.has_value() ||
            m_pendingVu1CoarseInvocation->submission.ready();
        entry.hasBarrierCommandType = true;
        entry.barrierCommandType = commandType;
        debugRecordVu1Timing(std::move(entry));
    }
#endif
    m_vu1CoarseForcedBarrierCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1CoarseForcedBarriersByCommandType[
        vu1CommandTypeIndex(commandType)]
        .fetch_add(1u, std::memory_order_relaxed);
}

void PS2Runtime::finishVu1SynchronousCommandBarrier(
    const Vu1SynchronousCommandBarrier &barrier)
{
    if (m_vu1SynchronousCommandBatchActive)
    {
        if (barrier.resolution !=
            Vu1SpeculationResolution::None)
        {
            m_vu1SpeculationPublicationState =
                Vu1SpeculationPublicationState::None;
        }
        return;
    }
    if (barrier.resolution !=
        Vu1SpeculationResolution::None)
    {
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
    }
    if (barrier.coarseInvocationCompleted)
    {
        (void)resumeVif1AfterVuWithTimingTrace();
        if (!m_publishedVu1Active.load(
                std::memory_order_acquire) &&
            !m_memory.vif1DmaSnapshot().active)
        {
            m_memory.finishVif1DmaTrace();
        }
    }
    if (!barrier.requeue ||
        !usesExactVu1Scheduling(m_vu1ExecutionMode))
    {
        return;
    }

    requeueOrDeferVu1SpeculativeSlice(
        *barrier.requeue);
}

void PS2Runtime::beginVu1SynchronousCommandBatch()
{
    if (!usesBufferedVu1OwnerStream(m_vu1ExecutionMode))
    {
        return;
    }
    if (m_vu1SynchronousCommandBatchActive ||
        m_batchedVu1SynchronousCommandBarrier.resolution !=
            Vu1SpeculationResolution::None ||
        m_batchedVu1SynchronousCommandBarrier.requeue ||
        m_batchedVu1SynchronousCommandBarrier.
            coarseInvocationCompleted ||
        m_batchedVu1VifState ||
        !m_batchedVu1DecodedUnpacks.empty() ||
        m_batchedVu1DecodedUnpackFinalVifState ||
        m_batchedVu1DecodedUnpackPayloadBytes != 0u)
    {
        throw std::logic_error(
            "nested VU1 synchronous command batch");
    }
    m_vu1SynchronousCommandBatchActive = true;
}

void PS2Runtime::finishVu1SynchronousCommandBatch(bool requeue)
{
    if (!usesBufferedVu1OwnerStream(m_vu1ExecutionMode))
    {
        return;
    }
    if (!m_vu1SynchronousCommandBatchActive)
    {
        throw std::logic_error(
            "VU1 synchronous command batch is not active");
    }

    if (requeue &&
        (!m_batchedVu1DecodedUnpacks.empty() ||
         m_batchedVu1VifState))
    {
        try
        {
            flushBatchedVu1DecodedUnpacks();
            flushBatchedVu1VifStateUpdate();
        }
        catch (...)
        {
            m_batchedVu1VifState.reset();
            m_batchedVu1DecodedUnpacks.clear();
            m_batchedVu1DecodedUnpackFinalVifState.reset();
            m_batchedVu1DecodedUnpackPayloadBytes = 0u;
            m_vu1SynchronousCommandBatchActive = false;
            Vu1SynchronousCommandBarrier barrier =
                std::move(m_batchedVu1SynchronousCommandBarrier);
            m_batchedVu1SynchronousCommandBarrier = {};
            if (barrier.resolution !=
                Vu1SpeculationResolution::None)
            {
                m_vu1SpeculationPublicationState =
                    Vu1SpeculationPublicationState::None;
            }
            throw;
        }
    }

    m_batchedVu1VifState.reset();
    m_batchedVu1DecodedUnpacks.clear();
    m_batchedVu1DecodedUnpackFinalVifState.reset();
    m_batchedVu1DecodedUnpackPayloadBytes = 0u;
    m_vu1SynchronousCommandBatchActive = false;
    Vu1SynchronousCommandBarrier barrier =
        std::move(m_batchedVu1SynchronousCommandBarrier);
    m_batchedVu1SynchronousCommandBarrier = {};
    if (!requeue)
    {
        if (barrier.resolution !=
            Vu1SpeculationResolution::None)
        {
            m_vu1SpeculationPublicationState =
                Vu1SpeculationPublicationState::None;
        }
        return;
    }
    finishVu1SynchronousCommandBarrier(barrier);
}

std::shared_ptr<const Vu1DecodedUnpackBatch>
PS2Runtime::takeBatchedVu1DecodedUnpacks()
{
    if (m_batchedVu1DecodedUnpacks.empty())
        return {};
    if (!m_vu1SynchronousCommandBatchActive ||
        !usesBufferedVu1OwnerStream(m_vu1ExecutionMode))
    {
        throw std::logic_error(
            "buffered VU1 UNPACKs require an asynchronous command batch");
    }

    auto batch =
        std::make_shared<const Vu1DecodedUnpackBatch>(
            Vu1DecodedUnpackBatch{
                std::move(m_batchedVu1DecodedUnpacks)});
    m_batchedVu1DecodedUnpacks.clear();
    m_batchedVu1DecodedUnpackFinalVifState.reset();
    m_batchedVu1DecodedUnpackPayloadBytes = 0u;
    return batch;
}

void PS2Runtime::flushBatchedVu1DecodedUnpacks()
{
    if (m_batchedVu1DecodedUnpacks.empty())
        return;

    m_batchedVu1DecodedUnpackFinalVifState.reset();
    const std::shared_ptr<const Vu1DecodedUnpackBatch> batch =
        takeBatchedVu1DecodedUnpacks();
    const Vu1SynchronousCommandBarrier barrier =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? beginVu1SynchronousCommandBarrier(
                  Vu1CommandType::DecodedUnpack)
            : Vu1SynchronousCommandBarrier{};
    (void)m_threadedVu1Executor
        ->submitResultlessResolvingSpeculation(
            Vu1CommandPayload{
                Vu1DecodedUnpackBatchCommand{batch}},
            barrier.resolution,
            currentEeTick().raw(),
            m_vu1ExecutionTiming.generation);
    // Admission is sufficient because this path contains only row-stable
    // UNPACKs. A later observation command drains the ordered prefix; mode-2
    // ROW feedback never reaches this resultless path.
    m_vu1MemoryMirrorDirty = true;
    finishVu1SynchronousCommandBarrier(barrier);
}

void PS2Runtime::flushBatchedVu1VifStateUpdate()
{
    if (!m_batchedVu1VifState)
        return;
    if (!m_vu1SynchronousCommandBatchActive ||
        !usesBufferedVu1OwnerStream(m_vu1ExecutionMode))
    {
        throw std::logic_error(
            "buffered VU1 VIF state requires an asynchronous command batch");
    }

    Vu1VifStateUpdateCommand command{
        *m_batchedVu1VifState};
    m_batchedVu1VifState.reset();
    const bool preservesPendingCoarseInvocation =
        m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        m_pendingVu1CoarseInvocation.has_value();
    const Vu1SynchronousCommandBarrier barrier =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? beginVu1SynchronousCommandBarrier(
                  Vu1CommandType::VifStateUpdate)
            : Vu1SynchronousCommandBarrier{};
    (void)m_threadedVu1Executor
        ->submitResultlessResolvingSpeculation(
            Vu1CommandPayload{command},
            barrier.resolution,
            currentEeTick().raw(),
            m_vu1ExecutionTiming.generation);
    // The command's typed result is exactly the immutable input state, so EE
    // can retain that parser-owned value without waiting for owner execution.
    m_knownVu1VifState = command.state;
    if (preservesPendingCoarseInvocation)
    {
        m_vu1CoarseNonPublishingVifStateUpdateCount.fetch_add(
            1u, std::memory_order_relaxed);
    }
    finishVu1SynchronousCommandBarrier(barrier);
}

bool PS2Runtime::isCurrentVu1PendingSlicePlan(
    const Vu1PendingSlicePlan &plan) const noexcept
{
    return plan.cycleBudget != 0u &&
           plan.eventGeneration != 0u &&
           plan.executionGeneration ==
               m_vu1ExecutionTiming.generation &&
           plan.eventGeneration ==
               m_vu1ExecutionTiming.eventToken.generation &&
           m_publishedVu1Active.load(
               std::memory_order_acquire);
}

bool PS2Runtime::hasPrecedingVif1Event(
    const Vu1PendingSlicePlan &plan) const noexcept
{
    const auto vif1Event = m_eeEventScheduler.event(
        ps2x::timing::EeEventSource::DmacVif1);
    return vif1Event.has_value() &&
           vif1Event->deadline <= plan.deadline;
}

void PS2Runtime::clearDeferredVu1SpeculativeSlice() noexcept
{
    m_deferredVu1SlicePlan.reset();
    m_vu1AsyncDeferredSlice.store(
        false, std::memory_order_release);
}

void PS2Runtime::requeueOrDeferVu1SpeculativeSlice(
    const Vu1PendingSlicePlan &plan)
{
    if (!isCurrentVu1PendingSlicePlan(plan))
    {
        return;
    }
    if (m_pendingVu1Slice ||
        m_pendingVu1ExecutionEpoch ||
        m_deferredVu1SlicePlan)
    {
        throw std::logic_error(
            "VU1 hazard requeue conflicts with retained speculative work");
    }

    if (hasPrecedingVif1Event(plan))
    {
        m_deferredVu1SlicePlan = plan;
        m_vu1AsyncDeferredSlice.store(
            true, std::memory_order_release);
        m_vu1AsyncDeferredSliceCount.fetch_add(
            1u, std::memory_order_relaxed);
        return;
    }

    enqueueVu1SpeculativeSlice(
        plan, Vu1SpeculationResolution::None);
    m_vu1AsyncHazardRequeueCount.fetch_add(
        1u, std::memory_order_relaxed);
}

void PS2Runtime::armDeferredVu1SpeculativeSlice(
    uint64_t requiredEventGeneration)
{
    if (!m_deferredVu1SlicePlan)
    {
        return;
    }

    const Vu1PendingSlicePlan plan =
        *m_deferredVu1SlicePlan;
    if (!isCurrentVu1PendingSlicePlan(plan))
    {
        clearDeferredVu1SpeculativeSlice();
        return;
    }
    if (requiredEventGeneration != 0u &&
        plan.eventGeneration != requiredEventGeneration)
    {
        return;
    }
    if (requiredEventGeneration == 0u &&
        hasPrecedingVif1Event(plan))
    {
        return;
    }
    if (m_pendingVu1Slice ||
        m_pendingVu1ExecutionEpoch ||
        m_vu1SpeculationPublicationState !=
            Vu1SpeculationPublicationState::None)
    {
        throw std::logic_error(
            "deferred VU1 slice conflicts with active speculation");
    }

    clearDeferredVu1SpeculativeSlice();
    enqueueVu1SpeculativeSlice(
        plan, Vu1SpeculationResolution::None);
    m_vu1AsyncHazardRequeueCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1AsyncDeferredSliceArmCount.fetch_add(
        1u, std::memory_order_relaxed);
    if (requiredEventGeneration != 0u)
    {
        m_vu1AsyncForcedDeferredSliceArmCount.fetch_add(
            1u, std::memory_order_relaxed);
    }
}

void PS2Runtime::enqueueVu1SpeculativeSlice(
    const Vu1PendingSlicePlan &plan,
    Vu1SpeculationResolution previousResolution)
{
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedAsync ||
        !m_threadedVu1Executor)
    {
        throw std::logic_error(
            "VU1 speculative enqueue requires asynchronous ownership");
    }
    if (m_deferredVu1SlicePlan)
    {
        if (isCurrentVu1PendingSlicePlan(
                *m_deferredVu1SlicePlan))
        {
            throw std::logic_error(
                "new VU1 speculation conflicts with a current deferred slice");
        }
        clearDeferredVu1SpeculativeSlice();
    }
    if (m_pendingVu1Slice ||
        m_pendingVu1ExecutionEpoch ||
        plan.cycleBudget == 0u ||
        plan.eventGeneration == 0u ||
        plan.executionGeneration !=
            m_vu1ExecutionTiming.generation)
    {
        throw std::logic_error(
            "invalid VU1 speculative slice plan");
    }
    if ((previousResolution ==
             Vu1SpeculationResolution::None &&
         m_vu1SpeculationPublicationState !=
             Vu1SpeculationPublicationState::None) ||
        (previousResolution ==
             Vu1SpeculationResolution::Commit &&
         m_vu1SpeculationPublicationState !=
             Vu1SpeculationPublicationState::Published))
    {
        throw std::logic_error(
            "VU1 speculative enqueue resolution does not match publication state");
    }

    const bool startsNewBatchSpeculationEpoch =
        m_vu1SynchronousCommandBatchActive &&
        (m_batchedVu1SynchronousCommandBarrier.resolution !=
             Vu1SpeculationResolution::None ||
         m_batchedVu1SynchronousCommandBarrier.requeue);
    if (startsNewBatchSpeculationEpoch &&
        m_batchedVu1SynchronousCommandBarrier.requeue)
    {
        const Vu1PendingSlicePlan &deferred =
            *m_batchedVu1SynchronousCommandBarrier.requeue;
        if (m_vu1ExecutionTiming.generation ==
                deferred.executionGeneration &&
            m_vu1ExecutionTiming.eventToken.generation ==
                deferred.eventGeneration &&
            m_publishedVu1Active.load(
                std::memory_order_acquire))
        {
            throw std::logic_error(
                "new VU1 speculation cannot replace a current batch requeue");
        }
    }

    const auto submitAt = std::chrono::steady_clock::now();
    try
    {
        if (m_vu1CheckpointEpochs)
        {
            if (previousResolution !=
                Vu1SpeculationResolution::None)
            {
                throw std::logic_error(
                    "checkpoint epochs do not use slice resolutions");
            }
            Vu1ExecutionEpoch epoch =
                m_threadedVu1Executor->submitExecutionEpoch(
                    Vu1ExecutionEpochOptions{
                        .initialCycles = plan.cycleBudget,
                        .followupCycles =
                            kVu1FollowupEventCycles,
                        .checkpointCapacity =
                            m_vu1CheckpointCapacity,
                    },
                    plan.deadline.raw(),
                    plan.eventGeneration);
            m_pendingVu1ExecutionEpoch.emplace(
                Vu1PendingExecutionEpoch{
                    .epoch = std::move(epoch),
                    .plan = plan,
                });
            m_vu1SpeculationPublicationState =
                Vu1SpeculationPublicationState::None;
        }
        else
        {
            Vu1CommandSubmission submission =
                m_threadedVu1Executor->submitSpeculativeAdvance(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = plan.cycleBudget,
                    },
                    previousResolution,
                    plan.deadline.raw(),
                    plan.eventGeneration);
            m_pendingVu1Slice.emplace(
                Vu1PendingSlice{
                    .submission = std::move(submission),
                    .plan = plan,
                });
            m_vu1SpeculationPublicationState =
                Vu1SpeculationPublicationState::Unpublished;
        }
        m_vu1AsyncPendingSlice.store(
            true, std::memory_order_release);
        if (startsNewBatchSpeculationEpoch)
        {
            // MSCAL/MSCNT can start a fresh VU generation while one VIF1
            // parser invocation is still active. The prior resolution has
            // already reached the owner, and its deferred plan is stale.
            // Subsequent commands in this parser call must therefore
            // resolve the new private startup slice as a distinct epoch.
            m_batchedVu1SynchronousCommandBarrier = {};
        }
    }
    catch (...)
    {
        if (previousResolution !=
            Vu1SpeculationResolution::None)
        {
            m_vu1SpeculationPublicationState =
                Vu1SpeculationPublicationState::None;
        }
        throw;
    }

    const uint64_t submitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - submitAt)
                .count());
    m_vu1AsyncSlicesSubmitted.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1AsyncSliceSubmitCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1AsyncSliceSubmitNanoseconds.fetch_add(
        submitNanoseconds, std::memory_order_relaxed);
    updateAtomicMaximum(
        m_vu1AsyncMaximumSliceSubmitNanoseconds,
        submitNanoseconds);
}

Vu1CommandResult
PS2Runtime::consumeVu1SpeculativeSliceAtEvent(
    const ps2x::timing::EeEventService &service,
    uint32_t cycleBudget)
{
    if (m_vu1SpeculationPublicationState !=
            Vu1SpeculationPublicationState::Unpublished ||
        !m_pendingVu1Slice)
    {
        throw std::logic_error(
            "VU1 event has no speculative slice");
    }
    if (m_pendingVu1Slice->plan.eventGeneration !=
            service.generation ||
        m_pendingVu1Slice->plan.executionGeneration !=
            m_vu1ExecutionTiming.generation ||
        m_pendingVu1Slice->plan.deadline !=
            service.scheduledTick)
    {
        throw std::logic_error(
            "VU1 speculative result does not match its event");
    }

    const uint32_t speculativeCycleBudget =
        m_pendingVu1Slice->plan.cycleBudget;
    Vu1CommandSubmission submission =
        std::move(m_pendingVu1Slice->submission);
    const Vu1WorkIdentity speculativeIdentity =
        submission.identity();
    m_pendingVu1Slice.reset();
    m_vu1AsyncPendingSlice.store(
        false, std::memory_order_release);
    const bool ready = submission.ready();
    const auto waitAt = std::chrono::steady_clock::now();
    Vu1CommandResult speculativeResult =
        m_threadedVu1Executor->wait(
            std::move(submission));
    // A synchronous hazard can requeue an overdue plan after a command at a
    // later guest tick. Command identities remain monotonic in that case;
    // the plan, not the identity timestamp, owns the publication deadline.
    if (speculativeResult.identity != speculativeIdentity ||
        speculativeResult.digest.type !=
            Vu1CommandType::AdvanceSlice ||
        !std::holds_alternative<Vu1SliceResult>(
            speculativeResult.payload))
    {
        throw std::logic_error(
            "VU1 speculative event returned an invalid result");
    }

    Vu1CommandResult result{};
    bool usedBudgetFallback = false;
    bool usedBudgetExtension = false;
    uint64_t budgetExtensionWaitNanoseconds = 0u;
    if (speculativeCycleBudget == cycleBudget)
    {
        result = std::move(speculativeResult);
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::Published;
    }
    else if (cycleBudget > speculativeCycleBudget)
    {
        Vu1SliceResult prefix = std::get<Vu1SliceResult>(
            std::move(speculativeResult.payload));
        if (prefix.run.executedCycles > cycleBudget)
        {
            throw std::logic_error(
                "VU1 speculative prefix exceeds its late event budget");
        }
        const bool exhaustedPrefixBudget =
            speculativeResult.active &&
            prefix.run.reason == VuExitReason::CycleBudget;
        const uint32_t extensionCycles =
            exhaustedPrefixBudget
                ? cycleBudget - prefix.run.executedCycles
                : 0u;

        const auto extensionAt =
            std::chrono::steady_clock::now();
        Vu1CommandResult extensionResult =
            m_threadedVu1Executor
                ->commitExtendedSpeculativeAdvance(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = cycleBudget,
                    },
                    extensionCycles,
                    service.serviceTick.raw(),
                    service.generation);
        budgetExtensionWaitNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() -
                    extensionAt)
                    .count());
        if (extensionResult.identity.sequence !=
                speculativeIdentity.sequence ||
            extensionResult.identity.generation !=
                speculativeIdentity.generation ||
            extensionResult.identity.publicationToken !=
                service.generation ||
            extensionResult.identity.guestTick !=
                service.serviceTick.raw() ||
            extensionResult.digest.type !=
                Vu1CommandType::AdvanceSlice ||
            !std::holds_alternative<Vu1SliceResult>(
                extensionResult.payload))
        {
            throw std::logic_error(
                "VU1 speculative extension returned an invalid result");
        }

        Vu1SliceResult extension =
            std::get<Vu1SliceResult>(
                std::move(extensionResult.payload));
        if (extensionCycles != 0u)
        {
            if (!speculativeResult.diagnostics.empty())
            {
                const auto *const boundary = std::get_if<
                    Vu1TraceInvocationEndDiagnostic>(
                    &speculativeResult.diagnostics.back());
                if (boundary && boundary->hitCycleLimit)
                {
                    speculativeResult.diagnostics.pop_back();
                }
            }
            prefix.path1Packets.reserve(
                prefix.path1Packets.size() +
                extension.path1Packets.size());
            prefix.path1Packets.insert(
                prefix.path1Packets.end(),
                std::make_move_iterator(
                    extension.path1Packets.begin()),
                std::make_move_iterator(
                    extension.path1Packets.end()));
            prefix.run.executedCycles +=
                extension.run.executedCycles;
            prefix.run.reason = extension.run.reason;
            prefix.run.activeAfter =
                extension.run.activeAfter;
            prefix.run.completed =
                extension.run.completed;
            prefix.state = std::move(extension.state);
            prefix.architecturalStateHash =
                extension.architecturalStateHash;
            if (prefix.diagnosticStateHashes &&
                extension.diagnosticStateHashes)
            {
                extension.diagnosticStateHashes->inputAggregate =
                    prefix.diagnosticStateHashes->inputAggregate;
                extension.diagnosticStateHashes->inputExecution =
                    prefix.diagnosticStateHashes->inputExecution;
                extension.diagnosticStateHashes->inputMicroMemory =
                    prefix.diagnosticStateHashes->inputMicroMemory;
                extension.diagnosticStateHashes->inputDataMemory =
                    prefix.diagnosticStateHashes->inputDataMemory;
                extension.diagnosticStateHashes->inputVif =
                    prefix.diagnosticStateHashes->inputVif;
            }
            prefix.diagnosticStateHashes =
                std::move(extension.diagnosticStateHashes);
            prefix.vifCanResume =
                extension.vifCanResume;
            prefix.fault = std::move(extension.fault);
        }
        prefix.run.requestedCycles = cycleBudget;

        speculativeResult.diagnostics.reserve(
            speculativeResult.diagnostics.size() +
            extensionResult.diagnostics.size());
        speculativeResult.diagnostics.insert(
            speculativeResult.diagnostics.end(),
            std::make_move_iterator(
                extensionResult.diagnostics.begin()),
            std::make_move_iterator(
                extensionResult.diagnostics.end()));
        extensionResult.diagnostics =
            std::move(speculativeResult.diagnostics);
        extensionResult.payload = std::move(prefix);
        result = std::move(extensionResult);
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
        usedBudgetExtension = true;
        m_vu1AsyncBudgetExtensionCount.fetch_add(
            1u, std::memory_order_relaxed);
        m_vu1AsyncBudgetExtensionCycleDelta.fetch_add(
            static_cast<uint64_t>(
                cycleBudget - speculativeCycleBudget),
            std::memory_order_relaxed);
        m_vu1AsyncBudgetExtensionExecutedCycles.fetch_add(
            extension.run.executedCycles,
            std::memory_order_relaxed);
        m_vu1AsyncBudgetExtensionWaitNanoseconds.fetch_add(
            budgetExtensionWaitNanoseconds,
            std::memory_order_relaxed);
    }
    else
    {
        result = m_threadedVu1Executor->submitResolvingSpeculation(
            Vu1AdvanceSliceCommand{
                .maximumCycles = cycleBudget,
            },
            Vu1SpeculationResolution::Rollback,
            service.serviceTick.raw(),
            service.generation);
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
        const uint64_t cycleDelta =
            cycleBudget >= speculativeCycleBudget
                ? static_cast<uint64_t>(
                      cycleBudget - speculativeCycleBudget)
                : static_cast<uint64_t>(
                      speculativeCycleBudget - cycleBudget);
        m_vu1AsyncBudgetFallbackCount.fetch_add(
            1u, std::memory_order_relaxed);
        m_vu1AsyncBudgetFallbackCycleDelta.fetch_add(
            cycleDelta, std::memory_order_relaxed);
        usedBudgetFallback = true;
    }

    const uint64_t expectedResultGuestTick =
        !usedBudgetFallback && !usedBudgetExtension
            ? speculativeIdentity.guestTick
            : service.serviceTick.raw();
    if (result.identity.publicationToken !=
            service.generation ||
        result.identity.guestTick != expectedResultGuestTick ||
        result.digest.type !=
            Vu1CommandType::AdvanceSlice ||
        !std::holds_alternative<Vu1SliceResult>(
            result.payload))
    {
        throw std::logic_error(
            "VU1 event returned an invalid published result");
    }

    const uint64_t waitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitAt)
                .count());
    m_vu1AsyncEventWaitCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1AsyncEventWaitNanoseconds.fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    updateAtomicMaximum(
        m_vu1AsyncMaximumEventWaitNanoseconds,
        waitNanoseconds);
    if (usedBudgetFallback)
    {
        m_vu1AsyncBudgetFallbackWaitNanoseconds.fetch_add(
            waitNanoseconds, std::memory_order_relaxed);
    }
    if (ready)
    {
        m_vu1AsyncResultsReadyAtEvent.fetch_add(
            1u, std::memory_order_relaxed);
    }
    else
    {
        m_vu1AsyncResultsLateAtEvent.fetch_add(
            1u, std::memory_order_relaxed);
    }
    return result;
}

Vu1CommandResult
PS2Runtime::consumeVu1ExecutionEpochAtEvent(
    const ps2x::timing::EeEventService &service,
    uint32_t cycleBudget)
{
    if (!m_vu1CheckpointEpochs ||
        !m_pendingVu1ExecutionEpoch ||
        !m_threadedVu1Executor)
    {
        throw std::logic_error(
            "VU1 event has no checkpoint epoch");
    }
    const Vu1PendingSlicePlan &plan =
        m_pendingVu1ExecutionEpoch->plan;
    if (plan.eventGeneration != service.generation ||
        plan.executionGeneration !=
            m_vu1ExecutionTiming.generation ||
        plan.deadline != service.scheduledTick)
    {
        throw std::logic_error(
            "VU1 checkpoint epoch does not match its event");
    }

    const bool ready =
        m_threadedVu1Executor
            ->executionEpochCheckpointReady(
                m_pendingVu1ExecutionEpoch->epoch,
                cycleBudget);
    const auto waitAt = std::chrono::steady_clock::now();
    Vu1CommandResult result =
        m_threadedVu1Executor
            ->waitExecutionEpochCheckpoint(
                m_pendingVu1ExecutionEpoch->epoch,
                cycleBudget,
                service.serviceTick.raw(),
                service.generation);
    const Vu1WorkIdentity epochIdentity =
        m_pendingVu1ExecutionEpoch->epoch.identity();
    if (result.identity.sequence != epochIdentity.sequence ||
        result.identity.generation != epochIdentity.generation ||
        result.identity.publicationToken !=
            service.generation ||
        result.identity.guestTick !=
            service.serviceTick.raw() ||
        result.digest.type !=
            Vu1CommandType::AdvanceSlice ||
        !std::holds_alternative<Vu1SliceResult>(
            result.payload))
    {
        throw std::logic_error(
            "VU1 checkpoint epoch returned an invalid result");
    }

    const uint64_t waitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitAt)
                .count());
    m_vu1AsyncEventWaitCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1AsyncEventWaitNanoseconds.fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    updateAtomicMaximum(
        m_vu1AsyncMaximumEventWaitNanoseconds,
        waitNanoseconds);
    if (ready)
    {
        m_vu1AsyncResultsReadyAtEvent.fetch_add(
            1u, std::memory_order_relaxed);
    }
    else
    {
        m_vu1AsyncResultsLateAtEvent.fetch_add(
            1u, std::memory_order_relaxed);
    }
    return result;
}

PS2Runtime::Vu1CoarseEstimatorIdentity
PS2Runtime::vu1CoarseEstimatorIdentity(
    const Vu1InvocationCommand &command) const
{
    return Vu1CoarseEstimatorIdentity{
        .kind = command.kind,
        .startPc = command.kind == Vu1InvocationKind::Mscnt
            ? m_publishedVu1ProgramCounter.load(
                  std::memory_order_acquire)
            : command.startPc,
        .top = command.top,
        .itop = command.itop,
        .codeGeneration = m_memory.getVU1CodeGeneration(),
    };
}

uint32_t PS2Runtime::estimateVu1CoarseCycles(
    const Vu1CoarseEstimatorIdentity &identity)
{
    if (m_vu1ExecutionMode !=
        Vu1ExecutionMode::ThreadedCoarse)
    {
        throw std::logic_error(
            "coarse VU1 estimate requested in an exact mode");
    }

    uint64_t inputHash = 0u;
    inputHash = appendVu1CoarseDigest(
        inputHash, static_cast<uint64_t>(identity.kind));
    inputHash = appendVu1CoarseDigest(
        inputHash, identity.startPc);
    inputHash = appendVu1CoarseDigest(inputHash, identity.top);
    inputHash = appendVu1CoarseDigest(inputHash, identity.itop);
    inputHash = appendVu1CoarseDigest(
        inputHash, identity.codeGeneration);

    const auto matchingEntry = std::find_if(
        m_vu1CoarseEstimatorEntries.begin(),
        m_vu1CoarseEstimatorEntries.end(),
        [&identity](const Vu1CoarseEstimatorEntry &entry)
        {
            return entry.identity == identity;
        });
    const bool identityHit =
        matchingEntry != m_vu1CoarseEstimatorEntries.end();
    (identityHit
         ? m_vu1CoarseEstimatorIdentityHitCount
         : m_vu1CoarseEstimatorIdentityMissCount)
        .fetch_add(1u, std::memory_order_relaxed);
    inputHash = appendVu1CoarseDigest(
        inputHash, identityHit ? 1u : 0u);

    const std::deque<uint32_t> &history = identityHit
        ? matchingEntry->cycleHistory
        : m_vu1CoarseCycleHistory;
    inputHash = appendVu1CoarseDigest(
        inputHash, history.size());

    uint64_t cycleSum = 0u;
    for (uint32_t cycles : history)
    {
        cycleSum += cycles;
        inputHash = appendVu1CoarseDigest(inputHash, cycles);
    }
    uint32_t estimate = m_vu1CoarseInitialEstimateCycles;
    if (!history.empty())
    {
        estimate = static_cast<uint32_t>(
            cycleSum / history.size());
    }
    estimate = std::clamp(
        estimate,
        m_vu1CoarseMinimumEstimateCycles,
        m_vu1CoarseMaximumLeadCycles);

    uint64_t inputStream =
        m_vu1CoarseEstimatorInputDigest.load(
            std::memory_order_relaxed);
    inputStream = appendVu1CoarseDigest(
        inputStream, inputHash);
    m_vu1CoarseEstimatorInputDigest.store(
        inputStream, std::memory_order_release);

    uint64_t outputStream =
        m_vu1CoarseEstimatorOutputDigest.load(
            std::memory_order_relaxed);
    outputStream = appendVu1CoarseDigest(
        outputStream, inputHash);
    outputStream = appendVu1CoarseDigest(
        outputStream, estimate);
    m_vu1CoarseEstimatorOutputDigest.store(
        outputStream, std::memory_order_release);
    return estimate;
}

void PS2Runtime::submitVu1CoarseInvocation(
    Vu1InvocationCommand command,
    ps2x::timing::EeTick startTick)
{
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedCoarse ||
        !m_threadedVu1Executor)
    {
        throw std::logic_error(
            "coarse VU1 submission requires coarse ownership");
    }
    if (m_pendingVu1CoarseInvocation)
    {
        throw std::logic_error(
            "coarse VU1 submission exceeded one invocation of lead");
    }

    refreshVu1DiagnosticsConfiguration(
        startTick.raw(),
        m_vu1ExecutionTiming.generation);
    if (m_vu1SynchronousCommandBatchActive &&
        !m_batchedVu1DecodedUnpacks.empty())
    {
        command.unpacksBefore =
            takeBatchedVu1DecodedUnpacks();
    }
    if (m_vu1SynchronousCommandBatchActive &&
        m_batchedVu1VifState)
    {
        command.vifStateBefore =
            std::make_shared<const Vu1VifState>(
                *m_batchedVu1VifState);
        m_batchedVu1VifState.reset();
    }
    command.maximumCycles =
        m_vu1CoarseMaximumInvocationCycles;
    command.maximumPath1Bytes =
        m_vu1CoarseMaximumPath1Bytes;
    command.captureState = false;

    const Vu1CoarseEstimatorIdentity estimatorIdentity =
        vu1CoarseEstimatorIdentity(command);
    const uint32_t estimate =
        estimateVu1CoarseCycles(estimatorIdentity);
    const uint64_t inputHash =
        m_vu1CoarseEstimatorInputDigest.load(
            std::memory_order_acquire);
    const Vu1InvocationKind invocationKind = command.kind;
    const uint32_t invocationStartPc = command.startPc;
    const GifArbiter::Path1ReservationToken path1Reservation =
        m_gifArbiter.reservePath1();
    const ps2x::timing::EeTick deadline =
        ps2x::timing::saturatingAdd(
            startTick,
            ps2x::timing::vuCyclesToEeTicks(estimate));
    try
    {
        Vu1CommandSubmission submission =
            m_threadedVu1Executor->submitAsync(
                Vu1CommandPayload{std::move(command)},
                startTick.raw(),
                m_vu1ExecutionTiming.generation);
        scheduleVU1Event(deadline);
        if (m_vu1ExecutionTiming.eventToken.generation == 0u)
        {
            throw std::runtime_error(
                "coarse VU1 completion event was not scheduled");
        }

        m_pendingVu1CoarseInvocation =
            Vu1PendingCoarseInvocation{
                .submission = std::move(submission),
                .estimatorIdentity = estimatorIdentity,
                .estimatedCycles = estimate,
                .estimatorInputHash = inputHash,
                .path1Reservation = path1Reservation,
                .startTick = startTick,
                .deadline = deadline,
                .eventGeneration =
                    m_vu1ExecutionTiming.eventToken.generation,
                .executionGeneration =
                    m_vu1ExecutionTiming.generation,
            };
    }
    catch (...)
    {
        (void)m_gifArbiter.releasePath1(path1Reservation);
        throw;
    }
    m_vu1CoarsePendingInvocation.store(
        true, std::memory_order_release);
    updateAtomicMaximum(
        m_vu1CoarsePendingInvocationHighWater,
        size_t{1u});
    m_vu1CoarseInvocationsSubmitted.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1CoarsePath1ReservationsStarted.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1CoarseEstimatedCycleSum.fetch_add(
        estimate, std::memory_order_relaxed);
    m_vu1CoarseLastEstimatedCycles.store(
        estimate, std::memory_order_release);

    m_publishedVu1Active.store(
        true, std::memory_order_release);
    if (invocationKind != Vu1InvocationKind::Mscnt)
    {
        m_publishedVu1ProgramCounter.store(
            invocationStartPc, std::memory_order_release);
    }
    m_publishedVu1ProgressActive.store(
        true, std::memory_order_release);
    m_vu1MemoryMirrorDirty = true;
}

bool PS2Runtime::completeVu1CoarseInvocation(
    ps2x::timing::EeTick publicationTick,
    bool schedulerEvent)
{
    if (!m_pendingVu1CoarseInvocation ||
        !m_threadedVu1Executor)
    {
        throw std::logic_error(
            "coarse VU1 completion has no pending invocation");
    }

    Vu1PendingCoarseInvocation &pending =
        *m_pendingVu1CoarseInvocation;

    if (!schedulerEvent &&
        m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
        m_vu1ExecutionTiming.eventToken = {
            ps2x::timing::EeEventSource::VifVu1Finish, 0u};
    }

    if (!pending.completedResult)
    {
        const bool ready = pending.submission.ready();
        const auto waitAt = std::chrono::steady_clock::now();
        Vu1CommandResult result =
            [&]()
            {
                try
                {
                    return m_threadedVu1Executor->wait(
                        std::move(pending.submission));
                }
                catch (...)
                {
                    discardVu1CoarseInvocationAfterFailure();
                    throw;
                }
            }();
        const uint64_t waitNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - waitAt)
                    .count());
        if (schedulerEvent)
        {
            (ready ? m_vu1CoarseResultsReadyAtEvent
                   : m_vu1CoarseResultsLateAtEvent)
                .fetch_add(1u, std::memory_order_relaxed);
        }
        if (!ready)
        {
            m_vu1CoarseForcedWaitCount.fetch_add(
                1u, std::memory_order_relaxed);
            m_vu1CoarseForcedWaitNanoseconds.fetch_add(
                waitNanoseconds, std::memory_order_relaxed);
            updateAtomicMaximum(
                m_vu1CoarseMaximumForcedWaitNanoseconds,
                waitNanoseconds);
        }

        if (result.disposition !=
                Vu1CommandDisposition::Completed ||
            result.digest.type != Vu1CommandType::Invocation)
        {
            discardVu1CoarseInvocationAfterFailure();
            throw std::runtime_error(
                "coarse VU1 owner rejected an invocation");
        }
        const auto *const invocation =
            std::get_if<Vu1SliceResult>(&result.payload);
        if (!invocation)
        {
            discardVu1CoarseInvocationAfterFailure();
            throw std::runtime_error(
                "coarse VU1 invocation returned no typed result");
        }
        if (invocation->fault)
        {
            const std::string fault = *invocation->fault;
            discardVu1CoarseInvocationAfterFailure();
            throw std::runtime_error(fault);
        }
        if (result.active || !invocation->vifCanResume)
        {
            discardVu1CoarseInvocationAfterFailure();
            throw std::runtime_error(
                "coarse VU1 invocation exceeded its cycle bound");
        }

        const uint32_t actual =
            invocation->run.executedCycles;
        const uint32_t estimate = pending.estimatedCycles;
        const int64_t error =
            static_cast<int64_t>(actual) -
            static_cast<int64_t>(estimate);
        const uint64_t absoluteError =
            error < 0
                ? static_cast<uint64_t>(-error)
                : static_cast<uint64_t>(error);
        if (error < 0)
        {
            m_vu1CoarseEarlyEstimateCount.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else if (error == 0)
        {
            m_vu1CoarseExactEstimateCount.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else
        {
            m_vu1CoarseLateEstimateCount.fetch_add(
                1u, std::memory_order_relaxed);
        }
        m_vu1CoarseActualCycleSum.fetch_add(
            actual, std::memory_order_relaxed);
        m_vu1CoarseAbsoluteErrorCycleSum.fetch_add(
            absoluteError, std::memory_order_relaxed);
        updateAtomicMaximum(
            m_vu1CoarseMaximumAbsoluteErrorCycles,
            absoluteError);
        m_vu1CoarseLastActualCycles.store(
            actual, std::memory_order_release);
        m_vu1CoarseLastEstimateErrorCycles.store(
            error, std::memory_order_release);

        uint64_t path1Bytes = 0u;
        for (const Vu1Path1Packet &packet :
             invocation->path1Packets)
        {
            path1Bytes += packet.bytes.size();
        }

        uint64_t resultDigest =
            m_vu1CoarseEstimatorResultDigest.load(
                std::memory_order_relaxed);
        resultDigest = appendVu1CoarseDigest(
            resultDigest, pending.estimatorInputHash);
        resultDigest = appendVu1CoarseDigest(
            resultDigest, estimate);
        resultDigest = appendVu1CoarseDigest(
            resultDigest, actual);
        resultDigest = appendVu1CoarseDigest(
            resultDigest, static_cast<uint64_t>(error));
        resultDigest = appendVu1CoarseDigest(
            resultDigest, path1Bytes);
        resultDigest = appendVu1CoarseDigest(
            resultDigest, invocation->architecturalStateHash);
        m_vu1CoarseEstimatorResultDigest.store(
            resultDigest, std::memory_order_release);

        m_vu1CoarseCycleHistory.push_back(actual);
        while (m_vu1CoarseCycleHistory.size() >
               m_vu1CoarseEstimatorHistoryCapacity)
        {
            m_vu1CoarseCycleHistory.pop_front();
        }
        m_vu1CoarseEstimatorHistorySize.store(
            m_vu1CoarseCycleHistory.size(),
            std::memory_order_release);
        updateAtomicMaximum(
            m_vu1CoarseEstimatorHistoryHighWater,
            m_vu1CoarseCycleHistory.size());

        auto matchingEntry = std::find_if(
            m_vu1CoarseEstimatorEntries.begin(),
            m_vu1CoarseEstimatorEntries.end(),
            [&pending](const Vu1CoarseEstimatorEntry &entry)
            {
                return entry.identity ==
                    pending.estimatorIdentity;
            });
        Vu1CoarseEstimatorEntry updatedEntry{};
        if (matchingEntry !=
            m_vu1CoarseEstimatorEntries.end())
        {
            updatedEntry = std::move(*matchingEntry);
            m_vu1CoarseEstimatorEntries.erase(
                matchingEntry);
        }
        else
        {
            if (m_vu1CoarseEstimatorEntries.size() >=
                m_vu1CoarseEstimatorIdentityCapacity)
            {
                m_vu1CoarseEstimatorEntries.pop_front();
            }
            updatedEntry.identity = pending.estimatorIdentity;
        }
        updatedEntry.cycleHistory.push_back(actual);
        while (updatedEntry.cycleHistory.size() >
               m_vu1CoarseEstimatorHistoryCapacity)
        {
            updatedEntry.cycleHistory.pop_front();
        }
        m_vu1CoarseEstimatorEntries.push_back(
            std::move(updatedEntry));
        m_vu1CoarseEstimatorIdentitySize.store(
            m_vu1CoarseEstimatorEntries.size(),
            std::memory_order_release);
        updateAtomicMaximum(
            m_vu1CoarseEstimatorIdentityHighWater,
            m_vu1CoarseEstimatorEntries.size());

        pending.ownerReadyAtResolution = ready;
        pending.completedResult.emplace(std::move(result));
        m_vu1CoarseInvocationsCompleted.fetch_add(
            1u, std::memory_order_relaxed);
    }

    const auto *const resolvedInvocation =
        std::get_if<Vu1SliceResult>(
            &pending.completedResult->payload);
    if (!resolvedInvocation)
    {
        throw std::logic_error(
            "coarse VU1 pending result lost its invocation payload");
    }
    const uint32_t actual =
        resolvedInvocation->run.executedCycles;
    const ps2x::timing::EeTick causalDeadline =
        ps2x::timing::saturatingAdd(
            pending.startTick,
            ps2x::timing::vuCyclesToEeTicks(actual));
    if (schedulerEvent && publicationTick < causalDeadline)
    {
        const uint64_t elapsedCycles =
            ps2x::timing::eeTicksToVuCyclesFloor(
                ps2x::timing::elapsedEeTicks(
                    pending.startTick, publicationTick));
        const uint64_t deferredCycles =
            actual > elapsedCycles
                ? static_cast<uint64_t>(actual) - elapsedCycles
                : 0u;
        scheduleVU1Event(causalDeadline);
        if (m_vu1ExecutionTiming.eventToken.generation == 0u)
        {
            throw std::runtime_error(
                "coarse VU1 causal deadline was not scheduled");
        }
        pending.deadline = causalDeadline;
        pending.eventGeneration =
            m_vu1ExecutionTiming.eventToken.generation;
        m_vu1CoarseCausalDeadlineDeferralCount.fetch_add(
            1u, std::memory_order_relaxed);
        m_vu1CoarseCausalDeadlineDeferredCycles.fetch_add(
            deferredCycles, std::memory_order_relaxed);
        updateAtomicMaximum(
            m_vu1CoarseMaximumCausalDeadlineDeferredCycles,
            deferredCycles);
        return false;
    }

    Vu1PendingCoarseInvocation completed =
        std::move(pending);
    m_pendingVu1CoarseInvocation.reset();
    m_vu1CoarsePendingInvocation.store(
        false, std::memory_order_release);
    struct Path1ReservationGuard
    {
        GifArbiter &arbiter;
        GifArbiter::Path1ReservationToken token = 0u;

        ~Path1ReservationGuard()
        {
            if (token != 0u)
                (void)arbiter.releasePath1(token);
        }

        bool release() noexcept
        {
            const bool released =
                arbiter.releasePath1(token);
            token = 0u;
            return released;
        }
    } path1Reservation{
        m_gifArbiter,
        completed.path1Reservation,
    };
    Vu1CommandResult result =
        std::move(*completed.completedResult);
    const auto *const invocation =
        std::get_if<Vu1SliceResult>(&result.payload);
    if (!invocation)
    {
        throw std::logic_error(
            "coarse VU1 publication lost its invocation payload");
    }

    uint64_t path1Bytes = 0u;
    for (const Vu1Path1Packet &packet :
         invocation->path1Packets)
    {
        path1Bytes += packet.bytes.size();
    }
    m_vu1CoarsePath1BytesPublished.fetch_add(
        path1Bytes, std::memory_order_relaxed);
    updateAtomicMaximum(
        m_vu1CoarseMaximumInvocationPath1Bytes,
        path1Bytes);

    publishVu1CommandResult(result);
    publishVu1SliceEffects(*invocation);
    const bool path1ReservationReleased =
        path1Reservation.release();
    (path1ReservationReleased
         ? m_vu1CoarsePath1ReservationsReleased
         : m_vu1CoarsePath1ReservationsInvalidated)
        .fetch_add(1u, std::memory_order_relaxed);
    m_gifArbiter.drain();
    m_vu1ExecutionTiming.totalAdvancedCycles += actual;
    m_vu1ExecutionTiming.lastAdvancedTick = publicationTick;
    setVU1BusyFlag(nullptr, false);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        DebugVu1TimingEntry entry = debugVu1TimingEntry(
            DebugVu1TimingEventKind::InvocationComplete,
            m_boundEeContext);
        entry.executedCycles = actual;
        entry.architecturalStateHash =
            invocation->architecturalStateHash;
        if (invocation->diagnosticStateHashes)
        {
            entry.inputExecutionStateHash =
                invocation->diagnosticStateHashes->inputExecution;
            entry.inputMicroMemoryHash =
                invocation->diagnosticStateHashes->inputMicroMemory;
            entry.inputDataMemoryHash =
                invocation->diagnosticStateHashes->inputDataMemory;
            entry.inputVifStateHash =
                invocation->diagnosticStateHashes->inputVif;
            entry.executionStateHash =
                invocation->diagnosticStateHashes->execution;
            entry.microMemoryHash =
                invocation->diagnosticStateHashes->microMemory;
            entry.dataMemoryHash =
                invocation->diagnosticStateHashes->dataMemory;
            entry.vifStateHash =
                invocation->diagnosticStateHashes->vif;
        }
        entry.exitReason = invocation->run.reason;
        entry.hasExitReason = true;
        entry.schedulerEvent = schedulerEvent;
        entry.ownerReady = completed.ownerReadyAtResolution;
        debugRecordVu1Timing(std::move(entry));
    }
#endif
    m_vu1CoarseInvocationsPublished.fetch_add(
        1u, std::memory_order_relaxed);
    return true;
}

void PS2Runtime::discardVu1CoarseInvocationAfterFailure() noexcept
{
    if (!m_pendingVu1CoarseInvocation)
        return;
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    m_vu1ExecutionTiming.eventToken = {
        ps2x::timing::EeEventSource::VifVu1Finish, 0u};

    const GifArbiter::Path1ReservationToken token =
        m_pendingVu1CoarseInvocation->path1Reservation;
    if (token != 0u)
    {
        if (m_gifArbiter.releasePath1(token))
        {
            m_vu1CoarsePath1ReservationsDiscardedAtFailure.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else
        {
            m_vu1CoarsePath1ReservationsInvalidated.fetch_add(
                1u, std::memory_order_relaxed);
        }
    }
    m_pendingVu1CoarseInvocation.reset();
    m_vu1CoarsePendingInvocation.store(
        false, std::memory_order_release);
    m_vu1CoarseInvocationsDiscardedAtFailure.fetch_add(
        1u, std::memory_order_relaxed);
    m_publishedVu1Active.store(
        false, std::memory_order_release);
    m_publishedVu1ProgressActive.store(
        false, std::memory_order_release);
    setVU1BusyFlag(nullptr, false);
    m_memory.cancelVIF1VuWait();
    m_memory.finishVif1DmaTrace();
}

void PS2Runtime::resolveVu1CoarseInvocationForShutdown() noexcept
{
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedCoarse ||
        !m_pendingVu1CoarseInvocation)
    {
        return;
    }
    const auto discardPath1Reservation = [this]() noexcept
    {
        if (!m_pendingVu1CoarseInvocation)
            return;
        const GifArbiter::Path1ReservationToken token =
            m_pendingVu1CoarseInvocation->path1Reservation;
        if (token == 0u)
            return;
        if (m_gifArbiter.releasePath1(token))
        {
            m_vu1CoarsePath1ReservationsDiscardedAtShutdown.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else
        {
            m_vu1CoarsePath1ReservationsInvalidated.fetch_add(
                1u, std::memory_order_relaxed);
        }
    };
    try
    {
        if (m_vu1ExecutionTiming.eventToken.generation != 0u)
        {
            (void)m_eeEventScheduler.cancel(
                m_vu1ExecutionTiming.eventToken);
            m_vu1ExecutionTiming.eventToken = {
                ps2x::timing::EeEventSource::VifVu1Finish,
                0u};
        }
        if (!m_pendingVu1CoarseInvocation->completedResult)
        {
            (void)m_threadedVu1Executor->wait(
                std::move(
                    m_pendingVu1CoarseInvocation->submission));
        }
        discardPath1Reservation();
        m_pendingVu1CoarseInvocation.reset();
        m_vu1CoarsePendingInvocation.store(
            false, std::memory_order_release);
        m_vu1CoarseInvocationsDiscardedAtShutdown.fetch_add(
            1u, std::memory_order_relaxed);
    }
    catch (...)
    {
        discardPath1Reservation();
        m_pendingVu1CoarseInvocation.reset();
        m_vu1CoarsePendingInvocation.store(
            false, std::memory_order_release);
    }
}

void PS2Runtime::resolveVu1SpeculationForShutdown() noexcept
{
    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedCoarse)
    {
        resolveVu1CoarseInvocationForShutdown();
        return;
    }
    clearDeferredVu1SpeculativeSlice();
    if (m_vu1CheckpointEpochs &&
        m_pendingVu1ExecutionEpoch &&
        m_threadedVu1Executor)
    {
        m_threadedVu1Executor->stopExecutionEpoch(
            m_pendingVu1ExecutionEpoch->epoch);
        m_pendingVu1ExecutionEpoch.reset();
        m_vu1AsyncPendingSlice.store(
            false, std::memory_order_release);
        return;
    }
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedAsync ||
        !m_threadedVu1Executor ||
        m_vu1SpeculationPublicationState ==
            Vu1SpeculationPublicationState::None)
    {
        return;
    }

    try
    {
        Vu1SynchronousCommandBarrier barrier =
            beginVu1SynchronousCommandBarrier(
                Vu1CommandType::Barrier);
        barrier.requeue.reset();
        const Vu1CommandResult result =
            m_threadedVu1Executor->submitResolvingSpeculation(
                Vu1BarrierCommand{}, barrier.resolution,
                currentEeTick().raw(),
                m_vu1ExecutionTiming.generation);
        publishVu1CommandResult(result);
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
    }
    catch (...)
    {
        m_pendingVu1Slice.reset();
        m_vu1AsyncPendingSlice.store(
            false, std::memory_order_release);
        m_vu1SpeculationPublicationState =
            Vu1SpeculationPublicationState::None;
    }
}

void PS2Runtime::cancelVU1Execution(bool resetInterpreter)
{
    clearDeferredVu1SpeculativeSlice();
    if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        m_pendingVu1CoarseInvocation)
    {
        recordVu1CoarseForcedBarrier(
            Vu1CommandType::Reset);
        (void)completeVu1CoarseInvocation(
            currentEeTick(), false);
    }
    if (m_pendingVu1ExecutionEpoch &&
        m_threadedVu1Executor)
    {
        m_threadedVu1Executor->stopExecutionEpoch(
            m_pendingVu1ExecutionEpoch->epoch);
        m_pendingVu1ExecutionEpoch.reset();
        m_vu1AsyncPendingSlice.store(
            false, std::memory_order_release);
        m_vu1AsyncHazardBarrierCount.fetch_add(
            1u, std::memory_order_relaxed);
    }
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    m_vu1ExecutionTiming.eventToken = {
        ps2x::timing::EeEventSource::VifVu1Finish, 0u};
    ++m_vu1ExecutionTiming.generation;
    if (m_vu1ExecutionTiming.generation == 0u)
        ++m_vu1ExecutionTiming.generation;
    m_vu1ExecutionTiming.lastAdvancedTick = currentEeTick();
    m_vu1ExecutionTiming.totalAdvancedCycles = 0u;
    if (resetInterpreter)
        (void)submitVu1Command(Vu1ResetCommand{});
    m_memory.cancelVIF1VuWait();
    m_memory.finishVif1DmaTrace();
    setVU1BusyFlag(nullptr, false);
}

Vu1CommandResult PS2Runtime::submitVu1Command(
    Vu1CommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!m_vu1Executor)
    {
        throw std::runtime_error(
            "VU1 command executor is unavailable");
    }
    const Vu1CommandType type = vu1CommandType(payload);
    if (type != Vu1CommandType::SetDiagnostics)
    {
        refreshVu1DiagnosticsConfiguration(
            guestTick, publicationToken);
    }
    bool foldedUnpacks = false;
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive &&
        !m_batchedVu1DecodedUnpacks.empty())
    {
        std::visit(
            [this, &foldedUnpacks](auto &command)
            {
                using T = std::decay_t<decltype(command)>;
                if constexpr (
                    std::is_same_v<T, Vu1MscalCommand> ||
                    std::is_same_v<T, Vu1MscalfCommand> ||
                    std::is_same_v<T, Vu1MscntCommand> ||
                    std::is_same_v<T, Vu1InvocationCommand>)
                {
                    command.unpacksBefore =
                        takeBatchedVu1DecodedUnpacks();
                    foldedUnpacks = true;
                }
            },
            payload);
        if (!foldedUnpacks)
            flushBatchedVu1DecodedUnpacks();
    }
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive &&
        m_batchedVu1VifState)
    {
        bool foldedState = false;
        std::visit(
            [this, &foldedState](auto &command)
            {
                using T = std::decay_t<decltype(command)>;
                if constexpr (
                    std::is_same_v<T, Vu1MscalCommand> ||
                    std::is_same_v<T, Vu1MscalfCommand> ||
                    std::is_same_v<T, Vu1MscntCommand> ||
                    std::is_same_v<T, Vu1InvocationCommand>)
                {
                    command.vifStateBefore =
                        std::make_shared<const Vu1VifState>(
                            *m_batchedVu1VifState);
                    m_batchedVu1VifState.reset();
                    foldedState = true;
                }
            },
            payload);
        if (!foldedState)
            flushBatchedVu1VifStateUpdate();
    }
    const bool resultlessMemoryWrite =
        usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        (type == Vu1CommandType::MicroMemoryWrite ||
         type == Vu1CommandType::DataMemoryWrite);
    if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        m_pendingVu1CoarseInvocation &&
        type == Vu1CommandType::SetBackend)
    {
        Vu1CommandResult result =
            m_threadedVu1Executor->submit(
                std::move(payload), guestTick,
                publicationToken);
        if (result.disposition !=
                Vu1CommandDisposition::Completed ||
            result.digest.type != Vu1CommandType::SetBackend)
        {
            throw std::runtime_error(
                "VU1 non-publishing backend command was rejected");
        }
        m_vu1CoarseNonPublishingBackendChangeCount.fetch_add(
            1u, std::memory_order_relaxed);
        return result;
    }
    if (resultlessMemoryWrite)
    {
        uint64_t admittedCodeGeneration =
            m_memory.getVU1CodeGeneration();
        if (type == Vu1CommandType::MicroMemoryWrite)
        {
            if (admittedCodeGeneration ==
                std::numeric_limits<uint64_t>::max())
            {
                throw std::overflow_error(
                    "VU1 code generation exhausted");
            }
            ++admittedCodeGeneration;
        }
        const uint64_t payloadSize =
            vu1CommandPayloadSize(payload);
        const Vu1SynchronousCommandBarrier barrier =
            usesExactVu1Scheduling(m_vu1ExecutionMode)
                ? beginVu1SynchronousCommandBarrier(type)
                : Vu1SynchronousCommandBarrier{};
        const Vu1CommandAdmission admission =
            m_threadedVu1Executor
                ->submitResultlessResolvingSpeculation(
                    std::move(payload),
                    barrier.resolution,
                    guestTick, publicationToken);
        m_vu1MemoryMirrorDirty = true;
        finishVu1SynchronousCommandBarrier(barrier);
        return Vu1CommandResult{
            .identity = admission.identity,
            .digest = {
                .sequence = admission.identity.sequence,
                .generation = admission.identity.generation,
                .guestTick = admission.identity.guestTick,
                .type = admission.type,
                .payloadSize = payloadSize,
            },
            .ownerGeneration = admission.identity.generation,
            .codeGeneration = admittedCodeGeneration,
            .programCounter =
                m_publishedVu1ProgramCounter.load(
                    std::memory_order_acquire),
            .active = m_publishedVu1Active.load(
                std::memory_order_acquire),
            .progress = vu1ProgressSnapshot(),
        };
    }
    const Vu1SynchronousCommandBarrier barrier =
        beginVu1SynchronousCommandBarrier(type);
    Vu1CommandResult result =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? m_threadedVu1Executor
                  ->submitResolvingSpeculation(
                      std::move(payload),
                      barrier.resolution,
                      guestTick, publicationToken)
            : m_vu1Executor->submit(
                  std::move(payload),
                  guestTick, publicationToken);
    if (result.disposition !=
        Vu1CommandDisposition::Completed)
    {
        throw std::runtime_error(
            std::string("VU1 command rejected: ") +
            vu1CommandTypeName(result.digest.type));
    }
    publishVu1CommandResult(result);
    if (foldedUnpacks)
        m_vu1MemoryMirrorDirty = true;
    if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        (type == Vu1CommandType::BindMemory ||
         type == Vu1CommandType::Reset ||
         type == Vu1CommandType::Restore))
    {
        m_vu1CoarseCycleHistory.clear();
        m_vu1CoarseEstimatorEntries.clear();
        m_vu1CoarseEstimatorHistorySize.store(
            0u, std::memory_order_release);
        m_vu1CoarseEstimatorIdentitySize.store(
            0u, std::memory_order_release);
    }
    finishVu1SynchronousCommandBarrier(barrier);
    return result;
}

Vu1CommandResult PS2Runtime::submitVu1AdvanceSlice(
    Vu1AdvanceSliceCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!m_vu1Executor)
    {
        throw std::runtime_error(
            "VU1 command executor is unavailable");
    }
    refreshVu1DiagnosticsConfiguration(
        guestTick, publicationToken);
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive &&
        !m_batchedVu1DecodedUnpacks.empty())
    {
        flushBatchedVu1DecodedUnpacks();
    }
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive &&
        m_batchedVu1VifState)
    {
        flushBatchedVu1VifStateUpdate();
    }
    const Vu1SynchronousCommandBarrier barrier =
        beginVu1SynchronousCommandBarrier(
            Vu1CommandType::AdvanceSlice);
    Vu1CommandResult result =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? m_threadedVu1Executor
                  ->submitResolvingSpeculation(
                      Vu1CommandPayload{command},
                      barrier.resolution,
                      guestTick, publicationToken)
            : m_vu1Executor->submitAdvanceSlice(
                  command, guestTick, publicationToken);
    if (result.disposition !=
        Vu1CommandDisposition::Completed)
    {
        throw std::runtime_error(
            std::string("VU1 command rejected: ") +
            vu1CommandTypeName(result.digest.type));
    }
    publishVu1CommandResult(result);
    finishVu1SynchronousCommandBarrier(barrier);
    return result;
}

Vu1CommandResult PS2Runtime::submitVu1DecodedUnpack(
    Vu1DecodedUnpackCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!m_vu1Executor)
    {
        throw std::runtime_error(
            "VU1 command executor is unavailable");
    }
    refreshVu1DiagnosticsConfiguration(
        guestTick, publicationToken);
    const bool asynchronousBatch =
        usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive;
    if (asynchronousBatch && m_batchedVu1VifState)
    {
        command.vifStateBefore =
            std::make_shared<const Vu1VifState>(
                *m_batchedVu1VifState);
        m_knownVu1VifState =
            *m_batchedVu1VifState;
        m_batchedVu1VifState.reset();
    }
    if (asynchronousBatch && command.vifStateBefore)
    {
        m_knownVu1VifState =
            *command.vifStateBefore;
    }

    const uint64_t commandPayloadBytes =
        11u + sizeof(uint64_t) + command.bytes.size() +
        (command.vifStateBefore
             ? 12u * sizeof(uint32_t)
             : 0u);
    constexpr uint64_t kFoldedCommandReserveBytes = 128u;
    const uint64_t maximumBatchPayloadBytes =
        m_vu1CommandPayloadCapacityBytes >
                kFoldedCommandReserveBytes
            ? m_vu1CommandPayloadCapacityBytes -
                  kFoldedCommandReserveBytes
            : 0u;
    const auto fitsEmptyBatch = [&]() noexcept
    {
        return maximumBatchPayloadBytes >=
                   sizeof(uint64_t) &&
               commandPayloadBytes <=
                   maximumBatchPayloadBytes -
                       sizeof(uint64_t);
    };
    const bool rowStable =
        m_knownVu1VifState &&
        (m_knownVu1VifState->mode & 3u) != 2u;
    if (asynchronousBatch && rowStable &&
        fitsEmptyBatch())
    {
        const bool fitsCurrentBatch =
            m_batchedVu1DecodedUnpackPayloadBytes <=
                maximumBatchPayloadBytes -
                    sizeof(uint64_t) -
                    commandPayloadBytes;
        if (!fitsCurrentBatch)
            flushBatchedVu1DecodedUnpacks();

        m_batchedVu1DecodedUnpackPayloadBytes +=
            commandPayloadBytes;
        m_batchedVu1DecodedUnpacks.push_back(
            std::move(command));
        m_batchedVu1DecodedUnpackFinalVifState =
            m_knownVu1VifState;

        Vu1CommandResult buffered{};
        buffered.identity = {
            .generation = publicationToken,
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        };
        buffered.digest = {
            .generation = publicationToken,
            .guestTick = guestTick,
            .type = Vu1CommandType::DecodedUnpack,
            .payloadSize = commandPayloadBytes,
        };
        buffered.ownerGeneration = publicationToken;
        buffered.codeGeneration =
            m_publishedVu1CodeGeneration.load(
                std::memory_order_acquire);
        buffered.programCounter =
            m_publishedVu1ProgramCounter.load(
                std::memory_order_acquire);
        buffered.active = m_publishedVu1Active.load(
            std::memory_order_acquire);
        return buffered;
    }

    if (asynchronousBatch)
        flushBatchedVu1DecodedUnpacks();
    const Vu1SynchronousCommandBarrier barrier =
        beginVu1SynchronousCommandBarrier(
            Vu1CommandType::DecodedUnpack);
    Vu1CommandResult result =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? m_threadedVu1Executor
                  ->submitResolvingSpeculation(
                      Vu1CommandPayload{
                          std::move(command)},
                      barrier.resolution,
                      guestTick, publicationToken)
            : m_vu1Executor->submitDecodedUnpack(
                  std::move(command), guestTick,
                  publicationToken);
    if (result.disposition !=
        Vu1CommandDisposition::Completed)
    {
        throw std::runtime_error(
            "VU1 command rejected: decoded-unpack");
    }
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode))
    {
        if (const auto *const vifState =
                std::get_if<Vu1VifStateResult>(
                    &result.payload))
        {
            m_knownVu1VifState =
                vifState->state;
        }
    }
    publishVu1CommandResult(result);
    finishVu1SynchronousCommandBarrier(barrier);
    return result;
}

Vu1CommandResult PS2Runtime::submitVu1VifStateUpdate(
    Vu1VifStateUpdateCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!m_vu1Executor)
    {
        throw std::runtime_error(
            "VU1 command executor is unavailable");
    }
    refreshVu1DiagnosticsConfiguration(
        guestTick, publicationToken);
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive)
    {
        m_batchedVu1VifState = command.state;
        m_knownVu1VifState = command.state;
        Vu1CommandResult buffered{};
        buffered.identity = {
            .generation = publicationToken,
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        };
        buffered.digest = {
            .generation = publicationToken,
            .guestTick = guestTick,
            .type = Vu1CommandType::VifStateUpdate,
            .payloadSize = 12u * sizeof(uint32_t),
        };
        buffered.ownerGeneration = publicationToken;
        buffered.codeGeneration =
            m_publishedVu1CodeGeneration.load(
                std::memory_order_acquire);
        buffered.programCounter =
            m_publishedVu1ProgramCounter.load(
                std::memory_order_acquire);
        buffered.active = m_publishedVu1Active.load(
            std::memory_order_acquire);
        buffered.payload = Vu1VifStateResult{command.state};
        return buffered;
    }
    return submitVu1VifStateUpdateImmediate(
        std::move(command), guestTick, publicationToken);
}

Vu1CommandResult PS2Runtime::submitVu1VifStateUpdateImmediate(
    Vu1VifStateUpdateCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive &&
        !m_batchedVu1DecodedUnpacks.empty())
    {
        flushBatchedVu1DecodedUnpacks();
    }
    if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        m_pendingVu1CoarseInvocation)
    {
        const Vu1VifState state = command.state;
        const Vu1CommandAdmission admission =
            m_threadedVu1Executor->submitResultless(
                Vu1CommandPayload{std::move(command)},
                guestTick, publicationToken);
        if (admission.type !=
                Vu1CommandType::VifStateUpdate)
        {
            throw std::runtime_error(
                "VU1 non-publishing VIF state update was rejected");
        }
        m_knownVu1VifState = state;
        m_vu1CoarseNonPublishingVifStateUpdateCount.fetch_add(
            1u, std::memory_order_relaxed);
        return Vu1CommandResult{
            .identity = admission.identity,
            .digest = {
                .sequence = admission.identity.sequence,
                .generation = admission.identity.generation,
                .guestTick = admission.identity.guestTick,
                .type = admission.type,
                .payloadSize = 12u * sizeof(uint32_t),
            },
            .ownerGeneration = admission.identity.generation,
            .codeGeneration =
                m_publishedVu1CodeGeneration.load(
                    std::memory_order_acquire),
            .programCounter =
                m_publishedVu1ProgramCounter.load(
                    std::memory_order_acquire),
            .active = m_publishedVu1Active.load(
                std::memory_order_acquire),
            .progress = vu1ProgressSnapshot(),
            .payload = Vu1VifStateResult{state},
        };
    }
    const Vu1SynchronousCommandBarrier barrier =
        beginVu1SynchronousCommandBarrier(
            Vu1CommandType::VifStateUpdate);
    Vu1CommandResult result =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? m_threadedVu1Executor
                  ->submitResolvingSpeculation(
                      Vu1CommandPayload{command},
                      barrier.resolution,
                      guestTick, publicationToken)
            : m_vu1Executor->submitVifStateUpdate(
                  command, guestTick, publicationToken);
    if (result.disposition !=
        Vu1CommandDisposition::Completed)
    {
        throw std::runtime_error(
            "VU1 command rejected: vif-state-update");
    }
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode))
    {
        if (const auto *const vifState =
                std::get_if<Vu1VifStateResult>(
                    &result.payload))
        {
            m_knownVu1VifState = vifState->state;
        }
    }
    publishVu1CommandResult(result);
    finishVu1SynchronousCommandBarrier(barrier);
    return result;
}

void PS2Runtime::refreshVu1DiagnosticsConfiguration(
    uint64_t guestTick,
    uint64_t publicationToken)
{
    const bool traceEnabled = m_memory.vuTraceEnabled();
    const bool workloadProfileEnabled =
        m_memory.vuWorkloadProfileEnabled();
    if (traceEnabled == m_publishedVu1TraceEnabled &&
        workloadProfileEnabled ==
            m_publishedVu1WorkloadProfileEnabled)
    {
        return;
    }

    // A diagnostics command is owner-visible and must not overtake parser
    // state that was already acknowledged on EE. This path is cold and the
    // immediate flush preserves the original command order.
    if (usesBufferedVu1OwnerStream(m_vu1ExecutionMode) &&
        m_vu1SynchronousCommandBatchActive)
    {
        flushBatchedVu1DecodedUnpacks();
        if (m_batchedVu1VifState)
            flushBatchedVu1VifStateUpdate();
    }

    Vu1CommandPayload payload =
        Vu1SetDiagnosticsCommand{
            .traceEnabled = traceEnabled,
            .workloadProfileEnabled =
                workloadProfileEnabled,
        };
    if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedCoarse &&
        m_pendingVu1CoarseInvocation)
    {
        (void)m_threadedVu1Executor->submitResultless(
            std::move(payload), guestTick, publicationToken);
        m_publishedVu1TraceEnabled = traceEnabled;
        m_publishedVu1WorkloadProfileEnabled =
            workloadProfileEnabled;
        m_vu1CoarseNonPublishingDiagnosticsUpdateCount.fetch_add(
            1u, std::memory_order_relaxed);
        return;
    }
    const Vu1SynchronousCommandBarrier barrier =
        beginVu1SynchronousCommandBarrier(
            Vu1CommandType::SetDiagnostics);
    Vu1CommandResult result =
        usesExactVu1Scheduling(m_vu1ExecutionMode)
            ? m_threadedVu1Executor
                  ->submitResolvingSpeculation(
                      std::move(payload),
                      barrier.resolution,
                      guestTick, publicationToken)
            : m_vu1Executor->submit(
                  std::move(payload),
                  guestTick, publicationToken);
    if (result.disposition !=
        Vu1CommandDisposition::Completed)
    {
        throw std::runtime_error(
            "VU1 command rejected: set-diagnostics");
    }
    m_publishedVu1TraceEnabled = traceEnabled;
    m_publishedVu1WorkloadProfileEnabled =
        workloadProfileEnabled;
    publishVu1CommandResult(result);
    finishVu1SynchronousCommandBarrier(barrier);
}

void PS2Runtime::publishVu1CommandResult(
    const Vu1CommandResult &result)
{
    m_publishedVu1Active.store(
        result.active, std::memory_order_release);
    m_publishedVu1ProgramCounter.store(
        result.programCounter,
        std::memory_order_release);
    m_publishedVu1CodeGeneration.store(
        result.codeGeneration,
        std::memory_order_release);
    m_publishedVu1ProgressEnabled.store(
        result.progress.enabled,
        std::memory_order_release);
    m_publishedVu1ProgressActive.store(
        result.progress.active,
        std::memory_order_release);
    m_publishedVu1ProgressInvocations.store(
        result.progress.invocations,
        std::memory_order_release);
    m_publishedVu1ProgressCycles.store(
        result.progress.cycles,
        std::memory_order_release);
    m_publishedVu1ProgressPc.store(
        result.progress.pc,
        std::memory_order_release);

    switch (result.digest.type)
    {
    case Vu1CommandType::BindMemory:
        m_vu1MemoryMirrorDirty = false;
        m_knownVu1VifState.reset();
        break;
    case Vu1CommandType::MicroMemoryWrite:
    case Vu1CommandType::DataMemoryWrite:
    case Vu1CommandType::DecodedUnpack:
    case Vu1CommandType::Invocation:
    case Vu1CommandType::AdvanceSlice:
        m_vu1MemoryMirrorDirty = true;
        break;
    case Vu1CommandType::Reset:
    case Vu1CommandType::Restore:
        m_vu1MemoryMirrorDirty = true;
        m_knownVu1VifState.reset();
        break;
    default:
        break;
    }

    if (!result.diagnostics.empty())
    {
        replayVu1Diagnostics(result.diagnostics, m_memory);
    }
}

std::shared_ptr<const Vu1Snapshot>
PS2Runtime::captureVu1OwnerSnapshot(
    bool includeBackendDiagnostics)
{
    const Vu1CommandResult result = submitVu1Command(
        Vu1SnapshotCommand{
            .includeBackendDiagnostics =
                includeBackendDiagnostics},
        currentEeTick().raw(),
        m_vu1ExecutionTiming.generation);
    const auto *const snapshot =
        std::get_if<Vu1SnapshotResult>(&result.payload);
    if (!snapshot || !snapshot->snapshot)
    {
        throw std::runtime_error(
            "VU1 snapshot command returned no state");
    }
    return snapshot->snapshot;
}

std::shared_ptr<const Vu1Snapshot>
PS2Runtime::captureVu1OwnerDiagnosticSnapshot(
    bool includeBackendDiagnostics)
{
    if (m_vu1ExecutionMode !=
            Vu1ExecutionMode::ThreadedCoarse ||
        !m_pendingVu1CoarseInvocation ||
        !m_threadedVu1Executor)
    {
        return captureVu1OwnerSnapshot(
            includeBackendDiagnostics);
    }
    if (m_vu1SynchronousCommandBatchActive ||
        !m_batchedVu1DecodedUnpacks.empty() ||
        m_batchedVu1VifState)
    {
        throw std::logic_error(
            "VU1 diagnostic snapshot cannot interleave a command batch");
    }

    const auto waitAt = std::chrono::steady_clock::now();
    const Vu1CommandResult result =
        m_threadedVu1Executor->submit(
            Vu1SnapshotCommand{
                .includeBackendDiagnostics =
                    includeBackendDiagnostics},
            currentEeTick().raw(),
            m_vu1ExecutionTiming.generation);
    const uint64_t waitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitAt)
                .count());
    m_vu1CoarseNonPublishingDiagnosticSnapshotCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1CoarseNonPublishingDiagnosticSnapshotWaitCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_vu1CoarseNonPublishingDiagnosticSnapshotWaitNanoseconds.fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    updateAtomicMaximum(
        m_vu1CoarseMaximumNonPublishingDiagnosticSnapshotWaitNanoseconds,
        waitNanoseconds);

    if (result.disposition !=
            Vu1CommandDisposition::Completed ||
        result.digest.type != Vu1CommandType::Snapshot)
    {
        throw std::runtime_error(
            "VU1 diagnostic snapshot command was rejected");
    }
    const auto *const snapshot =
        std::get_if<Vu1SnapshotResult>(&result.payload);
    if (!snapshot || !snapshot->snapshot)
    {
        throw std::runtime_error(
            "VU1 diagnostic snapshot command returned no state");
    }
    return snapshot->snapshot;
}

void PS2Runtime::synchronizeVu1MemoryMirror()
{
    if (!m_vu1MemoryMirrorDirty ||
        m_vu1MemoryMirrorSynchronizationActive)
    {
        return;
    }

    m_vu1MemoryMirrorSynchronizationActive = true;
    try
    {
        const std::shared_ptr<const Vu1Snapshot> snapshot =
            captureVu1OwnerSnapshot();
        m_memory.publishVu1MemorySnapshot(*snapshot);
        m_vu1MemoryMirrorDirty = false;
        m_vu1MemoryMirrorSynchronizationActive = false;
    }
    catch (...)
    {
        m_vu1MemoryMirrorSynchronizationActive = false;
        throw;
    }
}

void PS2Runtime::publishVu1SliceEffects(
    const Vu1SliceResult &slice)
{
    for (const Vu1Path1Packet &packet : slice.path1Packets)
    {
        if (packet.bytes.empty())
            continue;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
        debugRecordVu1Path1Packet(
            packet, slice.run.executedCycles);
#endif
        m_memory.submitGifPacket(
            GifPathId::Path1,
            packet.bytes.data(),
            static_cast<uint32_t>(packet.bytes.size()));
    }
}

void PS2Runtime::startVU1FromVif(
    uint32_t startPC, uint32_t top, uint32_t itop,
    bool flushBeforeStart)
{
    clearDeferredVu1SpeculativeSlice();
    const ps2x::timing::EeTick startTick =
        commitEeContextProgress(m_boundEeContext);
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    ++m_vu1ExecutionTiming.generation;
    if (m_vu1ExecutionTiming.generation == 0u)
        ++m_vu1ExecutionTiming.generation;
    m_vu1ExecutionTiming.lastAdvancedTick = startTick;
    m_vu1ExecutionTiming.totalAdvancedCycles = 0u;
    const Vu1InvocationKind invocationKind = flushBeforeStart
        ? Vu1InvocationKind::Mscalf
        : Vu1InvocationKind::Mscal;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    debugBeginVu1TimingInvocation(
        invocationKind, startPC, top, itop, startTick);
#endif
    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedCoarse)
    {
        submitVu1CoarseInvocation(
            Vu1InvocationCommand{
                .kind = invocationKind,
                .startPc = startPC,
                .top = top,
                .itop = itop,
            },
            startTick);
        setVU1BusyFlag(nullptr, true);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
        debugPublishVu1TimingInvocationStart(
            m_pendingVu1CoarseInvocation->deadline,
            m_pendingVu1CoarseInvocation->estimatedCycles);
#endif
        return;
    }
    if (flushBeforeStart)
    {
        (void)submitVu1Command(
            Vu1MscalfCommand{startPC, top, itop},
            startTick.raw(),
            m_vu1ExecutionTiming.generation);
    }
    else
    {
        (void)submitVu1Command(
            Vu1MscalCommand{startPC, top, itop},
            startTick.raw(),
            m_vu1ExecutionTiming.generation);
    }
    setVU1BusyFlag(nullptr, true);
    const ps2x::timing::EeTick deadline =
        ps2x::timing::saturatingAdd(
            startTick,
            ps2x::timing::vuCyclesToEeTicks(
                kVu1StartupEventCycles));
    scheduleVU1Event(deadline);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    debugPublishVu1TimingInvocationStart(
        deadline, kVu1StartupEventCycles);
#endif
    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedAsync)
    {
        enqueueVu1SpeculativeSlice(
            Vu1PendingSlicePlan{
                .cycleBudget = kVu1StartupEventCycles,
                .deadline = deadline,
                .eventGeneration =
                    m_vu1ExecutionTiming.eventToken.generation,
                .executionGeneration =
                    m_vu1ExecutionTiming.generation,
            },
            Vu1SpeculationResolution::None);
    }
}

void PS2Runtime::resumeVU1FromVif(
    uint32_t top, uint32_t itop)
{
    clearDeferredVu1SpeculativeSlice();
    const ps2x::timing::EeTick startTick =
        commitEeContextProgress(m_boundEeContext);
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    ++m_vu1ExecutionTiming.generation;
    if (m_vu1ExecutionTiming.generation == 0u)
        ++m_vu1ExecutionTiming.generation;
    m_vu1ExecutionTiming.lastAdvancedTick = startTick;
    m_vu1ExecutionTiming.totalAdvancedCycles = 0u;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    debugBeginVu1TimingInvocation(
        Vu1InvocationKind::Mscnt,
        m_publishedVu1ProgramCounter.load(
            std::memory_order_acquire),
        top, itop, startTick);
#endif
    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedCoarse)
    {
        submitVu1CoarseInvocation(
            Vu1InvocationCommand{
                .kind = Vu1InvocationKind::Mscnt,
                .top = top,
                .itop = itop,
            },
            startTick);
        setVU1BusyFlag(nullptr, true);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
        debugPublishVu1TimingInvocationStart(
            m_pendingVu1CoarseInvocation->deadline,
            m_pendingVu1CoarseInvocation->estimatedCycles);
#endif
        return;
    }
    (void)submitVu1Command(
        Vu1MscntCommand{top, itop},
        startTick.raw(),
        m_vu1ExecutionTiming.generation);
    setVU1BusyFlag(nullptr, true);
    const ps2x::timing::EeTick deadline =
        ps2x::timing::saturatingAdd(
            startTick,
            ps2x::timing::vuCyclesToEeTicks(
                kVu1StartupEventCycles));
    scheduleVU1Event(deadline);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    debugPublishVu1TimingInvocationStart(
        deadline, kVu1StartupEventCycles);
#endif
    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedAsync)
    {
        enqueueVu1SpeculativeSlice(
            Vu1PendingSlicePlan{
                .cycleBudget = kVu1StartupEventCycles,
                .deadline = deadline,
                .eventGeneration =
                    m_vu1ExecutionTiming.eventToken.generation,
                .executionGeneration =
                    m_vu1ExecutionTiming.generation,
            },
            Vu1SpeculationResolution::None);
    }
}

void PS2Runtime::serviceVU1AtEvent(
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    if (m_vu1ExecutionTiming.eventToken.generation == 0u ||
        service.generation !=
            m_vu1ExecutionTiming.eventToken.generation)
    {
        return;
    }
    m_vu1ExecutionTiming.eventToken = {
        ps2x::timing::EeEventSource::VifVu1Finish, 0u};

    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedCoarse)
    {
        if (!m_pendingVu1CoarseInvocation ||
            m_pendingVu1CoarseInvocation->eventGeneration !=
                service.generation ||
            m_pendingVu1CoarseInvocation->executionGeneration !=
                m_vu1ExecutionTiming.generation)
        {
            throw std::logic_error(
                "coarse VU1 event has no matching invocation");
        }
        setVU1BusyFlag(ctx, true);
        if (!completeVu1CoarseInvocation(
                service.serviceTick, true))
        {
            return;
        }
        (void)resumeVif1AfterVuWithTimingTrace();
        if (!m_publishedVu1Active.load(
                std::memory_order_acquire) &&
            !m_memory.vif1DmaSnapshot().active)
        {
            m_memory.finishVif1DmaTrace();
        }
        return;
    }

    if (!m_publishedVu1Active.load(std::memory_order_acquire))
    {
        if (m_vu1ExecutionMode ==
                Vu1ExecutionMode::ThreadedAsync &&
            (m_vu1SpeculationPublicationState !=
                 Vu1SpeculationPublicationState::None ||
             m_pendingVu1ExecutionEpoch))
        {
            throw std::logic_error(
                "active VU1 speculation has no published busy state");
        }
        setVU1BusyFlag(ctx, false);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
        if (m_debugVu1TimingTraceEnabled.load(
                std::memory_order_relaxed))
        {
            DebugVu1TimingEntry entry =
                debugVu1TimingEntry(
                    DebugVu1TimingEventKind::
                        InvocationComplete,
                    ctx);
            entry.executedCycles = static_cast<uint32_t>(
                std::min<uint64_t>(
                    m_vu1ExecutionTiming.totalAdvancedCycles,
                    UINT32_MAX));
            entry.schedulerEvent = true;
            debugRecordVu1Timing(std::move(entry));
        }
#endif
        (void)resumeVif1AfterVuWithTimingTrace();
        if (!m_publishedVu1Active.load(
                std::memory_order_acquire) &&
            !m_memory.vif1DmaSnapshot().active)
            m_memory.finishVif1DmaTrace();
        return;
    }

    setVU1BusyFlag(ctx, true);
    const uint64_t elapsedCycles =
        ps2x::timing::eeTicksToVuCyclesFloor(
            ps2x::timing::elapsedEeTicks(
                m_vu1ExecutionTiming.lastAdvancedTick,
                service.serviceTick));
    const uint32_t cycleBudget =
        static_cast<uint32_t>(
            std::min<uint64_t>(
                elapsedCycles,
                kVu1MaximumAdvanceCycles));
    if (cycleBudget == 0u)
    {
        if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedAsync)
        {
            throw std::logic_error(
                "asynchronous VU1 event has a zero cycle budget");
        }
        scheduleVU1Event(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::vuCyclesToEeTicks(1u)));
        return;
    }

    Vu1CommandResult commandResult{};
    if (m_vu1ExecutionMode ==
        Vu1ExecutionMode::ThreadedAsync)
    {
        if (m_vu1CheckpointEpochs)
        {
            if (!m_pendingVu1ExecutionEpoch)
            {
                throw std::logic_error(
                    "VU1 event has no pending checkpoint epoch");
            }
            commandResult =
                consumeVu1ExecutionEpochAtEvent(
                    service, cycleBudget);
        }
        else
        {
            if (!m_pendingVu1Slice)
            {
                throw std::logic_error(
                    "VU1 event has no pending speculative slice");
            }
            commandResult =
                consumeVu1SpeculativeSliceAtEvent(
                    service, cycleBudget);
        }
        publishVu1CommandResult(commandResult);
        m_vu1AsyncSlicesPublished.fetch_add(
            1u, std::memory_order_relaxed);
    }
    else
    {
        commandResult = submitVu1AdvanceSlice(
            Vu1AdvanceSliceCommand{
                .maximumCycles = cycleBudget,
            },
            service.serviceTick.raw(),
            service.generation);
    }
    const auto *const slice =
        std::get_if<Vu1SliceResult>(
            &commandResult.payload);
    if (!slice)
    {
        throw std::runtime_error(
            "VU1 advance command returned no slice result");
    }
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed) &&
        m_vu1ExecutionTiming.totalAdvancedCycles == 0u &&
        slice->diagnosticStateHashes)
    {
        m_debugVu1TimingInvocation.inputStateHashes.aggregate =
            slice->diagnosticStateHashes->inputAggregate;
        m_debugVu1TimingInvocation.inputStateHashes.execution =
            slice->diagnosticStateHashes->inputExecution;
        m_debugVu1TimingInvocation.inputStateHashes.microMemory =
            slice->diagnosticStateHashes->inputMicroMemory;
        m_debugVu1TimingInvocation.inputStateHashes.dataMemory =
            slice->diagnosticStateHashes->inputDataMemory;
        m_debugVu1TimingInvocation.inputStateHashes.vif =
            slice->diagnosticStateHashes->inputVif;
    }
#endif
    publishVu1SliceEffects(*slice);
    const VuRunResult &advance = slice->run;
    m_vu1ExecutionTiming.totalAdvancedCycles +=
        advance.executedCycles;

    if (commandResult.active)
    {
        m_vu1ExecutionTiming.lastAdvancedTick =
            ps2x::timing::saturatingAdd(
                m_vu1ExecutionTiming.lastAdvancedTick,
                ps2x::timing::vuCyclesToEeTicks(
                    advance.executedCycles));
        const ps2x::timing::EeTick deadline =
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::vuCyclesToEeTicks(
                    kVu1FollowupEventCycles));
        scheduleVU1Event(deadline);
        if (m_vu1ExecutionMode ==
            Vu1ExecutionMode::ThreadedAsync)
        {
            const uint64_t nextElapsedCycles =
                ps2x::timing::eeTicksToVuCyclesFloor(
                    ps2x::timing::elapsedEeTicks(
                        m_vu1ExecutionTiming.lastAdvancedTick,
                        deadline));
            const uint32_t nextCycleBudget =
                static_cast<uint32_t>(
                    std::min<uint64_t>(
                        nextElapsedCycles,
                        kVu1MaximumAdvanceCycles));
            const Vu1PendingSlicePlan nextPlan{
                    .cycleBudget = nextCycleBudget,
                    .deadline = deadline,
                    .eventGeneration =
                        m_vu1ExecutionTiming.eventToken.generation,
                    .executionGeneration =
                        m_vu1ExecutionTiming.generation,
                };
            if (m_vu1CheckpointEpochs)
            {
                if (!m_pendingVu1ExecutionEpoch)
                {
                    throw std::logic_error(
                        "active VU1 epoch lost its checkpoint horizon");
                }
                const auto submitAt =
                    std::chrono::steady_clock::now();
                m_pendingVu1ExecutionEpoch->plan = nextPlan;
                refreshVu1DiagnosticsConfiguration(
                    deadline.raw(),
                    m_vu1ExecutionTiming.eventToken.generation);
                const uint64_t submitNanoseconds =
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            submitAt)
                            .count());
                m_vu1AsyncSlicesSubmitted.fetch_add(
                    1u, std::memory_order_relaxed);
                m_vu1AsyncSliceSubmitCount.fetch_add(
                    1u, std::memory_order_relaxed);
                m_vu1AsyncSliceSubmitNanoseconds.fetch_add(
                    submitNanoseconds,
                    std::memory_order_relaxed);
                updateAtomicMaximum(
                    m_vu1AsyncMaximumSliceSubmitNanoseconds,
                    submitNanoseconds);
            }
            else
            {
                refreshVu1DiagnosticsConfiguration(
                    deadline.raw(),
                    m_vu1ExecutionTiming.eventToken.generation);
                enqueueVu1SpeculativeSlice(
                    nextPlan,
                    m_vu1SpeculationPublicationState ==
                            Vu1SpeculationPublicationState::Published
                        ? Vu1SpeculationResolution::Commit
                        : Vu1SpeculationResolution::None);
            }
        }
        return;
    }

    if (m_pendingVu1ExecutionEpoch)
    {
        m_pendingVu1ExecutionEpoch.reset();
        m_vu1AsyncPendingSlice.store(
            false, std::memory_order_release);
    }
    m_vu1ExecutionTiming.lastAdvancedTick =
        service.serviceTick;
    setVU1BusyFlag(ctx, false);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        DebugVu1TimingEntry entry = debugVu1TimingEntry(
            DebugVu1TimingEventKind::InvocationComplete,
            ctx);
        entry.executedCycles = static_cast<uint32_t>(
            std::min<uint64_t>(
                m_vu1ExecutionTiming.totalAdvancedCycles,
                UINT32_MAX));
        entry.architecturalStateHash =
            slice->architecturalStateHash;
        entry.inputExecutionStateHash =
            m_debugVu1TimingInvocation.
                inputStateHashes.execution;
        entry.inputMicroMemoryHash =
            m_debugVu1TimingInvocation.
                inputStateHashes.microMemory;
        entry.inputDataMemoryHash =
            m_debugVu1TimingInvocation.
                inputStateHashes.dataMemory;
        entry.inputVifStateHash =
            m_debugVu1TimingInvocation.
                inputStateHashes.vif;
        if (slice->diagnosticStateHashes)
        {
            entry.executionStateHash =
                slice->diagnosticStateHashes->execution;
            entry.microMemoryHash =
                slice->diagnosticStateHashes->microMemory;
            entry.dataMemoryHash =
                slice->diagnosticStateHashes->dataMemory;
            entry.vifStateHash =
                slice->diagnosticStateHashes->vif;
        }
        entry.exitReason = advance.reason;
        entry.hasExitReason = true;
        entry.schedulerEvent = true;
        debugRecordVu1Timing(std::move(entry));
    }
#endif
    (void)resumeVif1AfterVuWithTimingTrace();
    if (!m_publishedVu1Active.load(
            std::memory_order_acquire) &&
        !m_memory.vif1DmaSnapshot().active)
        m_memory.finishVif1DmaTrace();
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        m_eeCache.reset();
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }
        m_memory.installPostBiosTlbState();
        m_cpuContext.resetPostBiosTlbRegisters();

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PS2X_IOP_ENABLE_PLUGINS) && PS2X_IOP_ENABLE_PLUGINS && \
    !defined(PLATFORM_VITA) && (defined(_WIN32) || defined(__linux__))
        std::string pluginError;
        if (!m_iopSubsystem->loadPlugins(&pluginError))
        {
            std::cerr << "Failed to load IOP plugins: " << pluginError << std::endl;
            return false;
        }
#endif
        m_windowTitle =
            title && title[0] != '\0'
                ? title
                : "PS2 Game";
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, m_windowTitle.c_str()); // raylib vita does not support audio
#else
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, m_windowTitle.c_str());
#if defined(PLATFORM_DESKTOP)
        if (m_hostMonitorIndex.has_value())
        {
            const int monitorIndex = *m_hostMonitorIndex;
            const int monitorCount = GetMonitorCount();
            if (monitorIndex < monitorCount)
            {
                SetWindowMonitor(monitorIndex);
                const char *const monitorName =
                    GetMonitorName(monitorIndex);
                std::cout
                    << "Host monitor: " << monitorIndex;
                if (monitorName && monitorName[0] != '\0')
                {
                    std::cout << " (" << monitorName << ")";
                }
                std::cout
                    << ", " << GetMonitorWidth(monitorIndex)
                    << "x" << GetMonitorHeight(monitorIndex)
                    << std::endl;
            }
            else
            {
                std::cerr
                    << "Requested host monitor " << monitorIndex
                    << " is unavailable (" << monitorCount
                    << " connected); retaining default window placement"
                    << std::endl;
            }
        }
#endif
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
#endif
#if defined(PLATFORM_VITA)
        SetTargetFPS(60);
#else
        // The desktop runtime applies a blocking steady-clock deadline after
        // EndDrawing. Raylib's default limiter spins through the final 5% of
        // every wait, which needlessly consumes host CPU while the guest
        // executor is the bottleneck.
        SetTargetFPS(0);
#endif
        setRealtimeVSyncPacingEnabled(true);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
        m_debugServer = std::make_unique<PS2DebugServer>(*this);
        (void)m_debugServer->start();
#endif
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_memory.installPostBiosTlbState();
    m_cpuContext.resetPostBiosTlbRegisters();

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;
    std::vector<ElfAddressRange> executableLoadRanges;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max() && execEnd > ph.vaddr)
            {
                executableLoadRanges.push_back({
                    ph.vaddr,
                    static_cast<uint32_t>(execEnd),
                });
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    mergeAddressRanges(executableLoadRanges);
    std::vector<ElfAddressRange> executableSectionRanges;
    const bool hasExecutableSectionRanges =
        readExecutableSectionRanges(file,
                                    static_cast<uint64_t>(fileSize),
                                    header,
                                    executableLoadRanges,
                                    executableSectionRanges);
    const std::vector<ElfAddressRange> &codeRanges =
        hasExecutableSectionRanges ? executableSectionRanges : executableLoadRanges;
    for (const ElfAddressRange &range : codeRanges)
    {
        m_memory.registerCodeRegion(range.start, range.end);
    }
    RUNTIME_LOG("Registered " << codeRanges.size()
                              << (hasExecutableSectionRanges
                                      ? " executable ELF section range(s)"
                                      : " executable ELF segment fallback range(s)")
                              << std::endl);

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string iopError;
    if (!m_iopSubsystem->configure(identity, &iopError))
    {
        std::cerr << "[ps2xIOP] failed to configure profile: " << iopError << std::endl;
        return false;
    }

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

PS2Runtime::IoPaths PS2Runtime::ioPaths() const
{
    std::lock_guard<std::mutex> lock(m_ioPathsMutex);
    return m_ioPaths;
}

void PS2Runtime::configureIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    std::lock_guard<std::mutex> lock(m_ioPathsMutex);
    m_ioPaths = std::move(normalized);
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = ioPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    configureIoPaths(paths);
}

namespace
{
    constexpr size_t kMaxRecompiledModules = 32u;
    std::array<
        std::atomic<const PS2RecompiledModuleDescriptor *>,
        kMaxRecompiledModules>
        g_recompiledModules{};
    std::atomic<uint32_t> g_recompiledModuleHighWatermark{0u};
    constexpr uint64_t kEmptyRecompiledModuleAddressEnvelope =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) << 32u;
    std::atomic<uint64_t> g_recompiledModuleAddressEnvelope{
        kEmptyRecompiledModuleAddressEnvelope};

    uint64_t recompiledModuleAddressEnvelope(
        uint32_t firstAddress,
        uint32_t lastAddress) noexcept
    {
        return (static_cast<uint64_t>(firstAddress) << 32u) |
               lastAddress;
    }

    void publishRecompiledModuleBounds(
        uint32_t slotCount,
        uint32_t firstAddress,
        uint32_t lastAddress) noexcept
    {
        uint32_t observedCount =
            g_recompiledModuleHighWatermark.load(
                std::memory_order_relaxed);
        while (observedCount < slotCount &&
               !g_recompiledModuleHighWatermark.compare_exchange_weak(
                   observedCount,
                   slotCount,
                   std::memory_order_release,
                   std::memory_order_relaxed))
        {
        }

        uint64_t observedEnvelope =
            g_recompiledModuleAddressEnvelope.load(
                std::memory_order_relaxed);
        while (true)
        {
            const uint32_t observedFirst =
                static_cast<uint32_t>(observedEnvelope >> 32u);
            const uint32_t observedLast =
                static_cast<uint32_t>(observedEnvelope);
            const uint64_t expandedEnvelope =
                recompiledModuleAddressEnvelope(
                    std::min(observedFirst, firstAddress),
                    std::max(observedLast, lastAddress));
            if (expandedEnvelope == observedEnvelope ||
                g_recompiledModuleAddressEnvelope.compare_exchange_weak(
                    observedEnvelope,
                    expandedEnvelope,
                    std::memory_order_release,
                    std::memory_order_relaxed))
            {
                return;
            }
        }
    }

    uint32_t normalizeGuestFunctionAddress(uint32_t address)
    {
        uint32_t offset = 0u;
        if (ps2ResolveDirectRdramOffset(address, offset))
        {
            return offset;
        }
        return address;
    }

    bool generatedFunctionTableSlotNormalized(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }

    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        return generatedFunctionTableSlotNormalized(
            normalizeGuestFunctionAddress(address), slot);
    }

    PS2Runtime::RecompiledFunctionPair generatedFunctionAtNormalizedAddress(
        uint32_t address)
    {
        uint32_t slot = 0u;
        if (!generatedFunctionTableSlotNormalized(address, slot))
        {
            return {};
        }
        return g_ps2RecompiledFunctionTable[slot];
    }

    const uint8_t *recompiledModuleMatchData(
        const PS2RecompiledModuleDescriptor &module,
        const uint8_t *rdram)
    {
        if (rdram == nullptr)
        {
            return nullptr;
        }

        uint32_t offset = 0u;
        if (!ps2ResolveDirectRdramOffset(module.matchAddress, offset) ||
            offset > PS2_RAM_SIZE ||
            module.matchSize > PS2_RAM_SIZE - offset)
        {
            return nullptr;
        }
        return rdram + offset;
    }

    bool recompiledModuleMatches(
        const PS2RecompiledModuleDescriptor &module,
        const uint8_t *rdram)
    {
        const uint8_t *matchData =
            recompiledModuleMatchData(module, rdram);
        if (matchData == nullptr)
        {
            return false;
        }
        return std::memcmp(
                   matchData,
                   module.matchBytes,
                   module.matchSize) == 0;
    }

    PS2Runtime::RecompiledFunctionPair recompiledModuleFunction(
        const PS2RecompiledModuleDescriptor &module,
        uint32_t address)
    {
        if (address < module.functions[0].address ||
            address > module.functions[module.functionCount - 1u].address)
        {
            return {};
        }

        uint32_t first = 0u;
        uint32_t last = module.functionCount;
        while (first < last)
        {
            const uint32_t middle = first + ((last - first) / 2u);
            if (module.functions[middle].address < address)
            {
                first = middle + 1u;
            }
            else
            {
                last = middle;
            }
        }
        if (first < module.functionCount &&
            module.functions[first].address == address)
        {
            return module.functions[first].functions;
        }
        return {};
    }

    PS2Runtime::RecompiledFunctionPair activeRecompiledModuleFunction(
        uint32_t address,
        const uint8_t *rdram)
    {
        const uint64_t envelope =
            g_recompiledModuleAddressEnvelope.load(
                std::memory_order_acquire);
        const uint32_t firstAddress =
            static_cast<uint32_t>(envelope >> 32u);
        const uint32_t lastAddress =
            static_cast<uint32_t>(envelope);
        if (address < firstAddress || address > lastAddress)
        {
            return {};
        }

        const uint32_t moduleCount =
            g_recompiledModuleHighWatermark.load(
                std::memory_order_acquire);
        for (uint32_t index = 0u; index < moduleCount; ++index)
        {
            const PS2RecompiledModuleDescriptor *module =
                g_recompiledModules[index].load(
                    std::memory_order_acquire);
            if (module == nullptr)
            {
                continue;
            }

            constexpr size_t kMatchPrefixSize = sizeof(uint32_t);
            const uint8_t *matchData =
                recompiledModuleMatchData(*module, rdram);
            if (matchData == nullptr ||
                std::memcmp(
                    matchData,
                    module->matchBytes,
                    kMatchPrefixSize) != 0)
            {
                continue;
            }

            PS2Runtime::RecompiledFunctionPair functions =
                recompiledModuleFunction(*module, address);
            if (!functions.empty() &&
                (module->matchSize == kMatchPrefixSize ||
                 std::memcmp(
                     matchData + kMatchPrefixSize,
                     module->matchBytes + kMatchPrefixSize,
                     module->matchSize - kMatchPrefixSize) == 0))
            {
                return functions;
            }
        }
        return {};
    }

    PS2Runtime::RecompiledFunctionPair activeRecompiledFunction(
        uint32_t address,
        const uint8_t *rdram)
    {
        PS2Runtime::RecompiledFunctionPair moduleFunctions =
            activeRecompiledModuleFunction(address, rdram);
        return !moduleFunctions.empty()
                   ? moduleFunctions
                   : generatedFunctionAtNormalizedAddress(address);
    }

    bool activeRecompiledModuleCodeAddress(
        uint32_t address,
        const uint8_t *rdram)
    {
        const uint32_t moduleCount =
            g_recompiledModuleHighWatermark.load(
                std::memory_order_acquire);
        for (uint32_t index = 0u; index < moduleCount; ++index)
        {
            const PS2RecompiledModuleDescriptor *module =
                g_recompiledModules[index].load(
                    std::memory_order_acquire);
            if (module == nullptr ||
                !recompiledModuleMatches(*module, rdram))
            {
                continue;
            }
            for (uint32_t index = 0u; index < module->symbolCount; ++index)
            {
                const PS2GuestFunctionSymbol &symbol = module->symbols[index];
                if (address >= symbol.start && address < symbol.end)
                {
                    return true;
                }
            }
        }
        return false;
    }

    const PS2GuestFunctionSymbol *findGuestFunctionSymbol(uint32_t address)
    {
        address = normalizeGuestFunctionAddress(address);
        uint32_t first = 0u;
        uint32_t last = g_ps2GuestFunctionSymbolCount;
        while (first < last)
        {
            const uint32_t middle = first + ((last - first) / 2u);
            if (g_ps2GuestFunctionSymbols[middle].start <= address)
            {
                first = middle + 1u;
            }
            else
            {
                last = middle;
            }
        }

        if (first == 0u)
        {
            return nullptr;
        }

        const PS2GuestFunctionSymbol &symbol = g_ps2GuestFunctionSymbols[first - 1u];
        if (address < symbol.end && symbol.name != nullptr && symbol.name[0] != '\0')
        {
            return &symbol;
        }
        return nullptr;
    }
}

bool ps2RegisterRecompiledModule(
    const PS2RecompiledModuleDescriptor *descriptor) noexcept
{
    if (descriptor == nullptr ||
        descriptor->abiVersion !=
            PS2_RECOMPILED_FUNCTION_PAIR_ABI_VERSION ||
        descriptor->name == nullptr ||
        descriptor->name[0] == '\0' || descriptor->matchBytes == nullptr ||
        descriptor->matchSize < 4u || descriptor->matchSize > 4096u ||
        descriptor->functions == nullptr || descriptor->functionCount == 0u ||
        (descriptor->symbolCount != 0u && descriptor->symbols == nullptr))
    {
        return false;
    }

    uint32_t matchOffset = 0u;
    if (!ps2ResolveDirectRdramOffset(descriptor->matchAddress, matchOffset) ||
        matchOffset > PS2_RAM_SIZE ||
        descriptor->matchSize > PS2_RAM_SIZE - matchOffset)
    {
        return false;
    }

    uint32_t previousAddress = 0u;
    for (uint32_t index = 0u; index < descriptor->functionCount; ++index)
    {
        const PS2RecompiledModuleFunction &entry = descriptor->functions[index];
        if (!entry.functions.complete() || (entry.address & 3u) != 0u ||
            entry.address >= PS2_RAM_SIZE ||
            (index != 0u && entry.address <= previousAddress))
        {
            return false;
        }
        previousAddress = entry.address;
    }

    for (uint32_t index = 0u; index < descriptor->symbolCount; ++index)
    {
        const PS2GuestFunctionSymbol &symbol = descriptor->symbols[index];
        if (symbol.end <= symbol.start || symbol.end > PS2_RAM_SIZE ||
            (index != 0u &&
             symbol.start <= descriptor->symbols[index - 1u].start))
        {
            return false;
        }
    }

    for (auto &slot : g_recompiledModules)
    {
        const PS2RecompiledModuleDescriptor *existing =
            slot.load(std::memory_order_acquire);
        if (existing == descriptor)
        {
            return true;
        }
        if (existing != nullptr &&
            std::strcmp(existing->name, descriptor->name) == 0)
        {
            return false;
        }
    }

    for (uint32_t index = 0u;
         index < g_recompiledModules.size();
         ++index)
    {
        const PS2RecompiledModuleDescriptor *expected = nullptr;
        if (g_recompiledModules[index].compare_exchange_strong(
                expected,
                descriptor,
                std::memory_order_release,
                std::memory_order_relaxed))
        {
            publishRecompiledModuleBounds(
                index + 1u,
                descriptor->functions[0].address,
                descriptor->functions[
                    descriptor->functionCount - 1u].address);
            return true;
        }
    }
    return false;
}

std::string PS2Runtime::formatGuestPc(uint32_t address)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << address;

    const PS2GuestFunctionSymbol *symbol = findGuestFunctionSymbol(address);
    if (symbol != nullptr)
    {
        const uint32_t normalizedAddress = normalizeGuestFunctionAddress(address);
        const uint32_t offset = normalizedAddress - symbol->start;
        oss << '<' << symbol->name;
        if (offset != 0u)
        {
            oss << "+0x" << offset;
        }
        oss << '>';
    }

    return oss.str();
}

bool PS2Runtime::replaceFunction(
    uint32_t address,
    RecompiledFunctionPair functions)
{
    if (!functions.empty() && !functions.complete())
    {
        std::cerr
            << "[function-table] cannot install an incomplete fast/precise pair at guest PC 0x"
            << std::hex << address << std::dec << std::endl;
        return false;
    }

    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = functions;
    return true;
}

bool PS2Runtime::replaceFunction(
    uint32_t address,
    RecompiledFunction func)
{
    return replaceFunction(
        address,
        RecompiledFunctionPair::identical(func));
}

bool PS2Runtime::registerFunction(
    uint32_t address,
    RecompiledFunctionPair functions)
{
    return replaceFunction(address, functions);
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    return !activeRecompiledFunction(
                normalizeGuestFunctionAddress(address),
                m_memory.getRDRAM())
                .empty();
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

EeArchitecturalObservationMode
PS2Runtime::architecturalObservationMode(
    const R5900Context *ctx) noexcept
{
    constexpr uint32_t kBreakpointEnableMask =
        COP0_BPC_IAE | COP0_BPC_DRE | COP0_BPC_DWE;
    constexpr uint32_t kPerformanceCounterEnable = 1u << 31u;
    if (ctx != nullptr &&
        (ctx->cop0_bpc & kBreakpointEnableMask) == 0u &&
        (ctx->cop0_perf & kPerformanceCounterEnable) == 0u)
    {
        return EeArchitecturalObservationMode::Fast;
    }
    return EeArchitecturalObservationMode::Precise;
}

bool PS2Runtime::completeEeObservationModeTransition(
    uint8_t *rdram,
    R5900Context *ctx,
    EeArchitecturalObservationMode previousMode)
{
    if (architecturalObservationMode(ctx) == previousMode)
    {
        return false;
    }

    // Publish the unwind before fetch validation or event service. Either can
    // enter an exception or callback; in both cases no older specialization
    // may resume after that nested work completes.
    m_guestExecutionDispatcherExitEpoch.fetch_add(
        1u, std::memory_order_release);
    ValidateInstructionFetch(ctx, ctx->pc);
    serviceEeEventsAtBlockBoundary(rdram, ctx);
    return true;
}

bool PS2Runtime::completeEeObservationModeTransitionAtGuestBranch(
    uint8_t *rdram,
    R5900Context *ctx,
    EeArchitecturalObservationMode previousMode,
    GuestBranchKind kind)
{
    if (architecturalObservationMode(ctx) == previousMode)
    {
        return false;
    }

    m_guestExecutionDispatcherExitEpoch.fetch_add(
        1u, std::memory_order_release);
    (void)ValidateGuestBranchTarget(ctx, ctx->pc, kind);
    serviceEeEventsAtBlockBoundary(rdram, ctx);
    return true;
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(
    uint32_t address)
{
    return lookupFunction(address, nullptr);
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(
    uint32_t address,
    const R5900Context *ctx)
{
    pushDispatchPc(address);

    const uint32_t normalizedAddress = normalizeGuestFunctionAddress(address);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugControlActive.load(std::memory_order_acquire))
    {
        debugRecordBranch(normalizedAddress);
    }
#endif

    const EeArchitecturalObservationMode mode =
        architecturalObservationMode(ctx);
    RecompiledFunction fn =
        activeRecompiledFunction(normalizedAddress, m_memory.getRDRAM())
            .select(mode);
    if (fn != nullptr)
    {
        if (normalizedAddress != address)
        {
            static RecompiledFunction directMappedAlias =
                [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
            {
                if (!ctx || !runtime)
                {
                    return;
                }

                ctx->pc = normalizeGuestFunctionAddress(ctx->pc);
                RecompiledFunction target =
                    runtime->lookupFunction(ctx->pc, ctx);
                target(rdram, ctx, runtime);
            };
            return directMappedAlias;
        }
        return fn;
    }

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
    debugClearFault();
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    (void)rdram;
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc =
        ctx ? ctx->pc : m_debugPc.load(std::memory_order_relaxed);
    const uint32_t ra =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0))
            : m_debugRa.load(std::memory_order_relaxed);
    const uint32_t sp =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0))
            : m_debugSp.load(std::memory_order_relaxed);
    const uint32_t gp =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0))
            : m_debugGp.load(std::memory_order_relaxed);
    const uint32_t a0 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0)) : 0u;
    const uint32_t a1 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0)) : 0u;
    const uint32_t v0 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0)) : 0u;
    const uint32_t v1 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0)) : 0u;

    if (firstReport)
    {
        DebugFaultInfo fault{};
        fault.active = true;
        fault.type = "missing-recompiled-target";
        fault.operation = debugName ? debugName : "";
        fault.branchKind = kind;
        fault.sourcePc = sourcePc;
        fault.targetPc = targetPc;
        fault.pc = pc;
        fault.ra = ra;
        fault.sp = sp;
        fault.gp = gp;
        fault.a0 = a0;
        fault.a1 = a1;
        fault.v0 = v0;
        fault.v1 = v1;
        fault.codeRegion =
            m_memory.isCodeAddress(targetPc) ||
            activeRecompiledModuleCodeAddress(
                normalizeGuestFunctionAddress(targetPc),
                m_memory.getRDRAM());
        fault.policy = policy;
        debugRecordFault(fault);
        fault = debugFaultSnapshot();

        std::ostringstream oss;
        oss << "{\"schema_version\":1,"
            << "\"event\":\"missing-recompiled-target\","
            << "\"sequence\":" << fault.sequence << ','
            << "\"kind\":\"" << describeGuestBranchKind(kind) << "\","
            << "\"operation\":\""
            << escapeJsonString(debugName ? debugName : "") << "\","
            << "\"source\":\"" << rawGuestAddress(sourcePc) << "\","
            << "\"target\":\"" << rawGuestAddress(targetPc) << "\","
            << "\"pc\":\"" << rawGuestAddress(pc) << "\","
            << "\"ra\":\"" << rawGuestAddress(ra) << "\","
            << "\"sp\":\"" << rawGuestAddress(sp) << "\","
            << "\"gp\":\"" << rawGuestAddress(gp) << "\","
            << "\"code_region\":" << (fault.codeRegion ? "true" : "false")
            << ",\"policy\":" << static_cast<uint32_t>(policy)
            << ",\"trace_entries\":"
            << debugBranchHistory(kDebugBranchHistoryCapacity).size()
            << '}';

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

bool PS2Runtime::ValidateGuestBranchTarget(
    R5900Context *ctx,
    uint32_t targetPc,
    GuestBranchKind kind)
{
    // Host-invoked asynchronous callbacks use a zero return address as the
    // boundary back into the runtime. It is not a guest instruction fetch,
    // so consume it before architectural address translation. Keep the
    // exception narrowly scoped to the callback context that is currently
    // bound; an ordinary guest return to zero must still fault normally.
    if (ctx &&
        kind == GuestBranchKind::Return &&
        targetPc == 0u &&
        m_asyncCallbackInvocationDepth != 0u &&
        m_boundEeContext == ctx)
    {
        ctx->pc = 0u;
        return false;
    }

    ctx->pc = targetPc;
    ValidateInstructionFetch(ctx, targetPc);
    return true;
}

bool PS2Runtime::ValidateGuestBranchTargetWithoutObservation(
    R5900Context *ctx,
    uint32_t targetPc,
    GuestBranchKind kind)
{
    if (ctx &&
        kind == GuestBranchKind::Return &&
        targetPc == 0u &&
        m_asyncCallbackInvocationDepth != 0u &&
        m_boundEeContext == ctx)
    {
        ctx->pc = 0u;
        return false;
    }

    ctx->pc = targetPc;
    ValidateInstructionFetchWithoutObservation(ctx, targetPc);
    return true;
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    if (!ValidateGuestBranchTarget(ctx, targetPc, kind))
    {
        return false;
    }

    return dispatchValidatedGuestBranch(
        rdram,
        ctx,
        targetPc,
        sourcePc,
        fallthroughPc,
        kind,
        debugName,
        architecturalObservationMode(ctx));
}

bool PS2Runtime::dispatchValidatedGuestBranch(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t targetPc,
    uint32_t sourcePc,
    uint32_t fallthroughPc,
    GuestBranchKind kind,
    const char *debugName,
    EeArchitecturalObservationMode mode)
{
    targetPc = normalizeGuestFunctionAddress(targetPc);
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);
    RecompiledFunction targetFn =
        activeRecompiledFunction(targetPc, rdram)
            .select(mode);

    if (kind == GuestBranchKind::Return)
    {
        if (targetFn == nullptr)
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        // Prevent nested dispatch.
        ctx->pc = targetPc;
        return false;
    }

    if (targetFn == nullptr)
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            return true;
        }

        return false;
    }

    pushDispatchPc(targetPc);
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugControlActive.load(std::memory_order_acquire))
    {
        debugRecordBranch(targetPc);
    }
#endif
    const uint32_t entryPc = ctx->pc;
    const uint64_t dispatcherExitEpoch =
        m_guestExecutionDispatcherExitEpoch.load(
            std::memory_order_acquire);
    targetFn(rdram, ctx, this);

    if (isStopRequested() || ctx->pc == 0u)
    {
        return false;
    }

    // A stackless checkpoint returns generated code to the outer dispatcher.
    // In a nested call, the resume PC can legitimately equal the callee
    // entry, which is otherwise also how an empty implicit-return handler is
    // represented. Preserve the resume PC instead of converting it to the
    // call fallthrough when any nested target requested a dispatcher exit.
    // A stackful checkpoint suspends and resumes in place, so it does not
    // advance this epoch or unwind native callers.
    if (m_guestExecutionDispatcherExitEpoch.load(
            std::memory_order_acquire) !=
        dispatcherExitEpoch)
    {
        return false;
    }

    if (!isCall)
    {
        return false;
    }

    if (ctx->pc == entryPc)
    {
        ctx->pc = fallthroughPc;
    }

    return ctx->pc == fallthroughPc;
}

[[noreturn]] void PS2Runtime::raiseCop0Level1Exception(
    R5900Context *ctx,
    uint32_t exceptionCode,
    bool commitContextProgress,
    bool useTlbRefillVector)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }

    flushEeInstructionIssue(ctx);

    const ps2x::timing::EeTick now =
        commitContextProgress
            ? commitEeContextProgress(ctx)
            : currentEeTick();
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    synchronizeCop0Timing(ctx, cycle);

    // Performance counter exceptions outrank every level-1 exception. An
    // occurrence counter can overflow between scheduler boundaries, so honor
    // the sticky state before publishing the lower-priority exception.
    if (m_cop0Timing.performanceExceptionPending(
            ctx->cop0_status))
    {
        raiseCop0PerformanceException(ctx);
    }

    const bool alreadyInException =
        (ctx->cop0_status &
         COP0_STATUS_EXL) != 0u;
    if (!alreadyInException &&
        ctx->in_delay_slot)
    {
        ctx->cop0_epc = ctx->branch_pc;
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~COP0_CAUSE_EXCCODE_MASK) |
            ((exceptionCode << 2u) &
             COP0_CAUSE_EXCCODE_MASK) |
            COP0_CAUSE_BD;
    }
    else if (!alreadyInException)
    {
        ctx->cop0_epc = ctx->pc;
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~(COP0_CAUSE_EXCCODE_MASK |
               COP0_CAUSE_BD)) |
            ((exceptionCode << 2u) &
             COP0_CAUSE_EXCCODE_MASK);
    }
    else
    {
        // Nested level-1 exceptions update the cause but preserve the
        // original restart address and branch-delay state.
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~COP0_CAUSE_EXCCODE_MASK) |
            ((exceptionCode << 2u) &
             COP0_CAUSE_EXCCODE_MASK);
    }

    ctx->cop0_status |= COP0_STATUS_EXL;
    ctx->pc = selectExceptionVector(
        ctx, exceptionCode,
        alreadyInException,
        useTlbRefillVector);
    ctx->in_delay_slot = false;
    refreshCop0EventSchedules(ctx, cycle);
    throw PS2GuestException{};
}

[[noreturn]] void
PS2Runtime::raiseCop0PerformanceException(
    R5900Context *ctx)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }

    flushEeInstructionIssue(ctx);

    const ps2x::timing::EeTick now =
        currentEeTick();
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    synchronizeCop0Timing(ctx, cycle);

    if (ctx->in_delay_slot)
    {
        ctx->cop0_errorepc =
            ctx->branch_pc;
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~(COP0_CAUSE_EXC2_MASK |
               COP0_CAUSE_BD2)) |
            COP0_CAUSE_EXC2_PERFORMANCE |
            COP0_CAUSE_BD2;
    }
    else
    {
        ctx->cop0_errorepc = ctx->pc;
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~(COP0_CAUSE_EXC2_MASK |
               COP0_CAUSE_BD2)) |
            COP0_CAUSE_EXC2_PERFORMANCE;
    }

    ctx->cop0_status |= COP0_STATUS_ERL;
    ctx->pc =
        (ctx->cop0_status &
         COP0_STATUS_DEV) != 0u
            ? 0xBFC00280u
            : 0x80000080u;
    ctx->in_delay_slot = false;
    refreshCop0EventSchedules(ctx, cycle);
    throw PS2GuestException{};
}

[[noreturn]] void
PS2Runtime::raiseCop0DebugException(
    R5900Context *ctx,
    bool dataBreakpoint)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }

    flushEeInstructionIssue(ctx);

    const ps2x::timing::EeTick now =
        commitEeContextProgress(ctx);
    const uint64_t cycle =
        ps2x::timing::eeTickToCyclesFloor(now);
    synchronizeCop0Timing(ctx, cycle);

    // Performance overflow precedes both debug classes. Enabled interrupts
    // precede data breakpoints but follow instruction breakpoints.
    if (m_cop0Timing.performanceExceptionPending(
            ctx->cop0_status))
    {
        raiseCop0PerformanceException(ctx);
    }
    if (dataBreakpoint &&
        cop0TimerInterruptEnabled(ctx))
    {
        raiseCop0Level1Exception(
            ctx, EXCEPTION_INTERRUPT, false);
    }

    if (ctx->in_delay_slot)
    {
        ctx->cop0_errorepc =
            ctx->branch_pc;
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~(COP0_CAUSE_EXC2_MASK |
               COP0_CAUSE_BD2)) |
            COP0_CAUSE_EXC2_DEBUG |
            COP0_CAUSE_BD2;
    }
    else
    {
        ctx->cop0_errorepc = ctx->pc;
        ctx->cop0_cause =
            (ctx->cop0_cause &
             ~(COP0_CAUSE_EXC2_MASK |
               COP0_CAUSE_BD2)) |
            COP0_CAUSE_EXC2_DEBUG;
    }

    ctx->cop0_status |= COP0_STATUS_ERL;
    ctx->pc =
        (ctx->cop0_status &
         COP0_STATUS_DEV) != 0u
            ? 0xBFC00300u
            : 0x80000100u;
    ctx->in_delay_slot = false;
    refreshCop0EventSchedules(ctx, cycle);
    throw PS2GuestException{};
}

[[noreturn]] void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
    }

    raiseCop0Level1Exception(
        ctx, static_cast<uint32_t>(exception));
}

void PS2Runtime::RequireCoprocessorUsable(
    R5900Context *ctx,
    uint32_t coprocessor)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }
    if (coprocessor > 3u)
    {
        throw std::out_of_range("EE coprocessor number must fit Cause.CE");
    }

    const bool kernelMode =
        (ctx->cop0_status &
         (COP0_STATUS_EXL |
          COP0_STATUS_ERL)) != 0u ||
        (ctx->cop0_status & COP0_STATUS_KSU_MASK) == 0u;
    const bool enabled =
        (coprocessor == 0u && kernelMode) ||
        (ctx->cop0_status &
         (1u << (28u + coprocessor))) != 0u;
    if (enabled)
    {
        return;
    }

    ctx->cop0_cause =
        (ctx->cop0_cause & ~COP0_CAUSE_CE_MASK) |
        ((coprocessor << 28u) & COP0_CAUSE_CE_MASK);
    SignalException(
        ctx,
        EXCEPTION_COPROCESSOR_UNUSABLE);
}

[[noreturn]] void PS2Runtime::SignalMemoryException(R5900Context *ctx,
                                                    PS2Exception exception,
                                                    uint32_t badVAddr)
{
    ctx->cop0_badvaddr = badVAddr;

    if (exception == EXCEPTION_TLB_MODIFIED ||
        exception == EXCEPTION_TLB_REFILL_LOAD ||
        exception == EXCEPTION_TLB_REFILL_STORE)
    {
        ctx->cop0_context = (ctx->cop0_context & 0xFF800000u) |
                            ((badVAddr >> 9u) & 0x007FFFF0u);
        ctx->cop0_entryhi = (badVAddr & 0xFFFFE000u) |
                            (ctx->cop0_entryhi & 0x000000FFu);
    }

    SignalException(ctx, exception);
}

[[noreturn]] void PS2Runtime::raiseTlbFault(
    R5900Context *ctx,
    const PS2TlbFaultException &fault,
    bool storeAccess)
{
    const uint32_t badVAddr = fault.virtualAddress();
    ctx->cop0_badvaddr = badVAddr;
    ctx->cop0_context =
        (ctx->cop0_context & 0xff800000u) |
        ((badVAddr >> 9u) & 0x007ffff0u);
    ctx->cop0_entryhi =
        (badVAddr & 0xffffe000u) |
        (ctx->cop0_entryhi & 0x000000ffu);

    uint32_t exceptionCode =
        storeAccess
            ? EXCEPTION_TLB_REFILL_STORE
            : EXCEPTION_TLB_REFILL_LOAD;
    if (fault.kind() == PS2TlbFaultKind::Modified)
    {
        exceptionCode = EXCEPTION_TLB_MODIFIED;
    }

    raiseCop0Level1Exception(
        ctx,
        exceptionCode,
        true,
        fault.kind() == PS2TlbFaultKind::Refill);
}

void PS2Runtime::ValidateInstructionFetch(
    R5900Context *ctx,
    uint32_t virtualAddress)
{
    CheckEeInstructionBreakpoint(
        ctx, virtualAddress);

    ValidateInstructionFetchWithoutObservation(
        ctx, virtualAddress);
}

void PS2Runtime::ValidateInstructionFetchWithoutObservation(
    R5900Context *ctx,
    uint32_t virtualAddress)
{
    if (ctx && (virtualAddress & 3u) == 0u)
    {
        if (m_memory.tryReuseValidatedInstructionFetchRdramPage(
                eeInstructionFetchPageKey(
                    ctx, virtualAddress))) [[likely]]
        {
            return;
        }
    }

    validateInstructionFetchPageMissWithoutObservation(
        ctx, virtualAddress);
}

void PS2Runtime::validateInstructionFetchPageMissWithoutObservation(
    R5900Context *ctx,
    uint32_t virtualAddress)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }

    if ((virtualAddress & 3u) != 0u)
    {
        ctx->pc = virtualAddress;
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            virtualAddress);
    }

    const uint32_t translationPageKey =
        eeInstructionFetchPageKey(
            ctx, virtualAddress);
    constexpr uint32_t kFetchSize =
        sizeof(uint32_t);

    uint32_t physicalOffset = 0u;
    if (Ps2ResolveFastGuestRdramOffset(
            m_memory,
            guestAddressTranslation(ctx),
            virtualAddress,
            kFetchSize,
            false,
            physicalOffset))
    {
        m_memory.rememberValidatedInstructionFetchRdramPage(
            translationPageKey);
        return;
    }

    try
    {
        const uint32_t physicalAddress =
            m_memory.translateAddress(
                virtualAddress,
                guestAddressTranslation(ctx),
                false,
                kFetchSize);
        if (physicalAddress <=
            PS2_RAM_SIZE - kFetchSize)
        {
            m_memory.rememberValidatedInstructionFetchRdramPage(
                translationPageKey);
        }
    }
    catch (const PS2AddressErrorException &fault)
    {
        ctx->pc = fault.virtualAddress();
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        ctx->pc = fault.virtualAddress();
        raiseTlbFault(ctx, fault, false);
    }
}

void PS2Runtime::CheckEeInstructionBreakpoint(
    R5900Context *ctx,
    uint32_t virtualAddress)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }

    const uint32_t bpc = ctx->cop0_bpc;
    if ((bpc & COP0_BPC_IAE) == 0u ||
        !eeBreakpointModeEnabled(
            ctx->cop0_status, bpc, true) ||
        (virtualAddress & ctx->cop0_iabm) !=
            (ctx->cop0_iab & ctx->cop0_iabm))
    {
        return;
    }

    ctx->cop0_bpc |= COP0_BPC_IAB;
    if ((bpc & COP0_BPC_BED) != 0u)
    {
        return;
    }

    ctx->pc = virtualAddress;
    raiseCop0DebugException(ctx, false);
}

bool PS2Runtime::EeDataBreakpointEnabled(
    const R5900Context *ctx,
    bool write) const noexcept
{
    if (!ctx)
    {
        return false;
    }
    const uint32_t bpc = ctx->cop0_bpc;
    const uint32_t directionEnable =
        write ? COP0_BPC_DWE
              : COP0_BPC_DRE;
    return (bpc & directionEnable) != 0u &&
           eeBreakpointModeEnabled(
               ctx->cop0_status, bpc, false);
}

bool PS2Runtime::CheckEeDataAddressBreakpoint(
    R5900Context *ctx,
    uint32_t virtualAddress,
    bool write)
{
    if (!EeDataBreakpointEnabled(ctx, write) ||
        (virtualAddress & ctx->cop0_dabm) !=
            (ctx->cop0_dab & ctx->cop0_dabm))
    {
        return false;
    }

    if ((ctx->cop0_bpc & COP0_BPC_DVE) != 0u)
    {
        return true;
    }

    ctx->cop0_bpc |=
        write ? COP0_BPC_DWB
              : COP0_BPC_DRB;
    if ((ctx->cop0_bpc & COP0_BPC_BED) == 0u)
    {
        raiseCop0DebugException(ctx, true);
    }
    return false;
}

void PS2Runtime::CheckEeDataValueBreakpoint(
    R5900Context *ctx,
    uint32_t virtualAddress,
    uint32_t value,
    bool write)
{
    if (!EeDataBreakpointEnabled(ctx, write) ||
        (ctx->cop0_bpc & COP0_BPC_DVE) == 0u ||
        (virtualAddress & ctx->cop0_dabm) !=
            (ctx->cop0_dab & ctx->cop0_dabm) ||
        (value & ctx->cop0_dvbm) !=
            (ctx->cop0_dvb & ctx->cop0_dvbm))
    {
        return;
    }

    ctx->cop0_bpc |=
        write ? COP0_BPC_DWB
              : COP0_BPC_DRB;
    if ((ctx->cop0_bpc & COP0_BPC_BED) == 0u)
    {
        raiseCop0DebugException(ctx, true);
    }
}

void PS2Runtime::recordEeInstructionIssue(
    R5900Context *ctx,
    ps2x::performance::EeInstructionIssueProfile profile)
{
    if (!ctx)
    {
        return;
    }

    if ((ctx->cop0_perf & (1u << 31u)) == 0u ||
        (ctx->cop0_status & COP0_STATUS_ERL) != 0u)
    {
        ctx->ee_performance_issue_pending = false;
        ctx->ee_performance_pending_issue = {};
        return;
    }

    using namespace ps2x::performance;
    const auto recordIssue =
        [this, ctx](bool dual)
        {
            ps2x::timing::Cop0PerformanceEvents events{};
            if (dual)
            {
                events.dualInstructionIssued = 1u;
            }
            else
            {
                events.singleInstructionIssued = 1u;
            }
            recordEePerformanceEvents(ctx, events);
        };

    const bool dualIssueEnabled =
        (ctx->cop0_config & (1u << 18u)) != 0u;
    if (!dualIssueEnabled)
    {
        flushEeInstructionIssue(ctx);
        recordIssue(false);
        return;
    }

    if (!ctx->ee_performance_issue_pending)
    {
        ctx->ee_performance_pending_issue = profile;
        ctx->ee_performance_issue_pending = true;
        return;
    }

    const auto &previous =
        ctx->ee_performance_pending_issue;
    const bool distinctPipes =
        ((previous.pipeMask & kIssuePipe0) != 0u &&
         (profile.pipeMask & kIssuePipe1) != 0u) ||
        ((previous.pipeMask & kIssuePipe1) != 0u &&
         (profile.pipeMask & kIssuePipe0) != 0u);
    const bool forbiddenControlPair =
        ((previous.flags & kIssueBranch) != 0u &&
         (profile.flags &
          (kIssueBranch | kIssueEret)) != 0u);
    const bool conservativeBarrier =
        ((previous.flags | profile.flags) &
         kIssueConservativeBarrier) != 0u;
    const bool registerHazard =
        (previous.gprWriteMask &
         (profile.gprReadMask |
          profile.gprWriteMask)) != 0u;

    if (distinctPipes &&
        !forbiddenControlPair &&
        !conservativeBarrier &&
        !registerHazard)
    {
        recordIssue(true);
        ctx->ee_performance_issue_pending = false;
        ctx->ee_performance_pending_issue = {};
        return;
    }

    recordIssue(false);
    ctx->ee_performance_pending_issue = profile;
}

void PS2Runtime::flushEeInstructionIssue(
    R5900Context *ctx) noexcept
{
    if (!ctx || !ctx->ee_performance_issue_pending)
    {
        return;
    }

    ps2x::timing::Cop0PerformanceEvents events{};
    events.singleInstructionIssued = 1u;
    recordEePerformanceEvents(ctx, events);
    ctx->ee_performance_issue_pending = false;
    ctx->ee_performance_pending_issue = {};
}

void PS2Runtime::recordEePerformanceEvents(
    R5900Context *ctx,
    const ps2x::timing::Cop0PerformanceEvents &events) noexcept
{
    if (!ctx)
    {
        return;
    }

    const ps2x::timing::Cop0TimingAdvanceResult result =
        m_cop0Timing.recordPerformanceEvents(
            ctx->cop0_status, events);
    publishCop0TimingMirrors(ctx);
    if (result.performanceOverflowMask != 0u)
    {
        refreshCop0EventSchedules(
            ctx,
            ps2x::timing::eeTickToCyclesFloor(
                currentEeTick()));
    }
}

void PS2Runtime::recordEeInstructionCompletion(
    R5900Context *ctx,
    uint32_t completionEvents)
{
    if (!ctx || completionEvents == 0u)
    {
        return;
    }

    using namespace ps2x::performance;
    ps2x::timing::Cop0PerformanceEvents events{};
    events.instructionCompleted =
        (completionEvents & kInstructionCompleted) != 0u;
    events.nonBdsInstructionCompleted =
        (completionEvents &
         kNonBdsInstructionCompleted) != 0u;
    events.cop2InstructionCompleted =
        (completionEvents & kCop2InstructionCompleted) != 0u;
    events.cop1InstructionCompleted =
        (completionEvents & kCop1InstructionCompleted) != 0u;
    events.loadCompleted =
        (completionEvents & kLoadCompleted) != 0u;
    events.storeCompleted =
        (completionEvents & kStoreCompleted) != 0u;
    recordEePerformanceEvents(ctx, events);
}

void PS2Runtime::recordEePredictedBranch(
    R5900Context *ctx,
    uint32_t branchPc)
{
    if (!ctx)
    {
        return;
    }

    ps2x::timing::Cop0PerformanceEvents events{};
    events.branchIssued = 1u;
    events.lowOrderBranchIssued =
        (branchPc & 4u) == 0u ? 1u : 0u;
    recordEePerformanceEvents(ctx, events);
}

void PS2Runtime::recordEeConditionalBranch(
    R5900Context *ctx,
    uint32_t branchPc,
    bool taken)
{
    if (!ctx)
    {
        return;
    }

    constexpr uint32_t kConfigBranchPredictionEnable =
        1u << 12u;
    const size_t index =
        (branchPc >> 2u) &
        (kEeBranchHistoryEntries - 1u);
    if (m_eeBranchHistoryTags[index] != branchPc)
    {
        m_eeBranchHistoryTags[index] = branchPc;
        m_eeBranchHistoryStates[index] = 0u;
    }
    uint8_t &history = m_eeBranchHistoryStates[index];
    const bool predictedTaken = history >= 2u;
    const bool mispredicted =
        (ctx->cop0_config &
         kConfigBranchPredictionEnable) != 0u &&
        predictedTaken != taken;
    if (taken)
    {
        history = std::min<uint8_t>(history + 1u, 3u);
    }
    else if (history != 0u)
    {
        --history;
    }

    ps2x::timing::Cop0PerformanceEvents events{};
    events.branchIssued = 1u;
    events.branchMispredicted =
        mispredicted ? 1u : 0u;
    events.lowOrderBranchIssued =
        (branchPc & 4u) == 0u ? 1u : 0u;
    recordEePerformanceEvents(ctx, events);
}

PS2X_EE_OBSERVATION_COLD
void PS2Runtime::observeEeInstructionBeginPrecise(
    R5900Context *ctx,
    const EeObservationInstructionDescriptor *instruction,
    bool breakpointAlreadyChecked)
{
    if (!instruction)
    {
        return;
    }
    if (!breakpointAlreadyChecked)
    {
        CheckEeInstructionBreakpoint(ctx, instruction->pc);
    }
    if (ctx && (ctx->cop0_perf & 0x80000000u) != 0u)
    {
        recordEeInstructionIssue(ctx, instruction->issue);
    }
}

PS2X_EE_OBSERVATION_COLD
void PS2Runtime::observeEeInstructionCompletePrecise(
    R5900Context *ctx,
    const EeObservationInstructionDescriptor *instruction)
{
    if (ctx && instruction &&
        (ctx->cop0_perf & 0x80000000u) != 0u)
    {
        recordEeInstructionCompletion(
            ctx, instruction->completionEvents);
    }
}

PS2X_EE_OBSERVATION_COLD
bool PS2Runtime::observeEeDataAddressPrecise(
    R5900Context *ctx,
    uint32_t virtualAddress,
    bool write)
{
    return CheckEeDataAddressBreakpoint(
        ctx, virtualAddress, write);
}

PS2X_EE_OBSERVATION_COLD
void PS2Runtime::observeEeDataValuePrecise(
    R5900Context *ctx,
    uint32_t virtualAddress,
    uint32_t value,
    bool write)
{
    CheckEeDataValueBreakpoint(
        ctx, virtualAddress, value, write);
}

PS2X_EE_OBSERVATION_COLD
void PS2Runtime::observeEePredictedBranchPrecise(
    R5900Context *ctx,
    const EeObservationInstructionDescriptor *instruction)
{
    if (ctx && instruction &&
        (ctx->cop0_perf & 0x80000000u) != 0u)
    {
        recordEePredictedBranch(ctx, instruction->pc);
    }
}

PS2X_EE_OBSERVATION_COLD
void PS2Runtime::observeEeConditionalBranchPrecise(
    R5900Context *ctx,
    const EeObservationInstructionDescriptor *instruction,
    bool taken)
{
    if (ctx && instruction &&
        (ctx->cop0_perf & 0x80000000u) != 0u)
    {
        recordEeConditionalBranch(
            ctx, instruction->pc, taken);
    }
}

void PS2Runtime::SignalBusError(R5900Context *ctx,
                                PS2BusErrorAccess access,
                                uint32_t physicalAddress)
{
    if (!ctx)
    {
        throw PS2GuestException{};
    }

    // Bus errors cannot recursively enter either exception level. BEM also
    // suppresses both BadPAddr capture and delivery until software clears it.
    if ((ctx->cop0_status &
         (COP0_STATUS_BEM |
          COP0_STATUS_EXL |
          COP0_STATUS_ERL)) != 0u)
    {
        return;
    }

    if (access == PS2BusErrorAccess::DataLoad)
    {
        recordEeInstructionCompletion(
            ctx, ps2x::performance::kLoadCompleted);
    }
    else if (access == PS2BusErrorAccess::DataStore)
    {
        recordEeInstructionCompletion(
            ctx, ps2x::performance::kStoreCompleted);
    }

    ctx->cop0_badpaddr = physicalAddress & ~0xFu;
    ctx->cop0_status |= COP0_STATUS_BEM;
    raiseCop0Level1Exception(
        ctx,
        access == PS2BusErrorAccess::InstructionFetch
            ? EXCEPTION_BUS_ERROR_INSTRUCTION
            : EXCEPTION_BUS_ERROR_DATA);
}

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
void PS2Runtime::debugArmVu0TracesSlow(const R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }

    if (m_debugVu0SyncTraceEnabled.load(std::memory_order_acquire) &&
        !m_debugVu0SyncTraceTriggered.load(std::memory_order_acquire) &&
        m_debugVu0SyncTraceHasTrigger.load(std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugVu0SyncTraceTriggerEePc.load(std::memory_order_relaxed))
    {
        // VCALLMS synchronizes before the new microprogram is active. Arm at
        // that boundary so the following invocation is retained without
        // fabricating an inactive sync entry.
        const DebugEeScheduler scheduler =
            debugEeSchedulerSnapshot();
        const uint64_t vsyncTick =
            ps2_syscalls::GetCurrentVSyncTick(this);
        {
            std::lock_guard<std::mutex> lock(
                m_debugVu0SyncTraceMutex);
            if (m_debugVu0SyncTraceEnabled.load(
                    std::memory_order_relaxed) &&
                !m_debugVu0SyncTraceTriggered.load(
                    std::memory_order_relaxed))
            {
                m_debugVu0SyncTraceTriggerScheduler =
                    scheduler;
                m_debugVu0SyncTraceTriggerVsyncTick =
                    vsyncTick;
                m_debugVu0SyncTraceHasTriggerSchedulerSnapshot =
                    true;
            }
        }
        m_debugVu0SyncTraceTriggered.store(true, std::memory_order_release);
    }

    if (m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_acquire) &&
        !m_debugVu0InstructionTraceTriggered.load(
            std::memory_order_acquire) &&
        m_debugVu0InstructionTraceHasTrigger.load(
            std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugVu0InstructionTraceTriggerEePc.load(
                std::memory_order_relaxed))
    {
        m_debugVu0InstructionTraceTriggered.store(
            true, std::memory_order_release);
    }

    if (m_debugEeEventTraceEnabled.load(
            std::memory_order_acquire) &&
        !m_debugEeEventTraceTriggered.load(
            std::memory_order_acquire) &&
        m_debugEeEventTraceHasTrigger.load(
            std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugEeEventTraceTriggerEePc.load(
                std::memory_order_relaxed))
    {
        // EE-event correlation uses the same architectural VCALL boundary
        // as the VU traces. Arming here also retains the first event after
        // that boundary when no device deadline is due on the VCALL itself.
        m_debugEeEventTraceTriggered.store(
            true, std::memory_order_release);
    }
}

void PS2Runtime::beginVu0Invocation()
{
    const uint64_t invocation =
        m_vu0InvocationSequence.fetch_add(
            1u, std::memory_order_relaxed) +
        1u;
    m_vu0CurrentInvocation.store(
        invocation, std::memory_order_relaxed);
    m_vu0CurrentInvocationInstruction.store(
        0u, std::memory_order_relaxed);
    if (m_debugVu0SyncTraceEnabled.load(
            std::memory_order_acquire) &&
        !m_debugVu0SyncTraceTriggered.load(
            std::memory_order_acquire) &&
        m_debugVu0SyncTraceHasInvocationTrigger.load(
            std::memory_order_relaxed) &&
        invocation ==
            m_debugVu0SyncTraceTriggerInvocation.load(
                std::memory_order_relaxed))
    {
        m_debugVu0SyncTraceTriggered.store(
            true, std::memory_order_release);
    }
    if (m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_acquire) &&
        !m_debugVu0InstructionTraceTriggered.load(
            std::memory_order_acquire) &&
        m_debugVu0InstructionTraceHasInvocationTrigger.load(
            std::memory_order_relaxed) &&
        invocation ==
            m_debugVu0InstructionTraceTriggerInvocation.load(
                std::memory_order_relaxed))
    {
        m_debugVu0InstructionTraceTriggered.store(
            true, std::memory_order_release);
    }
}
#endif

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    beginVu0Invocation();
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop);
    copyVu0StateToContext(m_vu0.state(), ctx);
    ctx->vu0_vpu_stat =
        (ctx->vu0_vpu_stat & ~1u) | (m_vu0.isActive() ? 1u : 0u);
    const ps2x::timing::EeTick eeCycleTick = currentEeTick();
    m_vu0CycleTick = eeCycleTick;
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    synchronizeVU0Microprogram(rdram, ctx, true);

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    beginVu0Invocation();
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, kVu0MinimumRunCycles);
    copyVu0StateToContext(m_vu0.state(), ctx);
    ctx->vu0_vpu_stat =
        (ctx->vu0_vpu_stat & ~1u) | (m_vu0.isActive() ? 1u : 0u);
    const ps2x::timing::EeTick eeCycleTick = currentEeTick();
    m_vu0CycleTick = eeCycleTick;
    if (m_vu0.isActive())
    {
        m_vu0CycleTick = ps2x::timing::saturatingAdd(
            m_vu0CycleTick,
            ps2x::timing::vuCyclesToEeTicks(
                kVu0MinimumRunCycles));
    }
}

void PS2Runtime::synchronizeVU0Microprogram(
    uint8_t *rdram, R5900Context *ctx, bool interlocked)
{
    debugArmVu0Traces(ctx);
    if (!ctx)
    {
        return;
    }

    const ps2x::timing::EeTick eeCycleTick =
        commitEeContextProgress(ctx);
    if (m_eeEventScheduler.deadlineDue(eeCycleTick))
    {
        // A COP2 synchronization is also a safe EE event boundary. Service
        // the complete due device batch before publishing VU0 state, just
        // as PCSX2 does when an architectural access reaches an event test.
        serviceEeEventsAtTick(rdram, ctx, eeCycleTick);
    }
    if (!m_vu0.isActive())
    {
        ctx->vu0_vpu_stat &= ~1u;
        m_vu0CycleTick = eeCycleTick;
        return;
    }

    ps2x::timing::EeTick nextEventTick{};
    const auto scheduled =
        m_eeEventScheduler.nextDeadline();
    if (scheduled.has_value())
    {
        nextEventTick = *scheduled;
    }

    synchronizeVU0MicroprogramAtTick(
        rdram, ctx, interlocked, false, false,
        eeCycleTick, nextEventTick);
}

void PS2Runtime::publishVU0VectorRegisterWrite(
    const R5900Context *ctx, uint32_t index) noexcept
{
    if (!ctx || index >= 32u)
    {
        return;
    }

    VuExecutionState &state = m_vu0.state();
    _mm_storeu_ps(state.vf[index], ctx->vu0_vf[index]);
    if (index == 0u)
    {
        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
    }
}

void PS2Runtime::publishVU0ControlRegisterWrite(
    const R5900Context *ctx, uint32_t index) noexcept
{
    if (!ctx)
    {
        return;
    }

    VuExecutionState &state = m_vu0.state();
    if (index < 16u)
    {
        state.vi[index] = static_cast<int16_t>(ctx->vi[index]);
        state.vi[0] = 0;
        return;
    }

    // These are the writable control registers represented in the live
    // micro-mode state. CMSAR/FBRST and R are owned outside VuExecutionState;
    // read-only or reserved control-register writes have no state to publish.
    switch (index)
    {
    case 16u: // STATUS
        state.status = ctx->vu0_status;
        break;
    case 18u: // CLIP
        state.clip = ctx->vu0_clip_flags;
        break;
    case 21u: // I
        state.i = ctx->vu0_i;
        break;
    case 22u: // Q
        state.q = ctx->vu0_q;
        break;
    default:
        break;
    }
}

void PS2Runtime::serviceEeEventsAtBlockBoundary(
    uint8_t *rdram, R5900Context *ctx)
{
    debugArmVu0Traces(ctx);
    if (!ctx)
    {
        return;
    }

    const ps2x::timing::EeTick eeCycleTick =
        finishEeContextBlock(ctx);
    serviceEeEventsAtTick(rdram, ctx, eeCycleTick);
}

void PS2Runtime::serviceEeEventsAtGeneratedBlockBoundary(
    uint8_t *rdram, R5900Context *ctx)
{
    debugArmVu0Traces(ctx);
    if (!ctx)
    {
        return;
    }

    // Fast generated code cannot create a pending precise issue pair, but a
    // mode transition may hand one to it. Keep that publication path out of
    // the overwhelmingly common no-event boundary.
    if (ctx->ee_performance_issue_pending) [[unlikely]]
    {
        serviceEeEventsAtGeneratedPendingIssueBoundary(
            rdram, ctx);
        return;
    }

    // Generated Fast-mode code may carry a counted Random retirement run
    // across an internal control-flow edge. Block timing and issue state are
    // still published at every edge; Random becomes observable only when a
    // due event can enter callbacks/exceptions or another explicit
    // publication barrier is reached.
    const ps2x::timing::EeTick eeCycleTick =
        publishEeElapsed(ctx->finishEeBasicBlock());
    if (!m_eeEventScheduler.deadlineDue(eeCycleTick))
    {
        return;
    }

    serviceEeEventsAtGeneratedDeadline(
        rdram, ctx, eeCycleTick);
}

void PS2Runtime::serviceEeEventsAtGeneratedPendingIssueBoundary(
    uint8_t *rdram, R5900Context *ctx)
{
    flushEeInstructionIssue(ctx);
    const ps2x::timing::EeTick eeCycleTick =
        publishEeElapsed(ctx->finishEeBasicBlock());
    if (!m_eeEventScheduler.deadlineDue(eeCycleTick))
    {
        return;
    }

    ctx->finishEeInstruction();
    serviceEeEventsAtTick(rdram, ctx, eeCycleTick);
}

void PS2Runtime::serviceEeEventsAtGeneratedDeadline(
    uint8_t *rdram,
    R5900Context *ctx,
    ps2x::timing::EeTick eeCycleTick)
{
    ctx->finishEeInstruction();
    serviceEeEventsAtTick(rdram, ctx, eeCycleTick);
}

void PS2Runtime::serviceEeVSyncAtEvent(
    uint8_t *rdram,
    const ps2x::timing::EeEventService &service)
{
    if (!m_eeVSyncTiming.enabled ||
        m_eeVSyncTiming.eventToken.generation == 0u ||
        service.generation !=
            m_eeVSyncTiming.eventToken.generation)
    {
        return;
    }

    m_eeVSyncTiming.eventToken = {
        ps2x::timing::EeEventSource::VSync, 0u};

    switch (m_eeVSyncTiming.phase)
    {
    case EeVSyncPhase::Start:
    {
        paceEeVSyncStart(service.scheduledTick);
        if (isStopRequested())
        {
            return;
        }
        m_eeVSyncTiming.startTick =
            service.scheduledTick;
        m_eeVSyncTiming.currentVSyncTick =
            ps2_syscalls::PublishVSyncStart(
                rdram, this);
        if (!queuePendingVSyncDelivery(
                PendingVSyncDelivery{
                    .kind = PendingVSyncKind::Start,
                    .tick =
                        m_eeVSyncTiming.currentVSyncTick,
                    .eventSequence = service.sequence,
                }))
        {
            std::cerr
                << "EE VSync delivery queue exceeded its fixed bound"
                << std::endl;
            requestStop();
            return;
        }
        m_eeVSyncTiming.phase =
            EeVSyncPhase::GsBlank;
        scheduleEeVSyncEvent(
            ps2x::timing::saturatingAdd(
                m_eeVSyncTiming.startTick,
                ps2x::timing::eeCyclesToTicks(
                    m_eeVSyncTiming
                        .durations.gsBlankCycles)));
        requestGuestPreemption();
        return;
    }
    case EeVSyncPhase::GsBlank:
        publishEeVSyncField();
        m_eeVSyncTiming.phase = EeVSyncPhase::End;
        scheduleEeVSyncEvent(
            ps2x::timing::saturatingAdd(
                m_eeVSyncTiming.startTick,
                ps2x::timing::eeCyclesToTicks(
                    m_eeVSyncTiming
                        .durations.blankCycles)));
        return;
    case EeVSyncPhase::End:
        if (!queuePendingVSyncDelivery(
                PendingVSyncDelivery{
                    .kind = PendingVSyncKind::End,
                    .tick =
                        m_eeVSyncTiming.currentVSyncTick,
                    .eventSequence = service.sequence,
                }))
        {
            std::cerr
                << "EE VSync delivery queue exceeded its fixed bound"
                << std::endl;
            requestStop();
            return;
        }
        m_eeVSyncTiming.phase = EeVSyncPhase::Start;
        scheduleEeVSyncEvent(
            ps2x::timing::saturatingAdd(
                m_eeVSyncTiming.startTick,
                ps2x::timing::eeCyclesToTicks(
                    m_eeVSyncTiming
                        .durations.periodCycles)));
        requestGuestPreemption();
        return;
    }
}

void PS2Runtime::dispatchEeEvent(
    uint8_t *rdram,
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    switch (service.source)
    {
    case ps2x::timing::EeEventSource::
        Cop0Performance:
        serviceCop0PerformanceAtEvent(
            ctx, service);
        return;
    case ps2x::timing::EeEventSource::VSync:
        serviceEeVSyncAtEvent(rdram, service);
        return;
    case ps2x::timing::EeEventSource::EeCounters:
        serviceEeCountersAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::Cop0Timer:
        serviceCop0TimerAtEvent(ctx, service);
        return;
    case ps2x::timing::EeEventSource::
        DmacCompletion:
        (void)publishReadyDmacCompletions(
            service.sequence);
        if (m_dmacCompletionReady.load(
                std::memory_order_acquire))
        {
            (void)m_eeEventScheduler.scheduleAbsolute(
                ps2x::timing::EeEventSource::
                    DmacCompletion,
                service.serviceTick);
        }
        return;
    case ps2x::timing::EeEventSource::DmacVif1:
        serviceVif1DmaAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::DmacGif:
        serviceGifDmaAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::HleSif1:
        serviceHleSifDmaAtEvent(rdram, service);
        return;
    case ps2x::timing::EeEventSource::DmacVif0:
        serviceVif0DmaAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::DmacFromIpu:
        serviceFromIpuDmaAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::DmacToIpu:
        serviceToIpuDmaAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::
        DmacFromScratchpad:
        serviceScratchpadDmaAtEvent(
            DmacChannel::FromScratchpad, service);
        return;
    case ps2x::timing::EeEventSource::
        DmacToScratchpad:
        serviceScratchpadDmaAtEvent(
            DmacChannel::ToScratchpad, service);
        return;
    case ps2x::timing::EeEventSource::VifVu1Finish:
        serviceVU1AtEvent(ctx, service);
        return;
    case ps2x::timing::EeEventSource::VifVu0Finish:
        serviceVU0AtVifFinishEvent(
            rdram, ctx, service);
        return;
    case ps2x::timing::EeEventSource::Count:
        return;
    }
}

void PS2Runtime::serviceEeEventsAtTick(
    uint8_t *rdram,
    R5900Context *ctx,
    ps2x::timing::EeTick eeCycleTick)
{
    if (!m_eeEventScheduler.deadlineDue(eeCycleTick))
    {
        armDeferredVu1SpeculativeSlice();
        return;
    }

    const ps2x::timing::EeEventServiceResult result =
        m_eeEventScheduler.serviceDue(
            eeCycleTick,
            [&](const ps2x::timing::EeEventService &service)
            {
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
                const bool traceEnabled =
                    m_debugEeEventTraceEnabled.load(
                        std::memory_order_relaxed);
                DebugEeEventEntry entry{};
                if (traceEnabled)
                {
                    entry.source = service.source;
                    entry.eventSequence = service.sequence;
                    entry.generation = service.generation;
                    entry.scheduledTick =
                        service.scheduledTick.raw();
                    entry.serviceTick =
                        service.serviceTick.raw();
                    entry.latenessTicks =
                        service.latenessTicks.raw();
                    entry.eePc = ctx ? ctx->pc : 0u;
                    entry.deviceBefore =
                        debugEeEventDeviceState(
                            service.source);
                }
#endif

                if (service.source ==
                    ps2x::timing::EeEventSource::
                        VifVu1Finish)
                {
                    // The scheduler has already serviced every equal-tick
                    // higher-priority VIF1 callback. Arm exactly once from
                    // the resulting canonical owner state before publishing.
                    armDeferredVu1SpeculativeSlice(
                        service.generation);
                }
                dispatchEeEvent(rdram, ctx, service);

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
                if (traceEnabled)
                {
                    entry.deviceAfter =
                        debugEeEventDeviceState(
                            service.source);
                    const auto followup =
                        m_eeEventScheduler.event(
                            service.source);
                    if (followup.has_value())
                    {
                        entry.hasFollowup = true;
                        entry.rescheduled = true;
                        entry.followupTick =
                            followup->deadline.raw();
                    }
                    debugRecordEeEvent(
                        std::move(entry));
                }
#endif

                if (service.source ==
                        ps2x::timing::EeEventSource::
                            Cop0Performance ||
                    service.source ==
                        ps2x::timing::EeEventSource::
                            Cop0Timer)
                {
                    // Record the hardware transition before unwinding
                    // generated code into the selected exception vector.
                    deliverPendingCop0Exceptions(ctx);
                }
            });
    if (result.limitExceeded)
    {
        std::cerr << "EE scheduler service limit exceeded";
        if (result.offendingSource.has_value())
        {
            std::cerr << " for "
                      << ps2x::timing::eeEventSourceName(
                             *result.offendingSource);
        }
        std::cerr << std::endl;
        requestStop();
        return;
    }

    // Retain the plan while another known VIF1 callback can run first. Once
    // that chain ends, restore lookahead without waiting for an unrelated
    // future scheduler event.
    armDeferredVu1SpeculativeSlice();

    // PCSX2 services the complete fixed-priority device batch before its
    // ordinary VU0 catch-up. A same-tick callback scheduled by another
    // callback therefore remains part of this batch, and VU0 advances only
    // once to the actual shared-event service time.
    if (result.serviced != 0u)
    {
        ps2x::timing::EeTick nextEventTick{};
        const auto nextDeadline =
            m_eeEventScheduler.nextDeadline();
        if (nextDeadline.has_value())
            nextEventTick = *nextDeadline;

        if (m_vu0.isActive())
        {
            synchronizeVU0MicroprogramAtTick(
                rdram, ctx, false, true, true,
                eeCycleTick, nextEventTick);
        }
        else
        {
            ctx->vu0_vpu_stat &= ~1u;
            m_vu0CycleTick = eeCycleTick;
        }
    }
}

PS2Runtime::DebugEeEventDeviceState
PS2Runtime::debugEeEventDeviceState(
    ps2x::timing::EeEventSource source) const noexcept
{
    DebugEeEventDeviceState result{};
    switch (source)
    {
    case ps2x::timing::EeEventSource::
        Cop0Performance:
    {
        const ps2x::timing::Cop0TimingSnapshot state =
            m_cop0Timing.snapshot();
        result.kind =
            DebugEeEventDeviceKind::
                Cop0Performance;
        result.operationGeneration = state.pccr;
        result.lastAdvancedTick =
            ps2x::timing::eeCyclesToTicks(
                std::max(
                    state.lastPerformanceCycle[0],
                    state.lastPerformanceCycle[1]))
                .raw();
        result.phase = state.pccr;
        result.qwc = state.pcr[0];
        result.tagsProcessed = state.pcr[1];
        result.active =
            (state.pccr & 0x80000000u) != 0u;
        return result;
    }
    case ps2x::timing::EeEventSource::
        Cop0Timer:
    {
        const ps2x::timing::Cop0TimingSnapshot state =
            m_cop0Timing.snapshot();
        result.kind =
            DebugEeEventDeviceKind::Cop0Timer;
        result.lastAdvancedTick =
            ps2x::timing::eeCyclesToTicks(
                state.lastCountCycle)
                .raw();
        result.phase = state.count;
        result.stall = state.compare;
        result.qwc = state.timerPending ? 1u : 0u;
        result.active = state.timerArmed;
        return result;
    }
    case ps2x::timing::EeEventSource::VSync:
        result.kind = DebugEeEventDeviceKind::VSync;
        result.operationGeneration =
            m_eeVSyncTiming.currentVSyncTick;
        result.lastAdvancedTick =
            m_eeVSyncTiming.startTick.raw();
        result.phase = static_cast<uint32_t>(
            m_eeVSyncTiming.phase);
        result.active = m_eeVSyncTiming.enabled;
        return result;
    case ps2x::timing::EeEventSource::EeCounters:
    {
        const ps2x::timing::EeCounterBankSnapshot state =
            m_memory.eeCounterSnapshot();
        result.kind =
            DebugEeEventDeviceKind::EeCounters;
        result.lastAdvancedTick =
            ps2x::timing::eeCyclesToTicks(
                state.currentCycle)
                .raw();
        result.phase =
            static_cast<uint32_t>(
                state.hSyncPhase);
        result.stall =
            static_cast<uint32_t>(
                state.vSyncPhase);
        result.qwc =
            state.counters[0].count & 0xffffu;
        result.tagsProcessed =
            state.counters[1].count & 0xffffu;
        result.active = true;
        return result;
    }
    case ps2x::timing::EeEventSource::DmacVif1:
    {
        const Vif1DmaSnapshot state =
            m_memory.vif1DmaSnapshot();
        result.kind = DebugEeEventDeviceKind::Vif1Dma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.stall =
            static_cast<uint32_t>(state.stall);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed = state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::DmacGif:
    {
        const GifDmaSnapshot state =
            m_memory.gifDmaSnapshot();
        result.kind = DebugEeEventDeviceKind::GifDma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.stall =
            static_cast<uint32_t>(state.stall);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed = state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::DmacFromIpu:
    {
        const FromIpuDmaSnapshot state =
            m_memory.fromIpuDmaSnapshot();
        result.kind =
            DebugEeEventDeviceKind::FromIpuDma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.stall =
            static_cast<uint32_t>(state.stall);
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::DmacToIpu:
    {
        const ToIpuDmaSnapshot state =
            m_memory.toIpuDmaSnapshot();
        result.kind =
            DebugEeEventDeviceKind::ToIpuDma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.stall =
            static_cast<uint32_t>(state.stall);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed =
            state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::HleSif1:
    {
        result.kind =
            DebugEeEventDeviceKind::HleSifDma;
        if (m_hleSifDmaQueueCount == 0u)
        {
            return result;
        }
        const HleSifDmaOperation &state =
            m_hleSifDmaQueue[
                m_hleSifDmaQueueHead];
        result.operationGeneration =
            state.operationGeneration;
        result.totalAdvancedCycles =
            state.iopCycles;
        result.phase = static_cast<uint32_t>(
            m_hleSifDmaQueueCount);
        result.qwc = static_cast<uint32_t>(
            std::min<uint64_t>(
                state.iopCycles,
                std::numeric_limits<uint32_t>::max()));
        result.tagsProcessed =
            state.submission.transferCount;
        if (state.submission.transferCount != 0u)
        {
            result.madr =
                state.submission.transfers[0].src;
            result.tadr =
                state.submission.transfers[0].dest;
        }
        result.active = true;
        return result;
    }
    case ps2x::timing::EeEventSource::DmacVif0:
    {
        const Vif0DmaSnapshot state =
            m_memory.vif0DmaSnapshot();
        result.kind = DebugEeEventDeviceKind::Vif0Dma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.stall =
            static_cast<uint32_t>(state.stall);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed = state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::
        DmacFromScratchpad:
    case ps2x::timing::EeEventSource::
        DmacToScratchpad:
    {
        const DmacChannel channel =
            source ==
                    ps2x::timing::EeEventSource::
                        DmacFromScratchpad
                ? DmacChannel::FromScratchpad
                : DmacChannel::ToScratchpad;
        const ScratchpadDmaSnapshot state =
            m_memory.scratchpadDmaSnapshot(channel);
        result.kind =
            DebugEeEventDeviceKind::ScratchpadDma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed = state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::VifVu1Finish:
        result.kind = DebugEeEventDeviceKind::Vu1;
        result.operationGeneration =
            m_vu1ExecutionTiming.generation;
        result.lastAdvancedTick =
            m_vu1ExecutionTiming.lastAdvancedTick.raw();
        result.totalAdvancedCycles =
            m_vu1ExecutionTiming.totalAdvancedCycles;
        result.pc = m_publishedVu1ProgramCounter.load(
            std::memory_order_acquire);
        result.active = m_publishedVu1Active.load(
            std::memory_order_acquire);
        return result;
    case ps2x::timing::EeEventSource::VifVu0Finish:
        result.kind = DebugEeEventDeviceKind::Vu0;
        result.lastAdvancedTick =
            m_vu0CycleTick.raw();
        result.pc = m_vu0.state().pc;
        result.active = m_vu0.isActive();
        return result;
    case ps2x::timing::EeEventSource::DmacCompletion:
    case ps2x::timing::EeEventSource::Count:
        return result;
    }
    return result;
}

void PS2Runtime::synchronizeVU0MicroprogramAtTick(
    uint8_t *rdram, R5900Context *ctx, bool interlocked,
    bool blockBoundary, bool eventDue,
    ps2x::timing::EeTick eeCycleTick,
    ps2x::timing::EeTick nextEventTick)
{
    (void)rdram;

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    const bool traceEnabled =
        m_debugVu0SyncTraceEnabled.load(std::memory_order_acquire);
    DebugVu0SyncEntry traceEntry{};
    bool traceSync =
        traceEnabled &&
        m_debugVu0SyncTraceTriggered.load(std::memory_order_acquire);
    if (traceEnabled && !traceSync &&
        m_debugVu0SyncTraceHasTrigger.load(std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugVu0SyncTraceTriggerEePc.load(std::memory_order_relaxed))
    {
        m_debugVu0SyncTraceTriggered.store(true, std::memory_order_release);
        traceSync = true;
    }
    if (traceSync)
    {
        const VuExecutionState &state = m_vu0.state();
        traceEntry.invocation =
            m_vu0CurrentInvocation.load(std::memory_order_relaxed);
        traceEntry.invocationInstruction =
            m_vu0CurrentInvocationInstruction.load(
                std::memory_order_relaxed);
        traceEntry.eeCycleTicks = eeCycleTick.raw();
        traceEntry.vsyncTick =
            ps2_syscalls::GetCurrentVSyncTick(this);
        traceEntry.vuCycleTicks = m_vu0CycleTick.raw();
        traceEntry.nextEventCycleTicks =
            nextEventTick.raw();
        traceEntry.eePc = ctx->pc;
        traceEntry.vuPcBefore = state.pc;
        traceEntry.contextVi1 = ctx->vi[1];
        traceEntry.contextVi2 = ctx->vi[2];
        traceEntry.vuVi1Before = static_cast<uint16_t>(state.vi[1]);
        traceEntry.vuVi2Before = static_cast<uint16_t>(state.vi[2]);
        traceEntry.interlocked = interlocked;
        traceEntry.blockBoundary = blockBoundary;
        traceEntry.activeBefore = m_vu0.isActive();
        for (size_t index = 0u; index < 16u; ++index)
        {
            if (ctx->vi[index] != static_cast<uint16_t>(state.vi[index]))
            {
                traceEntry.pendingViMask |=
                    static_cast<uint16_t>(1u << index);
            }
        }
        for (size_t index = 0u; index < 32u; ++index)
        {
            if (std::memcmp(
                    &ctx->vu0_vf[index], state.vf[index],
                    sizeof(state.vf[index])) != 0)
            {
                traceEntry.pendingVfMask |= 1u << index;
            }
        }
    }
#endif

    uint32_t cycleBudget = 0u;
    if (interlocked)
    {
        cycleBudget = 65536u;
    }
    else if (eeCycleTick > m_vu0CycleTick)
    {
        const uint64_t elapsedCycles =
            ps2x::timing::eeTicksToVuCyclesFloor(
                ps2x::timing::elapsedEeTicks(
                    m_vu0CycleTick, eeCycleTick));
        if (blockBoundary)
        {
            if (eventDue)
            {
                cycleBudget = static_cast<uint32_t>(
                    std::min<uint64_t>(
                        std::max<uint64_t>(
                            elapsedCycles,
                            kVu0MinimumRunCycles),
                        65536u));
            }
        }
        else if (elapsedCycles >= kVu0SyncThresholdCycles)
        {
            cycleBudget = static_cast<uint32_t>(
                std::min<uint64_t>(
                    std::max<uint64_t>(
                        elapsedCycles,
                        kVu0MinimumRunCycles),
                    65536u));
        }
    }
    copyVu0ContextRegistersToState(ctx, m_vu0.state());
    if (cycleBudget != 0u)
    {
        m_vu0.continueExecution(
            m_memory.getVU0Code(), PS2_VU0_CODE_SIZE,
            m_memory.getVU0Data(), PS2_VU0_DATA_SIZE,
            m_gs, &m_memory, cycleBudget);
    }
    copyVu0StateToContext(m_vu0.state(), ctx);
    ctx->vu0_vpu_stat =
        (ctx->vu0_vpu_stat & ~1u) | (m_vu0.isActive() ? 1u : 0u);
    if (interlocked || !m_vu0.isActive())
    {
        m_vu0CycleTick = eeCycleTick;
    }
    else
    {
        m_vu0CycleTick = ps2x::timing::saturatingAdd(
            m_vu0CycleTick,
            ps2x::timing::vuCyclesToEeTicks(cycleBudget));
    }
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (traceSync)
    {
        const VuExecutionState &state = m_vu0.state();
        traceEntry.cycleBudget = cycleBudget;
        traceEntry.eventDue = eventDue;
        traceEntry.vuPcAfter = state.pc;
        traceEntry.vuVi1After = static_cast<uint16_t>(state.vi[1]);
        traceEntry.vuVi2After = static_cast<uint16_t>(state.vi[2]);
        traceEntry.activeAfter = m_vu0.isActive();
        debugRecordVu0Sync(std::move(traceEntry));
    }
#endif
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        raiseCop0Level1Exception(ctx, EXCEPTION_SYSCALL);
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Level1Exception(
        ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainPendingAlarmCallbacks()
{
    EeAlarmRuntimeState &state =
        eeAlarmRuntimeState();
    if (!state.pendingCallbackPublished.load(
            std::memory_order_acquire))
    {
        return;
    }

    std::deque<std::shared_ptr<AlarmInfo>> pending;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        pending.swap(state.pendingCallbacks);
        state.pendingCallbackPublished.store(
            false, std::memory_order_release);
    }

    for (const std::shared_ptr<AlarmInfo> &alarm :
         pending)
    {
        if (isStopRequested())
        {
            break;
        }
        if (!alarm ||
            !alarm->rdram ||
            !alarm->handler ||
            !hasFunction(alarm->handler))
        {
            continue;
        }

        try
        {
            R5900Context &callbackCtx =
                state.callbackContext;

            AsyncCallbackInvocationScope callbackExecution(
                this, &callbackCtx);
            SET_GPR_U32(&callbackCtx, 28, alarm->gp);
            SET_GPR_U32(&callbackCtx, 31, 0u);
            SET_GPR_U32(
                &callbackCtx,
                4,
                static_cast<uint32_t>(alarm->id));
            SET_GPR_U32(
                &callbackCtx,
                5,
                static_cast<uint32_t>(alarm->ticks));
            SET_GPR_U32(
                &callbackCtx, 6, alarm->commonArg);
            SET_GPR_U32(&callbackCtx, 7, 0u);
            SET_GPR_U32(
                &callbackCtx,
                29,
                callbackExecution.stackTop());
            callbackCtx.pc = alarm->handler;
            RecompiledFunction function =
                lookupFunction(alarm->handler, &callbackCtx);
            executeGuestStep(
                alarm->rdram,
                &callbackCtx,
                function);
        }
        catch (const ThreadExitException &)
        {
        }
        catch (const std::exception &e)
        {
            if (state.exceptionLogCount < 8u)
            {
                std::cerr
                    << "[SetAlarm] callback exception: "
                    << e.what() << std::endl;
                ++state.exceptionLogCount;
            }
        }
    }
}

void PS2Runtime::drainPendingVSyncHandlers(uint8_t *rdram)
{
    if (m_pendingVSyncDeliveryCount == 0u)
    {
        return;
    }

    std::array<
        PendingVSyncDelivery,
        kMaximumPendingVSyncDeliveries>
        pending{};
    const size_t pendingCount =
        m_pendingVSyncDeliveryCount;
    std::copy_n(
        m_pendingVSyncDeliveries.begin(),
        pendingCount,
        pending.begin());
    m_pendingVSyncDeliveryCount = 0u;
    for (size_t index = 0u;
         index < pendingCount;
         ++index)
    {
        const PendingVSyncDelivery &delivery =
            pending[index];
        switch (delivery.kind)
        {
        case PendingVSyncKind::Start:
            ps2_syscalls::DispatchVSyncStartHandlers(
                rdram, this, delivery.tick);
            break;
        case PendingVSyncKind::End:
            ps2_syscalls::DispatchVSyncEndHandlers(
                rdram, this);
            break;
        }
    }
}

void PS2Runtime::drainPendingEeCounterHandlers(
    uint8_t *rdram)
{
    uint32_t pending =
        m_pendingEeCounterInterrupts;
    m_pendingEeCounterInterrupts = 0u;
    uint32_t retained = 0u;
    for (uint32_t cause = 9u;
         cause <= 12u;
         ++cause)
    {
        const uint32_t bit = 1u << cause;
        if ((pending & bit) == 0u ||
            (m_memory.intcStatus() & bit) == 0u)
        {
            continue;
        }
        if (!ps2_syscalls::isIntcCauseEnabled(
                this, cause))
        {
            retained |= bit;
            continue;
        }
        if ((m_memory.intcMask() & bit) == 0u)
        {
            retained |= bit;
            continue;
        }
        ps2_syscalls::dispatchIntcHandlersForCause(
            rdram, this, cause);
    }
    m_pendingEeCounterInterrupts |= retained;
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    if (!m_dmacInterruptDeliveryDirty.exchange(
            false, std::memory_order_acq_rel))
    {
        return;
    }
    m_dmacInterruptDrainPasses.fetch_add(
        1u, std::memory_order_relaxed);
    collectPendingDmacInterrupts();

    // A delivered handler can acknowledge this completion, start another DMA,
    // and publish that completion before it returns. Drain a detached batch so
    // such re-entrant publications append to stable pending storage instead of
    // invalidating this traversal or being overwritten below.
    std::vector<DmacPendingInterrupt> pending;
    pending.swap(m_pendingDmacInterrupts);
    std::vector<DmacPendingInterrupt> retained;
    retained.reserve(pending.size());
    for (const DmacPendingInterrupt &interrupt :
         pending)
    {
        if (!m_memory.dmacInterruptStatusLatched(
                interrupt.source))
        {
            // The guest acknowledged the level before handler delivery.
            continue;
        }
        if (!m_memory.dmacInterruptUnmasked(
                interrupt.source) ||
            !ps2_syscalls::isDmacCauseEnabled(
                this, interrupt.cause))
        {
            retained.push_back(interrupt);
            continue;
        }
        ps2_syscalls::dispatchDmacHandlersForCause(
            rdram, this, interrupt.cause);
    }

    if (!retained.empty())
    {
        std::vector<DmacPendingInterrupt> reentrant;
        reentrant.swap(m_pendingDmacInterrupts);
        m_pendingDmacInterrupts = std::move(retained);
        mergePendingDmacInterrupts(
            std::move(reentrant));
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Level1Exception(
        ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    EeTlbEntry entry{};

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, entry))
    {
        raiseCop0Level1Exception(
            ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    ctx->cop0_pagemask = entry.pageMask;
    ctx->cop0_entryhi = entry.entryHi;
    ctx->cop0_entrylo0 = entry.entryLo0;
    ctx->cop0_entrylo1 = entry.entryLo1;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const EeTlbEntry entry{
        ctx->cop0_pagemask,
        ctx->cop0_entryhi,
        ctx->cop0_entrylo0,
        ctx->cop0_entrylo1,
    };

    if (!m_memory.tlbWrite(index, entry))
    {
        raiseCop0Level1Exception(
            ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Level1Exception(
            ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const EeTlbEntry entry{
        ctx->cop0_pagemask,
        ctx->cop0_entryhi,
        ctx->cop0_entrylo0,
        ctx->cop0_entrylo1,
    };

    if (!m_memory.tlbWrite(random, entry))
    {
        raiseCop0Level1Exception(
            ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    // Automatic Random retirement occurs after this instruction consumes
    // the selected slot, at the shared EE instruction boundary.
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index =
        m_memory.tlbProbe(ctx->cop0_entryhi);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

void PS2Runtime::executeGuestStep(uint8_t *rdram,
                                  R5900Context *ctx,
                                  RecompiledFunction function)
{
    struct CodeInvalidationFlush
    {
        PS2Memory &memory;
        ~CodeInvalidationFlush()
        {
            try
            {
                memory.flushCodeInvalidationLog();
            }
            catch (...)
            {
            }
        }
    } flush{m_memory};

    // Outer dispatch is an architectural publication boundary. Generated
    // fast code may return with a completed straight-line retirement run, so
    // materialize Random before either debugger hook can snapshot the context.
    // Address normalization is architectural dispatch work, not debugger
    // bookkeeping, and must remain in diagnostics-free builds.
    if (ctx)
    {
        ctx->finishEeInstruction();
        ctx->pc = normalizeGuestFunctionAddress(ctx->pc);
    }
    debugBeforeGuestStep(ctx);
    if (isStopRequested())
    {
        return;
    }

    try
    {
        if (ctx)
        {
            ValidateInstructionFetch(ctx, ctx->pc);
        }
        function(rdram, ctx, this);
    }
    catch (const PS2GuestException &)
    {
    }
    catch (...)
    {
        if (ctx)
        {
            ctx->finishEeInstruction();
        }
        debugAfterGuestStep(ctx);
        throw;
    }
    if (ctx)
    {
        ctx->finishEeInstruction();
    }
    debugAfterGuestStep(ctx);
}

void PS2Runtime::dispatchLoop(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t lastPc = std::numeric_limits<uint32_t>::max();
    uint32_t samePcCount = 0;
    constexpr uint32_t kSamePcYieldInterval = 0x4000u;

    while (!isStopRequested())
    {
        try
        {
            ValidateInstructionFetch(ctx, ctx->pc);
        }
        catch (const PS2GuestException &)
        {
            continue;
        }
        const uint32_t pc = normalizeGuestFunctionAddress(ctx->pc);
        ctx->pc = pc;

        if (pc == lastPc)
        {
            ++samePcCount;
            if ((samePcCount % kSamePcYieldInterval) == 0u)
            {
                PS2_IF_AGRESSIVE_LOGS({
                    RUNTIME_LOG("CPU is doing some work at PC " << formatGuestPc(pc) << ". PC not updating.");
                });
                std::this_thread::yield();
            }
        }
        else
        {
            samePcCount = 0;
            lastPc = pc;
        }

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
        m_debugPc.store(pc, std::memory_order_relaxed);
        m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)), std::memory_order_relaxed);
        m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)), std::memory_order_relaxed);
        m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)), std::memory_order_relaxed);
#endif

        RecompiledFunction fn = lookupFunction(pc, ctx);
        const uint32_t dispatchedPc = pc;
        const uint32_t dispatchedRa = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));

        uint64_t handoffBaseline = 0u;
        std::shared_ptr<ThreadInfo>
            returnedThreadInfo;
        {
            GuestExecutionScope guestExecution(this, ctx);
            executeGuestStep(rdram, ctx, fn);
            if (!m_eeRuntimeExecutor &&
                ctx->pc == 0u)
            {
                EeThreadRuntimeState &state =
                    *m_eeThreadRuntimeState;
                {
                    std::lock_guard<std::mutex> lock(
                        state.threadMapMutex);
                    const auto context =
                        state.contextThreadIds.find(ctx);
                    if (context !=
                        state.contextThreadIds.end())
                    {
                        const auto thread =
                            state.threads.find(
                                context->second);
                        if (thread !=
                            state.threads.end())
                        {
                            returnedThreadInfo =
                                thread->second;
                        }
                    }
                }
                if (returnedThreadInfo)
                {
                    std::lock_guard<std::mutex> lock(
                        returnedThreadInfo->m);
                    returnedThreadInfo->guestState
                        .makeDormant();
                }
            }
            handoffBaseline = guestExecutionHandoffEpochSnapshot();
        }
        if (returnedThreadInfo)
        {
            returnedThreadInfo->cv.notify_all();
        }

        waitForGuestExecutionHandoff(handoffBaseline);

        if (ctx->pc == 0u)
        {
            const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
            const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
            const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
            RUNTIME_LOG(
                "[dispatch:pc-zero] from="
                << formatGuestPc(dispatchedPc)
                << " fromRa=" << formatGuestPc(dispatchedRa)
                << " ra=" << formatGuestPc(ra)
                << " sp=0x" << std::hex << sp
                << " gp=0x" << gp
                << " trace=" << formatDispatchHistory()
                << std::dec << std::endl);

            // PC=0 means this guest thread returned (usually via jr $ra with RA=0).
            // Do not request a global runtime stop here: other guest threads may still run.
            break;
        }
    }
}

int PS2Runtime::threadIdForEeContext(
    const R5900Context *context) noexcept
{
    if (!context)
    {
        return 0;
    }
    if (context == &m_cpuContext)
    {
        return 1;
    }

    std::lock_guard<std::mutex> lock(
        m_eeThreadRuntimeState->threadMapMutex);
    const auto it =
        m_eeThreadRuntimeState->contextThreadIds.find(context);
    return it !=
                   m_eeThreadRuntimeState->contextThreadIds.end()
               ? it->second
               : 0;
}

static void copyEeProcessorGlobalState(
    R5900Context &destination,
    const R5900Context &source) noexcept
{
    copyEeCop2ProcessorGlobalState(destination, source);

    destination.cop0_index = source.cop0_index;
    destination.cop0_random = source.cop0_random;
    destination.cop0_entrylo0 = source.cop0_entrylo0;
    destination.cop0_entrylo1 = source.cop0_entrylo1;
    destination.cop0_context = source.cop0_context;
    destination.cop0_pagemask = source.cop0_pagemask;
    destination.cop0_wired = source.cop0_wired;
    destination.cop0_badvaddr = source.cop0_badvaddr;
    destination.cop0_count = source.cop0_count;
    destination.cop0_entryhi = source.cop0_entryhi;
    destination.cop0_compare = source.cop0_compare;
    destination.cop0_status = source.cop0_status;
    destination.cop0_cause = source.cop0_cause;
    destination.cop0_epc = source.cop0_epc;
    destination.cop0_prid = source.cop0_prid;
    destination.cop0_config = source.cop0_config;
    destination.cop0_badpaddr = source.cop0_badpaddr;
    destination.cop0_bpc = source.cop0_bpc;
    destination.cop0_iab = source.cop0_iab;
    destination.cop0_iabm = source.cop0_iabm;
    destination.cop0_dab = source.cop0_dab;
    destination.cop0_dabm = source.cop0_dabm;
    destination.cop0_dvb = source.cop0_dvb;
    destination.cop0_dvbm = source.cop0_dvbm;
    destination.cop0_perf = source.cop0_perf;
    destination.cop0_pcr0 = source.cop0_pcr0;
    destination.cop0_pcr1 = source.cop0_pcr1;
    destination.cop0_taglo = source.cop0_taglo;
    destination.cop0_taghi = source.cop0_taghi;
    destination.cop0_errorepc = source.cop0_errorepc;

}

void PS2Runtime::publishEeProcessorGlobalState(
    R5900Context *context) noexcept
{
    if (!context ||
        context == &m_cpuContext ||
        threadIdForEeContext(context) <= 0)
    {
        return;
    }
    copyEeProcessorGlobalState(
        m_cpuContext, *context);
}

void PS2Runtime::restoreEeProcessorGlobalState(
    R5900Context *context) noexcept
{
    if (!context ||
        context == &m_cpuContext ||
        threadIdForEeContext(context) <= 0)
    {
        return;
    }
    copyEeProcessorGlobalState(
        *context, m_cpuContext);
}

void PS2Runtime::selectEeThread(int threadId) noexcept
{
    if (threadId < 0)
    {
        threadId = 1;
    }
    m_eeThreadRuntimeState->currentThreadId.store(
        threadId, std::memory_order_release);
    m_boundEeThreadId = threadId;
    g_currentThreadRuntime = this;
    g_currentThreadId = threadId;
}

void PS2Runtime::switchGuestExecutionContext(
    R5900Context *context,
    int selectionHint) noexcept
{
    if (!context || context == m_boundEeContext)
    {
        return;
    }

    R5900Context *const previous =
        m_boundEeContext;
    int selectedThreadId = selectionHint > 0
                               ? selectionHint
                               : threadIdForEeContext(
                                     context);
    if (selectedThreadId <= 0)
    {
        if (previous)
        {
            selectedThreadId = m_boundEeThreadId;
        }
        else
        {
            selectedThreadId = 1;
        }
    }

    const ps2x::timing::EeTick now =
        finishEeContextBlock(previous);
    synchronizeCop0Timing(
        previous,
        ps2x::timing::eeTickToCyclesFloor(
            now));
    publishEeProcessorGlobalState(previous);
    restoreEeProcessorGlobalState(context);
    context->resetEeBlockTiming();
    m_boundEeContext = context;
    selectEeThread(selectedThreadId);
    publishCop0TimingMirrors(context);
    refreshCop0EventSchedules(
        context,
        ps2x::timing::eeTickToCyclesFloor(
            now));
}

void PS2Runtime::restoreGuestExecutionContext(
    R5900Context *context,
    R5900Context *previousContext,
    int previousThreadId) noexcept
{
    if (!context ||
        context == previousContext ||
        m_boundEeContext != context)
    {
        return;
    }

    const ps2x::timing::EeTick now =
        finishEeContextBlock(context);
    synchronizeCop0Timing(
        context,
        ps2x::timing::eeTickToCyclesFloor(
            now));
    publishEeProcessorGlobalState(context);
    restoreEeProcessorGlobalState(previousContext);
    m_boundEeContext = previousContext;
    selectEeThread(previousThreadId);
    publishCop0TimingMirrors(previousContext);
    refreshCop0EventSchedules(
        previousContext,
        ps2x::timing::eeTickToCyclesFloor(
            now));
}

R5900Context *PS2Runtime::enterGuestExecution(
    R5900Context *context,
    int *previousThreadId,
    int selectionHint)
{
    uint32_t &depth = g_guestExecutionDepths[this];

    if (m_eeRuntimeExecutor)
    {
        if (!m_eeRuntimeExecutor->ownsCurrentThread())
        {
            throw std::logic_error(
                "dedicated EE guest execution must run "
                "on its executor thread");
        }

        R5900Context *const previousContext =
            m_boundEeContext;
        if (previousThreadId)
        {
            *previousThreadId =
                m_boundEeThreadId;
        }
        ++depth;
        if (depth == 1u)
        {
            recordOuterGuestExecutionAcquisition(
                context);
        }
        switchGuestExecutionContext(
            context, selectionHint);
        return previousContext;
    }

    if (depth != 0u)
    {
        lockGuestExecutionMutex(false);
        ++depth;
        R5900Context *const previousContext =
            m_boundEeContext;
        if (previousThreadId)
        {
            *previousThreadId =
                m_boundEeThreadId;
        }
        switchGuestExecutionContext(
            context, selectionHint);
        return previousContext;
    }

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (!m_debugControlActive.load(
            std::memory_order_acquire))
    {
#endif
        m_guestExecutionWaiters.fetch_add(1u, std::memory_order_acq_rel);
        lockGuestExecutionMutex(true);
        m_guestExecutionWaiters.fetch_sub(1u, std::memory_order_acq_rel);
        depth = 1u;
        recordOuterGuestExecutionAcquisition(context);
        markGuestExecutionAcquired();
        flushDeferredLegacyEeWakeNotifications();
        R5900Context *const previousContext =
            m_boundEeContext;
        if (previousThreadId)
        {
            *previousThreadId =
                m_boundEeThreadId;
        }
        switchGuestExecutionContext(
            context, selectionHint);
        return previousContext;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    }

    for (;;)
    {
        {
            std::unique_lock<std::mutex> controlLock(m_debugControlMutex);
            m_debugControlCv.wait(controlLock, [this]()
                                  { return !m_debugPauseRequested.load(std::memory_order_acquire) ||
                                           isStopRequested(); });
        }

        m_guestExecutionWaiters.fetch_add(1u, std::memory_order_acq_rel);
        lockGuestExecutionMutex(true);
        m_guestExecutionWaiters.fetch_sub(1u, std::memory_order_acq_rel);

        // Close the race where a pause is requested after the condition check
        // but before this thread acquires the guest-execution mutex.
        if (!m_debugPauseRequested.load(std::memory_order_acquire) ||
            isStopRequested())
        {
            break;
        }
        m_guestExecutionMutex.unlock();
    }
    depth = 1u;
    recordOuterGuestExecutionAcquisition(context);
    markGuestExecutionAcquired();
    flushDeferredLegacyEeWakeNotifications();
    R5900Context *const previousContext =
        m_boundEeContext;
    if (previousThreadId)
    {
        *previousThreadId =
            m_boundEeThreadId;
    }
    switchGuestExecutionContext(
        context, selectionHint);
    return previousContext;
#endif
}

bool PS2Runtime::tryEnterGuestExecution(
    R5900Context *context,
    std::chrono::milliseconds timeout,
    R5900Context *&previousContext,
    int &previousThreadId,
    int selectionHint)
{
    previousContext = nullptr;
    previousThreadId = 1;

    if (m_eeRuntimeExecutor)
    {
        if (!m_eeRuntimeExecutor->ownsCurrentThread())
        {
            return false;
        }
        previousContext =
            enterGuestExecution(
                context,
                &previousThreadId,
                selectionHint);
        return true;
    }

    auto depthIt = g_guestExecutionDepths.find(this);
    if (depthIt != g_guestExecutionDepths.end() &&
        depthIt->second != 0u)
    {
        if (!m_guestExecutionMutex.try_lock_for(timeout))
        {
            return false;
        }
        ++depthIt->second;
        previousContext = m_boundEeContext;
        previousThreadId = m_boundEeThreadId;
        switchGuestExecutionContext(
            context, selectionHint);
        return true;
    }

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugControlActive.load(std::memory_order_acquire) &&
        m_debugPauseRequested.load(std::memory_order_acquire))
    {
        return false;
    }
#endif

    m_guestExecutionWaiters.fetch_add(
        1u, std::memory_order_acq_rel);
    const bool acquired =
        m_guestExecutionMutex.try_lock_for(timeout);
    m_guestExecutionWaiters.fetch_sub(
        1u, std::memory_order_acq_rel);
    if (!acquired)
    {
        return false;
    }

    // Close the same debugger-pause race as the blocking acquisition path.
    // The caller owns no guest state until after this check.
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_debugControlActive.load(std::memory_order_acquire) &&
        m_debugPauseRequested.load(std::memory_order_acquire))
    {
        m_guestExecutionMutex.unlock();
        return false;
    }
#endif

    g_guestExecutionDepths[this] = 1u;
    recordOuterGuestExecutionAcquisition(context);
    markGuestExecutionAcquired();
    flushDeferredLegacyEeWakeNotifications();
    previousContext = m_boundEeContext;
    previousThreadId = m_boundEeThreadId;
    switchGuestExecutionContext(
        context, selectionHint);
    return true;
}

void PS2Runtime::lockGuestExecutionMutex(bool mayContend)
{
    if (!m_eeThreadDiagnosticsEnabled)
    {
        m_guestExecutionMutex.lock();
        return;
    }

    m_eeThreadDiagnosticGuestLockRequests.fetch_add(
        1u, std::memory_order_relaxed);
    if (mayContend && !m_guestExecutionMutex.try_lock())
    {
        m_eeThreadDiagnosticGuestLockContentions.fetch_add(
            1u, std::memory_order_relaxed);
        m_guestExecutionMutex.lock();
    }
    else if (!mayContend)
    {
        m_guestExecutionMutex.lock();
    }
    m_eeThreadDiagnosticGuestLockAcquisitions.fetch_add(
        1u, std::memory_order_relaxed);
}

void PS2Runtime::recordOuterGuestExecutionAcquisition(
    R5900Context *context) noexcept
{
    if (!m_eeThreadDiagnosticsEnabled)
    {
        return;
    }

    m_eeThreadDiagnosticOuterGuestExecutionAcquisitions.fetch_add(
        1u, std::memory_order_relaxed);
    if (m_eeThreadDiagnosticsHasLastOuterContext &&
        m_eeThreadDiagnosticsLastOuterContext != context)
    {
        m_eeThreadDiagnosticGuestContextChanges.fetch_add(
            1u, std::memory_order_relaxed);
    }
    m_eeThreadDiagnosticsLastOuterContext = context;
    m_eeThreadDiagnosticsHasLastOuterContext = true;
}

void PS2Runtime::leaveGuestExecution(
    R5900Context *context,
    R5900Context *previousContext,
    int previousThreadId)
{
    auto it = g_guestExecutionDepths.find(this);
    if (it == g_guestExecutionDepths.end() || it->second == 0u)
    {
        return;
    }

    const bool legacyBoundary =
        it->second == 1u && !m_eeRuntimeExecutor;
    const bool hasPublishedCallback =
        legacyBoundary &&
        (m_eeAlarmRuntimeState
                 ->pendingCallbackPublished.load(
                     std::memory_order_acquire) ||
         m_pendingVSyncDeliveryCount != 0u ||
         m_pendingEeCounterInterrupts != 0u ||
         m_dmacInterruptDeliveryDirty.load(
             std::memory_order_acquire));
    if (hasPublishedCallback)
    {
        captureEeInterruptContinuation(
            context, m_boundEeThreadId);
    }
    else if (legacyBoundary)
    {
        clearEeInterruptContinuation();
    }

    restoreGuestExecutionContext(
        context,
        previousContext,
        previousThreadId);

    // The legacy host-thread backend drains published alarm/device work after
    // the outermost recompiled invocation returns. The dedicated executor
    // drains the same work from applyNextConsequence instead because a
    // suspended fiber can retain this scope across scheduler boundaries.
    if (hasPublishedCallback)
    {
        try
        {
            drainPendingAlarmCallbacks();
            drainPendingVSyncHandlers(
                m_memory.getRDRAM());
            drainPendingEeCounterHandlers(
                m_memory.getRDRAM());
            drainCompletedDmacHandlers(
                m_memory.getRDRAM());
        }
        catch (...)
        {
            clearEeInterruptContinuation();
            throw;
        }
        clearEeInterruptContinuation();
        acknowledgeGuestPreemptionRequests();
        it = g_guestExecutionDepths.find(this);
        if (it == g_guestExecutionDepths.end() || it->second == 0u)
        {
            return;
        }
    }

    if (it->second == 1u &&
        !m_eeRuntimeExecutor &&
        (isStopRequested() ||
         (context && context->pc == 0u)))
    {
        flushDeferredLegacyEeWakeNotifications();
    }

    --it->second;
    if (!m_eeRuntimeExecutor)
    {
        m_guestExecutionMutex.unlock();
    }
    if (it->second == 0u)
    {
        g_guestExecutionDepths.erase(it);
    }
}

uint32_t PS2Runtime::releaseGuestExecution(
    R5900Context *&context,
    int &threadId)
{
    context = nullptr;
    threadId = 1;
    auto it = g_guestExecutionDepths.find(this);
    if (it == g_guestExecutionDepths.end() || it->second == 0u)
    {
        return 0u;
    }

    if (!m_eeRuntimeExecutor)
    {
        flushDeferredLegacyEeWakeNotifications();
    }

    context = m_boundEeContext;
    threadId = m_boundEeThreadId;
    const ps2x::timing::EeTick now =
        finishEeContextBlock(m_boundEeContext);
    synchronizeCop0Timing(
        m_boundEeContext,
        ps2x::timing::eeTickToCyclesFloor(
            now));
    publishEeProcessorGlobalState(
        m_boundEeContext);
    m_boundEeContext = nullptr;
    m_boundEeThreadId = 1;
    m_eeThreadRuntimeState->currentThreadId.store(
        1, std::memory_order_release);

    const uint32_t depth = it->second;
    if (!m_eeRuntimeExecutor)
    {
        for (uint32_t i = 0; i < depth; ++i)
        {
            m_guestExecutionMutex.unlock();
        }
    }
    g_guestExecutionDepths.erase(it);
    return depth;
}

void PS2Runtime::reacquireGuestExecution(
    uint32_t depth,
    R5900Context *context,
    int threadId)
{
    if (depth == 0u)
    {
        return;
    }

    uint32_t &heldDepth = g_guestExecutionDepths[this];
    if (m_eeRuntimeExecutor)
    {
        if (!m_eeRuntimeExecutor->ownsCurrentThread() ||
            heldDepth != 0u)
        {
            throw std::logic_error(
                "dedicated EE logical ownership could "
                "not be restored");
        }
        heldDepth = depth;
        switchGuestExecutionContext(
            context, threadId);
        return;
    }

    uint32_t remaining = depth;

    if (heldDepth == 0u)
    {
        (void)enterGuestExecution(
            context, nullptr, threadId);
        heldDepth = g_guestExecutionDepths[this];
        --remaining;
    }
    else
    {
        switchGuestExecutionContext(
            context, threadId);
    }

    for (uint32_t i = 0; i < remaining; ++i)
    {
        lockGuestExecutionMutex(false);
        ++heldDepth;
    }
}

void PS2Runtime::markGuestExecutionAcquired()
{
    if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticHandoffNotifications.fetch_add(
            1u, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lock(m_guestExecutionHandoffMutex);
        m_guestExecutionHandoffEpoch.fetch_add(1u, std::memory_order_acq_rel);
    }
    m_guestExecutionHandoffCv.notify_all();
}

bool PS2Runtime::shouldDeferLegacyEeWakeNotification()
    const noexcept
{
    if (m_eeRuntimeExecutor)
    {
        return false;
    }
    const auto it = g_guestExecutionDepths.find(
        const_cast<PS2Runtime *>(this));
    return it != g_guestExecutionDepths.end() &&
           it->second != 0u;
}

void PS2Runtime::deferLegacyEeWakeNotification(
    const std::shared_ptr<ThreadInfo> &thread)
{
    if (!thread || m_eeRuntimeExecutor)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(
        m_legacyDeferredEeWakeMutex);
    m_legacyDeferredEeWakeNotifications.emplace_back(
        thread);
}

void PS2Runtime::flushDeferredLegacyEeWakeNotifications()
{
    if (m_eeRuntimeExecutor)
    {
        return;
    }

    std::vector<std::weak_ptr<ThreadInfo>> pending;
    {
        std::lock_guard<std::mutex> lock(
            m_legacyDeferredEeWakeMutex);
        pending.swap(
            m_legacyDeferredEeWakeNotifications);
    }
    for (const std::weak_ptr<ThreadInfo> &entry : pending)
    {
        if (const std::shared_ptr<ThreadInfo> thread =
                entry.lock())
        {
            thread->cv.notify_one();
        }
    }
}

void PS2Runtime::waitForGuestExecutionHandoff()
{
    waitForGuestExecutionHandoff(guestExecutionHandoffEpochSnapshot());
}

void PS2Runtime::waitForGuestExecutionHandoff(uint64_t baselineEpoch)
{
    if (m_eeRuntimeExecutor)
    {
        static_cast<void>(baselineEpoch);
        if (g_eeExecutorConsequenceContext.runtime ==
            this)
        {
            // Callback code is already running between continuations on the
            // executor stack. Any scheduler publication it made will be
            // consumed by this same boundary; there is no guest fiber to
            // suspend here.
            return;
        }
        yieldEeExecutorCurrent(
            isStopRequested()
                ? ps2x::ee::EeSchedulerExitReason::
                      StopRequested
                : ps2x::ee::EeSchedulerExitReason::
                      Preempted);
        return;
    }

    if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticHandoffWaitRequests.fetch_add(
            1u, std::memory_order_relaxed);
    }

    // Lock-free fast path
    if (m_guestExecutionWaiters.load(std::memory_order_acquire) == 0u)
    {
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticHandoffWaitFastPaths.fetch_add(
                1u, std::memory_order_relaxed);
        }
        return;
    }

    std::unique_lock<std::mutex> lock(m_guestExecutionHandoffMutex);

    if (m_guestExecutionWaiters.load(std::memory_order_acquire) == 0u)
    {
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticHandoffWaitFastPaths.fetch_add(
                1u, std::memory_order_relaxed);
        }
        return;
    }

    if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticHandoffCvWaits.fetch_add(
            1u, std::memory_order_relaxed);
    }
    const bool handedOff = m_guestExecutionHandoffCv.wait_for(
        lock,
        std::chrono::milliseconds(2),
        [&]()
        {
            return m_guestExecutionWaiters.load(std::memory_order_acquire) == 0u ||
                   m_guestExecutionHandoffEpoch.load(std::memory_order_relaxed) != baselineEpoch ||
                   isStopRequested();
        });

    if (!handedOff)
    {
        m_guestExecutionHandoffTimeouts.fetch_add(1u, std::memory_order_relaxed);
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticHandoffTimeouts.fetch_add(
                1u, std::memory_order_relaxed);
        }
    }
    else if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticHandoffCompletions.fetch_add(
            1u, std::memory_order_relaxed);
    }
}

void PS2Runtime::debugRecordStopLocked(
    const char *reason,
    uint32_t pc,
    uint64_t observedControlGeneration)
{
    ++m_debugStopSequence;
    m_debugLastStop.completed = true;
    m_debugLastStop.reason = reason ? reason : "unknown";
    m_debugLastStop.pauseSource.clear();
    m_debugLastStop.pc = normalizeGuestFunctionAddress(pc);
    m_debugLastStop.sequence = m_debugStopSequence;
    m_debugLastStop.controlGeneration =
        m_debugControlGeneration;
    m_debugLastStop.observedControlGeneration =
        observedControlGeneration;
    m_debugLastStop.staleBoundaryStops =
        m_debugStaleBoundaryStops;
}

void PS2Runtime::debugAdvanceControlGenerationLocked(
    const char *writer)
{
    ++m_debugControlGeneration;
    m_debugPauseWriter =
        writer ? writer : "unknown";
}

void PS2Runtime::debugSetPauseRequestedLocked(
    bool requested,
    const char *writer)
{
    debugAdvanceControlGenerationLocked(writer);
    m_debugPauseRequested.store(
        requested, std::memory_order_release);
}

void PS2Runtime::debugRefreshControlActiveLocked()
{
    const bool active =
        m_debugPauseRequested.load(std::memory_order_relaxed) ||
        m_debugRunUntilActive ||
        m_debugRunUntilMpegUniquePicturesActive ||
        m_debugRunBeforeNextMpegUniquePictureActive ||
        m_debugRunForVSyncFieldsActive ||
        m_debugStepActive ||
        !m_debugBreakpoints.empty() ||
        !m_debugWatchpoints.empty();
    m_debugControlActive.store(active, std::memory_order_release);
}

void PS2Runtime::debugPublishVuBackendDiagnostics()
{
    const uint64_t sequence =
        ++m_debugVuBackendDiagnosticsSequence;
    const auto convertSnapshot =
        [sequence](
            const VuExecutionState &state,
            VuExitReason lastExitReason,
            const VuVerifyDiagnostics &verify,
            const VuProgramCacheDiagnostics *cacheSnapshot,
            const Vu1RecompilerSnapshot *recompilerSnapshot)
        {
            DebugVuBackendDiagnostics result{};
            result.snapshotSequence = sequence;
            result.issuedCycles = state.issuedCycles;
            result.pc = state.pc;
            result.lastExitReason = lastExitReason;
            result.captured = true;
            result.verify = {
                .runs = verify.runs,
                .comparedPairs = verify.comparedPairs,
                .publishedPairs = verify.publishedPairs,
                .publishedPath1Packets =
                    verify.publishedPath1Packets,
                .mismatches = verify.mismatches,
                .lastMismatch = verify.lastMismatch,
            };

            if (cacheSnapshot)
            {
                const VuProgramCacheDiagnostics &source =
                    *cacheSnapshot;
                result.cacheCreated = true;
                result.cache = {
                    .hits = source.hits,
                    .misses = source.misses,
                    .compilations = source.compilations,
                    .invalidations = source.invalidations,
                    .invalidatedPrograms =
                        source.invalidatedPrograms,
                    .generationRetentions =
                        source.generationRetentions,
                    .retainedPrograms =
                        source.retainedPrograms,
                    .crossGenerationHits =
                        source.crossGenerationHits,
                    .evictionFlushes = source.evictionFlushes,
                    .selectiveEvictions =
                        source.selectiveEvictions,
                    .evictedPrograms = source.evictedPrograms,
                    .manualFlushes = source.manualFlushes,
                    .rejectedPrograms = source.rejectedPrograms,
                    .linkResolutions =
                        source.linkResolutions,
                    .linkResolutionFailures =
                        source.linkResolutionFailures,
                    .linkInvalidations =
                        source.linkInvalidations,
                    .generatedBytes = source.generatedBytes,
                    .compilationNanoseconds =
                        source.compilationNanoseconds,
                    .residentPrograms = static_cast<uint64_t>(
                        source.residentPrograms),
                    .residentExecutableBytes =
                        static_cast<uint64_t>(
                            source.residentExecutableBytes),
                    .highWaterPrograms = static_cast<uint64_t>(
                        source.highWaterPrograms),
                    .highWaterExecutableBytes =
                        static_cast<uint64_t>(
                            source.highWaterExecutableBytes),
                    .maximumPrograms = static_cast<uint64_t>(
                        source.maximumPrograms),
                    .maximumExecutableBytes =
                        static_cast<uint64_t>(
                            source.maximumExecutableBytes),
                };
            }

            if (recompilerSnapshot)
            {
                const VuRecompilerDiagnostics &source =
                    recompilerSnapshot->diagnostics;
                result.recompilerCreated = true;
                result.recompiler = {
                    .blockLinkingEnabled =
                        recompilerSnapshot->
                            blockLinkingEnabled,
                    .blockBudgetGuardsEnabled =
                        recompilerSnapshot->
                            blockBudgetGuardsEnabled,
                    .blockLocalVfRegistersEnabled =
                        recompilerSnapshot->
                            blockLocalVfRegistersEnabled,
                    .blockLocalVfRegistersAutomatic =
                        recompilerSnapshot->
                            blockLocalVfRegistersAutomatic,
                    .inlineXgkickEnabled =
                        recompilerSnapshot->
                            inlineXgkickEnabled,
                    .nativeEntries = source.nativeEntries,
                    .nativeBlocks = source.nativeBlocks,
                    .nativePairs = source.nativePairs,
                    .instrumentedNativeEntries =
                        source.instrumentedNativeEntries,
                    .instrumentedNativeBlocks =
                        source.instrumentedNativeBlocks,
                    .instrumentedNativePairs =
                        source.instrumentedNativePairs,
                    .inlinePairs = source.inlinePairs,
                    .helperPairs = source.helperPairs,
                    .linkedEdges = source.linkedEdges,
                    .slowLinkExits =
                        source.slowLinkExits,
                    .resolvedLinks =
                        source.resolvedLinks,
                    .abandonedLinkResolutions =
                        source.abandonedLinkResolutions,
                    .incompatibleLinkExits =
                        source.incompatibleLinkExits,
                    .fullBlockGuards =
                        source.fullBlockGuards,
                    .preciseTailEntries =
                        source.preciseTailEntries,
                    .fullGuardPairs =
                        source.fullGuardPairs,
                    .preciseTailPairs =
                        source.preciseTailPairs,
                    .budgetGuardComparisons =
                        source.budgetGuardComparisons,
                    .nativeBudgetExits =
                        source.nativeBudgetExits,
                    .blockCompletes = source.blockCompletes,
                    .cycleBudgetExits =
                        source.cycleBudgetExits,
                    .xgkickExits = source.xgkickExits,
                    .xgkickAdvanceHelperCalls =
                        source.xgkickAdvanceHelperCalls,
                    .unsupportedExits = source.unsupportedExits,
                    .codeInvalidationExits =
                        source.codeInvalidationExits,
                    .faultExits = source.faultExits,
                    .nativeExitReasons =
                        source.nativeExitReasons,
                    .interpreterInstrumentationFallbacks =
                        source.interpreterInstrumentationFallbacks,
                    .interpreterFallbackPairs =
                        source.interpreterFallbackPairs,
                    .codeImageIdentities =
                        source.codeImageIdentities,
                    .codeImageReuses =
                        source.codeImageReuses,
                    .codeImageCatalogEvictions =
                        source.codeImageCatalogEvictions,
                    .jitDumpRegistrations =
                        source.jitDumpRegistrations,
                    .jitDumpFailures =
                        source.jitDumpFailures,
                    .lastJitDiagnostic =
                        recompilerSnapshot->lastJitDiagnostic,
                };

                const VuBlockProfilingSnapshot &profile =
                    recompilerSnapshot->blockProfile;
                auto &destination = result.recompiler;
                destination.blockProfileEnabled =
                    profile.enabled;
                destination.blockProfileMaximumRecords =
                    profile.maximumRecords;
                destination.blockProfileDroppedRecords =
                    profile.droppedRecords;
                destination.blockProfiles.reserve(
                    profile.blocks.size());
                for (const VuBlockProfileSnapshot &block :
                     profile.blocks)
                {
                    DebugVuRecompilerDiagnostics::
                        BlockProfile converted{
                            .compilationIdentity =
                                block.compilationIdentity,
                            .codeContentIdentity =
                                block.key.codeContentIdentity,
                            .compilationGeneration =
                                block.key.codeGeneration,
                            .nativeAddress =
                                block.nativeAddress,
                            .nativeBytes =
                                block.nativeBytes,
                            .entryPc =
                                block.key.entryPc,
                            .firstPc = block.firstPc,
                            .lastPc = block.lastPc,
                            .blockPairs =
                                block.blockPairs,
                            .fixedCycles =
                                block.fixedCycles,
                            .compilationMode =
                                block.key.compilationMode ==
                                        VuCompilationMode::
                                            Instrumented
                                    ? "instrumented"
                                    : "normal",
                            .blockForm =
                                block.key.blockForm ==
                                        VuIrBlockForm::
                                            Basic
                                    ? "basic"
                                    : "linear-trace",
                            .blockExit =
                                std::string(
                                    vuIrBlockExitName(
                                        block.blockExit)),
                            .resident = block.resident,
                            .executions =
                                block.executions,
                            .guestPairs =
                                block.guestPairs,
                            .fullBudgetEntries =
                                block.fullBudgetEntries,
                            .boundedEntries =
                                block.boundedEntries,
                            .linkedEdges =
                                block.linkedEdges,
                            .helperBarriers =
                                block.helperBarriers,
                            .residentVfRegisters =
                                block.residentVfRegisters,
                            .residentVfDirtyRegisters =
                                block.residentVfDirtyRegisters,
                            .residentVfDirtyLanes =
                                block.residentVfDirtyLanes,
                            .maximumLiveVfRegisters =
                                block.maximumLiveVfRegisters,
                            .vfAccesses =
                                block.vfAccesses,
                            .allocatedVfAccesses =
                                block.allocatedVfAccesses,
                            .vfRegisterLoads =
                                block.vfRegisterLoads,
                            .vfRegisterStores =
                                block.vfRegisterStores,
                            .vfRegisterSpills =
                                block.vfRegisterSpills,
                            .vfRegisterReloads =
                                block.vfRegisterReloads,
                            .registerMaterializations =
                                block.registerMaterializations,
                            .exitReasons =
                                block.exitReasons,
                        };
                    for (size_t opcode = 0u;
                         opcode <
                             block.opcodeOperations.size();
                         ++opcode)
                    {
                        const uint64_t operations =
                            block.opcodeOperations[opcode];
                        if (operations == 0u)
                            continue;
                        converted.opcodeCounts.push_back({
                            .name = std::string(
                                vuIrOpcodeName(
                                    static_cast<
                                        VuIrOpcode>(
                                        opcode))),
                            .operations = operations,
                        });
                    }
                    converted.jitRegistrations.reserve(
                        block.jitRegistrations.size());
                    for (const auto &registration :
                         block.jitRegistrations)
                    {
                        converted.jitRegistrations
                            .push_back({
                                .generation =
                                    registration.generation,
                                .codeIndex =
                                    registration.codeIndex,
                            });
                    }
                    destination.blockProfiles.push_back(
                        std::move(converted));
                }
            }
            return result;
        };

    const auto captureUnit =
        [&](const VuUnit &unit)
        {
            std::optional<VuProgramCacheDiagnostics> cache;
            if (const VuProgramCache *const source =
                    unit.programCacheIfCreated())
            {
                cache =
                    source->
                        diagnosticsWhileExecutionQuiescent();
            }
            std::optional<Vu1RecompilerSnapshot> recompiler;
            if (const VuRecompilerBackend *const source =
                    unit.recompilerIfCreated())
            {
                recompiler = Vu1RecompilerSnapshot{
                    .diagnostics = source->diagnostics(),
                    .blockProfile =
                        source->
                            blockProfilingSnapshotWhileExecutionQuiescent(),
                    .lastJitDiagnostic =
                        source->lastJitDiagnostic(),
                    .blockLinkingEnabled =
                        source->blockLinkingEnabled(),
                    .blockBudgetGuardsEnabled =
                        source->blockBudgetGuardsEnabled(),
                    .blockLocalVfRegistersEnabled =
                        source->blockLocalVfRegistersEnabled(),
                    .blockLocalVfRegistersAutomatic =
                        source->
                            blockLocalVfRegistersAutomatic(),
                    .inlineXgkickEnabled =
                        source->inlineXgkickEnabled(),
                };
            }
            return convertSnapshot(
                unit.state(), unit.lastExitReason(),
                unit.verifyDiagnostics(),
                cache ? &*cache : nullptr,
                recompiler ? &*recompiler : nullptr);
        };

    DebugVuBackendDiagnostics vu1Diagnostics{};
    if (m_threadedVu1Executor)
    {
        const std::shared_ptr<const Vu1Snapshot> snapshot =
            captureVu1OwnerDiagnosticSnapshot(true);
        const Vu1BackendDiagnosticsSnapshot *const backend =
            snapshot->backendDiagnostics
                ? &*snapshot->backendDiagnostics
                : nullptr;
        vu1Diagnostics = convertSnapshot(
            snapshot->state, snapshot->lastExitReason,
            snapshot->verify,
            backend && backend->programCache
                ? &*backend->programCache
                : nullptr,
            backend && backend->recompiler
                ? &*backend->recompiler
                : nullptr);
    }
    else
    {
        vu1Diagnostics = captureUnit(m_vu1);
    }

    const std::array<DebugVuBackendDiagnostics, 2u> snapshots{
        captureUnit(m_vu0),
        std::move(vu1Diagnostics),
    };
    std::lock_guard<std::mutex> lock(
        m_debugVuBackendDiagnosticsMutex);
    m_debugVuBackendDiagnostics = snapshots;
}

void PS2Runtime::debugWaitUntilResumed()
{
    std::unique_lock<std::mutex> lock(m_debugControlMutex);
    m_debugControlCv.wait(lock, [this]()
                          { return !m_debugPauseRequested.load(std::memory_order_acquire) ||
                                   isStopRequested(); });
}

bool PS2Runtime::debugBlockGuestAtBoundary(
    R5900Context *ctx,
    const char *reason,
    std::optional<uint64_t> observedControlGeneration)
{
    const uint32_t pc = ctx ? ctx->pc : m_debugPc.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (observedControlGeneration.has_value() &&
            *observedControlGeneration !=
                m_debugControlGeneration)
        {
            ++m_debugStaleBoundaryStops;
            return false;
        }
        const bool acknowledgesPause =
            reason && std::strcmp(reason, "pause") == 0;
        if (acknowledgesPause)
        {
            if (!m_debugPauseRequested.load(
                    std::memory_order_acquire))
            {
                ++m_debugStaleBoundaryStops;
                return false;
            }
        }
        else
        {
            debugSetPauseRequestedLocked(
                true,
                reason ? reason : "boundary-stop");
            m_debugRunUntilActive = false;
            m_debugRunUntilMpegUniquePicturesActive = false;
            m_debugRunBeforeNextMpegUniquePictureActive = false;
            m_debugRunForVSyncFieldsActive = false;
            m_debugStepActive = false;
            m_debugStepRemaining = 0u;
            debugRecordStopLocked(
                reason,
                pc,
                observedControlGeneration.value_or(0u));
        }
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(true);

    // Cache ownership remains on this GameThread. Publish one coherent copy
    // before the debugger thread is allowed to inspect the paused runtime.
    debugPublishVuBackendDiagnostics();

    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugRequestPause();
        yieldEeExecutorCurrent(
            ps2x::ee::EeSchedulerExitReason::
                Preempted);
        return true;
    }

    // executeGuestStep() is called while this host thread owns the recursive
    // guest-execution mutex. Release all levels so debugger snapshots can be
    // proven quiescent while the guest remains stopped.
    R5900Context *releasedContext = nullptr;
    int releasedThreadId = 1;
    const uint32_t depth =
        releaseGuestExecution(
            releasedContext, releasedThreadId);
    debugWaitUntilResumed();
    if (depth != 0u)
    {
        reacquireGuestExecution(
            depth,
            releasedContext,
            releasedThreadId);
    }
    return true;
}

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
void PS2Runtime::debugBeforeGuestStep(R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }

    const uint32_t pc = ctx->pc;
    const bool controlActive =
        m_debugControlActive.load(std::memory_order_acquire);
    if (!controlActive)
    {
        // Keep a useful process-wide trail at outer dispatch boundaries
        // without making every nested guest call contend on the shared ring.
        debugRecordBranch(pc);
    }
    m_debugPc.store(pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)),
                    std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)),
                    std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)),
                    std::memory_order_relaxed);
    m_debugEeInstructions.store(ctx->insn_count, std::memory_order_relaxed);
    m_debugDispatches.fetch_add(1u, std::memory_order_relaxed);

    if (!controlActive)
    {
        return;
    }

    const char *stopReason = nullptr;
    uint64_t stopControlGeneration = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (m_debugPauseRequested.load(std::memory_order_acquire))
        {
            stopReason = "pause";
        }
        else if (m_debugRunUntilActive && pc == m_debugRunUntilPc)
        {
            stopReason = "predicate";
        }
        else if (m_debugRunUntilMpegUniquePicturesActive &&
                 m_debugMpegUniquePicturesServed.load(
                     std::memory_order_relaxed) >=
                     m_debugRunUntilMpegUniquePicturesTarget)
        {
            stopReason = "mpeg-unique-pictures";
        }
        else if (m_debugRunForVSyncFieldsActive &&
                 m_debugVSyncFields.load(std::memory_order_relaxed) >=
                     m_debugRunForVSyncFieldsTarget)
        {
            stopReason = "vsync-fields";
        }
        else
        {
            const bool hasBreakpoint =
                std::binary_search(m_debugBreakpoints.begin(), m_debugBreakpoints.end(), pc);
            if (hasBreakpoint)
            {
                if (m_debugSkipBreakpoint && m_debugSkipBreakpointPc == pc)
                {
                    m_debugSkipBreakpoint = false;
                }
                else
                {
                    stopReason = "breakpoint";
                }
            }
            else if (m_debugSkipBreakpoint)
            {
                m_debugSkipBreakpoint = false;
            }
        }
        if (stopReason)
        {
            stopControlGeneration =
                m_debugControlGeneration;
        }
    }

    if (stopReason)
    {
        (void)debugBlockGuestAtBoundary(
            ctx, stopReason,
            stopControlGeneration);
    }
}

void PS2Runtime::debugAfterGuestStep(R5900Context *ctx)
{
    if (ctx)
    {
        m_debugRa.store(
            static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)),
            std::memory_order_relaxed);
        m_debugSp.store(
            static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)),
            std::memory_order_relaxed);
        m_debugGp.store(
            static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)),
            std::memory_order_relaxed);
        m_debugEeInstructions.store(
            ctx->insn_count, std::memory_order_relaxed);
    }

    if (!m_debugControlActive.load(std::memory_order_acquire))
    {
        return;
    }

    bool stopForStep = false;
    uint64_t stopControlGeneration = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (m_debugStepActive)
        {
            if (m_debugStepRemaining > 0u)
            {
                --m_debugStepRemaining;
            }
            stopForStep = m_debugStepRemaining == 0u;
            if (stopForStep)
            {
                stopControlGeneration =
                    m_debugControlGeneration;
            }
        }
    }

    if (stopForStep)
    {
        (void)debugBlockGuestAtBoundary(
            ctx, "step", stopControlGeneration);
    }
}
#endif

bool PS2Runtime::debugPause(
    std::chrono::milliseconds timeout,
    const char *source)
{
    m_debugControlActive.store(true, std::memory_order_release);
    bool newlyRequested = false;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (!m_debugPauseRequested.load(std::memory_order_acquire))
        {
            newlyRequested = true;
            m_debugPauseSource =
                source ? source : "unknown";
            debugSetPauseRequestedLocked(
                true, m_debugPauseSource.c_str());
            m_debugRunUntilActive = false;
            m_debugRunUntilMpegUniquePicturesActive = false;
            m_debugRunBeforeNextMpegUniquePictureActive = false;
            m_debugRunForVSyncFieldsActive = false;
            m_debugStepActive = false;
            m_debugStepRemaining = 0u;
            debugRecordStopLocked("pause", m_debugPc.load(std::memory_order_relaxed));
            m_debugLastStop.pauseSource =
                m_debugPauseSource;
            m_debugLastStop.pauseWriter =
                m_debugPauseWriter;
            debugRefreshControlActiveLocked();
        }
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(true);

    if (m_eeRuntimeExecutor &&
        m_eeRuntimeExecutor->running() &&
        !m_eeRuntimeExecutor->ownsCurrentThread())
    {
        requestGuestPreemption();
        m_eeRuntimeExecutor->debugRequestPause();
        const bool paused =
            m_eeRuntimeExecutor->debugWaitUntilPaused(
                timeout);
        if (paused)
        {
            debugPublishVuBackendDiagnostics();
            return true;
        }

        if (newlyRequested)
        {
            {
                std::lock_guard<std::mutex> lock(
                    m_debugControlMutex);
                debugSetPauseRequestedLocked(
                    false,
                    "debug-pause-executor-timeout");
                debugRefreshControlActiveLocked();
            }
            m_eeRuntimeExecutor->debugResume();
            m_debugControlCv.notify_all();
            m_audioBackend.setDebuggerPaused(false);
        }
        return false;
    }

    const bool locked =
        m_guestExecutionMutex.try_lock_for(timeout);
    if (locked)
    {
        debugPublishVuBackendDiagnostics();
        m_guestExecutionMutex.unlock();
        return true;
    }

    if (newlyRequested)
    {
        {
            std::lock_guard<std::mutex> lock(m_debugControlMutex);
            debugSetPauseRequestedLocked(
                false,
                "debug-pause-lock-timeout");
            debugRefreshControlActiveLocked();
        }
        m_debugControlCv.notify_all();
        m_audioBackend.setDebuggerPaused(false);
    }
    return false;
}

void PS2Runtime::debugResume()
{
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (m_debugLastStop.reason == "breakpoint")
        {
            m_debugSkipBreakpoint = true;
            m_debugSkipBreakpointPc = m_debugLastStop.pc;
        }
        m_debugRunUntilActive = false;
        m_debugRunUntilMpegUniquePicturesActive = false;
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        m_debugRunForVSyncFieldsActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugSetPauseRequestedLocked(
            false, "debug-resume");
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugResume();
    }
    m_audioBackend.setDebuggerPaused(false);
}

bool PS2Runtime::debugIsPaused() const
{
    return m_debugPauseRequested.load(std::memory_order_acquire);
}

bool PS2Runtime::debugSetVuBackend(
    VuUnitId unit, VuBackendKind backend,
    std::string *diagnostic,
    std::chrono::milliseconds timeout)
{
    const auto fail =
        [&](std::string message)
        {
            if (diagnostic)
                *diagnostic = std::move(message);
            return false;
        };
    if (!debugIsPaused())
    {
        return fail(
            "VU backend changes require paused guest execution");
    }

    const auto applyBackend =
        [this, unit, backend, diagnostic]()
        {
            if (unit == VuUnitId::Vu1 &&
                m_threadedVu1Executor)
            {
                const Vu1CommandResult result =
                    submitVu1Command(
                        Vu1SetBackendCommand{backend},
                        currentEeTick().raw(),
                        m_vu1ExecutionTiming.generation);
                const auto *const status =
                    std::get_if<Vu1BackendStatusResult>(
                        &result.payload);
                if (!status)
                {
                    if (diagnostic)
                    {
                        *diagnostic =
                            "VU1 owner returned no backend status";
                    }
                    return false;
                }
                if (diagnostic)
                    *diagnostic = status->diagnostic;
                return status->accepted;
            }

            VuUnit *target = nullptr;
            switch (unit)
            {
            case VuUnitId::Vu0:
                target = &m_vu0;
                break;
            case VuUnitId::Vu1:
                target = &m_vu1;
                break;
            }
            if (!target)
            {
                if (diagnostic)
                    *diagnostic = "unknown VU unit";
                return false;
            }
            return target->setBackend(backend, diagnostic);
        };

    if (m_eeRuntimeExecutor &&
        m_eeRuntimeExecutor->running() &&
        !m_eeRuntimeExecutor->ownsCurrentThread())
    {
        bool changed = false;
        const bool invoked =
            invokeEeExecutorTaskAtBoundary(
                [this, diagnostic, &changed, applyBackend]()
                {
                    if (!debugIsPaused())
                    {
                        return;
                    }

                    changed = applyBackend();
                    if (changed)
                    {
                        debugPublishVuBackendDiagnostics();
                        if (diagnostic)
                        {
                            diagnostic->clear();
                        }
                    }
                });
        if (!invoked)
        {
            return fail(
                "VU backend change could not reach "
                "the EE executor boundary");
        }
        if (!debugIsPaused())
        {
            return fail(
                "guest execution resumed before the VU backend change");
        }
        return changed;
    }

    std::unique_lock<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex, std::defer_lock);
    if (!lock.try_lock_for(timeout))
    {
        return fail(
            "timed out waiting for a guest safe point");
    }
    if (!debugIsPaused())
    {
        return fail(
            "guest execution resumed before the VU backend change");
    }

    if (!applyBackend())
        return false;

    debugPublishVuBackendDiagnostics();
    if (diagnostic)
        diagnostic->clear();
    return true;
}

PS2Runtime::DebugStopInfo PS2Runtime::debugRunUntilPc(
    uint32_t pc, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    pc = normalizeGuestFunctionAddress(pc);
    if (!debugPause(timeout, "run-until-pc-entry"))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (normalizeGuestFunctionAddress(m_debugPc.load(std::memory_order_relaxed)) == pc)
        {
            debugRecordStopLocked("already-matched", pc);
            return m_debugLastStop;
        }
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = true;
        m_debugRunUntilPc = pc;
        m_debugRunUntilMpegUniquePicturesActive = false;
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        m_debugRunForVSyncFieldsActive = false;
        m_debugStepActive = false;
        debugSetPauseRequestedLocked(
            false, "run-until-pc-start");
        debugRefreshControlActiveLocked();
    }
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugResume();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                const DebugStopInfo stop =
                    m_debugLastStop;
                lock.unlock();
                if (m_eeRuntimeExecutor &&
                    !m_eeRuntimeExecutor
                         ->debugWaitUntilPaused(
                             timeout))
                {
                    return {
                        false,
                        "pause-timeout",
                        stop.pc,
                        stop.sequence,
                    };
                }
                return stop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugRunUntilActive = false;
        debugAdvanceControlGenerationLocked(
            "run-until-pc-timeout-clear");
    }

    (void)debugPause(timeout, "run-until-pc-timeout");
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

PS2Runtime::DebugStopInfo PS2Runtime::debugRunUntilMpegUniquePictures(
    uint64_t target, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    if (!debugPause(timeout, "run-until-mpeg-entry"))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    const uint64_t current =
        m_debugMpegUniquePicturesServed.load(std::memory_order_relaxed);
    if (current > target)
    {
        return {false, "target-already-passed",
                m_debugPc.load(std::memory_order_relaxed), 0u};
    }
    if (current == target)
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        debugRecordStopLocked(
            "already-matched", m_debugPc.load(std::memory_order_relaxed));
        return m_debugLastStop;
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = false;
        m_debugRunUntilMpegUniquePicturesActive = true;
        m_debugRunUntilMpegUniquePicturesTarget = target;
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        m_debugRunForVSyncFieldsActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugSetPauseRequestedLocked(
            false, "run-until-mpeg-start");
        debugRefreshControlActiveLocked();
    }
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugResume();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                const DebugStopInfo stop = m_debugLastStop;
                lock.unlock();
                if (m_eeRuntimeExecutor &&
                    !m_eeRuntimeExecutor->debugWaitUntilPaused(timeout))
                {
                    return {false, "pause-timeout", stop.pc, stop.sequence};
                }
                return stop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugRunUntilMpegUniquePicturesActive = false;
        debugAdvanceControlGenerationLocked(
            "run-until-mpeg-timeout-clear");
    }

    (void)debugPause(timeout, "run-until-mpeg-timeout");
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

PS2Runtime::DebugStopInfo
PS2Runtime::debugRunUntilBeforeNextMpegUniquePicture(
    uint64_t current, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    if (!debugPause(
            timeout,
            "run-before-next-mpeg-entry"))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    if (m_debugMpegUniquePicturesServed.load(std::memory_order_relaxed) != current)
    {
        return {false, "target-not-current",
                m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = false;
        m_debugRunUntilMpegUniquePicturesActive = false;
        m_debugRunBeforeNextMpegUniquePictureActive = true;
        m_debugRunBeforeNextMpegUniquePictureCurrent = current;
        m_debugRunForVSyncFieldsActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugSetPauseRequestedLocked(
            false, "run-before-next-mpeg-start");
        debugRefreshControlActiveLocked();
    }
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugResume();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                const DebugStopInfo stop = m_debugLastStop;
                lock.unlock();
                if (m_eeRuntimeExecutor &&
                    !m_eeRuntimeExecutor->debugWaitUntilPaused(timeout))
                {
                    return {false, "pause-timeout", stop.pc, stop.sequence};
                }
                return stop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        debugAdvanceControlGenerationLocked(
            "run-before-next-mpeg-timeout-clear");
    }

    (void)debugPause(
        timeout,
        "run-before-next-mpeg-timeout");
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

PS2Runtime::DebugStopInfo PS2Runtime::debugRunForVSyncFields(
    uint64_t count, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    if (count == 0u)
    {
        return {false, "invalid-count", m_debugPc.load(std::memory_order_relaxed), 0u};
    }
    if (!debugPause(timeout, "run-for-vsync-entry"))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    const uint64_t current =
        m_debugVSyncFields.load(std::memory_order_relaxed);
    if (count > std::numeric_limits<uint64_t>::max() - current)
    {
        return {false, "count-overflow", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = false;
        m_debugRunUntilMpegUniquePicturesActive = false;
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        m_debugRunForVSyncFieldsActive = true;
        m_debugRunForVSyncFieldsTarget = current + count;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugSetPauseRequestedLocked(
            false, "run-for-vsync-start");
        debugRefreshControlActiveLocked();
    }
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugResume();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                const DebugStopInfo stop = m_debugLastStop;
                lock.unlock();
                if (m_eeRuntimeExecutor &&
                    !m_eeRuntimeExecutor->debugWaitUntilPaused(timeout))
                {
                    return {false, "pause-timeout", stop.pc, stop.sequence};
                }
                return stop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugRunForVSyncFieldsActive = false;
        debugAdvanceControlGenerationLocked(
            "run-for-vsync-timeout-clear");
    }

    (void)debugPause(timeout, "run-for-vsync-timeout");
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

PS2Runtime::DebugStopInfo PS2Runtime::debugStepDispatches(
    uint64_t count, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    if (count == 0u)
    {
        return {false, "invalid-count", m_debugPc.load(std::memory_order_relaxed), 0u};
    }
    if (!debugPause(timeout, "step-dispatches-entry"))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = false;
        m_debugRunUntilMpegUniquePicturesActive = false;
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        m_debugRunForVSyncFieldsActive = false;
        m_debugStepActive = true;
        m_debugStepRemaining = count;
        debugSetPauseRequestedLocked(
            false, "step-dispatches-start");
        debugRefreshControlActiveLocked();
    }
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->debugResume();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                const DebugStopInfo stop =
                    m_debugLastStop;
                lock.unlock();
                if (m_eeRuntimeExecutor &&
                    !m_eeRuntimeExecutor
                         ->debugWaitUntilPaused(
                             timeout))
                {
                    return {
                        false,
                        "pause-timeout",
                        stop.pc,
                        stop.sequence,
                    };
                }
                return stop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugAdvanceControlGenerationLocked(
            "step-dispatches-timeout-clear");
    }

    (void)debugPause(timeout, "step-dispatches-timeout");
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

R5900Context PS2Runtime::debugCpuSnapshot()
{
    R5900Context result{};
    const auto capture =
        [this, &result]()
        {
            R5900Context *const context =
                m_boundEeContext
                    ? m_boundEeContext
                    : &m_cpuContext;
            synchronizeCop0Timing(
                context,
                ps2x::timing::eeTickToCyclesFloor(
                    currentEeTick()));
            result = m_cpuContext;
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return result;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

VuExecutionState PS2Runtime::debugVu0ArchitecturalSnapshot()
{
    VuExecutionState result{};
    const auto capture =
        [this, &result]()
        {
            if (m_vu0.isActive())
            {
                result = m_vu0.state();
                return;
            }

            const R5900Context *const context =
                m_boundEeContext
                    ? m_boundEeContext
                    : &m_cpuContext;
            copyVu0ContextToState(context, result);
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return result;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

PS2Runtime::DebugEeTiming PS2Runtime::debugEeTimingSnapshot()
{
    DebugEeTiming result{};
    const auto capture =
        [this, &result]()
        {
            const R5900Context *const context =
                m_boundEeContext
                    ? m_boundEeContext
                    : &m_cpuContext;
            result.currentTick =
                currentEeTick().raw();
            result.currentCycle =
                ps2x::timing::eeTickToCyclesFloor(
                    currentEeTick());
            result.localBlockTicks =
                context->ee_block_cycle_ticks;
            result.localBlockActive =
                context->ee_block_cycle_active;
            result.contextBound =
                m_boundEeContext != nullptr;
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return result;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

PS2Runtime::DebugEeScheduler
PS2Runtime::debugEeSchedulerSnapshot()
{
    DebugEeScheduler result{};
    const auto capture =
        [this, &result]()
        {
            result.currentTick = currentEeTick().raw();
            const auto next =
                m_eeEventScheduler.nextDeadline();
            if (next.has_value())
            {
                result.hasNextDeadline = true;
                result.nextDeadlineTick = next->raw();
            }
            result.statistics =
                m_eeEventScheduler.statistics();

            for (size_t index = 0u;
                 index < result.slots.size(); ++index)
            {
                DebugEeEventSlot &target =
                    result.slots[index];
                target.source =
                    static_cast<
                        ps2x::timing::EeEventSource>(
                        index);
                target.device =
                    debugEeEventDeviceState(
                        target.source);
                const auto source =
                    m_eeEventScheduler.event(
                        target.source);
                if (!source.has_value())
                {
                    continue;
                }
                target.pending = true;
                target.deadlineTick =
                    source->deadline.raw();
                target.generation =
                    source->token.generation;
                target.sequence = source->sequence;
            }
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return result;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

PS2Runtime::DebugVu1Timing
PS2Runtime::debugVu1TimingSnapshot()
{
    DebugVu1Timing result{};
    const auto capture =
        [this, &result]()
        {
            result.currentTick =
                currentEeTick().raw();
            result.generation =
                m_vu1ExecutionTiming.generation;
            result.lastAdvancedTick =
                m_vu1ExecutionTiming
                    .lastAdvancedTick.raw();
            result.totalAdvancedCycles =
                m_vu1ExecutionTiming
                    .totalAdvancedCycles;
            result.pc = m_publishedVu1ProgramCounter.load(
                std::memory_order_acquire);
            result.active = m_publishedVu1Active.load(
                std::memory_order_acquire);
            result.vifWaitingForVu =
                m_memory.vif1WaitingForVu();
            const auto event =
                m_eeEventScheduler.event(
                    ps2x::timing::EeEventSource::
                        VifVu1Finish);
            if (event.has_value())
            {
                result.eventPending = true;
                result.eventGeneration =
                    event->token.generation;
                result.eventDeadlineTick =
                    event->deadline.raw();
            }
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return result;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return result;
}

std::array<PS2Runtime::DebugVuBackendDiagnostics, 2u>
PS2Runtime::debugVuBackendDiagnosticsSnapshot() const
{
    std::lock_guard<std::mutex> lock(
        m_debugVuBackendDiagnosticsMutex);
    return m_debugVuBackendDiagnostics;
}

bool PS2Runtime::debugReadRdram(uint32_t address,
                                uint32_t size,
                                std::vector<uint8_t> &output)
{
    uint32_t offset = 0u;
    if (!ps2ResolveDirectRdramOffset(address, offset) ||
        size > PS2_RAM_SIZE ||
        offset > (PS2_RAM_SIZE - size) ||
        !m_memory.getRDRAM())
    {
        return false;
    }

    const auto capture =
        [this, offset, size, &output]()
        {
            output.assign(
                m_memory.getRDRAM() + offset,
                m_memory.getRDRAM() + offset + size);
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return true;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return true;
}

bool PS2Runtime::debugReadMemory(uint32_t address,
                                 uint32_t size,
                                 std::vector<uint8_t> &output)
{
    uint8_t *base = nullptr;
    uint32_t offset = 0u;
    uint32_t limit = 0u;
    if (ps2ResolveDirectRdramOffset(address, offset))
    {
        base = m_memory.getRDRAM();
        limit = PS2_RAM_SIZE;
    }
    else if (ps2IsScratchpadAddress(address))
    {
        base = m_memory.getScratchpad();
        offset = ps2ScratchpadOffset(address);
        limit = PS2_SCRATCHPAD_SIZE;
    }

    if (!base || size > limit || offset > (limit - size))
    {
        return false;
    }

    const auto capture =
        [base, offset, size, &output]()
        {
            output.assign(
                base + offset,
                base + offset + size);
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return true;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return true;
}

bool PS2Runtime::debugCopyGsVram(std::vector<uint8_t> &output)
{
    bool copied = false;
    const auto capture =
        [this, &output, &copied]()
        {
            GsVramSnapshotResult snapshot =
                takeGsCommandResult<GsVramSnapshotResult>(
                    submitGsCommand(GsCopyVramCommand{}));
            copied = snapshot.available;
            output = std::move(snapshot.bytes);
        };
    if (invokeEeExecutorTaskAtBoundary(capture))
    {
        return copied;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    capture();
    return copied;
}

void PS2Runtime::recordEeThreadQueueRotation(
    int threadId,
    int requestedPriority,
    int effectivePriority,
    bool accepted) noexcept
{
    if (!m_eeThreadDiagnosticsEnabled)
    {
        return;
    }

    m_eeThreadDiagnosticRotationRequests.fetch_add(
        1u, std::memory_order_relaxed);
    if (requestedPriority == 0)
    {
        m_eeThreadDiagnosticPriorityZeroRotationRequests.fetch_add(
            1u, std::memory_order_relaxed);
    }
    if (!accepted)
    {
        m_eeThreadDiagnosticRejectedRotationRequests.fetch_add(
            1u, std::memory_order_relaxed);
        return;
    }

    m_eeThreadDiagnosticAcceptedRotationRequests.fetch_add(
        1u, std::memory_order_relaxed);
    const auto hashWord = [](uint64_t hash, int value) noexcept
    {
        constexpr uint64_t kFnvPrime = 1099511628211ull;
        const uint32_t word = static_cast<uint32_t>(value);
        for (uint32_t shift = 0u; shift < 32u; shift += 8u)
        {
            hash ^= static_cast<uint8_t>(word >> shift);
            hash *= kFnvPrime;
        }
        return hash;
    };
    uint64_t sequenceHash =
        m_eeThreadDiagnosticAcceptedRotationSequenceHash.load(
            std::memory_order_relaxed);
    while (true)
    {
        uint64_t nextHash = hashWord(sequenceHash, threadId);
        nextHash = hashWord(nextHash, requestedPriority);
        nextHash = hashWord(nextHash, effectivePriority);
        if (m_eeThreadDiagnosticAcceptedRotationSequenceHash
                .compare_exchange_weak(
                    sequenceHash,
                    nextHash,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
        {
            break;
        }
    }
    if (effectivePriority >= 0 &&
        static_cast<size_t>(effectivePriority) <
            m_eeThreadDiagnosticAcceptedRotationsByPriority.size())
    {
        m_eeThreadDiagnosticAcceptedRotationsByPriority[
            static_cast<size_t>(effectivePriority)]
            .fetch_add(1u, std::memory_order_relaxed);
    }
    if (threadId >= 0 &&
        static_cast<size_t>(threadId) <
            m_eeThreadDiagnosticAcceptedRotationsByThread.size())
    {
        m_eeThreadDiagnosticAcceptedRotationsByThread[
            static_cast<size_t>(threadId)]
            .fetch_add(1u, std::memory_order_relaxed);
    }
    else
    {
        m_eeThreadDiagnosticUntrackedThreadRotationRequests.fetch_add(
            1u, std::memory_order_relaxed);
    }
}

void PS2Runtime::recordMpegPictureServed(
    R5900Context *ctx, bool repeated)
{
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    bool stopBeforeUniquePicture = false;
    uint64_t stopControlGeneration = 0u;
    if (!repeated && m_debugControlActive.load(std::memory_order_acquire))
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        stopBeforeUniquePicture =
            m_debugRunBeforeNextMpegUniquePictureActive &&
            m_debugMpegUniquePicturesServed.load(std::memory_order_relaxed) ==
                m_debugRunBeforeNextMpegUniquePictureCurrent;
        if (stopBeforeUniquePicture)
        {
            stopControlGeneration =
                m_debugControlGeneration;
        }
    }
    if (stopBeforeUniquePicture)
    {
        (void)debugBlockGuestAtBoundary(
            ctx,
            "before-mpeg-unique-picture",
            stopControlGeneration);
    }

    m_debugMpegPicturesServed.fetch_add(
        1u, std::memory_order_relaxed);
    if (repeated)
    {
        m_debugMpegRepeatedPicturesServed.fetch_add(
            1u, std::memory_order_relaxed);
    }
    else
    {
        m_debugMpegUniquePicturesServed.fetch_add(
            1u, std::memory_order_relaxed);
    }
#else
    (void)ctx;
    (void)repeated;
#endif
}

PS2Runtime::DebugRuntimeProgress PS2Runtime::debugRuntimeProgress() const
{
    DebugRuntimeProgress progress{};
    progress.dispatches = m_debugDispatches.load(std::memory_order_relaxed);
    progress.eeInstructions =
        m_debugEeInstructions.load(std::memory_order_relaxed);
    progress.vsyncFields =
        m_debugVSyncFields.load(std::memory_order_relaxed);
    progress.mpegPicturesServed =
        m_debugMpegPicturesServed.load(std::memory_order_relaxed);
    progress.mpegUniquePicturesServed =
        m_debugMpegUniquePicturesServed.load(std::memory_order_relaxed);
    progress.mpegRepeatedPicturesServed =
        m_debugMpegRepeatedPicturesServed.load(std::memory_order_relaxed);
    progress.pc = m_debugPc.load(std::memory_order_relaxed);
    progress.ra = m_debugRa.load(std::memory_order_relaxed);
    progress.sp = m_debugSp.load(std::memory_order_relaxed);
    progress.gp = m_debugGp.load(std::memory_order_relaxed);
    progress.guestExecutionWaiters =
        m_guestExecutionWaiters.load(std::memory_order_relaxed);
    progress.guestExecutionHandoffTimeouts =
        m_guestExecutionHandoffTimeouts.load(std::memory_order_relaxed);
    return progress;
}

PS2Runtime::DebugEeThreadDiagnostics
PS2Runtime::debugEeThreadDiagnosticsSnapshot() const
{
    DebugEeThreadDiagnostics result{};
    result.enabled = m_eeThreadDiagnosticsEnabled;
    result.guestLockRequests =
        m_eeThreadDiagnosticGuestLockRequests.load(
            std::memory_order_relaxed);
    result.guestLockAcquisitions =
        m_eeThreadDiagnosticGuestLockAcquisitions.load(
            std::memory_order_relaxed);
    result.guestLockContentions =
        m_eeThreadDiagnosticGuestLockContentions.load(
            std::memory_order_relaxed);
    result.outerGuestExecutionAcquisitions =
        m_eeThreadDiagnosticOuterGuestExecutionAcquisitions.load(
            std::memory_order_relaxed);
    result.guestContextChanges =
        m_eeThreadDiagnosticGuestContextChanges.load(
            std::memory_order_relaxed);
    result.handoffNotifications =
        m_eeThreadDiagnosticHandoffNotifications.load(
            std::memory_order_relaxed);
    result.handoffWaitRequests =
        m_eeThreadDiagnosticHandoffWaitRequests.load(
            std::memory_order_relaxed);
    result.handoffWaitFastPaths =
        m_eeThreadDiagnosticHandoffWaitFastPaths.load(
            std::memory_order_relaxed);
    result.handoffCvWaits =
        m_eeThreadDiagnosticHandoffCvWaits.load(
            std::memory_order_relaxed);
    result.handoffCompletions =
        m_eeThreadDiagnosticHandoffCompletions.load(
            std::memory_order_relaxed);
    result.handoffTimeouts =
        m_eeThreadDiagnosticHandoffTimeouts.load(
            std::memory_order_relaxed);
    result.yieldRequests =
        m_eeThreadDiagnosticYieldRequests.load(
            std::memory_order_relaxed);
    result.deferredYields =
        m_eeThreadDiagnosticDeferredYields.load(
            std::memory_order_relaxed);
    result.hostThreadYields =
        m_eeThreadDiagnosticHostThreadYields.load(
            std::memory_order_relaxed);
    result.requestedGuestSwitches =
        m_eeThreadDiagnosticRequestedGuestSwitches.load(
            std::memory_order_relaxed);
    result.guestSwitchCvWaits =
        m_eeThreadDiagnosticGuestSwitchCvWaits.load(
            std::memory_order_relaxed);
    result.completedGuestSwitches =
        m_eeThreadDiagnosticCompletedGuestSwitches.load(
            std::memory_order_relaxed);
    result.guestSwitchTimeouts =
        m_eeThreadDiagnosticGuestSwitchTimeouts.load(
            std::memory_order_relaxed);
    result.rotationRequests =
        m_eeThreadDiagnosticRotationRequests.load(
            std::memory_order_relaxed);
    result.acceptedRotationRequests =
        m_eeThreadDiagnosticAcceptedRotationRequests.load(
            std::memory_order_relaxed);
    result.rejectedRotationRequests =
        m_eeThreadDiagnosticRejectedRotationRequests.load(
            std::memory_order_relaxed);
    result.priorityZeroRotationRequests =
        m_eeThreadDiagnosticPriorityZeroRotationRequests.load(
            std::memory_order_relaxed);
    result.untrackedThreadRotationRequests =
        m_eeThreadDiagnosticUntrackedThreadRotationRequests.load(
            std::memory_order_relaxed);
    result.acceptedRotationSequenceHash =
        m_eeThreadDiagnosticAcceptedRotationSequenceHash.load(
            std::memory_order_relaxed);
    for (size_t priority = 0u;
         priority < result.acceptedRotationsByPriority.size();
         ++priority)
    {
        result.acceptedRotationsByPriority[priority] =
            m_eeThreadDiagnosticAcceptedRotationsByPriority[priority]
                .load(std::memory_order_relaxed);
    }
    for (size_t thread = 0u;
         thread < result.acceptedRotationsByThread.size();
         ++thread)
    {
        result.acceptedRotationsByThread[thread] =
            m_eeThreadDiagnosticAcceptedRotationsByThread[thread]
                .load(std::memory_order_relaxed);
    }
    return result;
}

std::optional<ps2x::ee::EeRuntimeExecutorStatistics>
PS2Runtime::debugEeRuntimeExecutorStatisticsSnapshot() const
{
    if (!m_eeRuntimeExecutor)
    {
        return std::nullopt;
    }
    return m_eeRuntimeExecutor->statistics();
}

void PS2Runtime::debugRecordBranch(uint32_t pc)
{
    const uint64_t sequence =
        m_debugBranchSequence.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    DebugBranchSlot &slot =
        m_debugBranchHistory[(sequence - 1u) % kDebugBranchHistoryCapacity];
    slot.pc.store(normalizeGuestFunctionAddress(pc), std::memory_order_relaxed);
    slot.sequence.store(sequence, std::memory_order_release);
}

std::vector<PS2Runtime::DebugBranchEntry> PS2Runtime::debugBranchHistory(
    size_t maximumEntries) const
{
    maximumEntries =
        std::min(maximumEntries, kDebugBranchHistoryCapacity);
    const uint64_t last =
        m_debugBranchSequence.load(std::memory_order_acquire);
    const uint64_t count =
        std::min<uint64_t>(last, static_cast<uint64_t>(maximumEntries));
    const uint64_t first = last - count + 1u;

    std::vector<DebugBranchEntry> entries;
    entries.reserve(static_cast<size_t>(count));
    for (uint64_t sequence = first; sequence <= last && count != 0u;
         ++sequence)
    {
        const DebugBranchSlot &slot =
            m_debugBranchHistory[(sequence - 1u) %
                                 kDebugBranchHistoryCapacity];
        const uint64_t observed =
            slot.sequence.load(std::memory_order_acquire);
        if (observed != sequence)
        {
            continue;
        }
        entries.push_back(
            {sequence, slot.pc.load(std::memory_order_relaxed)});
    }
    return entries;
}

void PS2Runtime::debugRecordFault(const DebugFaultInfo &fault)
{
    DebugFaultInfo copy = fault;
    copy.active = true;
    copy.sequence =
        m_debugFaultSequence.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    std::lock_guard<std::mutex> lock(m_debugFaultMutex);
    m_debugFault = std::move(copy);
}

PS2Runtime::DebugFaultInfo PS2Runtime::debugFaultSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_debugFaultMutex);
    return m_debugFault;
}

void PS2Runtime::debugClearFault()
{
    std::lock_guard<std::mutex> lock(m_debugFaultMutex);
    m_debugFault = {};
}

void PS2Runtime::debugStartVu0SyncTrace(
    size_t maximumEntries, std::optional<uint32_t> triggerEePc,
    bool stopOnFull, std::optional<uint64_t> triggerInvocation)
{
    if (triggerEePc.has_value() && triggerInvocation.has_value())
    {
        throw std::invalid_argument(
            "VU0 sync trace accepts only one trigger");
    }
    maximumEntries = std::clamp<size_t>(maximumEntries, 1u, 16384u);
    std::lock_guard<std::mutex> lock(m_debugVu0SyncTraceMutex);
    m_debugVu0SyncTraceEnabled.store(false, std::memory_order_release);
    m_debugVu0SyncTrace.clear();
    m_debugVu0SyncTrace.reserve(maximumEntries);
    m_debugVu0SyncTraceCapacity = maximumEntries;
    m_debugVu0SyncTraceNext = 0u;
    m_debugVu0SyncTraceTotal = 0u;
    m_debugVu0SyncTraceStopOnFull = stopOnFull;
    m_debugVu0SyncTraceHasTriggerSchedulerSnapshot = false;
    m_debugVu0SyncTraceTriggerVsyncTick = 0u;
    m_debugVu0SyncTraceTriggerScheduler = {};
    m_debugVu0SyncTraceTriggerEePc.store(
        triggerEePc.value_or(0u), std::memory_order_relaxed);
    m_debugVu0SyncTraceHasTrigger.store(
        triggerEePc.has_value(), std::memory_order_relaxed);
    m_debugVu0SyncTraceTriggerInvocation.store(
        triggerInvocation.value_or(0u), std::memory_order_relaxed);
    m_debugVu0SyncTraceHasInvocationTrigger.store(
        triggerInvocation.has_value(), std::memory_order_relaxed);
    m_debugVu0SyncTraceTriggered.store(
        !triggerEePc.has_value() && !triggerInvocation.has_value(),
        std::memory_order_release);
    if (triggerEePc.has_value())
    {
        m_debugEePcTraceTriggerEverConfigured.store(
            true, std::memory_order_release);
    }
    m_debugVu0SyncTraceEnabled.store(true, std::memory_order_release);
    m_vu0.setInstructionObserverEnabled(true);
}

void PS2Runtime::debugRecordVu0Sync(DebugVu0SyncEntry entry)
{
    if (!m_debugVu0SyncTraceEnabled.load(std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_debugVu0SyncTraceMutex);
    if (!m_debugVu0SyncTraceEnabled.load(std::memory_order_relaxed) ||
        m_debugVu0SyncTraceCapacity == 0u)
    {
        return;
    }

    entry.sequence = ++m_debugVu0SyncTraceTotal;
    if (m_debugVu0SyncTrace.size() < m_debugVu0SyncTraceCapacity)
    {
        m_debugVu0SyncTrace.push_back(std::move(entry));
        if (m_debugVu0SyncTraceStopOnFull &&
            m_debugVu0SyncTrace.size() == m_debugVu0SyncTraceCapacity)
        {
            m_debugVu0SyncTraceEnabled.store(
                false, std::memory_order_release);
        }
        return;
    }

    m_debugVu0SyncTrace[m_debugVu0SyncTraceNext] = std::move(entry);
    m_debugVu0SyncTraceNext =
        (m_debugVu0SyncTraceNext + 1u) % m_debugVu0SyncTraceCapacity;
}

PS2Runtime::DebugVu0SyncTrace PS2Runtime::debugVu0SyncTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_debugVu0SyncTraceEnabled.store(false, std::memory_order_release);
        m_vu0.setInstructionObserverEnabled(
            m_debugVu0InstructionTraceEnabled.load(
                std::memory_order_acquire));
    }

    std::lock_guard<std::mutex> lock(m_debugVu0SyncTraceMutex);
    DebugVu0SyncTrace snapshot{};
    snapshot.enabled =
        m_debugVu0SyncTraceEnabled.load(std::memory_order_acquire);
    snapshot.triggered =
        m_debugVu0SyncTraceTriggered.load(std::memory_order_acquire);
    snapshot.stopOnFull = m_debugVu0SyncTraceStopOnFull;
    snapshot.hasTriggerSchedulerSnapshot =
        m_debugVu0SyncTraceHasTriggerSchedulerSnapshot;
    snapshot.triggerVsyncTick =
        m_debugVu0SyncTraceTriggerVsyncTick;
    snapshot.triggerScheduler =
        m_debugVu0SyncTraceTriggerScheduler;
    if (m_debugVu0SyncTraceHasTrigger.load(std::memory_order_relaxed))
    {
        snapshot.triggerEePc =
            m_debugVu0SyncTraceTriggerEePc.load(std::memory_order_relaxed);
    }
    if (m_debugVu0SyncTraceHasInvocationTrigger.load(
            std::memory_order_relaxed))
    {
        snapshot.triggerInvocation =
            m_debugVu0SyncTraceTriggerInvocation.load(
                std::memory_order_relaxed);
    }
    snapshot.totalEntries = m_debugVu0SyncTraceTotal;
    snapshot.droppedEntries =
        m_debugVu0SyncTraceTotal > m_debugVu0SyncTrace.size()
            ? m_debugVu0SyncTraceTotal - m_debugVu0SyncTrace.size()
            : 0u;
    snapshot.entries.reserve(m_debugVu0SyncTrace.size());
    if (m_debugVu0SyncTrace.size() < m_debugVu0SyncTraceCapacity ||
        m_debugVu0SyncTrace.empty())
    {
        snapshot.entries = m_debugVu0SyncTrace;
        return snapshot;
    }

    for (size_t offset = 0u; offset < m_debugVu0SyncTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugVu0SyncTrace[
                (m_debugVu0SyncTraceNext + offset) %
                m_debugVu0SyncTrace.size()]);
    }
    return snapshot;
}

void PS2Runtime::debugStartEeEventTrace(
    size_t maximumEntries,
    std::optional<uint32_t> triggerEePc,
    bool stopOnFull)
{
    maximumEntries =
        std::clamp<size_t>(maximumEntries, 1u, 16384u);
    std::lock_guard<std::mutex> lock(
        m_debugEeEventTraceMutex);
    m_debugEeEventTraceEnabled.store(
        false, std::memory_order_release);
    m_debugEeEventTrace.clear();
    m_debugEeEventTrace.reserve(maximumEntries);
    m_debugEeEventTraceCapacity = maximumEntries;
    m_debugEeEventTraceNext = 0u;
    m_debugEeEventTraceTotal = 0u;
    m_debugEeEventTraceStopOnFull = stopOnFull;
    m_debugEeEventTraceTriggerEePc.store(
        triggerEePc.value_or(0u), std::memory_order_relaxed);
    m_debugEeEventTraceHasTrigger.store(
        triggerEePc.has_value(), std::memory_order_relaxed);
    m_debugEeEventTraceTriggered.store(
        !triggerEePc.has_value(), std::memory_order_release);
    if (triggerEePc.has_value())
    {
        m_debugEePcTraceTriggerEverConfigured.store(
            true, std::memory_order_release);
    }
    m_debugEeEventTraceEnabled.store(
        true, std::memory_order_release);
}

void PS2Runtime::debugRecordEeEvent(
    DebugEeEventEntry entry)
{
    if (!m_debugEeEventTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }

    if (!m_debugEeEventTraceTriggered.load(
            std::memory_order_acquire) &&
        m_debugEeEventTraceHasTrigger.load(
            std::memory_order_relaxed) &&
        entry.eePc != m_debugEeEventTraceTriggerEePc.load(
                          std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(
        m_debugEeEventTraceMutex);
    if (!m_debugEeEventTraceEnabled.load(
            std::memory_order_relaxed) ||
        m_debugEeEventTraceCapacity == 0u)
    {
        return;
    }

    if (!m_debugEeEventTraceTriggered.load(
            std::memory_order_relaxed))
    {
        if (!m_debugEeEventTraceHasTrigger.load(
                std::memory_order_relaxed) ||
            entry.eePc ==
                m_debugEeEventTraceTriggerEePc.load(
                    std::memory_order_relaxed))
        {
            m_debugEeEventTraceTriggered.store(
                true, std::memory_order_release);
        }
        else
        {
            return;
        }
    }

    entry.sequence = ++m_debugEeEventTraceTotal;
    if (m_debugEeEventTrace.size() <
        m_debugEeEventTraceCapacity)
    {
        m_debugEeEventTrace.push_back(std::move(entry));
        if (m_debugEeEventTraceStopOnFull &&
            m_debugEeEventTrace.size() ==
                m_debugEeEventTraceCapacity)
        {
            m_debugEeEventTraceEnabled.store(
                false, std::memory_order_release);
        }
        return;
    }

    m_debugEeEventTrace[m_debugEeEventTraceNext] =
        std::move(entry);
    m_debugEeEventTraceNext =
        (m_debugEeEventTraceNext + 1u) %
        m_debugEeEventTraceCapacity;
}

PS2Runtime::DebugEeEventTrace
PS2Runtime::debugEeEventTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_debugEeEventTraceEnabled.store(
            false, std::memory_order_release);
    }

    std::lock_guard<std::mutex> lock(
        m_debugEeEventTraceMutex);
    DebugEeEventTrace snapshot{};
    snapshot.enabled = m_debugEeEventTraceEnabled.load(
        std::memory_order_acquire);
    snapshot.triggered =
        m_debugEeEventTraceTriggered.load(
            std::memory_order_acquire);
    snapshot.stopOnFull = m_debugEeEventTraceStopOnFull;
    if (m_debugEeEventTraceHasTrigger.load(
            std::memory_order_relaxed))
    {
        snapshot.triggerEePc =
            m_debugEeEventTraceTriggerEePc.load(
                std::memory_order_relaxed);
    }
    snapshot.totalEntries = m_debugEeEventTraceTotal;
    snapshot.droppedEntries =
        m_debugEeEventTraceTotal >
                m_debugEeEventTrace.size()
            ? m_debugEeEventTraceTotal -
                  m_debugEeEventTrace.size()
            : 0u;
    snapshot.entries.reserve(m_debugEeEventTrace.size());
    if (m_debugEeEventTrace.size() <
            m_debugEeEventTraceCapacity ||
        m_debugEeEventTrace.empty())
    {
        snapshot.entries = m_debugEeEventTrace;
        return snapshot;
    }

    for (size_t offset = 0u;
         offset < m_debugEeEventTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugEeEventTrace[
                (m_debugEeEventTraceNext + offset) %
                m_debugEeEventTrace.size()]);
    }
    return snapshot;
}

void PS2Runtime::debugStartVu1TimingTrace(
    size_t maximumEntries,
    bool stopOnFull)
{
    maximumEntries =
        std::clamp<size_t>(maximumEntries, 1u, 16384u);
    std::lock_guard<std::mutex> lock(
        m_debugVu1TimingTraceMutex);
    m_debugVu1TimingTraceEnabled.store(
        false, std::memory_order_release);
    m_debugVu1TimingTrace.clear();
    m_debugVu1TimingTrace.reserve(maximumEntries);
    m_debugVu1TimingTraceCapacity = maximumEntries;
    m_debugVu1TimingTraceNext = 0u;
    m_debugVu1TimingTraceTotal = 0u;
    m_debugVu1TimingTraceStopOnFull = stopOnFull;
    m_vu1CommandProcessor.
        setDiagnosticArchitecturalStateHashes(true);
    m_debugVu1TimingTraceEnabled.store(
        true, std::memory_order_release);
}

void PS2Runtime::debugBeginVu1TimingInvocation(
    Vu1InvocationKind kind,
    uint32_t startPc,
    uint32_t top,
    uint32_t itop,
    ps2x::timing::EeTick startTick) noexcept
{
    if (!m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }
    ++m_debugVu1TimingNextInvocation;
    if (m_debugVu1TimingNextInvocation == 0u)
        ++m_debugVu1TimingNextInvocation;
    m_debugVu1TimingInvocation = {
        .sequence = m_debugVu1TimingNextInvocation,
        .kind = kind,
        .startPc = startPc,
        .top = top,
        .itop = itop,
        .startTick = startTick,
    };
    m_debugVu1TimingNextPath1Packet = 0u;
}

void PS2Runtime::debugPublishVu1TimingInvocationStart(
    ps2x::timing::EeTick deadline,
    uint32_t estimatedCycles)
{
    if (!m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }
    m_debugVu1TimingInvocation.deadline = deadline;
    m_debugVu1TimingInvocation.estimatedCycles =
        estimatedCycles;
    m_debugVu1TimingInvocation.eventGeneration =
        m_vu1ExecutionTiming.eventToken.generation;
    DebugVu1TimingEntry entry = debugVu1TimingEntry(
        DebugVu1TimingEventKind::InvocationStart,
        m_boundEeContext);
    debugRecordVu1Timing(std::move(entry));
}

PS2Runtime::DebugVu1TimingEntry
PS2Runtime::debugVu1TimingEntry(
    DebugVu1TimingEventKind kind,
    const R5900Context *ctx) const noexcept
{
    const R5900Context *const context = ctx
        ? ctx
        : (m_boundEeContext
               ? m_boundEeContext
               : &m_cpuContext);
    const bool waiting = m_memory.vif1WaitingForVu();
    DebugVu1TimingEntry entry{};
    entry.kind = kind;
    entry.mode = m_vu1ExecutionMode;
    entry.invocationKind =
        m_debugVu1TimingInvocation.kind;
    entry.invocation =
        m_debugVu1TimingInvocation.sequence;
    entry.eeTick = currentEeTick().raw();
    entry.eeInstructions = m_debugEeInstructions.load(
        std::memory_order_relaxed);
    entry.vsyncField = m_debugVSyncFields.load(
        std::memory_order_relaxed);
    entry.executionGeneration =
        m_vu1ExecutionTiming.generation;
    entry.eventGeneration =
        m_debugVu1TimingInvocation.eventGeneration;
    entry.startTick =
        m_debugVu1TimingInvocation.startTick.raw();
    entry.deadlineTick =
        m_debugVu1TimingInvocation.deadline.raw();
    entry.totalAdvancedCycles =
        m_vu1ExecutionTiming.totalAdvancedCycles;
    entry.dmaStarts = m_memory.dmaStartCount();
    entry.gsCsr = m_memory.gs().csr.load(
        std::memory_order_relaxed);
    entry.estimatedCycles =
        m_debugVu1TimingInvocation.estimatedCycles;
    entry.startPc = m_debugVu1TimingInvocation.startPc;
    entry.top = m_debugVu1TimingInvocation.top;
    entry.itop = m_debugVu1TimingInvocation.itop;
    entry.eePc = context ? context->pc : 0u;
    entry.vuPc = m_publishedVu1ProgramCounter.load(
        std::memory_order_acquire);
    entry.vifStat = m_memory.vif1_regs.stat;
    entry.vifCode = m_memory.vif1_regs.code;
    entry.intcStatus = m_memory.intcStatus();
    entry.intcMask = m_memory.intcMask();
    entry.vif1Dma = debugEeEventDeviceState(
        ps2x::timing::EeEventSource::DmacVif1);
    entry.active = m_publishedVu1Active.load(
        std::memory_order_acquire);
    entry.busy = context != nullptr &&
        (context->vu0_vpu_stat & kVu1BusyMask) != 0u;
    entry.vifWaitingBefore = waiting;
    entry.vifWaitingAfter = waiting;
    return entry;
}

void PS2Runtime::debugRecordVu1Timing(
    DebugVu1TimingEntry entry) const
{
    if (!m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(
        m_debugVu1TimingTraceMutex);
    if (!m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed) ||
        m_debugVu1TimingTraceCapacity == 0u)
    {
        return;
    }

    entry.sequence = ++m_debugVu1TimingTraceTotal;
    if (m_debugVu1TimingTrace.size() <
        m_debugVu1TimingTraceCapacity)
    {
        m_debugVu1TimingTrace.push_back(std::move(entry));
        if (m_debugVu1TimingTraceStopOnFull &&
            m_debugVu1TimingTrace.size() ==
                m_debugVu1TimingTraceCapacity)
        {
            m_debugVu1TimingTraceEnabled.store(
                false, std::memory_order_release);
            m_vu1CommandProcessor.
                setDiagnosticArchitecturalStateHashes(false);
        }
        return;
    }

    m_debugVu1TimingTrace[m_debugVu1TimingTraceNext] =
        std::move(entry);
    m_debugVu1TimingTraceNext =
        (m_debugVu1TimingTraceNext + 1u) %
        m_debugVu1TimingTraceCapacity;
}

PS2Runtime::DebugVu1TimingTrace
PS2Runtime::debugVu1TimingTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_debugVu1TimingTraceEnabled.store(
            false, std::memory_order_release);
        m_vu1CommandProcessor.
            setDiagnosticArchitecturalStateHashes(false);
    }

    std::lock_guard<std::mutex> lock(
        m_debugVu1TimingTraceMutex);
    DebugVu1TimingTrace snapshot{};
    snapshot.enabled = m_debugVu1TimingTraceEnabled.load(
        std::memory_order_acquire);
    snapshot.stopOnFull =
        m_debugVu1TimingTraceStopOnFull;
    snapshot.totalEntries =
        m_debugVu1TimingTraceTotal;
    snapshot.droppedEntries =
        m_debugVu1TimingTraceTotal >
                m_debugVu1TimingTrace.size()
            ? m_debugVu1TimingTraceTotal -
                  m_debugVu1TimingTrace.size()
            : 0u;
    snapshot.entries.reserve(
        m_debugVu1TimingTrace.size());
    if (m_debugVu1TimingTrace.size() <
            m_debugVu1TimingTraceCapacity ||
        m_debugVu1TimingTrace.empty())
    {
        snapshot.entries = m_debugVu1TimingTrace;
        return snapshot;
    }

    for (size_t offset = 0u;
         offset < m_debugVu1TimingTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugVu1TimingTrace[
                (m_debugVu1TimingTraceNext + offset) %
                m_debugVu1TimingTrace.size()]);
    }
    return snapshot;
}

void PS2Runtime::debugRecordVu1Path1Packet(
    const Vu1Path1Packet &packet,
    uint32_t defaultCycleOffset)
{
    if (!m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed) ||
        packet.bytes.empty())
    {
        return;
    }

    DebugVu1TimingEntry entry = debugVu1TimingEntry(
        DebugVu1TimingEventKind::Path1Packet,
        m_boundEeContext);
    entry.path1Packet =
        ++m_debugVu1TimingNextPath1Packet;
    entry.path1CycleOffset =
        m_vu1ExecutionTiming.totalAdvancedCycles +
        packet.cycleOffset.value_or(defaultCycleOffset);
    entry.path1Bytes = static_cast<uint32_t>(
        std::min<size_t>(
            packet.bytes.size(),
            static_cast<size_t>(UINT32_MAX)));
    entry.path1Digest = debugVu1TimingPacketDigest(
        packet.bytes.data(), packet.bytes.size());
    debugRecordVu1Timing(std::move(entry));
}

bool PS2Runtime::resumeVif1AfterVuWithTimingTrace()
{
    const bool waitingBefore =
        m_memory.vif1WaitingForVu();
    const bool resumed = m_memory.resumeVIF1AfterVu();
    if (m_debugVu1TimingTraceEnabled.load(
            std::memory_order_relaxed))
    {
        DebugVu1TimingEntry entry = debugVu1TimingEntry(
            DebugVu1TimingEventKind::VifResume,
            m_boundEeContext);
        entry.observedValue = resumed;
        entry.vifWaitingBefore = waitingBefore;
        entry.vifWaitingAfter =
            m_memory.vif1WaitingForVu();
        debugRecordVu1Timing(std::move(entry));
    }
    return resumed;
}

void PS2Runtime::debugStartVu0InstructionTrace(
    size_t maximumEntries, std::optional<uint32_t> triggerEePc,
    bool stopOnFull, std::optional<uint64_t> triggerInvocation)
{
    if (triggerEePc.has_value() && triggerInvocation.has_value())
    {
        throw std::invalid_argument(
            "VU0 instruction trace accepts only one trigger");
    }
    maximumEntries = std::clamp<size_t>(maximumEntries, 1u, 8192u);
    std::lock_guard<std::mutex> lock(m_debugVu0InstructionTraceMutex);
    m_debugVu0InstructionTraceEnabled.store(
        false, std::memory_order_release);
    m_debugVu0InstructionTrace.clear();
    m_debugVu0InstructionTrace.reserve(maximumEntries);
    m_debugVu0InstructionTraceCapacity = maximumEntries;
    m_debugVu0InstructionTraceNext = 0u;
    m_debugVu0InstructionTraceTotal = 0u;
    m_debugVu0InstructionTraceStopOnFull = stopOnFull;
    m_debugVu0InstructionTraceTriggerEePc.store(
        triggerEePc.value_or(0u), std::memory_order_relaxed);
    m_debugVu0InstructionTraceHasTrigger.store(
        triggerEePc.has_value(), std::memory_order_relaxed);
    m_debugVu0InstructionTraceTriggerInvocation.store(
        triggerInvocation.value_or(0u), std::memory_order_relaxed);
    m_debugVu0InstructionTraceHasInvocationTrigger.store(
        triggerInvocation.has_value(), std::memory_order_relaxed);
    m_debugVu0InstructionTraceTriggered.store(
        !triggerEePc.has_value() && !triggerInvocation.has_value(),
        std::memory_order_release);
    if (triggerEePc.has_value())
    {
        m_debugEePcTraceTriggerEverConfigured.store(
            true, std::memory_order_release);
    }
    m_debugVu0InstructionTraceEnabled.store(
        true, std::memory_order_release);
    m_vu0.setInstructionObserverEnabled(true);
}

void PS2Runtime::debugRecordVu0Instruction(
    DebugVu0InstructionEntry entry)
{
    if (!m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_debugVu0InstructionTraceMutex);
    if (!m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_relaxed) ||
        m_debugVu0InstructionTraceCapacity == 0u)
    {
        return;
    }

    entry.sequence = ++m_debugVu0InstructionTraceTotal;
    if (m_debugVu0InstructionTrace.size() <
        m_debugVu0InstructionTraceCapacity)
    {
        m_debugVu0InstructionTrace.push_back(std::move(entry));
        if (m_debugVu0InstructionTraceStopOnFull &&
            m_debugVu0InstructionTrace.size() ==
                m_debugVu0InstructionTraceCapacity)
        {
            m_debugVu0InstructionTraceEnabled.store(
                false, std::memory_order_release);
        }
        return;
    }

    m_debugVu0InstructionTrace[m_debugVu0InstructionTraceNext] =
        std::move(entry);
    m_debugVu0InstructionTraceNext =
        (m_debugVu0InstructionTraceNext + 1u) %
        m_debugVu0InstructionTraceCapacity;
}

PS2Runtime::DebugVu0InstructionTrace
PS2Runtime::debugVu0InstructionTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_debugVu0InstructionTraceEnabled.store(
            false, std::memory_order_release);
        m_vu0.setInstructionObserverEnabled(
            m_debugVu0SyncTraceEnabled.load(
                std::memory_order_acquire));
    }

    std::lock_guard<std::mutex> lock(m_debugVu0InstructionTraceMutex);
    DebugVu0InstructionTrace snapshot{};
    snapshot.enabled =
        m_debugVu0InstructionTraceEnabled.load(std::memory_order_acquire);
    snapshot.triggered =
        m_debugVu0InstructionTraceTriggered.load(
            std::memory_order_acquire);
    snapshot.stopOnFull = m_debugVu0InstructionTraceStopOnFull;
    if (m_debugVu0InstructionTraceHasTrigger.load(
            std::memory_order_relaxed))
    {
        snapshot.triggerEePc =
            m_debugVu0InstructionTraceTriggerEePc.load(
                std::memory_order_relaxed);
    }
    if (m_debugVu0InstructionTraceHasInvocationTrigger.load(
            std::memory_order_relaxed))
    {
        snapshot.triggerInvocation =
            m_debugVu0InstructionTraceTriggerInvocation.load(
                std::memory_order_relaxed);
    }
    snapshot.totalEntries = m_debugVu0InstructionTraceTotal;
    snapshot.droppedEntries =
        m_debugVu0InstructionTraceTotal >
                m_debugVu0InstructionTrace.size()
            ? m_debugVu0InstructionTraceTotal -
                  m_debugVu0InstructionTrace.size()
            : 0u;
    snapshot.entries.reserve(m_debugVu0InstructionTrace.size());
    if (m_debugVu0InstructionTrace.size() <
            m_debugVu0InstructionTraceCapacity ||
        m_debugVu0InstructionTrace.empty())
    {
        snapshot.entries = m_debugVu0InstructionTrace;
        return snapshot;
    }

    for (size_t offset = 0u;
         offset < m_debugVu0InstructionTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugVu0InstructionTrace[
                (m_debugVu0InstructionTraceNext + offset) %
                m_debugVu0InstructionTrace.size()]);
    }
    return snapshot;
}

std::vector<uint32_t> PS2Runtime::debugBreakpoints() const
{
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    return m_debugBreakpoints;
}

void PS2Runtime::debugAddBreakpoint(uint32_t address)
{
    m_debugControlActive.store(true, std::memory_order_release);
    address = normalizeGuestFunctionAddress(address);
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const auto position =
        std::lower_bound(m_debugBreakpoints.begin(), m_debugBreakpoints.end(), address);
    if (position == m_debugBreakpoints.end() || *position != address)
    {
        m_debugBreakpoints.insert(position, address);
        debugAdvanceControlGenerationLocked(
            "breakpoint-add");
    }
    debugRefreshControlActiveLocked();
}

void PS2Runtime::debugRemoveBreakpoint(uint32_t address)
{
    address = normalizeGuestFunctionAddress(address);
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const auto position =
        std::lower_bound(m_debugBreakpoints.begin(), m_debugBreakpoints.end(), address);
    if (position != m_debugBreakpoints.end() && *position == address)
    {
        m_debugBreakpoints.erase(position);
        debugAdvanceControlGenerationLocked(
            "breakpoint-remove");
    }
    debugRefreshControlActiveLocked();
}

std::vector<PS2Runtime::DebugWatchpoint> PS2Runtime::debugWatchpoints() const
{
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    return m_debugWatchpoints;
}

uint64_t PS2Runtime::debugAddWatchpoint(uint32_t start,
                                        uint32_t size,
                                        DebugMemoryAccess access)
{
    if (size == 0u || static_cast<uint64_t>(start) + size > 0x100000000ull)
    {
        return 0u;
    }

    m_debugControlActive.store(true, std::memory_order_release);
    uint32_t normalized = 0u;
    if (ps2ResolveDirectRdramOffset(start, normalized))
    {
        start = normalized;
    }

    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const uint64_t id = m_debugNextWatchpointId++;
    m_debugWatchpoints.push_back({id, start, size, access});
    debugAdvanceControlGenerationLocked(
        "watchpoint-add");
    m_debugWatchpointsActive.store(true, std::memory_order_release);
    debugRefreshControlActiveLocked();
    return id;
}

bool PS2Runtime::debugRemoveWatchpoint(uint64_t id)
{
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const auto position = std::find_if(
        m_debugWatchpoints.begin(), m_debugWatchpoints.end(),
        [id](const DebugWatchpoint &watchpoint)
        { return watchpoint.id == id; });
    if (position == m_debugWatchpoints.end())
    {
        return false;
    }
    m_debugWatchpoints.erase(position);
    debugAdvanceControlGenerationLocked(
        "watchpoint-remove");
    m_debugWatchpointsActive.store(!m_debugWatchpoints.empty(),
                                   std::memory_order_release);
    debugRefreshControlActiveLocked();
    return true;
}

bool PS2Runtime::debugWatchpointsEnabled() const
{
    return m_debugWatchpointsActive.load(std::memory_order_acquire);
}

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
void PS2Runtime::debugObserveMemoryAccess(uint32_t address,
                                          uint32_t size,
                                          DebugMemoryAccess access,
                                          const R5900Context *ctx)
{
    if (!debugWatchpointsEnabled() || size == 0u)
    {
        return;
    }

    uint32_t normalized = 0u;
    if (ps2ResolveDirectRdramOffset(address, normalized))
    {
        address = normalized;
    }

    bool matched = false;
    uint64_t stopControlGeneration = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        const uint64_t accessBegin = address;
        const uint64_t accessEnd = accessBegin + size;
        const uint8_t requested = static_cast<uint8_t>(access);
        for (const DebugWatchpoint &watchpoint : m_debugWatchpoints)
        {
            const uint8_t watched = static_cast<uint8_t>(watchpoint.access);
            const uint64_t watchBegin = watchpoint.start;
            const uint64_t watchEnd = watchBegin + watchpoint.size;
            if ((requested & watched) != 0u &&
                accessBegin < watchEnd && watchBegin < accessEnd)
            {
                matched = true;
                stopControlGeneration =
                    m_debugControlGeneration;
                break;
            }
        }
    }

    if (matched)
    {
        R5900Context *const mutableContext =
            const_cast<R5900Context *>(ctx);
        if (mutableContext)
        {
            // Fast direct-RDRAM accesses may carry the current retirement in
            // the counted Random run. A matched watchpoint publishes the
            // context before the memory effect, so materialize it first.
            mutableContext->finishEeInstruction();
        }
        (void)debugBlockGuestAtBoundary(
            mutableContext,
            "watchpoint",
            stopControlGeneration);
    }
}
#endif

PS2Runtime::DeferredGuestYieldScope::DeferredGuestYieldScope(bool &pendingOut) noexcept
    : m_pendingOut(pendingOut)
{
    ++g_deferredGuestYieldDepth;
}

PS2Runtime::DeferredGuestYieldScope::~DeferredGuestYieldScope()
{
    if (--g_deferredGuestYieldDepth == 0u && g_deferredGuestYieldPending)
    {
        g_deferredGuestYieldPending = false;
        m_pendingOut = true;
    }
}

void PS2Runtime::yieldGuestExecutionAtBoundary()
{
    if (m_eeRuntimeExecutor)
    {
        if (g_deferredGuestYieldDepth != 0u)
        {
            g_deferredGuestYieldPending = true;
            return;
        }
        yieldEeExecutorCurrent(
            isStopRequested()
                ? ps2x::ee::EeSchedulerExitReason::
                      StopRequested
                : ps2x::ee::EeSchedulerExitReason::
                      Preempted);
        return;
    }

    auto depthIt = g_guestExecutionDepths.find(this);
    if (g_deferredGuestYieldDepth != 0u ||
        depthIt == g_guestExecutionDepths.end() ||
        depthIt->second == 0u ||
        m_guestExecutionWaiters.load(
            std::memory_order_acquire) != 0u)
    {
        yieldGuestExecutionAfterWake();
        return;
    }

    // Raw operations can publish READY immediately before this ordinary
    // boundary, while their legacy host worker has not yet reached the guest
    // mutex. Give that worker the same bounded acquisition window as a wake
    // handoff. If no worker appears, the boundary had no alternate runnable
    // guest and must not be classified as a failed switch.
    if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticYieldRequests.fetch_add(
            1u, std::memory_order_relaxed);
    }

    const uint64_t handoffEpoch =
        m_guestExecutionHandoffEpoch.load(
            std::memory_order_acquire);
    bool handedOff = false;
    {
        GuestExecutionReleaseScope releaseGuestExecution(this);
        std::unique_lock<std::mutex> lock(
            m_guestExecutionHandoffMutex);
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticGuestSwitchCvWaits.fetch_add(
                1u, std::memory_order_relaxed);
        }
        handedOff = m_guestExecutionHandoffCv.wait_for(
            lock,
            std::chrono::milliseconds(2),
            [&]()
            {
                return m_guestExecutionHandoffEpoch.load(
                           std::memory_order_acquire) !=
                       handoffEpoch;
            });
    }

    if (m_eeThreadDiagnosticsEnabled)
    {
        if (handedOff)
        {
            m_eeThreadDiagnosticRequestedGuestSwitches.fetch_add(
                1u, std::memory_order_relaxed);
            m_eeThreadDiagnosticCompletedGuestSwitches.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else if (m_guestExecutionWaiters.load(
                     std::memory_order_acquire) != 0u)
        {
            m_eeThreadDiagnosticRequestedGuestSwitches.fetch_add(
                1u, std::memory_order_relaxed);
            m_eeThreadDiagnosticGuestSwitchTimeouts.fetch_add(
                1u, std::memory_order_relaxed);
        }
    }
}

void PS2Runtime::yieldGuestExecutionAfterWake()
{
    if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticYieldRequests.fetch_add(
            1u, std::memory_order_relaxed);
    }

    if (g_deferredGuestYieldDepth != 0u)
    {
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticDeferredYields.fetch_add(
                1u, std::memory_order_relaxed);
        }
        g_deferredGuestYieldPending = true;
        return;
    }

    if (m_eeRuntimeExecutor)
    {
        if (g_eeExecutorConsequenceContext.runtime ==
            this)
        {
            return;
        }
        yieldEeExecutorCurrent(
            isStopRequested()
                ? ps2x::ee::EeSchedulerExitReason::
                      StopRequested
                : ps2x::ee::EeSchedulerExitReason::
                      Preempted);
        return;
    }

    auto it = g_guestExecutionDepths.find(this);
    if (it == g_guestExecutionDepths.end() || it->second == 0u)
    {
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticHostThreadYields.fetch_add(
                1u, std::memory_order_relaxed);
        }
        std::this_thread::yield();
        return;
    }

    if (m_eeThreadDiagnosticsEnabled)
    {
        m_eeThreadDiagnosticRequestedGuestSwitches.fetch_add(
            1u, std::memory_order_relaxed);
    }
    const uint64_t handoffEpoch = m_guestExecutionHandoffEpoch.load(std::memory_order_acquire);
    bool handedOff = false;
    {
        GuestExecutionReleaseScope releaseGuestExecution(this);
        std::unique_lock<std::mutex> lock(m_guestExecutionHandoffMutex);
        if (m_eeThreadDiagnosticsEnabled)
        {
            m_eeThreadDiagnosticGuestSwitchCvWaits.fetch_add(
                1u, std::memory_order_relaxed);
        }
        handedOff = m_guestExecutionHandoffCv.wait_for(
            lock, std::chrono::milliseconds(2), [&]()
            {
                return m_guestExecutionHandoffEpoch.load(
                           std::memory_order_acquire) != handoffEpoch;
            });
    }
    if (m_eeThreadDiagnosticsEnabled)
    {
        if (handedOff)
        {
            m_eeThreadDiagnosticCompletedGuestSwitches.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else
        {
            m_eeThreadDiagnosticGuestSwitchTimeouts.fetch_add(
                1u, std::memory_order_relaxed);
        }
    }
}

void PS2Runtime::requestGuestPreemption()
{
    m_guestExecutionPreemptionGeneration.fetch_add(
        1u, std::memory_order_release);
    if (m_eeRuntimeExecutor)
    {
        // External alarm/device publishers may be the only runnable source
        // while every guest fiber is waiting. The preemption flag makes the
        // work visible; this notification releases the executor's idle wait.
        m_eeRuntimeExecutor->notify();
    }
}

void PS2Runtime::acknowledgeGuestPreemptionRequests() noexcept
{
    const uint64_t requestedGeneration =
        m_guestExecutionPreemptionGeneration.load(
            std::memory_order_acquire);
    m_guestExecutionConsumedPreemptionGeneration.store(
        requestedGeneration, std::memory_order_release);
}

bool PS2Runtime::consumeGuestPreemptionRequest() noexcept
{
    // Guest execution ownership serializes consumers. If a producer advances
    // the requested generation after this load, that newer request remains
    // visible to the next checkpoint instead of being cleared accidentally.
    const uint64_t requestedGeneration =
        m_guestExecutionPreemptionGeneration.load(
            std::memory_order_acquire);
    const uint64_t consumedGeneration =
        m_guestExecutionConsumedPreemptionGeneration.load(
            std::memory_order_relaxed);
    if (requestedGeneration == consumedGeneration)
    {
        return false;
    }

    m_guestExecutionConsumedPreemptionGeneration.store(
        requestedGeneration, std::memory_order_release);
    return true;
}

PS2GuestCheckpointResult
PS2Runtime::checkpointGuestExecution(
    R5900Context *ctx)
{
    static_cast<void>(ctx);
    const bool explicitlyRequested =
        consumeGuestPreemptionRequest();

    thread_local uint32_t s_backEdgeYieldCounter = 0u;
    const uint32_t waiterCount =
        m_guestExecutionWaiters.load(
            std::memory_order_acquire);
    const uint32_t yieldInterval =
        waiterCount != 0u ? 64u : 100u;
    const bool periodicCheckpoint =
        !explicitlyRequested &&
        ++s_backEdgeYieldCounter >= yieldInterval;
    if (!explicitlyRequested &&
        !periodicCheckpoint)
    {
        return PS2GuestCheckpointResult::Continue;
    }

    // The common generated back-edge check is not an architectural
    // publication point. A real checkpoint is: it can release ownership,
    // suspend the continuation, or unwind to the outer dispatcher.
    if (ctx)
    {
        ctx->finishEeInstruction();
    }

    if (periodicCheckpoint)
    {
        s_backEdgeYieldCounter = 0u;
    }

    if (g_deferredGuestYieldDepth != 0u)
    {
        // Interrupt and callback adapters may temporarily require one
        // uninterrupted guest-safe region. Materialize the resume PC by
        // returning from generated code, then let the outermost defer scope
        // hand control to the scheduler after that region closes.
        g_deferredGuestYieldPending = true;
        m_guestExecutionDispatcherExitEpoch.fetch_add(
            1u, std::memory_order_release);
        return PS2GuestCheckpointResult::
            ExitToDispatcher;
    }

    if (g_eeExecutorConsequenceContext.runtime ==
        this)
    {
        // Inline interrupt/callback generated code runs on the executor
        // stack, not inside a resumable guest fiber. Materialize its resume
        // PC for the bounded callback dispatcher instead of attempting a
        // fiber-to-executor switch with no running continuation.
        m_guestExecutionDispatcherExitEpoch.fetch_add(
            1u, std::memory_order_release);
        return PS2GuestCheckpointResult::
            ExitToDispatcher;
    }

    if (m_eeExecutionBackend &&
        m_eeExecutionBackend->checkpointMode() ==
            EeExecutionCheckpointMode::
                SuspendContinuation)
    {
        yieldEeExecutorCurrent(
            isStopRequested()
                ? ps2x::ee::EeSchedulerExitReason::
                      StopRequested
                : ps2x::ee::EeSchedulerExitReason::
                      Preempted);
        return PS2GuestCheckpointResult::Continue;
    }

    if (periodicCheckpoint &&
        waiterCount != 0u)
    {
        // Returning to the dispatcher is not itself a fair handoff: the
        // current host thread can release and immediately reacquire the
        // recursive mutex, indefinitely starving a guest thread which is
        // already queued. Transfer ownership before asking generated code to
        // return to its dispatcher.
        yieldGuestExecutionAfterWake();
    }
    m_guestExecutionDispatcherExitEpoch.fetch_add(
        1u, std::memory_order_release);
    return PS2GuestCheckpointResult::
        ExitToDispatcher;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    synchronizeEeCounterMmio(ctx, vaddr, 1u);
    debugObserveMemoryAccess(vaddr, 1u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read8(
            vaddr, guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, false);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataLoad, fault.physicalAddress());
        return 0u;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    synchronizeEeCounterMmio(ctx, vaddr, 2u);
    debugObserveMemoryAccess(vaddr, 2u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read16(
            vaddr, guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, false);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataLoad, fault.physicalAddress());
        return 0u;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    synchronizeEeCounterMmio(ctx, vaddr, 4u);
    debugObserveMemoryAccess(vaddr, 4u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read32(
            vaddr, guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, false);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataLoad, fault.physicalAddress());
        return 0u;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    synchronizeEeCounterMmio(ctx, vaddr, 8u);
    debugObserveMemoryAccess(vaddr, 8u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read64(
            vaddr, guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, false);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataLoad, fault.physicalAddress());
        return 0u;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    synchronizeEeCounterMmio(ctx, vaddr, 16u);
    debugObserveMemoryAccess(vaddr, 16u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read128(
            vaddr, guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_LOAD,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, false);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataLoad, fault.physicalAddress());
        return _mm_setzero_si128();
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    synchronizeEeCounterMmio(ctx, vaddr, 1u);
    debugObserveMemoryAccess(vaddr, 1u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 1u, value, 0u);
    try
    {
        m_memory.write8(
            vaddr,
            value,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataStore, fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    synchronizeEeCounterMmio(ctx, vaddr, 2u);
    debugObserveMemoryAccess(vaddr, 2u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 2u, value, 0u);
    try
    {
        m_memory.write16(
            vaddr,
            value,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataStore, fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    synchronizeEeCounterMmio(ctx, vaddr, 4u);
    debugObserveMemoryAccess(vaddr, 4u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 4u, value, 0u);
    try
    {
        m_memory.write32(
            vaddr,
            value,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataStore, fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    synchronizeEeCounterMmio(ctx, vaddr, 8u);
    debugObserveMemoryAccess(vaddr, 8u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 8u, value, 0u);
    try
    {
        m_memory.write64(
            vaddr,
            value,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataStore, fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::StoreMasked32(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t vaddr,
    uint32_t value,
    uint8_t byteEnable)
{
    const Ps2ByteEnableSpan span =
        Ps2DecodeByteEnableSpan(byteEnable, 4u);
    if (span.valid && span.size != 0u)
    {
        const uint32_t selectedAddress =
            vaddr + span.offset;
        const uint32_t selectedValue =
            value >> (span.offset * 8u);
        synchronizeEeCounterMmio(
            ctx, selectedAddress, span.size);
        debugObserveMemoryAccess(
            selectedAddress,
            span.size,
            DebugMemoryAccess::Write,
            ctx);
        ps2TraceGuestWrite(
            rdram,
            selectedAddress,
            span.size,
            selectedValue,
            0u,
            "WRITE_MASKED32",
            ctx);
        traceVif0MmioWrite(
            m_memory,
            ctx,
            selectedAddress,
            span.size,
            selectedValue,
            0u);
    }
    try
    {
        m_memory.writeMasked32(
            vaddr,
            value,
            byteEnable,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(
            ctx,
            PS2BusErrorAccess::DataStore,
            fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            vaddr);
    }
}

void PS2Runtime::StoreMasked64(
    uint8_t *rdram,
    R5900Context *ctx,
    uint32_t vaddr,
    uint64_t value,
    uint8_t byteEnable)
{
    const Ps2ByteEnableSpan span =
        Ps2DecodeByteEnableSpan(byteEnable, 8u);
    if (span.valid && span.size != 0u)
    {
        const uint32_t selectedAddress =
            vaddr + span.offset;
        const uint64_t selectedValue =
            value >> (span.offset * 8u);
        synchronizeEeCounterMmio(
            ctx, selectedAddress, span.size);
        debugObserveMemoryAccess(
            selectedAddress,
            span.size,
            DebugMemoryAccess::Write,
            ctx);
        ps2TraceGuestWrite(
            rdram,
            selectedAddress,
            span.size,
            selectedValue,
            0u,
            "WRITE_MASKED64",
            ctx);
        traceVif0MmioWrite(
            m_memory,
            ctx,
            selectedAddress,
            span.size,
            selectedValue,
            0u);
    }
    try
    {
        m_memory.writeMasked64(
            vaddr,
            value,
            byteEnable,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(
            ctx,
            PS2BusErrorAccess::DataStore,
            fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            vaddr);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    synchronizeEeCounterMmio(ctx, vaddr, 16u);
    debugObserveMemoryAccess(vaddr, 16u, DebugMemoryAccess::Write, ctx);
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 16u, _parts[0], _parts[1]);
    try
    {
        m_memory.write128(
            vaddr,
            value,
            ctx ? ctx->pc : 0u,
            guestAddressTranslation(ctx));
    }
    catch (const PS2AddressErrorException &fault)
    {
        SignalMemoryException(
            ctx,
            EXCEPTION_ADDRESS_ERROR_STORE,
            fault.virtualAddress());
    }
    catch (const PS2TlbFaultException &fault)
    {
        raiseTlbFault(ctx, fault, true);
    }
    catch (const PS2BusErrorException &fault)
    {
        SignalBusError(ctx, PS2BusErrorAccess::DataStore, fault.physicalAddress());
        return;
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_CHCR, chcr);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        m_debugRunUntilActive = false;
        m_debugRunUntilMpegUniquePicturesActive = false;
        m_debugRunBeforeNextMpegUniquePictureActive = false;
        m_debugRunForVSyncFieldsActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugSetPauseRequestedLocked(
            false, "runtime-stop");
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);
    if (m_eeRuntimeExecutor)
    {
        m_eeRuntimeExecutor->requestStop();
    }
    ps2_syscalls::notifyRuntimeStop(this);
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

[[noreturn]] void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Level1Exception(
        ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_syscalls::resetFileIoState(this);
    ps2_syscalls::resetDeci2State(this);
    ps2_stubs::resetCdState(this);
    ps2_stubs::resetDmaState(this);
    ps2_stubs::resetLibCState(this);
    ps2_stubs::resetMemoryCardState(this);
    ps2_stubs::resetPadState(this);
    ps2_stubs::resetSifState(this);
    resetIop();
    ps2_stubs::resetAudioStubState(this);
    ps2_stubs::resetGsSyncVCallbackState(this);
    ps2_stubs::resetMpegStubState(this);
    m_hostPresentationUploadState->reset();
    ps2_syscalls::initializeGuestKernelState(
        m_memory.getRDRAM(), this);
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);
    {
        std::lock_guard<std::recursive_timed_mutex> lock(
            m_guestExecutionMutex);
        resetEeVSyncPacingUnlocked();
    }

    const char *const gsHistoryDumpPath = std::getenv(GS_HISTORY_DUMP_ENV);
    if (gsHistoryDumpPath && gsHistoryDumpPath[0] != '\0')
    {
        (void)submitGsCommand(
            GsClearDebugHistoryCommand{});
        (void)submitGsCommand(
            GsSetDebugHistoryPausedCommand{
                .paused = false});
    }

    std::clog
        << "[ee-execution] "
        << eeExecutionBackendDiagnostics(
               m_eeExecutionBackend->kind())
        << std::endl;
    RUNTIME_LOG(
        "Starting execution at address 0x"
        << std::hex << m_cpuContext.pc << std::dec
        << " with EE backend "
        << eeExecutionBackendName());

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(
        FB_WIDTH, DEFAULT_DISPLAY_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);

    const bool dedicatedExecutor =
        usesDedicatedEeExecutor();
    m_eeThreadRuntimeState->activeHostThreads.store(
        dedicatedExecutor ? 0 : 1,
        std::memory_order_relaxed);

    // Arm emulated VSync before the first guest instruction so event mode has
    // a deterministic phase independent of host thread startup.
    ps2_syscalls::EnsureVSyncScheduled(
        m_memory.getRDRAM(), this);

    if (dedicatedExecutor)
    {
        startDedicatedEeExecution();
    }
    else
    {
        m_eeExecutionBackend->create(
            1,
            [this]()
            {
                ThreadNaming::SetCurrentThreadName(
                    "GameThread");
                try
                {
                    dispatchLoop(
                        m_memory.getRDRAM(),
                        &m_cpuContext);
                    const uint32_t pc =
                        m_debugPc.load(
                            std::memory_order_relaxed);
                    RUNTIME_LOG(
                        "Game thread returned. PC=0x"
                        << std::hex << pc
                        << " RA=0x"
                        << static_cast<uint32_t>(
                               _mm_extract_epi32(
                                   m_cpuContext.r[31],
                                   0))
                        << std::dec << std::endl);
                }
                catch (const std::exception &e)
                {
                    std::cerr
                        << "Error during program "
                           "execution: "
                        << e.what() << std::endl;
                }
                catch (...)
                {
                    std::cerr
                        << "Error during program "
                           "execution: unknown exception"
                        << std::endl;
                }
                m_eeThreadRuntimeState
                    ->activeHostThreads.fetch_sub(
                        1,
                        std::memory_order_relaxed);
            });
    }

    uint64_t tick = 0;
    using WindowRateClock = std::chrono::steady_clock;
    constexpr auto windowTitleUpdateInterval =
        std::chrono::seconds(2);
    auto windowTitleSampleTime =
        WindowRateClock::now();
    uint64_t windowTitleSampleFields =
        m_debugVSyncFields.load(
            std::memory_order_relaxed);
    uint64_t windowTitleSamplePresentations =
        takeGsCommandResult<GsProgressSnapshotResult>(
            submitGsCommand(
                GsProgressSnapshotCommand{}))
            .snapshot.presentations;
    uint64_t windowTitleSamplePeriodCycles =
        m_debugVSyncPeriodCycles.load(
            std::memory_order_relaxed);
#if !defined(PLATFORM_VITA)
    const WindowRateClock::duration hostFramePeriod =
        std::chrono::duration_cast<
            WindowRateClock::duration>(
            std::chrono::duration<double>(1.0 / 60.0));
    WindowRateClock::time_point hostFrameDeadline =
        WindowRateClock::now();
#endif
    while (
        !isStopRequested() &&
        (dedicatedExecutor
             ? m_eeRuntimeExecutor->running()
             : m_eeThreadRuntimeState
                       ->activeHostThreads.load(
                           std::memory_order_relaxed) >
                   0))
    {
        if (dedicatedExecutor &&
            m_eeExecutionBackend
                    ->managedThreadCount() == 0u)
        {
            requestStop();
            break;
        }
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const int activeThreads = m_eeThreadRuntimeState->activeHostThreads.load(std::memory_order_relaxed);

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << activeThreads
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);
            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        UploadFrame(
            frameTex,
            this,
            *m_hostPresentationUploadState,
            presentWidth,
            presentHeight);

        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(
            std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(
            std::max<uint32_t>(1u, presentHeight));
        const float screenWidth =
            static_cast<float>(GetScreenWidth());
        const float screenHeight =
            static_cast<float>(GetScreenHeight());
        float presentationAspect = 4.0f / 3.0f;
        switch (m_hostPresentationVideoModeClass.load(
            std::memory_order_relaxed))
        {
        case EeVSyncVideoModeClass::Vesa:
            presentationAspect = srcWidth / srcHeight;
            break;
        case EeVSyncVideoModeClass::Hdtv720P:
        case EeVSyncVideoModeClass::Hdtv1080I:
        case EeVSyncVideoModeClass::Hdtv1080P:
            presentationAspect = 16.0f / 9.0f;
            break;
        default:
            break;
        }
        const float dstWidth = std::min(
            screenWidth, screenHeight * presentationAspect);
        const float dstHeight = dstWidth / presentationAspect;
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        DrawTexturePro(frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

        const WindowRateClock::time_point windowTitleNow =
            WindowRateClock::now();
        const WindowRateClock::duration windowTitleElapsed =
            windowTitleNow - windowTitleSampleTime;
        if (windowTitleElapsed >=
            windowTitleUpdateInterval)
        {
            const uint64_t currentFields =
                m_debugVSyncFields.load(
                    std::memory_order_relaxed);
            const uint64_t currentPresentations =
                takeGsCommandResult<GsProgressSnapshotResult>(
                    submitGsCommand(
                        GsProgressSnapshotCommand{}))
                    .snapshot.presentations;
            const uint64_t currentPeriodCycles =
                m_debugVSyncPeriodCycles.load(
                    std::memory_order_relaxed);

            if (currentPeriodCycles != 0u &&
                currentPeriodCycles ==
                    windowTitleSamplePeriodCycles &&
                currentFields >=
                    windowTitleSampleFields &&
                currentPresentations >=
                    windowTitleSamplePresentations)
            {
                const double elapsedSeconds =
                    std::chrono::duration<double>(
                        windowTitleElapsed)
                        .count();
                const double guestFramesPerSecond =
                    static_cast<double>(
                        currentPresentations -
                        windowTitleSamplePresentations) /
                    elapsedSeconds;
                const double expectedFields =
                    elapsedSeconds *
                    static_cast<double>(
                        EE_CYCLES_PER_SECOND) /
                    static_cast<double>(
                        currentPeriodCycles);
                const double runRatePercent =
                    expectedFields > 0.0
                        ? static_cast<double>(
                              currentFields -
                              windowTitleSampleFields) *
                              100.0 /
                              expectedFields
                        : 0.0;
                std::array<char, 512u> title{};
                std::snprintf(
                    title.data(),
                    title.size(),
                    "%.1f FPS | %.1f%% speed",
                    guestFramesPerSecond,
                    runRatePercent);
                SetWindowTitle(title.data());
            }

            windowTitleSampleTime =
                windowTitleNow;
            windowTitleSampleFields =
                currentFields;
            windowTitleSamplePresentations =
                currentPresentations;
            windowTitleSamplePeriodCycles =
                currentPeriodCycles;
        }

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }

#if !defined(PLATFORM_VITA)
        hostFrameDeadline += hostFramePeriod;
        const WindowRateClock::time_point hostFrameNow =
            WindowRateClock::now();
        if (hostFrameNow < hostFrameDeadline)
        {
            std::this_thread::sleep_until(
                hostFrameDeadline);
        }
        else if (hostFrameNow - hostFrameDeadline >=
                 hostFramePeriod)
        {
            // Do not replay missed presentation slots after a slow frame.
            hostFrameDeadline = hostFrameNow;
        }
#endif
    }

    requestStop();
    if (dedicatedExecutor)
    {
        try
        {
            joinDedicatedEeExecution();
        }
        catch (const std::exception &error)
        {
            std::cerr
                << "[run] EE executor failure: "
                << error.what() << std::endl;
        }
    }
    else
    {
        const auto joinDeadline =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(2);
        while (!m_eeExecutionBackend->isFinished(1) &&
               std::chrono::steady_clock::now() <
                   joinDeadline)
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
        }

        if (m_eeExecutionBackend->isFinished(1))
        {
            m_eeExecutionBackend->destroy(1);
        }
        else
        {
            std::cerr
                << "[run] game thread did not stop within "
                   "timeout; detaching"
                << std::endl;
            m_eeExecutionBackend->detach(1);
        }

        const auto workerDeadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(1000);
        while (
            m_eeThreadRuntimeState
                    ->activeHostThreads.load(
                        std::memory_order_relaxed) >
                0 &&
            std::chrono::steady_clock::now() <
                workerDeadline)
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
        }

        if (m_eeThreadRuntimeState
                ->activeHostThreads.load(
                    std::memory_order_relaxed) > 0)
        {
            requestStop();
            const auto finalWorkerDeadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(1000);
            while (
                m_eeThreadRuntimeState
                        ->activeHostThreads.load(
                            std::memory_order_relaxed) >
                    0 &&
                std::chrono::steady_clock::now() <
                    finalWorkerDeadline)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));
            }
        }

        if (m_eeThreadRuntimeState
                ->activeHostThreads.load(
                    std::memory_order_relaxed) == 0)
        {
            ps2_syscalls::joinAllGuestHostThreads(
                this);
        }
        else
        {
            std::cerr
                << "[run] guest host threads did not stop "
                   "within timeout; detaching remaining "
                   "worker threads"
                << std::endl;
            ps2_syscalls::detachAllGuestHostThreads(
                this);
        }
    }

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    if (gsHistoryDumpPath && gsHistoryDumpPath[0] != '\0')
    {
        (void)submitGsCommand(
            GsSetDebugHistoryPausedCommand{
                .paused = true});
        const GSDebugSnapshot snapshot =
            takeGsCommandResult<GsDebugSnapshotResult>(
                submitGsCommand(
                    GsDebugSnapshotCommand{}))
                .snapshot;
        const std::vector<GSDebugHistoryEntry> history =
            takeGsCommandResult<GsDebugHistoryResult>(
                submitGsCommand(
                    GsDebugHistoryCommand{}))
                .entries;
        std::ofstream out(gsHistoryDumpPath, std::ios::out | std::ios::trunc);
        if (!out)
        {
            std::cerr << "[gs:history] failed to open dump path: "
                      << gsHistoryDumpPath << std::endl;
        }
        else
        {
            out << "snapshot"
                << " ctx0.frame=" << snapshot.ctx[0].frame.fbp
                << "/" << snapshot.ctx[0].frame.fbw
                << "/0x" << std::hex << static_cast<uint32_t>(snapshot.ctx[0].frame.psm)
                << " ctx1.frame=" << std::dec << snapshot.ctx[1].frame.fbp
                << "/" << snapshot.ctx[1].frame.fbw
                << "/0x" << std::hex << static_cast<uint32_t>(snapshot.ctx[1].frame.psm)
                << " present=" << std::dec << snapshot.hostPresentationWidth
                << "x" << snapshot.hostPresentationHeight
                << " display=" << snapshot.hostPresentationDisplayFbp
                << " source=" << snapshot.hostPresentationSourceFbp
                << " preferred=" << snapshot.hostPresentationUsedPreferred
                << '\n';
            out << "seq\tframe\ttick\tkind\tprim\tflags\tframebuf\tzbuf\ttex0"
                   "\ttest\talpha\tbounds\tgif\treg\ttransfer\tpresent\n";
            for (const GSDebugHistoryEntry &entry : history)
            {
                out << entry.seq << '\t'
                    << entry.frameIndex << '\t'
                    << entry.vsyncTick << '\t'
                    << static_cast<uint32_t>(entry.kind) << '\t'
                    << static_cast<uint32_t>(entry.prim.type) << '\t'
                    << "iip=" << entry.prim.iip
                    << "/tme=" << entry.prim.tme
                    << "/abe=" << entry.prim.abe
                    << "/fst=" << entry.prim.fst
                    << "/ctxt=" << entry.prim.ctxt << '\t'
                    << entry.frame.fbp << "/" << entry.frame.fbw
                    << "/0x" << std::hex << static_cast<uint32_t>(entry.frame.psm)
                    << "/0x" << entry.frame.fbmsk << std::dec << '\t'
                    << entry.zbuf.zbp << "/0x" << std::hex
                    << static_cast<uint32_t>(entry.zbuf.psm)
                    << "/m=" << std::dec << entry.zbuf.zmask << '\t'
                    << entry.tex0.tbp0 << "/" << static_cast<uint32_t>(entry.tex0.tbw)
                    << "/0x" << std::hex << static_cast<uint32_t>(entry.tex0.psm)
                    << "/" << std::dec << (1u << entry.tex0.tw)
                    << "x" << (1u << entry.tex0.th)
                    << "/cbp=" << entry.tex0.cbp
                    << "/cpsm=0x" << std::hex << static_cast<uint32_t>(entry.tex0.cpsm)
                    << std::dec << '\t'
                    << "0x" << std::hex << entry.test << '\t'
                    << "0x" << entry.alpha << std::dec << '\t'
                    << entry.xMin << ".." << entry.xMax
                    << "/" << entry.yMin << ".." << entry.yMax
                    << "/z=" << entry.zMin << ".." << entry.zMax
                    << "/a=" << static_cast<uint32_t>(entry.aMin)
                    << ".." << static_cast<uint32_t>(entry.aMax) << '\t'
                    << "nloop=" << entry.gifNloop
                    << "/flg=" << static_cast<uint32_t>(entry.gifFlg)
                    << "/nreg=" << static_cast<uint32_t>(entry.gifNreg)
                    << "/bytes=" << entry.gifSizeBytes << '\t'
                    << "0x" << std::hex << static_cast<uint32_t>(entry.reg)
                    << "=0x" << entry.regValue << std::dec << '\t'
                    << entry.trxreg.rrw << "x" << entry.trxreg.rrh
                    << "/dir=" << entry.trxdir
                    << "/pixels=" << entry.transferPixels << '\t'
                    << entry.width << "x" << entry.height
                    << "/display=" << entry.displayFbp
                    << "/source=" << entry.sourceFbp
                    << "/preferred=" << entry.usedPreferred
                    << '\n';
            }
            std::cout << "[gs:history] wrote " << history.size()
                      << " entries to " << gsHistoryDumpPath << std::endl;
        }
    }

    UnloadTexture(frameTex);
    CloseWindow();

    const int remainingThreads = m_eeThreadRuntimeState->activeHostThreads.load(std::memory_order_relaxed);
    RUNTIME_LOG("[run] exiting loop, activeThreads=" << remainingThreads);
    if (remainingThreads > 0)
    {
        std::cerr << "[run] warning: " << remainingThreads
                  << " guest worker thread(s) still active during shutdown." << std::endl;
    }
    emitEeTlbTranslationCacheDiagnostics(
        m_memory,
        m_emitTlbTranslationCacheDiagnosticsOnDestruction);
}

#undef PS2X_EE_OBSERVATION_COLD
