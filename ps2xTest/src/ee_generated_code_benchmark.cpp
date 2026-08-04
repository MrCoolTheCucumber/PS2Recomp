#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    using namespace ps2x::performance;

    constexpr uint32_t kOutputSchemaVersion = 2u;
    constexpr uint32_t kInstructionsPerBlock = 10u;
    constexpr uint32_t kLoadsPerBlock = 1u;
    constexpr uint32_t kStoresPerBlock = 1u;
    constexpr uint32_t kCycleTicksPerInstruction = 8u;
    constexpr uint32_t kCodeBase = 0x00180004u;
    constexpr uint32_t kVirtualBase = 0x00400000u;
    constexpr uint32_t kIdentityPhysicalBase = kVirtualBase;
    constexpr uint32_t kMappedPhysicalBase = 0x00800000u;
    constexpr uint32_t kDirectBase =
        0x80000000u | kMappedPhysicalBase;
    constexpr uint32_t kWorkingSetBytes = 4096u;
    constexpr uint32_t kWorkingSetWords =
        kWorkingSetBytes / sizeof(uint32_t);
    constexpr uint32_t kAddressSlots = 256u;
    constexpr uint32_t kAsid = 0x46u;

    constexpr uint32_t kPccrCte = 1u << 31u;
    constexpr uint32_t kPccrK0 = 1u << 2u;
    constexpr uint32_t kPccrK1 = 1u << 12u;
    constexpr uint32_t kConfigBpe = 1u << 12u;
    constexpr uint32_t kConfigDie = 1u << 18u;

    constexpr uint32_t kBpcIae = 1u << 31u;
    constexpr uint32_t kBpcDre = 1u << 30u;
    constexpr uint32_t kBpcDwe = 1u << 29u;
    constexpr uint32_t kBpcDve = 1u << 28u;
    constexpr uint32_t kBpcIke = 1u << 24u;
    constexpr uint32_t kBpcDke = 1u << 19u;
    constexpr uint32_t kBpcBed = 1u << 15u;
    constexpr uint32_t kBpcDwb = 1u << 2u;
    constexpr uint32_t kBpcDrb = 1u << 1u;
    constexpr uint32_t kBpcIab = 1u << 0u;
    constexpr uint32_t kBreakpointStickyMask =
        kBpcDwb | kBpcDrb | kBpcIab;

    constexpr uint32_t kCauseExc2Mask = 0x00070000u;
    constexpr uint32_t kCauseExc2Performance = 0x00020000u;
    constexpr uint32_t kStatusErl = 1u << 2u;

#if defined(_MSC_VER)
#define PS2_BENCHMARK_NOINLINE __declspec(noinline)
#define PS2_BENCHMARK_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_BENCHMARK_NOINLINE __attribute__((noinline, noipa))
#define PS2_BENCHMARK_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define PS2_BENCHMARK_NOINLINE __attribute__((noinline))
#define PS2_BENCHMARK_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

    enum class ObservationEntry : uint8_t
    {
        FastCompiledOut,
        PolicySelected,
        Precise,
    };

    enum class MemoryPath : uint8_t
    {
        MappedKuseg,
        DirectKseg,
    };

    enum class Feature : uint8_t
    {
        Inactive,
        InstructionBreakpoint,
        DataAddressBreakpoint,
        DataValueBreakpoint,
        PccrCycles,
        PccrIssue,
        PccrBranch,
        PccrLowOrderBranch,
        PccrCompletion,
        PccrCoprocessor,
        PccrMemory,
        PccrOverflowLastCompletion,
    };

    struct BenchmarkCase
    {
        std::string_view name;
        ObservationEntry entry = ObservationEntry::Precise;
        MemoryPath memory = MemoryPath::MappedKuseg;
        Feature feature = Feature::Inactive;
    };

    constexpr std::array<BenchmarkCase, 17u> kCases{{
        {"fast_control_mapped", ObservationEntry::FastCompiledOut,
         MemoryPath::MappedKuseg, Feature::Inactive},
        {"fast_control_direct", ObservationEntry::FastCompiledOut,
         MemoryPath::DirectKseg, Feature::Inactive},
        {"policy_inactive_mapped", ObservationEntry::PolicySelected,
         MemoryPath::MappedKuseg, Feature::Inactive},
        {"policy_inactive_direct", ObservationEntry::PolicySelected,
         MemoryPath::DirectKseg, Feature::Inactive},
        {"precise_inactive_mapped", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::Inactive},
        {"precise_inactive_direct", ObservationEntry::Precise,
         MemoryPath::DirectKseg, Feature::Inactive},
        {"instruction_breakpoint", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::InstructionBreakpoint},
        {"data_address_breakpoint", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::DataAddressBreakpoint},
        {"data_value_breakpoint", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::DataValueBreakpoint},
        {"pccr_cycles", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrCycles},
        {"pccr_issue", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrIssue},
        {"pccr_branch", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrBranch},
        {"pccr_low_order_branch", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrLowOrderBranch},
        {"pccr_completion", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrCompletion},
        {"pccr_coprocessor", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrCoprocessor},
        {"pccr_memory", ObservationEntry::Precise,
         MemoryPath::MappedKuseg, Feature::PccrMemory},
        {"pccr_overflow_last_completion", ObservationEntry::Precise,
         MemoryPath::MappedKuseg,
         Feature::PccrOverflowLastCompletion},
    }};

    struct Configuration
    {
        uint64_t blocks = 1'000'000u;
        uint64_t warmupBlocks = 20'000u;
        uint32_t samples = 5u;
        std::string_view caseFilter;
    };

    struct Measurement
    {
        const BenchmarkCase *benchmarkCase = nullptr;
        uint32_t sample = 0u;
        uint64_t blocks = 0u;
        uint64_t guestInstructions = 0u;
        uint64_t guestLoads = 0u;
        uint64_t guestStores = 0u;
        uint64_t hostNanoseconds = 0u;
        uint64_t registerHash = 0u;
        uint64_t memoryHash = 0u;
        uint64_t identityMemoryHash = 0u;
        uint64_t initialIdentityMemoryHash = 0u;
        uint64_t workHash = 0u;
        uint64_t counterHash = 0u;
        uint64_t insnCount = 0u;
        uint32_t random = 0u;
        uint32_t bpc = 0u;
        uint32_t pccr = 0u;
        uint32_t pcr0 = 0u;
        uint32_t pcr1 = 0u;
        uint32_t pc = 0u;
        uint32_t status = 0u;
        uint32_t cause = 0u;
        uint32_t errorEpc = 0u;
        bool issuePending = false;
        bool overflowDelivered = false;
    };

    class Fnv1a64
    {
    public:
        void addBytes(const void *data, size_t size) noexcept
        {
            const auto *bytes = static_cast<const uint8_t *>(data);
            for (size_t index = 0u; index < size; ++index)
            {
                m_value ^= bytes[index];
                m_value *= 1099511628211ull;
            }
        }

        template <typename Value>
        void add(const Value &value) noexcept
        {
            addBytes(&value, sizeof(value));
        }

        uint64_t value() const noexcept { return m_value; }

    private:
        uint64_t m_value = 1469598103934665603ull;
    };

    constexpr uint32_t makeEntryLo(
        uint32_t pfn,
        bool dirty,
        bool valid,
        bool global)
    {
        return ((pfn & 0x000fffffu) << 6u) |
               (2u << 3u) |
               (dirty ? 0x4u : 0u) |
               (valid ? 0x2u : 0u) |
               (global ? 0x1u : 0u);
    }

    constexpr uint32_t performanceControl(
        uint32_t event0,
        uint32_t event1)
    {
        return kPccrCte |
               kPccrK0 |
               kPccrK1 |
               (event0 << 5u) |
               (event1 << 15u);
    }

    constexpr uint32_t rotateLeft(uint32_t value, uint32_t amount)
    {
        return (value << amount) |
               (value >> (32u - amount));
    }

    bool parseUnsigned(std::string_view text, uint64_t &value)
    {
        if (text.empty())
            return false;
        const char *const begin = text.data();
        const char *const end = begin + text.size();
        const auto result = std::from_chars(begin, end, value);
        return result.ec == std::errc{} && result.ptr == end;
    }

    bool parseArguments(
        int argc,
        char **argv,
        Configuration &configuration)
    {
        for (int index = 1; index < argc; index += 2)
        {
            if (index + 1 >= argc)
                return false;
            const std::string_view option = argv[index];
            if (option == "--case")
            {
                configuration.caseFilter = argv[index + 1];
                continue;
            }
            uint64_t value = 0u;
            if (!parseUnsigned(argv[index + 1], value))
                return false;

            if (option == "--blocks")
            {
                configuration.blocks = value;
            }
            else if (option == "--warmup-blocks")
            {
                configuration.warmupBlocks = value;
            }
            else if (option == "--samples")
            {
                if (value == 0u ||
                    value > std::numeric_limits<uint32_t>::max())
                {
                    return false;
                }
                configuration.samples = static_cast<uint32_t>(value);
            }
            else
            {
                return false;
            }
        }

        constexpr uint64_t maximumBlocksFromTicks =
            std::numeric_limits<uint32_t>::max() /
            (kInstructionsPerBlock * kCycleTicksPerInstruction);
        constexpr uint64_t maximumBlocksFromOverflowFixture =
            0x7fffffffull / kInstructionsPerBlock;
        const uint64_t maximumBlocks = std::min(
            maximumBlocksFromTicks,
            maximumBlocksFromOverflowFixture);
        return configuration.blocks != 0u &&
               configuration.blocks <= maximumBlocks &&
               configuration.warmupBlocks <= maximumBlocks;
    }

    std::string_view entryName(ObservationEntry entry)
    {
        switch (entry)
        {
        case ObservationEntry::FastCompiledOut:
            return "fast_compiled_out";
        case ObservationEntry::PolicySelected:
            return "policy_selected";
        case ObservationEntry::Precise:
            return "precise";
        }
        return "unknown";
    }

    std::string_view memoryPathName(MemoryPath memory)
    {
        return memory == MemoryPath::MappedKuseg
                   ? "mapped_kuseg_non_identity"
                   : "direct_kseg";
    }

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE void beginInstruction(
        PS2Runtime *runtime,
        R5900Context *ctx,
        uint32_t pc,
        uint32_t pipeMask,
        uint32_t gprReadMask,
        uint32_t gprWriteMask,
        uint32_t flags)
    {
        ctx->pc = pc;
        if constexpr (Observe)
        {
            runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);
            if ((ctx->cop0_perf & kPccrCte) != 0u)
            {
                runtime->recordEeInstructionIssue(
                    ctx,
                    {pipeMask, gprReadMask, gprWriteMask, flags});
            }
        }
        ctx->beginEeInstruction();
        ctx->advanceEeCycleTicks(kCycleTicksPerInstruction);
    }

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE void completeInstruction(
        PS2Runtime *runtime,
        R5900Context *ctx,
        uint32_t completionEvents)
    {
        if constexpr (Observe)
        {
            if ((ctx->cop0_perf & kPccrCte) != 0u)
            {
                runtime->recordEeInstructionCompletion(
                    ctx, completionEvents);
            }
        }
    }

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE uint32_t generatedLoad32(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address)
    {
        if constexpr (Observe)
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

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE void generatedStore32(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address,
        uint32_t value)
    {
        if constexpr (Observe)
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

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE void recordConditionalBranch(
        PS2Runtime *runtime,
        R5900Context *ctx,
        uint32_t pc,
        bool taken)
    {
        if constexpr (Observe)
        {
            if ((ctx->cop0_perf & kPccrCte) != 0u)
            {
                runtime->recordEeConditionalBranch(
                    ctx, pc, taken);
            }
        }
    }

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE void recordPredictedBranch(
        PS2Runtime *runtime,
        R5900Context *ctx,
        uint32_t pc)
    {
        if constexpr (Observe)
        {
            if ((ctx->cop0_perf & kPccrCte) != 0u)
            {
                runtime->recordEePredictedBranch(ctx, pc);
            }
        }
    }

    template <bool Observe>
    PS2_BENCHMARK_ALWAYS_INLINE void generatedFixedWork(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t addressBase,
        uint64_t blocks)
    {
        constexpr uint32_t normalCompletion =
            kInstructionCompleted |
            kNonBdsInstructionCompleted;
        constexpr uint32_t branchCompletion =
            kInstructionCompleted;

        for (uint64_t block = 0u; block < blocks; ++block)
        {
            const uint32_t address =
                addressBase +
                static_cast<uint32_t>(block & (kAddressSlots - 1u)) *
                    sizeof(uint32_t);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 0u,
                kIssuePipe0, 1u << 2u, 1u << 2u, 0u);
            SET_GPR_U32(
                ctx, 2u,
                ADD32(GPR_U32(ctx, 2u), 0x9e3779b9u));
            completeInstruction<Observe>(
                runtime, ctx, normalCompletion);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 4u,
                kIssuePipe1, 1u << 3u, 1u << 3u, 0u);
            SET_GPR_U32(
                ctx, 3u,
                rotateLeft(GPR_U32(ctx, 3u), 5u) ^
                    0xa5a5a5a5u);
            completeInstruction<Observe>(
                runtime, ctx, normalCompletion);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 8u,
                kIssuePipe1, 0u, 1u << 4u, 0u);
            const uint32_t loaded = generatedLoad32<Observe>(
                rdram, ctx, runtime, address);
            SET_GPR_U32(ctx, 4u, loaded);
            completeInstruction<Observe>(
                runtime, ctx,
                normalCompletion | kLoadCompleted);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 12u,
                kIssuePipe0, 1u << 6u, 1u << 6u, 0u);
            SET_GPR_U32(
                ctx, 6u,
                ADD32(
                    MUL32(GPR_U32(ctx, 6u), 1664525u),
                    1013904223u));
            completeInstruction<Observe>(
                runtime, ctx, normalCompletion);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 16u,
                kIssuePipe1,
                (1u << 2u) | (1u << 3u) |
                    (1u << 4u) | (1u << 6u),
                0u, 0u);
            const uint32_t stored =
                GPR_U32(ctx, 2u) ^ GPR_U32(ctx, 3u) ^
                GPR_U32(ctx, 4u) ^ GPR_U32(ctx, 6u);
            generatedStore32<Observe>(
                rdram, ctx, runtime, address, stored);
            completeInstruction<Observe>(
                runtime, ctx,
                normalCompletion | kStoreCompleted);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 20u,
                kIssuePipe0, 0u, 0u, 0u);
            ctx->f[1] = FPU_ADD_S(ctx->f[1], ctx->f[2]);
            completeInstruction<Observe>(
                runtime, ctx,
                normalCompletion | kCop1InstructionCompleted);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 24u,
                kIssuePipe0, 0u, 0u, 0u);
            ctx->vi[1] = static_cast<uint16_t>(
                ctx->vi[1] + ctx->vi[2]);
            completeInstruction<Observe>(
                runtime, ctx,
                normalCompletion | kCop2InstructionCompleted);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 28u,
                kIssuePipe0 | kIssuePipe1,
                0u, 0u, kIssueBranch);
            const bool taken = (block & 1u) == 0u;
            recordConditionalBranch<Observe>(
                runtime, ctx, kCodeBase + 28u, taken);
            SET_GPR_U32(
                ctx, 7u,
                taken
                    ? ADD32(GPR_U32(ctx, 7u), GPR_U32(ctx, 2u))
                    : XOR32(GPR_U32(ctx, 7u), GPR_U32(ctx, 3u)));
            completeInstruction<Observe>(
                runtime, ctx, branchCompletion);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 32u,
                kIssuePipe0, 1u << 7u, 1u << 7u, 0u);
            SET_GPR_U32(
                ctx, 7u,
                XOR32(
                    GPR_U32(ctx, 7u),
                    rotateLeft(GPR_U32(ctx, 6u), 13u)));
            completeInstruction<Observe>(
                runtime, ctx, normalCompletion);

            beginInstruction<Observe>(
                runtime, ctx, kCodeBase + 36u,
                kIssuePipe0 | kIssuePipe1,
                0u, 0u, kIssueBranch);
            recordPredictedBranch<Observe>(
                runtime, ctx, kCodeBase + 36u);
            completeInstruction<Observe>(
                runtime, ctx, branchCompletion);
        }
    }

    extern "C" PS2_BENCHMARK_NOINLINE
    void EeGeneratedBenchmarkFastEntry(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t addressBase,
        uint64_t blocks)
    {
        generatedFixedWork<false>(
            rdram, ctx, runtime, addressBase, blocks);
    }

    extern "C" PS2_BENCHMARK_NOINLINE
    void EeGeneratedBenchmarkPreciseEntry(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t addressBase,
        uint64_t blocks)
    {
        generatedFixedWork<true>(
            rdram, ctx, runtime, addressBase, blocks);
    }

    extern "C" PS2_BENCHMARK_NOINLINE
    void EeGeneratedBenchmarkPolicyEntry(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t addressBase,
        uint64_t blocks)
    {
        if (PS2Runtime::architecturalObservationMode(ctx) ==
            EeArchitecturalObservationMode::Fast)
        {
            EeGeneratedBenchmarkFastEntry(
                rdram, ctx, runtime, addressBase, blocks);
        }
        else
        {
            EeGeneratedBenchmarkPreciseEntry(
                rdram, ctx, runtime, addressBase, blocks);
        }
    }

    class Fixture
    {
    public:
        explicit Fixture(
            const BenchmarkCase &benchmarkCase,
            uint64_t blocks)
            : m_case(benchmarkCase),
              m_blocks(blocks)
        {
            if (!runtime.memory().initialize())
            {
                throw std::runtime_error(
                    "failed to initialize PS2 memory");
            }
            for (uint32_t index = 0u;
                 index < runtime.memory().tlbEntryCount();
                 ++index)
            {
                if (!runtime.memory().tlbWrite(
                        index, EeTlbEntry{}))
                {
                    throw std::runtime_error(
                        "failed to clear the EE TLB");
                }
            }

            const uint32_t evenPfn =
                kMappedPhysicalBase >> 12u;
            if (!runtime.memory().tlbWrite(
                    12u,
                    EeTlbEntry{
                        0u,
                        kVirtualBase | kAsid,
                        makeEntryLo(
                            evenPfn, true, true, false),
                        makeEntryLo(
                            evenPfn + 1u,
                            true, true, false),
                    }))
            {
                throw std::runtime_error(
                    "failed to install the non-identity TLB mapping");
            }

            rdram = runtime.memory().getRDRAM();
            if (!rdram)
            {
                throw std::runtime_error("RDRAM is unavailable");
            }
            initializeMemory();
            initializeContext();
            configureFeature();
            initialIdentityMemoryHash = hashMemory(
                rdram + kIdentityPhysicalBase,
                kWorkingSetBytes);
        }

        void run()
        {
            const uint32_t addressBase =
                m_case.memory == MemoryPath::MappedKuseg
                    ? kVirtualBase
                    : kDirectBase;
            if (m_case.entry == ObservationEntry::FastCompiledOut)
            {
                EeGeneratedBenchmarkFastEntry(
                    rdram, &context, &runtime,
                    addressBase, m_blocks);
            }
            else if (m_case.entry == ObservationEntry::PolicySelected)
            {
                EeGeneratedBenchmarkPolicyEntry(
                    rdram, &context, &runtime,
                    addressBase, m_blocks);
            }
            else
            {
                EeGeneratedBenchmarkPreciseEntry(
                    rdram, &context, &runtime,
                    addressBase, m_blocks);
            }
        }

        void finish()
        {
            try
            {
                runtime.serviceEeEventsAtBlockBoundary(
                    rdram, &context);
            }
            catch (const PS2GuestException &)
            {
                overflowDelivered = true;
            }

            const bool expectsOverflow =
                m_case.feature ==
                Feature::PccrOverflowLastCompletion;
            if (overflowDelivered != expectsOverflow)
            {
                throw std::runtime_error(
                    expectsOverflow
                        ? "last-completion overflow was not delivered"
                        : "unexpected guest exception while finishing work");
            }

            if (!expectsOverflow)
            {
                (void)runtime.readCop0Performance(
                    rdram, &context, 1u);
            }
        }

        Measurement measurement(
            uint32_t sample,
            uint64_t hostNanoseconds) const
        {
            Measurement result{};
            result.benchmarkCase = &m_case;
            result.sample = sample;
            result.blocks = m_blocks;
            result.guestInstructions =
                m_blocks * kInstructionsPerBlock;
            result.guestLoads = m_blocks * kLoadsPerBlock;
            result.guestStores = m_blocks * kStoresPerBlock;
            result.hostNanoseconds = hostNanoseconds;
            result.registerHash = hashRegisters(context);
            result.memoryHash = hashMemory(
                rdram + kMappedPhysicalBase,
                kWorkingSetBytes);
            result.identityMemoryHash = hashMemory(
                rdram + kIdentityPhysicalBase,
                kWorkingSetBytes);
            result.initialIdentityMemoryHash =
                initialIdentityMemoryHash;

            Fnv1a64 work;
            work.add(result.registerHash);
            work.add(result.memoryHash);
            work.add(context.insn_count);
            work.add(context.cop0_random);
            result.workHash = work.value();

            result.insnCount = context.insn_count;
            result.random = context.cop0_random;
            result.bpc = context.cop0_bpc;
            result.pccr = context.cop0_perf;
            result.pcr0 = context.cop0_pcr0;
            result.pcr1 = context.cop0_pcr1;
            result.pc = context.pc;
            result.status = context.cop0_status;
            result.cause = context.cop0_cause;
            result.errorEpc = context.cop0_errorepc;
            result.issuePending =
                context.ee_performance_issue_pending;
            result.overflowDelivered = overflowDelivered;
            result.counterHash = hashCounters(context);
            return result;
        }

    private:
        static uint64_t hashMemory(
            const uint8_t *data,
            size_t size)
        {
            Fnv1a64 hash;
            hash.addBytes(data, size);
            return hash.value();
        }

        static uint64_t hashRegisters(
            const R5900Context &context)
        {
            Fnv1a64 hash;
            for (uint32_t reg = 1u; reg <= 8u; ++reg)
            {
                hash.addBytes(&context.r[reg], sizeof(context.r[reg]));
            }
            hash.add(context.f[1]);
            hash.add(context.f[2]);
            hash.add(context.vi[1]);
            hash.add(context.vi[2]);
            hash.add(context.hi);
            hash.add(context.lo);
            hash.add(context.hi1);
            hash.add(context.lo1);
            return hash.value();
        }

        static uint64_t hashCounters(
            const R5900Context &context)
        {
            Fnv1a64 hash;
            hash.add(context.insn_count);
            hash.add(context.cop0_random);
            hash.add(context.cop0_wired);
            hash.add(context.cop0_bpc);
            hash.add(context.cop0_perf);
            hash.add(context.cop0_pcr0);
            hash.add(context.cop0_pcr1);
            hash.add(context.pc);
            hash.add(context.cop0_status);
            hash.add(context.cop0_cause);
            hash.add(context.cop0_errorepc);
            hash.add(context.cop0_random_instruction_pending);
            hash.add(context.ee_performance_issue_pending);
            return hash.value();
        }

        void initializeMemory()
        {
            for (uint32_t word = 0u;
                 word < kWorkingSetWords;
                 ++word)
            {
                const uint32_t mappedValue =
                    0x6d2b79f5u ^
                    (word * 0x9e3779b9u);
                const uint32_t identityValue =
                    0xc001d00du ^
                    (word * 0x85ebca6bu);
                std::memcpy(
                    rdram + kMappedPhysicalBase +
                        word * sizeof(uint32_t),
                    &mappedValue, sizeof(mappedValue));
                std::memcpy(
                    rdram + kIdentityPhysicalBase +
                        word * sizeof(uint32_t),
                    &identityValue, sizeof(identityValue));
            }
        }

        void initializeContext()
        {
            context = R5900Context{};
            context.cop0_status = 0u;
            context.cop0_entryhi = kAsid;
            context.cop0_config |= kConfigBpe | kConfigDie;
            context.cop0_wired = 0u;
            context.cop0_random = 47u;
            SET_GPR_U32(&context, 2u, 0x13579bdfu);
            SET_GPR_U32(&context, 3u, 0x2468ace0u);
            SET_GPR_U32(&context, 4u, 0u);
            SET_GPR_U32(&context, 6u, 0x10203040u);
            SET_GPR_U32(&context, 7u, 0x89abcdefu);
            context.f[1] = 1.25f;
            context.f[2] = 0.25f;
            context.vi[1] = 0x1234u;
            context.vi[2] = 0x31u;
        }

        void configurePerformance(
            uint32_t event0,
            uint32_t event1,
            uint32_t pcr0 = 0u,
            uint32_t pcr1 = 0u)
        {
            runtime.writeCop0Performance(
                rdram, &context, 1u, pcr0);
            runtime.writeCop0Performance(
                rdram, &context, 3u, pcr1);
            runtime.writeCop0Performance(
                rdram, &context, 0u,
                performanceControl(event0, event1));
        }

        void configureFeature()
        {
            switch (m_case.feature)
            {
            case Feature::Inactive:
                return;
            case Feature::InstructionBreakpoint:
                context.cop0_bpc =
                    kBpcIae | kBpcIke | kBpcBed;
                context.cop0_iab = kCodeBase;
                context.cop0_iabm = 0xfffff000u;
                return;
            case Feature::DataAddressBreakpoint:
                context.cop0_bpc =
                    kBpcDre | kBpcDwe |
                    kBpcDke | kBpcBed;
                context.cop0_dab = kVirtualBase;
                context.cop0_dabm = 0xfffff000u;
                return;
            case Feature::DataValueBreakpoint:
                context.cop0_bpc =
                    kBpcDre | kBpcDwe | kBpcDve |
                    kBpcDke | kBpcBed;
                context.cop0_dab = kVirtualBase;
                context.cop0_dabm = 0xfffff000u;
                context.cop0_dvb = 0u;
                context.cop0_dvbm = 0u;
                return;
            case Feature::PccrCycles:
                configurePerformance(1u, 1u);
                return;
            case Feature::PccrIssue:
                configurePerformance(2u, 2u);
                return;
            case Feature::PccrBranch:
                configurePerformance(3u, 3u);
                return;
            case Feature::PccrLowOrderBranch:
                configurePerformance(3u, 0u);
                return;
            case Feature::PccrCompletion:
                configurePerformance(12u, 13u);
                return;
            case Feature::PccrCoprocessor:
                configurePerformance(14u, 14u);
                return;
            case Feature::PccrMemory:
                configurePerformance(15u, 15u);
                return;
            case Feature::PccrOverflowLastCompletion:
            {
                const uint64_t instructions =
                    m_blocks * kInstructionsPerBlock;
                const uint32_t initial =
                    0x7fffffffu -
                    static_cast<uint32_t>(instructions - 1u);
                configurePerformance(12u, 16u, initial, 0u);
                return;
            }
            }
        }

        const BenchmarkCase &m_case;
        uint64_t m_blocks = 0u;

    public:
        PS2Runtime runtime;
        R5900Context context{};
        uint8_t *rdram = nullptr;
        uint64_t initialIdentityMemoryHash = 0u;
        bool overflowDelivered = false;
    };

    uint32_t expectedRandom(uint64_t guestInstructions)
    {
        return 47u -
               static_cast<uint32_t>(guestInstructions % 48u);
    }

    uint32_t expectedBreakpointSticky(Feature feature)
    {
        switch (feature)
        {
        case Feature::InstructionBreakpoint:
            return kBpcIab;
        case Feature::DataAddressBreakpoint:
        case Feature::DataValueBreakpoint:
            return kBpcDrb | kBpcDwb;
        default:
            return 0u;
        }
    }

    void verifyPerformanceCounters(const Measurement &measurement)
    {
        const Feature feature = measurement.benchmarkCase->feature;
        const uint64_t blocks = measurement.blocks;
        const uint64_t instructions = measurement.guestInstructions;
        uint64_t expected0 = 0u;
        uint64_t expected1 = 0u;

        switch (feature)
        {
        case Feature::PccrCycles:
            expected0 = instructions;
            expected1 = instructions;
            break;
        case Feature::PccrIssue:
            expected0 = 0u;
            expected1 = blocks * 5u;
            break;
        case Feature::PccrBranch:
            expected0 = blocks * 2u;
            expected1 = (blocks + 1u) / 2u;
            break;
        case Feature::PccrLowOrderBranch:
            expected0 = blocks * 2u;
            expected1 = blocks * 2u;
            break;
        case Feature::PccrCompletion:
            expected0 = instructions;
            expected1 = blocks * 8u;
            break;
        case Feature::PccrCoprocessor:
            expected0 = blocks;
            expected1 = blocks;
            break;
        case Feature::PccrMemory:
            expected0 = blocks;
            expected1 = blocks;
            break;
        case Feature::PccrOverflowLastCompletion:
            expected0 = 0x80000000u;
            expected1 = 0u;
            break;
        default:
            break;
        }

        if (measurement.pcr0 != static_cast<uint32_t>(expected0) ||
            measurement.pcr1 != static_cast<uint32_t>(expected1))
        {
            throw std::runtime_error(
                "PCCR result does not match fixed-work expectation");
        }
    }

    void verifyMeasurement(const Measurement &measurement)
    {
        if (measurement.insnCount !=
                measurement.guestInstructions ||
            measurement.random !=
                expectedRandom(measurement.guestInstructions))
        {
            throw std::runtime_error(
                "instruction count or Random result is incorrect");
        }
        if (measurement.identityMemoryHash !=
            measurement.initialIdentityMemoryHash)
        {
            throw std::runtime_error(
                "non-identity mapped work modified identity RDRAM");
        }

        const uint32_t actualSticky =
            measurement.bpc & kBreakpointStickyMask;
        if (actualSticky != expectedBreakpointSticky(
                                measurement.benchmarkCase->feature))
        {
            throw std::runtime_error(
                "hardware-breakpoint sticky result is incorrect");
        }

        const bool expectsOverflow =
            measurement.benchmarkCase->feature ==
            Feature::PccrOverflowLastCompletion;
        if (measurement.overflowDelivered != expectsOverflow)
        {
            throw std::runtime_error(
                "performance-overflow delivery result is incorrect");
        }
        if (expectsOverflow &&
            (measurement.pc != 0x80000080u ||
             (measurement.status & kStatusErl) == 0u ||
             (measurement.cause & kCauseExc2Mask) !=
                 kCauseExc2Performance))
        {
            throw std::runtime_error(
                "performance overflow published incorrect exception state");
        }
        if (!expectsOverflow && measurement.issuePending)
        {
            throw std::runtime_error(
                "PCCR issue state remained pending at the benchmark boundary");
        }
        verifyPerformanceCounters(measurement);
    }

    Measurement runMeasurement(
        const BenchmarkCase &benchmarkCase,
        uint64_t blocks,
        uint32_t sample)
    {
        Fixture fixture(benchmarkCase, blocks);
        const auto started = std::chrono::steady_clock::now();
        fixture.run();
        const auto stopped = std::chrono::steady_clock::now();
        fixture.finish();
        const uint64_t hostNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    stopped - started)
                    .count());
        Measurement result =
            fixture.measurement(sample, hostNanoseconds);
        verifyMeasurement(result);
        return result;
    }

    void runWarmup(
        const BenchmarkCase &benchmarkCase,
        uint64_t blocks)
    {
        if (blocks == 0u)
            return;
        Fixture fixture(benchmarkCase, blocks);
        fixture.run();
        fixture.finish();
        verifyMeasurement(fixture.measurement(0u, 0u));
    }

    void printHex(uint64_t value)
    {
        std::cout << "0x" << std::hex << value << std::dec;
    }

    void printMeasurement(const Measurement &measurement)
    {
        const auto &benchmarkCase = *measurement.benchmarkCase;
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"measurement\""
            << ",\"case\":\"" << benchmarkCase.name << "\""
            << ",\"entry\":\"" << entryName(benchmarkCase.entry) << "\""
            << ",\"memory_path\":\""
            << memoryPathName(benchmarkCase.memory) << "\""
            << ",\"sample\":" << measurement.sample
            << ",\"blocks\":" << measurement.blocks
            << ",\"guest_instructions\":"
            << measurement.guestInstructions
            << ",\"guest_loads\":" << measurement.guestLoads
            << ",\"guest_stores\":" << measurement.guestStores
            << ",\"host_time_ns\":" << measurement.hostNanoseconds
            << ",\"host_ns_per_guest_instruction\":"
            << (measurement.guestInstructions == 0u
                    ? 0.0
                    : static_cast<double>(measurement.hostNanoseconds) /
                          static_cast<double>(measurement.guestInstructions))
            << ",\"register_hash\":\"";
        printHex(measurement.registerHash);
        std::cout << "\",\"memory_hash\":\"";
        printHex(measurement.memoryHash);
        std::cout << "\",\"identity_memory_hash\":\"";
        printHex(measurement.identityMemoryHash);
        std::cout << "\",\"work_hash\":\"";
        printHex(measurement.workHash);
        std::cout << "\",\"counter_hash\":\"";
        printHex(measurement.counterHash);
        std::cout
            << "\",\"insn_count\":" << measurement.insnCount
            << ",\"random\":" << measurement.random
            << ",\"bpc\":" << measurement.bpc
            << ",\"pccr\":" << measurement.pccr
            << ",\"pcr0\":" << measurement.pcr0
            << ",\"pcr1\":" << measurement.pcr1
            << ",\"pc\":" << measurement.pc
            << ",\"status\":" << measurement.status
            << ",\"cause\":" << measurement.cause
            << ",\"error_epc\":" << measurement.errorEpc
            << ",\"issue_pending\":"
            << (measurement.issuePending ? "true" : "false")
            << ",\"overflow_delivered\":"
            << (measurement.overflowDelivered ? "true" : "false")
            << "}\n";
    }

    uint64_t median(std::vector<uint64_t> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    }

    void printSummary(
        const BenchmarkCase &benchmarkCase,
        uint64_t blocks,
        const std::vector<uint64_t> &hostTimes,
        uint64_t workHash,
        uint64_t counterHash)
    {
        const uint64_t guestInstructions =
            blocks * kInstructionsPerBlock;
        const uint64_t medianTime = median(hostTimes);
        const auto [minimum, maximum] =
            std::minmax_element(hostTimes.begin(), hostTimes.end());
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"summary\""
            << ",\"case\":\"" << benchmarkCase.name << "\""
            << ",\"entry\":\"" << entryName(benchmarkCase.entry) << "\""
            << ",\"memory_path\":\""
            << memoryPathName(benchmarkCase.memory) << "\""
            << ",\"samples\":" << hostTimes.size()
            << ",\"blocks\":" << blocks
            << ",\"guest_instructions\":" << guestInstructions
            << ",\"median_host_time_ns\":" << medianTime
            << ",\"minimum_host_time_ns\":" << *minimum
            << ",\"maximum_host_time_ns\":" << *maximum
            << ",\"median_host_ns_per_guest_instruction\":"
            << static_cast<double>(medianTime) /
                   static_cast<double>(guestInstructions)
            << ",\"work_hash\":\"";
        printHex(workHash);
        std::cout << "\",\"counter_hash\":\"";
        printHex(counterHash);
        std::cout << "\"}\n";
    }
}

int main(int argc, char **argv)
{
    Configuration configuration{};
    if (!parseArguments(argc, argv, configuration))
    {
        std::cerr
            << "usage: ee_generated_code_benchmark "
               "[--blocks N] [--warmup-blocks N] [--samples N] "
               "[--case NAME]\n";
        return 2;
    }

    try
    {
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"configuration\""
            << ",\"blocks\":" << configuration.blocks
            << ",\"warmup_blocks\":" << configuration.warmupBlocks
            << ",\"samples\":" << configuration.samples
            << ",\"case_filter\":\""
            << configuration.caseFilter << "\""
            << ",\"instructions_per_block\":"
            << kInstructionsPerBlock
            << ",\"loads_per_block\":" << kLoadsPerBlock
            << ",\"stores_per_block\":" << kStoresPerBlock
            << ",\"virtual_base\":" << kVirtualBase
            << ",\"mapped_physical_base\":" << kMappedPhysicalBase
            << ",\"direct_kseg_base\":" << kDirectBase
            << ",\"non_identity_mapping\":"
            << (kVirtualBase != kMappedPhysicalBase
                    ? "true" : "false")
            << "}\n";

        std::array<std::vector<uint64_t>, kCases.size()> hostTimes;
        std::array<uint64_t, kCases.size()> workHashes{};
        std::array<uint64_t, kCases.size()> counterHashes{};
        uint64_t commonWorkHash = 0u;
        std::vector<size_t> selectedCases;

        for (size_t index = 0u; index < kCases.size(); ++index)
        {
            if (configuration.caseFilter.empty() ||
                kCases[index].name == configuration.caseFilter)
            {
                selectedCases.push_back(index);
            }
        }
        if (selectedCases.empty())
        {
            throw std::runtime_error("unknown benchmark case");
        }

        for (size_t index : selectedCases)
        {
            runWarmup(
                kCases[index],
                configuration.warmupBlocks);
        }

        for (uint32_t sample = 0u;
             sample < configuration.samples;
             ++sample)
        {
            // Rotate the first case so thermal/order effects do not always
            // favor the same entry while retaining deterministic JSON order
            // within each sample.
            for (size_t offset = 0u;
                 offset < selectedCases.size();
                 ++offset)
            {
                const size_t index =
                    selectedCases[
                        (offset + sample) % selectedCases.size()];
                const Measurement measurement = runMeasurement(
                    kCases[index], configuration.blocks, sample);
                printMeasurement(measurement);

                if (commonWorkHash == 0u)
                {
                    commonWorkHash = measurement.workHash;
                }
                else if (measurement.workHash != commonWorkHash)
                {
                    throw std::runtime_error(
                        "observation or address mode changed fixed guest work");
                }

                if (hostTimes[index].empty())
                {
                    workHashes[index] = measurement.workHash;
                    counterHashes[index] = measurement.counterHash;
                }
                else if (workHashes[index] != measurement.workHash ||
                         counterHashes[index] != measurement.counterHash)
                {
                    throw std::runtime_error(
                        "fixed-work hashes changed between samples");
                }
                hostTimes[index].push_back(
                    measurement.hostNanoseconds);
            }
        }

        for (size_t index : selectedCases)
        {
            printSummary(
                kCases[index], configuration.blocks,
                hostTimes[index], workHashes[index],
                counterHashes[index]);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "ee_generated_code_benchmark: "
            << error.what() << '\n';
        return 1;
    }
    return 0;
}
