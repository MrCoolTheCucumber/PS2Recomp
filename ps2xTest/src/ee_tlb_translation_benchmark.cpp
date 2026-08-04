#include "runtime/ps2_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
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
    constexpr uint32_t kOutputSchemaVersion = 1u;
    constexpr uint32_t kVirtualBase = 0x00400000u;
    constexpr uint32_t kIdentityPhysicalBase = kVirtualBase;
    constexpr uint32_t kMappedPhysicalBase = 0x00800000u;
    constexpr uint32_t kDirectBase =
        0x80000000u | kMappedPhysicalBase;
    constexpr uint32_t kWorkingSetBytes = 0x1000u;
    constexpr uint32_t kAccessesPerBlock = 8u;
    constexpr uint32_t kAsid = 0x46u;

#if defined(_MSC_VER)
#define PS2_TLB_BENCHMARK_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_TLB_BENCHMARK_NOINLINE __attribute__((noinline, noipa))
#else
#define PS2_TLB_BENCHMARK_NOINLINE __attribute__((noinline))
#endif

    enum class TranslationPath : uint8_t
    {
        MappedCached,
        MappedForcedUncached,
        DirectKseg,
    };

    struct BenchmarkCase
    {
        std::string_view name;
        TranslationPath path;
    };

    constexpr std::array<BenchmarkCase, 3u> kCases{{
        {"mapped_cached", TranslationPath::MappedCached},
        {"mapped_forced_uncached",
         TranslationPath::MappedForcedUncached},
        {"direct_kseg", TranslationPath::DirectKseg},
    }};

    struct Configuration
    {
        uint64_t blocks = 1'000'000u;
        uint64_t warmupBlocks = 20'000u;
        uint32_t samples = 5u;
        std::string_view caseFilter;
    };

    struct WorkResult
    {
        uint32_t state = 0u;
        uint64_t checksum = 0u;
    };

    struct Measurement
    {
        const BenchmarkCase *benchmarkCase = nullptr;
        uint32_t sample = 0u;
        uint64_t blocks = 0u;
        uint64_t warmupBlocks = 0u;
        uint64_t guestAccesses = 0u;
        uint64_t hostNanoseconds = 0u;
        uint64_t memoryHash = 0u;
        uint64_t identityMemoryHash = 0u;
        uint64_t initialIdentityMemoryHash = 0u;
        uint64_t workHash = 0u;
        uint64_t diagnosticHits = 0u;
        uint64_t diagnosticMisses = 0u;
        uint32_t finalState = 0u;
        uint64_t checksum = 0u;
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

    constexpr uint32_t makeEntryLo(uint32_t pfn)
    {
        return ((pfn & 0x000fffffu) << 6u) |
               (2u << 3u) |
               0x4u |
               0x2u;
    }

    bool parseUnsigned(std::string_view text, uint64_t &value)
    {
        if (text.empty())
        {
            return false;
        }
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
            {
                return false;
            }
            const std::string_view option = argv[index];
            if (option == "--case")
            {
                configuration.caseFilter = argv[index + 1];
                continue;
            }

            uint64_t value = 0u;
            if (!parseUnsigned(argv[index + 1], value))
            {
                return false;
            }
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
                configuration.samples =
                    static_cast<uint32_t>(value);
            }
            else
            {
                return false;
            }
        }

        constexpr uint64_t maximumBlocks =
            std::numeric_limits<uint64_t>::max() /
            (2u * kAccessesPerBlock);
        return configuration.blocks != 0u &&
               configuration.blocks <= maximumBlocks &&
               configuration.warmupBlocks <= maximumBlocks;
    }

    void initializeWorkingSet(PS2Memory &memory)
    {
        uint8_t *const rdram = memory.getRDRAM();
        for (uint32_t offset = 0u;
             offset < kWorkingSetBytes;
             offset += sizeof(uint32_t))
        {
            const uint32_t word =
                0x9e3779b9u * (offset / sizeof(uint32_t) + 1u) ^
                std::rotl(offset, 13);
            std::memcpy(
                rdram + kMappedPhysicalBase + offset,
                &word,
                sizeof(word));
        }
        std::memset(
            rdram + kIdentityPhysicalBase,
            0xa5,
            kWorkingSetBytes);
    }

    uint64_t hashBytes(const void *data, size_t size)
    {
        Fnv1a64 hash;
        hash.addBytes(data, size);
        return hash.value();
    }

    PS2_TLB_BENCHMARK_NOINLINE WorkResult runWork(
        PS2Memory &memory,
        uint32_t guestBase,
        EeAddressTranslationContext translation,
        uint64_t blocks,
        uint32_t initialState)
    {
        uint32_t state = initialState;
        uint64_t checksum = 0xcbf29ce484222325ull;
        for (uint64_t block = 0u; block < blocks; ++block)
        {
            const uint32_t blockMix =
                static_cast<uint32_t>(block) * 0x85ebca6bu;
            for (uint32_t access = 0u;
                 access < kAccessesPerBlock;
                 ++access)
            {
                const uint32_t wordIndex =
                    (static_cast<uint32_t>(block) *
                         kAccessesPerBlock +
                     access * 37u) &
                    ((kWorkingSetBytes / sizeof(uint32_t)) - 1u);
                const uint32_t address =
                    guestBase + wordIndex * sizeof(uint32_t);
                const uint32_t loaded =
                    memory.read32(address, translation);
                state = std::rotl(
                    state ^ loaded ^ blockMix,
                    (access + 5u) & 31u);
                const uint32_t stored =
                    loaded + state +
                    access * 0x27d4eb2du;
                memory.write32(
                    address,
                    stored,
                    0u,
                    translation);
                checksum ^= stored;
                checksum *= 1099511628211ull;
            }
        }
        return {state, checksum};
    }

    Measurement measure(
        const BenchmarkCase &benchmarkCase,
        const Configuration &configuration,
        uint32_t sample)
    {
        PS2Memory memory;
        if (!memory.initialize())
        {
            throw std::runtime_error(
                "PS2Memory initialize failed");
        }
        if (!memory.tlbWrite(
                7u,
                EeTlbEntry{
                    0u,
                    kVirtualBase | kAsid,
                    makeEntryLo(kMappedPhysicalBase >> 12u),
                    makeEntryLo(
                        (kMappedPhysicalBase >> 12u) + 1u),
                }))
        {
            throw std::runtime_error("TLB mapping install failed");
        }
        initializeWorkingSet(memory);
        const uint64_t initialIdentityMemoryHash = hashBytes(
            memory.getRDRAM() + kIdentityPhysicalBase,
            kWorkingSetBytes);

        const bool mapped =
            benchmarkCase.path != TranslationPath::DirectKseg;
        const uint32_t guestBase = mapped
            ? kVirtualBase
            : kDirectBase;
        const auto translation = mapped
            ? EeAddressTranslationContext::fromCop0Status(
                  0x10u,
                  static_cast<uint8_t>(kAsid))
            : EeAddressTranslationContext::unchecked();
        memory.setTlbTranslationCacheEnabled(
            benchmarkCase.path !=
            TranslationPath::MappedForcedUncached);

        constexpr uint32_t initialState = 0x13579bdfu;
        (void)runWork(
            memory,
            guestBase,
            translation,
            configuration.warmupBlocks,
            initialState);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto start = std::chrono::steady_clock::now();
        const WorkResult result = runWork(
            memory,
            guestBase,
            translation,
            configuration.blocks,
            initialState);
        const auto end = std::chrono::steady_clock::now();
        std::atomic_signal_fence(std::memory_order_seq_cst);

        const uint64_t memoryHash = hashBytes(
            memory.getRDRAM() + kMappedPhysicalBase,
            kWorkingSetBytes);
        const uint64_t identityMemoryHash = hashBytes(
            memory.getRDRAM() + kIdentityPhysicalBase,
            kWorkingSetBytes);
        Fnv1a64 workHash;
        workHash.add(result.state);
        workHash.add(result.checksum);
        workHash.add(memoryHash);
        workHash.add(identityMemoryHash);

        uint64_t diagnosticHits = 0u;
        uint64_t diagnosticMisses = 0u;
        if (benchmarkCase.path == TranslationPath::MappedCached)
        {
            memory.setTlbTranslationCacheDiagnosticsEnabled(true);
            memory.resetTlbTranslationCacheStats();
            for (uint32_t offset = 0u;
                 offset < kWorkingSetBytes;
                 offset += sizeof(uint32_t))
            {
                (void)memory.resolveAddress(
                    kVirtualBase + offset,
                    translation,
                    false,
                    sizeof(uint32_t));
            }
            const EeTlbTranslationCacheStats stats =
                memory.tlbTranslationCacheStats();
            diagnosticHits = stats.hits;
            diagnosticMisses = stats.misses;
        }

        return {
            &benchmarkCase,
            sample,
            configuration.blocks,
            configuration.warmupBlocks,
            configuration.blocks *
                (2u * kAccessesPerBlock),
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(end - start)
                    .count()),
            memoryHash,
            identityMemoryHash,
            initialIdentityMemoryHash,
            workHash.value(),
            diagnosticHits,
            diagnosticMisses,
            result.state,
            result.checksum,
        };
    }

    void printMeasurement(const Measurement &measurement)
    {
        const double nsPerAccess =
            static_cast<double>(measurement.hostNanoseconds) /
            static_cast<double>(measurement.guestAccesses);
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"case\":\"" << measurement.benchmarkCase->name
            << "\",\"sample\":" << measurement.sample
            << ",\"blocks\":" << measurement.blocks
            << ",\"warmup_blocks\":"
            << measurement.warmupBlocks
            << ",\"guest_accesses\":"
            << measurement.guestAccesses
            << ",\"host_nanoseconds\":"
            << measurement.hostNanoseconds
            << ",\"ns_per_access\":"
            << std::fixed << std::setprecision(6)
            << nsPerAccess
            << ",\"memory_hash\":\"0x" << std::hex
            << std::setw(16) << std::setfill('0')
            << measurement.memoryHash
            << "\",\"identity_memory_hash\":\"0x"
            << std::setw(16) << measurement.identityMemoryHash
            << "\",\"initial_identity_memory_hash\":\"0x"
            << std::setw(16)
            << measurement.initialIdentityMemoryHash
            << "\",\"work_hash\":\"0x"
            << std::setw(16) << measurement.workHash
            << "\",\"final_state\":\"0x"
            << std::setw(8) << measurement.finalState
            << "\",\"checksum\":\"0x"
            << std::setw(16) << measurement.checksum
            << '"' << std::dec << std::setfill(' ')
            << ",\"diagnostic_hits\":"
            << measurement.diagnosticHits
            << ",\"diagnostic_misses\":"
            << measurement.diagnosticMisses
            << ",\"cache_sets\":"
            << PS2Memory::kTlbTranslationCacheSetCount
            << ",\"cache_ways\":"
            << PS2Memory::kTlbTranslationCacheWayCount
            << ",\"cache_subpage_bytes\":"
            << PS2Memory::kTlbTranslationCachePageSize
            << ",\"diagnostics_timed\":false"
            << "}\n";
    }
}

int main(int argc, char **argv)
{
    Configuration configuration;
    if (!parseArguments(argc, argv, configuration))
    {
        std::cerr
            << "usage: ee_tlb_translation_benchmark "
               "[--blocks N] [--warmup-blocks N] "
               "[--samples N] [--case NAME]\n";
        return 2;
    }

    std::vector<const BenchmarkCase *> selectedCases;
    for (const BenchmarkCase &benchmarkCase : kCases)
    {
        if (configuration.caseFilter.empty() ||
            benchmarkCase.name == configuration.caseFilter)
        {
            selectedCases.push_back(&benchmarkCase);
        }
    }
    if (selectedCases.empty())
    {
        std::cerr << "unknown or empty benchmark case\n";
        return 2;
    }

    try
    {
        for (uint32_t sample = 0u;
             sample < configuration.samples;
             ++sample)
        {
            std::vector<Measurement> measurements;
            measurements.reserve(selectedCases.size());
            for (size_t index = 0u;
                 index < selectedCases.size();
                 ++index)
            {
                const size_t rotatedIndex =
                    (index + sample) % selectedCases.size();
                measurements.push_back(measure(
                    *selectedCases[rotatedIndex],
                    configuration,
                    sample));
            }

            if (measurements.size() > 1u)
            {
                const uint64_t expectedWorkHash =
                    measurements.front().workHash;
                const uint64_t expectedIdentityHash =
                    measurements.front().initialIdentityMemoryHash;
                for (const Measurement &measurement : measurements)
                {
                    if (measurement.workHash != expectedWorkHash)
                    {
                        throw std::runtime_error(
                            "equal-work hash mismatch");
                    }
                    if (measurement.identityMemoryHash !=
                            expectedIdentityHash ||
                        measurement.initialIdentityMemoryHash !=
                            expectedIdentityHash)
                    {
                        throw std::runtime_error(
                            "non-identity mapping touched identity RDRAM");
                    }
                }
            }

            for (const Measurement &measurement : measurements)
            {
                printMeasurement(measurement);
            }
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
