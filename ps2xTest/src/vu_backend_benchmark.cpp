#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_recompiler.h"
#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    std::atomic<bool> g_trackAllocations{false};
    std::atomic<uint64_t> g_allocationCount{0u};
    std::atomic<uint64_t> g_allocationBytes{0u};

    void recordAllocation(std::size_t size) noexcept
    {
        if (!g_trackAllocations.load(
                std::memory_order_relaxed))
        {
            return;
        }
        g_allocationCount.fetch_add(
            1u, std::memory_order_relaxed);
        g_allocationBytes.fetch_add(
            static_cast<uint64_t>(size),
            std::memory_order_relaxed);
    }

    void *allocate(std::size_t size)
    {
        size = std::max<std::size_t>(size, 1u);
        void *const result = std::malloc(size);
        if (!result)
            throw std::bad_alloc();
        recordAllocation(size);
        return result;
    }

    void *allocateAligned(
        std::size_t size, std::size_t alignment)
    {
        size = std::max<std::size_t>(size, 1u);
        void *result = nullptr;
#if defined(_WIN32)
        result = _aligned_malloc(size, alignment);
        if (!result)
            throw std::bad_alloc();
#else
        if (posix_memalign(
                &result, alignment, size) != 0)
        {
            throw std::bad_alloc();
        }
#endif
        recordAllocation(size);
        return result;
    }

    void freeAligned(void *pointer) noexcept
    {
#if defined(_WIN32)
        _aligned_free(pointer);
#else
        std::free(pointer);
#endif
    }
}

void *operator new(std::size_t size)
{
    return allocate(size);
}

void *operator new[](std::size_t size)
{
    return allocate(size);
}

void *operator new(
    std::size_t size, const std::nothrow_t &) noexcept
{
    try
    {
        return allocate(size);
    }
    catch (...)
    {
        return nullptr;
    }
}

void *operator new[](
    std::size_t size, const std::nothrow_t &) noexcept
{
    try
    {
        return allocate(size);
    }
    catch (...)
    {
        return nullptr;
    }
}

void *operator new(
    std::size_t size, std::align_val_t alignment)
{
    return allocateAligned(
        size, static_cast<std::size_t>(alignment));
}

void *operator new[](
    std::size_t size, std::align_val_t alignment)
{
    return allocateAligned(
        size, static_cast<std::size_t>(alignment));
}

void *operator new(
    std::size_t size, std::align_val_t alignment,
    const std::nothrow_t &) noexcept
{
    try
    {
        return allocateAligned(
            size, static_cast<std::size_t>(alignment));
    }
    catch (...)
    {
        return nullptr;
    }
}

void *operator new[](
    std::size_t size, std::align_val_t alignment,
    const std::nothrow_t &) noexcept
{
    try
    {
        return allocateAligned(
            size, static_cast<std::size_t>(alignment));
    }
    catch (...)
    {
        return nullptr;
    }
}

void operator delete(void *pointer) noexcept
{
    std::free(pointer);
}

void operator delete[](void *pointer) noexcept
{
    std::free(pointer);
}

void operator delete(
    void *pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete[](
    void *pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete(
    void *pointer, std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete[](
    void *pointer, std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete(
    void *pointer, std::size_t,
    std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete[](
    void *pointer, std::size_t,
    std::align_val_t) noexcept
{
    freeAligned(pointer);
}

namespace
{
    using Clock = std::chrono::steady_clock;

    struct FixturePaths
    {
        std::filesystem::path data;
        std::filesystem::path micro;
        std::filesystem::path state;
    };

    struct Configuration
    {
        std::filesystem::path fixtureDirectory;
        uint32_t maximumPairs = 0u;
        uint32_t iterations = 1000u;
        uint32_t warmupIterations = 10u;
        uint32_t samples = 7u;
    };

    bool parseUnsigned(
        std::string_view text, uint64_t &value)
    {
        if (text.empty())
            return false;
        const char *const begin = text.data();
        const char *const end = begin + text.size();
        const auto result =
            std::from_chars(begin, end, value);
        return
            result.ec == std::errc{} &&
            result.ptr == end;
    }

    bool parseArguments(
        int argc, char **argv,
        Configuration &configuration)
    {
        if (argc < 3)
            return false;
        configuration.fixtureDirectory = argv[1];
        uint64_t maximumPairs = 0u;
        if (!parseUnsigned(argv[2], maximumPairs) ||
            maximumPairs == 0u ||
            maximumPairs > UINT32_MAX)
        {
            return false;
        }
        configuration.maximumPairs =
            static_cast<uint32_t>(maximumPairs);

        for (int index = 3; index < argc; index += 2)
        {
            if (index + 1 >= argc)
                return false;
            uint64_t value = 0u;
            if (!parseUnsigned(argv[index + 1], value) ||
                value == 0u || value > UINT32_MAX)
            {
                return false;
            }
            const std::string_view option(argv[index]);
            if (option == "--iterations")
            {
                configuration.iterations =
                    static_cast<uint32_t>(value);
            }
            else if (option == "--warmup")
            {
                configuration.warmupIterations =
                    static_cast<uint32_t>(value);
            }
            else if (option == "--samples")
            {
                configuration.samples =
                    static_cast<uint32_t>(value);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    bool resolveFixture(
        const std::filesystem::path &directory,
        FixturePaths &paths)
    {
        const FixturePaths standard{
            directory / "vu1-data.bin",
            directory / "vu1-code.bin",
            directory / "vu1-replay-state.bin",
        };
        if (std::filesystem::is_regular_file(
                standard.data) &&
            std::filesystem::is_regular_file(
                standard.micro) &&
            std::filesystem::is_regular_file(
                standard.state))
        {
            paths = standard;
            return true;
        }

        std::vector<FixturePaths> candidates;
        std::error_code error;
        for (const auto &entry :
             std::filesystem::directory_iterator(
                 directory, error))
        {
            if (error || !entry.is_regular_file())
                continue;
            const std::string filename =
                entry.path().filename().string();
            constexpr std::string_view suffix =
                "-state.bin";
            if (!filename.ends_with(suffix))
                continue;
            const std::string prefix = filename.substr(
                0u, filename.size() - suffix.size());
            FixturePaths candidate{
                directory / (prefix + "-data.bin"),
                directory / (prefix + "-micro.bin"),
                entry.path(),
            };
            if (!std::filesystem::is_regular_file(
                    candidate.micro))
            {
                candidate.micro =
                    directory / (prefix + "-code.bin");
            }
            if (std::filesystem::is_regular_file(
                    candidate.data) &&
                std::filesystem::is_regular_file(
                    candidate.micro))
            {
                candidates.push_back(
                    std::move(candidate));
            }
        }
        if (candidates.size() != 1u)
            return false;
        paths = std::move(candidates.front());
        return true;
    }

    template <typename Container>
    bool readExact(
        const std::filesystem::path &path,
        Container &destination)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            return false;
        input.read(
            reinterpret_cast<char *>(
                destination.data()),
            static_cast<std::streamsize>(
                destination.size() *
                sizeof(destination[0])));
        return
            input.gcount() ==
            static_cast<std::streamsize>(
                destination.size() *
                sizeof(destination[0]));
    }

    class HashingSink final : public IVuSideEffectSink
    {
    public:
        void submitPath1Packet(
            const uint8_t *data,
            uint32_t sizeBytes) override
        {
            ++packets;
            bytes += sizeBytes;
            mix(sizeBytes);
            for (uint32_t index = 0u;
                 index < sizeBytes; ++index)
            {
                hash ^= data[index];
                hash *= 1099511628211ull;
            }
        }

        void reset()
        {
            packets = 0u;
            bytes = 0u;
            hash = 1469598103934665603ull;
        }

        bool operator==(
            const HashingSink &other) const
        {
            return
                packets == other.packets &&
                bytes == other.bytes &&
                hash == other.hash;
        }

        uint64_t packets = 0u;
        uint64_t bytes = 0u;
        uint64_t hash = 1469598103934665603ull;

    private:
        void mix(uint32_t value)
        {
            for (uint32_t shift = 0u;
                 shift < 32u; shift += 8u)
            {
                hash ^=
                    static_cast<uint8_t>(
                        value >> shift);
                hash *= 1099511628211ull;
            }
        }
    };

    struct Invocation
    {
        VuExecutionState state{};
        std::array<uint8_t, PS2_VU1_DATA_SIZE> data{};
        HashingSink effects;
    };

    struct RunSummary
    {
        uint32_t pairs = 0u;
        VuExitReason reason = VuExitReason::Inactive;
        bool valid = true;
    };

    RunSummary runBackend(
        IVuExecutionBackend &backend,
        VuExecutionContext &context,
        uint32_t maximumPairs)
    {
        RunSummary summary;
        while (context.state.active &&
               summary.pairs < maximumPairs)
        {
            const uint32_t remaining =
                maximumPairs - summary.pairs;
            const VuRunResult result =
                backend.run(context, remaining);
            if (result.executedCycles > remaining)
            {
                summary.valid = false;
                summary.reason = VuExitReason::Fault;
                break;
            }
            summary.pairs += result.executedCycles;
            summary.reason = result.reason;
            if (result.reason ==
                    VuExitReason::XgkickBoundary &&
                result.executedCycles != 0u)
            {
                continue;
            }
            if (result.reason ==
                    VuExitReason::CycleBudget &&
                summary.pairs == maximumPairs)
            {
                break;
            }
            if (result.executedCycles == 0u &&
                context.state.active)
            {
                summary.valid = false;
            }
            break;
        }
        return summary;
    }

    VuExecutionState loadInitialState(
        const std::array<uint32_t, 159u> &words,
        PS2Memory &memory)
    {
        VuUnit seed(VuUnitId::Vu1);
        VuExecutionState &state = seed.state();
        size_t cursor = 5u;
        std::memcpy(
            state.vf, words.data() + cursor,
            sizeof(state.vf));
        cursor += 32u * 4u;
        for (size_t index = 0u; index < 16u; ++index)
        {
            state.vi[index] =
                static_cast<int32_t>(words[cursor++]);
        }
        std::memcpy(
            state.acc, words.data() + cursor,
            sizeof(state.acc));
        cursor += 4u;
        std::memcpy(
            &state.q, &words[cursor++],
            sizeof(state.q));
        std::memcpy(
            &state.p, &words[cursor++],
            sizeof(state.p));
        std::memcpy(
            &state.i, &words[cursor++],
            sizeof(state.i));
        state.status = words[cursor++];
        state.mac = words[cursor++];
        state.clip = words[cursor++];
        seed.start(
            words[2], words[3], words[4],
            &memory);
        return state;
    }

    void prepareInvocations(
        std::vector<Invocation> &invocations,
        const VuExecutionState &initialState,
        const std::array<
            uint8_t, PS2_VU1_DATA_SIZE> &initialData)
    {
        for (Invocation &invocation : invocations)
        {
            invocation.state = initialState;
            std::memcpy(
                invocation.data.data(),
                initialData.data(),
                initialData.size());
            invocation.effects.reset();
        }
    }

    struct Measurement
    {
        uint64_t nanoseconds = 0u;
        uint64_t pairs = 0u;
        uint64_t allocations = 0u;
        uint64_t allocationBytes = 0u;
        uint64_t packets = 0u;
        uint64_t packetBytes = 0u;
        bool valid = true;
        VuRecompilerDiagnostics nativeDelta{};
        VuProgramCacheDiagnostics cacheDelta{};
    };

    uint64_t difference(
        uint64_t after, uint64_t before)
    {
        return after - before;
    }

    Measurement measure(
        IVuExecutionBackend &backend,
        VuRecompilerBackend *native,
        VuUnit *nativeUnit,
        std::vector<Invocation> &invocations,
        const VuExecutionState &initialState,
        const std::array<
            uint8_t, PS2_VU1_DATA_SIZE> &initialData,
        const VuExecutionState &expectedState,
        const std::array<
            uint8_t, PS2_VU1_DATA_SIZE> &expectedData,
        const HashingSink &expectedEffects,
        const uint8_t *code, PS2Memory &memory,
        uint32_t maximumPairs,
        uint32_t expectedPairs)
    {
        prepareInvocations(
            invocations, initialState, initialData);
        const VuRecompilerDiagnostics nativeBefore =
            native ? native->diagnostics()
                   : VuRecompilerDiagnostics{};
        const VuProgramCacheDiagnostics cacheBefore =
            nativeUnit
                ? nativeUnit->programCache().diagnostics()
                : VuProgramCacheDiagnostics{};

        g_allocationCount.store(
            0u, std::memory_order_relaxed);
        g_allocationBytes.store(
            0u, std::memory_order_relaxed);
        g_trackAllocations.store(
            true, std::memory_order_release);
        const auto started = Clock::now();
        Measurement result;
        for (Invocation &invocation : invocations)
        {
            VuExecutionContext context{
                .state = invocation.state,
                .code = code,
                .codeSize = PS2_VU1_CODE_SIZE,
                .data = invocation.data.data(),
                .dataSize = PS2_VU1_DATA_SIZE,
                .sideEffects = invocation.effects,
                .memory = &memory,
                .enableInstrumentation = false,
            };
            const RunSummary run =
                runBackend(
                    backend, context, maximumPairs);
            result.pairs += run.pairs;
            result.packets +=
                invocation.effects.packets;
            result.packetBytes +=
                invocation.effects.bytes;
            if (!run.valid ||
                run.pairs != expectedPairs)
            {
                result.valid = false;
            }
        }
        const auto stopped = Clock::now();
        g_trackAllocations.store(
            false, std::memory_order_release);

        result.nanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    stopped - started)
                    .count());
        result.allocations =
            g_allocationCount.load(
                std::memory_order_relaxed);
        result.allocationBytes =
            g_allocationBytes.load(
                std::memory_order_relaxed);

        for (const Invocation &invocation : invocations)
        {
            std::string differenceField;
            if (!vuExecutionStatesEqual(
                    invocation.state, expectedState,
                    &differenceField) ||
                invocation.data != expectedData ||
                !(invocation.effects ==
                  expectedEffects))
            {
                result.valid = false;
                break;
            }
        }

        if (native)
        {
            const VuRecompilerDiagnostics after =
                native->diagnostics();
            result.nativeDelta.nativeEntries =
                difference(
                    after.nativeEntries,
                    nativeBefore.nativeEntries);
            result.nativeDelta.nativePairs =
                difference(
                    after.nativePairs,
                    nativeBefore.nativePairs);
            result.nativeDelta.inlinePairs =
                difference(
                    after.inlinePairs,
                    nativeBefore.inlinePairs);
            result.nativeDelta.helperPairs =
                difference(
                    after.helperPairs,
                    nativeBefore.helperPairs);
            result.nativeDelta.blockCompletes =
                difference(
                    after.blockCompletes,
                    nativeBefore.blockCompletes);
            result.nativeDelta.cycleBudgetExits =
                difference(
                    after.cycleBudgetExits,
                    nativeBefore.cycleBudgetExits);
            result.nativeDelta.xgkickExits =
                difference(
                    after.xgkickExits,
                    nativeBefore.xgkickExits);
            result.nativeDelta.xgkickAdvanceHelperCalls =
                difference(
                    after.xgkickAdvanceHelperCalls,
                    nativeBefore.xgkickAdvanceHelperCalls);
            result.nativeDelta.unsupportedExits =
                difference(
                    after.unsupportedExits,
                    nativeBefore.unsupportedExits);
            result.nativeDelta.codeInvalidationExits =
                difference(
                    after.codeInvalidationExits,
                    nativeBefore.codeInvalidationExits);
            result.nativeDelta.faultExits =
                difference(
                    after.faultExits,
                    nativeBefore.faultExits);
        }
        if (nativeUnit)
        {
            const VuProgramCacheDiagnostics after =
                nativeUnit->programCache().diagnostics();
            result.cacheDelta.hits =
                difference(
                    after.hits, cacheBefore.hits);
            result.cacheDelta.misses =
                difference(
                    after.misses, cacheBefore.misses);
            result.cacheDelta.compilations =
                difference(
                    after.compilations,
                    cacheBefore.compilations);
        }
        return result;
    }

    uint64_t median(std::vector<uint64_t> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    }

    void printMeasurement(
        std::string_view backend, uint32_t sample,
        uint32_t invocations,
        const Measurement &measurement)
    {
        const double pairsPerSecond =
            measurement.nanoseconds == 0u
                ? 0.0
                : static_cast<double>(
                      measurement.pairs) *
                      1.0e9 /
                      static_cast<double>(
                          measurement.nanoseconds);
        std::cout
            << "{\"schema_version\":1"
            << ",\"event\":\"warm-sample\""
            << ",\"backend\":\"" << backend << "\""
            << ",\"sample\":" << sample
            << ",\"invocations\":" << invocations
            << ",\"guest_pairs\":"
            << measurement.pairs
            << ",\"host_nanoseconds\":"
            << measurement.nanoseconds
            << ",\"guest_pairs_per_second\":"
            << std::fixed << std::setprecision(3)
            << pairsPerSecond
            << ",\"allocation_count\":"
            << measurement.allocations
            << ",\"allocation_bytes\":"
            << measurement.allocationBytes
            << ",\"path1_packets\":"
            << measurement.packets
            << ",\"path1_bytes\":"
            << measurement.packetBytes
            << ",\"valid\":"
            << (measurement.valid
                    ? "true" : "false");
        if (backend == "recompiler")
        {
            std::cout
                << ",\"native_entries\":"
                << measurement.nativeDelta.nativeEntries
                << ",\"native_pairs\":"
                << measurement.nativeDelta.nativePairs
                << ",\"inline_pairs\":"
                << measurement.nativeDelta.inlinePairs
                << ",\"helper_pairs\":"
                << measurement.nativeDelta.helperPairs
                << ",\"block_completes\":"
                << measurement.nativeDelta.blockCompletes
                << ",\"cycle_budget_exits\":"
                << measurement.nativeDelta.cycleBudgetExits
                << ",\"xgkick_exits\":"
                << measurement.nativeDelta.xgkickExits
                << ",\"xgkick_advance_helper_calls\":"
                << measurement.nativeDelta
                       .xgkickAdvanceHelperCalls
                << ",\"unsupported_exits\":"
                << measurement.nativeDelta.unsupportedExits
                << ",\"code_invalidation_exits\":"
                << measurement.nativeDelta.codeInvalidationExits
                << ",\"fault_exits\":"
                << measurement.nativeDelta.faultExits
                << ",\"cache_hits\":"
                << measurement.cacheDelta.hits
                << ",\"cache_misses\":"
                << measurement.cacheDelta.misses
                << ",\"cache_compilations\":"
                << measurement.cacheDelta.compilations;
        }
        std::cout << "}\n";
    }
}

int main(int argc, char **argv)
{
    Configuration configuration;
    if (!parseArguments(
            argc, argv, configuration))
    {
        std::cerr
            << "usage: vu_backend_benchmark "
               "FIXTURE_DIRECTORY INSTRUCTION_PAIRS "
               "[--iterations N] [--warmup N] "
               "[--samples N]\n";
        return 2;
    }
    if (!VuRecompilerBackend::supported())
    {
        std::cerr
            << "the x86-64 VU recompiler is unavailable\n";
        return 2;
    }

    FixturePaths paths;
    if (!resolveFixture(
            configuration.fixtureDirectory, paths))
    {
        std::cerr << "failed to resolve VU1 fixture\n";
        return 2;
    }
    std::error_code sizeError;
    if (std::filesystem::file_size(
            paths.data, sizeError) !=
            PS2_VU1_DATA_SIZE ||
        sizeError ||
        std::filesystem::file_size(
            paths.micro, sizeError) !=
            PS2_VU1_CODE_SIZE ||
        sizeError)
    {
        std::cerr
            << "benchmark requires full-size VU1 data and code\n";
        return 2;
    }

    std::array<uint8_t, PS2_VU1_DATA_SIZE>
        initialData{};
    std::array<uint8_t, PS2_VU1_CODE_SIZE> micro{};
    std::array<uint32_t, 159u> stateWords{};
    if (!readExact(paths.data, initialData) ||
        !readExact(paths.micro, micro) ||
        !readExact(paths.state, stateWords) ||
        stateWords[0] != 0x31555652u ||
        stateWords[1] != 1u)
    {
        std::cerr << "failed to read VU1 fixture\n";
        return 2;
    }

    PS2Memory memory;
    if (!memory.initialize())
    {
        std::cerr << "failed to initialize memory\n";
        return 2;
    }
    std::memcpy(
        memory.getVU1Code(), micro.data(),
        micro.size());
    memory.markVU1CodeModified();
    const VuExecutionState initialState =
        loadInitialState(stateWords, memory);

    VuUnit interpreterUnit(VuUnitId::Vu1);
    VuUnit nativeUnit(VuUnitId::Vu1);
    VuInterpreterBackend interpreter(interpreterUnit);
    VuRecompilerBackend native(nativeUnit);

    Invocation reference;
    Invocation cold;
    reference.state = initialState;
    reference.data = initialData;
    reference.effects.reset();
    VuExecutionContext referenceContext{
        .state = reference.state,
        .code = memory.getVU1Code(),
        .codeSize = PS2_VU1_CODE_SIZE,
        .data = reference.data.data(),
        .dataSize = PS2_VU1_DATA_SIZE,
        .sideEffects = reference.effects,
        .memory = &memory,
        .enableInstrumentation = false,
    };
    const RunSummary referenceRun = runBackend(
        interpreter, referenceContext,
        configuration.maximumPairs);
    if (!referenceRun.valid ||
        referenceRun.pairs == 0u)
    {
        std::cerr
            << "interpreter fixture execution failed\n";
        return 1;
    }

    cold.state = initialState;
    cold.data = initialData;
    cold.effects.reset();
    VuExecutionContext coldContext{
        .state = cold.state,
        .code = memory.getVU1Code(),
        .codeSize = PS2_VU1_CODE_SIZE,
        .data = cold.data.data(),
        .dataSize = PS2_VU1_DATA_SIZE,
        .sideEffects = cold.effects,
        .memory = &memory,
        .enableInstrumentation = false,
    };
    g_allocationCount.store(
        0u, std::memory_order_relaxed);
    g_allocationBytes.store(
        0u, std::memory_order_relaxed);
    g_trackAllocations.store(
        true, std::memory_order_release);
    const auto coldStarted = Clock::now();
    const RunSummary coldRun = runBackend(
        native, coldContext,
        configuration.maximumPairs);
    const auto coldStopped = Clock::now();
    g_trackAllocations.store(
        false, std::memory_order_release);
    const uint64_t coldNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                coldStopped - coldStarted)
                .count());
    std::string stateDifference;
    const bool coldMatches =
        coldRun.valid &&
        coldRun.pairs == referenceRun.pairs &&
        vuExecutionStatesEqual(
            cold.state, reference.state,
            &stateDifference) &&
        cold.data == reference.data &&
        cold.effects == reference.effects;
    const VuProgramCacheDiagnostics coldCache =
        nativeUnit.programCache().diagnostics();
    std::cout
        << "{\"schema_version\":1"
        << ",\"event\":\"cold-recompiler\""
        << ",\"guest_pairs\":" << coldRun.pairs
        << ",\"host_nanoseconds\":"
        << coldNanoseconds
        << ",\"allocation_count\":"
        << g_allocationCount.load(
               std::memory_order_relaxed)
        << ",\"allocation_bytes\":"
        << g_allocationBytes.load(
               std::memory_order_relaxed)
        << ",\"compilations\":"
        << coldCache.compilations
        << ",\"compilation_nanoseconds\":"
        << coldCache.compilationNanoseconds
        << ",\"generated_bytes\":"
        << coldCache.generatedBytes
        << ",\"resident_programs\":"
        << coldCache.residentPrograms
        << ",\"resident_executable_bytes\":"
        << coldCache.residentExecutableBytes
        << ",\"matches_interpreter\":"
        << (coldMatches ? "true" : "false")
        << "}\n";
    if (!coldMatches)
    {
        std::cerr
            << "cold recompiler mismatch: "
            << stateDifference << "\n";
        return 1;
    }

    Invocation warmup;
    for (uint32_t iteration = 0u;
         iteration < configuration.warmupIterations;
         ++iteration)
    {
        warmup.state = initialState;
        warmup.data = initialData;
        warmup.effects.reset();
        VuExecutionContext context{
            .state = warmup.state,
            .code = memory.getVU1Code(),
            .codeSize = PS2_VU1_CODE_SIZE,
            .data = warmup.data.data(),
            .dataSize = PS2_VU1_DATA_SIZE,
            .sideEffects = warmup.effects,
            .memory = &memory,
            .enableInstrumentation = false,
        };
        if (!runBackend(
                 interpreter, context,
                 configuration.maximumPairs)
                 .valid)
        {
            std::cerr
                << "interpreter warmup failed\n";
            return 1;
        }

        warmup.state = initialState;
        warmup.data = initialData;
        warmup.effects.reset();
        if (!runBackend(
                 native, context,
                 configuration.maximumPairs)
                 .valid)
        {
            std::cerr
                << "recompiler warmup failed\n";
            return 1;
        }
    }

    std::vector<Invocation> invocations(
        configuration.iterations);
    std::vector<uint64_t> interpreterTimes;
    std::vector<uint64_t> nativeTimes;
    bool allValid = true;
    for (uint32_t sample = 0u;
         sample < configuration.samples; ++sample)
    {
        const auto measureInterpreter = [&]
        {
            const Measurement measurement =
                measure(
                    interpreter, nullptr, nullptr,
                    invocations, initialState, initialData,
                    reference.state, reference.data,
                    reference.effects,
                    memory.getVU1Code(), memory,
                    configuration.maximumPairs,
                    referenceRun.pairs);
            printMeasurement(
                "interpreter", sample,
                configuration.iterations,
                measurement);
            interpreterTimes.push_back(
                measurement.nanoseconds);
            allValid = allValid && measurement.valid;
        };
        const auto measureRecompiler = [&]
        {
            const Measurement measurement =
                measure(
                    native, &native, &nativeUnit,
                    invocations, initialState, initialData,
                    reference.state, reference.data,
                    reference.effects,
                    memory.getVU1Code(), memory,
                    configuration.maximumPairs,
                    referenceRun.pairs);
            printMeasurement(
                "recompiler", sample,
                configuration.iterations,
                measurement);
            nativeTimes.push_back(
                measurement.nanoseconds);
            allValid = allValid && measurement.valid;
        };
        if ((sample & 1u) == 0u)
        {
            measureInterpreter();
            measureRecompiler();
        }
        else
        {
            measureRecompiler();
            measureInterpreter();
        }
    }

    const uint64_t interpreterMedian =
        median(interpreterTimes);
    const uint64_t nativeMedian =
        median(nativeTimes);
    const VuProgramCacheDiagnostics finalCache =
        nativeUnit.programCache().diagnostics();
    std::cout
        << "{\"schema_version\":1"
        << ",\"event\":\"summary\""
        << ",\"fixture\":\""
        << configuration.fixtureDirectory.filename().string()
        << "\""
        << ",\"pairs_per_invocation\":"
        << referenceRun.pairs
        << ",\"iterations_per_sample\":"
        << configuration.iterations
        << ",\"samples\":"
        << configuration.samples
        << ",\"interpreter_median_nanoseconds\":"
        << interpreterMedian
        << ",\"recompiler_median_nanoseconds\":"
        << nativeMedian
        << ",\"speedup\":"
        << std::fixed << std::setprecision(6)
        << (nativeMedian == 0u
                ? 0.0
                : static_cast<double>(
                      interpreterMedian) /
                      static_cast<double>(
                          nativeMedian))
        << ",\"cache_hits\":"
        << finalCache.hits
        << ",\"cache_misses\":"
        << finalCache.misses
        << ",\"compilations\":"
        << finalCache.compilations
        << ",\"generated_bytes\":"
        << finalCache.generatedBytes
        << ",\"high_water_executable_bytes\":"
        << finalCache.highWaterExecutableBytes
        << ",\"valid\":"
        << (allValid ? "true" : "false")
        << "}\n";
    return allValid ? 0 : 1;
}
