#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_analysis.h"
#include "runtime/ps2_vu_recompiler.h"
#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
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
#include <thread>
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

    enum class TimingScope : uint8_t
    {
        VuCore,
        VuPlusPath1,
    };

    std::string_view timingScopeName(
        TimingScope scope)
    {
        switch (scope)
        {
        case TimingScope::VuCore:
            return "vu-core";
        case TimingScope::VuPlusPath1:
            return "vu-plus-path1";
        }
        return "unknown";
    }

    struct Configuration
    {
        std::filesystem::path fixtureDirectory;
        uint32_t maximumPairs = 0u;
        uint32_t iterations = 1000u;
        uint32_t warmupIterations = 10u;
        uint32_t samples = 7u;
        uint32_t cacheChurnIterations = 0u;
        bool measureVuCore = true;
        bool measureVuPlusPath1 = true;
        bool measureInterpreter = true;
        bool measureRecompiler = true;
        bool analysisCheck = false;
        bool budgetSweep = false;
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
            const std::string_view option(argv[index]);
            const std::string_view argument(argv[index + 1]);
            if (option == "--scope")
            {
                configuration.measureVuCore =
                    argument == "core" || argument == "all";
                configuration.measureVuPlusPath1 =
                    argument == "path1" || argument == "all";
                if (!configuration.measureVuCore &&
                    !configuration.measureVuPlusPath1)
                {
                    return false;
                }
                continue;
            }
            if (option == "--backend")
            {
                configuration.measureInterpreter =
                    argument == "interpreter" ||
                    argument == "both";
                configuration.measureRecompiler =
                    argument == "recompiler" ||
                    argument == "both";
                if (!configuration.measureInterpreter &&
                    !configuration.measureRecompiler)
                {
                    return false;
                }
                continue;
            }
            if (option == "--analysis-check")
            {
                if (argument == "1" ||
                    argument == "true" ||
                    argument == "on")
                {
                    configuration.analysisCheck = true;
                }
                else if (
                    argument == "0" ||
                    argument == "false" ||
                    argument == "off")
                {
                    configuration.analysisCheck = false;
                }
                else
                {
                    return false;
                }
                continue;
            }
            if (option == "--budget-sweep")
            {
                if (argument == "1" ||
                    argument == "true" ||
                    argument == "on")
                {
                    configuration.budgetSweep = true;
                }
                else if (
                    argument == "0" ||
                    argument == "false" ||
                    argument == "off")
                {
                    configuration.budgetSweep = false;
                }
                else
                {
                    return false;
                }
                continue;
            }

            uint64_t value = 0u;
            if (!parseUnsigned(argument, value) ||
                value == 0u || value > UINT32_MAX)
            {
                return false;
            }
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
            else if (option == "--cache-churn-iterations")
            {
                configuration.cacheChurnIterations =
                    static_cast<uint32_t>(value);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    class PerfStatControl
    {
    public:
        PerfStatControl()
        {
            const char *const controlPath =
                std::getenv(
                    "PS2X_PERF_CONTROL_FIFO");
            const char *const acknowledgePath =
                std::getenv(
                    "PS2X_PERF_ACK_FIFO");
            if (!controlPath && !acknowledgePath)
                return;
            m_requested = true;
            if (!controlPath || !acknowledgePath)
            {
                m_error =
                    "both PS2X_PERF_CONTROL_FIFO and "
                    "PS2X_PERF_ACK_FIFO are required";
                return;
            }
            m_control.open(controlPath);
            if (!m_control)
            {
                m_error =
                    "failed to open perf control FIFO";
                return;
            }
            m_acknowledge.open(acknowledgePath);
            if (!m_acknowledge)
            {
                m_error =
                    "failed to open perf acknowledgement FIFO";
                return;
            }
            m_valid = true;
        }

        [[nodiscard]] bool valid() const
        {
            return !m_requested || m_valid;
        }

        [[nodiscard]] const std::string &error() const
        {
            return m_error;
        }

        bool enable()
        {
            return command("enable");
        }

        bool disable()
        {
            return command("disable");
        }

    private:
        bool command(std::string_view text)
        {
            if (!m_requested)
                return true;
            if (!m_valid)
                return false;
            m_control << text << '\n';
            m_control.flush();
            for (uint32_t attempt = 0u;
                 attempt < 1000u; ++attempt)
            {
                std::string response;
                m_acknowledge.clear();
                if (std::getline(
                        m_acknowledge, response))
                {
                    const size_t firstText =
                        response.find_first_not_of('\0');
                    if (firstText == std::string::npos)
                        response.clear();
                    else if (firstText != 0u)
                        response.erase(0u, firstText);
                    if (response == "ack")
                        return true;
                    m_error =
                        "perf control returned an "
                        "unexpected acknowledgement: " +
                        response;
                    m_valid = false;
                    return false;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));
            }
            m_error =
                "perf control command was not "
                "acknowledged";
            m_valid = false;
            return false;
        }

        bool m_requested = false;
        bool m_valid = false;
        std::string m_error;
        std::ofstream m_control;
        std::ifstream m_acknowledge;
    };

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

    constexpr uint64_t kFnvOffset =
        1469598103934665603ull;
    constexpr uint64_t kFnvPrime =
        1099511628211ull;

    template <typename Value>
    void hashValue(uint64_t &hash, const Value &value)
    {
        const auto *const bytes =
            reinterpret_cast<const uint8_t *>(&value);
        for (size_t index = 0u;
             index < sizeof(Value); ++index)
        {
            hash ^= bytes[index];
            hash *= kFnvPrime;
        }
    }

    uint64_t hashBytes(
        const uint8_t *data, size_t size)
    {
        uint64_t hash = kFnvOffset;
        for (size_t index = 0u; index < size; ++index)
        {
            hash ^= data[index];
            hash *= kFnvPrime;
        }
        return hash;
    }

    class BufferedPath1Sink final : public IVuSideEffectSink
    {
    public:
        void submitPath1Packet(
            const uint8_t *data,
            uint32_t sizeBytes) override
        {
            m_packetSizes.push_back(sizeBytes);
            m_bytes.insert(
                m_bytes.end(), data, data + sizeBytes);
        }

        void reset()
        {
            m_packetSizes.clear();
            m_bytes.clear();
        }

        void reserveLike(
            const BufferedPath1Sink &other)
        {
            m_packetSizes.reserve(
                other.m_packetSizes.size());
            m_bytes.reserve(other.m_bytes.size());
        }

        bool reserveRepeated(
            const BufferedPath1Sink &other,
            size_t repetitions)
        {
            if ((other.m_packetSizes.size() != 0u &&
                 repetitions >
                     m_packetSizes.max_size() /
                         other.m_packetSizes.size()) ||
                (other.m_bytes.size() != 0u &&
                 repetitions >
                     m_bytes.max_size() /
                         other.m_bytes.size()))
            {
                return false;
            }
            m_packetSizes.reserve(
                other.m_packetSizes.size() *
                repetitions);
            m_bytes.reserve(
                other.m_bytes.size() * repetitions);
            return true;
        }

        [[nodiscard]] uint64_t packetCount() const
        {
            return m_packetSizes.size();
        }

        [[nodiscard]] uint64_t byteCount() const
        {
            return m_bytes.size();
        }

        [[nodiscard]] uint64_t hash() const
        {
            uint64_t result = kFnvOffset;
            for (const uint32_t size : m_packetSizes)
                hashValue(result, size);
            for (const uint8_t byte : m_bytes)
            {
                result ^= byte;
                result *= kFnvPrime;
            }
            return result;
        }

        [[nodiscard]] uint64_t hashRepeated(
            size_t repetitions) const
        {
            uint64_t result = kFnvOffset;
            for (size_t repetition = 0u;
                 repetition < repetitions;
                 ++repetition)
            {
                for (const uint32_t size :
                     m_packetSizes)
                {
                    hashValue(result, size);
                }
            }
            for (size_t repetition = 0u;
                 repetition < repetitions;
                 ++repetition)
            {
                for (const uint8_t byte : m_bytes)
                {
                    result ^= byte;
                    result *= kFnvPrime;
                }
            }
            return result;
        }

        bool operator==(
            const BufferedPath1Sink &other) const
        {
            return
                m_packetSizes == other.m_packetSizes &&
                m_bytes == other.m_bytes;
        }

        bool equalsRepeated(
            const BufferedPath1Sink &other,
            size_t repetitions) const
        {
            if ((other.m_packetSizes.size() != 0u &&
                 repetitions >
                     m_packetSizes.max_size() /
                         other.m_packetSizes.size()) ||
                (other.m_bytes.size() != 0u &&
                 repetitions >
                     m_bytes.max_size() /
                         other.m_bytes.size()) ||
                m_packetSizes.size() !=
                    other.m_packetSizes.size() *
                        repetitions ||
                m_bytes.size() !=
                    other.m_bytes.size() * repetitions)
            {
                return false;
            }
            for (size_t repetition = 0u;
                 repetition < repetitions;
                 ++repetition)
            {
                if (!std::equal(
                        other.m_packetSizes.begin(),
                        other.m_packetSizes.end(),
                        m_packetSizes.begin() +
                            static_cast<ptrdiff_t>(
                                repetition *
                                other.m_packetSizes
                                    .size())) ||
                    !std::equal(
                        other.m_bytes.begin(),
                        other.m_bytes.end(),
                        m_bytes.begin() +
                            static_cast<ptrdiff_t>(
                                repetition *
                                other.m_bytes.size())))
                {
                    return false;
                }
            }
            return true;
        }

    private:
        std::vector<uint32_t> m_packetSizes;
        std::vector<uint8_t> m_bytes;
    };

    class ProductionPath1Transfer final
        : public IVuSideEffectSink
    {
    public:
        explicit ProductionPath1Transfer(
            PS2Memory &memory)
            : m_memory(memory),
              m_arbiter(
                  [this](const uint8_t *data,
                         uint32_t sizeBytes)
                  {
                      if (m_output)
                      {
                          m_output->submitPath1Packet(
                              data, sizeBytes);
                      }
                  })
        {
        }

        ~ProductionPath1Transfer() override
        {
            m_memory.setGifArbiter(nullptr);
        }

        void begin(BufferedPath1Sink &output)
        {
            if (!m_arbiter.empty())
                m_arbiter.reset();
            m_output = &output;
            m_memory.setGifArbiter(&m_arbiter);
        }

        void submitPath1Packet(
            const uint8_t *data,
            uint32_t sizeBytes) override
        {
            m_memory.submitGifPacket(
                GifPathId::Path1, data, sizeBytes,
                false);
        }

        void drain()
        {
            m_arbiter.drain();
        }

        [[nodiscard]] bool empty() const
        {
            return m_arbiter.empty();
        }

        void end()
        {
            m_memory.setGifArbiter(nullptr);
            m_output = nullptr;
        }

    private:
        PS2Memory &m_memory;
        GifArbiter m_arbiter;
        BufferedPath1Sink *m_output = nullptr;
    };

    struct Invocation
    {
        VuExecutionState state{};
        std::array<uint8_t, PS2_VU1_DATA_SIZE> data{};
        BufferedPath1Sink effects;
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
        summary.reason =
            context.state.active
                ? VuExitReason::CycleBudget
                : VuExitReason::Inactive;
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
        if (context.state.active &&
            summary.pairs == maximumPairs &&
            summary.reason ==
                VuExitReason::XgkickBoundary)
        {
            // XGKICK is an internal native block split. The VU adapter
            // continues through it and reports the exhausted public budget.
            summary.reason = VuExitReason::CycleBudget;
        }
        return summary;
    }

    struct BudgetSweepSummary
    {
        bool valid = false;
        uint64_t budgets = 0u;
        uint64_t pairs = 0u;
        VuRecompilerDiagnostics diagnostics{};
        std::string diagnostic;
    };

    BudgetSweepSummary validateBudgetSweep(
        const uint8_t *code,
        const VuExecutionState &initialState,
        const std::array<
            uint8_t, PS2_VU1_DATA_SIZE> &initialData,
        PS2Memory &memory,
        uint32_t completionPairs)
    {
        BudgetSweepSummary summary;
        VuUnit interpreterUnit(VuUnitId::Vu1);
        VuUnit nativeUnit(VuUnitId::Vu1);
        VuInterpreterBackend interpreter(
            interpreterUnit);
        VuRecompilerBackend native(nativeUnit);
        const VuRecompilerDiagnostics before =
            native.diagnostics();

        for (uint64_t budget = 0u;
             budget <=
                 static_cast<uint64_t>(
                     completionPairs) +
                     1u;
             ++budget)
        {
            Invocation reference;
            Invocation recompiled;
            reference.state = initialState;
            reference.data = initialData;
            reference.effects.reset();
            recompiled.state = initialState;
            recompiled.data = initialData;
            recompiled.effects.reset();
            VuExecutionContext referenceContext{
                .state = reference.state,
                .code = code,
                .codeSize = PS2_VU1_CODE_SIZE,
                .data = reference.data.data(),
                .dataSize = PS2_VU1_DATA_SIZE,
                .sideEffects = reference.effects,
                .memory = &memory,
                .enableInstrumentation = false,
            };
            VuExecutionContext recompiledContext{
                .state = recompiled.state,
                .code = code,
                .codeSize = PS2_VU1_CODE_SIZE,
                .data = recompiled.data.data(),
                .dataSize = PS2_VU1_DATA_SIZE,
                .sideEffects = recompiled.effects,
                .memory = &memory,
                .enableInstrumentation = false,
            };
            const uint32_t maximumPairs =
                static_cast<uint32_t>(budget);
            const RunSummary referenceRun =
                runBackend(
                    interpreter,
                    referenceContext,
                    maximumPairs);
            const RunSummary recompiledRun =
                runBackend(
                    native,
                    recompiledContext,
                    maximumPairs);

            std::string differenceField;
            if (!referenceRun.valid ||
                !recompiledRun.valid ||
                referenceRun.pairs !=
                    recompiledRun.pairs ||
                referenceRun.reason !=
                    recompiledRun.reason ||
                !vuExecutionStatesEqual(
                    reference.state,
                    recompiled.state,
                    &differenceField) ||
                reference.data != recompiled.data ||
                !(reference.effects ==
                  recompiled.effects))
            {
                summary.diagnostic =
                    "budget " +
                    std::to_string(budget) +
                    " diverged at " +
                    differenceField +
                    " (interpreter " +
                    std::string(
                        vuExitReasonName(
                            referenceRun.reason)) +
                    "/" +
                    std::to_string(
                        referenceRun.pairs) +
                    ", recompiler " +
                    std::string(
                        vuExitReasonName(
                            recompiledRun.reason)) +
                    "/" +
                    std::to_string(
                        recompiledRun.pairs) +
                    ")";
                return summary;
            }
            ++summary.budgets;
            summary.pairs += referenceRun.pairs;
        }

        const VuRecompilerDiagnostics after =
            native.diagnostics();
        summary.diagnostics.nativeBlocks =
            after.nativeBlocks - before.nativeBlocks;
        summary.diagnostics.nativePairs =
            after.nativePairs - before.nativePairs;
        summary.diagnostics.fullBlockGuards =
            after.fullBlockGuards -
            before.fullBlockGuards;
        summary.diagnostics.preciseTailEntries =
            after.preciseTailEntries -
            before.preciseTailEntries;
        summary.diagnostics.fullGuardPairs =
            after.fullGuardPairs -
            before.fullGuardPairs;
        summary.diagnostics.preciseTailPairs =
            after.preciseTailPairs -
            before.preciseTailPairs;
        summary.diagnostics.budgetGuardComparisons =
            after.budgetGuardComparisons -
            before.budgetGuardComparisons;
        summary.diagnostics.nativeBudgetExits =
            after.nativeBudgetExits -
            before.nativeBudgetExits;
        summary.diagnostics.faultExits =
            after.faultExits - before.faultExits;
        if (summary.diagnostics.fullBlockGuards +
                    summary.diagnostics.
                        preciseTailEntries !=
                summary.diagnostics.nativeBlocks ||
            summary.diagnostics.fullGuardPairs +
                    summary.diagnostics.
                        preciseTailPairs !=
                summary.diagnostics.nativePairs ||
            summary.diagnostics.faultExits != 0u)
        {
            summary.diagnostic =
                "budget-guard diagnostics did not reconcile";
            return summary;
        }
        summary.valid = true;
        return summary;
    }

    struct AnalysisCheckSummary
    {
        bool valid = false;
        bool bounded = false;
        uint64_t analyzedBlocks = 0u;
        uint64_t executedBlocks = 0u;
        uint64_t pairs = 0u;
        uint64_t cycles = 0u;
        uint64_t staticEdges = 0u;
        uint64_t dynamicEdges = 0u;
        uint64_t xgkickBoundaries = 0u;
        uint64_t helperPairs = 0u;
        uint64_t vfReadRegisters = 0u;
        uint64_t vfWriteRegisters = 0u;
        uint64_t viReadRegisters = 0u;
        uint64_t viWriteRegisters = 0u;
        uint64_t vfLiveInRegisters = 0u;
        uint64_t viLiveInRegisters = 0u;
        uint64_t blockVfWorkingSet = 0u;
        uint64_t blockViWorkingSet = 0u;
        uint64_t blockVfAccesses = 0u;
        uint64_t top3BlockVfAccesses = 0u;
        uint64_t top8BlockVfAccesses = 0u;
        uint64_t top13BlockVfAccesses = 0u;
        uint64_t blocksFitting3VfRegisters = 0u;
        uint64_t blocksFitting8VfRegisters = 0u;
        uint64_t blocksFitting13VfRegisters = 0u;
        uint32_t maximumVfLiveInRegisters = 0u;
        uint32_t maximumViLiveInRegisters = 0u;
        uint32_t maximumBlockVfWorkingSet = 0u;
        uint32_t maximumBlockViWorkingSet = 0u;
        std::array<uint64_t, 32u> vfReads{};
        std::array<uint64_t, 32u> vfWrites{};
        std::array<uint64_t, 16u> viReads{};
        std::array<uint64_t, 16u> viWrites{};
        VuExitReason reason = VuExitReason::Inactive;
        std::string diagnostic;
    };

    VuAnalysisEntryState analysisEntry(
        const VuExecutionState &state)
    {
        VuAnalysisEntryState entry{
            .pc = state.pc,
            .endPending = state.ebit,
        };
        if (state.branchPending)
        {
            entry.pendingBranch =
                VuAnalysisPendingBranchKind::Static;
            entry.pendingBranchTarget =
                state.branchTarget;
        }
        return entry;
    }

    bool analysisEntryMatchesState(
        const VuAnalysisEntryState &entry,
        const VuExecutionState &state)
    {
        if (entry.pc != state.pc ||
            entry.endPending != state.ebit)
        {
            return false;
        }
        switch (entry.pendingBranch)
        {
        case VuAnalysisPendingBranchKind::None:
            return !state.branchPending;
        case VuAnalysisPendingBranchKind::Static:
            return
                state.branchPending &&
                state.branchDelay == 0u &&
                entry.pendingBranchTarget ==
                    state.branchTarget;
        case VuAnalysisPendingBranchKind::Dynamic:
            return
                state.branchPending &&
                state.branchDelay == 0u;
        }
        return false;
    }

    AnalysisCheckSummary validateAnalysisTrace(
        const uint8_t *code, uint32_t codeSize,
        const uint8_t *initialData, uint32_t dataSize,
        const VuExecutionState &initialState,
        PS2Memory &memory, uint32_t maximumPairs)
    {
        AnalysisCheckSummary summary;
        if (!code ||
            (dataSize != 0u && !initialData))
        {
            summary.diagnostic =
                "analysis trace requires code and data";
            return summary;
        }

        VuExecutionState state = initialState;
        std::vector<uint8_t> data(dataSize);
        if (dataSize != 0u)
        {
            std::memcpy(
                data.data(), initialData, dataSize);
        }
        BufferedPath1Sink effects;
        VuExecutionContext context{
            .state = state,
            .code = code,
            .codeSize = codeSize,
            .data = data.data(),
            .dataSize = dataSize,
            .sideEffects = effects,
            .memory = &memory,
            .enableInstrumentation = false,
            .enableProgressAccounting = false,
        };
        VuUnit unit(VuUnitId::Vu1);
        VuIrInterpreterBackend interpreter(unit);
        const VuAnalysisConfiguration configuration{
            .maximumPairsPerBlock = 64u,
            .maximumBlocks = 16384u,
            .hostFeatures =
                VuRecompilerBackend::hostFeatures(),
        };

        VuControlFlowGraph graph =
            analyzeVuControlFlow(
                code, codeSize,
                analysisEntry(state),
                configuration);
        if (!graph.valid || graph.blocks.empty())
        {
            summary.diagnostic =
                "initial CFG failed: " +
                graph.diagnostic;
            return summary;
        }
        summary.analyzedBlocks += graph.blocks.size();
        uint32_t blockIndex = 0u;
        const uint64_t issuedBefore =
            state.issuedCycles;

        while (state.active &&
               summary.pairs < maximumPairs)
        {
            if (blockIndex >= graph.blocks.size())
            {
                summary.diagnostic =
                    "resolved CFG block index is out of range";
                return summary;
            }
            const VuAnalysisBlock &block =
                graph.blocks[blockIndex];
            if (!analysisEntryMatchesState(
                    block.entry, state))
            {
                summary.diagnostic =
                    "interpreter state does not match CFG block entry";
                return summary;
            }

            ++summary.executedBlocks;
            uint32_t blockVfRegisters = 0u;
            uint16_t blockViRegisters = 0u;
            std::array<uint32_t, 32u>
                blockVfAccessCounts{};
            for (const VuAnalysisPair &pair :
                 block.pairs)
            {
                for (uint32_t reg = 1u;
                     reg < 32u; ++reg)
                {
                    if (pair.reads.vf[reg] != 0u ||
                        pair.writes.vf[reg] != 0u)
                    {
                        blockVfRegisters |= 1u << reg;
                    }
                    blockVfAccessCounts[reg] +=
                        pair.reads.vf[reg] != 0u
                            ? 1u : 0u;
                    blockVfAccessCounts[reg] +=
                        pair.writes.vf[reg] != 0u
                            ? 1u : 0u;
                }
                blockViRegisters |= static_cast<uint16_t>(
                    (pair.reads.vi | pair.writes.vi) &
                    ~uint16_t{1u});
            }
            const uint32_t blockVfWorkingSet =
                std::popcount(blockVfRegisters);
            const uint32_t blockViWorkingSet =
                std::popcount(blockViRegisters);
            summary.blockVfWorkingSet +=
                blockVfWorkingSet;
            summary.blockViWorkingSet +=
                blockViWorkingSet;
            summary.maximumBlockVfWorkingSet =
                std::max(
                    summary.maximumBlockVfWorkingSet,
                    blockVfWorkingSet);
            summary.maximumBlockViWorkingSet =
                std::max(
                    summary.maximumBlockViWorkingSet,
                    blockViWorkingSet);
            summary.blocksFitting3VfRegisters +=
                blockVfWorkingSet <= 3u ? 1u : 0u;
            summary.blocksFitting8VfRegisters +=
                blockVfWorkingSet <= 8u ? 1u : 0u;
            summary.blocksFitting13VfRegisters +=
                blockVfWorkingSet <= 13u ? 1u : 0u;
            std::array<uint8_t, 31u> vfOrder{};
            for (uint8_t index = 0u;
                 index < vfOrder.size(); ++index)
            {
                vfOrder[index] =
                    static_cast<uint8_t>(index + 1u);
            }
            std::stable_sort(
                vfOrder.begin(), vfOrder.end(),
                [&](uint8_t left, uint8_t right)
                {
                    return
                        blockVfAccessCounts[left] >
                        blockVfAccessCounts[right];
                });
            uint64_t blockAccesses = 0u;
            for (uint32_t reg = 1u;
                 reg < 32u; ++reg)
            {
                blockAccesses +=
                    blockVfAccessCounts[reg];
            }
            summary.blockVfAccesses += blockAccesses;
            for (size_t rank = 0u;
                 rank < vfOrder.size(); ++rank)
            {
                const uint64_t accesses =
                    blockVfAccessCounts[
                        vfOrder[rank]];
                if (rank < 3u)
                    summary.top3BlockVfAccesses += accesses;
                if (rank < 8u)
                    summary.top8BlockVfAccesses += accesses;
                if (rank < 13u)
                    summary.top13BlockVfAccesses += accesses;
            }
            uint32_t blockPairs = 0u;
            VuRunResult lastResult{};
            bool sawUnsupported = false;
            for (const VuAnalysisPair &pair :
                 block.pairs)
            {
                if (summary.pairs >= maximumPairs)
                {
                    summary.bounded = true;
                    summary.reason =
                        VuExitReason::CycleBudget;
                    break;
                }
                if (state.pc != pair.ir.pc)
                {
                    summary.diagnostic =
                        "interpreter PC diverged inside a static block";
                    return summary;
                }

                summary.helperPairs +=
                    pair.usesNativeHelper ? 1u : 0u;
                uint32_t vfLive = 0u;
                for (uint32_t reg = 1u;
                     reg < 32u; ++reg)
                {
                    if (pair.reads.vf[reg] != 0u)
                    {
                        ++summary.vfReadRegisters;
                        ++summary.vfReads[reg];
                    }
                    if (pair.writes.vf[reg] != 0u)
                    {
                        ++summary.vfWriteRegisters;
                        ++summary.vfWrites[reg];
                    }
                    if (pair.liveIn.vf[reg] != 0u)
                        ++vfLive;
                }
                const uint16_t viReads =
                    pair.reads.vi & ~uint16_t{1u};
                const uint16_t viWrites =
                    pair.writes.vi & ~uint16_t{1u};
                const uint16_t viLive =
                    pair.liveIn.vi & ~uint16_t{1u};
                summary.viReadRegisters +=
                    std::popcount(viReads);
                summary.viWriteRegisters +=
                    std::popcount(viWrites);
                summary.vfLiveInRegisters += vfLive;
                summary.viLiveInRegisters +=
                    std::popcount(viLive);
                summary.maximumVfLiveInRegisters =
                    std::max(
                        summary.maximumVfLiveInRegisters,
                        vfLive);
                summary.maximumViLiveInRegisters =
                    std::max(
                        summary.maximumViLiveInRegisters,
                        static_cast<uint32_t>(
                            std::popcount(viLive)));
                for (uint32_t reg = 1u;
                     reg < 16u; ++reg)
                {
                    const uint16_t bit =
                        static_cast<uint16_t>(1u << reg);
                    if ((viReads & bit) != 0u)
                        ++summary.viReads[reg];
                    if ((viWrites & bit) != 0u)
                        ++summary.viWrites[reg];
                }

                lastResult =
                    interpreter.run(context, 1u);
                if (!pair.retires)
                {
                    if (lastResult.executedCycles != 0u ||
                        lastResult.reason !=
                            VuExitReason::
                                UnsupportedInstruction)
                    {
                        summary.diagnostic =
                            "unsupported analysis pair retired in the interpreter";
                        return summary;
                    }
                    sawUnsupported = true;
                    break;
                }
                if (lastResult.executedCycles !=
                    pair.ir.cycles)
                {
                    summary.diagnostic =
                        "analysis pair cost did not match the interpreter";
                    return summary;
                }
                ++blockPairs;
                summary.pairs +=
                    lastResult.executedCycles;
                summary.cycles +=
                    lastResult.executedCycles;
                summary.reason = lastResult.reason;
                if (!state.active)
                    break;
            }

            if (summary.bounded)
                break;
            if (blockPairs != block.fixedPairs ||
                blockPairs != block.fixedCycles)
            {
                summary.diagnostic =
                    "fixed block cost did not match the interpreter trace";
                return summary;
            }

            if (block.terminal ==
                    VuAnalysisBlockTerminal::
                        UnsupportedInstruction)
            {
                if (!sawUnsupported)
                {
                    summary.diagnostic =
                        "analysis expected an unsupported exit that the interpreter did not take";
                    return summary;
                }
                summary.reason =
                    VuExitReason::UnsupportedInstruction;
                break;
            }
            if (block.terminal ==
                    VuAnalysisBlockTerminal::ProgramEnd)
            {
                if (state.active ||
                    lastResult.reason !=
                        VuExitReason::ProgramEnded)
                {
                    summary.diagnostic =
                        "analysis program-end cost did not stop the interpreter";
                    return summary;
                }
                summary.reason =
                    VuExitReason::ProgramEnded;
                break;
            }
            if (!state.active)
            {
                summary.diagnostic =
                    "interpreter stopped at a non-terminal analysis block";
                return summary;
            }
            if (block.terminal ==
                VuAnalysisBlockTerminal::XgkickBoundary)
            {
                ++summary.xgkickBoundaries;
            }

            const VuAnalysisSuccessor *matched = nullptr;
            const VuAnalysisSuccessor *dynamic = nullptr;
            for (const VuAnalysisSuccessor &successor :
                 block.successors)
            {
                if (successor.fixedPairs !=
                        block.fixedPairs ||
                    successor.fixedCycles !=
                        block.fixedCycles)
                {
                    summary.diagnostic =
                        "CFG edge lost its source block cost";
                    return summary;
                }
                if (successor.dynamicTarget)
                {
                    dynamic = &successor;
                    continue;
                }
                if (analysisEntryMatchesState(
                        successor.entry, state))
                {
                    matched = &successor;
                    break;
                }
            }

            if (matched)
            {
                ++summary.staticEdges;
                if (matched->targetBlock >=
                    graph.blocks.size())
                {
                    summary.diagnostic =
                        "static CFG edge has no resolved target";
                    return summary;
                }
                blockIndex = matched->targetBlock;
                continue;
            }
            if (!dynamic ||
                (state.pc & 7u) != 0u ||
                state.pc >= codeSize)
            {
                summary.diagnostic =
                    "interpreter successor is absent from the CFG";
                return summary;
            }

            ++summary.dynamicEdges;
            graph = analyzeVuControlFlow(
                code, codeSize,
                analysisEntry(state),
                configuration);
            if (!graph.valid || graph.blocks.empty())
            {
                summary.diagnostic =
                    "dynamic CFG continuation failed: " +
                    graph.diagnostic;
                return summary;
            }
            summary.analyzedBlocks +=
                graph.blocks.size();
            blockIndex = 0u;
        }

        if (state.issuedCycles - issuedBefore !=
                summary.cycles ||
            summary.pairs != summary.cycles)
        {
            summary.diagnostic =
                "analysis trace cycle accounting did not reconcile";
            return summary;
        }
        if (summary.pairs == maximumPairs &&
            state.active)
        {
            summary.bounded = true;
            summary.reason = VuExitReason::CycleBudget;
        }
        summary.valid = true;
        return summary;
    }

    uint64_t hashState(
        const VuExecutionState &state)
    {
        uint64_t hash = kFnvOffset;
        hashValue(hash, state.vf);
        hashValue(hash, state.vi);
        hashValue(hash, state.acc);
        hashValue(hash, state.q);
        hashValue(hash, state.p);
        hashValue(hash, state.i);
        hashValue(hash, state.pc);
        hashValue(hash, state.mac);
        hashValue(hash, state.clip);
        hashValue(hash, state.status);
        hashValue(hash, state.ebit);
        hashValue(hash, state.top);
        hashValue(hash, state.itop);
        hashValue(hash, state.branchPending);
        hashValue(hash, state.branchTarget);
        hashValue(hash, state.branchDelay);
        hashValue(hash, state.viBackupCycles);
        hashValue(hash, state.viBackupRegister);
        hashValue(hash, state.viBackupValue);
        hashValue(hash, state.active);
        hashValue(hash, state.issuedCycles);

        const VuPipelineState &pipeline =
            state.pipeline;
        hashValue(hash, pipeline.xgkick.active);
        hashValue(hash, pipeline.xgkick.address);
        hashValue(
            hash, pipeline.xgkick.tagBytesRemaining);
        hashValue(hash, pipeline.xgkick.cycleCredit);
        hashValue(hash, pipeline.xgkick.tagEop);
        const uint64_t packetSize =
            pipeline.xgkick.packet.size();
        hashValue(hash, packetSize);
        for (const uint8_t byte :
             pipeline.xgkick.packet)
        {
            hash ^= byte;
            hash *= kFnvPrime;
        }
        hashValue(hash, pipeline.delayedQ.active);
        hashValue(hash, pipeline.delayedQ.result);
        hashValue(
            hash, pipeline.delayedQ.cyclesRemaining);
        hashValue(hash, pipeline.delayedP.active);
        hashValue(hash, pipeline.delayedP.result);
        hashValue(
            hash, pipeline.delayedP.cyclesRemaining);
        for (const auto &stage :
             pipeline.fmacFlags)
        {
            hashValue(hash, stage.active);
            hashValue(hash, stage.mac);
            hashValue(hash, stage.status);
        }
        hashValue(hash, pipeline.fmacFlagIndex);
        hashValue(hash, pipeline.workingMac);
        return hash;
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
            uint8_t, PS2_VU1_DATA_SIZE> &initialData,
        const BufferedPath1Sink &expectedEffects)
    {
        for (Invocation &invocation : invocations)
        {
            invocation.state = initialState;
            std::memcpy(
                invocation.data.data(),
                initialData.data(),
                initialData.size());
            invocation.effects.reset();
            invocation.effects.reserveLike(
                expectedEffects);
        }
    }

    struct Measurement
    {
        uint64_t hostNanoseconds = 0u;
        uint64_t timedPlusDrainNanoseconds = 0u;
        uint64_t drainNanoseconds = 0u;
        uint64_t validationNanoseconds = 0u;
        uint64_t pairs = 0u;
        uint64_t cycles = 0u;
        uint64_t allocations = 0u;
        uint64_t allocationBytes = 0u;
        uint64_t drainAllocations = 0u;
        uint64_t drainAllocationBytes = 0u;
        uint64_t packets = 0u;
        uint64_t packetBytes = 0u;
        uint64_t stateHash = 0u;
        uint64_t dataHash = 0u;
        uint64_t path1Hash = 0u;
        uint64_t samplePath1Hash = 0u;
        VuExitReason exitReason =
            VuExitReason::Inactive;
        bool blockLinkingEnabled = false;
        bool blockBudgetGuardsEnabled = false;
        bool blockLocalVfRegistersEnabled = false;
        bool blockLocalVfRegistersAutomatic = false;
        bool inlineXgkickEnabled = false;
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
        TimingScope scope,
        IVuExecutionBackend &backend,
        VuRecompilerBackend *native,
        VuUnit *nativeUnit,
        ProductionPath1Transfer &path1Transfer,
        PerfStatControl &perfControl,
        std::vector<Invocation> &invocations,
        const VuExecutionState &initialState,
        const std::array<
            uint8_t, PS2_VU1_DATA_SIZE> &initialData,
        const VuExecutionState &expectedState,
        const std::array<
            uint8_t, PS2_VU1_DATA_SIZE> &expectedData,
        const BufferedPath1Sink &expectedEffects,
        const uint8_t *code, PS2Memory &memory,
        uint32_t maximumPairs,
        uint32_t expectedPairs,
        VuExitReason expectedReason,
        bool churnCache)
    {
        Measurement result;
        result.blockLinkingEnabled =
            native &&
            native->blockLinkingEnabled();
        result.blockBudgetGuardsEnabled =
            native &&
            native->blockBudgetGuardsEnabled();
        result.blockLocalVfRegistersEnabled =
            native &&
            native->
                blockLocalVfRegistersEnabled();
        result.blockLocalVfRegistersAutomatic =
            native &&
            native->
                blockLocalVfRegistersAutomatic();
        result.inlineXgkickEnabled =
            native &&
            native->inlineXgkickEnabled();
        prepareInvocations(
            invocations, initialState, initialData,
            expectedEffects);
        memory.setGifArbiter(nullptr);
        BufferedPath1Sink drainedEffects;
        if (scope == TimingScope::VuPlusPath1)
        {
            if (!drainedEffects.reserveRepeated(
                    expectedEffects,
                    invocations.size()))
            {
                std::cerr
                    << "PATH1 validation buffer is too large\n";
                result.valid = false;
                return result;
            }
            path1Transfer.begin(drainedEffects);
        }
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
        if (!perfControl.enable())
        {
            if (scope == TimingScope::VuPlusPath1)
                path1Transfer.end();
            std::cerr
                << "failed to enable perf counters: "
                << perfControl.error() << "\n";
            result.valid = false;
            return result;
        }
        g_trackAllocations.store(
            true, std::memory_order_release);
        const auto started = Clock::now();
        bool firstRun = true;
        for (Invocation &invocation : invocations)
        {
            if (churnCache)
            {
                if (!nativeUnit)
                {
                    result.valid = false;
                    break;
                }
                nativeUnit->programCache().flush();
            }
            IVuSideEffectSink *sideEffects =
                scope == TimingScope::VuPlusPath1
                    ? static_cast<IVuSideEffectSink *>(
                          &path1Transfer)
                    : static_cast<IVuSideEffectSink *>(
                          &invocation.effects);
            VuExecutionContext context{
                .state = invocation.state,
                .code = code,
                .codeSize = PS2_VU1_CODE_SIZE,
                .data = invocation.data.data(),
                .dataSize = PS2_VU1_DATA_SIZE,
                .sideEffects = *sideEffects,
                .memory = &memory,
                .enableInstrumentation = false,
            };
            const RunSummary run =
                runBackend(
                    backend, context, maximumPairs);
            if (firstRun)
            {
                result.exitReason = run.reason;
                firstRun = false;
            }
            else if (run.reason != result.exitReason)
            {
                result.valid = false;
            }
            result.pairs += run.pairs;
            result.cycles += run.pairs;
            if (scope == TimingScope::VuCore)
            {
                result.packets +=
                    invocation.effects.packetCount();
                result.packetBytes +=
                    invocation.effects.byteCount();
            }
            if (!run.valid ||
                run.pairs != expectedPairs ||
                run.reason != expectedReason)
            {
                result.valid = false;
            }
        }
        const auto stopped = Clock::now();
        g_trackAllocations.store(
            false, std::memory_order_release);
        if (!perfControl.disable())
        {
            std::cerr
                << "failed to disable perf counters: "
                << perfControl.error() << "\n";
            result.valid = false;
        }

        result.hostNanoseconds =
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
        if (scope == TimingScope::VuPlusPath1)
        {
            g_allocationCount.store(
                0u, std::memory_order_relaxed);
            g_allocationBytes.store(
                0u, std::memory_order_relaxed);
            g_trackAllocations.store(
                true, std::memory_order_release);
            const auto drainStarted = Clock::now();
            path1Transfer.drain();
            const auto drainStopped = Clock::now();
            g_trackAllocations.store(
                false, std::memory_order_release);
            result.drainNanoseconds =
                static_cast<uint64_t>(
                    std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                        drainStopped - drainStarted)
                        .count());
            result.drainAllocations =
                g_allocationCount.load(
                    std::memory_order_relaxed);
            result.drainAllocationBytes =
                g_allocationBytes.load(
                    std::memory_order_relaxed);
            result.packets =
                drainedEffects.packetCount();
            result.packetBytes =
                drainedEffects.byteCount();
            if (!path1Transfer.empty())
                result.valid = false;
            path1Transfer.end();
        }
        result.timedPlusDrainNanoseconds =
            result.hostNanoseconds +
            result.drainNanoseconds;

        const auto validationStarted = Clock::now();
        bool firstInvocation = true;
        for (const Invocation &invocation : invocations)
        {
            const uint64_t stateHash =
                hashState(invocation.state);
            const uint64_t dataHash =
                hashBytes(
                    invocation.data.data(),
                    invocation.data.size());
            if (firstInvocation)
            {
                result.stateHash = stateHash;
                result.dataHash = dataHash;
                firstInvocation = false;
            }
            else if (
                stateHash != result.stateHash ||
                dataHash != result.dataHash)
            {
                result.valid = false;
            }
            std::string differenceField;
            if (!vuExecutionStatesEqual(
                    invocation.state, expectedState,
                    &differenceField) ||
                invocation.data != expectedData ||
                (scope == TimingScope::VuCore &&
                 !(invocation.effects ==
                   expectedEffects)))
            {
                result.valid = false;
                break;
            }
        }
        result.path1Hash = expectedEffects.hash();
        if (scope == TimingScope::VuCore)
        {
            result.samplePath1Hash =
                invocations.front()
                    .effects.hashRepeated(
                        invocations.size());
        }
        else
        {
            result.samplePath1Hash =
                drainedEffects.hash();
            if (!drainedEffects.equalsRepeated(
                    expectedEffects,
                    invocations.size()))
            {
                result.valid = false;
            }
        }
        const auto validationStopped = Clock::now();
        result.validationNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    validationStopped -
                    validationStarted)
                    .count());

        if (native)
        {
            const VuRecompilerDiagnostics after =
                native->diagnostics();
            result.nativeDelta.nativeEntries =
                difference(
                    after.nativeEntries,
                    nativeBefore.nativeEntries);
            result.nativeDelta.nativeBlocks =
                difference(
                    after.nativeBlocks,
                    nativeBefore.nativeBlocks);
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
            result.nativeDelta.linkedEdges =
                difference(
                    after.linkedEdges,
                    nativeBefore.linkedEdges);
            result.nativeDelta.slowLinkExits =
                difference(
                    after.slowLinkExits,
                    nativeBefore.slowLinkExits);
            result.nativeDelta.resolvedLinks =
                difference(
                    after.resolvedLinks,
                    nativeBefore.resolvedLinks);
            result.nativeDelta.abandonedLinkResolutions =
                difference(
                    after.abandonedLinkResolutions,
                    nativeBefore.
                        abandonedLinkResolutions);
            result.nativeDelta.incompatibleLinkExits =
                difference(
                    after.incompatibleLinkExits,
                    nativeBefore.incompatibleLinkExits);
            result.nativeDelta.fullBlockGuards =
                difference(
                    after.fullBlockGuards,
                    nativeBefore.fullBlockGuards);
            result.nativeDelta.preciseTailEntries =
                difference(
                    after.preciseTailEntries,
                    nativeBefore.preciseTailEntries);
            result.nativeDelta.fullGuardPairs =
                difference(
                    after.fullGuardPairs,
                    nativeBefore.fullGuardPairs);
            result.nativeDelta.preciseTailPairs =
                difference(
                    after.preciseTailPairs,
                    nativeBefore.preciseTailPairs);
            result.nativeDelta.budgetGuardComparisons =
                difference(
                    after.budgetGuardComparisons,
                    nativeBefore.
                        budgetGuardComparisons);
            result.nativeDelta.nativeBudgetExits =
                difference(
                    after.nativeBudgetExits,
                    nativeBefore.nativeBudgetExits);
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
            for (size_t exit = 0u;
                 exit <
                     result.nativeDelta.
                         nativeExitReasons.size();
                 ++exit)
            {
                result.nativeDelta.
                    nativeExitReasons[exit] =
                        difference(
                            after.nativeExitReasons[exit],
                            nativeBefore.
                                nativeExitReasons[exit]);
            }
            result.nativeDelta.interpreterFallbackPairs =
                difference(
                    after.interpreterFallbackPairs,
                    nativeBefore.interpreterFallbackPairs);
            result.valid =
                result.valid &&
                result.nativeDelta.fullBlockGuards +
                        result.nativeDelta.preciseTailEntries ==
                    result.nativeDelta.nativeBlocks &&
                result.nativeDelta.fullGuardPairs +
                        result.nativeDelta.preciseTailPairs ==
                    result.nativeDelta.nativePairs;
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
            result.cacheDelta.invalidations =
                difference(
                    after.invalidations,
                    cacheBefore.invalidations);
            result.cacheDelta.invalidatedPrograms =
                difference(
                    after.invalidatedPrograms,
                    cacheBefore.invalidatedPrograms);
            result.cacheDelta.generationRetentions =
                difference(
                    after.generationRetentions,
                    cacheBefore.generationRetentions);
            result.cacheDelta.retainedPrograms =
                difference(
                    after.retainedPrograms,
                    cacheBefore.retainedPrograms);
            result.cacheDelta.crossGenerationHits =
                difference(
                    after.crossGenerationHits,
                    cacheBefore.crossGenerationHits);
            result.cacheDelta.evictionFlushes =
                difference(
                    after.evictionFlushes,
                    cacheBefore.evictionFlushes);
            result.cacheDelta.evictedPrograms =
                difference(
                    after.evictedPrograms,
                    cacheBefore.evictedPrograms);
            result.cacheDelta.manualFlushes =
                difference(
                    after.manualFlushes,
                    cacheBefore.manualFlushes);
            result.cacheDelta.rejectedPrograms =
                difference(
                    after.rejectedPrograms,
                    cacheBefore.rejectedPrograms);
            result.cacheDelta.linkResolutions =
                difference(
                    after.linkResolutions,
                    cacheBefore.linkResolutions);
            result.cacheDelta.linkResolutionFailures =
                difference(
                    after.linkResolutionFailures,
                    cacheBefore.linkResolutionFailures);
            result.cacheDelta.linkInvalidations =
                difference(
                    after.linkInvalidations,
                    cacheBefore.linkInvalidations);
            result.cacheDelta.generatedBytes =
                difference(
                    after.generatedBytes,
                    cacheBefore.generatedBytes);
            result.cacheDelta.compilationNanoseconds =
                difference(
                    after.compilationNanoseconds,
                    cacheBefore.compilationNanoseconds);
            result.cacheDelta.residentPrograms =
                after.residentPrograms;
            result.cacheDelta.residentExecutableBytes =
                after.residentExecutableBytes;
            result.cacheDelta.highWaterPrograms =
                after.highWaterPrograms;
            result.cacheDelta.highWaterExecutableBytes =
                after.highWaterExecutableBytes;
            if (churnCache)
            {
                result.valid =
                    result.valid &&
                    result.cacheDelta.manualFlushes ==
                        invocations.size() &&
                    result.cacheDelta.compilations != 0u;
            }
            else
            {
                result.valid =
                    result.valid &&
                    result.cacheDelta.misses == 0u &&
                    result.cacheDelta.compilations == 0u;
            }
        }
        return result;
    }

    uint64_t median(std::vector<uint64_t> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    }

    void printMeasurement(
        std::string_view event,
        TimingScope scope,
        std::string_view backend, uint32_t sample,
        uint32_t invocations,
        const Measurement &measurement)
    {
        const double pairsPerSecond =
            measurement.hostNanoseconds == 0u
                ? 0.0
                : static_cast<double>(
                      measurement.pairs) *
                      1.0e9 /
                      static_cast<double>(
                          measurement.hostNanoseconds);
        const double nanosecondsPerPair =
            measurement.pairs == 0u
                ? 0.0
                : static_cast<double>(
                      measurement.hostNanoseconds) /
                      static_cast<double>(
                          measurement.pairs);
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"" << event << "\""
            << ",\"backend\":\"" << backend << "\""
            << ",\"timing_scope\":\""
            << timingScopeName(scope) << "\""
            << ",\"sample\":" << sample
            << ",\"invocations\":" << invocations
            << ",\"guest_pairs\":"
            << measurement.pairs
            << ",\"guest_cycles\":"
            << measurement.cycles
            << ",\"exit_reason\":\""
            << vuExitReasonName(
                   measurement.exitReason)
            << "\""
            << ",\"host_nanoseconds\":"
            << measurement.hostNanoseconds
            << ",\"timed_plus_final_drain_nanoseconds\":"
            << measurement.timedPlusDrainNanoseconds
            << ",\"final_drain_nanoseconds\":"
            << measurement.drainNanoseconds
            << ",\"validation_nanoseconds\":"
            << measurement.validationNanoseconds
            << ",\"guest_pairs_per_second\":"
            << std::fixed << std::setprecision(3)
            << pairsPerSecond
            << ",\"host_nanoseconds_per_guest_pair\":"
            << nanosecondsPerPair
            << ",\"allocation_count\":"
            << measurement.allocations
            << ",\"allocation_bytes\":"
            << measurement.allocationBytes
            << ",\"execution_allocation_count\":"
            << measurement.allocations
            << ",\"execution_allocation_bytes\":"
            << measurement.allocationBytes
            << ",\"final_drain_allocation_count\":"
            << measurement.drainAllocations
            << ",\"final_drain_allocation_bytes\":"
            << measurement.drainAllocationBytes
            << ",\"path1_packets\":"
            << measurement.packets
            << ",\"path1_bytes\":"
            << measurement.packetBytes
            << ",\"state_fnv1a64\":\"0x"
            << std::hex << std::setw(16)
            << std::setfill('0')
            << measurement.stateHash
            << "\",\"vu_data_fnv1a64\":\"0x"
            << std::setw(16)
            << measurement.dataHash
            << "\",\"path1_fnv1a64\":\"0x"
            << std::setw(16)
            << measurement.path1Hash
            << "\",\"path1_sample_fnv1a64\":\"0x"
            << std::setw(16)
            << measurement.samplePath1Hash
            << "\"" << std::dec << std::setfill(' ')
            << ",\"valid\":"
            << (measurement.valid
                    ? "true" : "false");
        if (backend == "recompiler")
        {
            std::cout
                << ",\"block_linking_enabled\":"
                << (measurement.blockLinkingEnabled
                        ? "true" : "false")
                << ",\"block_budget_guards_enabled\":"
                << (measurement.blockBudgetGuardsEnabled
                        ? "true" : "false")
                << ",\"block_local_vf_registers_enabled\":"
                << (measurement.
                            blockLocalVfRegistersEnabled
                        ? "true" : "false")
                << ",\"block_local_vf_registers_automatic\":"
                << (measurement.
                            blockLocalVfRegistersAutomatic
                        ? "true" : "false")
                << ",\"inline_xgkick_enabled\":"
                << (measurement.inlineXgkickEnabled
                        ? "true" : "false")
                << ",\"native_entries\":"
                << measurement.nativeDelta.nativeEntries
                << ",\"native_blocks\":"
                << measurement.nativeDelta.nativeBlocks
                << ",\"native_pairs\":"
                << measurement.nativeDelta.nativePairs
                << ",\"inline_pairs\":"
                << measurement.nativeDelta.inlinePairs
                << ",\"helper_pairs\":"
                << measurement.nativeDelta.helperPairs
                << ",\"linked_edges\":"
                << measurement.nativeDelta.linkedEdges
                << ",\"slow_link_exits\":"
                << measurement.nativeDelta.slowLinkExits
                << ",\"resolved_links\":"
                << measurement.nativeDelta.resolvedLinks
                << ",\"abandoned_link_resolutions\":"
                << measurement.nativeDelta
                       .abandonedLinkResolutions
                << ",\"incompatible_link_exits\":"
                << measurement.nativeDelta
                       .incompatibleLinkExits
                << ",\"full_block_guards\":"
                << measurement.nativeDelta
                       .fullBlockGuards
                << ",\"precise_tail_entries\":"
                << measurement.nativeDelta
                       .preciseTailEntries
                << ",\"full_guard_pairs\":"
                << measurement.nativeDelta
                       .fullGuardPairs
                << ",\"precise_tail_pairs\":"
                << measurement.nativeDelta
                       .preciseTailPairs
                << ",\"budget_guard_comparisons\":"
                << measurement.nativeDelta
                       .budgetGuardComparisons
                << ",\"native_budget_exits\":"
                << measurement.nativeDelta
                       .nativeBudgetExits
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
                << measurement.nativeDelta.faultExits;
            std::cout
                << ",\"native_exit_reasons\":{";
            for (size_t exit = 0u;
                 exit <
                     measurement.nativeDelta.
                         nativeExitReasons.size();
                 ++exit)
            {
                if (exit != 0u)
                    std::cout << ",";
                std::cout
                    << "\""
                    << vuNativeBlockExitName(
                           static_cast<
                               VuNativeBlockExit>(
                               exit))
                    << "\":"
                    << measurement.nativeDelta.
                           nativeExitReasons[exit];
            }
            std::cout
                << "}"
                << ",\"interpreter_fallback_pairs\":"
                << measurement.nativeDelta
                       .interpreterFallbackPairs
                << ",\"cache_hits\":"
                << measurement.cacheDelta.hits
                << ",\"cache_misses\":"
                << measurement.cacheDelta.misses
                << ",\"cache_compilations\":"
                << measurement.cacheDelta.compilations
                << ",\"cache_invalidations\":"
                << measurement.cacheDelta.invalidations
                << ",\"cache_invalidated_programs\":"
                << measurement.cacheDelta
                       .invalidatedPrograms
                << ",\"cache_generation_retentions\":"
                << measurement.cacheDelta
                       .generationRetentions
                << ",\"cache_retained_programs\":"
                << measurement.cacheDelta.retainedPrograms
                << ",\"cache_cross_generation_hits\":"
                << measurement.cacheDelta
                       .crossGenerationHits
                << ",\"cache_eviction_flushes\":"
                << measurement.cacheDelta.evictionFlushes
                << ",\"cache_evicted_programs\":"
                << measurement.cacheDelta.evictedPrograms
                << ",\"cache_manual_flushes\":"
                << measurement.cacheDelta.manualFlushes
                << ",\"cache_rejected_programs\":"
                << measurement.cacheDelta.rejectedPrograms
                << ",\"cache_link_resolutions\":"
                << measurement.cacheDelta.linkResolutions
                << ",\"cache_link_resolution_failures\":"
                << measurement.cacheDelta
                       .linkResolutionFailures
                << ",\"cache_link_invalidations\":"
                << measurement.cacheDelta.linkInvalidations
                << ",\"cache_generated_bytes\":"
                << measurement.cacheDelta.generatedBytes
                << ",\"cache_compilation_nanoseconds\":"
                << measurement.cacheDelta
                       .compilationNanoseconds
                << ",\"cache_resident_programs\":"
                << measurement.cacheDelta.residentPrograms
                << ",\"cache_resident_executable_bytes\":"
                << measurement.cacheDelta
                       .residentExecutableBytes
                << ",\"cache_high_water_programs\":"
                << measurement.cacheDelta.highWaterPrograms
                << ",\"cache_high_water_executable_bytes\":"
                << measurement.cacheDelta
                       .highWaterExecutableBytes;
        }
        std::cout << "}\n";
    }

    void printBlockProfiles(
        const VuBlockProfilingSnapshot &profile,
        const VuRecompilerDiagnostics &diagnostics)
    {
        if (!profile.enabled)
            return;
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"block-profile-summary\""
            << ",\"maximum_records\":"
            << profile.maximumRecords
            << ",\"dropped_records\":"
            << profile.droppedRecords
            << ",\"record_count\":"
            << profile.blocks.size()
            << ",\"backend_cumulative_native_entries\":"
            << diagnostics.nativeEntries
            << ",\"backend_cumulative_native_blocks\":"
            << diagnostics.nativeBlocks
            << ",\"backend_cumulative_native_pairs\":"
            << diagnostics.nativePairs
            << ",\"backend_cumulative_helper_pairs\":"
            << diagnostics.helperPairs
            << ",\"backend_cumulative_linked_edges\":"
            << diagnostics.linkedEdges
            << ",\"backend_cumulative_slow_link_exits\":"
            << diagnostics.slowLinkExits
            << ",\"backend_cumulative_full_block_guards\":"
            << diagnostics.fullBlockGuards
            << ",\"backend_cumulative_precise_tail_entries\":"
            << diagnostics.preciseTailEntries
            << ",\"backend_cumulative_full_guard_pairs\":"
            << diagnostics.fullGuardPairs
            << ",\"backend_cumulative_precise_tail_pairs\":"
            << diagnostics.preciseTailPairs
            << ",\"backend_cumulative_budget_guard_comparisons\":"
            << diagnostics.budgetGuardComparisons
            << ",\"jitdump_registrations\":"
            << diagnostics.jitDumpRegistrations
            << ",\"jitdump_failures\":"
            << diagnostics.jitDumpFailures
            << "}\n";

        for (const VuBlockProfileSnapshot &block :
             profile.blocks)
        {
            std::cout
                << "{\"schema_version\":2"
                << ",\"event\":\"block-profile\""
                << ",\"unit\":\""
                << (block.key.unit == VuUnitId::Vu0
                        ? "vu0" : "vu1")
                << "\""
                << ",\"compilation_id\":"
                << block.compilationIdentity
                << ",\"code_content_identity\":"
                << block.key.codeContentIdentity
                << ",\"compilation_generation\":"
                << block.key.codeGeneration
                << ",\"native_address\":\"0x"
                << std::hex << std::setw(16)
                << std::setfill('0')
                << block.nativeAddress
                << "\",\"native_bytes\":"
                << std::dec << std::setfill(' ')
                << block.nativeBytes
                << ",\"entry_pc\":\"0x"
                << std::hex << std::setw(4)
                << std::setfill('0')
                << block.key.entryPc
                << "\",\"first_pc\":\"0x"
                << std::setw(4) << block.firstPc
                << "\",\"last_pc\":\"0x"
                << std::setw(4) << block.lastPc
                << "\"" << std::dec
                << std::setfill(' ')
                << ",\"block_pairs\":"
                << block.blockPairs
                << ",\"fixed_cycles\":"
                << block.fixedCycles
                << ",\"compilation_mode\":\""
                << (block.key.compilationMode ==
                            VuCompilationMode::Instrumented
                        ? "instrumented" : "normal")
                << "\""
                << ",\"block_form\":\""
                << (block.key.blockForm ==
                            VuIrBlockForm::Basic
                        ? "basic" : "linear-trace")
                << "\""
                << ",\"block_exit\":\""
                << vuIrBlockExitName(block.blockExit)
                << "\""
                << ",\"resident\":"
                << (block.resident
                        ? "true" : "false")
                << ",\"executions\":"
                << block.executions
                << ",\"guest_pairs\":"
                << block.guestPairs
                << ",\"full_budget_entries\":"
                << block.fullBudgetEntries
                << ",\"bounded_entries\":"
                << block.boundedEntries
                << ",\"linked_edges\":"
                << block.linkedEdges
                << ",\"helper_barriers\":"
                << block.helperBarriers
                << ",\"resident_vf_registers\":"
                << block.residentVfRegisters
                << ",\"resident_vf_dirty_registers\":"
                << block.residentVfDirtyRegisters
                << ",\"resident_vf_dirty_lanes\":"
                << block.residentVfDirtyLanes
                << ",\"maximum_live_vf_registers\":"
                << block.maximumLiveVfRegisters
                << ",\"vf_accesses\":"
                << block.vfAccesses
                << ",\"allocated_vf_accesses\":"
                << block.allocatedVfAccesses
                << ",\"vf_register_loads\":"
                << block.vfRegisterLoads
                << ",\"vf_register_stores\":"
                << block.vfRegisterStores
                << ",\"vf_register_spills\":"
                << block.vfRegisterSpills
                << ",\"vf_register_reloads\":"
                << block.vfRegisterReloads
                << ",\"register_materializations\":{";
            for (size_t cause = 0u;
                 cause <
                     block.registerMaterializations.
                         size();
                 ++cause)
            {
                if (cause != 0u)
                    std::cout << ",";
                std::cout
                    << "\""
                    << vuRegisterMaterializationCauseName(
                           static_cast<
                               VuRegisterMaterializationCause>(
                               cause))
                    << "\":"
                    << block.
                           registerMaterializations[
                               cause];
            }
            std::cout
                << "},\"exit_reasons\":{";
            for (size_t exit = 0u;
                 exit < block.exitReasons.size();
                 ++exit)
            {
                if (exit != 0u)
                    std::cout << ",";
                std::cout
                    << "\""
                    << vuNativeBlockExitName(
                           static_cast<
                               VuNativeBlockExit>(
                               exit))
                    << "\":"
                    << block.exitReasons[exit];
            }
            std::cout << "},\"opcodes\":[";
            bool firstOpcode = true;
            for (size_t opcode = 0u;
                 opcode <
                     block.opcodeOperations.size();
                 ++opcode)
            {
                const uint64_t operations =
                    block.opcodeOperations[opcode];
                if (operations == 0u)
                    continue;
                if (!firstOpcode)
                    std::cout << ",";
                firstOpcode = false;
                std::cout
                    << "{\"name\":\""
                    << vuIrOpcodeName(
                           static_cast<VuIrOpcode>(
                               opcode))
                    << "\",\"operations\":"
                    << operations << "}";
            }
            std::cout << "],\"jit_registrations\":[";
            for (size_t registration = 0u;
                 registration <
                     block.jitRegistrations.size();
                 ++registration)
            {
                if (registration != 0u)
                    std::cout << ",";
                const auto &source =
                    block.jitRegistrations[
                        registration];
                std::cout
                    << "{\"generation\":"
                    << source.generation
                    << ",\"code_index\":"
                    << source.codeIndex
                    << "}";
            }
            std::cout << "]}\n";
        }
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
               "[--samples N] "
               "[--cache-churn-iterations N] "
               "[--analysis-check on|off] "
               "[--budget-sweep on|off] "
               "[--scope core|path1|all] "
               "[--backend interpreter|recompiler|both]\n";
        return 2;
    }
    if (!VuRecompilerBackend::supported())
    {
        std::cerr
            << "the x86-64 VU recompiler is unavailable\n";
        return 2;
    }
    if (configuration.cacheChurnIterations != 0u &&
        !configuration.measureRecompiler)
    {
        std::cerr
            << "cache churn requires the recompiler backend\n";
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

    if (configuration.analysisCheck)
    {
        const AnalysisCheckSummary analysis =
            validateAnalysisTrace(
                memory.getVU1Code(),
                PS2_VU1_CODE_SIZE,
                initialData.data(),
                PS2_VU1_DATA_SIZE,
                initialState,
                memory,
                configuration.maximumPairs);
        if (!analysis.valid)
        {
            std::cerr
                << "VU analysis trace failed: "
                << analysis.diagnostic << "\n";
            return 1;
        }
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"analysis-check\""
            << ",\"fixture\":\""
            << configuration.fixtureDirectory.
                   filename().string()
            << "\""
            << ",\"valid\":true"
            << ",\"bounded\":"
            << (analysis.bounded ? "true" : "false")
            << ",\"analyzed_blocks\":"
            << analysis.analyzedBlocks
            << ",\"executed_blocks\":"
            << analysis.executedBlocks
            << ",\"pairs\":" << analysis.pairs
            << ",\"cycles\":" << analysis.cycles
            << ",\"static_edges\":"
            << analysis.staticEdges
            << ",\"dynamic_edges\":"
            << analysis.dynamicEdges
            << ",\"xgkick_boundaries\":"
            << analysis.xgkickBoundaries
            << ",\"helper_pairs\":"
            << analysis.helperPairs
            << ",\"vf_read_registers\":"
            << analysis.vfReadRegisters
            << ",\"vf_write_registers\":"
            << analysis.vfWriteRegisters
            << ",\"vi_read_registers\":"
            << analysis.viReadRegisters
            << ",\"vi_write_registers\":"
            << analysis.viWriteRegisters
            << ",\"vf_live_in_registers\":"
            << analysis.vfLiveInRegisters
            << ",\"vi_live_in_registers\":"
            << analysis.viLiveInRegisters
            << ",\"block_vf_working_set\":"
            << analysis.blockVfWorkingSet
            << ",\"block_vi_working_set\":"
            << analysis.blockViWorkingSet
            << ",\"block_vf_accesses\":"
            << analysis.blockVfAccesses
            << ",\"top3_block_vf_accesses\":"
            << analysis.top3BlockVfAccesses
            << ",\"top8_block_vf_accesses\":"
            << analysis.top8BlockVfAccesses
            << ",\"top13_block_vf_accesses\":"
            << analysis.top13BlockVfAccesses
            << ",\"blocks_fitting_3_vf_registers\":"
            << analysis.blocksFitting3VfRegisters
            << ",\"blocks_fitting_8_vf_registers\":"
            << analysis.blocksFitting8VfRegisters
            << ",\"blocks_fitting_13_vf_registers\":"
            << analysis.blocksFitting13VfRegisters
            << ",\"maximum_vf_live_in_registers\":"
            << analysis.maximumVfLiveInRegisters
            << ",\"maximum_vi_live_in_registers\":"
            << analysis.maximumViLiveInRegisters
            << ",\"maximum_block_vf_working_set\":"
            << analysis.maximumBlockVfWorkingSet
            << ",\"maximum_block_vi_working_set\":"
            << analysis.maximumBlockViWorkingSet
            << ",\"vf_reads\":[";
        for (size_t reg = 0u;
             reg < analysis.vfReads.size(); ++reg)
        {
            if (reg != 0u)
                std::cout << ",";
            std::cout << analysis.vfReads[reg];
        }
        std::cout << "],\"vf_writes\":[";
        for (size_t reg = 0u;
             reg < analysis.vfWrites.size(); ++reg)
        {
            if (reg != 0u)
                std::cout << ",";
            std::cout << analysis.vfWrites[reg];
        }
        std::cout << "],\"vi_reads\":[";
        for (size_t reg = 0u;
             reg < analysis.viReads.size(); ++reg)
        {
            if (reg != 0u)
                std::cout << ",";
            std::cout << analysis.viReads[reg];
        }
        std::cout << "],\"vi_writes\":[";
        for (size_t reg = 0u;
             reg < analysis.viWrites.size(); ++reg)
        {
            if (reg != 0u)
                std::cout << ",";
            std::cout << analysis.viWrites[reg];
        }
        std::cout
            << "]"
            << ",\"exit\":\""
            << vuExitReasonName(analysis.reason)
            << "\"}\n";
    }

    if (configuration.budgetSweep)
    {
        const BudgetSweepSummary sweep =
            validateBudgetSweep(
                memory.getVU1Code(),
                initialState,
                initialData,
                memory,
                configuration.maximumPairs);
        if (!sweep.valid)
        {
            std::cerr
                << "VU budget sweep failed: "
                << sweep.diagnostic << "\n";
            return 1;
        }
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"budget-sweep\""
            << ",\"fixture\":\""
            << configuration.fixtureDirectory.
                   filename().string()
            << "\""
            << ",\"valid\":true"
            << ",\"budgets\":" << sweep.budgets
            << ",\"guest_pairs\":"
            << sweep.pairs
            << ",\"native_blocks\":"
            << sweep.diagnostics.nativeBlocks
            << ",\"native_pairs\":"
            << sweep.diagnostics.nativePairs
            << ",\"full_block_guards\":"
            << sweep.diagnostics.fullBlockGuards
            << ",\"precise_tail_entries\":"
            << sweep.diagnostics.preciseTailEntries
            << ",\"full_guard_pairs\":"
            << sweep.diagnostics.fullGuardPairs
            << ",\"precise_tail_pairs\":"
            << sweep.diagnostics.preciseTailPairs
            << ",\"budget_guard_comparisons\":"
            << sweep.diagnostics.
                   budgetGuardComparisons
            << ",\"native_budget_exits\":"
            << sweep.diagnostics.nativeBudgetExits
            << ",\"fault_exits\":"
            << sweep.diagnostics.faultExits
            << "}\n";
    }

    VuUnit interpreterUnit(VuUnitId::Vu1);
    VuUnit nativeUnit(VuUnitId::Vu1);
    VuInterpreterBackend interpreter(interpreterUnit);
    VuRecompilerBackend native(nativeUnit);
    ProductionPath1Transfer path1Transfer(memory);

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
    const auto coldValidationStarted = Clock::now();
    std::string stateDifference;
    const bool coldMatches =
        coldRun.valid &&
        coldRun.pairs == referenceRun.pairs &&
        vuExecutionStatesEqual(
            cold.state, reference.state,
            &stateDifference) &&
        cold.data == reference.data &&
        cold.effects == reference.effects;
    const uint64_t coldStateHash =
        hashState(cold.state);
    const uint64_t coldDataHash =
        hashBytes(cold.data.data(), cold.data.size());
    const uint64_t coldPath1Hash =
        cold.effects.hash();
    const auto coldValidationStopped = Clock::now();
    const uint64_t coldValidationNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                coldValidationStopped -
                coldValidationStarted)
                .count());
    const VuProgramCacheDiagnostics coldCache =
        nativeUnit.programCache().diagnostics();
    std::cout
        << "{\"schema_version\":2"
        << ",\"event\":\"cold-recompiler\""
        << ",\"timing_scope\":\"vu-core\""
        << ",\"guest_pairs\":" << coldRun.pairs
        << ",\"guest_cycles\":" << coldRun.pairs
        << ",\"exit_reason\":\""
        << vuExitReasonName(coldRun.reason)
        << "\""
        << ",\"host_nanoseconds\":"
        << coldNanoseconds
        << ",\"validation_nanoseconds\":"
        << coldValidationNanoseconds
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
        << ",\"generation_retentions\":"
        << coldCache.generationRetentions
        << ",\"cross_generation_hits\":"
        << coldCache.crossGenerationHits
        << ",\"resident_programs\":"
        << coldCache.residentPrograms
        << ",\"resident_executable_bytes\":"
        << coldCache.residentExecutableBytes
        << ",\"path1_packets\":"
        << cold.effects.packetCount()
        << ",\"path1_bytes\":"
        << cold.effects.byteCount()
        << ",\"state_fnv1a64\":\"0x"
        << std::hex << std::setw(16)
        << std::setfill('0') << coldStateHash
        << "\",\"vu_data_fnv1a64\":\"0x"
        << std::setw(16) << coldDataHash
        << "\",\"path1_fnv1a64\":\"0x"
        << std::setw(16) << coldPath1Hash
        << "\"" << std::dec << std::setfill(' ')
        << ",\"matches_interpreter\":"
        << (coldMatches ? "true" : "false")
        << "}\n";
    if (!coldMatches)
    {
        std::cerr
            << "cold recompiler mismatch: "
            << stateDifference;
        if (!native.lastDiagnostic().empty())
        {
            std::cerr
                << " (" << native.lastDiagnostic()
                << ")";
        }
        std::cerr << "\n";
        return 1;
    }

    PerfStatControl perfControl;
    if (!perfControl.valid())
    {
        std::cerr
            << "failed to configure perf counters: "
            << perfControl.error() << "\n";
        return 2;
    }

    Invocation warmup;
    warmup.effects.reserveLike(reference.effects);
    const auto runWarmup =
        [&](IVuExecutionBackend &backend,
            TimingScope scope)
    {
        warmup.state = initialState;
        warmup.data = initialData;
        warmup.effects.reset();
        IVuSideEffectSink *sideEffects =
            &warmup.effects;
        if (scope == TimingScope::VuPlusPath1)
        {
            path1Transfer.begin(warmup.effects);
            sideEffects = &path1Transfer;
        }
        VuExecutionContext context{
            .state = warmup.state,
            .code = memory.getVU1Code(),
            .codeSize = PS2_VU1_CODE_SIZE,
            .data = warmup.data.data(),
            .dataSize = PS2_VU1_DATA_SIZE,
            .sideEffects = *sideEffects,
            .memory = &memory,
            .enableInstrumentation = false,
        };
        const RunSummary run = runBackend(
            backend, context,
            configuration.maximumPairs);
        if (scope == TimingScope::VuPlusPath1)
        {
            path1Transfer.drain();
            const bool empty =
                path1Transfer.empty();
            path1Transfer.end();
            if (!empty)
                return false;
        }
        std::string differenceField;
        return
            run.valid &&
            run.pairs == referenceRun.pairs &&
            vuExecutionStatesEqual(
                warmup.state, reference.state,
                &differenceField) &&
            warmup.data == reference.data &&
            warmup.effects == reference.effects;
    };

    std::vector<TimingScope> scopes;
    if (configuration.measureVuCore)
        scopes.push_back(TimingScope::VuCore);
    if (configuration.measureVuPlusPath1)
        scopes.push_back(TimingScope::VuPlusPath1);
    for (uint32_t iteration = 0u;
         iteration < configuration.warmupIterations;
         ++iteration)
    {
        for (const TimingScope scope : scopes)
        {
            if (configuration.measureInterpreter &&
                !runWarmup(interpreter, scope))
            {
                std::cerr
                    << "interpreter warmup failed for "
                    << timingScopeName(scope) << "\n";
                return 1;
            }
            if (configuration.measureRecompiler &&
                !runWarmup(native, scope))
            {
                std::cerr
                    << "recompiler warmup failed for "
                    << timingScopeName(scope) << "\n";
                return 1;
            }
        }
    }
    native
        .resetBlockProfilingCountersWhileExecutionQuiescent();

    std::vector<Invocation> invocations(
        configuration.iterations);
    std::array<std::vector<uint64_t>, 2u>
        interpreterTimes;
    std::array<std::vector<uint64_t>, 2u>
        nativeTimes;
    bool allValid = true;
    for (uint32_t sample = 0u;
         sample < configuration.samples; ++sample)
    {
        const auto measureInterpreter = [&]
        {
            std::vector<TimingScope> orderedScopes =
                scopes;
            if ((sample & 1u) != 0u)
            {
                std::reverse(
                    orderedScopes.begin(),
                    orderedScopes.end());
            }
            for (const TimingScope scope :
                 orderedScopes)
            {
                const Measurement measurement =
                    measure(
                        scope, interpreter,
                        nullptr, nullptr,
                        path1Transfer,
                        perfControl,
                        invocations, initialState,
                        initialData, reference.state,
                        reference.data,
                        reference.effects,
                        memory.getVU1Code(), memory,
                        configuration.maximumPairs,
                        referenceRun.pairs,
                        referenceRun.reason,
                        false);
                printMeasurement(
                    "warm-sample", scope,
                    "interpreter", sample,
                    configuration.iterations,
                    measurement);
                interpreterTimes[
                    static_cast<size_t>(scope)]
                    .push_back(
                        measurement.hostNanoseconds);
                allValid =
                    allValid && measurement.valid;
            }
        };
        const auto measureRecompiler = [&]
        {
            std::vector<TimingScope> orderedScopes =
                scopes;
            if ((sample & 1u) != 0u)
            {
                std::reverse(
                    orderedScopes.begin(),
                    orderedScopes.end());
            }
            for (const TimingScope scope :
                 orderedScopes)
            {
                const Measurement measurement =
                    measure(
                        scope, native,
                        &native, &nativeUnit,
                        path1Transfer,
                        perfControl,
                        invocations, initialState,
                        initialData, reference.state,
                        reference.data,
                        reference.effects,
                        memory.getVU1Code(), memory,
                        configuration.maximumPairs,
                        referenceRun.pairs,
                        referenceRun.reason,
                        false);
                printMeasurement(
                    "warm-sample", scope,
                    "recompiler", sample,
                    configuration.iterations,
                    measurement);
                nativeTimes[
                    static_cast<size_t>(scope)]
                    .push_back(
                        measurement.hostNanoseconds);
                allValid =
                    allValid && measurement.valid;
            }
        };
        if (configuration.measureInterpreter &&
            configuration.measureRecompiler &&
            (sample & 1u) == 0u)
        {
            measureInterpreter();
            measureRecompiler();
        }
        else if (
            configuration.measureInterpreter &&
            configuration.measureRecompiler)
        {
            measureRecompiler();
            measureInterpreter();
        }
        else if (configuration.measureInterpreter)
        {
            measureInterpreter();
        }
        else
        {
            measureRecompiler();
        }
    }

    const bool warmAllValid = allValid;
    const VuProgramCacheDiagnostics warmCache =
        nativeUnit.programCache().diagnostics();
    std::array<std::vector<uint64_t>, 2u>
        cacheChurnTimes;
    std::array<bool, 2u> cacheChurnValid{
        true, true};
    if (configuration.cacheChurnIterations != 0u)
    {
        std::vector<Invocation> cacheChurnInvocations(
            configuration.cacheChurnIterations);
        for (uint32_t sample = 0u;
             sample < configuration.samples; ++sample)
        {
            std::vector<TimingScope> orderedScopes =
                scopes;
            if ((sample & 1u) != 0u)
            {
                std::reverse(
                    orderedScopes.begin(),
                    orderedScopes.end());
            }
            for (const TimingScope scope :
                 orderedScopes)
            {
                const Measurement measurement =
                    measure(
                        scope, native,
                        &native, &nativeUnit,
                        path1Transfer,
                        perfControl,
                        cacheChurnInvocations,
                        initialState, initialData,
                        reference.state,
                        reference.data,
                        reference.effects,
                        memory.getVU1Code(), memory,
                        configuration.maximumPairs,
                        referenceRun.pairs,
                        referenceRun.reason,
                        true);
                printMeasurement(
                    "cache-churn-sample", scope,
                    "recompiler", sample,
                    configuration.cacheChurnIterations,
                    measurement);
                const size_t scopeIndex =
                    static_cast<size_t>(scope);
                cacheChurnTimes[scopeIndex].push_back(
                    measurement.hostNanoseconds);
                cacheChurnValid[scopeIndex] =
                    cacheChurnValid[scopeIndex] &&
                    measurement.valid;
                allValid =
                    allValid && measurement.valid;
            }
        }
    }

    const VuProgramCacheDiagnostics finalCache =
        nativeUnit.programCache().diagnostics();
    for (const TimingScope scope : scopes)
    {
        const auto &scopeInterpreterTimes =
            interpreterTimes[
                static_cast<size_t>(scope)];
        const auto &scopeNativeTimes =
            nativeTimes[
                static_cast<size_t>(scope)];
        const bool hasInterpreter =
            !scopeInterpreterTimes.empty();
        const bool hasNative =
            !scopeNativeTimes.empty();
        const uint64_t interpreterMedian =
            hasInterpreter
                ? median(scopeInterpreterTimes)
                : 0u;
        const uint64_t nativeMedian =
            hasNative
                ? median(scopeNativeTimes)
                : 0u;
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"summary\""
            << ",\"timing_scope\":\""
            << timingScopeName(scope) << "\""
            << ",\"fixture\":\""
            << configuration.fixtureDirectory
                   .filename().string()
            << "\""
            << ",\"pairs_per_invocation\":"
            << referenceRun.pairs
            << ",\"cycles_per_invocation\":"
            << referenceRun.pairs
            << ",\"iterations_per_sample\":"
            << configuration.iterations
            << ",\"samples\":"
            << configuration.samples
            << ",\"interpreter_median_nanoseconds\":";
        if (hasInterpreter)
            std::cout << interpreterMedian;
        else
            std::cout << "null";
        std::cout
            << ",\"recompiler_median_nanoseconds\":";
        if (hasNative)
            std::cout << nativeMedian;
        else
            std::cout << "null";
        std::cout << ",\"speedup\":";
        if (hasInterpreter &&
            hasNative &&
            nativeMedian != 0u)
        {
            std::cout
                << std::fixed << std::setprecision(6)
                << static_cast<double>(
                       interpreterMedian) /
                       static_cast<double>(
                           nativeMedian);
        }
        else
        {
            std::cout << "null";
        }
        std::cout
            << ",\"cache_hits\":"
            << warmCache.hits
            << ",\"cache_misses\":"
            << warmCache.misses
            << ",\"compilations\":"
            << warmCache.compilations
            << ",\"generated_bytes\":"
            << warmCache.generatedBytes
            << ",\"high_water_executable_bytes\":"
            << warmCache.highWaterExecutableBytes
            << ",\"valid\":"
            << (warmAllValid ? "true" : "false")
            << "}\n";
    }
    for (const TimingScope scope : scopes)
    {
        const size_t scopeIndex =
            static_cast<size_t>(scope);
        const auto &scopeTimes =
            cacheChurnTimes[scopeIndex];
        if (scopeTimes.empty())
            continue;
        const uint64_t scopeMedian =
            median(scopeTimes);
        const uint64_t pairsPerSample =
            static_cast<uint64_t>(
                referenceRun.pairs) *
            configuration.cacheChurnIterations;
        const double nanosecondsPerPair =
            pairsPerSample == 0u
                ? 0.0
                : static_cast<double>(
                      scopeMedian) /
                      static_cast<double>(
                          pairsPerSample);
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"cache-churn-summary\""
            << ",\"timing_scope\":\""
            << timingScopeName(scope) << "\""
            << ",\"fixture\":\""
            << configuration.fixtureDirectory
                   .filename().string()
            << "\""
            << ",\"pairs_per_invocation\":"
            << referenceRun.pairs
            << ",\"cycles_per_invocation\":"
            << referenceRun.pairs
            << ",\"iterations_per_sample\":"
            << configuration.cacheChurnIterations
            << ",\"samples\":"
            << configuration.samples
            << ",\"recompiler_median_nanoseconds\":"
            << scopeMedian
            << ",\"host_nanoseconds_per_guest_pair\":"
            << std::fixed << std::setprecision(6)
            << nanosecondsPerPair
            << ",\"valid\":"
            << (cacheChurnValid[scopeIndex]
                    ? "true" : "false")
            << "}\n";
    }
    if (configuration.cacheChurnIterations != 0u)
    {
        std::cout
            << "{\"schema_version\":2"
            << ",\"event\":\"cache-churn-cache-summary\""
            << ",\"cache_hits\":"
            << difference(
                   finalCache.hits, warmCache.hits)
            << ",\"cache_misses\":"
            << difference(
                   finalCache.misses, warmCache.misses)
            << ",\"cache_compilations\":"
            << difference(
                   finalCache.compilations,
                   warmCache.compilations)
            << ",\"cache_manual_flushes\":"
            << difference(
                   finalCache.manualFlushes,
                   warmCache.manualFlushes)
            << ",\"cache_rejected_programs\":"
            << difference(
                   finalCache.rejectedPrograms,
                   warmCache.rejectedPrograms)
            << ",\"cache_generated_bytes\":"
            << difference(
                   finalCache.generatedBytes,
                   warmCache.generatedBytes)
            << ",\"cache_compilation_nanoseconds\":"
            << difference(
                   finalCache.compilationNanoseconds,
                   warmCache.compilationNanoseconds)
            << ",\"cache_resident_programs\":"
            << finalCache.residentPrograms
            << ",\"cache_resident_executable_bytes\":"
            << finalCache.residentExecutableBytes
            << ",\"cache_high_water_programs\":"
            << finalCache.highWaterPrograms
            << ",\"cache_high_water_executable_bytes\":"
            << finalCache.highWaterExecutableBytes
            << ",\"valid\":"
            << (allValid ? "true" : "false")
            << "}\n";
    }
    printBlockProfiles(
        native
            .blockProfilingSnapshotWhileExecutionQuiescent(),
        native.diagnostics());
    return allValid ? 0 : 1;
}
