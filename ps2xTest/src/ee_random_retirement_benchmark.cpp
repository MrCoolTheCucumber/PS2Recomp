#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    constexpr uint32_t kOutputSchemaVersion = 1u;
    constexpr uint32_t kRoundsPerBlock = 4u;
    constexpr uint32_t kInstructionsPerRound = 16u;
    constexpr uint32_t kInstructionsPerBlock =
        kRoundsPerBlock * kInstructionsPerRound;
    constexpr uint32_t kCodeBase = 0x001a0000u;
    constexpr uint32_t kInitialRandom = 43u;
    constexpr uint32_t kInitialWired = 5u;

#if defined(_MSC_VER)
#define PS2_RANDOM_BENCHMARK_NOINLINE __declspec(noinline)
#define PS2_RANDOM_BENCHMARK_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define PS2_RANDOM_BENCHMARK_NOINLINE __attribute__((noinline, noipa))
#define PS2_RANDOM_BENCHMARK_ALWAYS_INLINE \
    inline __attribute__((always_inline))
#else
#define PS2_RANDOM_BENCHMARK_NOINLINE __attribute__((noinline))
#define PS2_RANDOM_BENCHMARK_ALWAYS_INLINE \
    inline __attribute__((always_inline))
#endif

    enum class RetirementStrategy : uint8_t
    {
        ScalarReference,
        GeneratedBatch,
        DirectBlockTotal,
    };

    using BenchmarkEntry = void (*)(R5900Context *, uint64_t);

    struct BenchmarkCase
    {
        std::string_view name;
        RetirementStrategy strategy;
        BenchmarkEntry entry;
    };

    struct Configuration
    {
        uint64_t blocks = 2'000'000u;
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
        uint64_t hostNanoseconds = 0u;
        uint64_t registerHash = 0u;
        uint64_t workHash = 0u;
        uint64_t counterHash = 0u;
        uint64_t insnCount = 0u;
        uint64_t pendingRetirements = 0u;
        uint32_t random = 0u;
        uint32_t wired = 0u;
        uint32_t pc = 0u;
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

    template <RetirementStrategy Strategy>
    PS2_RANDOM_BENCHMARK_ALWAYS_INLINE
    void beginArithmeticInstruction(R5900Context *ctx) noexcept
    {
        if constexpr (Strategy ==
                      RetirementStrategy::ScalarReference)
        {
            ctx->beginEeInstruction();
        }
    }

    template <RetirementStrategy Strategy>
    PS2_RANDOM_BENCHMARK_ALWAYS_INLINE
    void completeArithmeticBlock(R5900Context *ctx) noexcept
    {
        if constexpr (Strategy ==
                      RetirementStrategy::GeneratedBatch)
        {
            // This is the production generated-code operation. The outer
            // publication boundary materializes the counted Random update.
            ctx->retireEeInstructions(kInstructionsPerBlock);
            ctx->finishEeInstruction();
        }
        else if constexpr (Strategy ==
                           RetirementStrategy::DirectBlockTotal)
        {
            // Benchmark-only control: compile the same retirement count
            // directly into the block total, without the generic pending
            // accumulator. The bounded fixture cannot overflow insn_count.
            ctx->insn_count += kInstructionsPerBlock;
            ctx->advanceCop0Random(kInstructionsPerBlock);
        }
        else
        {
            ctx->finishEeInstruction();
        }
    }

    template <RetirementStrategy Strategy>
    PS2_RANDOM_BENCHMARK_ALWAYS_INLINE
    void runArithmeticBlocks(
        R5900Context *ctx,
        uint64_t blocks) noexcept
    {
        uint32_t a = GPR_U32(ctx, 2u);
        uint32_t b = GPR_U32(ctx, 3u);
        uint32_t c = GPR_U32(ctx, 4u);
        uint32_t d = GPR_U32(ctx, 5u);

        for (uint64_t block = 0u; block < blocks; ++block)
        {
            const uint32_t blockMix =
                static_cast<uint32_t>(block);
            for (uint32_t round = 0u;
                 round < kRoundsPerBlock;
                 ++round)
            {
                const uint32_t roundMix =
                    blockMix + round * 0x85ebca6bu;

                beginArithmeticInstruction<Strategy>(ctx);
                a += 0x9e3779b9u;
                beginArithmeticInstruction<Strategy>(ctx);
                b ^= a;
                beginArithmeticInstruction<Strategy>(ctx);
                c <<= 5u;
                beginArithmeticInstruction<Strategy>(ctx);
                d |= c;
                beginArithmeticInstruction<Strategy>(ctx);
                a -= b;
                beginArithmeticInstruction<Strategy>(ctx);
                b &= d;
                beginArithmeticInstruction<Strategy>(ctx);
                c >>= 3u;
                beginArithmeticInstruction<Strategy>(ctx);
                d ^= a;
                beginArithmeticInstruction<Strategy>(ctx);
                a |= c;
                beginArithmeticInstruction<Strategy>(ctx);
                b += 0x7f4a7c15u;
                beginArithmeticInstruction<Strategy>(ctx);
                c = static_cast<uint32_t>(
                    static_cast<int32_t>(c) >> 7u);
                beginArithmeticInstruction<Strategy>(ctx);
                d -= c;
                beginArithmeticInstruction<Strategy>(ctx);
                a = a < b ? 1u : 0u;
                beginArithmeticInstruction<Strategy>(ctx);
                b ^= d;
                beginArithmeticInstruction<Strategy>(ctx);
                c += b;
                beginArithmeticInstruction<Strategy>(ctx);
                d &= c ^ roundMix;
            }

            completeArithmeticBlock<Strategy>(ctx);
        }

        SET_GPR_U32(ctx, 2u, a);
        SET_GPR_U32(ctx, 3u, b);
        SET_GPR_U32(ctx, 4u, c);
        SET_GPR_U32(ctx, 5u, d);
        ctx->pc = kCodeBase + kInstructionsPerBlock * 4u;
    }

    extern "C" PS2_RANDOM_BENCHMARK_NOINLINE
    void EeRandomRetirementScalarReference(
        R5900Context *ctx,
        uint64_t blocks)
    {
        runArithmeticBlocks<
            RetirementStrategy::ScalarReference>(ctx, blocks);
    }

    extern "C" PS2_RANDOM_BENCHMARK_NOINLINE
    void EeRandomRetirementGeneratedBatch(
        R5900Context *ctx,
        uint64_t blocks)
    {
        runArithmeticBlocks<
            RetirementStrategy::GeneratedBatch>(ctx, blocks);
    }

    extern "C" PS2_RANDOM_BENCHMARK_NOINLINE
    void EeRandomRetirementDirectBlockTotal(
        R5900Context *ctx,
        uint64_t blocks)
    {
        runArithmeticBlocks<
            RetirementStrategy::DirectBlockTotal>(ctx, blocks);
    }

    constexpr std::array<BenchmarkCase, 3u> kCases{{
        {"scalar_reference",
         RetirementStrategy::ScalarReference,
         EeRandomRetirementScalarReference},
        {"generated_batch",
         RetirementStrategy::GeneratedBatch,
         EeRandomRetirementGeneratedBatch},
        {"direct_block_total",
         RetirementStrategy::DirectBlockTotal,
         EeRandomRetirementDirectBlockTotal},
    }};

    bool parseUnsigned(
        std::string_view text,
        uint64_t &value)
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
            kInstructionsPerBlock;
        return configuration.blocks != 0u &&
               configuration.blocks <= maximumBlocks &&
               configuration.warmupBlocks <= maximumBlocks;
    }

    std::string_view strategyName(
        RetirementStrategy strategy)
    {
        switch (strategy)
        {
        case RetirementStrategy::ScalarReference:
            return "scalar_reference";
        case RetirementStrategy::GeneratedBatch:
            return "production_counted_batch";
        case RetirementStrategy::DirectBlockTotal:
            return "benchmark_direct_total";
        }
        return "unknown";
    }

    uint32_t expectedRandom(uint64_t retirementCount)
    {
        constexpr uint32_t upperBound = 47u;
        constexpr uint32_t lowerBound = kInitialWired;
        constexpr uint32_t interval =
            upperBound - lowerBound + 1u;
        const uint32_t offset =
            kInitialRandom - lowerBound;
        const uint32_t decrement =
            static_cast<uint32_t>(retirementCount % interval);
        return lowerBound +
               ((offset + interval - decrement) % interval);
    }

    uint64_t hashRegisters(const R5900Context &ctx)
    {
        Fnv1a64 hash;
        for (uint32_t reg = 2u; reg <= 5u; ++reg)
        {
            hash.addBytes(&ctx.r[reg], sizeof(ctx.r[reg]));
        }
        return hash.value();
    }

    Measurement runMeasurement(
        const BenchmarkCase &benchmarkCase,
        uint64_t blocks,
        uint32_t sample)
    {
        R5900Context ctx{};
        ctx.cop0_random = kInitialRandom;
        ctx.cop0_wired = kInitialWired;
        SET_GPR_U32(&ctx, 2u, 0x13579bdfu);
        SET_GPR_U32(&ctx, 3u, 0x2468ace0u);
        SET_GPR_U32(&ctx, 4u, 0xf00dcafeu);
        SET_GPR_U32(&ctx, 5u, 0x0badc0deu);

        const auto started =
            std::chrono::steady_clock::now();
        benchmarkCase.entry(&ctx, blocks);
        const auto stopped =
            std::chrono::steady_clock::now();
        ctx.finishEeInstruction();

        Measurement measurement{};
        measurement.benchmarkCase = &benchmarkCase;
        measurement.sample = sample;
        measurement.blocks = blocks;
        measurement.guestInstructions =
            blocks * kInstructionsPerBlock;
        measurement.hostNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    stopped - started)
                    .count());
        measurement.registerHash = hashRegisters(ctx);
        measurement.insnCount = ctx.insn_count;
        measurement.pendingRetirements =
            ctx.cop0_random_pending_retirements;
        measurement.random = ctx.cop0_random;
        measurement.wired = ctx.cop0_wired;
        measurement.pc = ctx.pc;

        Fnv1a64 work;
        work.add(measurement.registerHash);
        work.add(measurement.insnCount);
        work.add(measurement.random);
        measurement.workHash = work.value();

        Fnv1a64 counters;
        counters.add(measurement.insnCount);
        counters.add(measurement.pendingRetirements);
        counters.add(measurement.random);
        counters.add(measurement.wired);
        counters.add(measurement.pc);
        measurement.counterHash = counters.value();

        const uint32_t expectedRandomValue =
            expectedRandom(measurement.guestInstructions);
        if (measurement.insnCount !=
                measurement.guestInstructions ||
            measurement.pendingRetirements != 0u ||
            measurement.random != expectedRandomValue ||
            measurement.wired != kInitialWired ||
            measurement.pc !=
                kCodeBase + kInstructionsPerBlock * 4u)
        {
            std::ostringstream message;
            message
                << "incorrect retirement result: case="
                << benchmarkCase.name
                << " instructions=" << measurement.insnCount
                << "/" << measurement.guestInstructions
                << " pending="
                << measurement.pendingRetirements
                << " random=" << measurement.random
                << "/" << expectedRandomValue
                << " wired=" << measurement.wired
                << " pc=0x" << std::hex << measurement.pc;
            throw std::runtime_error(message.str());
        }
        return measurement;
    }

    void runWarmup(
        const BenchmarkCase &benchmarkCase,
        uint64_t blocks)
    {
        if (blocks != 0u)
        {
            (void)runMeasurement(
                benchmarkCase, blocks, 0u);
        }
    }

    uint64_t median(std::vector<uint64_t> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    }

    void printHex(uint64_t value)
    {
        std::cout << "0x" << std::hex << value << std::dec;
    }

    void printMeasurement(const Measurement &measurement)
    {
        const BenchmarkCase &benchmarkCase =
            *measurement.benchmarkCase;
        std::cout
            << "{\"schema_version\":"
            << kOutputSchemaVersion
            << ",\"event\":\"measurement\""
            << ",\"case\":\"" << benchmarkCase.name << "\""
            << ",\"strategy\":\""
            << strategyName(benchmarkCase.strategy) << "\""
            << ",\"sample\":" << measurement.sample
            << ",\"blocks\":" << measurement.blocks
            << ",\"guest_instructions\":"
            << measurement.guestInstructions
            << ",\"host_time_ns\":"
            << measurement.hostNanoseconds
            << ",\"host_ns_per_guest_instruction\":"
            << static_cast<double>(
                   measurement.hostNanoseconds) /
                   static_cast<double>(
                       measurement.guestInstructions)
            << ",\"register_hash\":\"";
        printHex(measurement.registerHash);
        std::cout << "\",\"work_hash\":\"";
        printHex(measurement.workHash);
        std::cout << "\",\"counter_hash\":\"";
        printHex(measurement.counterHash);
        std::cout
            << "\",\"insn_count\":"
            << measurement.insnCount
            << ",\"random\":" << measurement.random
            << ",\"wired\":" << measurement.wired
            << ",\"pending_retirements\":"
            << measurement.pendingRetirements
            << ",\"pc\":" << measurement.pc
            << "}\n";
    }
}
int main(int argc, char **argv)
{
    Configuration configuration{};
    if (!parseArguments(argc, argv, configuration))
    {
        std::cerr
            << "usage: ee_random_retirement_benchmark "
               "[--blocks N] [--warmup-blocks N] "
               "[--samples N] [--case NAME]\n";
        return 2;
    }

    try
    {
        std::vector<const BenchmarkCase *> selectedCases;
        for (const BenchmarkCase &benchmarkCase : kCases)
        {
            if (configuration.caseFilter.empty() ||
                benchmarkCase.name ==
                    configuration.caseFilter)
            {
                selectedCases.push_back(&benchmarkCase);
            }
        }
        if (selectedCases.empty())
        {
            throw std::runtime_error(
                "the requested benchmark case does not exist");
        }

        for (const BenchmarkCase *benchmarkCase :
             selectedCases)
        {
            runWarmup(
                *benchmarkCase,
                configuration.warmupBlocks);
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout
            << "{\"schema_version\":"
            << kOutputSchemaVersion
            << ",\"event\":\"configuration\""
            << ",\"blocks\":" << configuration.blocks
            << ",\"warmup_blocks\":"
            << configuration.warmupBlocks
            << ",\"samples\":" << configuration.samples
            << ",\"instructions_per_block\":"
            << kInstructionsPerBlock
            << "}\n";

        std::array<std::vector<uint64_t>, kCases.size()>
            elapsedByCase;
        std::optional<uint64_t> expectedWorkHash;
        std::optional<uint64_t> expectedCounterHash;
        for (uint32_t sample = 0u;
             sample < configuration.samples;
             ++sample)
        {
            for (size_t offset = 0u;
                 offset < selectedCases.size();
                 ++offset)
            {
                const BenchmarkCase &benchmarkCase =
                    *selectedCases[
                        (offset + sample) %
                        selectedCases.size()];
                const Measurement measurement =
                    runMeasurement(
                        benchmarkCase,
                        configuration.blocks,
                        sample);
                if (!expectedWorkHash)
                {
                    expectedWorkHash = measurement.workHash;
                    expectedCounterHash =
                        measurement.counterHash;
                }
                else if (*expectedWorkHash !=
                             measurement.workHash ||
                         *expectedCounterHash !=
                             measurement.counterHash)
                {
                    throw std::runtime_error(
                        "retirement strategies produced different architectural state");
                }

                const size_t caseIndex =
                    static_cast<size_t>(
                        &benchmarkCase - kCases.data());
                elapsedByCase[caseIndex].push_back(
                    measurement.hostNanoseconds);
                printMeasurement(measurement);
            }
        }

        std::optional<uint64_t> directMedian;
        for (size_t index = 0u;
             index < kCases.size();
             ++index)
        {
            if (!elapsedByCase[index].empty() &&
                kCases[index].strategy ==
                    RetirementStrategy::DirectBlockTotal)
            {
                directMedian = median(
                    elapsedByCase[index]);
            }
        }

        for (size_t index = 0u;
             index < kCases.size();
             ++index)
        {
            const std::vector<uint64_t> &samples =
                elapsedByCase[index];
            if (samples.empty())
            {
                continue;
            }
            const uint64_t medianNanoseconds =
                median(samples);
            const auto [minimum, maximum] =
                std::minmax_element(
                    samples.begin(), samples.end());
            std::cout
                << "{\"schema_version\":"
                << kOutputSchemaVersion
                << ",\"event\":\"summary\""
                << ",\"case\":\"" << kCases[index].name
                << "\",\"samples\":" << samples.size()
                << ",\"median_host_time_ns\":"
                << medianNanoseconds
                << ",\"min_host_time_ns\":" << *minimum
                << ",\"max_host_time_ns\":" << *maximum
                << ",\"median_host_ns_per_guest_instruction\":"
                << static_cast<double>(medianNanoseconds) /
                       static_cast<double>(
                           configuration.blocks *
                           kInstructionsPerBlock);
            if (directMedian)
            {
                std::cout
                    << ",\"median_ratio_to_direct\":"
                    << static_cast<double>(medianNanoseconds) /
                           static_cast<double>(*directMedian);
            }
            std::cout << "}\n";
        }

        std::cout
            << "{\"schema_version\":"
            << kOutputSchemaVersion
            << ",\"event\":\"result\""
            << ",\"work_hash\":\"";
        printHex(*expectedWorkHash);
        std::cout << "\",\"counter_hash\":\"";
        printHex(*expectedCounterHash);
        std::cout << "\",\"status\":\"pass\"}\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
