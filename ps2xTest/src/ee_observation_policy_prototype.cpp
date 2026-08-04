#include "ee_observation_policy_prototype.h"

#include "ps2_runtime_macros.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace
{
    using namespace ps2x::test::observation;

    constexpr uint32_t kOutputSchemaVersion = 1u;
    constexpr uint32_t kPccrCte = 1u << 31u;
    constexpr uint32_t kPccrK0 = 1u << 2u;
    constexpr uint32_t kPccrK1 = 1u << 12u;
    constexpr uint32_t kConfigBpe = 1u << 12u;
    constexpr uint32_t kConfigDie = 1u << 18u;

    enum class Representation : uint8_t
    {
        Template,
        Explicit,
        CompactExecutor,
        OutlinedThunks,
    };

    enum class Feature : uint8_t
    {
        Inactive,
        PccrCompletion,
    };

    struct PrototypeCase
    {
        std::string_view name;
        Representation representation = Representation::Template;
        EeArchitecturalObservationMode mode =
            EeArchitecturalObservationMode::Precise;
        Feature feature = Feature::Inactive;
        PS2Runtime::RecompiledFunction entry = nullptr;
    };

    constexpr std::array<PrototypeCase, 10u> kCases{{
        {"template_fast_inactive",
         Representation::Template,
         EeArchitecturalObservationMode::Fast,
         Feature::Inactive,
         EeObservationTemplateFast},
        {"explicit_fast_inactive",
         Representation::Explicit,
         EeArchitecturalObservationMode::Fast,
         Feature::Inactive,
         EeObservationExplicitFast},
        {"template_precise_inactive",
         Representation::Template,
         EeArchitecturalObservationMode::Precise,
         Feature::Inactive,
         EeObservationTemplatePrecise},
        {"explicit_precise_inactive",
         Representation::Explicit,
         EeArchitecturalObservationMode::Precise,
         Feature::Inactive,
         EeObservationExplicitPrecise},
        {"compact_precise_inactive",
         Representation::CompactExecutor,
         EeArchitecturalObservationMode::Precise,
         Feature::Inactive,
         EeObservationCompactPrecise},
        {"outlined_precise_inactive",
         Representation::OutlinedThunks,
         EeArchitecturalObservationMode::Precise,
         Feature::Inactive,
         EeObservationOutlinedPrecise},
        {"template_precise_pccr",
         Representation::Template,
         EeArchitecturalObservationMode::Precise,
         Feature::PccrCompletion,
         EeObservationTemplatePrecise},
        {"explicit_precise_pccr",
         Representation::Explicit,
         EeArchitecturalObservationMode::Precise,
         Feature::PccrCompletion,
         EeObservationExplicitPrecise},
        {"compact_precise_pccr",
         Representation::CompactExecutor,
         EeArchitecturalObservationMode::Precise,
         Feature::PccrCompletion,
         EeObservationCompactPrecise},
        {"outlined_precise_pccr",
         Representation::OutlinedThunks,
         EeArchitecturalObservationMode::Precise,
         Feature::PccrCompletion,
         EeObservationOutlinedPrecise},
    }};

    struct Configuration
    {
        uint64_t blocks = 1'000'000u;
        uint64_t warmupBlocks = 20'000u;
        uint32_t samples = 5u;
    };

    struct Measurement
    {
        const PrototypeCase *prototypeCase = nullptr;
        uint32_t sample = 0u;
        uint64_t blocks = 0u;
        uint64_t guestInstructions = 0u;
        uint64_t hostNanoseconds = 0u;
        uint64_t registerHash = 0u;
        uint64_t memoryHash = 0u;
        uint64_t workHash = 0u;
        uint64_t counterHash = 0u;
        uint64_t insnCount = 0u;
        uint32_t random = 0u;
        uint32_t pccr = 0u;
        uint32_t pcr0 = 0u;
        uint32_t pcr1 = 0u;
        uint32_t pc = 0u;
        bool issuePending = false;
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

        constexpr uint64_t maximumBlocks =
            std::numeric_limits<uint32_t>::max() /
            (kInstructionsPerBlock * kCycleTicksPerInstruction);
        return configuration.blocks != 0u &&
               configuration.blocks <= maximumBlocks &&
               configuration.warmupBlocks <= maximumBlocks;
    }

    std::string_view representationName(Representation representation)
    {
        switch (representation)
        {
        case Representation::Template:
            return "parameterized_template";
        case Representation::Explicit:
            return "explicit_bodies";
        case Representation::CompactExecutor:
            return "compact_executor";
        case Representation::OutlinedThunks:
            return "outlined_thunks";
        }
        return "unknown";
    }

    std::string_view modeName(EeArchitecturalObservationMode mode)
    {
        return mode == EeArchitecturalObservationMode::Fast
                   ? "fast"
                   : "precise";
    }

    std::string_view featureName(Feature feature)
    {
        return feature == Feature::Inactive
                   ? "inactive"
                   : "pccr_completion";
    }

    uint32_t performanceControl(
        uint32_t event0,
        uint32_t event1)
    {
        return kPccrCte |
               kPccrK0 |
               kPccrK1 |
               (event0 << 5u) |
               (event1 << 15u);
    }

    uint32_t expectedRandom(uint64_t guestInstructions)
    {
        return 47u -
               static_cast<uint32_t>(guestInstructions % 48u);
    }

    class Fixture
    {
    public:
        Fixture(
            const PrototypeCase &prototypeCase,
            uint64_t blocks)
            : m_case(prototypeCase),
              m_blocks(blocks)
        {
            if (!runtime.memory().initialize())
            {
                throw std::runtime_error(
                    "failed to initialize PS2 memory");
            }
            rdram = runtime.memory().getRDRAM();
            if (!rdram)
            {
                throw std::runtime_error("RDRAM is unavailable");
            }
            initializeMemory();
            initializeContext();
            if (m_case.feature == Feature::PccrCompletion)
            {
                configurePerformance();
            }
        }

        void run()
        {
            for (uint64_t block = 0u;
                 block < m_blocks;
                 ++block)
            {
                m_case.entry(rdram, &context, &runtime);
            }
        }

        void finish()
        {
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            if (m_case.feature == Feature::PccrCompletion)
            {
                (void)runtime.readCop0Performance(
                    rdram, &context, 1u);
                (void)runtime.readCop0Performance(
                    rdram, &context, 3u);
            }
        }

        Measurement measurement(
            uint32_t sample,
            uint64_t hostNanoseconds) const
        {
            Measurement result{};
            result.prototypeCase = &m_case;
            result.sample = sample;
            result.blocks = m_blocks;
            result.guestInstructions =
                m_blocks * kInstructionsPerBlock;
            result.hostNanoseconds = hostNanoseconds;
            result.registerHash = hashRegisters();
            result.memoryHash = hashMemory();
            result.insnCount = context.insn_count;
            result.random = context.cop0_random;
            result.pccr = context.cop0_perf;
            result.pcr0 = context.cop0_pcr0;
            result.pcr1 = context.cop0_pcr1;
            result.pc = context.pc;
            result.issuePending =
                context.ee_performance_issue_pending;

            Fnv1a64 work;
            work.add(result.registerHash);
            work.add(result.memoryHash);
            work.add(result.insnCount);
            work.add(result.random);
            result.workHash = work.value();

            Fnv1a64 counters;
            counters.add(result.insnCount);
            counters.add(result.random);
            counters.add(result.pccr);
            counters.add(result.pcr0);
            counters.add(result.pcr1);
            counters.add(result.pc);
            counters.add(result.issuePending);
            result.counterHash = counters.value();
            return result;
        }

    private:
        void initializeMemory()
        {
            for (uint32_t word = 0u;
                 word < kWorkingSetWords;
                 ++word)
            {
                const uint32_t value =
                    0x6d2b79f5u ^
                    (word * 0x9e3779b9u);
                std::memcpy(
                    rdram + kPhysicalBase +
                        word * sizeof(uint32_t),
                    &value,
                    sizeof(value));
            }
        }

        void initializeContext()
        {
            context = R5900Context{};
            context.cop0_status = 0u;
            context.cop0_config |= kConfigBpe | kConfigDie;
            context.cop0_wired = 0u;
            context.cop0_random = 47u;
            SET_GPR_U32(&context, 2u, 0x13579bdfu);
            SET_GPR_U32(&context, 3u, 0x2468ace0u);
            SET_GPR_U32(&context, 4u, 0u);
        }

        void configurePerformance()
        {
            runtime.writeCop0Performance(
                rdram, &context, 1u, 0u);
            runtime.writeCop0Performance(
                rdram, &context, 3u, 0u);
            runtime.writeCop0Performance(
                rdram,
                &context,
                0u,
                performanceControl(12u, 12u));
        }

        uint64_t hashRegisters() const
        {
            Fnv1a64 hash;
            for (uint32_t reg = 2u; reg <= 4u; ++reg)
            {
                hash.addBytes(
                    &context.r[reg],
                    sizeof(context.r[reg]));
            }
            return hash.value();
        }

        uint64_t hashMemory() const
        {
            Fnv1a64 hash;
            hash.addBytes(
                rdram + kPhysicalBase,
                kWorkingSetBytes);
            return hash.value();
        }

        const PrototypeCase &m_case;
        uint64_t m_blocks = 0u;

    public:
        PS2Runtime runtime;
        R5900Context context{};
        uint8_t *rdram = nullptr;
    };

    void verifyMeasurement(const Measurement &measurement)
    {
        const uint64_t expectedInstructions =
            measurement.blocks * kInstructionsPerBlock;
        if (measurement.insnCount != expectedInstructions ||
            measurement.random != expectedRandom(expectedInstructions) ||
            measurement.pc != kCodeBase + 16u ||
            measurement.issuePending)
        {
            std::ostringstream message;
            message
                << "prototype instruction boundary is incorrect: case="
                << measurement.prototypeCase->name
                << " instructions=" << measurement.insnCount
                << "/" << expectedInstructions
                << " random=" << measurement.random
                << "/" << expectedRandom(expectedInstructions)
                << " pc=0x" << std::hex << measurement.pc
                << "/0x" << (kCodeBase + 16u) << std::dec
                << " issue_pending=" << measurement.issuePending;
            throw std::runtime_error(message.str());
        }

        const uint32_t expectedCounter =
            measurement.prototypeCase->feature ==
                    Feature::PccrCompletion
                ? static_cast<uint32_t>(expectedInstructions)
                : 0u;
        if (measurement.pcr0 != expectedCounter ||
            measurement.pcr1 != expectedCounter)
        {
            std::ostringstream message;
            message
                << "prototype PCCR result is incorrect: case="
                << measurement.prototypeCase->name
                << " pcr0=" << measurement.pcr0
                << " pcr1=" << measurement.pcr1
                << " expected=" << expectedCounter;
            throw std::runtime_error(message.str());
        }
    }

    Measurement runMeasurement(
        const PrototypeCase &prototypeCase,
        uint64_t blocks,
        uint32_t sample)
    {
        Fixture fixture(prototypeCase, blocks);
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
        const PrototypeCase &prototypeCase,
        uint64_t blocks)
    {
        if (blocks == 0u)
            return;
        Fixture fixture(prototypeCase, blocks);
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
        const PrototypeCase &prototypeCase =
            *measurement.prototypeCase;
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"measurement\""
            << ",\"case\":\"" << prototypeCase.name << "\""
            << ",\"representation\":\""
            << representationName(prototypeCase.representation)
            << "\""
            << ",\"mode\":\"" << modeName(prototypeCase.mode)
            << "\""
            << ",\"feature\":\""
            << featureName(prototypeCase.feature) << "\""
            << ",\"sample\":" << measurement.sample
            << ",\"blocks\":" << measurement.blocks
            << ",\"guest_instructions\":"
            << measurement.guestInstructions
            << ",\"host_time_ns\":"
            << measurement.hostNanoseconds
            << ",\"host_ns_per_guest_instruction\":"
            << (measurement.guestInstructions == 0u
                    ? 0.0
                    : static_cast<double>(
                          measurement.hostNanoseconds) /
                          static_cast<double>(
                              measurement.guestInstructions))
            << ",\"register_hash\":\"";
        printHex(measurement.registerHash);
        std::cout << "\",\"memory_hash\":\"";
        printHex(measurement.memoryHash);
        std::cout << "\",\"work_hash\":\"";
        printHex(measurement.workHash);
        std::cout << "\",\"counter_hash\":\"";
        printHex(measurement.counterHash);
        std::cout
            << "\",\"insn_count\":" << measurement.insnCount
            << ",\"random\":" << measurement.random
            << ",\"pccr\":" << measurement.pccr
            << ",\"pcr0\":" << measurement.pcr0
            << ",\"pcr1\":" << measurement.pcr1
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
            << "usage: ee_observation_policy_prototype "
               "[--blocks N] [--warmup-blocks N] [--samples N]\n";
        return 2;
    }

    try
    {
        for (const PrototypeCase &prototypeCase : kCases)
        {
            runWarmup(
                prototypeCase,
                configuration.warmupBlocks);
        }

        std::optional<uint64_t> expectedWorkHash;
        std::optional<uint64_t> inactiveCounterHash;
        std::optional<uint64_t> pccrCounterHash;
        std::cout << std::fixed << std::setprecision(6);
        for (uint32_t sample = 0u;
             sample < configuration.samples;
             ++sample)
        {
            for (size_t caseOffset = 0u;
                 caseOffset < kCases.size();
                 ++caseOffset)
            {
                const PrototypeCase &prototypeCase =
                    kCases[(caseOffset + sample) % kCases.size()];
                const Measurement measurement = runMeasurement(
                    prototypeCase,
                    configuration.blocks,
                    sample);
                if (!expectedWorkHash)
                {
                    expectedWorkHash = measurement.workHash;
                }
                else if (*expectedWorkHash != measurement.workHash)
                {
                    throw std::runtime_error(
                        "prototype representations performed different work");
                }

                std::optional<uint64_t> &expectedCounterHash =
                    prototypeCase.feature == Feature::Inactive
                        ? inactiveCounterHash
                        : pccrCounterHash;
                if (!expectedCounterHash)
                {
                    expectedCounterHash = measurement.counterHash;
                }
                else if (*expectedCounterHash !=
                         measurement.counterHash)
                {
                    throw std::runtime_error(
                        "prototype representations produced different counters");
                }
                printMeasurement(measurement);
            }
        }

        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"summary\""
            << ",\"cases\":" << kCases.size()
            << ",\"samples\":" << configuration.samples
            << ",\"blocks_per_case\":" << configuration.blocks
            << ",\"work_hash\":\"";
        printHex(*expectedWorkHash);
        std::cout << "\",\"status\":\"pass\"}\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
