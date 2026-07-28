#include "ps2_runtime.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    constexpr uint32_t kOutputSchemaVersion = 4u;

    std::atomic<bool> g_trackAllocations{false};
    std::atomic<uint64_t> g_allocationCount{0u};
    std::atomic<uint64_t> g_allocationBytes{0u};

    void recordAllocation(std::size_t size) noexcept
    {
        if (!g_trackAllocations.load(std::memory_order_relaxed))
            return;
        g_allocationCount.fetch_add(1u, std::memory_order_relaxed);
        g_allocationBytes.fetch_add(
            static_cast<uint64_t>(size), std::memory_order_relaxed);
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

    void *allocateAligned(std::size_t size, std::size_t alignment)
    {
        size = std::max<std::size_t>(size, 1u);
        void *result = nullptr;
#if defined(_WIN32)
        result = _aligned_malloc(size, alignment);
        if (!result)
            throw std::bad_alloc();
#else
        if (posix_memalign(&result, alignment, size) != 0)
            throw std::bad_alloc();
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

void *operator new(std::size_t size, const std::nothrow_t &) noexcept
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

void *operator new[](std::size_t size, const std::nothrow_t &) noexcept
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

void *operator new(std::size_t size, std::align_val_t alignment)
{
    return allocateAligned(size, static_cast<std::size_t>(alignment));
}

void *operator new[](std::size_t size, std::align_val_t alignment)
{
    return allocateAligned(size, static_cast<std::size_t>(alignment));
}

void *operator new(
    std::size_t size, std::align_val_t alignment,
    const std::nothrow_t &) noexcept
{
    try
    {
        return allocateAligned(size, static_cast<std::size_t>(alignment));
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
        return allocateAligned(size, static_cast<std::size_t>(alignment));
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

void operator delete(void *pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void *pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete(void *pointer, const std::nothrow_t &) noexcept
{
    std::free(pointer);
}

void operator delete[](void *pointer, const std::nothrow_t &) noexcept
{
    std::free(pointer);
}

void operator delete(void *pointer, std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete[](void *pointer, std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete(void *pointer, std::size_t, std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete[](void *pointer, std::size_t, std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete(
    void *pointer, std::align_val_t, const std::nothrow_t &) noexcept
{
    freeAligned(pointer);
}

void operator delete[](
    void *pointer, std::align_val_t, const std::nothrow_t &) noexcept
{
    freeAligned(pointer);
}

namespace
{
    constexpr uint32_t kVuUpperNop = 0x0000003Fu;
    constexpr uint32_t kVuUpperAdd =
        (0xFu << 21u) | (2u << 16u) | (1u << 11u) |
        (3u << 6u) | 0x28u;

    enum class VuWorkload
    {
        Nop,
        Fmac,
    };

    struct Configuration
    {
        uint64_t blocks = 1'000'000u;
        uint64_t warmupBlocks = 20'000u;
        uint32_t blockTicks = 64u;
        uint32_t samples = 5u;
        VuWorkload vuWorkload = VuWorkload::Nop;
    };

    struct Measurement
    {
        std::string_view mode;
        uint32_t sample = 0u;
        uint64_t eeBlocks = 0u;
        uint64_t eeTicks = 0u;
        uint64_t eventTests = 0u;
        uint64_t eventServices = 0u;
        uint64_t vuPairs = 0u;
        uint64_t hostNanoseconds = 0u;
        uint64_t allocations = 0u;
        uint64_t allocatedBytes = 0u;
        uint64_t stateHash = 0u;
        uint32_t vuPc = 0u;
        VuWorkload vuWorkload = VuWorkload::Nop;
    };

    std::string_view vuWorkloadName(VuWorkload workload)
    {
        switch (workload)
        {
        case VuWorkload::Nop:
            return "nop";
        case VuWorkload::Fmac:
            return "fmac";
        }
        return "unknown";
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

    bool parseArguments(int argc, char **argv, Configuration &configuration)
    {
        for (int index = 1; index < argc; index += 2)
        {
            if (index + 1 >= argc)
                return false;

            const std::string_view option = argv[index];
            const std::string_view argument = argv[index + 1];
            if (option == "--vu-workload")
            {
                if (argument == "nop")
                    configuration.vuWorkload = VuWorkload::Nop;
                else if (argument == "fmac")
                    configuration.vuWorkload = VuWorkload::Fmac;
                else
                    return false;
                continue;
            }

            uint64_t value = 0u;
            if (!parseUnsigned(argument, value))
                return false;

            if (option == "--blocks")
            {
                configuration.blocks = value;
            }
            else if (option == "--warmup-blocks")
            {
                configuration.warmupBlocks = value;
            }
            else if (option == "--block-ticks")
            {
                if (value == 0u ||
                    value > std::numeric_limits<uint32_t>::max())
                {
                    return false;
                }
                configuration.blockTicks = static_cast<uint32_t>(value);
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
        return configuration.blocks != 0u;
    }

    uint64_t hashState(const VU1State &state)
    {
        const auto *const bytes =
            reinterpret_cast<const uint8_t *>(&state);
        uint64_t hash = 1469598103934665603ull;
        for (std::size_t index = 0u; index < sizeof(state); ++index)
        {
            hash ^= bytes[index];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    void writeInstructionPair(
        uint8_t *code, uint32_t pc, uint32_t lower, uint32_t upper)
    {
        std::memcpy(code + pc, &lower, sizeof(lower));
        std::memcpy(code + pc + sizeof(lower), &upper, sizeof(upper));
    }

    bool initializeFixture(
        PS2Runtime &runtime, VuWorkload workload)
    {
        if (!runtime.memory().initialize() ||
            !runtime.syncCoreSubsystems())
        {
            return false;
        }

        uint8_t *const code = runtime.memory().getVU0Code();
        uint8_t *const data = runtime.memory().getVU0Data();
        if (!code || !data)
            return false;

        std::memset(code, 0, PS2_VU0_CODE_SIZE);
        std::memset(data, 0, PS2_VU0_DATA_SIZE);
        const uint32_t upper =
            workload == VuWorkload::Fmac
                ? kVuUpperAdd
                : kVuUpperNop;
        for (uint32_t pc = 0u; pc < PS2_VU0_CODE_SIZE; pc += 8u)
            writeInstructionPair(code, pc, 0u, upper);
        return true;
    }

    void accountUnscheduledBlock(
        R5900Context &context, uint32_t blockTicks)
    {
        context.advanceEeCycleTicks(blockTicks);
        (void)context.finishEeBasicBlock();
    }

    void advanceScheduledBlocks(
        PS2Runtime &runtime, R5900Context &context,
        uint64_t blocks, uint32_t blockTicks)
    {
        uint8_t *const rdram = runtime.memory().getRDRAM();
        for (uint64_t block = 0u; block < blocks; ++block)
        {
            context.advanceEeCycleTicks(blockTicks);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
        }
    }

    uint64_t committedEeCycles(
        uint64_t blocks, uint32_t blockTicks)
    {
        const uint64_t cyclesPerBlock =
            std::max<uint32_t>(blockTicks >> 3u, 1u);
        if (blocks >
            std::numeric_limits<uint64_t>::max() /
                cyclesPerBlock)
        {
            throw std::runtime_error(
                "EE cycle count overflows uint64_t");
        }
        return blocks * cyclesPerBlock;
    }

    uint64_t committedEeTicks(
        uint64_t blocks, uint32_t blockTicks)
    {
        const uint64_t cycles =
            committedEeCycles(blocks, blockTicks);
        if (cycles >
            std::numeric_limits<uint64_t>::max() / 8u)
        {
            throw std::runtime_error(
                "EE tick count overflows uint64_t");
        }
        return cycles * 8u;
    }

    uint32_t checkedVuBudget(
        uint64_t blocks, uint32_t blockTicks)
    {
        const uint64_t budget =
            committedEeCycles(blocks, blockTicks);
        if (budget >
            std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error(
                "VU budget exceeds uint32_t");
        }
        return static_cast<uint32_t>(budget);
    }

    void configureVuWorkloadState(
        PS2Runtime &runtime, VuWorkload workload)
    {
        if (workload != VuWorkload::Fmac)
            return;

        const float source1[4] = {
            -1.0f, 2.0f, -3.0f, 4.0f};
        const float source2[4] = {
            0.5f, -2.0f, 3.0f, -4.0f};
        std::memcpy(
            runtime.vu0().state().vf[1],
            source1, sizeof(source1));
        std::memcpy(
            runtime.vu0().state().vf[2],
            source2, sizeof(source2));
    }

    void startDirectVu(
        PS2Runtime &runtime, VuWorkload workload)
    {
        configureVuWorkloadState(runtime, workload);
        runtime.vu0().execute(
            runtime.memory().getVU0Code(),
            PS2_VU0_CODE_SIZE,
            runtime.memory().getVU0Data(),
            PS2_VU0_DATA_SIZE,
            runtime.gs(), &runtime.memory(),
            0u, 0u, 0u, 16u);
    }

    void prepareEventEqualWork(
        PS2Runtime &runtime, R5900Context &context,
        const Configuration &configuration)
    {
        runtime.vu0().setProgressTrackingEnabled(true);
        startDirectVu(runtime, configuration.vuWorkload);
        advanceScheduledBlocks(
            runtime, context, configuration.warmupBlocks,
            configuration.blockTicks);
        runtime.vu0().continueExecution(
            runtime.memory().getVU0Code(),
            PS2_VU0_CODE_SIZE,
            runtime.memory().getVU0Data(),
            PS2_VU0_DATA_SIZE,
            runtime.gs(), &runtime.memory(),
            checkedVuBudget(
                configuration.warmupBlocks,
                configuration.blockTicks));

        runtime.resetEeTiming(&context);
        runtime.vu0().reset();
        context = R5900Context{};
        startDirectVu(runtime, configuration.vuWorkload);
    }

    Measurement measureEventEqualWork(
        uint32_t sample, const Configuration &configuration)
    {
        PS2Runtime runtime;
        R5900Context context{};
        if (!initializeFixture(
                runtime, configuration.vuWorkload))
            throw std::runtime_error("failed to initialize scheduler fixture");
        prepareEventEqualWork(
            runtime, context, configuration);

        const VU1ProgressSnapshot before =
            runtime.vu0().getProgressSnapshot();
        const uint64_t beforeServices =
            runtime.debugEeSchedulerSnapshot().statistics.serviced;
        g_allocationCount.store(0u, std::memory_order_relaxed);
        g_allocationBytes.store(0u, std::memory_order_relaxed);
        g_trackAllocations.store(true, std::memory_order_release);
        const auto started = std::chrono::steady_clock::now();
        advanceScheduledBlocks(
            runtime, context, configuration.blocks,
            configuration.blockTicks);
        runtime.vu0().continueExecution(
            runtime.memory().getVU0Code(),
            PS2_VU0_CODE_SIZE,
            runtime.memory().getVU0Data(),
            PS2_VU0_DATA_SIZE,
            runtime.gs(), &runtime.memory(),
            checkedVuBudget(
                configuration.blocks,
                configuration.blockTicks));
        const auto stopped = std::chrono::steady_clock::now();
        g_trackAllocations.store(false, std::memory_order_release);

        const VU1ProgressSnapshot after =
            runtime.vu0().getProgressSnapshot();
        Measurement result{};
        result.mode = "event_scheduler_equal_work";
        result.sample = sample;
        result.eeBlocks = configuration.blocks;
        result.eeTicks = committedEeTicks(
            configuration.blocks,
            configuration.blockTicks);
        result.eventTests = configuration.blocks;
        result.eventServices =
            runtime.debugEeSchedulerSnapshot()
                    .statistics.serviced -
                beforeServices;
        result.vuPairs = after.cycles - before.cycles;
        result.hostNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    stopped - started)
                    .count());
        result.allocations =
            g_allocationCount.load(std::memory_order_relaxed);
        result.allocatedBytes =
            g_allocationBytes.load(std::memory_order_relaxed);
        result.stateHash = hashState(runtime.vu0().state());
        result.vuPc = runtime.vu0().state().pc;
        result.vuWorkload = configuration.vuWorkload;
        return result;
    }

    Measurement measureNoEventHotPath(
        uint32_t sample, const Configuration &configuration)
    {
        PS2Runtime runtime;
        R5900Context context{};
        if (!initializeFixture(
                runtime, configuration.vuWorkload))
            throw std::runtime_error(
                "failed to initialize no-event fixture");
        advanceScheduledBlocks(
            runtime, context, configuration.warmupBlocks,
            configuration.blockTicks);
        runtime.resetEeTiming(&context);

        const uint64_t beforeServices =
            runtime.debugEeSchedulerSnapshot()
                .statistics.serviced;
        g_allocationCount.store(0u, std::memory_order_relaxed);
        g_allocationBytes.store(0u, std::memory_order_relaxed);
        g_trackAllocations.store(true, std::memory_order_release);
        const auto started = std::chrono::steady_clock::now();
        advanceScheduledBlocks(
            runtime, context, configuration.blocks,
            configuration.blockTicks);
        const auto stopped = std::chrono::steady_clock::now();
        g_trackAllocations.store(false, std::memory_order_release);

        Measurement result{};
        result.mode = "event_scheduler_no_pending";
        result.sample = sample;
        result.eeBlocks = configuration.blocks;
        result.eeTicks = committedEeTicks(
            configuration.blocks,
            configuration.blockTicks);
        result.eventTests = configuration.blocks;
        result.eventServices =
            runtime.debugEeSchedulerSnapshot()
                    .statistics.serviced -
                beforeServices;
        result.hostNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    stopped - started)
                    .count());
        result.allocations =
            g_allocationCount.load(std::memory_order_relaxed);
        result.allocatedBytes =
            g_allocationBytes.load(std::memory_order_relaxed);
        result.stateHash = hashState(runtime.vu0().state());
        result.vuPc = runtime.vu0().state().pc;
        result.vuWorkload = configuration.vuWorkload;
        return result;
    }

    void prepareDirect(
        PS2Runtime &runtime, R5900Context &context,
        const Configuration &configuration)
    {
        VU1Interpreter &vu = runtime.vu0();
        vu.setProgressTrackingEnabled(true);
        startDirectVu(runtime, configuration.vuWorkload);
        for (uint64_t block = 0u;
             block < configuration.warmupBlocks; ++block)
        {
            accountUnscheduledBlock(
                context, configuration.blockTicks);
        }
        vu.continueExecution(
            runtime.memory().getVU0Code(), PS2_VU0_CODE_SIZE,
            runtime.memory().getVU0Data(), PS2_VU0_DATA_SIZE,
            runtime.gs(), &runtime.memory(),
            checkedVuBudget(
                configuration.warmupBlocks,
                configuration.blockTicks));

        vu.reset();
        context = R5900Context{};
        startDirectVu(runtime, configuration.vuWorkload);
    }

    Measurement measureDirect(
        uint32_t sample, const Configuration &configuration,
        uint64_t targetVuPairs)
    {
        if (targetVuPairs > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("direct VU budget exceeds uint32_t");

        PS2Runtime runtime;
        R5900Context context{};
        if (!initializeFixture(
                runtime, configuration.vuWorkload))
            throw std::runtime_error("failed to initialize direct fixture");
        prepareDirect(runtime, context, configuration);

        const VU1ProgressSnapshot before =
            runtime.vu0().getProgressSnapshot();
        g_allocationCount.store(0u, std::memory_order_relaxed);
        g_allocationBytes.store(0u, std::memory_order_relaxed);
        g_trackAllocations.store(true, std::memory_order_release);
        const auto started = std::chrono::steady_clock::now();
        for (uint64_t block = 0u; block < configuration.blocks; ++block)
        {
            accountUnscheduledBlock(
                context, configuration.blockTicks);
        }
        runtime.vu0().continueExecution(
            runtime.memory().getVU0Code(), PS2_VU0_CODE_SIZE,
            runtime.memory().getVU0Data(), PS2_VU0_DATA_SIZE,
            runtime.gs(), &runtime.memory(),
            static_cast<uint32_t>(targetVuPairs));
        const auto stopped = std::chrono::steady_clock::now();
        g_trackAllocations.store(false, std::memory_order_release);

        const VU1ProgressSnapshot after =
            runtime.vu0().getProgressSnapshot();
        Measurement result{};
        result.mode = "pre_scheduler_equal_work";
        result.sample = sample;
        result.eeBlocks = configuration.blocks;
        result.eeTicks = committedEeTicks(
            configuration.blocks,
            configuration.blockTicks);
        result.vuPairs = after.cycles - before.cycles;
        result.hostNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    stopped - started)
                    .count());
        result.allocations =
            g_allocationCount.load(std::memory_order_relaxed);
        result.allocatedBytes =
            g_allocationBytes.load(std::memory_order_relaxed);
        result.stateHash = hashState(runtime.vu0().state());
        result.vuPc = runtime.vu0().state().pc;
        result.vuWorkload = configuration.vuWorkload;
        return result;
    }

    void printMeasurement(const Measurement &measurement)
    {
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"measurement\","
            << "\"mode\":\"" << measurement.mode << "\","
            << "\"vu_workload\":\""
            << vuWorkloadName(measurement.vuWorkload) << "\","
            << "\"sample\":" << measurement.sample << ','
            << "\"ee_blocks\":" << measurement.eeBlocks << ','
            << "\"ee_cycle_ticks\":" << measurement.eeTicks << ','
            << "\"ee_cycles\":" << (measurement.eeTicks >> 3u) << ','
            << "\"event_tests\":" << measurement.eventTests << ','
            << "\"event_services\":" << measurement.eventServices << ','
            << "\"source_services\":{},"
            << "\"vu_instruction_pairs\":" << measurement.vuPairs << ','
            << "\"vu_cycles\":" << measurement.vuPairs << ','
            << "\"host_time_ns\":" << measurement.hostNanoseconds << ','
            << "\"guest_pairs_per_host_second\":"
            << (measurement.hostNanoseconds == 0u
                    ? 0.0
                    : static_cast<double>(measurement.vuPairs) *
                          1'000'000'000.0 /
                          static_cast<double>(
                              measurement.hostNanoseconds))
            << ','
            << "\"allocation_count\":" << measurement.allocations << ','
            << "\"allocation_bytes\":" << measurement.allocatedBytes << ','
            << "\"state_hash\":\"0x" << std::hex
            << measurement.stateHash << std::dec << "\","
            << "\"vu_pc\":\"0x" << std::hex << measurement.vuPc
            << std::dec << "\"}\n";
    }

    uint64_t medianHostTime(std::vector<uint64_t> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    }
}

int main(int argc, char **argv)
{
    Configuration configuration{};
    if (!parseArguments(argc, argv, configuration))
    {
        std::cerr
            << "usage: ee_timing_benchmark "
               "[--blocks N] [--warmup-blocks N] "
               "[--block-ticks N] [--samples N] "
               "[--vu-workload nop|fmac]\n";
        return 2;
    }

    try
    {
        std::vector<uint64_t> eventTimes;
        std::vector<uint64_t> noEventTimes;
        std::vector<uint64_t> directTimes;
        eventTimes.reserve(configuration.samples);
        noEventTimes.reserve(configuration.samples);
        directTimes.reserve(configuration.samples);

        const uint64_t expectedVuPairs =
            committedEeCycles(
                configuration.blocks,
                configuration.blockTicks);
        if (expectedVuPairs >
            std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error(
                "equal-work VU budget exceeds uint32_t");
        }
        uint64_t expectedStateHash = 0u;
        for (uint32_t sample = 0u; sample < configuration.samples; ++sample)
        {
            const Measurement event =
                measureEventEqualWork(
                    sample, configuration);
            if (sample == 0u)
            {
                expectedStateHash = event.stateHash;
            }
            if (event.vuPairs != expectedVuPairs ||
                event.stateHash != expectedStateHash ||
                event.eventServices != 0u)
            {
                throw std::runtime_error(
                    "event benchmark work is not deterministic");
            }
            printMeasurement(event);
            eventTimes.push_back(event.hostNanoseconds);

            const Measurement noEvent =
                measureNoEventHotPath(sample, configuration);
            if (noEvent.eventServices != 0u ||
                noEvent.vuPairs != 0u)
            {
                throw std::runtime_error(
                    "no-event hot path performed device work");
            }
            printMeasurement(noEvent);
            noEventTimes.push_back(
                noEvent.hostNanoseconds);

            const Measurement direct =
                measureDirect(sample, configuration, expectedVuPairs);
            if (direct.vuPairs != expectedVuPairs ||
                direct.stateHash != expectedStateHash)
            {
                throw std::runtime_error(
                    "equal-work direct result does not match event result");
            }
            printMeasurement(direct);
            directTimes.push_back(direct.hostNanoseconds);
        }

        const uint64_t eventMedian =
            medianHostTime(eventTimes);
        const uint64_t noEventMedian =
            medianHostTime(noEventTimes);
        const uint64_t directMedian = medianHostTime(directTimes);
        const double ratio =
            directMedian == 0u
                ? 0.0
                : static_cast<double>(eventMedian) /
                      static_cast<double>(directMedian);
        std::cout
            << "{\"schema_version\":" << kOutputSchemaVersion
            << ",\"event\":\"summary\","
            << "\"vu_workload\":\""
            << vuWorkloadName(configuration.vuWorkload) << "\","
            << "\"samples\":" << configuration.samples << ','
            << "\"ee_blocks\":" << configuration.blocks << ','
            << "\"ee_cycle_ticks\":"
            << committedEeTicks(
                   configuration.blocks,
                   configuration.blockTicks)
            << ','
            << "\"vu_instruction_pairs\":" << expectedVuPairs << ','
            << "\"event_equal_work_median_host_time_ns\":"
            << eventMedian << ','
            << "\"event_no_pending_median_host_time_ns\":"
            << noEventMedian << ','
            << "\"pre_scheduler_equal_work_median_host_time_ns\":"
            << directMedian << ','
            << "\"event_guest_pairs_per_host_second\":"
            << (eventMedian == 0u
                    ? 0.0
                    : static_cast<double>(expectedVuPairs) *
                          1'000'000'000.0 /
                          static_cast<double>(eventMedian))
            << ','
            << "\"pre_scheduler_guest_pairs_per_host_second\":"
            << (directMedian == 0u
                    ? 0.0
                    : static_cast<double>(expectedVuPairs) *
                          1'000'000'000.0 /
                          static_cast<double>(directMedian))
            << ','
            << "\"event_to_pre_scheduler_ratio\":" << ratio
            << "}\n";
    }
    catch (const std::exception &error)
    {
        g_trackAllocations.store(false, std::memory_order_release);
        std::cerr << "ee_timing_benchmark: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
