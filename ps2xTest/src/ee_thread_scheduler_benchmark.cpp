#include "ps2_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sys/resource.h>
#endif

namespace
{
    constexpr uint32_t kOutputSchemaVersion = 1u;

    std::atomic<bool> g_trackAllocations{false};
    std::atomic<uint64_t> g_allocationCount{0u};
    std::atomic<uint64_t> g_allocationBytes{0u};

    void recordAllocation(std::size_t size) noexcept
    {
        if (!g_trackAllocations.load(std::memory_order_relaxed))
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
        {
            throw std::bad_alloc();
        }
        recordAllocation(size);
        return result;
    }

    void *allocateAligned(
        std::size_t size,
        std::size_t alignment)
    {
        size = std::max<std::size_t>(size, 1u);
        void *result = nullptr;
#if defined(_WIN32)
        result = _aligned_malloc(size, alignment);
        if (!result)
        {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&result, alignment, size) != 0)
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
    std::size_t size,
    const std::nothrow_t &) noexcept
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
    std::size_t size,
    const std::nothrow_t &) noexcept
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
    std::size_t size,
    std::align_val_t alignment)
{
    return allocateAligned(
        size, static_cast<std::size_t>(alignment));
}

void *operator new[](
    std::size_t size,
    std::align_val_t alignment)
{
    return allocateAligned(
        size, static_cast<std::size_t>(alignment));
}

void *operator new(
    std::size_t size,
    std::align_val_t alignment,
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
    std::size_t size,
    std::align_val_t alignment,
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

void operator delete(void *pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void *pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete(
    void *pointer,
    const std::nothrow_t &) noexcept
{
    std::free(pointer);
}

void operator delete[](
    void *pointer,
    const std::nothrow_t &) noexcept
{
    std::free(pointer);
}

void operator delete(
    void *pointer,
    std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete[](
    void *pointer,
    std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete(
    void *pointer,
    std::size_t,
    std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete[](
    void *pointer,
    std::size_t,
    std::align_val_t) noexcept
{
    freeAligned(pointer);
}

void operator delete(
    void *pointer,
    std::align_val_t,
    const std::nothrow_t &) noexcept
{
    freeAligned(pointer);
}

void operator delete[](
    void *pointer,
    std::align_val_t,
    const std::nothrow_t &) noexcept
{
    freeAligned(pointer);
}

namespace
{
    struct Configuration
    {
        uint64_t switches = 10'000u;
        uint32_t guestInstructionsPerBoundary = 64u;
        uint32_t guestCyclesPerBoundary = 64u;
        bool diagnostics = true;
    };

    struct Measurement
    {
        PS2Runtime::DebugEeThreadDiagnostics diagnostics{};
        uint64_t wallNanoseconds = 0u;
        uint64_t processCpuNanoseconds = 0u;
        uint64_t allocationCount = 0u;
        uint64_t allocationBytes = 0u;
        uint64_t hostStackCapacityBytes = 0u;
        uint64_t peakRssKiB = 0u;
        uint64_t guestInstructions = 0u;
        uint64_t guestCycles = 0u;
        uint64_t stateHash = 0u;
    };

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
        const auto result =
            std::from_chars(begin, end, value);
        return result.ec == std::errc{} &&
               result.ptr == end;
    }

    bool parseArguments(
        int argc,
        char **argv,
        Configuration &configuration)
    {
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view argument(argv[index]);
            if (argument == "--switches" &&
                index + 1 < argc)
            {
                if (!parseUnsigned(
                        argv[++index],
                        configuration.switches))
                {
                    return false;
                }
            }
            else if (argument == "--guest-instructions" &&
                     index + 1 < argc)
            {
                uint64_t value = 0u;
                if (!parseUnsigned(argv[++index], value) ||
                    value >
                        std::numeric_limits<uint32_t>::max())
                {
                    return false;
                }
                configuration.guestInstructionsPerBoundary =
                    static_cast<uint32_t>(value);
            }
            else if (argument == "--guest-cycles" &&
                     index + 1 < argc)
            {
                uint64_t value = 0u;
                if (!parseUnsigned(argv[++index], value) ||
                    value >
                        std::numeric_limits<uint32_t>::max() /
                            ps2x::timing::kEeTicksPerCycle)
                {
                    return false;
                }
                configuration.guestCyclesPerBoundary =
                    static_cast<uint32_t>(value);
            }
            else if (argument == "--diagnostics" &&
                     index + 1 < argc)
            {
                const std::string_view value(argv[++index]);
                if (value == "on")
                {
                    configuration.diagnostics = true;
                }
                else if (value == "off")
                {
                    configuration.diagnostics = false;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        return configuration.switches >= 2u &&
               (configuration.switches & 1u) == 0u;
    }

    uint64_t processCpuNanoseconds()
    {
        const std::clock_t ticks = std::clock();
        if (ticks == static_cast<std::clock_t>(-1))
        {
            return 0u;
        }
        return static_cast<uint64_t>(
            static_cast<long double>(ticks) *
            1'000'000'000.0L /
            static_cast<long double>(CLOCKS_PER_SEC));
    }

    uint64_t currentThreadStackCapacity()
    {
#if defined(__linux__)
        pthread_attr_t attributes{};
        if (pthread_getattr_np(
                pthread_self(), &attributes) != 0)
        {
            return 0u;
        }
        size_t size = 0u;
        const int result =
            pthread_attr_getstacksize(&attributes, &size);
        pthread_attr_destroy(&attributes);
        return result == 0
                   ? static_cast<uint64_t>(size)
                   : 0u;
#else
        return 0u;
#endif
    }

    uint64_t peakRssKiB()
    {
#if defined(__linux__)
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            return static_cast<uint64_t>(usage.ru_maxrss);
        }
#endif
        return 0u;
    }

    uint64_t mixGuestWork(
        uint64_t state,
        uint32_t instructions)
    {
        for (uint32_t index = 0u;
             index < instructions;
             ++index)
        {
            state ^=
                state >> 12u;
            state ^=
                state << 25u;
            state ^=
                state >> 27u;
            state *= 0x2545F4914F6CDD1Dull;
        }
        return state;
    }

    PS2Runtime::DebugEeThreadDiagnostics subtract(
        const PS2Runtime::DebugEeThreadDiagnostics &after,
        const PS2Runtime::DebugEeThreadDiagnostics &before)
    {
        PS2Runtime::DebugEeThreadDiagnostics delta{};
        delta.enabled = after.enabled;
#define PS2X_SUBTRACT_COUNTER(name) \
        delta.name = after.name - before.name
        PS2X_SUBTRACT_COUNTER(guestLockRequests);
        PS2X_SUBTRACT_COUNTER(guestLockAcquisitions);
        PS2X_SUBTRACT_COUNTER(guestLockContentions);
        PS2X_SUBTRACT_COUNTER(outerGuestExecutionAcquisitions);
        PS2X_SUBTRACT_COUNTER(guestContextChanges);
        PS2X_SUBTRACT_COUNTER(handoffNotifications);
        PS2X_SUBTRACT_COUNTER(handoffWaitRequests);
        PS2X_SUBTRACT_COUNTER(handoffWaitFastPaths);
        PS2X_SUBTRACT_COUNTER(handoffCvWaits);
        PS2X_SUBTRACT_COUNTER(handoffCompletions);
        PS2X_SUBTRACT_COUNTER(handoffTimeouts);
        PS2X_SUBTRACT_COUNTER(yieldRequests);
        PS2X_SUBTRACT_COUNTER(deferredYields);
        PS2X_SUBTRACT_COUNTER(hostThreadYields);
        PS2X_SUBTRACT_COUNTER(requestedGuestSwitches);
        PS2X_SUBTRACT_COUNTER(guestSwitchCvWaits);
        PS2X_SUBTRACT_COUNTER(completedGuestSwitches);
        PS2X_SUBTRACT_COUNTER(guestSwitchTimeouts);
        PS2X_SUBTRACT_COUNTER(rotationRequests);
        PS2X_SUBTRACT_COUNTER(acceptedRotationRequests);
        PS2X_SUBTRACT_COUNTER(rejectedRotationRequests);
        PS2X_SUBTRACT_COUNTER(priorityZeroRotationRequests);
        PS2X_SUBTRACT_COUNTER(untrackedThreadRotationRequests);
#undef PS2X_SUBTRACT_COUNTER
        for (size_t index = 0u;
             index < delta.acceptedRotationsByPriority.size();
             ++index)
        {
            delta.acceptedRotationsByPriority[index] =
                after.acceptedRotationsByPriority[index] -
                before.acceptedRotationsByPriority[index];
        }
        for (size_t index = 0u;
             index < delta.acceptedRotationsByThread.size();
             ++index)
        {
            delta.acceptedRotationsByThread[index] =
                after.acceptedRotationsByThread[index] -
                before.acceptedRotationsByThread[index];
        }
        return delta;
    }

    Measurement measure(const Configuration &configuration)
    {
        PS2RuntimeConfiguration runtimeConfiguration{};
        runtimeConfiguration.eeThreadDiagnostics =
            configuration.diagnostics;
        runtimeConfiguration.useEeThreadDiagnosticsEnvironment =
            false;
        PS2Runtime runtime(runtimeConfiguration);
        std::array<R5900Context, 2u> contexts{};
        std::array<uint64_t, 2u> guestStates{
            0x123456789abcdef0ull,
            0xfedcba9876543210ull};
        std::array<std::atomic<uint64_t>, 2u>
            hostStackCapacities{};
        std::atomic<bool> firstHolding{false};
        std::atomic<bool> secondReady{false};
        std::atomic<bool> go{false};
        std::atomic<bool> abort{false};
        std::atomic<uint32_t> turn{0u};
        std::atomic<uint32_t> failure{0u};
        std::mutex completionMutex;
        std::condition_variable completionCv;
        uint32_t finished = 0u;
        const uint64_t iterations =
            configuration.switches / 2u;

        const auto executeTurns =
            [&](uint32_t worker)
        {
            uint64_t state = guestStates[worker];
            for (uint64_t iteration = 0u;
                 iteration < iterations &&
                 !abort.load(std::memory_order_acquire);
                 ++iteration)
            {
                if (turn.load(std::memory_order_acquire) !=
                    worker)
                {
                    failure.store(
                        1u, std::memory_order_release);
                    abort.store(
                        true, std::memory_order_release);
                    break;
                }

                state = mixGuestWork(
                    state,
                    configuration
                        .guestInstructionsPerBoundary);
                contexts[worker].insn_count +=
                    configuration
                        .guestInstructionsPerBoundary;
                contexts[worker].advanceEeCycleTicks(
                    configuration
                        .guestCyclesPerBoundary *
                    ps2x::timing::kEeTicksPerCycle);
                turn.store(
                    1u - worker,
                    std::memory_order_release);
                runtime.yieldGuestExecutionAfterWake();
            }
            guestStates[worker] = state;
        };

        const auto runWorker =
            [&](uint32_t worker)
        {
            hostStackCapacities[worker].store(
                currentThreadStackCapacity(),
                std::memory_order_release);
            if (worker == 0u)
            {
                {
                    PS2Runtime::GuestExecutionScope scope(
                        &runtime, &contexts[worker]);
                    firstHolding.store(
                        true, std::memory_order_release);
                    while (!go.load(std::memory_order_acquire) &&
                           !abort.load(std::memory_order_acquire))
                    {
                        std::this_thread::yield();
                    }
                    while (runtime
                               .guestExecutionWaiterCountForTesting() ==
                               0u &&
                           !abort.load(std::memory_order_acquire))
                    {
                        std::this_thread::yield();
                    }
                    executeTurns(worker);
                }
            }
            else
            {
                secondReady.store(
                    true, std::memory_order_release);
                while (!go.load(std::memory_order_acquire) &&
                       !abort.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                if (!abort.load(std::memory_order_acquire))
                {
                    PS2Runtime::GuestExecutionScope scope(
                        &runtime, &contexts[worker]);
                    executeTurns(worker);
                }
            }
            {
                std::lock_guard<std::mutex> lock(
                    completionMutex);
                ++finished;
            }
            completionCv.notify_one();
        };

        std::thread first(runWorker, 0u);
        const auto startupDeadline =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(5);
        while (!firstHolding.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() <
                   startupDeadline)
        {
            std::this_thread::yield();
        }
        if (!firstHolding.load(std::memory_order_acquire))
        {
            abort.store(true, std::memory_order_release);
            go.store(true, std::memory_order_release);
            first.join();
            throw std::runtime_error(
                "first worker did not acquire guest execution");
        }

        std::thread second(runWorker, 1u);
        while (!secondReady.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() <
                   startupDeadline)
        {
            std::this_thread::yield();
        }
        if (!secondReady.load(std::memory_order_acquire))
        {
            abort.store(true, std::memory_order_release);
            go.store(true, std::memory_order_release);
            first.join();
            second.join();
            throw std::runtime_error(
                "second worker did not reach the start barrier");
        }

        const auto before =
            runtime.debugEeThreadDiagnosticsSnapshot();
        g_allocationCount.store(0u, std::memory_order_relaxed);
        g_allocationBytes.store(0u, std::memory_order_relaxed);
        g_trackAllocations.store(true, std::memory_order_release);
        const uint64_t cpuStart = processCpuNanoseconds();
        const auto wallStart =
            std::chrono::steady_clock::now();
        go.store(true, std::memory_order_release);

        const auto completionDeadline =
            wallStart + std::chrono::seconds(30);
        {
            std::unique_lock<std::mutex> lock(
                completionMutex);
            if (!completionCv.wait_until(
                    lock,
                    completionDeadline,
                    [&]()
                    {
                        return finished == 2u;
                    }))
            {
                failure.store(2u, std::memory_order_release);
                abort.store(true, std::memory_order_release);
                runtime.requestStop();
            }
        }

        first.join();
        second.join();
        const auto wallEnd =
            std::chrono::steady_clock::now();
        const uint64_t cpuEnd = processCpuNanoseconds();
        g_trackAllocations.store(false, std::memory_order_release);
        if (failure.load(std::memory_order_acquire) != 0u)
        {
            throw std::runtime_error(
                failure.load(std::memory_order_relaxed) == 1u
                    ? "guest handoff violated strict alternation"
                    : "guest handoff benchmark timed out");
        }

        const auto after =
            runtime.debugEeThreadDiagnosticsSnapshot();
        const PS2Runtime::DebugEeTiming timing =
            runtime.debugEeTimingSnapshot();

        Measurement result{};
        result.diagnostics = subtract(after, before);
        result.wallNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    wallEnd - wallStart)
                    .count());
        result.processCpuNanoseconds =
            cpuEnd - cpuStart;
        result.allocationCount =
            g_allocationCount.load(
                std::memory_order_relaxed);
        result.allocationBytes =
            g_allocationBytes.load(
                std::memory_order_relaxed);
        result.hostStackCapacityBytes =
            hostStackCapacities[0u].load(
                std::memory_order_acquire) +
            hostStackCapacities[1u].load(
                std::memory_order_acquire);
        result.peakRssKiB = peakRssKiB();
        result.guestInstructions =
            contexts[0u].insn_count +
            contexts[1u].insn_count;
        result.guestCycles = timing.currentCycle;
        result.stateHash =
            guestStates[0u] ^
            (guestStates[1u] +
             0x9e3779b97f4a7c15ull +
             (guestStates[0u] << 6u) +
             (guestStates[0u] >> 2u));
        return result;
    }

    void validate(
        const Configuration &configuration,
        const Measurement &measurement)
    {
        const uint64_t expectedInstructions =
            configuration.switches *
            configuration.guestInstructionsPerBoundary;
        const uint64_t expectedCycles =
            configuration.switches *
            configuration.guestCyclesPerBoundary;
        if (measurement.guestInstructions !=
                expectedInstructions ||
            measurement.guestCycles != expectedCycles)
        {
            throw std::runtime_error(
                "fixed guest work did not reconcile");
        }
        if (configuration.diagnostics &&
            (measurement.diagnostics
                     .requestedGuestSwitches !=
                 configuration.switches ||
             measurement.diagnostics
                     .guestSwitchCvWaits !=
                 configuration.switches ||
             measurement.diagnostics
                     .completedGuestSwitches !=
                 configuration.switches ||
             measurement.diagnostics
                     .guestSwitchTimeouts != 0u))
        {
            throw std::runtime_error(
                "requested and completed guest switches did not reconcile");
        }
    }

    void printMeasurement(
        const Configuration &configuration,
        const Measurement &measurement)
    {
        const double wallNanosecondsPerSwitch =
            static_cast<double>(
                measurement.wallNanoseconds) /
            static_cast<double>(configuration.switches);
        const double cpuNanosecondsPerSwitch =
            static_cast<double>(
                measurement.processCpuNanoseconds) /
            static_cast<double>(configuration.switches);
        const auto &diagnostics =
            measurement.diagnostics;
        std::cout
            << "{\"schema_version\":"
            << kOutputSchemaVersion
            << ",\"event\":\"summary\""
            << ",\"backend\":\"legacy-host-thread\""
            << ",\"diagnostics_enabled\":"
            << (configuration.diagnostics
                    ? "true"
                    : "false")
            << ",\"requested_switches\":"
            << configuration.switches
            << ",\"completed_switches\":"
            << (configuration.diagnostics
                    ? diagnostics.completedGuestSwitches
                    : 0u)
            << ",\"yield_returns\":"
            << configuration.switches
            << ",\"wall_time_ns\":"
            << measurement.wallNanoseconds
            << ",\"process_cpu_time_ns\":"
            << measurement.processCpuNanoseconds
            << ",\"wall_time_ns_per_switch\":"
            << wallNanosecondsPerSwitch
            << ",\"process_cpu_time_ns_per_switch\":"
            << cpuNanosecondsPerSwitch
            << ",\"guest_lock_requests\":"
            << diagnostics.guestLockRequests
            << ",\"guest_lock_acquisitions\":"
            << diagnostics.guestLockAcquisitions
            << ",\"guest_lock_contentions\":"
            << diagnostics.guestLockContentions
            << ",\"outer_guest_execution_acquisitions\":"
            << diagnostics.outerGuestExecutionAcquisitions
            << ",\"guest_context_changes\":"
            << diagnostics.guestContextChanges
            << ",\"handoff_notifications\":"
            << diagnostics.handoffNotifications
            << ",\"handoff_wait_requests\":"
            << diagnostics.handoffWaitRequests
            << ",\"handoff_cv_waits\":"
            << diagnostics.handoffCvWaits
            << ",\"handoff_timeouts\":"
            << diagnostics.handoffTimeouts
            << ",\"requested_guest_switches\":"
            << diagnostics.requestedGuestSwitches
            << ",\"guest_switch_cv_waits\":"
            << diagnostics.guestSwitchCvWaits
            << ",\"completed_guest_switches\":"
            << diagnostics.completedGuestSwitches
            << ",\"guest_switch_timeouts\":"
            << diagnostics.guestSwitchTimeouts
            << ",\"allocation_count\":"
            << measurement.allocationCount
            << ",\"allocation_bytes\":"
            << measurement.allocationBytes
            << ",\"host_thread_count\":2"
            << ",\"host_stack_capacity_bytes\":"
            << measurement.hostStackCapacityBytes
            << ",\"fiber_stack_capacity_bytes\":0"
            << ",\"peak_rss_kib\":"
            << measurement.peakRssKiB
            << ",\"guest_instructions_per_boundary\":"
            << configuration.guestInstructionsPerBoundary
            << ",\"guest_cycles_per_boundary\":"
            << configuration.guestCyclesPerBoundary
            << ",\"guest_instructions\":"
            << measurement.guestInstructions
            << ",\"guest_cycles\":"
            << measurement.guestCycles
            << ",\"state_hash\":\"0x"
            << std::hex << measurement.stateHash
            << std::dec << "\"}\n";
    }
}

int main(int argc, char **argv)
{
    Configuration configuration{};
    if (!parseArguments(
            argc, argv, configuration))
    {
        std::cerr
            << "usage: ee_thread_scheduler_benchmark "
               "[--switches EVEN_N] "
               "[--guest-instructions N] "
               "[--guest-cycles N] "
               "[--diagnostics on|off]\n";
        return 2;
    }

    try
    {
        const Measurement measurement =
            measure(configuration);
        validate(configuration, measurement);
        printMeasurement(configuration, measurement);
    }
    catch (const std::exception &error)
    {
        g_trackAllocations.store(
            false, std::memory_order_release);
        std::cerr
            << "ee_thread_scheduler_benchmark: "
            << error.what() << '\n';
        return 1;
    }
    return 0;
}
