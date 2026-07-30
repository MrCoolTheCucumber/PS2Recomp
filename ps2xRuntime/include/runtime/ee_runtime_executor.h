#pragma once

#include "runtime/ee_execution_backend.h"
#include "runtime/ee_scheduler_executor.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <thread>

namespace ps2x::ee
{
    struct EeRuntimeExecutorStatistics
    {
        uint64_t starts = 0u;
        uint64_t dispatches = 0u;
        uint64_t publicationsQueued = 0u;
        uint64_t publicationsApplied = 0u;
        uint64_t idleWaits = 0u;
        uint64_t continuationsDestroyed = 0u;
        uint64_t failures = 0u;
        size_t peakQueuedPublications = 0u;
        ps2x::timing::EeTick modeledTick{};
        EeSchedulerExecutorStatistics boundary{};
    };

    // Owns the one host thread which is allowed to resume executor-backed EE
    // continuations. Host workers may enqueue state publications, but those
    // publications are applied only at a scheduler boundary after the prior
    // continuation's exit has committed its RUN/WAIT/DORMANT transition.
    //
    // This class deliberately does not support legacy-host-thread. That
    // backend remains a behavior oracle while its existing worker lifecycle
    // is migrated; silently recreating one worker per guest here would break
    // the single-executor contract.
    class EeRuntimeExecutor final
    {
    public:
        using Command = std::function<void(
            EeThreadScheduler &,
            IEeExecutionBackend &)>;

        struct Options
        {
            uint64_t tickBudget = 4096u;
        };

        explicit EeRuntimeExecutor(
            IEeExecutionBackend &backend,
            IEeSchedulerExecutorHooks *hooks = nullptr);
        EeRuntimeExecutor(
            IEeExecutionBackend &backend,
            IEeSchedulerExecutorHooks *hooks,
            Options options);
        ~EeRuntimeExecutor() noexcept;

        EeRuntimeExecutor(const EeRuntimeExecutor &) = delete;
        EeRuntimeExecutor &operator=(
            const EeRuntimeExecutor &) = delete;
        EeRuntimeExecutor(EeRuntimeExecutor &&) = delete;
        EeRuntimeExecutor &operator=(
            EeRuntimeExecutor &&) = delete;

        // Starts a fresh scheduler on a dedicated host thread. setup runs on
        // that executor stack and may create backend continuations and seed
        // scheduler records. This call returns after setup has completed.
        void start(Command setup);

        // Queue a host publication without running guest code on the caller.
        // False means stop/failure has closed the executor boundary.
        [[nodiscard]] bool publish(Command command);

        // Queue one command and wait until the executor applies it at a
        // boundary. This is intended for lifecycle operations and bounded
        // tests, not for a guest continuation running on the executor thread.
        void invokeAtBoundary(Command command);

        // Wake an idle executor after an external hook published work without
        // using publish(). The hook still applies that work at its declared
        // consequence stage.
        void notify() noexcept;

        void debugRequestPause() noexcept;
        void debugResume() noexcept;
        [[nodiscard]] bool debugStepBoundaries(
            uint64_t count) noexcept;
        [[nodiscard]] bool debugPaused() const noexcept;

        // Stop is cooperative while a guest continuation is running. The
        // backend-neutral checkpoint is responsible for returning that guest
        // to the executor; idle and paused executors stop immediately.
        void requestStop() noexcept;
        void join();

        [[nodiscard]] bool running() const noexcept;
        [[nodiscard]] bool
        acceptingPublications() const noexcept;
        [[nodiscard]] std::thread::id
        executorThreadId() const noexcept;
        [[nodiscard]] EeRuntimeExecutorStatistics
        statistics() const noexcept;
        [[nodiscard]] std::exception_ptr
        failure() const noexcept;
        void rethrowFailure() const;

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
