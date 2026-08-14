#pragma once

#include "runtime/ee_thread_scheduler.h"
#include "runtime/ee_timing.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ps2x::ee
{
    // Values are the deterministic order within one executor boundary.
    // The guest's explicit syscall mutation (including ready rotation) has
    // already completed before the boundary begins. Published host wakes
    // are then made visible, modeled wait deadlines expire, due devices
    // advance, and only afterward are their interrupt causes delivered.
    enum class EeSchedulerConsequenceStage : uint8_t
    {
        AsynchronousWake = 0u,
        WaitTimeout,
        HardwareEvent,
        InterruptCause,
        Count,
    };

    inline constexpr size_t
        kEeSchedulerConsequenceStageCount =
            static_cast<size_t>(
                EeSchedulerConsequenceStage::Count);

    [[nodiscard]] constexpr size_t
    eeSchedulerConsequenceStageIndex(
        EeSchedulerConsequenceStage stage) noexcept
    {
        return static_cast<size_t>(stage);
    }

    [[nodiscard]] constexpr std::string_view
    eeSchedulerConsequenceStageName(
        EeSchedulerConsequenceStage stage) noexcept
    {
        switch (stage)
        {
        case EeSchedulerConsequenceStage::
            AsynchronousWake:
            return "asynchronous_wake";
        case EeSchedulerConsequenceStage::WaitTimeout:
            return "wait_timeout";
        case EeSchedulerConsequenceStage::HardwareEvent:
            return "hardware_event";
        case EeSchedulerConsequenceStage::InterruptCause:
            return "interrupt_cause";
        case EeSchedulerConsequenceStage::Count:
            break;
        }
        return "unknown";
    }

    class IEeSchedulerExecutorHooks
    {
    public:
        virtual ~IEeSchedulerExecutorHooks() = default;

        // The executor advances its canonical timeline before this call.
        // The hook commits the prior context's backend-local work exactly
        // once and must not advance a second timeline.
        virtual void commitPriorContext(
            std::optional<int> priorThreadId,
            ps2x::timing::EeTickDelta elapsed,
            ps2x::timing::EeTick now) = 0;

        // Called for the provisional selection and every selection changed
        // by an immediately runnable consequence.
        virtual void publishSelectedContext(
            std::optional<int> selectedThreadId,
            ps2x::timing::EeTick now) = 0;

        // A selected continuation consumes its retained wait completion
        // immediately before resume. Runtime adapters may mirror the result
        // into their syscall frame/state, but must not reschedule here.
        virtual void publishWaitCompletion(
            int threadId,
            EeSchedulerCompletedWait completion,
            ps2x::timing::EeTick now) = 0;

        [[nodiscard]] virtual bool
        hasImmediateConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick now) const = 0;

        // Implementations may publish thread state only through scheduler
        // transitions. Host work is drained here on the one executor.
        [[nodiscard]] virtual
        EeSchedulerReschedulePolicy applyNextConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick now,
            EeThreadScheduler &scheduler) = 0;
    };

    enum class EeSchedulerExecutorDisposition : uint8_t
    {
        ResumeGuest,
        Paused,
        StopRequested,
    };

    struct EeSchedulerBoundaryResult
    {
        ps2x::timing::EeTick tick{};
        std::optional<int> selectedThreadId;
        size_t consequencesProcessed = 0u;
        size_t contextPublications = 0u;
        bool limitExceeded = false;
        bool reentrantBoundaryRejected = false;
        EeSchedulerExecutorDisposition disposition =
            EeSchedulerExecutorDisposition::ResumeGuest;
        std::optional<EeSchedulerConsequenceStage>
            offendingStage;
        std::array<
            size_t,
            kEeSchedulerConsequenceStageCount>
            stageCounts{};
    };

    struct EeSchedulerExecutorStatistics
    {
        uint64_t boundaries = 0u;
        uint64_t consequences = 0u;
        uint64_t contextPublications = 0u;
        uint64_t consequenceLimitHits = 0u;
        uint64_t reentrantBoundaryRejects = 0u;
        uint64_t pausedBoundaries = 0u;
        uint64_t stoppedBoundaries = 0u;
        uint64_t completedDebugSteps = 0u;
    };

    class EeSchedulerExecutor
    {
    public:
        static constexpr size_t
            kMaximumConsequencesPerBoundary = 1024u;

        [[nodiscard]] EeSchedulerBoundaryResult
        processBoundary(
            EeThreadScheduler &scheduler,
            IEeSchedulerExecutorHooks &hooks,
            std::optional<int> priorThreadId,
            ps2x::timing::EeTickDelta elapsed,
            EeSchedulerReschedulePolicy initialPolicy);

        [[nodiscard]] ps2x::timing::EeTick
        now() const noexcept;
        [[nodiscard]] const EeSchedulerExecutorStatistics &
        statistics() const noexcept;
        // True means this call created a new pause request. False means an
        // existing request is still active and remains idempotent.
        bool debugRequestPause() noexcept;
        void debugResume() noexcept;
        void debugRequestStop() noexcept;
        [[nodiscard]] bool debugStepBoundaries(
            uint64_t count) noexcept;
        [[nodiscard]] bool debugPaused() const noexcept;
        [[nodiscard]] bool debugStopRequested()
            const noexcept;
        [[nodiscard]] uint64_t
        debugStepBoundariesRemaining()
            const noexcept;
        void reset() noexcept;

    private:
        [[nodiscard]] std::optional<
            EeSchedulerConsequenceStage>
        nextImmediateStage(
            const IEeSchedulerExecutorHooks &hooks,
            ps2x::timing::EeTick now) const;
        void applyDebugControl(
            EeSchedulerBoundaryResult &result) noexcept;

        ps2x::timing::EeTimeline m_timeline;
        EeSchedulerExecutorStatistics m_statistics{};
        bool m_processingBoundary = false;
        std::atomic<bool> m_debugPauseRequested{false};
        std::atomic<bool> m_debugStopRequested{false};
        std::atomic<uint64_t>
            m_debugStepBoundariesRemaining{0u};
    };
}
