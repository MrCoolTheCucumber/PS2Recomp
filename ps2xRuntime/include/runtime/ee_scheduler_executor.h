#pragma once

#include "runtime/ee_thread_scheduler.h"
#include "runtime/ee_timing.h"

#include <array>
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

        [[nodiscard]] virtual bool
        hasImmediateConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick now) const = 0;

        // Implementations may publish thread state only through scheduler
        // transitions. Host work is drained here on the one executor.
        virtual void applyNextConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick now,
            EeThreadScheduler &scheduler) = 0;
    };

    struct EeSchedulerBoundaryResult
    {
        ps2x::timing::EeTick tick{};
        std::optional<int> selectedThreadId;
        size_t consequencesProcessed = 0u;
        size_t contextPublications = 0u;
        bool limitExceeded = false;
        bool reentrantBoundaryRejected = false;
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
        void reset() noexcept;

    private:
        [[nodiscard]] std::optional<
            EeSchedulerConsequenceStage>
        nextImmediateStage(
            const IEeSchedulerExecutorHooks &hooks,
            ps2x::timing::EeTick now) const;

        ps2x::timing::EeTimeline m_timeline;
        EeSchedulerExecutorStatistics m_statistics{};
        bool m_processingBoundary = false;
    };
}
