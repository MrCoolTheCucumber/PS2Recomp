#include "runtime/ee_scheduler_executor.h"

namespace ps2x::ee
{
    std::optional<EeSchedulerConsequenceStage>
    EeSchedulerExecutor::nextImmediateStage(
        const IEeSchedulerExecutorHooks &hooks,
        ps2x::timing::EeTick now) const
    {
        for (size_t stageIndex = 0u;
             stageIndex <
             kEeSchedulerConsequenceStageCount;
             ++stageIndex)
        {
            const auto stage =
                static_cast<
                    EeSchedulerConsequenceStage>(
                    stageIndex);
            if (hooks.hasImmediateConsequence(
                    stage, now))
            {
                return stage;
            }
        }
        return std::nullopt;
    }

    EeSchedulerBoundaryResult
    EeSchedulerExecutor::processBoundary(
        EeThreadScheduler &scheduler,
        IEeSchedulerExecutorHooks &hooks,
        std::optional<int> priorThreadId,
        ps2x::timing::EeTickDelta elapsed,
        EeSchedulerReschedulePolicy initialPolicy)
    {
        EeSchedulerBoundaryResult result{};
        result.tick = m_timeline.now();
        result.selectedThreadId =
            scheduler.currentThreadId();
        if (m_processingBoundary)
        {
            result.reentrantBoundaryRejected = true;
            ++m_statistics.reentrantBoundaryRejects;
            return result;
        }

        m_processingBoundary = true;
        try
        {
            result.tick =
                m_timeline.advance(elapsed);
            scheduler.publishCanonicalTick(
                result.tick);
            hooks.commitPriorContext(
                priorThreadId,
                elapsed,
                result.tick);

            result.selectedThreadId =
                scheduler.reschedule(initialPolicy);
            hooks.publishSelectedContext(
                result.selectedThreadId,
                result.tick);
            ++result.contextPublications;

            while (
                result.consequencesProcessed <
                kMaximumConsequencesPerBoundary)
            {
                const auto stage =
                    nextImmediateStage(
                        hooks, result.tick);
                if (!stage.has_value())
                {
                    break;
                }

                const std::optional<int>
                    selectionBefore =
                        scheduler.currentThreadId();
                const EeSchedulerReschedulePolicy
                    consequencePolicy =
                        hooks.applyNextConsequence(
                            *stage,
                            result.tick,
                            scheduler);
                ++result.consequencesProcessed;
                ++result.stageCounts[
                    eeSchedulerConsequenceStageIndex(
                        *stage)];

                result.selectedThreadId =
                    scheduler.reschedule(
                        consequencePolicy);
                if (selectionBefore !=
                    result.selectedThreadId)
                {
                    hooks.publishSelectedContext(
                        result.selectedThreadId,
                        result.tick);
                    ++result.contextPublications;
                }
            }

            if (result.consequencesProcessed ==
                kMaximumConsequencesPerBoundary)
            {
                result.offendingStage =
                    nextImmediateStage(
                        hooks, result.tick);
                result.limitExceeded =
                    result.offendingStage.has_value();
            }
            applyDebugControl(result);
        }
        catch (...)
        {
            m_processingBoundary = false;
            throw;
        }
        m_processingBoundary = false;

        ++m_statistics.boundaries;
        m_statistics.consequences +=
            result.consequencesProcessed;
        m_statistics.contextPublications +=
            result.contextPublications;
        if (result.limitExceeded)
        {
            ++m_statistics.consequenceLimitHits;
        }
        return result;
    }

    void EeSchedulerExecutor::applyDebugControl(
        EeSchedulerBoundaryResult &result) noexcept
    {
        if (m_debugStopRequested.load(
                std::memory_order_acquire))
        {
            result.disposition =
                EeSchedulerExecutorDisposition::
                    StopRequested;
            ++m_statistics.stoppedBoundaries;
            return;
        }

        if (m_debugPauseRequested.load(
                std::memory_order_acquire))
        {
            result.disposition =
                EeSchedulerExecutorDisposition::Paused;
            ++m_statistics.pausedBoundaries;
            return;
        }

        uint64_t remaining =
            m_debugStepBoundariesRemaining.load(
                std::memory_order_acquire);
        while (remaining != 0u)
        {
            if (m_debugStepBoundariesRemaining
                    .compare_exchange_weak(
                        remaining,
                        remaining - 1u,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
            {
                if (remaining == 1u)
                {
                    m_debugPauseRequested.store(
                        true,
                        std::memory_order_release);
                    result.disposition =
                        EeSchedulerExecutorDisposition::
                            Paused;
                    ++m_statistics.pausedBoundaries;
                    ++m_statistics.completedDebugSteps;
                }
                return;
            }
        }
    }

    ps2x::timing::EeTick
    EeSchedulerExecutor::now() const noexcept
    {
        return m_timeline.now();
    }

    const EeSchedulerExecutorStatistics &
    EeSchedulerExecutor::statistics()
        const noexcept
    {
        return m_statistics;
    }

    void EeSchedulerExecutor::debugRequestPause()
        noexcept
    {
        m_debugStepBoundariesRemaining.store(
            0u, std::memory_order_release);
        m_debugPauseRequested.store(
            true, std::memory_order_release);
    }

    void EeSchedulerExecutor::debugResume() noexcept
    {
        m_debugStepBoundariesRemaining.store(
            0u, std::memory_order_release);
        m_debugPauseRequested.store(
            false, std::memory_order_release);
    }

    void EeSchedulerExecutor::debugRequestStop()
        noexcept
    {
        m_debugStopRequested.store(
            true, std::memory_order_release);
    }

    bool EeSchedulerExecutor::debugStepBoundaries(
        uint64_t count) noexcept
    {
        if (count == 0u ||
            m_debugStopRequested.load(
                std::memory_order_acquire))
        {
            return false;
        }

        m_debugStepBoundariesRemaining.store(
            count, std::memory_order_release);
        m_debugPauseRequested.store(
            false, std::memory_order_release);
        return true;
    }

    bool EeSchedulerExecutor::debugPaused()
        const noexcept
    {
        return m_debugPauseRequested.load(
            std::memory_order_acquire);
    }

    bool EeSchedulerExecutor::debugStopRequested()
        const noexcept
    {
        return m_debugStopRequested.load(
            std::memory_order_acquire);
    }

    uint64_t
    EeSchedulerExecutor::
        debugStepBoundariesRemaining()
            const noexcept
    {
        return m_debugStepBoundariesRemaining.load(
            std::memory_order_acquire);
    }

    void EeSchedulerExecutor::reset() noexcept
    {
        if (m_processingBoundary)
        {
            return;
        }
        m_timeline.reset();
        m_statistics = {};
        m_debugStepBoundariesRemaining.store(
            0u, std::memory_order_release);
        m_debugPauseRequested.store(
            false, std::memory_order_release);
        m_debugStopRequested.store(
            false, std::memory_order_release);
    }
}
