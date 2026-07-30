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
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly);
                if (selectionBefore !=
                    result.selectedThreadId)
                {
                    hooks.publishSelectedContext(
                        result.selectedThreadId,
                        result.tick);
                    ++result.contextPublications;
                }
            }

            result.offendingStage =
                nextImmediateStage(
                    hooks, result.tick);
            result.limitExceeded =
                result.offendingStage.has_value();
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

    void EeSchedulerExecutor::reset() noexcept
    {
        if (m_processingBoundary)
        {
            return;
        }
        m_timeline.reset();
        m_statistics = {};
    }
}
