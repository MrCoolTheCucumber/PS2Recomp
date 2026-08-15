#include "runtime/ee_scheduler_executor.h"
#include "runtime/ps2_build_config.h"

#include <stdexcept>

namespace ps2x::ee
{
    void EeSchedulerExecutor::requireOwner()
    {
        const std::thread::id current =
            std::this_thread::get_id();
        if (m_ownerThreadId == std::thread::id{})
        {
            m_ownerThreadId = current;
            return;
        }
        if (m_ownerThreadId != current)
        {
            throw std::logic_error(
                "EE scheduler boundary requires its owner "
                "thread");
        }
    }

    void EeSchedulerExecutor::
        validateOwnerLocalTransition(
            const EeThreadScheduler &scheduler,
            const EeSchedulerBoundaryInput &input) const
    {
        const EeSchedulerOwnerLocalTransition &transition =
            input.ownerLocalTransition;
        if (!transition.structurallyValid())
        {
            throw std::logic_error(
                "EE scheduler boundary received an invalid "
                "owner-local transition");
        }
        if (!transition.pending())
        {
            return;
        }
        if (input.initialPolicy !=
            EeSchedulerReschedulePolicy::None)
        {
            throw std::logic_error(
                "an owner-local transition requires a "
                "suppressed provisional reschedule");
        }
        if (!input.priorThreadId.has_value() ||
            *input.priorThreadId !=
                transition.originatingThread.id ||
            scheduler.currentThreadId() !=
                input.priorThreadId)
        {
            throw std::logic_error(
                "an owner-local transition does not match "
                "the exiting scheduler owner");
        }
        const auto currentHandle =
            scheduler.threadHandle(
                transition.originatingThread.id);
        if (!currentHandle.has_value() ||
            *currentHandle !=
                transition.originatingThread)
        {
            throw std::logic_error(
                "an owner-local transition has a stale "
                "originating thread handle");
        }
    }

    std::optional<int>
    EeSchedulerExecutor::applyOwnerLocalTransition(
        EeThreadScheduler &scheduler,
        const EeSchedulerOwnerLocalTransition &transition)
    {
        switch (transition.kind)
        {
        case EeSchedulerOwnerLocalTransitionKind::
            RotateReadyQueue:
        {
            const auto origin =
                scheduler.thread(
                    transition.originatingThread.id);
            if (!origin.has_value() ||
                origin->generation !=
                    transition.originatingThread
                        .generation)
            {
                throw std::logic_error(
                    "an owner-local ready-queue rotation "
                    "lost its originating thread");
            }
            const bool currentPriority =
                origin->priority == transition.priority;
            if (!currentPriority &&
                !scheduler.rotateReadyQueue(
                    transition.priority))
            {
                throw std::logic_error(
                    "an owner-local ready-queue rotation "
                    "violated a scheduler invariant");
            }
            break;
        }
        case EeSchedulerOwnerLocalTransitionKind::None:
            throw std::logic_error(
                "an empty owner-local transition cannot be "
                "applied");
        default:
            throw std::logic_error(
                "unknown owner-local scheduler transition");
        }

        const std::optional<int> selected =
            scheduler.reschedule(
                transition.reschedulePolicy);
        if (!selected.has_value())
        {
            throw std::logic_error(
                "an owner-local transition lost the "
                "selected scheduler context");
        }
        return selected;
    }

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
        EeSchedulerBoundaryInput input)
    {
        requireOwner();
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
            validateOwnerLocalTransition(
                scheduler, input);
            result.tick =
                m_timeline.advance(input.elapsed);
            scheduler.publishCanonicalTick(
                result.tick);
            hooks.commitPriorContext(
                input.priorThreadId,
                input.elapsed,
                result.tick);

            if (input.ownerLocalTransition.pending())
            {
                result.selectedThreadId =
                    applyOwnerLocalTransition(
                        scheduler,
                        input.ownerLocalTransition);
                result.ownerLocalTransitionsApplied = 1u;
            }
            else
            {
                result.selectedThreadId =
                    scheduler.reschedule(
                        input.initialPolicy);
            }
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
        m_statistics.ownerLocalTransitionsApplied +=
            result.ownerLocalTransitionsApplied;
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

    EeSchedulerBoundaryResult
    EeSchedulerExecutor::processBoundary(
        EeThreadScheduler &scheduler,
        IEeSchedulerExecutorHooks &hooks,
        std::optional<int> priorThreadId,
        ps2x::timing::EeTickDelta elapsed,
        EeSchedulerReschedulePolicy initialPolicy)
    {
        return processBoundary(
            scheduler,
            hooks,
            EeSchedulerBoundaryInput{
                priorThreadId,
                elapsed,
                initialPolicy,
                {},
            });
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

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
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
#endif
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

    bool EeSchedulerExecutor::debugRequestPause()
        noexcept
    {
        m_debugStepBoundariesRemaining.store(
            0u, std::memory_order_release);
        return !m_debugPauseRequested.exchange(
            true, std::memory_order_acq_rel);
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
        m_ownerThreadId = {};
    }
}
