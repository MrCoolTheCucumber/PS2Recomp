#include "runtime/ee_thread_scheduler.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace ps2x::ee
{
    namespace
    {
        [[nodiscard]]
        EeSchedulerTransitionReason
        transitionReasonForWaitCompletion(
            EeSchedulerWaitCompletion completion)
        {
            switch (completion)
            {
            case EeSchedulerWaitCompletion::Satisfied:
                return EeSchedulerTransitionReason::
                    WaitSatisfied;
            case EeSchedulerWaitCompletion::Released:
                return EeSchedulerTransitionReason::
                    WaitReleased;
            case EeSchedulerWaitCompletion::
                ObjectDeleted:
                return EeSchedulerTransitionReason::
                    WaitObjectDeleted;
            case EeSchedulerWaitCompletion::TimedOut:
                return EeSchedulerTransitionReason::
                    WaitTimedOut;
            case EeSchedulerWaitCompletion::None:
                break;
            }
            return EeSchedulerTransitionReason::
                WaitReleased;
        }
    }

    bool EeSchedulerWaitKey::valid() const noexcept
    {
        switch (kind)
        {
        case EeSchedulerWaitKind::Sleep:
            return objectId == 0;
        case EeSchedulerWaitKind::Semaphore:
        case EeSchedulerWaitKind::EventFlag:
        case EeSchedulerWaitKind::HleSemaphore:
            return objectId >= 0;
        case EeSchedulerWaitKind::None:
        default:
            return false;
        }
    }

    bool operator<(
        const EeSchedulerWaitKey &left,
        const EeSchedulerWaitKey &right) noexcept
    {
        return std::tie(left.kind, left.objectId) <
               std::tie(right.kind, right.objectId);
    }

    void EeThreadScheduler::enableTransitionTracing(
        size_t capacity) noexcept
    {
        m_transitionTracingEnabled = capacity != 0u;
        m_transitionTraceCapacity = capacity;
        clearTransitionTrace();
    }

    void EeThreadScheduler::disableTransitionTracing()
        noexcept
    {
        m_transitionTracingEnabled = false;
    }

    void EeThreadScheduler::clearTransitionTrace()
        noexcept
    {
        m_transitionTrace.clear();
        m_nextTransitionTraceSequence = 1u;
        m_droppedTransitionTraceCount = 0u;
    }

    bool EeThreadScheduler::transitionTracingEnabled()
        const noexcept
    {
        return m_transitionTracingEnabled;
    }

    size_t EeThreadScheduler::transitionTraceCapacity()
        const noexcept
    {
        return m_transitionTraceCapacity;
    }

    uint64_t
    EeThreadScheduler::droppedTransitionTraceCount()
        const noexcept
    {
        return m_droppedTransitionTraceCount;
    }

    std::vector<EeSchedulerTransitionTrace>
    EeThreadScheduler::transitionTraceSnapshot() const
    {
        return {
            m_transitionTrace.begin(),
            m_transitionTrace.end()};
    }

    void EeThreadScheduler::publishCanonicalTick(
        ps2x::timing::EeTick tick) noexcept
    {
        if (tick >= m_canonicalTick)
        {
            m_canonicalTick = tick;
        }
    }

    void EeThreadScheduler::recordTransition(
        EeSchedulerTransitionReason reason,
        std::optional<int> oldThreadId,
        int threadId,
        EeSchedulerWaitKey wait)
    {
        if (!m_transitionTracingEnabled ||
            m_transitionTraceCapacity == 0u)
        {
            return;
        }

        const auto it = m_threads.find(threadId);
        if (it == m_threads.end())
        {
            return;
        }

        if (!wait.valid())
        {
            wait = it->second.wait;
            if (!wait.valid() &&
                it->second.completedWait.valid())
            {
                wait =
                    it->second.completedWait.wait;
            }
        }

        EeSchedulerTransitionTrace trace{};
        trace.tick = m_canonicalTick;
        trace.sequence =
            m_nextTransitionTraceSequence++;
        trace.reason = reason;
        trace.oldThreadId = oldThreadId;
        trace.newThreadId = m_currentThreadId;
        trace.threadId = threadId;
        trace.generation = it->second.generation;
        trace.priority = it->second.priority;
        trace.state = it->second.state;
        trace.wait = wait;
        trace.queueSequence =
            it->second.queueSequence;

        if (m_transitionTrace.size() ==
            m_transitionTraceCapacity)
        {
            m_transitionTrace.pop_front();
            ++m_droppedTransitionTraceCount;
        }
        m_transitionTrace.push_back(trace);
    }

    bool EeThreadScheduler::validThreadId(
        int threadId) noexcept
    {
        return threadId > 0;
    }

    bool EeThreadScheduler::validPriority(
        int priority) noexcept
    {
        return priority >= 0 &&
               priority < kEeThreadPriorityCount;
    }

    bool EeThreadScheduler::addDormantThread(
        int threadId,
        uint32_t generation,
        int priority)
    {
        if (!validThreadId(threadId) ||
            generation == 0u ||
            !validPriority(priority))
        {
            return false;
        }

        const bool added =
            m_threads
                .emplace(
                    threadId,
                    ThreadRecord{
                        generation,
                        priority,
                        EeSchedulerThreadState::Dormant,
                        {},
                        {},
                        false,
                        0u,
                        0u,
                        0u})
                .second;
        if (added)
        {
            recordTransition(
                EeSchedulerTransitionReason::
                    ThreadAdded,
                m_currentThreadId,
                threadId);
        }
        return added;
    }

    bool EeThreadScheduler::addRunningThread(
        int threadId,
        uint32_t generation,
        int priority)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        if (m_currentThreadId.has_value() ||
            !addDormantThread(
                threadId, generation, priority))
        {
            return false;
        }

        ThreadRecord &thread =
            m_threads.at(threadId);
        thread.state = EeSchedulerThreadState::Running;
        m_currentThreadId = threadId;
        recordTransition(
            EeSchedulerTransitionReason::
                RunningThreadAdded,
            oldThreadId,
            threadId);
        return true;
    }

    void EeThreadScheduler::enqueueReady(
        int threadId,
        ThreadRecord &thread)
    {
        thread.state = EeSchedulerThreadState::Ready;
        thread.wait = {};
        thread.queueSequence = m_nextQueueSequence++;
        m_readyQueues[
            static_cast<size_t>(thread.priority)]
            .push_back(
                EeSchedulerThreadHandle{
                    threadId,
                    thread.generation});
    }

    bool EeThreadScheduler::startThread(int threadId)
    {
        const auto handle = threadHandle(threadId);
        return handle.has_value() &&
               startThread(*handle);
    }

    bool EeThreadScheduler::startThread(
        EeSchedulerThreadHandle handle)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->state !=
                EeSchedulerThreadState::Dormant)
        {
            return false;
        }

        enqueueReady(handle.id, *thread);
        recordTransition(
            EeSchedulerTransitionReason::Started,
            oldThreadId,
            handle.id);
        return true;
    }

    std::optional<int>
    EeThreadScheduler::takeNextReady()
    {
        for (std::deque<EeSchedulerThreadHandle> &queue :
             m_readyQueues)
        {
            while (!queue.empty())
            {
                const EeSchedulerThreadHandle handle =
                    queue.front();
                queue.pop_front();
                ThreadRecord *thread =
                    findThread(handle);
                if (!thread ||
                    thread->state !=
                        EeSchedulerThreadState::Ready)
                {
                    continue;
                }

            thread->state =
                EeSchedulerThreadState::Running;
            thread->retainCompletedWaitReason = false;
            thread->queueSequence = 0u;
            m_currentThreadId = handle.id;
                return handle.id;
            }
        }
        return std::nullopt;
    }

    std::optional<int>
    EeThreadScheduler::selectNextThread()
    {
        if (m_currentThreadId.has_value())
        {
            return m_currentThreadId;
        }
        const std::optional<int> selected =
            takeNextReady();
        if (selected.has_value())
        {
            recordTransition(
                EeSchedulerTransitionReason::Selected,
                std::nullopt,
                *selected);
        }
        return selected;
    }

    std::optional<int>
    EeThreadScheduler::yieldCurrentThread()
    {
        if (!m_currentThreadId.has_value())
        {
            return selectNextThread();
        }

        const int threadId = *m_currentThreadId;
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running)
        {
            return std::nullopt;
        }

        m_currentThreadId.reset();
        enqueueReady(threadId, it->second);
        const std::optional<int> selected =
            takeNextReady();
        recordTransition(
            EeSchedulerTransitionReason::Yielded,
            oldThreadId,
            threadId);
        return selected;
    }

    std::optional<int>
    EeThreadScheduler::blockCurrentThread(
        EeSchedulerWaitKey wait)
    {
        if (!m_currentThreadId.has_value() ||
            !wait.valid())
        {
            return std::nullopt;
        }

        const int threadId = *m_currentThreadId;
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running)
        {
            return std::nullopt;
        }

        ThreadRecord &thread = it->second;
        if (thread.completedWait.valid() ||
            (wait.kind == EeSchedulerWaitKind::Sleep &&
             thread.wakeupCount != 0u))
        {
            return std::nullopt;
        }
        thread.state = EeSchedulerThreadState::Waiting;
        thread.wait = wait;
        thread.queueSequence = m_nextQueueSequence++;
        m_waitQueues[wait].push_back(
            EeSchedulerThreadHandle{
                threadId,
                thread.generation});
        m_currentThreadId.reset();
        const std::optional<int> selected =
            takeNextReady();
        recordTransition(
            EeSchedulerTransitionReason::Blocked,
            oldThreadId,
            threadId,
            wait);
        return selected;
    }

    EeSchedulerSleepResult
    EeThreadScheduler::sleepCurrentThread()
    {
        EeSchedulerSleepResult result{};
        if (!m_currentThreadId.has_value())
        {
            return result;
        }

        const int threadId = *m_currentThreadId;
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running ||
            it->second.completedWait.valid())
        {
            return result;
        }

        ThreadRecord &thread = it->second;
        if (thread.wakeupCount != 0u)
        {
            --thread.wakeupCount;
            result.disposition =
                EeSchedulerSleepDisposition::
                    WakeupConsumed;
            result.selectedThreadId = threadId;
            recordTransition(
                EeSchedulerTransitionReason::
                    WakeupConsumed,
                m_currentThreadId,
                threadId,
                EeSchedulerWaitKey{
                    EeSchedulerWaitKind::Sleep,
                    0});
            return result;
        }

        result.disposition =
            EeSchedulerSleepDisposition::Blocked;
        result.selectedThreadId =
            blockCurrentThread(
                EeSchedulerWaitKey{
                    EeSchedulerWaitKind::Sleep,
                    0});
        return result;
    }

    std::optional<int>
    EeThreadScheduler::wakeOne(
        EeSchedulerWaitKey wait)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        if (!wait.valid())
        {
            return std::nullopt;
        }

        const auto queueIt = m_waitQueues.find(wait);
        if (queueIt == m_waitQueues.end() ||
            queueIt->second.empty())
        {
            return std::nullopt;
        }

        const EeSchedulerThreadHandle handle =
            queueIt->second.front();
        queueIt->second.pop_front();
        if (queueIt->second.empty())
        {
            m_waitQueues.erase(queueIt);
        }

        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            (thread->state !=
                 EeSchedulerThreadState::Waiting &&
             thread->state !=
                 EeSchedulerThreadState::WaitSuspended) ||
            thread->wait != wait)
        {
            return std::nullopt;
        }

        thread->wait = {};
        thread->completedWait = {
            wait,
            EeSchedulerWaitCompletion::Satisfied};
        thread->retainCompletedWaitReason = false;
        if (thread->suspendCount != 0u)
        {
            thread->state =
                EeSchedulerThreadState::Suspended;
            thread->queueSequence = 0u;
        }
        else
        {
            enqueueReady(handle.id, *thread);
        }
        recordTransition(
            EeSchedulerTransitionReason::
                WaitSatisfied,
            oldThreadId,
            handle.id,
            wait);
        return handle.id;
    }

    EeSchedulerWakeResult
    EeThreadScheduler::wakeThread(
        EeSchedulerThreadHandle handle)
    {
        ThreadRecord *thread = findThread(handle);
        if (!thread)
        {
            return EeSchedulerWakeResult::
                InvalidHandle;
        }
        if (thread->state ==
            EeSchedulerThreadState::Dormant)
        {
            return EeSchedulerWakeResult::Dormant;
        }

        const EeSchedulerWaitKey sleep{
            EeSchedulerWaitKind::Sleep,
            0};
        if ((thread->state ==
                 EeSchedulerThreadState::Waiting ||
             thread->state ==
                 EeSchedulerThreadState::
                     WaitSuspended) &&
            thread->wait == sleep)
        {
            if (!releaseWaitThread(
                    handle,
                    EeSchedulerWaitCompletion::
                        Satisfied))
            {
                return EeSchedulerWakeResult::
                    InvalidHandle;
            }
            return EeSchedulerWakeResult::
                WokeSleeper;
        }

        if (thread->wakeupCount ==
            std::numeric_limits<uint32_t>::max())
        {
            return EeSchedulerWakeResult::
                WakeupCountOverflow;
        }
        ++thread->wakeupCount;
        recordTransition(
            EeSchedulerTransitionReason::
                WakeupCounted,
            m_currentThreadId,
            handle.id,
            sleep);
        return EeSchedulerWakeResult::
            WakeupCounted;
    }

    std::optional<uint32_t>
    EeThreadScheduler::cancelWakeups(
        EeSchedulerThreadHandle handle)
    {
        ThreadRecord *thread = findThread(handle);
        if (!thread)
        {
            return std::nullopt;
        }

        const uint32_t previous =
            thread->wakeupCount;
        thread->wakeupCount = 0u;
        recordTransition(
            EeSchedulerTransitionReason::
                WakeupsCancelled,
            m_currentThreadId,
            handle.id,
            EeSchedulerWaitKey{
                EeSchedulerWaitKind::Sleep,
                0});
        return previous;
    }

    bool EeThreadScheduler::releaseWaitThread(
        EeSchedulerThreadHandle handle,
        EeSchedulerWaitCompletion completion,
        bool retainReasonForStatus)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            completion ==
                EeSchedulerWaitCompletion::None ||
            (thread->state !=
                 EeSchedulerThreadState::Waiting &&
             thread->state !=
                 EeSchedulerThreadState::
                     WaitSuspended) ||
            !thread->wait.valid())
        {
            return false;
        }

        const EeSchedulerWaitKey wait =
            thread->wait;
        if (!removeWait(handle, wait))
        {
            return false;
        }

        thread->wait = {};
        thread->completedWait = {
            wait,
            completion};
        thread->retainCompletedWaitReason =
            retainReasonForStatus;
        if (thread->suspendCount != 0u)
        {
            thread->state =
                EeSchedulerThreadState::Suspended;
            thread->queueSequence = 0u;
        }
        else
        {
            enqueueReady(handle.id, *thread);
        }
        recordTransition(
            transitionReasonForWaitCompletion(
                completion),
            oldThreadId,
            handle.id,
            wait);
        return true;
    }

    std::optional<EeSchedulerCompletedWait>
    EeThreadScheduler::consumeWaitCompletion(
        EeSchedulerThreadHandle handle)
    {
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->state !=
                EeSchedulerThreadState::Running ||
            !thread->completedWait.valid())
        {
            return std::nullopt;
        }

        const EeSchedulerCompletedWait completed =
            thread->completedWait;
        thread->completedWait = {};
        thread->retainCompletedWaitReason = false;
        recordTransition(
            EeSchedulerTransitionReason::
                WaitCompletionConsumed,
            m_currentThreadId,
            handle.id,
            completed.wait);
        return completed;
    }

    bool EeThreadScheduler::retireCurrentThread()
    {
        if (!m_currentThreadId.has_value())
        {
            return false;
        }

        const int threadId = *m_currentThreadId;
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running)
        {
            return false;
        }

        it->second.state =
            EeSchedulerThreadState::Dormant;
        it->second.wait = {};
        it->second.completedWait = {};
        it->second.retainCompletedWaitReason = false;
        it->second.wakeupCount = 0u;
        it->second.suspendCount = 0u;
        it->second.queueSequence = 0u;
        m_currentThreadId.reset();
        return true;
    }

    std::optional<int>
    EeThreadScheduler::finishCurrentThread()
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        const int threadId =
            oldThreadId.value_or(0);
        if (!retireCurrentThread())
        {
            return std::nullopt;
        }
        const std::optional<int> selected =
            takeNextReady();
        recordTransition(
            EeSchedulerTransitionReason::Finished,
            oldThreadId,
            threadId);
        return selected;
    }

    EeThreadScheduler::ThreadRecord *
    EeThreadScheduler::findThread(
        EeSchedulerThreadHandle handle)
    {
        if (!handle.valid())
        {
            return nullptr;
        }
        const auto it = m_threads.find(handle.id);
        if (it == m_threads.end() ||
            it->second.generation !=
                handle.generation)
        {
            return nullptr;
        }
        return &it->second;
    }

    const EeThreadScheduler::ThreadRecord *
    EeThreadScheduler::findThread(
        EeSchedulerThreadHandle handle) const
    {
        if (!handle.valid())
        {
            return nullptr;
        }
        const auto it = m_threads.find(handle.id);
        if (it == m_threads.end() ||
            it->second.generation !=
                handle.generation)
        {
            return nullptr;
        }
        return &it->second;
    }

    bool EeThreadScheduler::removeReady(
        EeSchedulerThreadHandle handle,
        int priority)
    {
        if (!validPriority(priority))
        {
            return false;
        }
        std::deque<EeSchedulerThreadHandle> &queue =
            m_readyQueues[
                static_cast<size_t>(priority)];
        const auto it =
            std::find(
                queue.begin(), queue.end(), handle);
        if (it == queue.end())
        {
            return false;
        }
        queue.erase(it);
        return true;
    }

    bool EeThreadScheduler::removeWait(
        EeSchedulerThreadHandle handle,
        EeSchedulerWaitKey wait)
    {
        const auto queueIt = m_waitQueues.find(wait);
        if (queueIt == m_waitQueues.end())
        {
            return false;
        }
        std::deque<EeSchedulerThreadHandle> &queue =
            queueIt->second;
        const auto it =
            std::find(
                queue.begin(), queue.end(), handle);
        if (it == queue.end())
        {
            return false;
        }
        queue.erase(it);
        if (queue.empty())
        {
            m_waitQueues.erase(queueIt);
        }
        return true;
    }

    bool EeThreadScheduler::suspendThread(
        EeSchedulerThreadHandle handle)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->state ==
                EeSchedulerThreadState::Dormant ||
            thread->suspendCount ==
                std::numeric_limits<uint32_t>::max())
        {
            return false;
        }

        bool selectReplacement = false;
        switch (thread->state)
        {
        case EeSchedulerThreadState::Running:
            if (!m_currentThreadId.has_value() ||
                *m_currentThreadId != handle.id)
            {
                return false;
            }
            thread->state =
                EeSchedulerThreadState::Suspended;
            m_currentThreadId.reset();
            selectReplacement = true;
            break;
        case EeSchedulerThreadState::Ready:
            if (!removeReady(
                    handle, thread->priority))
            {
                return false;
            }
            thread->state =
                EeSchedulerThreadState::Suspended;
            thread->queueSequence = 0u;
            break;
        case EeSchedulerThreadState::Waiting:
            thread->state =
                EeSchedulerThreadState::WaitSuspended;
            break;
        case EeSchedulerThreadState::Suspended:
        case EeSchedulerThreadState::WaitSuspended:
            break;
        case EeSchedulerThreadState::Dormant:
            return false;
        }
        ++thread->suspendCount;
        if (selectReplacement)
        {
            (void)takeNextReady();
        }
        recordTransition(
            EeSchedulerTransitionReason::Suspended,
            oldThreadId,
            handle.id);
        return true;
    }

    bool EeThreadScheduler::resumeThread(
        EeSchedulerThreadHandle handle)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->suspendCount == 0u ||
            (thread->state !=
                 EeSchedulerThreadState::Suspended &&
             thread->state !=
                 EeSchedulerThreadState::WaitSuspended))
        {
            return false;
        }

        --thread->suspendCount;
        if (thread->suspendCount != 0u)
        {
            recordTransition(
                EeSchedulerTransitionReason::Resumed,
                oldThreadId,
                handle.id);
            return true;
        }

        if (thread->state ==
            EeSchedulerThreadState::WaitSuspended)
        {
            thread->state =
                EeSchedulerThreadState::Waiting;
        }
        else
        {
            enqueueReady(handle.id, *thread);
        }
        recordTransition(
            EeSchedulerTransitionReason::Resumed,
            oldThreadId,
            handle.id);
        return true;
    }

    bool EeThreadScheduler::terminateThread(
        EeSchedulerThreadHandle handle)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->state ==
                EeSchedulerThreadState::Dormant)
        {
            return false;
        }

        const EeSchedulerWaitKey previousWait =
            thread->wait;
        switch (thread->state)
        {
        case EeSchedulerThreadState::Running:
            if (!m_currentThreadId.has_value() ||
                *m_currentThreadId != handle.id)
            {
                return false;
            }
            m_currentThreadId.reset();
            break;
        case EeSchedulerThreadState::Ready:
            if (!removeReady(
                    handle, thread->priority))
            {
                return false;
            }
            break;
        case EeSchedulerThreadState::Waiting:
        case EeSchedulerThreadState::WaitSuspended:
            if (!removeWait(
                    handle, thread->wait))
            {
                return false;
            }
            break;
        case EeSchedulerThreadState::Suspended:
            break;
        case EeSchedulerThreadState::Dormant:
            return false;
        }

        thread->state =
            EeSchedulerThreadState::Dormant;
        thread->wait = {};
        thread->completedWait = {};
        thread->retainCompletedWaitReason = false;
        thread->wakeupCount = 0u;
        thread->suspendCount = 0u;
        thread->queueSequence = 0u;
        if (!m_currentThreadId.has_value())
        {
            (void)takeNextReady();
        }
        recordTransition(
            EeSchedulerTransitionReason::Terminated,
            oldThreadId,
            handle.id,
            previousWait);
        return true;
    }

    bool EeThreadScheduler::exitCurrentThread()
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        const int threadId =
            oldThreadId.value_or(0);
        if (!retireCurrentThread())
        {
            return false;
        }
        (void)takeNextReady();
        recordTransition(
            EeSchedulerTransitionReason::Exited,
            oldThreadId,
            threadId);
        return true;
    }

    bool EeThreadScheduler::deleteThread(
        EeSchedulerThreadHandle handle)
    {
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->state !=
                EeSchedulerThreadState::Dormant)
        {
            return false;
        }
        recordTransition(
            EeSchedulerTransitionReason::Deleted,
            m_currentThreadId,
            handle.id);
        return m_threads.erase(handle.id) == 1u;
    }

    std::optional<int>
    EeThreadScheduler::changeThreadPriority(
        EeSchedulerThreadHandle handle,
        int priority)
    {
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            !validPriority(priority) ||
            thread->state ==
                EeSchedulerThreadState::Dormant)
        {
            return std::nullopt;
        }

        const int previous = thread->priority;
        if (previous == priority)
        {
            return previous;
        }

        if (thread->state ==
            EeSchedulerThreadState::Ready)
        {
            if (!removeReady(handle, previous))
            {
                return std::nullopt;
            }
            thread->priority = priority;
            enqueueReady(handle.id, *thread);
        }
        else
        {
            thread->priority = priority;
        }
        recordTransition(
            EeSchedulerTransitionReason::
                PriorityChanged,
            m_currentThreadId,
            handle.id);
        return previous;
    }

    bool EeThreadScheduler::rotateReadyQueue(
        int priority)
    {
        if (!validPriority(priority))
        {
            return false;
        }

        std::deque<EeSchedulerThreadHandle> &queue =
            m_readyQueues[
                static_cast<size_t>(priority)];
        if (queue.size() < 2u)
        {
            return true;
        }

        const EeSchedulerThreadHandle handle =
            queue.front();
        ThreadRecord *thread = findThread(handle);
        if (!thread ||
            thread->state !=
                EeSchedulerThreadState::Ready ||
            thread->priority != priority)
        {
            return false;
        }

        queue.pop_front();
        thread->queueSequence =
            m_nextQueueSequence++;
        queue.push_back(handle);
        recordTransition(
            EeSchedulerTransitionReason::
                ReadyQueueRotated,
            m_currentThreadId,
            handle.id);
        return true;
    }

    std::optional<int>
    EeThreadScheduler::highestReadyPriority()
        const
    {
        for (size_t priority = 0u;
             priority < m_readyQueues.size();
             ++priority)
        {
            if (!m_readyQueues[priority].empty())
            {
                return static_cast<int>(priority);
            }
        }
        return std::nullopt;
    }

    std::optional<int>
    EeThreadScheduler::reschedule(
        EeSchedulerReschedulePolicy policy)
    {
        const std::optional<int> oldThreadId =
            m_currentThreadId;
        if (!m_currentThreadId.has_value())
        {
            const std::optional<int> selected =
                takeNextReady();
            if (selected.has_value())
            {
                recordTransition(
                    EeSchedulerTransitionReason::
                        Rescheduled,
                    oldThreadId,
                    *selected);
            }
            return selected;
        }

        const int threadId = *m_currentThreadId;
        const auto currentIt =
            m_threads.find(threadId);
        if (currentIt == m_threads.end() ||
            currentIt->second.state !=
                EeSchedulerThreadState::Running)
        {
            return std::nullopt;
        }

        const std::optional<int> readyPriority =
            highestReadyPriority();
        if (!readyPriority.has_value())
        {
            return threadId;
        }

        const int currentPriority =
            currentIt->second.priority;
        if (policy ==
            EeSchedulerReschedulePolicy::None)
        {
            return threadId;
        }
        const bool shouldPreempt =
            *readyPriority < currentPriority ||
            (policy ==
                 EeSchedulerReschedulePolicy::
                     EqualOrHigherPriority &&
             *readyPriority == currentPriority);
        if (!shouldPreempt)
        {
            return threadId;
        }

        m_currentThreadId.reset();
        enqueueReady(threadId, currentIt->second);
        const std::optional<int> selected =
            takeNextReady();
        if (selected.has_value())
        {
            recordTransition(
                EeSchedulerTransitionReason::
                    Rescheduled,
                oldThreadId,
                *selected);
        }
        return selected;
    }

    std::optional<EeSchedulerDispatch>
    EeThreadScheduler::dispatchOne(
        IEeSchedulerExecutionBackend &backend,
        uint64_t tickBudget)
    {
        const std::optional<int> selected =
            selectNextThread();
        if (!selected.has_value())
        {
            return std::nullopt;
        }

        EeSchedulerDispatch dispatch{};
        dispatch.resumedThreadId = *selected;
        dispatch.result =
            backend.resume(*selected, tickBudget);

        switch (dispatch.result.reason)
        {
        case EeSchedulerExitReason::Yielded:
            dispatch.selectedThreadId =
                yieldCurrentThread();
            break;
        case EeSchedulerExitReason::Blocked:
            dispatch.selectedThreadId =
                blockCurrentThread(
                    dispatch.result.wait);
            break;
        case EeSchedulerExitReason::Finished:
        case EeSchedulerExitReason::Exception:
            dispatch.selectedThreadId =
                finishCurrentThread();
            break;
        case EeSchedulerExitReason::Preempted:
        case EeSchedulerExitReason::StopRequested:
            dispatch.selectedThreadId =
                currentThreadId();
            break;
        }

        return dispatch;
    }

    std::optional<int>
    EeThreadScheduler::currentThreadId()
        const noexcept
    {
        return m_currentThreadId;
    }

    std::optional<EeSchedulerThreadHandle>
    EeThreadScheduler::threadHandle(
        int threadId) const
    {
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end())
        {
            return std::nullopt;
        }
        return EeSchedulerThreadHandle{
            threadId,
            it->second.generation};
    }

    std::optional<EeSchedulerThreadSnapshot>
    EeThreadScheduler::thread(int threadId) const
    {
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end())
        {
            return std::nullopt;
        }

        return EeSchedulerThreadSnapshot{
            threadId,
            it->second.generation,
            it->second.priority,
            it->second.state,
            it->second.wait,
            it->second.completedWait,
            it->second.retainCompletedWaitReason,
            it->second.wakeupCount,
            it->second.suspendCount,
            it->second.queueSequence};
    }

    std::vector<int>
    EeThreadScheduler::readyOrder(
        int priority) const
    {
        if (!validPriority(priority))
        {
            return {};
        }

        const std::deque<EeSchedulerThreadHandle>
            &queue =
            m_readyQueues[
                static_cast<size_t>(priority)];
        std::vector<int> order;
        order.reserve(queue.size());
        for (EeSchedulerThreadHandle handle :
             queue)
        {
            order.push_back(handle.id);
        }
        return order;
    }

    std::vector<int>
    EeThreadScheduler::waitOrder(
        EeSchedulerWaitKey wait) const
    {
        const auto it = m_waitQueues.find(wait);
        if (it == m_waitQueues.end())
        {
            return {};
        }
        std::vector<int> order;
        order.reserve(it->second.size());
        for (EeSchedulerThreadHandle handle :
             it->second)
        {
            order.push_back(handle.id);
        }
        return order;
    }

    size_t EeThreadScheduler::threadCount()
        const noexcept
    {
        return m_threads.size();
    }

    bool EeThreadScheduler::validate() const
    {
        std::unordered_map<int, size_t>
            readyMembership;
        std::unordered_map<int, size_t>
            waitMembership;

        for (size_t priority = 0u;
             priority < m_readyQueues.size();
             ++priority)
        {
            for (EeSchedulerThreadHandle handle :
                 m_readyQueues[priority])
            {
                ++readyMembership[handle.id];
                const auto it =
                    m_threads.find(handle.id);
                if (it == m_threads.end() ||
                    it->second.generation !=
                        handle.generation ||
                    it->second.state !=
                        EeSchedulerThreadState::Ready ||
                    it->second.priority !=
                        static_cast<int>(priority))
                {
                    return false;
                }
            }
        }

        for (const auto &[wait, queue] :
             m_waitQueues)
        {
            if (!wait.valid() || queue.empty())
            {
                return false;
            }
            for (EeSchedulerThreadHandle handle :
                 queue)
            {
                ++waitMembership[handle.id];
                const auto it =
                    m_threads.find(handle.id);
                if (it == m_threads.end() ||
                    it->second.generation !=
                        handle.generation ||
                    (it->second.state !=
                         EeSchedulerThreadState::Waiting &&
                     it->second.state !=
                         EeSchedulerThreadState::WaitSuspended) ||
                    it->second.wait != wait)
                {
                    return false;
                }
            }
        }

        size_t runningCount = 0u;
        for (const auto &[threadId, thread] :
             m_threads)
        {
            const size_t readyCount =
                readyMembership[threadId];
            const size_t waitCount =
                waitMembership[threadId];
            const bool noCompletedWait =
                thread.completedWait.completion ==
                EeSchedulerWaitCompletion::None;
            const bool knownCompletion =
                noCompletedWait ||
                thread.completedWait.completion ==
                    EeSchedulerWaitCompletion::
                        Satisfied ||
                thread.completedWait.completion ==
                    EeSchedulerWaitCompletion::
                        Released ||
                thread.completedWait.completion ==
                    EeSchedulerWaitCompletion::
                        ObjectDeleted ||
                thread.completedWait.completion ==
                    EeSchedulerWaitCompletion::
                        TimedOut;
            if (!knownCompletion ||
                noCompletedWait ==
                    thread.completedWait.wait.valid() ||
                (thread.retainCompletedWaitReason &&
                 noCompletedWait))
            {
                return false;
            }

            switch (thread.state)
            {
            case EeSchedulerThreadState::Dormant:
                if (readyCount != 0u ||
                    waitCount != 0u ||
                    thread.wait.valid() ||
                    !noCompletedWait ||
                    thread.retainCompletedWaitReason ||
                    thread.wakeupCount != 0u ||
                    thread.suspendCount != 0u ||
                    thread.queueSequence != 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::Ready:
                if (readyCount != 1u ||
                    waitCount != 0u ||
                    thread.wait.valid() ||
                    thread.suspendCount != 0u ||
                    thread.queueSequence == 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::Running:
                ++runningCount;
                if (!m_currentThreadId.has_value() ||
                    *m_currentThreadId != threadId ||
                    readyCount != 0u ||
                    waitCount != 0u ||
                    thread.wait.valid() ||
                    thread.retainCompletedWaitReason ||
                    thread.suspendCount != 0u ||
                    thread.queueSequence != 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::Waiting:
                if (readyCount != 0u ||
                    waitCount != 1u ||
                    !thread.wait.valid() ||
                    !noCompletedWait ||
                    thread.retainCompletedWaitReason ||
                    (thread.wait.kind ==
                         EeSchedulerWaitKind::Sleep &&
                     thread.wakeupCount != 0u) ||
                    thread.suspendCount != 0u ||
                    thread.queueSequence == 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::Suspended:
                if (readyCount != 0u ||
                    waitCount != 0u ||
                    thread.wait.valid() ||
                    thread.suspendCount == 0u ||
                    thread.queueSequence != 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::WaitSuspended:
                if (readyCount != 0u ||
                    waitCount != 1u ||
                    !thread.wait.valid() ||
                    !noCompletedWait ||
                    thread.retainCompletedWaitReason ||
                    (thread.wait.kind ==
                         EeSchedulerWaitKind::Sleep &&
                     thread.wakeupCount != 0u) ||
                    thread.suspendCount == 0u ||
                    thread.queueSequence == 0u)
                {
                    return false;
                }
                break;
            }
        }

        return runningCount ==
               (m_currentThreadId.has_value()
                    ? 1u
                    : 0u);
    }
}
