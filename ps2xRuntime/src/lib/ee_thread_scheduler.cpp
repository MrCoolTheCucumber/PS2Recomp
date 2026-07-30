#include "runtime/ee_thread_scheduler.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>

namespace ps2x::ee
{
    bool EeSchedulerWaitKey::valid() const noexcept
    {
        switch (kind)
        {
        case EeSchedulerWaitKind::Sleep:
            return objectId == 0;
        case EeSchedulerWaitKind::Semaphore:
        case EeSchedulerWaitKind::EventFlag:
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

        return m_threads
            .emplace(
                threadId,
                ThreadRecord{
                    generation,
                    priority,
                    EeSchedulerThreadState::Dormant,
                    {},
                    0u})
            .second;
    }

    bool EeThreadScheduler::addRunningThread(
        int threadId,
        uint32_t generation,
        int priority)
    {
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
            .push_back(threadId);
    }

    bool EeThreadScheduler::startThread(int threadId)
    {
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Dormant)
        {
            return false;
        }

        enqueueReady(threadId, it->second);
        return true;
    }

    std::optional<int>
    EeThreadScheduler::takeNextReady()
    {
        for (std::deque<int> &queue :
             m_readyQueues)
        {
            if (queue.empty())
            {
                continue;
            }

            const int threadId = queue.front();
            queue.pop_front();
            const auto it = m_threads.find(threadId);
            if (it == m_threads.end() ||
                it->second.state !=
                    EeSchedulerThreadState::Ready)
            {
                return std::nullopt;
            }

            it->second.state =
                EeSchedulerThreadState::Running;
            it->second.queueSequence = 0u;
            m_currentThreadId = threadId;
            return threadId;
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
        return takeNextReady();
    }

    std::optional<int>
    EeThreadScheduler::yieldCurrentThread()
    {
        if (!m_currentThreadId.has_value())
        {
            return selectNextThread();
        }

        const int threadId = *m_currentThreadId;
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running)
        {
            return std::nullopt;
        }

        m_currentThreadId.reset();
        enqueueReady(threadId, it->second);
        return takeNextReady();
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
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running)
        {
            return std::nullopt;
        }

        ThreadRecord &thread = it->second;
        thread.state = EeSchedulerThreadState::Waiting;
        thread.wait = wait;
        thread.queueSequence = m_nextQueueSequence++;
        m_waitQueues[wait].push_back(threadId);
        m_currentThreadId.reset();
        return takeNextReady();
    }

    std::optional<int>
    EeThreadScheduler::wakeOne(
        EeSchedulerWaitKey wait)
    {
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

        const int threadId = queueIt->second.front();
        queueIt->second.pop_front();
        if (queueIt->second.empty())
        {
            m_waitQueues.erase(queueIt);
        }

        const auto threadIt = m_threads.find(threadId);
        if (threadIt == m_threads.end() ||
            threadIt->second.state !=
                EeSchedulerThreadState::Waiting ||
            threadIt->second.wait != wait)
        {
            return std::nullopt;
        }

        enqueueReady(threadId, threadIt->second);
        return threadId;
    }

    std::optional<int>
    EeThreadScheduler::finishCurrentThread()
    {
        if (!m_currentThreadId.has_value())
        {
            return std::nullopt;
        }

        const int threadId = *m_currentThreadId;
        const auto it = m_threads.find(threadId);
        if (it == m_threads.end() ||
            it->second.state !=
                EeSchedulerThreadState::Running)
        {
            return std::nullopt;
        }

        it->second.state =
            EeSchedulerThreadState::Dormant;
        it->second.wait = {};
        it->second.queueSequence = 0u;
        m_currentThreadId.reset();
        return takeNextReady();
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

        const std::deque<int> &queue =
            m_readyQueues[
                static_cast<size_t>(priority)];
        return {queue.begin(), queue.end()};
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
        return {
            it->second.begin(),
            it->second.end()};
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
            for (int threadId :
                 m_readyQueues[priority])
            {
                ++readyMembership[threadId];
                const auto it =
                    m_threads.find(threadId);
                if (it == m_threads.end() ||
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
            for (int threadId : queue)
            {
                ++waitMembership[threadId];
                const auto it =
                    m_threads.find(threadId);
                if (it == m_threads.end() ||
                    it->second.state !=
                        EeSchedulerThreadState::Waiting ||
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
            switch (thread.state)
            {
            case EeSchedulerThreadState::Dormant:
                if (readyCount != 0u ||
                    waitCount != 0u ||
                    thread.wait.valid() ||
                    thread.queueSequence != 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::Ready:
                if (readyCount != 1u ||
                    waitCount != 0u ||
                    thread.wait.valid() ||
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
                    thread.queueSequence != 0u)
                {
                    return false;
                }
                break;
            case EeSchedulerThreadState::Waiting:
                if (readyCount != 0u ||
                    waitCount != 1u ||
                    !thread.wait.valid() ||
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
