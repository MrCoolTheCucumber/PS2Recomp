#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ps2x::ee
{
    constexpr int kEeThreadPriorityCount = 128;

    enum class EeSchedulerThreadState : uint8_t
    {
        Dormant,
        Ready,
        Running,
        Waiting,
    };

    enum class EeSchedulerWaitKind : uint8_t
    {
        None,
        Sleep,
        Semaphore,
        EventFlag,
    };

    struct EeSchedulerWaitKey
    {
        EeSchedulerWaitKind kind =
            EeSchedulerWaitKind::None;
        int objectId = 0;

        [[nodiscard]] bool valid() const noexcept;

        friend bool operator==(
            const EeSchedulerWaitKey &,
            const EeSchedulerWaitKey &) = default;

        friend bool operator<(
            const EeSchedulerWaitKey &left,
            const EeSchedulerWaitKey &right) noexcept;
    };

    struct EeSchedulerThreadSnapshot
    {
        int id = 0;
        uint32_t generation = 0u;
        int priority = 0;
        EeSchedulerThreadState state =
            EeSchedulerThreadState::Dormant;
        EeSchedulerWaitKey wait{};
        uint64_t queueSequence = 0u;
    };

    enum class EeSchedulerExitReason : uint8_t
    {
        Yielded,
        Blocked,
        Preempted,
        Finished,
        StopRequested,
    };

    struct EeSchedulerRunResult
    {
        EeSchedulerExitReason reason =
            EeSchedulerExitReason::Preempted;
        uint64_t elapsedTicks = 0u;
        EeSchedulerWaitKey wait{};
    };

    class IEeSchedulerExecutionBackend
    {
    public:
        virtual ~IEeSchedulerExecutionBackend() = default;

        virtual EeSchedulerRunResult resume(
            int threadId,
            uint64_t tickBudget) = 0;
    };

    struct EeSchedulerDispatch
    {
        int resumedThreadId = 0;
        EeSchedulerRunResult result{};
        std::optional<int> selectedThreadId;
    };

    // This core is single-owner by construction. Host workers publish events
    // elsewhere; the one EE executor applies them here at a scheduling
    // boundary. No method creates a host thread, fiber, or native stack.
    class EeThreadScheduler
    {
    public:
        [[nodiscard]] bool addDormantThread(
            int threadId,
            uint32_t generation,
            int priority);
        [[nodiscard]] bool addRunningThread(
            int threadId,
            uint32_t generation,
            int priority);

        [[nodiscard]] bool startThread(int threadId);
        [[nodiscard]] std::optional<int>
        selectNextThread();
        [[nodiscard]] std::optional<int>
        yieldCurrentThread();
        [[nodiscard]] std::optional<int>
        blockCurrentThread(EeSchedulerWaitKey wait);
        [[nodiscard]] std::optional<int>
        wakeOne(EeSchedulerWaitKey wait);
        [[nodiscard]] std::optional<int>
        finishCurrentThread();

        [[nodiscard]] std::optional<EeSchedulerDispatch>
        dispatchOne(
            IEeSchedulerExecutionBackend &backend,
            uint64_t tickBudget);

        [[nodiscard]] std::optional<int>
        currentThreadId() const noexcept;
        [[nodiscard]] std::optional<EeSchedulerThreadSnapshot>
        thread(int threadId) const;
        [[nodiscard]] std::vector<int>
        readyOrder(int priority) const;
        [[nodiscard]] std::vector<int>
        waitOrder(EeSchedulerWaitKey wait) const;
        [[nodiscard]] size_t threadCount() const noexcept;
        [[nodiscard]] bool validate() const;

    private:
        struct ThreadRecord
        {
            uint32_t generation = 0u;
            int priority = 0;
            EeSchedulerThreadState state =
                EeSchedulerThreadState::Dormant;
            EeSchedulerWaitKey wait{};
            uint64_t queueSequence = 0u;
        };

        [[nodiscard]] static bool validThreadId(
            int threadId) noexcept;
        [[nodiscard]] static bool validPriority(
            int priority) noexcept;
        void enqueueReady(
            int threadId,
            ThreadRecord &thread);
        [[nodiscard]] std::optional<int>
        takeNextReady();

        std::unordered_map<int, ThreadRecord> m_threads;
        std::array<
            std::deque<int>,
            kEeThreadPriorityCount>
            m_readyQueues;
        std::map<
            EeSchedulerWaitKey,
            std::deque<int>>
            m_waitQueues;
        std::optional<int> m_currentThreadId;
        uint64_t m_nextQueueSequence = 1u;
    };
}
