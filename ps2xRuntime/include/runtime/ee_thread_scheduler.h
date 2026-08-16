#pragma once

#include "runtime/ee_timing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <map>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ps2x::ee
{
    constexpr int kEeThreadPriorityCount = 128;
    inline constexpr size_t
        kDefaultEeSchedulerTransitionTraceCapacity =
            4096u;

    enum class EeSchedulerThreadState : uint8_t
    {
        Dormant,
        Ready,
        Running,
        Waiting,
        Suspended,
        WaitSuspended,
    };

    enum class EeSchedulerWaitKind : uint8_t
    {
        None,
        Sleep,
        Semaphore,
        EventFlag,
        HleSemaphore,
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

    enum class EeSchedulerWaitCompletion : uint8_t
    {
        None,
        Satisfied,
        Released,
        ObjectDeleted,
        TimedOut,
    };

    enum class EeSchedulerTransitionReason : uint8_t
    {
        ThreadAdded,
        RunningThreadAdded,
        Started,
        Selected,
        Yielded,
        Blocked,
        WakeupConsumed,
        WaitSatisfied,
        WakeupCounted,
        WakeupsCancelled,
        WaitReleased,
        WaitObjectDeleted,
        WaitTimedOut,
        WaitCompletionConsumed,
        Finished,
        Suspended,
        Resumed,
        Terminated,
        Exited,
        Deleted,
        PriorityChanged,
        ReadyQueueRotated,
        Rescheduled,
    };

    [[nodiscard]] constexpr std::string_view
    eeSchedulerTransitionReasonName(
        EeSchedulerTransitionReason reason) noexcept
    {
        switch (reason)
        {
        case EeSchedulerTransitionReason::ThreadAdded:
            return "thread_added";
        case EeSchedulerTransitionReason::
            RunningThreadAdded:
            return "running_thread_added";
        case EeSchedulerTransitionReason::Started:
            return "started";
        case EeSchedulerTransitionReason::Selected:
            return "selected";
        case EeSchedulerTransitionReason::Yielded:
            return "yielded";
        case EeSchedulerTransitionReason::Blocked:
            return "blocked";
        case EeSchedulerTransitionReason::
            WakeupConsumed:
            return "wakeup_consumed";
        case EeSchedulerTransitionReason::
            WaitSatisfied:
            return "wait_satisfied";
        case EeSchedulerTransitionReason::
            WakeupCounted:
            return "wakeup_counted";
        case EeSchedulerTransitionReason::
            WakeupsCancelled:
            return "wakeups_cancelled";
        case EeSchedulerTransitionReason::
            WaitReleased:
            return "wait_released";
        case EeSchedulerTransitionReason::
            WaitObjectDeleted:
            return "wait_object_deleted";
        case EeSchedulerTransitionReason::WaitTimedOut:
            return "wait_timed_out";
        case EeSchedulerTransitionReason::
            WaitCompletionConsumed:
            return "wait_completion_consumed";
        case EeSchedulerTransitionReason::Finished:
            return "finished";
        case EeSchedulerTransitionReason::Suspended:
            return "suspended";
        case EeSchedulerTransitionReason::Resumed:
            return "resumed";
        case EeSchedulerTransitionReason::Terminated:
            return "terminated";
        case EeSchedulerTransitionReason::Exited:
            return "exited";
        case EeSchedulerTransitionReason::Deleted:
            return "deleted";
        case EeSchedulerTransitionReason::
            PriorityChanged:
            return "priority_changed";
        case EeSchedulerTransitionReason::
            ReadyQueueRotated:
            return "ready_queue_rotated";
        case EeSchedulerTransitionReason::Rescheduled:
            return "rescheduled";
        }
        return "unknown";
    }

    struct EeSchedulerCompletedWait
    {
        EeSchedulerWaitKey wait{};
        EeSchedulerWaitCompletion completion =
            EeSchedulerWaitCompletion::None;

        [[nodiscard]] bool valid() const noexcept
        {
            return wait.valid() &&
                   completion !=
                       EeSchedulerWaitCompletion::None;
        }
    };

    struct EeSchedulerThreadSnapshot
    {
        int id = 0;
        uint32_t generation = 0u;
        int priority = 0;
        EeSchedulerThreadState state =
            EeSchedulerThreadState::Dormant;
        EeSchedulerWaitKey wait{};
        EeSchedulerCompletedWait completedWait{};
        bool retainCompletedWaitReason = false;
        uint32_t wakeupCount = 0u;
        uint32_t suspendCount = 0u;
        uint64_t queueSequence = 0u;
    };

    struct EeThreadSchedulerSnapshot
    {
        std::vector<EeSchedulerThreadSnapshot> threads;
        std::optional<int> currentThreadId;
        uint64_t nextQueueSequence = 1u;
        ps2x::timing::EeTick canonicalTick{};
    };

    // oldThreadId/newThreadId are the selected RUN owner before and after
    // the transition. threadId identifies the affected record; its remaining
    // fields are the post-transition snapshot. wait retains the affected wait
    // object even when the transition detached it from the active wait queue.
    struct EeSchedulerTransitionTrace
    {
        ps2x::timing::EeTick tick{};
        uint64_t sequence = 0u;
        EeSchedulerTransitionReason reason =
            EeSchedulerTransitionReason::ThreadAdded;
        std::optional<int> oldThreadId;
        std::optional<int> newThreadId;
        int threadId = 0;
        uint32_t generation = 0u;
        int priority = 0;
        EeSchedulerThreadState state =
            EeSchedulerThreadState::Dormant;
        EeSchedulerWaitKey wait{};
        uint64_t queueSequence = 0u;
    };

    struct EeSchedulerThreadHandle
    {
        int id = 0;
        uint32_t generation = 0u;

        [[nodiscard]] bool valid() const noexcept
        {
            return id > 0 && generation != 0u;
        }

        friend bool operator==(
            const EeSchedulerThreadHandle &,
            const EeSchedulerThreadHandle &) = default;
    };

    enum class EeSchedulerExitReason : uint8_t
    {
        Yielded,
        Blocked,
        Preempted,
        Finished,
        Exception,
        StopRequested,
    };

    enum class EeSchedulerReschedulePolicy : uint8_t
    {
        None,
        HigherPriorityOnly,
        EqualOrHigherPriority,
    };

    enum class EeSchedulerOwnerLocalTransitionKind : uint8_t
    {
        None,
        RotateReadyQueue,
    };

    // One typed scheduler mutation produced by the currently running guest
    // continuation and consumed by its immediately following executor
    // boundary. This deliberately cannot carry an arbitrary callback or
    // shared completion state.
    struct EeSchedulerOwnerLocalTransition
    {
        EeSchedulerOwnerLocalTransitionKind kind =
            EeSchedulerOwnerLocalTransitionKind::None;
        EeSchedulerThreadHandle originatingThread{};
        int priority = 0;
        EeSchedulerReschedulePolicy reschedulePolicy =
            EeSchedulerReschedulePolicy::None;

        [[nodiscard]] bool pending() const noexcept
        {
            return kind !=
                   EeSchedulerOwnerLocalTransitionKind::None;
        }

        [[nodiscard]] bool structurallyValid() const noexcept
        {
            switch (kind)
            {
            case EeSchedulerOwnerLocalTransitionKind::None:
                return originatingThread ==
                           EeSchedulerThreadHandle{} &&
                       priority == 0 &&
                       reschedulePolicy ==
                           EeSchedulerReschedulePolicy::None;
            case EeSchedulerOwnerLocalTransitionKind::
                RotateReadyQueue:
                return originatingThread.valid() &&
                       priority >= 0 &&
                       priority < kEeThreadPriorityCount &&
                       (reschedulePolicy ==
                            EeSchedulerReschedulePolicy::None ||
                        reschedulePolicy ==
                            EeSchedulerReschedulePolicy::
                                EqualOrHigherPriority);
            }
            return false;
        }

        friend bool operator==(
            const EeSchedulerOwnerLocalTransition &,
            const EeSchedulerOwnerLocalTransition &) = default;
    };

    enum class EeSchedulerSleepDisposition : uint8_t
    {
        InvalidCurrentThread,
        WakeupConsumed,
        Blocked,
    };

    struct EeSchedulerSleepResult
    {
        EeSchedulerSleepDisposition disposition =
            EeSchedulerSleepDisposition::
                InvalidCurrentThread;
        std::optional<int> selectedThreadId;
    };

    enum class EeSchedulerWakeResult : uint8_t
    {
        InvalidHandle,
        Dormant,
        WokeSleeper,
        WakeupCounted,
        WakeupCountOverflow,
    };

    struct EeSchedulerRunResult
    {
        EeSchedulerExitReason reason =
            EeSchedulerExitReason::Preempted;
        uint64_t elapsedTicks = 0u;
        EeSchedulerWaitKey wait{};
        std::exception_ptr failure;
        EeSchedulerOwnerLocalTransition
            ownerLocalTransition{};
    };

    // Validates the complete exit protocol, including legal reason/wait,
    // failure, and owner-local transition combinations.
    [[nodiscard]] bool eeSchedulerRunResultValid(
        const EeSchedulerRunResult &result) noexcept;

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
        [[nodiscard]] bool startThread(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] std::optional<int>
        selectNextThread();
        [[nodiscard]] std::optional<int>
        yieldCurrentThread();
        [[nodiscard]] std::optional<int>
        blockCurrentThread(EeSchedulerWaitKey wait);
        [[nodiscard]] EeSchedulerSleepResult
        sleepCurrentThread();
        [[nodiscard]] std::optional<int>
        wakeOne(EeSchedulerWaitKey wait);
        [[nodiscard]] EeSchedulerWakeResult
        wakeThread(EeSchedulerThreadHandle thread);
        [[nodiscard]] std::optional<uint32_t>
        cancelWakeups(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] bool releaseWaitThread(
            EeSchedulerThreadHandle thread,
            EeSchedulerWaitCompletion completion,
            bool retainReasonForStatus = false);
        [[nodiscard]] std::optional<
            EeSchedulerCompletedWait>
        consumeWaitCompletion(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] std::optional<int>
        finishCurrentThread();
        [[nodiscard]] bool suspendThread(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] bool resumeThread(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] bool terminateThread(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] bool exitCurrentThread();
        [[nodiscard]] bool deleteThread(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] std::optional<int>
        changeThreadPriority(
            EeSchedulerThreadHandle thread,
            int priority);
        [[nodiscard]] bool rotateReadyQueue(
            int priority);
        [[nodiscard]] std::optional<int>
        reschedule(
            EeSchedulerReschedulePolicy policy);

        [[nodiscard]] std::optional<EeSchedulerDispatch>
        dispatchOne(
            IEeSchedulerExecutionBackend &backend,
            uint64_t tickBudget);

        [[nodiscard]] std::optional<int>
        currentThreadId() const noexcept;
        [[nodiscard]] std::optional<
            EeSchedulerThreadHandle>
        threadHandle(int threadId) const;
        [[nodiscard]] std::optional<EeSchedulerThreadSnapshot>
        thread(int threadId) const;
        [[nodiscard]] std::vector<int>
        readyOrder(int priority) const;
        [[nodiscard]] std::vector<int>
        waitOrder(EeSchedulerWaitKey wait) const;
        [[nodiscard]] size_t threadCount() const noexcept;
        [[nodiscard]] bool validate() const;
        [[nodiscard]] EeThreadSchedulerSnapshot snapshot() const;
        [[nodiscard]] bool restore(
            const EeThreadSchedulerSnapshot &snapshot);
        // Enabling starts a fresh bounded capture. Overflow retains the newest
        // records and increments droppedTransitionTraceCount. Disabling stops
        // publication without discarding the captured snapshot.
        void enableTransitionTracing(
            size_t capacity =
                kDefaultEeSchedulerTransitionTraceCapacity)
            noexcept;
        void disableTransitionTracing() noexcept;
        void clearTransitionTrace() noexcept;
        [[nodiscard]] bool transitionTracingEnabled()
            const noexcept;
        [[nodiscard]] size_t transitionTraceCapacity()
            const noexcept;
        [[nodiscard]] uint64_t
        droppedTransitionTraceCount() const noexcept;
        [[nodiscard]] std::vector<
            EeSchedulerTransitionTrace>
        transitionTraceSnapshot() const;

    private:
        friend class EeSchedulerExecutor;

        struct ThreadRecord
        {
            uint32_t generation = 0u;
            int priority = 0;
            EeSchedulerThreadState state =
                EeSchedulerThreadState::Dormant;
            EeSchedulerWaitKey wait{};
            EeSchedulerCompletedWait
                completedWait{};
            bool retainCompletedWaitReason = false;
            uint32_t wakeupCount = 0u;
            uint32_t suspendCount = 0u;
            uint64_t queueSequence = 0u;
        };

        [[nodiscard]] static bool validThreadId(
            int threadId) noexcept;
        [[nodiscard]] static bool validPriority(
            int priority) noexcept;
        void enqueueReady(
            int threadId,
            ThreadRecord &thread);
        [[nodiscard]] ThreadRecord *findThread(
            EeSchedulerThreadHandle thread);
        [[nodiscard]] const ThreadRecord *findThread(
            EeSchedulerThreadHandle thread) const;
        [[nodiscard]] bool removeReady(
            EeSchedulerThreadHandle thread,
            int priority);
        [[nodiscard]] bool removeWait(
            EeSchedulerThreadHandle thread,
            EeSchedulerWaitKey wait);
        [[nodiscard]] bool retireCurrentThread();
        [[nodiscard]] std::optional<int>
        highestReadyPriority() const;
        [[nodiscard]] std::optional<int>
        takeNextReady();
        void publishCanonicalTick(
            ps2x::timing::EeTick tick) noexcept;
        void recordTransition(
            EeSchedulerTransitionReason reason,
            std::optional<int> oldThreadId,
            int threadId,
            EeSchedulerWaitKey wait = {});

        std::unordered_map<int, ThreadRecord> m_threads;
        std::array<
            std::deque<EeSchedulerThreadHandle>,
            kEeThreadPriorityCount>
            m_readyQueues;
        std::map<
            EeSchedulerWaitKey,
            std::deque<EeSchedulerThreadHandle>>
            m_waitQueues;
        std::optional<int> m_currentThreadId;
        uint64_t m_nextQueueSequence = 1u;
        bool m_transitionTracingEnabled = false;
        size_t m_transitionTraceCapacity =
            kDefaultEeSchedulerTransitionTraceCapacity;
        std::deque<EeSchedulerTransitionTrace>
            m_transitionTrace;
        uint64_t m_nextTransitionTraceSequence = 1u;
        uint64_t m_droppedTransitionTraceCount = 0u;
        ps2x::timing::EeTick m_canonicalTick{};
    };
}
