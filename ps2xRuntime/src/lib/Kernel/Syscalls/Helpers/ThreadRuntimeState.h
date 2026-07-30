#pragma once

#include "ps2_runtime.h"

#include <cassert>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

enum class EeThreadWaitQueue : uint8_t
{
    None = 0,
    Sleep = 1,
    Semaphore = 2,
    EventFlag = 3,
};

struct EeThreadGuestStateSnapshot
{
    int status = 0x10; // THS_DORMANT
    int waitType = 0;  // TSW_NONE
    int waitId = 0;
    int wakeupCount = 0;
    int currentPriority = 0;
    int suspendCount = 0;
    EeThreadWaitQueue waitQueue = EeThreadWaitQueue::None;
    int waitQueueId = 0;
    bool waitCompletionPending = false;
    bool started = false;
    uint64_t revision = 0u;
};

// Authoritative guest-visible EE thread state. Callers serialize access with
// ThreadInfo::m; host-worker cancellation and notification remain separate
// mechanics. In particular, a retained wait reason is not queue membership:
// raw DeleteSema publishes READY with a pending SEMA completion after the
// waiter has already been unlinked.
class EeThreadGuestState
{
public:
    static constexpr int kRun = 0x01;
    static constexpr int kReady = 0x02;
    static constexpr int kWait = 0x04;
    static constexpr int kSuspend = 0x08;
    static constexpr int kWaitSuspend = 0x0c;
    static constexpr int kDormant = 0x10;

    static constexpr int kWaitNone = 0;
    static constexpr int kWaitSleep = 1;
    static constexpr int kWaitSemaphore = 2;
    static constexpr int kWaitEvent = 3;

    const EeThreadGuestStateSnapshot &snapshot() const noexcept
    {
        return m_state;
    }

    bool isValid() const noexcept
    {
        return isValid(m_state);
    }

    bool isDormant() const noexcept
    {
        return m_state.status == kDormant;
    }

    bool isWaiting() const noexcept
    {
        return m_state.status == kWait ||
               m_state.status == kWaitSuspend;
    }

    bool isWaitingOn(int waitType, int waitId) const noexcept
    {
        return isWaiting() &&
               m_state.waitType == waitType &&
               m_state.waitId == waitId;
    }

    bool isLinkedTo(EeThreadWaitQueue queue, int id) const noexcept
    {
        return m_state.waitQueue == queue &&
               m_state.waitQueueId == id;
    }

    void initializeRunning(int priority)
    {
        EeThreadGuestStateSnapshot next{};
        next.status = kRun;
        next.currentPriority = priority;
        next.started = true;
        commit(next);
    }

    void setCurrentPriority(int priority)
    {
        mutate([&](EeThreadGuestStateSnapshot &next)
               { next.currentPriority = priority; });
    }

    int changeCurrentPriority(int priority)
    {
        const int previous = m_state.currentPriority;
        setCurrentPriority(priority);
        return previous;
    }

    bool start()
    {
        if (!isDormant())
        {
            return false;
        }

        EeThreadGuestStateSnapshot next = m_state;
        next.status = kReady;
        next.waitType = kWaitNone;
        next.waitId = 0;
        next.wakeupCount = 0;
        next.suspendCount = 0;
        next.waitQueue = EeThreadWaitQueue::None;
        next.waitQueueId = 0;
        next.waitCompletionPending = false;
        next.started = true;
        commit(next);
        return true;
    }

    void publishRunning()
    {
        mutate([](EeThreadGuestStateSnapshot &next)
               {
                   next.status = kRun;
                   next.waitType = kWaitNone;
                   next.waitId = 0;
                   next.waitQueue = EeThreadWaitQueue::None;
                   next.waitQueueId = 0;
                   next.waitCompletionPending = false;
                   next.started = true;
               });
    }

    void publishReady()
    {
        mutate([](EeThreadGuestStateSnapshot &next)
               {
                   next.status = kReady;
                   next.waitType = kWaitNone;
                   next.waitId = 0;
                   next.waitQueue = EeThreadWaitQueue::None;
                   next.waitQueueId = 0;
                   next.waitCompletionPending = false;
                   next.started = true;
               });
    }

    void makeDormant()
    {
        EeThreadGuestStateSnapshot next{};
        next.status = kDormant;
        next.currentPriority = m_state.currentPriority;
        commit(next);
    }

    bool block(EeThreadWaitQueue queue, int waitType, int waitId)
    {
        if (isDormant() || queue == EeThreadWaitQueue::None)
        {
            return false;
        }

        EeThreadGuestStateSnapshot next = m_state;
        next.status =
            next.suspendCount > 0 ? kWaitSuspend : kWait;
        next.waitType = waitType;
        next.waitId = waitId;
        next.waitQueue = queue;
        next.waitQueueId = waitId;
        next.waitCompletionPending = false;
        commit(next);
        return true;
    }

    // Detach a waiter and publish its post-wake state. retainReason is used
    // only for the retail raw-semaphore-delete completion: the thread is
    // runnable and no longer linked, but ReferThreadStatus retains SEMA/id
    // until that continuation is dispatched.
    bool releaseWait(
        EeThreadWaitQueue queue,
        int waitType,
        int waitId,
        bool retainReason)
    {
        if (!isWaitingOn(waitType, waitId) ||
            !isLinkedTo(queue, waitId))
        {
            return false;
        }

        EeThreadGuestStateSnapshot next = m_state;
        next.status =
            next.suspendCount > 0 ? kSuspend : kReady;
        next.waitQueue = EeThreadWaitQueue::None;
        next.waitQueueId = 0;
        next.waitCompletionPending = retainReason;
        if (!retainReason)
        {
            next.waitType = kWaitNone;
            next.waitId = 0;
        }
        commit(next);
        return true;
    }

    // Complete the host continuation after a wake. This also accepts a wait
    // already detached by a synchronous publisher.
    bool finishWait(
        EeThreadWaitQueue queue,
        int waitType,
        int waitId,
        bool publishRunningAfterWake)
    {
        const bool linked = isLinkedTo(queue, waitId);
        const bool matchingReason =
            m_state.waitType == waitType &&
            m_state.waitId == waitId;
        const bool alreadyPublished =
            m_state.waitQueue == EeThreadWaitQueue::None &&
            (m_state.status == kReady ||
             m_state.status == kSuspend) &&
            ((m_state.waitCompletionPending &&
              matchingReason) ||
             (!m_state.waitCompletionPending &&
              m_state.waitType == kWaitNone));
        if (!linked && !alreadyPublished)
        {
            return false;
        }

        EeThreadGuestStateSnapshot next = m_state;
        next.status = next.suspendCount > 0
                          ? kSuspend
                          : (publishRunningAfterWake ? kRun : kReady);
        next.waitType = kWaitNone;
        next.waitId = 0;
        next.waitQueue = EeThreadWaitQueue::None;
        next.waitQueueId = 0;
        next.waitCompletionPending = false;
        commit(next);
        return linked;
    }

    bool suspend()
    {
        if (isDormant())
        {
            return false;
        }

        mutate([](EeThreadGuestStateSnapshot &next)
               {
                   ++next.suspendCount;
                   next.status =
                       next.waitQueue != EeThreadWaitQueue::None
                           ? kWaitSuspend
                           : kSuspend;
               });
        return true;
    }

    bool resume(bool isCurrentThread, bool &becameRunnable)
    {
        becameRunnable = false;
        if (isDormant() || m_state.suspendCount <= 0)
        {
            return false;
        }

        mutate([&](EeThreadGuestStateSnapshot &next)
               {
                   --next.suspendCount;
                   if (next.suspendCount == 0)
                   {
                       if (next.waitQueue != EeThreadWaitQueue::None)
                       {
                           next.status = kWait;
                       }
                       else
                       {
                           next.status =
                               isCurrentThread ? kRun : kReady;
                           becameRunnable = !isCurrentThread;
                       }
                   }
               });
        return true;
    }

    bool consumeWakeup()
    {
        if (m_state.wakeupCount <= 0)
        {
            return false;
        }

        mutate([](EeThreadGuestStateSnapshot &next)
               {
                   --next.wakeupCount;
                   next.status = kRun;
                   next.waitType = kWaitNone;
                   next.waitId = 0;
                   next.waitQueue = EeThreadWaitQueue::None;
                   next.waitQueueId = 0;
                   next.waitCompletionPending = false;
               });
        return true;
    }

    bool wakeSleeping()
    {
        if (!isWaitingOn(kWaitSleep, 0) ||
            !isLinkedTo(EeThreadWaitQueue::Sleep, 0))
        {
            return false;
        }

        return releaseWait(
            EeThreadWaitQueue::Sleep,
            kWaitSleep,
            0,
            false);
    }

    int incrementWakeup()
    {
        int result = 0;
        mutate([&](EeThreadGuestStateSnapshot &next)
               {
                   ++next.wakeupCount;
                   result = next.wakeupCount;
               });
        return result;
    }

    int cancelWakeups()
    {
        const int previous = m_state.wakeupCount;
        mutate([](EeThreadGuestStateSnapshot &next)
               { next.wakeupCount = 0; });
        return previous;
    }

private:
    template <typename Mutator>
    void mutate(Mutator mutator)
    {
        EeThreadGuestStateSnapshot next = m_state;
        mutator(next);
        commit(next);
    }

    void commit(EeThreadGuestStateSnapshot next)
    {
        next.revision = m_state.revision + 1u;
        const bool valid = isValid(next);
        assert(valid && "invalid EE guest-thread state transition");
        if (valid)
        {
            m_state = next;
        }
    }

    static bool isValid(
        const EeThreadGuestStateSnapshot &state) noexcept
    {
        const bool knownStatus =
            state.status == kRun ||
            state.status == kReady ||
            state.status == kWait ||
            state.status == kSuspend ||
            state.status == kWaitSuspend ||
            state.status == kDormant;
        if (!knownStatus ||
            state.wakeupCount < 0 ||
            state.suspendCount < 0)
        {
            return false;
        }

        const bool hasQueue =
            state.waitQueue != EeThreadWaitQueue::None;
        const bool hasWaitReason =
            state.waitType != kWaitNone;
        if (!hasQueue && state.waitQueueId != 0)
        {
            return false;
        }

        if (hasQueue)
        {
            const bool queueMatchesReason =
                (state.waitQueue == EeThreadWaitQueue::Sleep &&
                 state.waitType == kWaitSleep &&
                 state.waitId == 0 &&
                 state.waitQueueId == 0) ||
                (state.waitQueue == EeThreadWaitQueue::Semaphore &&
                 state.waitType == kWaitSemaphore &&
                 state.waitQueueId == state.waitId) ||
                (state.waitQueue == EeThreadWaitQueue::EventFlag &&
                 state.waitType == kWaitEvent &&
                 state.waitQueueId == state.waitId);
            if (!queueMatchesReason)
            {
                return false;
            }
        }

        switch (state.status)
        {
        case kRun:
            return state.started &&
                   state.suspendCount == 0 &&
                   !hasQueue &&
                   !hasWaitReason &&
                   !state.waitCompletionPending;
        case kReady:
            return state.started &&
                   state.suspendCount == 0 &&
                   !hasQueue &&
                   (state.waitCompletionPending
                        ? hasWaitReason
                        : !hasWaitReason);
        case kWait:
            return state.started &&
                   state.suspendCount == 0 &&
                   hasQueue &&
                   hasWaitReason &&
                   !state.waitCompletionPending;
        case kSuspend:
            return state.started &&
                   state.suspendCount > 0 &&
                   !hasQueue &&
                   (state.waitCompletionPending
                        ? hasWaitReason
                        : !hasWaitReason);
        case kWaitSuspend:
            return state.started &&
                   state.suspendCount > 0 &&
                   hasQueue &&
                   hasWaitReason &&
                   !state.waitCompletionPending;
        case kDormant:
            return !state.started &&
                   state.suspendCount == 0 &&
                   state.wakeupCount == 0 &&
                   !hasQueue &&
                   !hasWaitReason &&
                   !state.waitCompletionPending;
        default:
            return false;
        }
    }

    EeThreadGuestStateSnapshot m_state{};
};

struct ThreadInfo
{
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    uint32_t priority = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t arg = 0;
    uint32_t generation = 0;
    bool ownsStack = false;
    uint32_t tlsBase = 0;

    EeThreadGuestState guestState;

    // The legacy host-thread backend executes directly from this
    // runtime-owned context. Main thread ID 1 continues to use
    // PS2Runtime::cpu() as its runtime-owned context.
    R5900Context context{};
    R5900Context *boundContext = nullptr;
    std::atomic<uint32_t> currentPc{0};

    std::mutex m;
    std::condition_variable cv;
    // The dedicated executor consumes the scheduler's completion immediately
    // before resuming a continuation. Keep one mutex-protected mirror for the
    // suspended syscall adapter so signal, forced release, deletion, and
    // timeout remain distinguishable after queue membership is detached.
    ps2x::ee::EeSchedulerWaitCompletion pendingWaitCompletion =
        ps2x::ee::EeSchedulerWaitCompletion::None;
    uint32_t pendingEventFlagResultBits = 0u;
    std::atomic<bool> forceRelease{false};
    std::atomic<bool> terminated{false};
};

struct EeThreadRuntimeState
{
    explicit EeThreadRuntimeState(uint64_t runtimeGenerationValue)
        : runtimeGeneration(runtimeGenerationValue)
    {
        assert(runtimeGeneration != 0u);
    }

    uint32_t allocateThreadGenerationLocked() noexcept
    {
        uint32_t generation = 0u;
        do
        {
            generation = nextGeneration++;
        } while (generation == 0u);
        return generation;
    }

    const uint64_t runtimeGeneration;
    std::mutex threadMapMutex;
    std::unordered_map<int, std::shared_ptr<ThreadInfo>> threads;
    std::unordered_map<const R5900Context *, int> contextThreadIds;
    int nextThreadId = 2;
    uint32_t nextGeneration = 1u;

    std::atomic<int> activeHostThreads{0};

    // This is the runtime authority. The host TLS ID remains a checked
    // adapter until all legacy direct-call sites bind through the executor.
    std::atomic<int> currentThreadId{1};
    std::atomic<uint64_t> legacyAdapterMismatches{0u};
};

// Compatibility view for legacy code that still observes a host TLS slot.
// Runtime context binding owns the value; syscall code must use
// PS2Runtime::currentEeThreadId() instead.
inline thread_local PS2Runtime *g_currentThreadRuntime = nullptr;
inline thread_local int g_currentThreadId = 1;
