#include "MiniTest.h"

#include "runtime/ee_scheduler_executor.h"
#include "runtime/ee_thread_scheduler.h"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

using namespace ps2x::ee;

namespace
{
    class ScriptedEeBackend final
        : public IEeSchedulerExecutionBackend
    {
    public:
        explicit ScriptedEeBackend(
            std::deque<EeSchedulerRunResult> script)
            : m_script(std::move(script)),
              m_owner(std::this_thread::get_id())
        {
        }

        EeSchedulerRunResult resume(
            int threadId,
            uint64_t tickBudget) override
        {
            m_resumedThreads.push_back(threadId);
            m_tickBudgets.push_back(tickBudget);
            m_singleHostOwner =
                m_singleHostOwner &&
                std::this_thread::get_id() == m_owner;
            if (m_script.empty())
            {
                return {
                    EeSchedulerExitReason::StopRequested,
                    0u,
                    {}};
            }

            const EeSchedulerRunResult result =
                m_script.front();
            m_script.pop_front();
            return result;
        }

        const std::vector<int> &resumedThreads() const
        {
            return m_resumedThreads;
        }

        const std::vector<uint64_t> &tickBudgets() const
        {
            return m_tickBudgets;
        }

        bool singleHostOwner() const
        {
            return m_singleHostOwner;
        }

    private:
        std::deque<EeSchedulerRunResult> m_script;
        std::thread::id m_owner;
        std::vector<int> m_resumedThreads;
        std::vector<uint64_t> m_tickBudgets;
        bool m_singleHostOwner = true;
    };

    bool equals(
        const std::vector<int> &actual,
        std::initializer_list<int> expected)
    {
        return actual ==
               std::vector<int>(expected);
    }

    class ScriptedBoundaryHooks final
        : public IEeSchedulerExecutorHooks
    {
    public:
        struct Commit
        {
            std::optional<int> priorThreadId;
            uint64_t elapsed = 0u;
            uint64_t tick = 0u;
        };

        using Action = std::function<
            void(
                EeThreadScheduler &,
                ScriptedBoundaryHooks &)>;

        void queue(
            EeSchedulerConsequenceStage stage,
            Action action)
        {
            m_actions[
                eeSchedulerConsequenceStageIndex(stage)]
                .push_back(std::move(action));
        }

        void setStorm(
            std::optional<
                EeSchedulerConsequenceStage>
                stage)
        {
            m_stormStage = stage;
        }

        void commitPriorContext(
            std::optional<int> priorThreadId,
            ps2x::timing::EeTickDelta elapsed,
            ps2x::timing::EeTick now) override
        {
            commits.push_back(
                Commit{
                    priorThreadId,
                    elapsed.raw(),
                    now.raw()});
        }

        void publishSelectedContext(
            std::optional<int> selectedThreadId,
            ps2x::timing::EeTick) override
        {
            publishedThreads.push_back(
                selectedThreadId.value_or(0));
        }

        [[nodiscard]] bool
        hasImmediateConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick) const override
        {
            return m_stormStage == stage ||
                   !m_actions[
                        eeSchedulerConsequenceStageIndex(
                            stage)]
                        .empty();
        }

        void applyNextConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick,
            EeThreadScheduler &scheduler) override
        {
            appliedStages.push_back(stage);
            std::deque<Action> &actions =
                m_actions[
                    eeSchedulerConsequenceStageIndex(
                        stage)];
            if (actions.empty())
            {
                return;
            }

            Action action =
                std::move(actions.front());
            actions.pop_front();
            action(scheduler, *this);
        }

        std::vector<Commit> commits;
        std::vector<int> publishedThreads;
        std::vector<EeSchedulerConsequenceStage>
            appliedStages;

    private:
        std::array<
            std::deque<Action>,
            kEeSchedulerConsequenceStageCount>
            m_actions;
        std::optional<EeSchedulerConsequenceStage>
            m_stormStage;
    };
}

void register_ee_thread_scheduler_tests()
{
    MiniTest::Case("EeThreadScheduler", [](TestCase &tc)
    {
        tc.Run("fixed priority buckets retain stable FIFO order", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 40),
                "the fixture should add its running thread");
            t.IsTrue(
                scheduler.addDormantThread(2, 1u, 40) &&
                    scheduler.startThread(2),
                "the first equal-priority thread should become ready");
            t.IsTrue(
                scheduler.addDormantThread(3, 1u, 40) &&
                    scheduler.startThread(3),
                "the second equal-priority thread should become ready");
            t.IsTrue(
                scheduler.addDormantThread(4, 1u, 10) &&
                    scheduler.startThread(4),
                "the higher-priority thread should become ready");

            t.IsTrue(
                equals(
                    scheduler.readyOrder(40),
                    {2, 3}),
                "equal-priority starts should append in FIFO order");
            t.IsTrue(
                equals(
                    scheduler.readyOrder(10),
                    {4}),
                "a separate priority should own a separate bucket");
            t.Equals(
                scheduler.yieldCurrentThread().value_or(0),
                4,
                "yield should select the numerically highest-priority bucket");
            t.IsTrue(
                equals(
                    scheduler.readyOrder(40),
                    {2, 3, 1}),
                "yielding should append the former runner to its bucket tail");
            t.Equals(
                scheduler.finishCurrentThread().value_or(0),
                2,
                "finishing the high-priority thread should select the FIFO head");
            t.Equals(
                scheduler.yieldCurrentThread().value_or(0),
                3,
                "equal-priority yield should select the next FIFO member");
            t.Equals(
                scheduler.yieldCurrentThread().value_or(0),
                1,
                "a second yield should preserve the original FIFO cycle");
            t.IsTrue(
                scheduler.validate(),
                "every fixed-bucket transition should preserve queue invariants");
        });

        tc.Run("explicit wait queues retain object-local FIFO order", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            const EeSchedulerWaitKey semaphore{
                EeSchedulerWaitKind::Semaphore,
                7};

            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 20),
                "the fixture should add its first runner");
            t.IsTrue(
                scheduler.addDormantThread(2, 1u, 20) &&
                    scheduler.startThread(2),
                "the fixture should add its second runner");
            t.Equals(
                scheduler
                    .blockCurrentThread(semaphore)
                    .value_or(0),
                2,
                "blocking the first thread should select the second");
            t.IsTrue(
                !scheduler
                     .blockCurrentThread(semaphore)
                     .has_value(),
                "blocking the second thread should leave the executor idle");
            t.IsTrue(
                equals(
                    scheduler.waitOrder(semaphore),
                    {1, 2}),
                "one wait object should retain waiter FIFO order");

            t.Equals(
                scheduler.wakeOne(semaphore).value_or(0),
                1,
                "the first wake should publish the first waiter");
            t.Equals(
                scheduler.wakeOne(semaphore).value_or(0),
                2,
                "the second wake should publish the second waiter");
            t.IsTrue(
                scheduler.waitOrder(semaphore).empty(),
                "the explicit wait queue should retire after its final wake");
            t.IsTrue(
                equals(
                    scheduler.readyOrder(20),
                    {1, 2}),
                "woken waiters should enter the ready bucket in publication order");
            t.Equals(
                scheduler.selectNextThread().value_or(0),
                1,
                "the executor should select the first published waiter");
            t.IsTrue(
                scheduler.validate(),
                "wait publication should preserve exact queue membership");
        });

        tc.Run("suspend and wake preserve wait-suspended ownership", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            const EeSchedulerWaitKey semaphore{
                EeSchedulerWaitKind::Semaphore,
                7};

            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 20) &&
                    scheduler.addDormantThread(2, 1u, 20) &&
                    scheduler.startThread(2),
                "the fixture should create one runner and one ready peer");
            const EeSchedulerThreadHandle first =
                scheduler.threadHandle(1).value_or(
                    EeSchedulerThreadHandle{});
            const EeSchedulerThreadHandle second =
                scheduler.threadHandle(2).value_or(
                    EeSchedulerThreadHandle{});
            t.Equals(
                scheduler
                    .blockCurrentThread(semaphore)
                    .value_or(0),
                2,
                "blocking the runner should select its peer");
            t.IsTrue(
                scheduler.suspendThread(first) &&
                    scheduler.suspendThread(first),
                "nested suspension should retain one wait membership");

            const auto waitSuspended =
                scheduler.thread(1);
            t.IsTrue(
                waitSuspended.has_value() &&
                    waitSuspended->state ==
                        EeSchedulerThreadState::WaitSuspended &&
                    waitSuspended->suspendCount == 2u &&
                    waitSuspended->wait == semaphore &&
                    equals(
                        scheduler.waitOrder(semaphore),
                        {1}),
                "a suspended waiter should report WAITSUSPEND without changing queues");
            t.Equals(
                scheduler.wakeOne(semaphore).value_or(0),
                1,
                "wake should synchronously detach the wait-suspended thread");

            const auto suspended =
                scheduler.thread(1);
            t.IsTrue(
                suspended.has_value() &&
                    suspended->state ==
                        EeSchedulerThreadState::Suspended &&
                    suspended->suspendCount == 2u &&
                    !suspended->wait.valid() &&
                    scheduler.waitOrder(semaphore).empty() &&
                    scheduler.readyOrder(20).empty(),
                "a woken suspended waiter should become SUSPEND without becoming ready");
            t.IsTrue(
                scheduler.resumeThread(first),
                "the first resume should consume one nested suspend");
            const auto onceResumed =
                scheduler.thread(1);
            t.IsTrue(
                onceResumed.has_value() &&
                    onceResumed->state ==
                        EeSchedulerThreadState::Suspended &&
                    onceResumed->suspendCount == 1u,
                "one retained suspension should keep the thread suspended");
            t.IsTrue(
                scheduler.resumeThread(first) &&
                    equals(
                        scheduler.readyOrder(20),
                        {1}) &&
                    scheduler.currentThreadId().value_or(0) ==
                        2,
                "the final resume should publish READY without preempting the caller");

            t.IsTrue(
                scheduler.suspendThread(second),
                "suspending the running thread should select the ready peer");
            t.IsTrue(
                scheduler.currentThreadId().value_or(0) ==
                        1 &&
                    scheduler.thread(2)->state ==
                        EeSchedulerThreadState::Suspended,
                "a running suspension should leave exactly one replacement RUN thread");
            t.IsTrue(
                scheduler.resumeThread(second) &&
                    equals(
                        scheduler.readyOrder(20),
                        {2}),
                "resuming the former runner should append it to READY");
            t.IsTrue(
                scheduler.validate(),
                "WAIT/SUSPEND composition should preserve exact ownership");
        });

        tc.Run("terminate exit and delete reject recycled generations", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            const EeSchedulerWaitKey semaphore{
                EeSchedulerWaitKind::Semaphore,
                4};

            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 40) &&
                    scheduler.addDormantThread(2, 1u, 40) &&
                    scheduler.startThread(2),
                "the fixture should create a RUN/READY pair");
            const EeSchedulerThreadHandle first =
                scheduler.threadHandle(1).value_or(
                    EeSchedulerThreadHandle{});
            const EeSchedulerThreadHandle staleSecond =
                scheduler.threadHandle(2).value_or(
                    EeSchedulerThreadHandle{});

            t.IsTrue(
                scheduler.terminateThread(staleSecond) &&
                    scheduler.thread(2)->state ==
                        EeSchedulerThreadState::Dormant &&
                    scheduler.readyOrder(40).empty(),
                "termination should unlink a READY target and retain its dormant object");
            t.IsTrue(
                scheduler.deleteThread(staleSecond) &&
                    !scheduler.thread(2).has_value(),
                "delete should remove only a dormant generation");
            t.IsTrue(
                scheduler.addDormantThread(2, 2u, 40),
                "the deleted numeric ID should accept a replacement generation");
            const EeSchedulerThreadHandle replacement =
                scheduler.threadHandle(2).value_or(
                    EeSchedulerThreadHandle{});
            t.IsTrue(
                !scheduler.startThread(staleSecond) &&
                    !scheduler.suspendThread(staleSecond) &&
                    !scheduler.terminateThread(staleSecond) &&
                    !scheduler.deleteThread(staleSecond),
                "an internal stale generation must not address the replacement");
            t.IsTrue(
                scheduler.startThread(replacement),
                "the replacement generation should start normally");
            t.Equals(
                scheduler
                    .blockCurrentThread(semaphore)
                    .value_or(0),
                2,
                "blocking the first generation should select the replacement");
            t.IsTrue(
                scheduler.terminateThread(first) &&
                    scheduler.waitOrder(semaphore).empty() &&
                    scheduler.thread(1)->state ==
                        EeSchedulerThreadState::Dormant,
                "termination should synchronously unlink a waiting target");
            t.IsTrue(
                scheduler.deleteThread(first),
                "a terminated waiter should be deletable");
            t.IsTrue(
                scheduler.exitCurrentThread() &&
                    !scheduler.currentThreadId().has_value() &&
                    scheduler.thread(2)->state ==
                        EeSchedulerThreadState::Dormant,
                "self-exit should retain a dormant object and leave an idle executor");
            t.IsTrue(
                scheduler.deleteThread(replacement) &&
                    scheduler.threadCount() == 0u &&
                    scheduler.validate(),
                "deleting the exited replacement should leave an empty valid scheduler");
        });

        tc.Run("named ready-queue rotation is distinct from rescheduling", [](TestCase &t)
        {
            {
                EeThreadScheduler scheduler;
                t.IsTrue(
                    scheduler.addRunningThread(1, 1u, 40) &&
                        scheduler.addDormantThread(2, 1u, 40) &&
                        scheduler.startThread(2),
                    "the current-priority fixture should create one equal-priority peer");
                t.IsTrue(
                    scheduler.rotateReadyQueue(40) &&
                        equals(
                            scheduler.readyOrder(40),
                            {2}),
                    "rotating a singleton READY queue should not pretend RUN belongs to it");
                t.Equals(
                    scheduler
                        .reschedule(
                            EeSchedulerReschedulePolicy::
                                HigherPriorityOnly)
                        .value_or(0),
                    1,
                    "a raw current-priority rotation should leave the caller RUN");
                t.Equals(
                    scheduler
                        .reschedule(
                            EeSchedulerReschedulePolicy::
                                EqualOrHigherPriority)
                        .value_or(0),
                    2,
                    "the ordinary rotation boundary should dispatch the equal-priority peer");
                t.IsTrue(
                    equals(
                        scheduler.readyOrder(40),
                        {1}) &&
                        scheduler.validate(),
                    "ordinary equal-priority dispatch should append the former runner");
            }

            {
                EeThreadScheduler scheduler;
                t.IsTrue(
                    scheduler.addRunningThread(1, 1u, 40) &&
                        scheduler.addDormantThread(2, 1u, 50) &&
                        scheduler.startThread(2) &&
                        scheduler.addDormantThread(3, 1u, 50) &&
                        scheduler.startThread(3),
                    "the non-current fixture should create two lower-priority waiters");
                const uint64_t firstSequence =
                    scheduler.thread(2)->queueSequence;
                const uint64_t secondSequence =
                    scheduler.thread(3)->queueSequence;
                t.IsTrue(
                    scheduler.rotateReadyQueue(50) &&
                        equals(
                            scheduler.readyOrder(50),
                            {3, 2}),
                    "named rotation should move only the selected FIFO head");
                t.IsTrue(
                    scheduler.thread(2)->queueSequence >
                            secondSequence &&
                        secondSequence > firstSequence,
                    "the rotated tail should receive the newest publication sequence");
                t.Equals(
                    scheduler
                        .reschedule(
                            EeSchedulerReschedulePolicy::
                                HigherPriorityOnly)
                        .value_or(0),
                    1,
                    "rotating a lower-priority queue should not run it");

                const EeSchedulerThreadHandle main =
                    scheduler.threadHandle(1).value_or(
                        EeSchedulerThreadHandle{});
                t.Equals(
                    scheduler
                        .changeThreadPriority(main, 60)
                        .value_or(-1),
                    40,
                    "lowering the caller should publish its new priority");
                t.Equals(
                    scheduler
                        .reschedule(
                            EeSchedulerReschedulePolicy::
                                HigherPriorityOnly)
                        .value_or(0),
                    3,
                    "the next boundary should select the rotated non-current FIFO head");
                t.IsTrue(
                    equals(
                        scheduler.readyOrder(50),
                        {2}) &&
                        equals(
                            scheduler.readyOrder(60),
                            {1}) &&
                        !scheduler.rotateReadyQueue(128) &&
                        scheduler.validate(),
                    "non-current rotation should preserve every other membership");
            }
        });

        tc.Run("raw publications defer ordinary higher-priority dispatch", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            const EeSchedulerWaitKey sleep{
                EeSchedulerWaitKind::Sleep,
                0};
            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 40) &&
                    scheduler.addDormantThread(2, 1u, 30),
                "the fixture should create a lower-priority caller and dormant worker");
            const EeSchedulerThreadHandle main =
                scheduler.threadHandle(1).value_or(
                    EeSchedulerThreadHandle{});
            const EeSchedulerThreadHandle worker =
                scheduler.threadHandle(2).value_or(
                    EeSchedulerThreadHandle{});

            t.IsTrue(
                scheduler.startThread(worker) &&
                    scheduler.currentThreadId().value_or(0) ==
                        1,
                "raw start should publish READY without dispatch");
            t.Equals(
                scheduler
                    .reschedule(
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly)
                    .value_or(0),
                2,
                "the ordinary start boundary should dispatch a higher-priority target");
            t.Equals(
                scheduler
                    .blockCurrentThread(sleep)
                    .value_or(0),
                1,
                "the higher-priority worker should be able to sleep beneath its caller");

            t.Equals(
                scheduler.wakeOne(sleep).value_or(0),
                2,
                "raw wake should synchronously publish the sleeper");
            t.IsTrue(
                scheduler.currentThreadId().value_or(0) ==
                    1,
                "raw wake should not preempt its caller");
            t.Equals(
                scheduler
                    .reschedule(
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly)
                    .value_or(0),
                2,
                "the ordinary wake boundary should dispatch the higher-priority sleeper");
            t.Equals(
                scheduler
                    .blockCurrentThread(sleep)
                    .value_or(0),
                1,
                "the worker should re-enter the same explicit sleep queue");

            t.IsTrue(
                scheduler.suspendThread(worker) &&
                    scheduler.wakeOne(sleep).value_or(0) ==
                        2 &&
                    scheduler.resumeThread(worker),
                "raw wake and resume should publish a wait-suspended worker as READY");
            t.IsTrue(
                scheduler.currentThreadId().value_or(0) ==
                        1 &&
                    scheduler.thread(2)->state ==
                        EeSchedulerThreadState::Ready,
                "raw resume should preserve the caller until a boundary");
            t.Equals(
                scheduler
                    .reschedule(
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly)
                    .value_or(0),
                2,
                "the ordinary resume boundary should dispatch the higher-priority worker");

            t.Equals(
                scheduler
                    .changeThreadPriority(main, 20)
                    .value_or(-1),
                40,
                "raw priority change should publish a promoted READY target");
            t.IsTrue(
                scheduler.currentThreadId().value_or(0) ==
                    2,
                "raw priority change should not preempt the worker");
            t.Equals(
                scheduler
                    .reschedule(
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly)
                    .value_or(0),
                1,
                "the ordinary priority boundary should dispatch the promoted target");

            t.IsTrue(
                scheduler.addDormantThread(3, 1u, 10) &&
                    scheduler.startThread(3) &&
                    scheduler.terminateThread(worker),
                "raw termination should coexist with an unrelated higher-priority READY thread");
            t.IsTrue(
                scheduler.currentThreadId().value_or(0) ==
                    1,
                "raw termination should leave its caller in control");
            t.Equals(
                scheduler
                    .reschedule(
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly)
                    .value_or(0),
                3,
                "the ordinary termination boundary should dispatch unrelated higher-priority work");
            t.IsTrue(
                scheduler.validate(),
                "ordinary/raw publication boundaries should preserve exact queue ownership");
        });

        tc.Run("executor drains same-tick consequences to a stable selection", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            EeSchedulerExecutor executor;
            ScriptedBoundaryHooks hooks;
            const EeSchedulerWaitKey wakeWait{
                EeSchedulerWaitKind::Semaphore,
                1};
            const EeSchedulerWaitKey timeoutWait{
                EeSchedulerWaitKind::Semaphore,
                2};

            t.IsTrue(
                scheduler.addRunningThread(3, 1u, 20) &&
                    scheduler.addDormantThread(1, 1u, 40) &&
                    scheduler.startThread(1) &&
                    scheduler.addDormantThread(2, 1u, 40) &&
                    scheduler.startThread(2),
                "the fixture should create its first high-priority waiter");
            t.Equals(
                scheduler
                    .blockCurrentThread(wakeWait)
                    .value_or(0),
                1,
                "the first high-priority thread should wait for an asynchronous wake");
            t.IsTrue(
                scheduler.addDormantThread(4, 1u, 10) &&
                    scheduler.startThread(4),
                "the fixture should publish a timeout target");
            t.Equals(
                scheduler
                    .reschedule(
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly)
                    .value_or(0),
                4,
                "the timeout target should run before it blocks");
            t.Equals(
                scheduler
                    .blockCurrentThread(timeoutWait)
                    .value_or(0),
                2,
                "blocking the timeout target should restore a priority-40 runner");
            t.IsTrue(
                scheduler.addDormantThread(5, 1u, 40) &&
                    scheduler.startThread(5) &&
                    scheduler.rotateReadyQueue(40) &&
                    equals(
                        scheduler.readyOrder(40),
                        {5, 1}),
                "the explicit same-tick rotation should occur before boundary consequences");

            hooks.queue(
                EeSchedulerConsequenceStage::
                    AsynchronousWake,
                [wakeWait](
                    EeThreadScheduler &target,
                    ScriptedBoundaryHooks &)
                {
                    (void)target.wakeOne(wakeWait);
                });
            hooks.queue(
                EeSchedulerConsequenceStage::WaitTimeout,
                [timeoutWait](
                    EeThreadScheduler &target,
                    ScriptedBoundaryHooks &)
                {
                    (void)target.wakeOne(timeoutWait);
                });
            hooks.queue(
                EeSchedulerConsequenceStage::HardwareEvent,
                [](
                    EeThreadScheduler &,
                    ScriptedBoundaryHooks &targetHooks)
                {
                    targetHooks.queue(
                        EeSchedulerConsequenceStage::
                            AsynchronousWake,
                        [](
                            EeThreadScheduler &,
                            ScriptedBoundaryHooks &)
                        {
                        });
                });
            hooks.queue(
                EeSchedulerConsequenceStage::InterruptCause,
                [](
                    EeThreadScheduler &target,
                    ScriptedBoundaryHooks &)
                {
                    (void)target.rotateReadyQueue(40);
                });

            const EeSchedulerBoundaryResult first =
                executor.processBoundary(
                    scheduler,
                    hooks,
                    2,
                    ps2x::timing::eeTickDeltaFromRaw(
                        8u),
                    EeSchedulerReschedulePolicy::
                        EqualOrHigherPriority);
            t.IsTrue(
                first.tick.raw() == 8u &&
                    first.selectedThreadId.value_or(0) ==
                        4 &&
                    first.consequencesProcessed == 5u &&
                    first.contextPublications == 3u &&
                    !first.limitExceeded,
                "one boundary should commit time and drain every immediate consequence");
            t.IsTrue(
                hooks.appliedStages ==
                    std::vector<
                        EeSchedulerConsequenceStage>({
                        EeSchedulerConsequenceStage::
                            AsynchronousWake,
                        EeSchedulerConsequenceStage::
                            WaitTimeout,
                        EeSchedulerConsequenceStage::
                            HardwareEvent,
                        EeSchedulerConsequenceStage::
                            AsynchronousWake,
                        EeSchedulerConsequenceStage::
                            InterruptCause,
                    }),
                "same-tick stages should use fixed priority and restart after recursive publication");
            t.IsTrue(
                equals(
                    hooks.publishedThreads,
                    {5, 3, 4}),
                "context mirrors should publish the provisional and each newly selected thread");
            t.IsTrue(
                equals(
                    scheduler.readyOrder(40),
                    {2, 5, 1}) &&
                    equals(
                        scheduler.readyOrder(20),
                        {3}),
                "the same-tick interrupt rotation should preserve all prior wake selections");
            t.IsTrue(
                first.stageCounts[
                    eeSchedulerConsequenceStageIndex(
                        EeSchedulerConsequenceStage::
                            AsynchronousWake)] == 2u &&
                    first.stageCounts[
                        eeSchedulerConsequenceStageIndex(
                            EeSchedulerConsequenceStage::
                                WaitTimeout)] == 1u &&
                    first.stageCounts[
                        eeSchedulerConsequenceStageIndex(
                            EeSchedulerConsequenceStage::
                                HardwareEvent)] == 1u &&
                    first.stageCounts[
                        eeSchedulerConsequenceStageIndex(
                            EeSchedulerConsequenceStage::
                                InterruptCause)] == 1u,
                "the boundary result should retain exact stage diagnostics");

            const EeSchedulerBoundaryResult second =
                executor.processBoundary(
                    scheduler,
                    hooks,
                    4,
                    ps2x::timing::eeTickDeltaFromRaw(
                        12u),
                    EeSchedulerReschedulePolicy::
                        HigherPriorityOnly);
            t.IsTrue(
                second.tick.raw() == 20u &&
                    second.consequencesProcessed == 0u &&
                    second.selectedThreadId.value_or(0) ==
                        4 &&
                    hooks.commits.size() == 2u &&
                    hooks.commits[0].priorThreadId ==
                        std::optional<int>(2) &&
                    hooks.commits[0].elapsed == 8u &&
                    hooks.commits[0].tick == 8u &&
                    hooks.commits[1].priorThreadId ==
                        std::optional<int>(4) &&
                    hooks.commits[1].elapsed == 12u &&
                    hooks.commits[1].tick == 20u,
                "successive boundaries should commit each prior context without rewinding time");
            t.IsTrue(
                executor.statistics().boundaries == 2u &&
                    executor.statistics().consequences ==
                        5u &&
                    executor.statistics().
                            contextPublications == 4u &&
                    scheduler.validate(),
                "stable-boundary statistics and scheduler invariants should agree");
        });

        tc.Run("executor rejects recursion and bounds consequence storms", [](TestCase &t)
        {
            {
                EeThreadScheduler scheduler;
                EeSchedulerExecutor executor;
                ScriptedBoundaryHooks hooks;
                std::optional<
                    EeSchedulerBoundaryResult>
                    nested;
                t.IsTrue(
                    scheduler.addRunningThread(
                        1, 1u, 40),
                    "the recursion fixture should have one runner");
                hooks.queue(
                    EeSchedulerConsequenceStage::
                        AsynchronousWake,
                    [&executor, &nested](
                        EeThreadScheduler &target,
                        ScriptedBoundaryHooks &targetHooks)
                    {
                        nested =
                            executor.processBoundary(
                                target,
                                targetHooks,
                                1,
                                ps2x::timing::
                                    eeTickDeltaFromRaw(
                                        99u),
                                EeSchedulerReschedulePolicy::
                                    HigherPriorityOnly);
                    });

                const EeSchedulerBoundaryResult outer =
                    executor.processBoundary(
                        scheduler,
                        hooks,
                        1,
                        ps2x::timing::
                            eeTickDeltaFromRaw(4u),
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly);
                t.IsTrue(
                    !outer.reentrantBoundaryRejected &&
                        nested.has_value() &&
                        nested->
                            reentrantBoundaryRejected &&
                        nested->tick.raw() == 4u &&
                        executor.now().raw() == 4u &&
                        executor.statistics().
                                reentrantBoundaryRejects ==
                            1u,
                    "a nested boundary should be rejected without committing its elapsed time");
            }

            {
                EeThreadScheduler scheduler;
                EeSchedulerExecutor executor;
                ScriptedBoundaryHooks hooks;
                t.IsTrue(
                    scheduler.addRunningThread(
                        1, 1u, 40),
                    "the storm fixture should have one runner");
                hooks.setStorm(
                    EeSchedulerConsequenceStage::
                        AsynchronousWake);
                const EeSchedulerBoundaryResult result =
                    executor.processBoundary(
                        scheduler,
                        hooks,
                        1,
                        ps2x::timing::
                            eeTickDeltaFromRaw(1u),
                        EeSchedulerReschedulePolicy::
                            HigherPriorityOnly);

                t.IsTrue(
                    result.limitExceeded &&
                        result.consequencesProcessed ==
                            EeSchedulerExecutor::
                                kMaximumConsequencesPerBoundary &&
                        result.offendingStage ==
                            EeSchedulerConsequenceStage::
                                AsynchronousWake &&
                        executor.statistics().
                                consequenceLimitHits ==
                            1u,
                    "an immediate consequence storm should stop at the fixed bound with its stage");
                t.IsTrue(
                    scheduler.currentThreadId().
                            value_or(0) ==
                        1 &&
                        scheduler.validate(),
                    "storm diagnostics should leave the selected scheduler state valid");
            }
        });

        tc.Run("priority changes migrate only ready membership", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 40) &&
                    scheduler.addDormantThread(2, 1u, 40) &&
                    scheduler.startThread(2) &&
                    scheduler.addDormantThread(3, 1u, 40) &&
                    scheduler.startThread(3),
                "the fixture should create one runner and two FIFO peers");
            const EeSchedulerThreadHandle second =
                scheduler.threadHandle(2).value_or(
                    EeSchedulerThreadHandle{});
            const EeSchedulerThreadHandle third =
                scheduler.threadHandle(3).value_or(
                    EeSchedulerThreadHandle{});

            t.Equals(
                scheduler
                    .changeThreadPriority(third, 10)
                    .value_or(-1),
                40,
                "priority change should return the previous priority");
            t.IsTrue(
                equals(
                    scheduler.readyOrder(10),
                    {3}) &&
                    equals(
                        scheduler.readyOrder(40),
                        {2}),
                "a READY target should move between fixed buckets");
            t.Equals(
                scheduler.yieldCurrentThread().value_or(0),
                3,
                "the promoted READY target should win at the next boundary");
            t.Equals(
                scheduler
                    .changeThreadPriority(third, 50)
                    .value_or(-1),
                10,
                "changing RUN priority should not manufacture queue membership");
            t.IsTrue(
                scheduler.currentThreadId().value_or(0) ==
                        3 &&
                    scheduler.readyOrder(50).empty(),
                "a raw RUN priority publication should preserve the caller");

            t.Equals(
                scheduler
                    .changeThreadPriority(second, 5)
                    .value_or(-1),
                40,
                "a second READY priority change should retain its old value");
            t.IsTrue(
                !scheduler
                     .changeThreadPriority(second, 128)
                     .has_value() &&
                    scheduler.thread(2)->priority == 5,
                "an invalid priority should leave the target unchanged");
            t.IsTrue(
                scheduler.suspendThread(second),
                "the READY target should be removable into SUSPEND");
            t.Equals(
                scheduler
                    .changeThreadPriority(second, 15)
                    .value_or(-1),
                5,
                "a suspended priority should change without entering READY");
            t.IsTrue(
                scheduler.resumeThread(second) &&
                    equals(
                        scheduler.readyOrder(15),
                        {2}),
                "resume should publish the suspended target at its new priority");
            t.Equals(
                scheduler.finishCurrentThread().value_or(0),
                2,
                "finishing the runner should select the promoted resumed target");
            t.IsTrue(
                !scheduler
                     .changeThreadPriority(third, 1)
                     .has_value() &&
                    scheduler.validate(),
                "a dormant thread should reject priority changes without corrupting queues");
        });

        tc.Run("scripted backend drives exits without host continuations", [](TestCase &t)
        {
            EeThreadScheduler scheduler;
            const EeSchedulerWaitKey event{
                EeSchedulerWaitKind::EventFlag,
                3};

            t.IsTrue(
                scheduler.addRunningThread(1, 1u, 40),
                "the fixture should add its main runner");
            t.IsTrue(
                scheduler.addDormantThread(2, 1u, 40) &&
                    scheduler.startThread(2),
                "the fixture should add an equal-priority worker");
            t.IsTrue(
                scheduler.addDormantThread(3, 1u, 10) &&
                    scheduler.startThread(3),
                "the fixture should add a high-priority worker");

            ScriptedEeBackend backend({
                {
                    EeSchedulerExitReason::Yielded,
                    8u,
                    {},
                },
                {
                    EeSchedulerExitReason::Blocked,
                    16u,
                    event,
                },
                {
                    EeSchedulerExitReason::Finished,
                    24u,
                    {},
                },
            });

            const auto first =
                scheduler.dispatchOne(backend, 64u);
            const auto second =
                scheduler.dispatchOne(backend, 64u);
            const auto third =
                scheduler.dispatchOne(backend, 64u);
            t.IsTrue(
                first.has_value() &&
                    first->resumedThreadId == 1 &&
                    first->selectedThreadId == 3,
                "a scripted yield should select the higher-priority worker");
            t.IsTrue(
                second.has_value() &&
                    second->resumedThreadId == 3 &&
                    second->selectedThreadId == 2,
                "a scripted block should enter its explicit wait queue");
            t.IsTrue(
                third.has_value() &&
                    third->resumedThreadId == 2 &&
                    third->selectedThreadId == 1,
                "a scripted finish should select the retained FIFO runner");
            t.IsTrue(
                equals(
                    backend.resumedThreads(),
                    {1, 3, 2}),
                "the fake backend should observe deterministic resume order");
            t.IsTrue(
                backend.tickBudgets() ==
                    std::vector<uint64_t>(
                        {64u, 64u, 64u}),
                "the scheduler should pass the exact fixed budget");
            t.IsTrue(
                backend.singleHostOwner(),
                "every scripted resume should execute on the one caller thread");
            t.IsTrue(
                equals(
                    scheduler.waitOrder(event),
                    {3}),
                "the scripted block should retain explicit wait ownership");
            t.IsTrue(
                scheduler.validate(),
                "scripted exits should preserve scheduler invariants");
        });
    });
}
