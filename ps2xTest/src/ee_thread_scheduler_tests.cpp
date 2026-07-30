#include "MiniTest.h"

#include "runtime/ee_thread_scheduler.h"

#include <cstdint>
#include <deque>
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
