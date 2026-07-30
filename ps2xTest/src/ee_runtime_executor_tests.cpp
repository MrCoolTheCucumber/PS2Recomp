#include "MiniTest.h"

#include "runtime/ee_runtime_executor.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ps2x::ee;

namespace
{
    using namespace std::chrono_literals;

    template <typename Predicate>
    bool waitFor(
        std::condition_variable &cv,
        std::mutex &mutex,
        Predicate predicate)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(
            lock, 2s, std::move(predicate));
    }

    class ScriptedRuntimeBackend final
        : public IEeExecutionBackend
    {
    public:
        using Step =
            std::function<EeSchedulerRunResult()>;

        void plan(
            int threadId,
            std::vector<Step> steps)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::deque<Step> queue;
            for (Step &step : steps)
            {
                queue.push_back(std::move(step));
            }
            m_plans[threadId] = std::move(queue);
        }

        [[nodiscard]] EeExecutionBackendKind
        kind() const noexcept override
        {
            return EeExecutionBackendKind::LegacyCppFiber;
        }

        [[nodiscard]] std::string_view
        name() const noexcept override
        {
            return "scripted-runtime-executor";
        }

        [[nodiscard]] bool
        executorResumable() const noexcept override
        {
            return true;
        }

        void create(
            int threadId,
            ThreadEntry entry) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            requireOwnerLocked("create");
            if (threadId <= 0 || !entry)
            {
                throw std::invalid_argument(
                    "invalid scripted continuation");
            }
            const auto plan = m_plans.find(threadId);
            if (plan == m_plans.end() ||
                plan->second.empty() ||
                m_managed.contains(threadId))
            {
                throw std::logic_error(
                    "missing or duplicate scripted "
                    "continuation plan");
            }
            m_managed.emplace(
                threadId,
                Record{std::move(plan->second), false});
            m_plans.erase(plan);
        }

        EeSchedulerRunResult resume(
            int threadId,
            uint64_t) override
        {
            Step step;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                requireOwnerLocked("resume");
                const auto record =
                    m_managed.find(threadId);
                if (record == m_managed.end() ||
                    record->second.finished ||
                    record->second.steps.empty())
                {
                    throw std::logic_error(
                        "scripted continuation is not "
                        "resumable");
                }
                step =
                    std::move(
                        record->second.steps.front());
                record->second.steps.pop_front();
            }

            EeSchedulerRunResult result = step();
            if (result.reason ==
                    EeSchedulerExitReason::Finished ||
                result.reason ==
                    EeSchedulerExitReason::Exception)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                const auto record =
                    m_managed.find(threadId);
                if (record != m_managed.end())
                {
                    record->second.finished = true;
                }
            }
            return result;
        }

        void yieldCurrent(
            EeSchedulerRunResult) override
        {
            throw std::logic_error(
                "scripted backend has no native "
                "continuation");
        }

        [[nodiscard]] bool
        isFinished(int threadId) const override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto record = m_managed.find(threadId);
            return record == m_managed.end() ||
                   record->second.finished;
        }

        void destroy(int threadId) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            requireOwnerLocked("destroy");
            m_managed.erase(threadId);
        }

        void detach(int threadId) override
        {
            destroy(threadId);
        }

        void joinAll() override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            requireOwnerLocked("join all");
            m_managed.clear();
            m_owner = {};
        }

        void detachAll() override
        {
            joinAll();
        }

        [[nodiscard]] size_t
        managedThreadCount() const override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_managed.size();
        }

    private:
        struct Record
        {
            std::deque<Step> steps;
            bool finished = false;
        };

        void requireOwnerLocked(
            const char *operation)
        {
            const std::thread::id current =
                std::this_thread::get_id();
            if (m_owner == std::thread::id{})
            {
                m_owner = current;
            }
            else if (m_owner != current)
            {
                throw std::logic_error(
                    std::string(
                        "scripted backend wrong thread: ") +
                    operation);
            }
        }

        mutable std::mutex m_mutex;
        std::thread::id m_owner;
        std::unordered_map<int, std::deque<Step>>
            m_plans;
        std::unordered_map<int, Record> m_managed;
    };
}

void register_ee_runtime_executor_tests()
{
    MiniTest::Case("EeRuntimeExecutor", [](TestCase &tc)
    {
        tc.Run(
            "rejects the per-guest host-thread backend",
            [](TestCase &t)
            {
                auto backend =
                    createEeExecutionBackend(
                        EeExecutionBackendKind::
                            LegacyHostThread);
                bool rejected = false;
                try
                {
                    EeRuntimeExecutor executor(*backend);
                }
                catch (const std::invalid_argument &)
                {
                    rejected = true;
                }
                t.IsTrue(
                    rejected,
                    "the single executor must not silently "
                    "create legacy guest workers");
            });

        tc.Run(
            "serializes concurrent host publications",
            [](TestCase &t)
            {
                ScriptedRuntimeBackend backend;
                EeRuntimeExecutor executor(backend);
                int applied = 0;
                std::atomic<int> accepted{0};
                constexpr int publisherCount = 4;
                constexpr int publicationsPerPublisher =
                    128;

                executor.start(
                    [](EeThreadScheduler &,
                       IEeExecutionBackend &)
                    {
                    });
                std::vector<std::thread> publishers;
                for (int publisher = 0;
                     publisher < publisherCount;
                     ++publisher)
                {
                    publishers.emplace_back([&]()
                    {
                        for (int publication = 0;
                             publication <
                             publicationsPerPublisher;
                             ++publication)
                        {
                            if (executor.publish(
                                    [&](EeThreadScheduler &,
                                        IEeExecutionBackend &)
                                    {
                                        ++applied;
                                    }))
                            {
                                accepted.fetch_add(
                                    1,
                                    std::memory_order_relaxed);
                            }
                        }
                    });
                }
                for (std::thread &publisher : publishers)
                {
                    publisher.join();
                }
                executor.invokeAtBoundary(
                    [](EeThreadScheduler &,
                       IEeExecutionBackend &)
                    {
                    });
                executor.requestStop();
                executor.join();
                executor.rethrowFailure();

                const int expected =
                    publisherCount *
                    publicationsPerPublisher;
                const EeRuntimeExecutorStatistics stats =
                    executor.statistics();
                t.IsTrue(
                    accepted.load(
                        std::memory_order_relaxed) ==
                            expected &&
                        applied == expected,
                    "every accepted host publication should "
                    "run exactly once on the executor");
                t.IsTrue(
                    stats.publicationsQueued ==
                            static_cast<uint64_t>(
                                expected + 1) &&
                        stats.publicationsApplied ==
                            static_cast<uint64_t>(
                                expected + 1) &&
                        stats.peakQueuedPublications > 0u,
                    "publication accounting should include "
                    "the final FIFO synchronization command");
            });

        tc.Run(
            "scripted wake publication observes committed wait",
            [](TestCase &t)
            {
                ScriptedRuntimeBackend backend;
                EeRuntimeExecutor executor(backend);
                const EeSchedulerWaitKey wait{
                    EeSchedulerWaitKind::Semaphore,
                    19};
                std::mutex stageMutex;
                std::condition_variable stageCv;
                bool resumeEntered = false;
                bool allowBlockingExit = false;
                bool finished = false;
                bool sawWaiting = false;
                bool released = false;

                backend.plan(
                    1,
                    {
                        [&]()
                        {
                            std::unique_lock<std::mutex>
                                lock(stageMutex);
                            resumeEntered = true;
                            stageCv.notify_all();
                            stageCv.wait(
                                lock,
                                [&]()
                                {
                                    return allowBlockingExit;
                                });
                            return EeSchedulerRunResult{
                                EeSchedulerExitReason::
                                    Blocked,
                                12u,
                                wait,
                                {}};
                        },
                        [&]()
                        {
                            {
                                std::lock_guard<std::mutex>
                                    lock(stageMutex);
                                finished = true;
                            }
                            stageCv.notify_all();
                            return EeSchedulerRunResult{
                                EeSchedulerExitReason::
                                    Finished,
                                3u,
                                {},
                                {}};
                        },
                    });
                executor.start(
                    [](EeThreadScheduler &scheduler,
                       IEeExecutionBackend &selectedBackend)
                    {
                        selectedBackend.create(1, []()
                        {
                        });
                        if (!scheduler.addRunningThread(
                                1, 1u, 10))
                        {
                            throw std::logic_error(
                                "failed to seed scripted "
                                "wake fixture");
                        }
                    });
                const bool reachedWindow = waitFor(
                    stageCv,
                    stageMutex,
                    [&]()
                    {
                        return resumeEntered;
                    });
                const bool published =
                    reachedWindow &&
                    executor.publish(
                        [&](EeThreadScheduler &scheduler,
                            IEeExecutionBackend &)
                        {
                            const auto snapshot =
                                scheduler.thread(1);
                            sawWaiting =
                                snapshot.has_value() &&
                                snapshot->state ==
                                    EeSchedulerThreadState::
                                        Waiting &&
                                snapshot->wait == wait;
                            const auto handle =
                                scheduler.threadHandle(1);
                            released =
                                handle.has_value() &&
                                scheduler
                                    .releaseWaitThread(
                                        *handle,
                                        EeSchedulerWaitCompletion::
                                            Satisfied);
                        });
                {
                    std::lock_guard<std::mutex> lock(
                        stageMutex);
                    allowBlockingExit = true;
                }
                stageCv.notify_all();
                const bool completed = waitFor(
                    stageCv,
                    stageMutex,
                    [&]()
                    {
                        return finished;
                    });
                if (completed)
                {
                    executor.invokeAtBoundary(
                        [](EeThreadScheduler &,
                           IEeExecutionBackend &)
                        {
                        });
                }
                executor.requestStop();
                executor.join();
                executor.rethrowFailure();

                t.IsTrue(
                    reachedWindow && published &&
                        sawWaiting && released &&
                        completed,
                    "a host publication queued before the "
                    "exit must run only after Blocked owns "
                    "the wait");
                t.IsTrue(
                    executor.statistics()
                            .modeledTick.raw() ==
                        15u,
                    "both scripted regions should commit "
                    "their elapsed ticks exactly once");
            });

        tc.Run(
            "backend stop exits after committing prior work",
            [](TestCase &t)
            {
                ScriptedRuntimeBackend backend;
                EeRuntimeExecutor executor(backend);
                std::mutex stageMutex;
                std::condition_variable stageCv;
                bool returnedStop = false;

                backend.plan(
                    1,
                    {
                        [&]()
                        {
                            {
                                std::lock_guard<std::mutex>
                                    lock(stageMutex);
                                returnedStop = true;
                            }
                            stageCv.notify_all();
                            return EeSchedulerRunResult{
                                EeSchedulerExitReason::
                                    StopRequested,
                                23u,
                                {},
                                {}};
                        },
                    });
                executor.start(
                    [](EeThreadScheduler &scheduler,
                       IEeExecutionBackend &selectedBackend)
                    {
                        selectedBackend.create(1, []()
                        {
                        });
                        if (!scheduler.addRunningThread(
                                1, 1u, 10))
                        {
                            throw std::logic_error(
                                "failed to seed stop "
                                "fixture");
                        }
                    });

                const bool returned = waitFor(
                    stageCv,
                    stageMutex,
                    [&]()
                    {
                        return returnedStop;
                    });
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    2s;
                while (executor.running() &&
                       std::chrono::steady_clock::now() <
                           deadline)
                {
                    std::this_thread::sleep_for(1ms);
                }
                const bool stoppedFromBackend =
                    !executor.running();
                if (!stoppedFromBackend)
                {
                    executor.requestStop();
                }
                executor.join();
                executor.rethrowFailure();

                t.IsTrue(
                    returned && stoppedFromBackend,
                    "StopRequested should close the "
                    "executor without an external stop");
                t.IsTrue(
                    executor.statistics()
                            .modeledTick.raw() ==
                            23u &&
                        backend.managedThreadCount() == 0u,
                    "the final region should commit before "
                    "cooperative continuation teardown");
            });

        tc.Run(
            "guest exception commits prior work before surfacing",
            [](TestCase &t)
            {
                ScriptedRuntimeBackend backend;
                EeRuntimeExecutor executor(backend);
                std::mutex stageMutex;
                std::condition_variable stageCv;
                bool returnedException = false;

                backend.plan(
                    1,
                    {
                        [&]()
                        {
                            {
                                std::lock_guard<std::mutex>
                                    lock(stageMutex);
                                returnedException = true;
                            }
                            stageCv.notify_all();
                            return EeSchedulerRunResult{
                                EeSchedulerExitReason::
                                    Exception,
                                31u,
                                {},
                                std::make_exception_ptr(
                                    std::runtime_error(
                                        "scripted guest "
                                        "failure"))};
                        },
                    });
                executor.start(
                    [](EeThreadScheduler &scheduler,
                       IEeExecutionBackend &selectedBackend)
                    {
                        selectedBackend.create(1, []()
                        {
                        });
                        if (!scheduler.addRunningThread(
                                1, 1u, 10))
                        {
                            throw std::logic_error(
                                "failed to seed exception "
                                "fixture");
                        }
                    });

                const bool returned = waitFor(
                    stageCv,
                    stageMutex,
                    [&]()
                    {
                        return returnedException;
                    });
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    2s;
                while (executor.running() &&
                       std::chrono::steady_clock::now() <
                           deadline)
                {
                    std::this_thread::sleep_for(1ms);
                }
                const bool stoppedFromFailure =
                    !executor.running();
                if (!stoppedFromFailure)
                {
                    executor.requestStop();
                }
                executor.join();

                bool surfacedExpectedFailure = false;
                try
                {
                    executor.rethrowFailure();
                }
                catch (const std::runtime_error &error)
                {
                    surfacedExpectedFailure =
                        std::string(error.what()) ==
                        "scripted guest failure";
                }

                const EeRuntimeExecutorStatistics stats =
                    executor.statistics();
                t.IsTrue(
                    returned && stoppedFromFailure &&
                        surfacedExpectedFailure,
                    "the guest exception should close the "
                    "executor and retain its payload");
                t.IsTrue(
                    stats.modeledTick.raw() == 31u &&
                        stats.failures == 1u &&
                        backend.managedThreadCount() == 0u,
                    "the failing region should commit "
                    "before exception propagation and "
                    "continuation teardown");
            });

        tc.Run(
            "runs every continuation on one dedicated executor",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                auto backend =
                    createEeExecutionBackend(
                        EeExecutionBackendKind::
                            LegacyCppFiber);
                EeRuntimeExecutor executor(*backend);
                const std::thread::id callerThread =
                    std::this_thread::get_id();
                std::thread::id setupThread;
                std::thread::id firstEntryThread;
                std::thread::id firstResumeThread;
                std::thread::id secondEntryThread;
                std::mutex completionMutex;
                std::condition_variable completionCv;
                int completions = 0;

                executor.start(
                    [&](EeThreadScheduler &scheduler,
                        IEeExecutionBackend &selectedBackend)
                    {
                        setupThread =
                            std::this_thread::get_id();
                        selectedBackend.create(1, [&]()
                        {
                            firstEntryThread =
                                std::this_thread::get_id();
                            selectedBackend.yieldCurrent(
                                {
                                    EeSchedulerExitReason::
                                        Yielded,
                                    8u,
                                    {},
                                    {},
                                });
                            firstResumeThread =
                                std::this_thread::get_id();
                            {
                                std::lock_guard<std::mutex>
                                    lock(completionMutex);
                                ++completions;
                            }
                            completionCv.notify_all();
                        });
                        selectedBackend.create(2, [&]()
                        {
                            secondEntryThread =
                                std::this_thread::get_id();
                            {
                                std::lock_guard<std::mutex>
                                    lock(completionMutex);
                                ++completions;
                            }
                            completionCv.notify_all();
                        });
                        if (!scheduler.addRunningThread(
                                1, 1u, 40) ||
                            !scheduler.addDormantThread(
                                2, 1u, 40) ||
                            !scheduler.startThread(2))
                        {
                            throw std::logic_error(
                                "failed to seed executor "
                                "fixture");
                        }
                    });

                const bool completed = waitFor(
                    completionCv,
                    completionMutex,
                    [&]()
                    {
                        return completions == 2;
                    });
                bool validFinalState = false;
                size_t managedAtBoundary = 99u;
                if (completed)
                {
                    executor.invokeAtBoundary(
                        [&](EeThreadScheduler &scheduler,
                            IEeExecutionBackend &
                                selectedBackend)
                        {
                            const auto first =
                                scheduler.thread(1);
                            const auto second =
                                scheduler.thread(2);
                            validFinalState =
                                first.has_value() &&
                                second.has_value() &&
                                first->state ==
                                    EeSchedulerThreadState::
                                        Dormant &&
                                second->state ==
                                    EeSchedulerThreadState::
                                        Dormant &&
                                scheduler.validate();
                            managedAtBoundary =
                                selectedBackend
                                    .managedThreadCount();
                        });
                }

                executor.requestStop();
                executor.join();
                executor.rethrowFailure();

                const EeRuntimeExecutorStatistics stats =
                    executor.statistics();
                t.IsTrue(
                    completed,
                    "both fibers should finish within the "
                    "bounded run");
                t.IsTrue(
                    setupThread != callerThread &&
                        firstEntryThread == setupThread &&
                        firstResumeThread == setupThread &&
                        secondEntryThread == setupThread &&
                        executor.executorThreadId() ==
                            std::thread::id{},
                    "setup, entry, yield resume, and peer "
                    "entry should share one dedicated host "
                    "thread");
                t.IsTrue(
                    validFinalState &&
                        managedAtBoundary == 0u &&
                        backend->managedThreadCount() == 0u,
                    "normal completion should leave dormant "
                    "scheduler records and no continuation");
                t.IsTrue(
                    stats.dispatches == 3u &&
                        stats.continuationsDestroyed == 2u &&
                        stats.boundary.boundaries >= 3u &&
                        stats.modeledTick.raw() == 8u,
                    "executor accounting should retain each "
                    "dispatch, terminal destroy, boundary, "
                    "and committed elapsed tick");
            });

        tc.Run(
            "orders a published wake after the blocking exit",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                auto backend =
                    createEeExecutionBackend(
                        EeExecutionBackendKind::
                            LegacyCppFiber);
                EeRuntimeExecutor executor(*backend);
                const EeSchedulerWaitKey wait{
                    EeSchedulerWaitKind::Semaphore,
                    7};
                std::mutex stageMutex;
                std::condition_variable stageCv;
                bool entered = false;
                bool allowBlock = false;
                bool resumed = false;
                bool sawCommittedWait = false;
                bool released = false;

                executor.start(
                    [&](EeThreadScheduler &scheduler,
                        IEeExecutionBackend &selectedBackend)
                    {
                        selectedBackend.create(1, [&]()
                        {
                            {
                                std::unique_lock<std::mutex>
                                    lock(stageMutex);
                                entered = true;
                                stageCv.notify_all();
                                stageCv.wait(
                                    lock,
                                    [&]()
                                    {
                                        return allowBlock;
                                    });
                            }

                            selectedBackend.yieldCurrent(
                                {
                                    EeSchedulerExitReason::
                                        Blocked,
                                    16u,
                                    wait,
                                    {},
                                });
                            {
                                std::lock_guard<std::mutex>
                                    lock(stageMutex);
                                resumed = true;
                            }
                            stageCv.notify_all();
                        });
                        if (!scheduler.addRunningThread(
                                1, 1u, 20))
                        {
                            throw std::logic_error(
                                "failed to seed blocking "
                                "executor fixture");
                        }
                    });

                const bool reachedWindow = waitFor(
                    stageCv,
                    stageMutex,
                    [&]()
                    {
                        return entered;
                    });
                const bool published =
                    reachedWindow &&
                    executor.publish(
                        [&](EeThreadScheduler &scheduler,
                            IEeExecutionBackend &)
                        {
                            const auto snapshot =
                                scheduler.thread(1);
                            sawCommittedWait =
                                snapshot.has_value() &&
                                snapshot->state ==
                                    EeSchedulerThreadState::
                                        Waiting &&
                                snapshot->wait == wait;
                            const auto handle =
                                scheduler.threadHandle(1);
                            released =
                                handle.has_value() &&
                                scheduler
                                    .releaseWaitThread(
                                        *handle,
                                        EeSchedulerWaitCompletion::
                                            Satisfied);
                        });
                {
                    std::lock_guard<std::mutex> lock(
                        stageMutex);
                    allowBlock = true;
                }
                stageCv.notify_all();

                const bool resumedAfterWake = waitFor(
                    stageCv,
                    stageMutex,
                    [&]()
                    {
                        return resumed;
                    });
                if (resumedAfterWake)
                {
                    executor.invokeAtBoundary(
                        [](EeThreadScheduler &,
                           IEeExecutionBackend &)
                        {
                        });
                }
                executor.requestStop();
                executor.join();
                executor.rethrowFailure();

                const EeRuntimeExecutorStatistics stats =
                    executor.statistics();
                t.IsTrue(
                    reachedWindow && published,
                    "the host should publish while the guest "
                    "is still running before its park");
                t.IsTrue(
                    sawCommittedWait && released &&
                        resumedAfterWake,
                    "the queued wake must observe WAIT, "
                    "release it, and resume the same fiber");
                t.IsTrue(
                    stats.publicationsQueued == 2u &&
                        stats.publicationsApplied == 2u &&
                        stats.boundary.consequences >= 2u,
                    "both the wake and synchronization "
                    "commands should pass through the "
                    "asynchronous boundary stage");
            });

        tc.Run(
            "shutdown unwinds suspended and never-run fibers then resets",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct UnwindMarker
                {
                    std::atomic<int> &count;

                    ~UnwindMarker()
                    {
                        count.fetch_add(
                            1, std::memory_order_release);
                    }
                };

                auto backend =
                    createEeExecutionBackend(
                        EeExecutionBackendKind::
                            LegacyCppFiber);
                EeRuntimeExecutor executor(*backend);
                const EeSchedulerWaitKey wait{
                    EeSchedulerWaitKind::Sleep,
                    0};
                std::atomic<int> unwindCount{0};
                std::atomic<bool> neverRunEntered{false};
                bool observedWaiting = false;

                executor.start(
                    [&](EeThreadScheduler &scheduler,
                        IEeExecutionBackend &selectedBackend)
                    {
                        selectedBackend.create(10, [&]()
                        {
                            UnwindMarker marker{unwindCount};
                            selectedBackend.yieldCurrent(
                                {
                                    EeSchedulerExitReason::
                                        Blocked,
                                    4u,
                                    wait,
                                    {},
                                });
                        });
                        selectedBackend.create(11, [&]()
                        {
                            neverRunEntered.store(
                                true,
                                std::memory_order_release);
                        });
                        if (!scheduler.addRunningThread(
                                10, 1u, 30) ||
                            !scheduler.addDormantThread(
                                11, 1u, 30))
                        {
                            throw std::logic_error(
                                "failed to seed shutdown "
                                "executor fixture");
                        }
                    });
                executor.invokeAtBoundary(
                    [&](EeThreadScheduler &scheduler,
                        IEeExecutionBackend &)
                    {
                        const auto snapshot =
                            scheduler.thread(10);
                        observedWaiting =
                            snapshot.has_value() &&
                            snapshot->state ==
                                EeSchedulerThreadState::
                                    Waiting &&
                            snapshot->wait == wait;
                    });
                executor.requestStop();
                executor.join();
                executor.rethrowFailure();

                t.IsTrue(
                    observedWaiting &&
                        unwindCount.load(
                            std::memory_order_acquire) == 1 &&
                        !neverRunEntered.load(
                            std::memory_order_acquire) &&
                        backend->managedThreadCount() == 0u,
                    "stop should force-unwind the suspended "
                    "frame and destroy an untouched fiber "
                    "without entering it");

                std::mutex completionMutex;
                std::condition_variable completionCv;
                bool replacementFinished = false;
                std::thread::id replacementSetupThread;
                std::thread::id replacementEntryThread;
                executor.start(
                    [&](EeThreadScheduler &scheduler,
                        IEeExecutionBackend &selectedBackend)
                    {
                        replacementSetupThread =
                            std::this_thread::get_id();
                        selectedBackend.create(12, [&]()
                        {
                            replacementEntryThread =
                                std::this_thread::get_id();
                            {
                                std::lock_guard<std::mutex>
                                    lock(completionMutex);
                                replacementFinished = true;
                            }
                            completionCv.notify_all();
                        });
                        if (!scheduler.addRunningThread(
                                12, 2u, 30))
                        {
                            throw std::logic_error(
                                "failed to seed replacement "
                                "executor fixture");
                        }
                    });
                const bool replacementCompleted = waitFor(
                    completionCv,
                    completionMutex,
                    [&]()
                    {
                        return replacementFinished;
                    });
                if (replacementCompleted)
                {
                    executor.invokeAtBoundary(
                        [](EeThreadScheduler &,
                           IEeExecutionBackend &)
                        {
                        });
                }
                executor.requestStop();
                executor.join();
                executor.rethrowFailure();

                t.IsTrue(
                    replacementCompleted &&
                        replacementEntryThread ==
                            replacementSetupThread &&
                        backend->managedThreadCount() == 0u &&
                        executor.statistics().starts == 2u,
                    "a complete teardown should let a fresh "
                    "executor generation bind and finish");
            });
    });
}
