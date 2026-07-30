#include "MiniTest.h"

#include "runtime/ee_execution_backend.h"
#include "runtime/ee_thread_scheduler.h"

#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ps2x::ee;

void register_ee_execution_backend_tests()
{
    MiniTest::Case("EeExecutionBackend", [](TestCase &tc)
    {
        tc.Run(
            "backend names parse without changing invalid selections",
            [](TestCase &t)
            {
                EeExecutionBackendKind parsed =
                    EeExecutionBackendKind::
                        LegacyHostThread;
                t.IsTrue(
                    parseEeExecutionBackendKind(
                        "legacy-cpp-fiber", parsed) &&
                        parsed ==
                            EeExecutionBackendKind::
                                LegacyCppFiber,
                    "the fiber artifact name should select the fiber backend");
                t.IsTrue(
                    parseEeExecutionBackendKind(
                        "legacy-host-thread", parsed) &&
                        parsed ==
                            EeExecutionBackendKind::
                                LegacyHostThread,
                    "the host-thread artifact name should select the compatibility backend");
                t.IsFalse(
                    parseEeExecutionBackendKind(
                        "fiber", parsed),
                    "an abbreviated or unknown backend name should be rejected");
                t.Equals(
                    parsed,
                    EeExecutionBackendKind::
                        LegacyHostThread,
                    "a rejected name should preserve the caller's selection");
            });

        tc.Run(
            "fiber selection is explicit and never silently falls back",
            [](TestCase &t)
            {
                const EeExecutionBackendBuildInfo build =
                    eeExecutionBackendBuildInfo();
                const std::string diagnostics =
                    eeExecutionBackendDiagnostics(
                        EeExecutionBackendKind::
                            LegacyCppFiber);
                t.IsTrue(
                    diagnostics.find(
                        "selected=legacy-cpp-fiber") !=
                        std::string::npos,
                    "diagnostics should retain the requested mode");

                auto hostBackend =
                    createEeExecutionBackend(
                        EeExecutionBackendKind::
                            LegacyHostThread);
                t.Equals(
                    hostBackend->checkpointMode(),
                    EeExecutionCheckpointMode::
                        DispatcherExit,
                    "the host-thread backend should exit generated code at a checkpoint");

                if (!build.boostContextFcontextAvailable)
                {
                    bool rejected = false;
                    try
                    {
                        auto backend =
                            createEeExecutionBackend(
                                EeExecutionBackendKind::
                                    LegacyCppFiber);
                    }
                    catch (const std::runtime_error &)
                    {
                        rejected = true;
                    }
                    t.IsTrue(
                        rejected,
                        "an unavailable fiber mode must fail instead of selecting host threads");
                    return;
                }

                auto backend =
                    createEeExecutionBackend(
                        EeExecutionBackendKind::
                            LegacyCppFiber);
                t.Equals(
                    backend->kind(),
                    EeExecutionBackendKind::
                        LegacyCppFiber,
                    "the factory should retain the requested backend kind");
                t.Equals(
                    std::string(backend->name()),
                    std::string("legacy-cpp-fiber"),
                    "the selected mode should have a stable artifact name");
                t.Equals(
                    backend->checkpointMode(),
                    EeExecutionCheckpointMode::
                        SuspendContinuation,
                    "the stackful backend should suspend and resume inside a checkpoint");
                t.Equals(
                    backend->managedThreadCount(),
                    size_t{0u},
                    "a new fiber backend should own no continuations");
            });

        tc.Run(
            "fiber exits drive the deterministic scheduler interface",
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
                const EeSchedulerWaitKey wait{
                    EeSchedulerWaitKind::Semaphore,
                    7};
                std::vector<std::string> stages;

                backend->create(1, [&]()
                {
                    stages.push_back("one-enter");
                    backend->yieldCurrent(
                        {
                            EeSchedulerExitReason::Yielded,
                            8u,
                            {},
                            {},
                        });
                    stages.push_back("one-resume");
                    backend->yieldCurrent(
                        {
                            EeSchedulerExitReason::Blocked,
                            16u,
                            wait,
                            {},
                        });
                    stages.push_back("one-finish");
                });
                backend->create(2, [&]()
                {
                    stages.push_back("two-finish");
                });

                t.IsTrue(
                    stages.empty(),
                    "fiber creation must not enter guest code");
                t.Equals(
                    backend->managedThreadCount(),
                    size_t{2u},
                    "one continuation should be owned per guest thread");

                EeThreadScheduler scheduler;
                t.IsTrue(
                    scheduler.addRunningThread(1, 1u, 40) &&
                        scheduler.addDormantThread(
                            2, 1u, 40) &&
                        scheduler.startThread(2),
                    "the fixture should establish two equal-priority threads");

                const auto first =
                    scheduler.dispatchOne(*backend, 64u);
                const auto second =
                    scheduler.dispatchOne(*backend, 64u);
                const auto third =
                    scheduler.dispatchOne(*backend, 64u);
                t.IsTrue(
                    first.has_value() &&
                        first->resumedThreadId == 1 &&
                        first->result.reason ==
                            EeSchedulerExitReason::Yielded &&
                        first->result.elapsedTicks == 8u &&
                        first->selectedThreadId == 2,
                    "a yielded fiber should return through the scheduler and rotate FIFO");
                t.IsTrue(
                    second.has_value() &&
                        second->resumedThreadId == 2 &&
                        second->result.reason ==
                            EeSchedulerExitReason::Finished &&
                        second->selectedThreadId == 1,
                    "normal fiber return should finish through the scheduler");
                t.IsTrue(
                    third.has_value() &&
                        third->resumedThreadId == 1 &&
                        third->result.reason ==
                            EeSchedulerExitReason::Blocked &&
                        third->result.elapsedTicks == 16u &&
                        third->result.wait == wait &&
                        !third->selectedThreadId.has_value(),
                    "a blocking exit should transfer exact wait ownership to the scheduler");
                t.IsTrue(
                    stages ==
                        std::vector<std::string>(
                            {
                                "one-enter",
                                "two-finish",
                                "one-resume",
                            }),
                    "the executor should never switch directly between guest fibers");

                const auto firstHandle =
                    scheduler.threadHandle(1);
                t.IsTrue(
                    firstHandle.has_value() &&
                        scheduler.releaseWaitThread(
                            *firstHandle,
                            EeSchedulerWaitCompletion::
                                Satisfied),
                    "the fixture should publish one scheduler-owned wait completion");
                const auto fourth =
                    scheduler.dispatchOne(*backend, 64u);
                t.IsTrue(
                    fourth.has_value() &&
                        fourth->resumedThreadId == 1 &&
                        fourth->result.reason ==
                            EeSchedulerExitReason::Finished &&
                        !fourth->selectedThreadId.has_value(),
                    "the released fiber should resume at the suspended call and finish");
                t.IsTrue(
                    stages ==
                        std::vector<std::string>(
                            {
                                "one-enter",
                                "two-finish",
                                "one-resume",
                                "one-finish",
                            }) &&
                        scheduler.validate(),
                    "fiber execution should preserve deterministic scheduler state");

                backend->destroy(1);
                backend->destroy(2);
                t.Equals(
                    backend->managedThreadCount(),
                    size_t{0u},
                    "executor destruction should release every completed continuation");
            });

        tc.Run(
            "fiber exceptions and affinity return to the executor",
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
                backend->create(3, []()
                {
                    throw std::runtime_error(
                        "backend guest failure");
                });

                EeThreadScheduler scheduler;
                t.IsTrue(
                    scheduler.addRunningThread(3, 1u, 20),
                    "the failing fixture should own the scheduler");
                const auto failed =
                    scheduler.dispatchOne(*backend, 32u);
                std::string failureMessage;
                if (failed.has_value() &&
                    failed->result.failure)
                {
                    try
                    {
                        std::rethrow_exception(
                            failed->result.failure);
                    }
                    catch (const std::runtime_error &error)
                    {
                        failureMessage = error.what();
                    }
                }
                t.IsTrue(
                    failed.has_value() &&
                        failed->result.reason ==
                            EeSchedulerExitReason::Exception &&
                        failureMessage ==
                            "backend guest failure" &&
                        backend->isFinished(3),
                    "ordinary exceptions should become terminal executor results");
                backend->destroy(3);

                bool targetRan = false;
                bool directResumeRejected = false;
                backend->create(4, [&]()
                {
                    targetRan = true;
                });
                backend->create(5, [&]()
                {
                    try
                    {
                        static_cast<void>(
                            backend->resume(4, 1u));
                    }
                    catch (const std::logic_error &)
                    {
                        directResumeRejected = true;
                    }
                    backend->yieldCurrent(
                        {
                            EeSchedulerExitReason::Yielded,
                            1u,
                            {},
                            {},
                        });
                });

                const auto sourceExit =
                    backend->resume(5, 8u);
                t.IsTrue(
                    sourceExit.reason ==
                            EeSchedulerExitReason::Yielded &&
                        directResumeRejected &&
                        !targetRan,
                    "a guest must return to its executor before another guest resumes");

                std::atomic<bool> wrongThreadRejected{
                    false};
                std::thread wrongThread([&]()
                {
                    try
                    {
                        static_cast<void>(
                            backend->resume(4, 8u));
                    }
                    catch (const std::logic_error &)
                    {
                        wrongThreadRejected.store(
                            true,
                            std::memory_order_release);
                    }
                });
                wrongThread.join();
                t.IsTrue(
                    wrongThreadRejected.load(
                        std::memory_order_acquire),
                    "resume should reject a host thread other than the executor");

                const auto targetExit =
                    backend->resume(4, 8u);
                t.IsTrue(
                    targetExit.reason ==
                            EeSchedulerExitReason::Finished &&
                        targetRan,
                    "the executor should still enter the untouched target");
                backend->destroy(4);
                backend->destroy(5);

                backend->joinAll();
                std::atomic<bool> rebound{false};
                std::atomic<bool> reboundFinished{false};
                std::thread replacementExecutor([&]()
                {
                    backend->create(6, [&]()
                    {
                        rebound.store(
                            true,
                            std::memory_order_release);
                    });
                    const auto result =
                        backend->resume(6, 8u);
                    reboundFinished.store(
                        result.reason ==
                            EeSchedulerExitReason::Finished,
                        std::memory_order_release);
                    backend->destroy(6);
                    backend->joinAll();
                });
                replacementExecutor.join();
                t.IsTrue(
                    rebound.load(
                        std::memory_order_acquire) &&
                        reboundFinished.load(
                            std::memory_order_acquire) &&
                        backend->managedThreadCount() ==
                            0u,
                    "an empty backend should bind a replacement executor after reset");
            });
    });
}
