#include "MiniTest.h"

#include "runtime/boost_ee_fiber.h"
#include "runtime/ee_execution_backend.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef PS2X_TEST_EE_FIBER_GUARD_PROBE
#define PS2X_TEST_EE_FIBER_GUARD_PROBE ""
#endif

namespace
{
#if defined(_MSC_VER)
#define PS2X_TEST_NOINLINE __declspec(noinline)
#else
#define PS2X_TEST_NOINLINE __attribute__((noinline))
#endif

    volatile uint64_t g_fiberStackSink = 0u;

    PS2X_TEST_NOINLINE void
    consumeFiberStackAndYield(size_t depth)
    {
        volatile unsigned char storage[4096];
        for (size_t offset = 0u;
             offset < sizeof(storage);
             offset += 256u)
        {
            storage[offset] =
                static_cast<unsigned char>(
                    depth + offset);
        }

        if (depth == 0u)
        {
            BoostEeFiber::yieldCurrent();
        }
        else
        {
            consumeFiberStackAndYield(depth - 1u);
        }

        g_fiberStackSink += storage[depth & 0xffu];
    }

    void nestedYield(
        std::vector<int> &stages,
        int depth)
    {
        stages.push_back(depth);
        if (depth == 0)
        {
            BoostEeFiber::yieldCurrent();
        }
        else
        {
            nestedYield(stages, depth - 1);
        }
        stages.push_back(-depth);
    }

    bool guardProbeFaulted()
    {
        const char *const path =
            PS2X_TEST_EE_FIBER_GUARD_PROBE;
        if (path[0] == '\0')
        {
            return false;
        }

#if defined(_WIN32)
        std::string command =
            std::string("\"") + path + "\"";
        std::vector<char> mutableCommand(
            command.begin(),
            command.end());
        mutableCommand.push_back('\0');

        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION process{};
        if (::CreateProcessA(
                nullptr,
                mutableCommand.data(),
                nullptr,
                nullptr,
                FALSE,
                CREATE_NO_WINDOW,
                nullptr,
                nullptr,
                &startup,
                &process) == FALSE)
        {
            return false;
        }

        const DWORD wait =
            ::WaitForSingleObject(process.hProcess, 5000u);
        if (wait != WAIT_OBJECT_0)
        {
            ::TerminateProcess(process.hProcess, 0xdeadu);
            ::WaitForSingleObject(process.hProcess, 1000u);
            ::CloseHandle(process.hThread);
            ::CloseHandle(process.hProcess);
            return false;
        }

        DWORD exitCode = 0u;
        const bool queried =
            ::GetExitCodeProcess(
                process.hProcess,
                &exitCode) != FALSE;
        ::CloseHandle(process.hThread);
        ::CloseHandle(process.hProcess);
        return queried &&
               (exitCode == 0xc00000fdu ||
                exitCode == 0xc0000005u);
#else
        const pid_t child = ::fork();
        if (child < 0)
        {
            return false;
        }
        if (child == 0)
        {
            ::execl(path, path, nullptr);
            std::_Exit(127);
        }

        int status = 0;
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() <
               deadline)
        {
            const pid_t result =
                ::waitpid(child, &status, WNOHANG);
            if (result == child)
            {
                return WIFSIGNALED(status) &&
                       (WTERMSIG(status) == SIGSEGV ||
                        WTERMSIG(status) == SIGBUS);
            }
            if (result < 0 && errno != EINTR)
            {
                return false;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(10));
        }

        ::kill(child, SIGKILL);
        ::waitpid(child, &status, 0);
        return false;
#endif
    }
}

void register_boost_ee_fiber_tests()
{
    MiniTest::Case("BoostEeFiber", [](TestCase &tc)
    {
        tc.Run(
            "availability rejects construction without production fcontext",
            [](TestCase &t)
            {
                if (BoostEeFiber::available())
                {
                    const EeExecutionBackendBuildInfo build =
                        eeExecutionBackendBuildInfo();
                    t.IsTrue(
                        build.boostContextFcontextAvailable,
                        "an available wrapper should match the build diagnostic");
                    return;
                }

                bool rejected = false;
                try
                {
                    BoostEeFiber fiber([] {});
                }
                catch (const std::logic_error &)
                {
                    rejected = true;
                }
                t.IsTrue(
                    rejected,
                    "the explicit fallback must reject Boost fiber construction");
            });

        tc.Run(
            "native stack ownership distinguishes fiber frames from external storage",
            [](TestCase &t)
            {
                unsigned char executorFrame = 0u;
                t.IsFalse(
                    BoostEeFiber::
                        currentStackContainsAddressForTesting(
                            &executorFrame),
                    "the executor stack must not be reported as a running fiber stack");

                if (!BoostEeFiber::available())
                {
                    return;
                }

                std::vector<unsigned char>
                    externalStorage(256u, 0u);
                bool nativeFrameInsideBeforeYield = false;
                bool nativeFrameInsideAfterYield = false;
                bool externalStorageOutsideBeforeYield =
                    false;
                bool externalStorageOutsideAfterYield =
                    false;
                bool nativeFramePreserved = false;

                BoostEeFiber fiber([&]()
                {
                    unsigned char nativeFrame[256u]{};
                    nativeFrame[0u] = 0x3cu;
                    nativeFrame[255u] = 0xc3u;

                    nativeFrameInsideBeforeYield =
                        BoostEeFiber::
                            currentStackContainsAddressForTesting(
                                nativeFrame);
                    externalStorageOutsideBeforeYield =
                        !BoostEeFiber::
                            currentStackContainsAddressForTesting(
                                externalStorage.data());

                    BoostEeFiber::yieldCurrent();

                    nativeFrameInsideAfterYield =
                        BoostEeFiber::
                            currentStackContainsAddressForTesting(
                                nativeFrame);
                    externalStorageOutsideAfterYield =
                        !BoostEeFiber::
                            currentStackContainsAddressForTesting(
                                externalStorage.data());
                    nativeFramePreserved =
                        nativeFrame[0u] == 0x3cu &&
                        nativeFrame[255u] == 0xc3u;
                });

                fiber.resume();
                t.Equals(
                    fiber.state(),
                    BoostEeFiber::State::Suspended,
                    "the native frame should remain suspended on its protected stack");
                t.IsFalse(
                    BoostEeFiber::
                        currentStackContainsAddressForTesting(
                            externalStorage.data()),
                    "the executor must report no current fiber while the continuation is suspended");

                fiber.resume();
                t.Equals(
                    fiber.state(),
                    BoostEeFiber::State::Finished,
                    "the ownership probe continuation should finish");
                t.IsTrue(
                    nativeFrameInsideBeforeYield &&
                        nativeFrameInsideAfterYield,
                    "a materialized native frame should stay in the fiber mapping across a yield");
                t.IsTrue(
                    externalStorageOutsideBeforeYield &&
                        externalStorageOutsideAfterYield,
                    "external storage must stay outside the fiber mapping");
                t.IsTrue(
                    nativeFramePreserved,
                    "the protected native frame should retain its contents across a yield");
            });

        tc.Run(
            "continuation moves survive repeated and nested yields",
            [](TestCase &t)
            {
                if (!BoostEeFiber::available())
                {
                    return;
                }

                constexpr size_t YieldCount = 4096u;
                size_t guestYields = 0u;
                BoostEeFiber stress([&]()
                {
                    for (size_t index = 0u;
                         index < YieldCount;
                         ++index)
                    {
                        ++guestYields;
                        BoostEeFiber::yieldCurrent();
                    }
                });

                t.Equals(
                    stress.state(),
                    BoostEeFiber::State::Created,
                    "fiber construction must not run its entry");
                for (size_t index = 0u;
                     index < YieldCount;
                     ++index)
                {
                    stress.resume();
                    t.Equals(
                        stress.state(),
                        BoostEeFiber::State::Suspended,
                        "every returned continuation should remain resumable");
                }
                stress.resume();
                t.Equals(
                    guestYields,
                    YieldCount,
                    "every move-only continuation should resume exactly once");
                t.Equals(
                    stress.state(),
                    BoostEeFiber::State::Finished,
                    "the final continuation should return normally");

                std::vector<int> stages;
                BoostEeFiber nested([&]()
                {
                    nestedYield(stages, 3);
                });
                nested.resume();
                t.Equals(
                    stages,
                    std::vector<int>({3, 2, 1, 0}),
                    "a yield should preserve all nested native frames");
                nested.resume();
                t.Equals(
                    stages,
                    std::vector<int>(
                        {3, 2, 1, 0, 0, -1, -2, -3}),
                    "resumption should unwind through the preserved nested frames");
                t.Equals(
                    nested.state(),
                    BoostEeFiber::State::Finished,
                    "the nested fiber should complete");
            });

        tc.Run(
            "regular exceptions are captured on the executor",
            [](TestCase &t)
            {
                if (!BoostEeFiber::available())
                {
                    return;
                }

                int destructorCount = 0;
                BoostEeFiber fiber([&]()
                {
                    struct Scope
                    {
                        int &count;
                        ~Scope()
                        {
                            ++count;
                        }
                    } scope{destructorCount};
                    throw std::runtime_error(
                        "guest fiber failure");
                });

                fiber.resume();
                t.Equals(
                    fiber.state(),
                    BoostEeFiber::State::Failed,
                    "a C++ exception should become a failed executor result");
                t.Equals(
                    destructorCount,
                    1,
                    "ordinary exception unwinding should run guest destructors");
                t.IsTrue(
                    static_cast<bool>(fiber.failure()),
                    "the executor should receive the exception pointer");

                std::string message;
                try
                {
                    fiber.rethrowFailure();
                }
                catch (const std::runtime_error &error)
                {
                    message = error.what();
                }
                t.Equals(
                    message,
                    std::string("guest fiber failure"),
                    "the recorded exception should retain its payload");
            });

        tc.Run(
            "destroying a suspended fiber forces one clean unwind",
            [](TestCase &t)
            {
                if (!BoostEeFiber::available())
                {
                    return;
                }

                int destructorCount = 0;
                bool passedYield = false;
                BoostEeFiber fiber([&]()
                {
                    struct Scope
                    {
                        int &count;
                        ~Scope()
                        {
                            ++count;
                        }
                    } scope{destructorCount};
                    BoostEeFiber::yieldCurrent();
                    passedYield = true;
                });

                fiber.resume();
                t.Equals(
                    destructorCount,
                    0,
                    "a suspended native frame should remain alive");
                fiber.destroy();
                fiber.destroy();
                t.Equals(
                    fiber.state(),
                    BoostEeFiber::State::Destroyed,
                    "destroy should be idempotent on the executor");
                t.Equals(
                    destructorCount,
                    1,
                    "forced unwind should run each destructor once");
                t.IsFalse(
                    passedYield,
                    "forced unwind must not continue after the yield");

                const BoostEeFiber::StackUsage usage =
                    fiber.stackUsage();
                t.Equals(
                    usage.allocationCount,
                    size_t{1u},
                    "one fiber should allocate one protected stack");
                t.Equals(
                    usage.deallocationCount,
                    size_t{1u},
                    "forced unwind should release that stack once");
                t.Equals(
                    usage.activeAllocationCount,
                    size_t{0u},
                    "destroy should leave no live stack allocation");
            });

        tc.Run(
            "executor affinity forbids cross-thread and guest-to-guest switches",
            [](TestCase &t)
            {
                if (!BoostEeFiber::available())
                {
                    return;
                }

                bool targetRan = false;
                BoostEeFiber target([&]()
                {
                    targetRan = true;
                });
                bool directSwitchRejected = false;
                BoostEeFiber source([&]()
                {
                    try
                    {
                        target.resume();
                    }
                    catch (const std::logic_error &)
                    {
                        directSwitchRejected = true;
                    }
                    BoostEeFiber::yieldCurrent();
                });

                source.resume();
                t.IsTrue(
                    directSwitchRejected,
                    "a guest fiber must return through its executor before another guest runs");
                t.IsFalse(
                    targetRan,
                    "the rejected direct switch must not enter its target");
                source.destroy();

                std::atomic<int> rejected{0};
                std::thread wrongThread([&]()
                {
                    try
                    {
                        target.resume();
                    }
                    catch (const std::logic_error &)
                    {
                        rejected.fetch_add(
                            1,
                            std::memory_order_relaxed);
                    }
                    try
                    {
                        target.destroy();
                    }
                    catch (const std::logic_error &)
                    {
                        rejected.fetch_add(
                            1,
                            std::memory_order_relaxed);
                    }
                });
                wrongThread.join();
                t.Equals(
                    rejected.load(std::memory_order_relaxed),
                    2,
                    "resume and destroy should both reject a non-owner host thread");

                bool outsideYieldRejected = false;
                try
                {
                    BoostEeFiber::yieldCurrent();
                }
                catch (const std::logic_error &)
                {
                    outsideYieldRejected = true;
                }
                t.IsTrue(
                    outsideYieldRejected,
                    "the executor stack must not masquerade as a guest fiber");

                target.resume();
                t.IsTrue(
                    targetRan,
                    "the constructing executor should still run the untouched target");
            });

        tc.Run(
            "protected stack accounting and setup failures are deterministic",
            [](TestCase &t)
            {
                if (!BoostEeFiber::available())
                {
                    return;
                }

                bool allocationFailed = false;
                try
                {
                    BoostEeFiber::Options options;
                    options.failureForTesting =
                        BoostEeFiber::StackFailureForTesting::
                            Allocation;
                    BoostEeFiber fiber([] {}, options);
                }
                catch (const std::bad_alloc &)
                {
                    allocationFailed = true;
                }
                t.IsTrue(
                    allocationFailed,
                    "injected allocation failure should report bad_alloc");

                bool guardFailed = false;
                try
                {
                    BoostEeFiber::Options options;
                    options.failureForTesting =
                        BoostEeFiber::StackFailureForTesting::
                            GuardSetup;
                    BoostEeFiber fiber([] {}, options);
                }
                catch (const std::runtime_error &error)
                {
                    guardFailed =
                        std::string(error.what()).find(
                            "guard setup failed") !=
                        std::string::npos;
                }
                t.IsTrue(
                    guardFailed,
                    "injected guard failure should unwind allocation and report an error");

                BoostEeFiber fiber([]()
                {
                    consumeFiberStackAndYield(16u);
                });
                const BoostEeFiber::StackUsage initial =
                    fiber.stackUsage();
                t.Equals(
                    initial.requestedUsableBytes,
                    BoostEeFiber::DefaultUsableStackBytes,
                    "the initial usable stack request should be one MiB");
                t.IsTrue(
                    initial.usableBytes >=
                            BoostEeFiber::
                                DefaultUsableStackBytes &&
                        initial.guardBytes != 0u &&
                        initial.committedBytes ==
                            initial.usableBytes +
                                initial.guardBytes,
                    "the protected mapping should contain one usable stack plus one guard page");
                t.Equals(
                    initial.allocationCount,
                    size_t{1u},
                    "construction should allocate exactly once");
                t.Equals(
                    initial.activeAllocationCount,
                    size_t{1u},
                    "the suspended initial continuation should retain its mapping");

                fiber.resume();
                fiber.sampleStackUsage();
                const BoostEeFiber::StackUsage suspended =
                    fiber.stackUsage();
                t.IsTrue(
                    suspended.highWaterBytes >=
                        16u * 4096u,
                    "explicit sampling should observe the retained deep native frames");
                t.IsTrue(
                    suspended.peakCommittedBytes ==
                            initial.committedBytes &&
                        suspended.peakResidentBytes != 0u &&
                        suspended.peakResidentBytes <=
                            initial.committedBytes,
                    "accounting should retain committed and resident peaks");

                fiber.resume();
                const BoostEeFiber::StackUsage finished =
                    fiber.stackUsage();
                t.Equals(
                    fiber.state(),
                    BoostEeFiber::State::Finished,
                    "the sampled fiber should finish normally");
                t.Equals(
                    finished.deallocationCount,
                    size_t{1u},
                    "normal return should deallocate on the executor");
                t.Equals(
                    finished.committedBytes,
                    size_t{0u},
                    "normal return should leave no committed stack");
                t.Equals(
                    finished.residentBytes,
                    size_t{0u},
                    "normal return should leave no resident stack");
            });

        tc.Run(
            "protected stack guard faults in a bounded subprocess",
            [](TestCase &t)
            {
                if (!BoostEeFiber::available())
                {
                    return;
                }
                t.IsTrue(
                    guardProbeFaulted(),
                    "an overflowing fiber should fault at its protected guard instead of corrupting the executor");
            });
    });
}
