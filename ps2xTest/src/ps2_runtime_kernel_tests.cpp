#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2_syscalls.h"
#include "ps2_stubs.h"

#include <chrono>
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

// g_currentThreadId is an `inline thread_local int` defined in the kernel's
// internal ThreadRuntimeState.h (default 1).
// The current-thread authority test deliberately corrupts this compatibility
// slot to prove that runtime context binding detects and repairs divergence.
//
// ODR-safety: this declaration MUST stay byte-for-byte type-compatible with that
// definition (`thread_local int`, same name, no namespace). It is an `extern`
// declaration of an existing inline thread_local, NOT a second definition, so the
// linker binds to the runtime's instance. If the runtime ever changes the type or
// moves it into a namespace, update this line in lockstep or the build will break.
extern thread_local int g_currentThreadId;

using namespace ps2_syscalls;

namespace
{
    constexpr uint32_t K_PARAM_ADDR = 0x1000u;
    constexpr uint32_t K_STATUS_ADDR = 0x1400u;

    constexpr int KE_OK = 0;
    constexpr int KE_ERROR = -1;
    constexpr int KE_ILLEGAL_THID = -406;
    constexpr int KE_UNKNOWN_THID = -407;
    constexpr int KE_UNKNOWN_SEMID = -408;
    constexpr int KE_DORMANT = -413;
    constexpr int KE_SEMA_ZERO = -419;
    constexpr int KE_SEMA_OVF = -420;
    constexpr uint32_t K_SEMA_WAIT_READY_ADDR = 0x1900u;

    constexpr int THS_WAIT = 0x04;
    constexpr int THS_READY = 0x02;
    constexpr int THS_SUSPEND = 0x08;
    constexpr int THS_WAITSUSPEND = 0x0C;
    constexpr int THS_DORMANT = 0x10;
    constexpr uint32_t TSW_SLEEP = 1u;
    constexpr uint32_t TSW_SEMA = 2u;
    constexpr uint32_t TSW_EVENT = 3u;

    constexpr uint32_t K_EVENT_WAIT_READY_ADDR = 0x1800u;
    constexpr uint32_t K_EVENT_WAIT_GATE_ADDR = 0x1804u;
    constexpr uint32_t K_TERMINATE_SEMA_WAIT_READY_ADDR = 0x1810u;
    constexpr uint32_t K_SLEEP_GATE_ADDR = 0x1820u;
    constexpr uint32_t K_SLEEP_STAGE_ADDR = 0x1824u;
    constexpr uint32_t K_SLEEP_RETURN_ADDR = 0x1830u;
    constexpr uint32_t K_WAITSUSPEND_STAGE_ADDR = 0x1840u;
    constexpr uint32_t K_WAITSUSPEND_RETURN_ADDR = 0x1844u;
    constexpr uint32_t K_ID_REUSE_STAGE_ADDR = 0x1850u;
    constexpr uint32_t K_ID_REUSE_RETURN_ADDR = 0x1854u;
    constexpr uint32_t K_SEMA_ORACLE_STAGE_ADDR = 0x1860u;
    constexpr uint32_t K_SEMA_ORACLE_RETURN_ADDR = 0x1864u;
    constexpr uint32_t K_TERMINATE_CANDIDATE_STAGE_ADDR = 0x1870u;
    constexpr uint32_t K_TERMINATE_TARGET_STAGE_ADDR = 0x1874u;
    constexpr uint32_t K_EXIT_THREAD_STAGE_ADDR = 0x1880u;
    constexpr uint32_t K_EXIT_DELETE_THREAD_STAGE_ADDR = 0x1884u;
    constexpr uint32_t K_EXIT_THREAD_ID_ADDR = 0x1888u;
    constexpr uint32_t K_EXIT_DELETE_THREAD_ID_ADDR = 0x188Cu;
    constexpr uint32_t K_CONTROL_EVENT_PARAM_ADDR = 0x1890u;
    constexpr uint32_t K_CONTROL_SEMA_PARAM_ADDR = 0x18A0u;
    constexpr uint32_t K_RESCHEDULE_STAGE_ADDR = 0x18C0u;
    constexpr uint32_t K_CURRENT_THREAD_ID_ADDR = 0x18D0u;

    struct EeThreadStatus
    {
        int32_t status;
        uint32_t func;
        uint32_t stack;
        int32_t stack_size;
        uint32_t gp_reg;
        int32_t initial_priority;
        int32_t current_priority;
        uint32_t attr;
        uint32_t option;
        uint32_t waitType;
        uint32_t waitId;
        uint32_t wakeupCount;
    };

    struct EeSemaStatus
    {
        int32_t count;
        int32_t max_count;
        int32_t init_count;
        int32_t wait_threads;
        uint32_t attr;
        uint32_t option;
    };

    static_assert(sizeof(EeThreadStatus) == 0x30u, "Unexpected ee_thread_status_t size.");
    static_assert(sizeof(EeSemaStatus) == 0x18u, "Unexpected ee_sema_t size.");

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        SET_GPR_U32(&ctx, reg, value);
    }

    int32_t getRegS32(const R5900Context &ctx, int reg)
    {
        return static_cast<int32_t>(::getRegU32(&ctx, reg));
    }

    void writeGuestU32(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    void writeGuestWords(uint8_t *rdram, uint32_t addr, const uint32_t *words, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            writeGuestU32(rdram, addr + static_cast<uint32_t>(i * sizeof(uint32_t)), words[i]);
        }
    }

    uint32_t readGuestU32(const uint8_t *rdram, uint32_t addr)
    {
        uint32_t value = 0;
        std::memcpy(&value, rdram + addr, sizeof(value));
        return value;
    }

    template <typename Predicate>
    bool waitUntil(Predicate pred, std::chrono::milliseconds timeout)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (pred())
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return pred();
    }

    bool callSyscall(uint32_t syscallNumber, uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        return dispatchNumericSyscall(syscallNumber, rdram, ctx, runtime);
    }

    void overrideReturnHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        setReturnU32(ctx, ::getRegU32(ctx, 4) + ::getRegU32(ctx, 5));
        ctx->pc = ::getRegU32(ctx, 31);
    }

    void overrideBrokenHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        setReturnU32(ctx, 0xDEADBEEFu);
        ctx->pc = 0x12345678u;
    }

    void overrideRecursiveFindAddressHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        runtime->handleSyscall(rdram, ctx, 0x83u);
        ctx->pc = ::getRegU32(ctx, 31);
    }

    void overrideKsegCompareHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        auto getLowU64 = [](const R5900Context *cpu, int reg) -> uint64_t
        {
            return (reg == 0) ? 0u : static_cast<uint64_t>(_mm_extract_epi64(cpu->r[reg], 0));
        };
        auto setLowS32 = [](R5900Context *cpu, int reg, uint32_t value)
        {
            SET_GPR_S32(cpu, reg, value);
        };
        auto setLowU64 = [](R5900Context *cpu, int reg, uint64_t value)
        {
            SET_GPR_U64(cpu, reg, value);
        };

        const uint32_t nextA0 = static_cast<uint32_t>(::getRegU32(ctx, 4) + 4u);
        setLowS32(ctx, 4, nextA0);
        setLowU64(ctx, 2, (getLowU64(ctx, 4) < getLowU64(ctx, 5)) ? 1u : 0u);
        if (getLowU64(ctx, 2) == 0u)
        {
            ctx->r[4] = _mm_setzero_si128();
        }
        setLowU64(ctx, 2, getLowU64(ctx, 4));
        ctx->pc = ::getRegU32(ctx, 31);
    }

    constexpr uint64_t K_EXPECTED_UPPER64 = 0x1122334455667788ull;

    void overridePreserveUpper64Handler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        const uint64_t hi = static_cast<uint64_t>(_mm_extract_epi64(ctx->r[4], 1));
        const uint64_t low = static_cast<uint64_t>(_mm_extract_epi64(ctx->r[4], 0));
        const uint64_t expectedLow = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(0x80000000u)));
        setReturnU32(ctx, (hi == K_EXPECTED_UPPER64 && low == expectedLow) ? 1u : 0u);
        ctx->pc = ::getRegU32(ctx, 31);
    }

    void waitEventAfterSuspendHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_EVENT_WAIT_READY_ADDR, 1u);
        while (readGuestU32(rdram, K_EVENT_WAIT_GATE_ADDR) == 0u)
        {
            if (runtime && runtime->isStopRequested())
            {
                ctx->pc = 0u;
                return;
            }
            std::this_thread::yield();
        }

        const uint32_t eid = ::getRegU32(ctx, 4);
        setRegU32(*ctx, 4, eid);
        setRegU32(*ctx, 5, 0x4u);
        setRegU32(*ctx, 6, 1u);
        setRegU32(*ctx, 7, 0u);
        WaitEventFlag(rdram, ctx, runtime);
        ctx->pc = 0u;
    }

    void waitSemaUntilTerminatedHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_TERMINATE_SEMA_WAIT_READY_ADDR, 1u);
        WaitSema(rdram, ctx, runtime);
        writeGuestU32(rdram, K_TERMINATE_SEMA_WAIT_READY_ADDR, 2u);
        ctx->pc = 0u;
    }

    void sleepWakeCountHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_SLEEP_STAGE_ADDR, 1u);
        while (readGuestU32(rdram, K_SLEEP_GATE_ADDR) == 0u)
        {
            if (runtime && runtime->isStopRequested())
            {
                ctx->pc = 0u;
                return;
            }
            std::this_thread::yield();
        }

        for (uint32_t index = 0u; index < 4u; ++index)
        {
            SleepThread(rdram, ctx, runtime);
            writeGuestU32(
                rdram,
                K_SLEEP_RETURN_ADDR + index * sizeof(uint32_t),
                static_cast<uint32_t>(getRegS32(*ctx, 2)));
            writeGuestU32(rdram, K_SLEEP_STAGE_ADDR, index + 2u);
        }
        ctx->pc = 0u;
    }

    void waitSuspendSleepHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_WAITSUSPEND_STAGE_ADDR, 1u);
        SleepThread(rdram, ctx, runtime);
        writeGuestU32(
            rdram,
            K_WAITSUSPEND_RETURN_ADDR,
            static_cast<uint32_t>(getRegS32(*ctx, 2)));
        writeGuestU32(rdram, K_WAITSUSPEND_STAGE_ADDR, 2u);
        ctx->pc = 0u;
    }

    void idReuseSleepHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_ID_REUSE_STAGE_ADDR, 1u);
        SleepThread(rdram, ctx, runtime);
        writeGuestU32(
            rdram,
            K_ID_REUSE_RETURN_ADDR,
            static_cast<uint32_t>(getRegS32(*ctx, 2)));
        writeGuestU32(rdram, K_ID_REUSE_STAGE_ADDR, 2u);
        ctx->pc = 0u;
    }

    void semaWaitOracleHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_SEMA_ORACLE_STAGE_ADDR, 1u);
        WaitSema(rdram, ctx, runtime);
        writeGuestU32(
            rdram,
            K_SEMA_ORACLE_RETURN_ADDR,
            static_cast<uint32_t>(getRegS32(*ctx, 2)));
        writeGuestU32(rdram, K_SEMA_ORACLE_STAGE_ADDR, 2u);
        ctx->pc = 0u;
    }

    void terminateCandidateSleepHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_TERMINATE_CANDIDATE_STAGE_ADDR, 1u);
        SleepThread(rdram, ctx, runtime);
        writeGuestU32(rdram, K_TERMINATE_CANDIDATE_STAGE_ADDR, 2u);
        ctx->pc = 0u;
    }

    void terminateReadyTargetHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        writeGuestU32(rdram, K_TERMINATE_TARGET_STAGE_ADDR, 1u);
        ctx->pc = 0u;
    }

    void selfExitThreadHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        GetThreadId(rdram, ctx, runtime);
        writeGuestU32(
            rdram,
            K_EXIT_THREAD_ID_ADDR,
            static_cast<uint32_t>(getRegS32(*ctx, 2)));
        writeGuestU32(rdram, K_EXIT_THREAD_STAGE_ADDR, 1u);
        ExitThread(rdram, ctx, runtime);
        writeGuestU32(rdram, K_EXIT_THREAD_STAGE_ADDR, 99u);
    }

    void selfExitDeleteThreadHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        GetThreadId(rdram, ctx, runtime);
        writeGuestU32(
            rdram,
            K_EXIT_DELETE_THREAD_ID_ADDR,
            static_cast<uint32_t>(getRegS32(*ctx, 2)));
        writeGuestU32(rdram, K_EXIT_DELETE_THREAD_STAGE_ADDR, 1u);
        ExitDeleteThread(rdram, ctx, runtime);
        writeGuestU32(rdram, K_EXIT_DELETE_THREAD_STAGE_ADDR, 99u);
    }

    void poisonedCurrentThreadIdHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_currentThreadId = 0x6A6A;
        GetThreadId(rdram, ctx, runtime);
        writeGuestU32(
            rdram,
            K_CURRENT_THREAD_ID_ADDR,
            static_cast<uint32_t>(getRegS32(*ctx, 2)));
        ctx->pc = 0u;
    }

    void controlSleepWaitHandler(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SleepThread(rdram, ctx, runtime);
        ctx->pc = 0u;
    }

    void controlSemaWaitHandler(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sid = ::getRegU32(ctx, 4);
        setRegU32(*ctx, 4, sid);
        WaitSema(rdram, ctx, runtime);
        ctx->pc = 0u;
    }

    void controlEventWaitHandler(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t eid = ::getRegU32(ctx, 4);
        setRegU32(*ctx, 4, eid);
        setRegU32(*ctx, 5, 1u);
        setRegU32(*ctx, 6, 1u); // WEF_OR
        setRegU32(*ctx, 7, 0u);
        WaitEventFlag(rdram, ctx, runtime);
        ctx->pc = 0u;
    }

    void controlSuspendWaitHandler(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setRegU32(*ctx, 4, 0u);
        SuspendThread(rdram, ctx, runtime);
        ctx->pc = 0u;
    }

    void rescheduleMarkerHandler(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        writeGuestU32(
            rdram,
            K_RESCHEDULE_STAGE_ADDR,
            readGuestU32(rdram, K_RESCHEDULE_STAGE_ADDR) + 1u);
        ctx->pc = 0u;
    }

    void alarmNoopHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = 0u;
    }

    struct TestEnv
    {
        std::vector<uint8_t> rdram;
        R5900Context ctx{};
        PS2Runtime runtime;

        TestEnv() : rdram(PS2_RAM_SIZE, 0)
        {
            std::memset(&ctx, 0, sizeof(ctx));
        }
    };
}

void register_ps2_runtime_kernel_tests()
{
    MiniTest::Case("PS2RuntimeKernel", [](TestCase &tc)
    {
        tc.Run("thread create/refer/delete follows EE status layout", [](TestCase &t)
        {
            TestEnv env;

            const uint32_t threadParam[9] = {
                0x11223344u, // status (ignored by CreateThread)
                0x00200000u, // entry
                0x00300000u, // stack
                0x00000800u, // stack size
                0x00120000u, // gp
                5u,          // initial priority
                77u,         // current priority (ignored)
                0x55667788u, // attr (ignored)
                0x19AABBCCu  // option (ignored)
            };

            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, threadParam, std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(env.rdram.data(), &env.ctx, &env.runtime);

            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(tid >= 2, "CreateThread should return a valid non-main thread id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferThreadStatus(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                THS_DORMANT,
                "ReferThreadStatus should return the dormant status");

            EeThreadStatus status{};
            std::memcpy(&status, env.rdram.data() + K_STATUS_ADDR, sizeof(status));
            t.Equals(status.status, THS_DORMANT, "new thread should be dormant before StartThread");
            t.Equals(status.func, threadParam[1], "status.func should match entry");
            t.Equals(status.stack, threadParam[2], "status.stack should match configured stack");
            t.Equals(status.stack_size, static_cast<int32_t>(threadParam[3]), "status.stack_size should match thread param");
            t.Equals(status.gp_reg, threadParam[4], "status.gp_reg should match configured gp");
            t.Equals(status.initial_priority, 5, "status.initial_priority should match thread param");
            t.Equals(status.current_priority, 5, "status.current_priority should start at initial priority");
            t.Equals(status.attr, 0u, "CreateThread should ignore input attr");
            t.Equals(status.option, 0u, "CreateThread should ignore input option");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "DeleteThread should return the dormant thread id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferThreadStatus(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_THID, "deleted thread id should no longer be referable");
        });

        tc.Run("EE thread registries are isolated per runtime lifetime", [](TestCase &t)
        {
            notifyRuntimeStop();
            auto first = std::make_unique<TestEnv>();

            auto createDormantThread =
                [&](TestEnv &env,
                    uint32_t entry,
                    uint32_t stack)
                {
                    const uint32_t threadParam[9] = {
                        0u,
                        entry,
                        stack,
                        0x00000800u,
                        0u,
                        40u,
                        0u,
                        0u,
                        0u,
                    };
                    writeGuestWords(
                        env.rdram.data(),
                        K_PARAM_ADDR,
                        threadParam,
                        std::size(threadParam));
                    R5900Context createCtx{};
                    setRegU32(createCtx, 4, K_PARAM_ADDR);
                    CreateThread(
                        env.rdram.data(),
                        &createCtx,
                        &env.runtime);
                    return getRegS32(createCtx, 2);
                };
            auto referThread =
                [&](TestEnv &env,
                    int32_t tid,
                    EeThreadStatus &status)
                {
                    std::memset(&status, 0, sizeof(status));
                    R5900Context statusCtx{};
                    setRegU32(
                        statusCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferThreadStatus(
                        env.rdram.data(),
                        &statusCtx,
                        &env.runtime);
                    std::memcpy(
                        &status,
                        env.rdram.data() + K_STATUS_ADDR,
                        sizeof(status));
                    return getRegS32(statusCtx, 2);
                };

            constexpr uint32_t kFirstEntry = 0x00201000u;
            constexpr uint32_t kSecondEntry = 0x00202000u;
            const int32_t firstTid =
                createDormantThread(
                    *first, kFirstEntry, 0x00301000u);
            t.Equals(
                firstTid,
                2,
                "the first runtime should allocate thread id 2");

            {
                auto second = std::make_unique<TestEnv>();
                const int32_t secondTid =
                    createDormantThread(
                        *second,
                        kSecondEntry,
                        0x00302000u);
                t.Equals(
                    secondTid,
                    2,
                    "an independent runtime should own an independent id ring");

                EeThreadStatus firstStatus{};
                t.Equals(
                    referThread(
                        *first, firstTid, firstStatus),
                    THS_DORMANT,
                    "the first runtime's thread should remain queryable");
                t.Equals(
                    firstStatus.func,
                    kFirstEntry,
                    "the first runtime should retain its own thread payload");

                EeThreadStatus secondStatus{};
                t.Equals(
                    referThread(
                        *second, secondTid, secondStatus),
                    THS_DORMANT,
                    "the second runtime's thread should be queryable");
                t.Equals(
                    secondStatus.func,
                    kSecondEntry,
                    "the second runtime should retain its own thread payload");
            }

            EeThreadStatus survivingStatus{};
            t.Equals(
                referThread(
                    *first, firstTid, survivingStatus),
                THS_DORMANT,
                "destroying a second runtime must not reset the first runtime");
            t.Equals(
                survivingStatus.func,
                kFirstEntry,
                "the surviving runtime should retain its thread after peer destruction");
        });

        tc.Run("runtime context binding owns current EE thread selection", [](TestCase &t)
        {
            TestEnv env;

            g_currentThreadId = 0x5A5A;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                GetThreadId(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
            }
            t.Equals(
                getRegS32(env.ctx, 2),
                1,
                "a bound main context should ignore a corrupted TLS adapter");
            g_currentThreadId = 1;

            constexpr uint32_t kEntry = 0x00202500u;
            env.runtime.registerFunction(
                kEntry, &poisonedCurrentThreadIdHandler);
            const uint32_t threadParam[9] = {
                0u,
                kEntry,
                0x00302500u,
                0x00000800u,
                0u,
                40u,
                0u,
                0u,
                0u,
            };
            writeGuestWords(
                env.rdram.data(),
                K_PARAM_ADDR,
                threadParam,
                std::size(threadParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, K_PARAM_ADDR);
            CreateThread(
                env.rdram.data(),
                &createCtx,
                &env.runtime);
            const int32_t tid = getRegS32(createCtx, 2);
            t.IsTrue(
                tid >= 2,
                "the current-selection worker should be created");

            R5900Context startCtx{};
            setRegU32(
                startCtx, 4, static_cast<uint32_t>(tid));
            StartThread(
                env.rdram.data(),
                &startCtx,
                &env.runtime);
            t.Equals(
                getRegS32(startCtx, 2),
                tid,
                "the current-selection worker should start");

            const bool dormant = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(),
                    &statusCtx,
                    &env.runtime);
                return getRegS32(statusCtx, 2) ==
                       THS_DORMANT;
            }, std::chrono::milliseconds(500));
            t.IsTrue(
                dormant,
                "the current-selection worker should finish");
            t.Equals(
                static_cast<int32_t>(readGuestU32(
                    env.rdram.data(),
                    K_CURRENT_THREAD_ID_ADDR)),
                tid,
                "a bound worker context should ignore a corrupted TLS adapter");
            t.IsTrue(
                env.runtime.eeThreadLegacyAdapterMismatchCount() >= 1u,
                "the runtime should detect TLS adapter divergence");
        });

        tc.Run("start thread validates target and entry registration", [](TestCase &t)
        {
            TestEnv env;

            const uint32_t threadParam[7] = {
                0u,
                0x00250000u, // entry not registered in runtime
                0x00300000u,
                0x00000400u,
                0x00110000u,
                8u,
                0u
            };

            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, threadParam, std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(tid >= 2, "CreateThread should return an id before StartThread check");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, 0x12345678u);
            StartThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ERROR, "StartThread should fail when entry is not registered");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferThreadStatus(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                THS_DORMANT,
                "failed StartThread should retain the dormant status return");

            EeThreadStatus status{};
            std::memcpy(&status, env.rdram.data() + K_STATUS_ADDR, sizeof(status));
            t.Equals(status.status, THS_DORMANT, "thread should remain dormant when StartThread fails early");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "DeleteThread should return the failed-start thread id");
        });

        tc.Run("thread scheduling syscalls preserve EE BIOS success values", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00252000u;
            constexpr int32_t kPriority = 40;
            env.runtime.registerFunction(
                kThreadEntry, &alarmNoopHandler);

            setRegU32(env.ctx, 4, 0u);
            setRegU32(
                env.ctx, 5, static_cast<uint32_t>(kPriority));
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                0,
                "bare EE bootstrap should begin at priority zero");

            setRegU32(env.ctx, 4, 0u);
            RotateThreadReadyQueue(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                0,
                "RotateThreadReadyQueue should treat priority zero literally");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 0u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                kPriority,
                "ChangeThreadPriority should return 40 when changing to priority zero");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferThreadStatus(
                env.rdram.data(), &env.ctx, &env.runtime);
            EeThreadStatus bootstrapStatus{};
            std::memcpy(
                &bootstrapStatus,
                env.rdram.data() + K_STATUS_ADDR,
                sizeof(bootstrapStatus));
            t.Equals(
                bootstrapStatus.current_priority,
                0,
                "priority zero should remain the current priority");

            InitThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                1,
                "InitThread should preserve its existing helper return");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferThreadStatus(
                env.rdram.data(), &env.ctx, &env.runtime);
            EeThreadStatus initializedStatus{};
            std::memcpy(
                &initializedStatus,
                env.rdram.data() + K_STATUS_ADDR,
                sizeof(initializedStatus));
            t.Equals(
                initializedStatus.initial_priority,
                0,
                "InitThread should preserve the bootstrap initial priority");
            t.Equals(
                initializedStatus.current_priority,
                1,
                "InitThread should promote the current thread to priority one");

            const uint32_t zeroPriorityParam[7] = {
                0u,
                kThreadEntry,
                0x00303000u,
                0x00000800u,
                0u,
                0u,
                0u
            };
            writeGuestWords(
                env.rdram.data(),
                K_PARAM_ADDR,
                zeroPriorityParam,
                std::size(zeroPriorityParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t zeroPriorityTid = getRegS32(env.ctx, 2);
            t.IsTrue(
                zeroPriorityTid >= 2,
                "CreateThread should accept priority zero");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(zeroPriorityTid));
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferThreadStatus(
                env.rdram.data(), &env.ctx, &env.runtime);
            EeThreadStatus zeroPriorityStatus{};
            std::memcpy(
                &zeroPriorityStatus,
                env.rdram.data() + K_STATUS_ADDR,
                sizeof(zeroPriorityStatus));
            t.Equals(
                zeroPriorityStatus.initial_priority,
                0,
                "CreateThread should preserve initial priority zero");
            t.Equals(
                zeroPriorityStatus.current_priority,
                0,
                "CreateThread should preserve current priority zero");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(zeroPriorityTid));
            DeleteThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                zeroPriorityTid,
                "DeleteThread should remove the dormant priority-zero thread");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(
                env.ctx, 5, static_cast<uint32_t>(kPriority));
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                1,
                "ChangeThreadPriority should return the post-InitThread priority");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(kPriority));
            RotateThreadReadyQueue(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                kPriority,
                "RotateThreadReadyQueue should return the requested priority");

            const uint32_t threadParam[7] = {
                0u,
                kThreadEntry,
                0x00304000u,
                0x00000800u,
                0u,
                static_cast<uint32_t>(kPriority),
                0u
            };
            writeGuestWords(
                env.rdram.data(),
                K_PARAM_ADDR,
                threadParam,
                std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(
                tid >= 2,
                "CreateThread should return a worker thread id");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, 0u);
            StartThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                tid,
                "StartThread should return the started thread id");

            const bool dormant = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(),
                    &statusCtx,
                    &env.runtime);
                if (getRegS32(statusCtx, 2) != THS_DORMANT)
                {
                    return false;
                }

                EeThreadStatus status{};
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return status.status == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                dormant,
                "the no-op worker should return to dormant");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                tid,
                "DeleteThread should return the deleted thread id");
        });

        tc.Run("thread id and wakeup guard rails match kernel-style errors", [](TestCase &t)
        {
            TestEnv env;

            GetThreadId(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t selfTid = getRegS32(env.ctx, 2);
            t.IsTrue(selfTid > 0, "GetThreadId should return a positive thread id");

            setRegU32(env.ctx, 4, 0u);
            WakeupThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ILLEGAL_THID, "WakeupThread(TH_SELF/0) should be illegal");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(selfTid));
            WakeupThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ILLEGAL_THID, "WakeupThread(self) should be illegal");

            setRegU32(env.ctx, 4, 0u);
            iCancelWakeupThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ILLEGAL_THID, "iCancelWakeupThread(0) should be illegal");

            setRegU32(env.ctx, 4, 0u);
            CancelWakeupThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "CancelWakeupThread(TH_SELF) should return previous count (0)");
        });

        tc.Run("sleep and wakeup counts follow the EE BIOS contract", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00253000u;
            env.runtime.registerFunction(kThreadEntry, &sleepWakeCountHandler);
            writeGuestU32(env.rdram.data(), K_SLEEP_GATE_ADDR, 0u);
            writeGuestU32(env.rdram.data(), K_SLEEP_STAGE_ADDR, 0u);

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);

            const uint32_t threadParam[9] = {
                0u,
                kThreadEntry,
                0x00308000u,
                0x00000800u,
                0u,
                30u,
                0u,
                0u,
                0u
            };
            writeGuestWords(
                env.rdram.data(),
                K_PARAM_ADDR,
                threadParam,
                std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(tid >= 2, "CreateThread should return a worker id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, 0u);
            StartThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "StartThread should return the worker id");

            const bool entered = waitUntil([&]()
            {
                return readGuestU32(env.rdram.data(), K_SLEEP_STAGE_ADDR) == 1u;
            }, std::chrono::milliseconds(200));
            t.IsTrue(entered, "worker should reach the pre-sleep gate");

            for (int call = 0; call < 3; ++call)
            {
                R5900Context wakeCtx{};
                setRegU32(wakeCtx, 4, static_cast<uint32_t>(tid));
                if (call == 1)
                {
                    WakeupThread(
                        env.rdram.data(), &wakeCtx, &env.runtime);
                }
                else
                {
                    iWakeupThread(
                        env.rdram.data(), &wakeCtx, &env.runtime);
                }
                t.Equals(
                    getRegS32(wakeCtx, 2),
                    tid,
                    "waking a runnable thread should return its id");
            }

            R5900Context statusCtx{};
            setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
            setRegU32(statusCtx, 5, K_STATUS_ADDR);
            ReferThreadStatus(
                env.rdram.data(), &statusCtx, &env.runtime);
            EeThreadStatus status{};
            std::memcpy(
                &status,
                env.rdram.data() + K_STATUS_ADDR,
                sizeof(status));
            t.Equals(status.wakeupCount, 3u, "three early wakes should accumulate");

            R5900Context cancelCtx{};
            setRegU32(cancelCtx, 4, static_cast<uint32_t>(tid));
            CancelWakeupThread(
                env.rdram.data(), &cancelCtx, &env.runtime);
            t.Equals(
                getRegS32(cancelCtx, 2),
                3,
                "CancelWakeupThread should return and clear the accumulated count");

            for (int call = 0; call < 2; ++call)
            {
                R5900Context wakeCtx{};
                setRegU32(wakeCtx, 4, static_cast<uint32_t>(tid));
                if (call == 0)
                {
                    WakeupThread(
                        env.rdram.data(), &wakeCtx, &env.runtime);
                }
                else
                {
                    iWakeupThread(
                        env.rdram.data(), &wakeCtx, &env.runtime);
                }
                t.Equals(
                    getRegS32(wakeCtx, 2),
                    tid,
                    "replenishing the wake count should return the worker id");
            }

            writeGuestU32(env.rdram.data(), K_SLEEP_GATE_ADDR, 1u);
            const bool sleeping = waitUntil([&]()
            {
                R5900Context currentStatusCtx{};
                setRegU32(
                    currentStatusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(currentStatusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(),
                    &currentStatusCtx,
                    &env.runtime);
                EeThreadStatus currentStatus{};
                std::memcpy(
                    &currentStatus,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(currentStatus));
                return readGuestU32(
                           env.rdram.data(), K_SLEEP_STAGE_ADDR) == 3u &&
                       getRegS32(currentStatusCtx, 2) == THS_WAIT &&
                       currentStatus.status == THS_WAIT &&
                       currentStatus.waitType == TSW_SLEEP &&
                       currentStatus.wakeupCount == 0u;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                sleeping,
                "two accumulated wakes should satisfy two sleeps before the third blocks");
            t.Equals(
                readGuestU32(env.rdram.data(), K_SLEEP_RETURN_ADDR),
                static_cast<uint32_t>(tid),
                "an immediate SleepThread should return the current thread id");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_SLEEP_RETURN_ADDR + sizeof(uint32_t)),
                static_cast<uint32_t>(tid),
                "the second immediate SleepThread should return the current thread id");

            R5900Context emptyCancelCtx{};
            setRegU32(emptyCancelCtx, 4, static_cast<uint32_t>(tid));
            iCancelWakeupThread(
                env.rdram.data(), &emptyCancelCtx, &env.runtime);
            t.Equals(
                getRegS32(emptyCancelCtx, 2),
                0,
                "canceling an empty wake count should return zero");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context rawWakeCtx{};
                setRegU32(rawWakeCtx, 4, static_cast<uint32_t>(tid));
                iWakeupThread(
                    env.rdram.data(), &rawWakeCtx, &env.runtime);
                t.Equals(
                    getRegS32(rawWakeCtx, 2),
                    tid,
                    "raw wake should return the worker id");

                R5900Context readyStatusCtx{};
                setRegU32(
                    readyStatusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(readyStatusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(),
                    &readyStatusCtx,
                    &env.runtime);
                EeThreadStatus readyStatus{};
                std::memcpy(
                    &readyStatus,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(readyStatus));
                t.Equals(
                    getRegS32(readyStatusCtx, 2),
                    THS_READY,
                    "raw wake should leave the sleeper ready");
                t.Equals(
                    readyStatus.status,
                    THS_READY,
                    "raw wake should publish ready status");
                t.Equals(
                    readyStatus.waitType,
                    0u,
                    "raw wake should clear the sleep wait reason");
                t.Equals(
                    readyStatus.wakeupCount,
                    0u,
                    "waking a sleeper should not expose an accumulated count");
                t.Equals(
                    readGuestU32(env.rdram.data(), K_SLEEP_STAGE_ADDR),
                    3u,
                    "raw wake should not dispatch the worker before returning");
            }

            const bool sleepingAgain = waitUntil([&]()
            {
                R5900Context currentStatusCtx{};
                setRegU32(
                    currentStatusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(currentStatusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(),
                    &currentStatusCtx,
                    &env.runtime);
                return readGuestU32(
                           env.rdram.data(), K_SLEEP_STAGE_ADDR) == 4u &&
                       getRegS32(currentStatusCtx, 2) == THS_WAIT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                sleepingAgain,
                "the raw-woken worker should return from sleep and block again");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_SLEEP_RETURN_ADDR + 2u * sizeof(uint32_t)),
                static_cast<uint32_t>(tid),
                "a blocking SleepThread should return the current thread id after wake");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context wakeCtx{};
                setRegU32(wakeCtx, 4, static_cast<uint32_t>(tid));
                WakeupThread(
                    env.rdram.data(), &wakeCtx, &env.runtime);
                t.Equals(
                    getRegS32(wakeCtx, 2),
                    tid,
                    "ordinary wake should return the worker id");
                t.Equals(
                    readGuestU32(env.rdram.data(), K_SLEEP_STAGE_ADDR),
                    5u,
                    "ordinary wake should dispatch the worker before returning");
            }

            const bool dormant = waitUntil([&]()
            {
                R5900Context currentStatusCtx{};
                setRegU32(
                    currentStatusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(currentStatusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(),
                    &currentStatusCtx,
                    &env.runtime);
                return getRegS32(currentStatusCtx, 2) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(dormant, "the worker should finish after its fourth sleep");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_SLEEP_RETURN_ADDR + 3u * sizeof(uint32_t)),
                static_cast<uint32_t>(tid),
                "the ordinary-woken SleepThread should return the current thread id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                tid,
                "DeleteThread should remove the completed sleep worker");
        });

        tc.Run("sleeping suspended thread follows the EE state transitions", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00254000u;
            env.runtime.registerFunction(kThreadEntry, &waitSuspendSleepHandler);
            writeGuestU32(env.rdram.data(), K_WAITSUSPEND_STAGE_ADDR, 0u);

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);

            const uint32_t threadParam[9] = {
                0u,
                kThreadEntry,
                0x00309000u,
                0x00000800u,
                0u,
                30u,
                0u,
                0u,
                0u
            };
            writeGuestWords(
                env.rdram.data(),
                K_PARAM_ADDR,
                threadParam,
                std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(tid >= 2, "CreateThread should return a worker id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, 0u);
            StartThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "StartThread should return the worker id");

            const bool sleeping = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return readGuestU32(
                           env.rdram.data(), K_WAITSUSPEND_STAGE_ADDR) == 1u &&
                       getRegS32(statusCtx, 2) == THS_WAIT &&
                       status.status == THS_WAIT &&
                       status.waitType == TSW_SLEEP;
            }, std::chrono::milliseconds(200));
            t.IsTrue(sleeping, "worker should block in SleepThread");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            SuspendThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                tid,
                "SuspendThread should return the suspended worker id");

            R5900Context waitSuspendStatusCtx{};
            setRegU32(
                waitSuspendStatusCtx, 4, static_cast<uint32_t>(tid));
            setRegU32(waitSuspendStatusCtx, 5, K_STATUS_ADDR);
            ReferThreadStatus(
                env.rdram.data(),
                &waitSuspendStatusCtx,
                &env.runtime);
            EeThreadStatus waitSuspendStatus{};
            std::memcpy(
                &waitSuspendStatus,
                env.rdram.data() + K_STATUS_ADDR,
                sizeof(waitSuspendStatus));
            t.Equals(
                getRegS32(waitSuspendStatusCtx, 2),
                THS_WAITSUSPEND,
                "a suspended sleeper status query should return WAITSUSPEND");
            t.Equals(
                waitSuspendStatus.status,
                THS_WAITSUSPEND,
                "a suspended sleeper should publish WAITSUSPEND");
            t.Equals(
                waitSuspendStatus.waitType,
                TSW_SLEEP,
                "WAITSUSPEND should retain the sleep wait reason");
            t.Equals(
                waitSuspendStatus.wakeupCount,
                0u,
                "WAITSUSPEND should retain a zero wake count");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);

                R5900Context rawWakeCtx{};
                setRegU32(rawWakeCtx, 4, static_cast<uint32_t>(tid));
                iWakeupThread(
                    env.rdram.data(), &rawWakeCtx, &env.runtime);
                t.Equals(
                    getRegS32(rawWakeCtx, 2),
                    tid,
                    "raw wake should return the wait-suspended worker id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(), K_WAITSUSPEND_STAGE_ADDR),
                    1u,
                    "raw wake should not dispatch a wait-suspended worker");

                R5900Context suspendedStatusCtx{};
                setRegU32(
                    suspendedStatusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(suspendedStatusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    env.rdram.data(),
                    &suspendedStatusCtx,
                    &env.runtime);
                EeThreadStatus suspendedStatus{};
                std::memcpy(
                    &suspendedStatus,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(suspendedStatus));
                t.Equals(
                    getRegS32(suspendedStatusCtx, 2),
                    THS_SUSPEND,
                    "raw wake should leave a wait-suspended worker suspended");
                t.Equals(
                    suspendedStatus.status,
                    THS_SUSPEND,
                    "raw wake should publish SUSPEND");
                t.Equals(
                    suspendedStatus.waitType,
                    0u,
                    "raw wake should clear the sleep wait reason");
                t.Equals(
                    suspendedStatus.wakeupCount,
                    0u,
                    "raw wake should not expose an accumulated count");

                R5900Context resumeCtx{};
                setRegU32(resumeCtx, 4, static_cast<uint32_t>(tid));
                ResumeThread(
                    env.rdram.data(), &resumeCtx, &env.runtime);
                t.Equals(
                    getRegS32(resumeCtx, 2),
                    tid,
                    "ResumeThread should return the resumed worker id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(), K_WAITSUSPEND_STAGE_ADDR),
                    2u,
                    "ordinary resume should dispatch the higher-priority worker before returning");
            }

            const bool dormant = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                return getRegS32(statusCtx, 2) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                dormant,
                "the resumed higher-priority worker should finish before cleanup");

            if (!dormant)
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context cleanupWakeCtx{};
                setRegU32(
                    cleanupWakeCtx, 4, static_cast<uint32_t>(tid));
                WakeupThread(
                    env.rdram.data(),
                    &cleanupWakeCtx,
                    &env.runtime);
            }

            const bool cleanedUp = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                return getRegS32(statusCtx, 2) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(cleanedUp, "worker should be dormant before deletion");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), K_WAITSUSPEND_RETURN_ADDR),
                static_cast<uint32_t>(tid),
                "the resumed SleepThread should return the current thread id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                tid,
                "DeleteThread should remove the resumed sleep worker");
        });

        tc.Run("thread ids wrap and stale numeric ids address their replacement", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00255000u;
            constexpr uint32_t kStackAddr = 0x0030A000u;
            env.runtime.registerFunction(kThreadEntry, &idReuseSleepHandler);
            writeGuestU32(env.rdram.data(), K_ID_REUSE_STAGE_ADDR, 0u);
            writeGuestU32(env.rdram.data(), K_ID_REUSE_RETURN_ADDR, 0u);

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);

            const uint32_t threadParam[9] = {
                0u,
                kThreadEntry,
                kStackAddr,
                0x00001000u,
                0u,
                30u,
                0u,
                0u,
                0u
            };
            writeGuestWords(
                env.rdram.data(),
                K_PARAM_ADDR,
                threadParam,
                std::size(threadParam));

            auto createThread = [&]()
            {
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto deleteThread = [&](int32_t tid)
            {
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx, 4, static_cast<uint32_t>(tid));
                DeleteThread(
                    env.rdram.data(), &deleteCtx, &env.runtime);
                return getRegS32(deleteCtx, 2);
            };

            const int32_t firstTid = createThread();
            const int32_t keeperTid = createThread();
            t.Equals(firstTid, 2, "the first allocatable thread id should be 2");
            t.Equals(keeperTid, 3, "the next allocatable thread id should be 3");
            t.Equals(
                deleteThread(firstTid),
                firstTid,
                "deleting the first dormant thread should return id 2");

            R5900Context staleBeforeReuseCtx{};
            setRegU32(
                staleBeforeReuseCtx,
                4,
                static_cast<uint32_t>(firstTid));
            iWakeupThread(
                env.rdram.data(),
                &staleBeforeReuseCtx,
                &env.runtime);
            t.Equals(
                getRegS32(staleBeforeReuseCtx, 2),
                KE_ERROR,
                "a deleted thread id should reject a raw wake before reuse");

            std::vector<int32_t> dormantIds;
            dormantIds.reserve(252u);
            for (int32_t expectedTid = 4;
                 expectedTid <= 0xFF;
                 ++expectedTid)
            {
                const int32_t tid = createThread();
                t.Equals(
                    tid,
                    expectedTid,
                    "thread ids should advance through the remaining EE id ring");
                if (tid > 0)
                {
                    dormantIds.push_back(tid);
                }
            }

            const int32_t replacementTid = createThread();
            t.Equals(
                replacementTid,
                firstTid,
                "the allocator should wrap and reuse deleted id 2");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(replacementTid));
            setRegU32(env.ctx, 5, 0u);
            StartThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                replacementTid,
                "StartThread should return the replacement id");

            const bool sleeping = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx,
                    4,
                    static_cast<uint32_t>(replacementTid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return readGuestU32(
                           env.rdram.data(), K_ID_REUSE_STAGE_ADDR) == 1u &&
                       getRegS32(statusCtx, 2) == THS_WAIT &&
                       status.waitType == TSW_SLEEP;
            }, std::chrono::milliseconds(200));
            t.IsTrue(sleeping, "the replacement thread should block in SleepThread");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context staleAfterReuseCtx{};
                setRegU32(
                    staleAfterReuseCtx,
                    4,
                    static_cast<uint32_t>(firstTid));
                iWakeupThread(
                    env.rdram.data(),
                    &staleAfterReuseCtx,
                    &env.runtime);
                t.Equals(
                    getRegS32(staleAfterReuseCtx, 2),
                    replacementTid,
                    "the old numeric id should address its replacement after reuse");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(), K_ID_REUSE_STAGE_ADDR),
                    1u,
                    "raw wake through the reused id should not dispatch");

                R5900Context readyStatusCtx{};
                setRegU32(
                    readyStatusCtx,
                    4,
                    static_cast<uint32_t>(replacementTid));
                setRegU32(readyStatusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    env.rdram.data(),
                    &readyStatusCtx,
                    &env.runtime);
                EeThreadStatus readyStatus{};
                std::memcpy(
                    &readyStatus,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(readyStatus));
                t.Equals(
                    getRegS32(readyStatusCtx, 2),
                    THS_READY,
                    "raw stale-id wake should make the replacement READY");
                t.Equals(
                    readyStatus.status,
                    THS_READY,
                    "the replacement should publish READY");
                t.Equals(
                    readyStatus.waitType,
                    0u,
                    "the replacement wake should clear its sleep reason");
                t.Equals(
                    readyStatus.wakeupCount,
                    0u,
                    "the replacement wake should not accumulate a count");
            }

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 60u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                40,
                "lowering the main priority should return its previous priority");

            const bool dormant = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx,
                    4,
                    static_cast<uint32_t>(replacementTid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                return getRegS32(statusCtx, 2) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(dormant, "the replacement worker should finish after its wake");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), K_ID_REUSE_RETURN_ADDR),
                static_cast<uint32_t>(replacementTid),
                "the replacement SleepThread should return reused id 2");
            t.Equals(
                deleteThread(replacementTid),
                replacementTid,
                "the dormant replacement should be deletable");

            for (const int32_t tid : dormantIds)
            {
                t.Equals(
                    deleteThread(tid),
                    tid,
                    "every unreused dormant id should remain independently deletable");
            }
            t.Equals(
                deleteThread(keeperTid),
                keeperTid,
                "the original keeper thread should remain independently deletable");
        });

        tc.Run("semaphore wait release and delete match the EE BIOS boundary", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00256000u;
            constexpr uint32_t kRawStackAddr = 0x0030B000u;
            constexpr uint32_t kOrdinaryStackAddr = 0x0030C000u;
            constexpr uint32_t kDeleteStackAddr = 0x0030D000u;
            env.runtime.registerFunction(kThreadEntry, &semaWaitOracleHandler);

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                0,
                "main should change from bootstrap priority zero to 40");

            const uint32_t semaParam[6] = {
                0u,
                1u,
                0u,
                0u,
                0x44u,
                0x12345678u
            };
            auto createSema = [&]()
            {
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    semaParam,
                    std::size(semaParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateSema(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto referSema = [&](int32_t sid, EeSemaStatus &status)
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(sid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                iReferSemaStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return getRegS32(statusCtx, 2);
            };
            auto referThread = [&](int32_t tid, EeThreadStatus &status)
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return getRegS32(statusCtx, 2);
            };
            auto createWorker = [&](uint32_t stackAddr)
            {
                const uint32_t threadParam[9] = {
                    0u,
                    kThreadEntry,
                    stackAddr,
                    0x00001000u,
                    0u,
                    30u,
                    0u,
                    0u,
                    0u
                };
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    threadParam,
                    std::size(threadParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto startWorker = [&](int32_t tid, int32_t sid)
            {
                writeGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_STAGE_ADDR, 0u);
                writeGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_RETURN_ADDR, 0u);
                R5900Context startCtx{};
                setRegU32(
                    startCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(
                    startCtx, 5, static_cast<uint32_t>(sid));
                StartThread(
                    env.rdram.data(), &startCtx, &env.runtime);
                return getRegS32(startCtx, 2);
            };
            auto waitForSemaBlock = [&](int32_t tid, int32_t sid)
            {
                return waitUntil([&]()
                {
                    EeThreadStatus threadStatus{};
                    EeSemaStatus semaStatus{};
                    return readGuestU32(
                               env.rdram.data(),
                               K_SEMA_ORACLE_STAGE_ADDR) == 1u &&
                           referThread(tid, threadStatus) == THS_WAIT &&
                           threadStatus.status == THS_WAIT &&
                           threadStatus.waitType == TSW_SEMA &&
                           threadStatus.waitId ==
                               static_cast<uint32_t>(sid) &&
                           referSema(sid, semaStatus) == KE_OK &&
                           semaStatus.wait_threads == 1;
                }, std::chrono::milliseconds(200));
            };
            auto deleteWorker = [&](int32_t tid)
            {
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx, 4, static_cast<uint32_t>(tid));
                DeleteThread(
                    env.rdram.data(), &deleteCtx, &env.runtime);
                return getRegS32(deleteCtx, 2);
            };

            const int32_t releaseSid = createSema();
            t.Equals(
                releaseSid,
                0,
                "the first EE semaphore id should be valid id zero");

            EeSemaStatus initialSemaStatus{};
            t.Equals(
                referSema(releaseSid, initialSemaStatus),
                KE_OK,
                "ReferSemaStatus should return zero for semaphore zero");
            t.Equals(initialSemaStatus.count, 0, "initial semaphore count should be zero");
            t.Equals(initialSemaStatus.max_count, 1, "maximum semaphore count should be one");
            t.Equals(initialSemaStatus.init_count, 0, "recorded initial count should be zero");
            t.Equals(initialSemaStatus.wait_threads, 0, "new semaphore should have no waiters");
            t.Equals(initialSemaStatus.attr, 0x44u, "semaphore attr should be retained");
            t.Equals(initialSemaStatus.option, 0x12345678u, "semaphore option should be retained");

            const int32_t rawTid = createWorker(kRawStackAddr);
            t.Equals(rawTid, 2, "the raw-release waiter should use thread id 2");
            t.Equals(
                startWorker(rawTid, releaseSid),
                rawTid,
                "StartThread should return the raw-release waiter id");
            t.IsTrue(
                waitForSemaBlock(rawTid, releaseSid),
                "raw-release worker should block in WaitSema");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);

                R5900Context releaseCtx{};
                setRegU32(
                    releaseCtx, 4, static_cast<uint32_t>(rawTid));
                iReleaseWaitThread(
                    env.rdram.data(), &releaseCtx, &env.runtime);
                t.Equals(
                    getRegS32(releaseCtx, 2),
                    rawTid,
                    "raw release should return the waiter thread id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(), K_SEMA_ORACLE_STAGE_ADDR),
                    1u,
                    "raw release should not dispatch the waiter");

                EeThreadStatus readyStatus{};
                t.Equals(
                    referThread(rawTid, readyStatus),
                    THS_READY,
                    "raw release should return READY from thread status");
                t.Equals(readyStatus.status, THS_READY, "raw-released waiter should publish READY");
                t.Equals(readyStatus.waitType, 0u, "raw release should clear the wait reason");
                t.Equals(readyStatus.waitId, 0u, "raw release should clear the wait object id");
                t.Equals(readyStatus.wakeupCount, 0u, "raw release should not change wake count");

                EeSemaStatus releasedSemaStatus{};
                t.Equals(
                    referSema(releaseSid, releasedSemaStatus),
                    KE_OK,
                    "raw-released semaphore should remain queryable");
                t.Equals(
                    releasedSemaStatus.wait_threads,
                    0,
                    "raw release should synchronously remove the waiter");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    getRegS32(env.ctx, 2),
                    40,
                    "forced raw-release dispatch should return old main priority");
            }

            const bool rawDormant = waitUntil([&]()
            {
                EeThreadStatus status{};
                return readGuestU32(
                           env.rdram.data(),
                           K_SEMA_ORACLE_STAGE_ADDR) == 2u &&
                       referThread(rawTid, status) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                rawDormant,
                "raw-released worker should finish after the scheduling boundary");
            t.Equals(
                static_cast<int32_t>(readGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_RETURN_ADDR)),
                KE_ERROR,
                "force-released WaitSema should return generic -1");
            EeThreadStatus rawFinalStatus{};
            t.Equals(
                referThread(rawTid, rawFinalStatus),
                THS_DORMANT,
                "raw-released worker should finish dormant");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                60,
                "restoring main priority should return 60");
            t.Equals(
                deleteWorker(rawTid),
                rawTid,
                "raw-released dormant worker should be deletable");

            const int32_t ordinaryTid =
                createWorker(kOrdinaryStackAddr);
            t.Equals(ordinaryTid, 3, "ordinary-release waiter should use thread id 3");
            t.Equals(
                startWorker(ordinaryTid, releaseSid),
                ordinaryTid,
                "StartThread should return the ordinary-release waiter id");
            t.IsTrue(
                waitForSemaBlock(ordinaryTid, releaseSid),
                "ordinary-release worker should block in WaitSema");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context releaseCtx{};
                setRegU32(
                    releaseCtx,
                    4,
                    static_cast<uint32_t>(ordinaryTid));
                ReleaseWaitThread(
                    env.rdram.data(), &releaseCtx, &env.runtime);
                t.Equals(
                    getRegS32(releaseCtx, 2),
                    ordinaryTid,
                    "ordinary release should return the waiter thread id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(), K_SEMA_ORACLE_STAGE_ADDR),
                    2u,
                    "ordinary release should dispatch the higher-priority waiter before returning");
            }

            t.Equals(
                static_cast<int32_t>(readGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_RETURN_ADDR)),
                KE_ERROR,
                "ordinary force-released WaitSema should return generic -1");
            const bool ordinaryDormant = waitUntil([&]()
            {
                EeThreadStatus status{};
                return referThread(ordinaryTid, status) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                ordinaryDormant,
                "ordinary-released worker should complete its host-thread epilogue");
            EeThreadStatus ordinaryFinalStatus{};
            t.Equals(
                referThread(ordinaryTid, ordinaryFinalStatus),
                THS_DORMANT,
                "ordinary-released worker should already be dormant");
            EeSemaStatus ordinarySemaStatus{};
            t.Equals(
                referSema(releaseSid, ordinarySemaStatus),
                KE_OK,
                "release-test semaphore should remain queryable");
            t.Equals(
                ordinarySemaStatus.wait_threads,
                0,
                "ordinary release should remove its waiter");
            t.Equals(
                deleteWorker(ordinaryTid),
                ordinaryTid,
                "ordinary-released dormant worker should be deletable");

            R5900Context deleteReleaseSemaCtx{};
            setRegU32(
                deleteReleaseSemaCtx,
                4,
                static_cast<uint32_t>(releaseSid));
            DeleteSema(
                env.rdram.data(),
                &deleteReleaseSemaCtx,
                &env.runtime);
            t.Equals(
                getRegS32(deleteReleaseSemaCtx, 2),
                releaseSid,
                "DeleteSema should return valid semaphore id zero");

            const int32_t deleteSid = createSema();
            t.Equals(
                deleteSid,
                0,
                "the next semaphore should immediately reuse deleted id zero");
            const int32_t deleteTid =
                createWorker(kDeleteStackAddr);
            t.Equals(deleteTid, 4, "delete waiter should use thread id 4");
            t.Equals(
                startWorker(deleteTid, deleteSid),
                deleteTid,
                "StartThread should return the delete waiter id");
            t.IsTrue(
                waitForSemaBlock(deleteTid, deleteSid),
                "delete worker should block in WaitSema");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context deleteSemaCtx{};
                setRegU32(
                    deleteSemaCtx,
                    4,
                    static_cast<uint32_t>(deleteSid));
                DeleteSema(
                    env.rdram.data(), &deleteSemaCtx, &env.runtime);
                t.Equals(
                    getRegS32(deleteSemaCtx, 2),
                    deleteSid,
                    "DeleteSema should return id zero while waking a waiter");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(), K_SEMA_ORACLE_STAGE_ADDR),
                    2u,
                    "DeleteSema should dispatch a higher-priority waiter before returning");

                EeSemaStatus deletedSemaStatus{};
                t.Equals(
                    referSema(deleteSid, deletedSemaStatus),
                    KE_ERROR,
                    "querying a deleted semaphore should return generic -1");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    getRegS32(env.ctx, 2),
                    40,
                    "post-delete priority boundary should return old main priority");
            }

            const bool deleteDormant = waitUntil([&]()
            {
                EeThreadStatus status{};
                return readGuestU32(
                           env.rdram.data(),
                           K_SEMA_ORACLE_STAGE_ADDR) == 2u &&
                       referThread(deleteTid, status) == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                deleteDormant,
                "deleted-semaphore waiter should complete its host-thread epilogue");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_STAGE_ADDR),
                2u,
                "delete waiter should finish by the forced boundary");
            t.Equals(
                static_cast<int32_t>(readGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_RETURN_ADDR)),
                KE_ERROR,
                "deleted-semaphore WaitSema should return generic -1");
            EeThreadStatus deleteFinalStatus{};
            t.Equals(
                referThread(deleteTid, deleteFinalStatus),
                THS_DORMANT,
                "deleted-semaphore waiter should finish dormant");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                60,
                "restoring main priority after delete should return 60");
            t.Equals(
                deleteWorker(deleteTid),
                deleteTid,
                "deleted-semaphore waiter should be deletable");
        });

        tc.Run("semaphore signal and raw delete preserve EE reschedule boundaries", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00256400u;
            env.runtime.registerFunction(
                kThreadEntry, &semaWaitOracleHandler);

            R5900Context priorityCtx{};
            setRegU32(priorityCtx, 4, 0u);
            setRegU32(priorityCtx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &priorityCtx, &env.runtime);

            const uint32_t semaParam[6] = {
                0u,
                1u,
                0u,
                0u,
                0u,
                0u,
            };
            auto createSema = [&]()
            {
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    semaParam,
                    std::size(semaParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateSema(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto createWorker = [&](uint32_t stack)
            {
                const uint32_t threadParam[9] = {
                    0u,
                    kThreadEntry,
                    stack,
                    0x00001000u,
                    0u,
                    30u,
                    0u,
                    0u,
                    0u,
                };
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    threadParam,
                    std::size(threadParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto referThread =
                [&](int32_t tid, EeThreadStatus &status)
                {
                    R5900Context statusCtx{};
                    setRegU32(
                        statusCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    iReferThreadStatus(
                        env.rdram.data(),
                        &statusCtx,
                        &env.runtime);
                    std::memcpy(
                        &status,
                        env.rdram.data() + K_STATUS_ADDR,
                        sizeof(status));
                    return getRegS32(statusCtx, 2);
                };
            auto startAndWait =
                [&](int32_t tid, int32_t sid)
                {
                    writeGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_STAGE_ADDR,
                        0u);
                    writeGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_RETURN_ADDR,
                        0x7f7f7f7fu);
                    R5900Context startCtx{};
                    setRegU32(
                        startCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(
                        startCtx,
                        5,
                        static_cast<uint32_t>(sid));
                    StartThread(
                        env.rdram.data(), &startCtx, &env.runtime);
                    return waitUntil(
                        [&]()
                        {
                            EeThreadStatus status{};
                            return readGuestU32(
                                       env.rdram.data(),
                                       K_SEMA_ORACLE_STAGE_ADDR) ==
                                       1u &&
                                   referThread(tid, status) ==
                                       THS_WAIT &&
                                   status.status == THS_WAIT &&
                                   status.waitType == TSW_SEMA;
                        },
                        std::chrono::milliseconds(200));
                };
            auto waitForDormant = [&](int32_t tid)
            {
                return waitUntil(
                    [&]()
                    {
                        EeThreadStatus status{};
                        return readGuestU32(
                                   env.rdram.data(),
                                   K_SEMA_ORACLE_STAGE_ADDR) ==
                                   2u &&
                               referThread(tid, status) ==
                                   THS_DORMANT;
                    },
                    std::chrono::milliseconds(200));
            };
            auto restoreMainAndDeleteWorker =
                [&](int32_t tid)
                {
                    R5900Context restoreCtx{};
                    setRegU32(restoreCtx, 4, 0u);
                    setRegU32(restoreCtx, 5, 40u);
                    ChangeThreadPriority(
                        env.rdram.data(),
                        &restoreCtx,
                        &env.runtime);
                    R5900Context deleteCtx{};
                    setRegU32(
                        deleteCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    DeleteThread(
                        env.rdram.data(),
                        &deleteCtx,
                        &env.runtime);
                    return getRegS32(deleteCtx, 2);
                };

            const int32_t rawSignalSid = createSema();
            const int32_t rawSignalTid =
                createWorker(0x0030E000u);
            t.IsTrue(
                startAndWait(rawSignalTid, rawSignalSid),
                "raw-signal worker should block in WaitSema");
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context signalCtx{};
                setRegU32(
                    signalCtx,
                    4,
                    static_cast<uint32_t>(rawSignalSid));
                iSignalSema(
                    env.rdram.data(), &signalCtx, &env.runtime);
                t.Equals(
                    getRegS32(signalCtx, 2),
                    rawSignalSid,
                    "raw signal should return the semaphore id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_STAGE_ADDR),
                    1u,
                    "raw signal should not dispatch its waiter");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_STAGE_ADDR),
                    2u,
                    "the next ordinary boundary should dispatch the raw-signaled waiter");
            }
            t.IsTrue(
                waitForDormant(rawSignalTid),
                "raw-signaled waiter should finish dormant");
            t.Equals(
                static_cast<int32_t>(readGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_RETURN_ADDR)),
                rawSignalSid,
                "raw-signaled WaitSema should return the semaphore id");
            t.Equals(
                restoreMainAndDeleteWorker(rawSignalTid),
                rawSignalTid,
                "raw-signaled worker should be deletable");
            R5900Context deleteRawSignalSemaCtx{};
            setRegU32(
                deleteRawSignalSemaCtx,
                4,
                static_cast<uint32_t>(rawSignalSid));
            DeleteSema(
                env.rdram.data(),
                &deleteRawSignalSemaCtx,
                &env.runtime);

            const int32_t ordinarySignalSid = createSema();
            const int32_t ordinarySignalTid =
                createWorker(0x0030F000u);
            t.IsTrue(
                startAndWait(
                    ordinarySignalTid,
                    ordinarySignalSid),
                "ordinary-signal worker should block in WaitSema");
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context signalCtx{};
                setRegU32(
                    signalCtx,
                    4,
                    static_cast<uint32_t>(ordinarySignalSid));
                SignalSema(
                    env.rdram.data(), &signalCtx, &env.runtime);
                t.Equals(
                    getRegS32(signalCtx, 2),
                    ordinarySignalSid,
                    "ordinary signal should return the semaphore id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_STAGE_ADDR),
                    2u,
                    "ordinary signal should dispatch its higher-priority waiter before returning");
            }
            t.IsTrue(
                waitForDormant(ordinarySignalTid),
                "ordinary-signaled waiter should finish dormant");
            t.Equals(
                restoreMainAndDeleteWorker(
                    ordinarySignalTid),
                ordinarySignalTid,
                "ordinary-signaled worker should be deletable");
            R5900Context deleteOrdinarySignalSemaCtx{};
            setRegU32(
                deleteOrdinarySignalSemaCtx,
                4,
                static_cast<uint32_t>(ordinarySignalSid));
            DeleteSema(
                env.rdram.data(),
                &deleteOrdinarySignalSemaCtx,
                &env.runtime);

            const int32_t rawDeleteSid = createSema();
            const int32_t rawDeleteTid =
                createWorker(0x00310000u);
            t.IsTrue(
                startAndWait(rawDeleteTid, rawDeleteSid),
                "raw-delete worker should block in WaitSema");
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx,
                    4,
                    static_cast<uint32_t>(rawDeleteSid));
                iDeleteSema(
                    env.rdram.data(), &deleteCtx, &env.runtime);
                t.Equals(
                    getRegS32(deleteCtx, 2),
                    rawDeleteSid,
                    "raw semaphore deletion should return its id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_STAGE_ADDR),
                    1u,
                    "raw semaphore deletion should not dispatch its waiter");

                EeThreadStatus rawDeleteStatus{};
                t.Equals(
                    referThread(rawDeleteTid, rawDeleteStatus),
                    THS_READY,
                    "raw semaphore deletion should synchronously publish READY");
                t.Equals(
                    rawDeleteStatus.status,
                    THS_READY,
                    "raw-deleted waiter should be READY before dispatch");
                t.Equals(
                    rawDeleteStatus.waitType,
                    TSW_SEMA,
                    "raw-deleted READY waiter should retain its semaphore wait reason");
                t.Equals(
                    rawDeleteStatus.waitId,
                    static_cast<uint32_t>(rawDeleteSid),
                    "raw-deleted READY waiter should retain its semaphore id");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_STAGE_ADDR),
                    2u,
                    "the next ordinary boundary should dispatch the raw-delete waiter");
            }
            t.IsTrue(
                waitForDormant(rawDeleteTid),
                "raw-delete waiter should finish dormant");
            t.Equals(
                static_cast<int32_t>(readGuestU32(
                    env.rdram.data(), K_SEMA_ORACLE_RETURN_ADDR)),
                KE_ERROR,
                "raw-deleted WaitSema should return generic -1");
            t.Equals(
                restoreMainAndDeleteWorker(rawDeleteTid),
                rawDeleteTid,
                "raw-delete waiter should remain deletable");
        });

        tc.Run("priority change and suspend resume preserve EE reschedule boundaries", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kMarkerEntry = 0x00256800u;
            constexpr uint32_t kSleeperEntry = 0x00256C00u;
            env.runtime.registerFunction(
                kMarkerEntry, &rescheduleMarkerHandler);
            env.runtime.registerFunction(
                kSleeperEntry, &waitSuspendSleepHandler);

            R5900Context mainPriorityCtx{};
            setRegU32(mainPriorityCtx, 4, 0u);
            setRegU32(mainPriorityCtx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &mainPriorityCtx, &env.runtime);
            t.Equals(
                getRegS32(mainPriorityCtx, 2),
                0,
                "main should begin the reschedule matrix at priority 40");

            auto createWorker =
                [&](uint32_t entry,
                    uint32_t stack,
                    uint32_t priority)
                {
                    const uint32_t threadParam[9] = {
                        0u,
                        entry,
                        stack,
                        0x00000800u,
                        0u,
                        priority,
                        0u,
                        0u,
                        0u,
                    };
                    writeGuestWords(
                        env.rdram.data(),
                        K_PARAM_ADDR,
                        threadParam,
                        std::size(threadParam));
                    R5900Context createCtx{};
                    setRegU32(createCtx, 4, K_PARAM_ADDR);
                    CreateThread(
                        env.rdram.data(),
                        &createCtx,
                        &env.runtime);
                    return getRegS32(createCtx, 2);
                };
            auto startWorker = [&](int32_t tid)
            {
                R5900Context startCtx{};
                setRegU32(
                    startCtx,
                    4,
                    static_cast<uint32_t>(tid));
                setRegU32(startCtx, 5, 0u);
                StartThread(
                    env.rdram.data(), &startCtx, &env.runtime);
                return getRegS32(startCtx, 2);
            };
            auto referThread =
                [&](int32_t tid, EeThreadStatus &status)
                {
                    R5900Context statusCtx{};
                    setRegU32(
                        statusCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    iReferThreadStatus(
                        env.rdram.data(),
                        &statusCtx,
                        &env.runtime);
                    std::memcpy(
                        &status,
                        env.rdram.data() + K_STATUS_ADDR,
                        sizeof(status));
                    return getRegS32(statusCtx, 2);
                };
            auto waitForDormant = [&](int32_t tid)
            {
                return waitUntil(
                    [&]()
                    {
                        EeThreadStatus status{};
                        return referThread(tid, status) ==
                               THS_DORMANT;
                    },
                    std::chrono::milliseconds(200));
            };
            auto deleteWorker = [&](int32_t tid)
            {
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx,
                    4,
                    static_cast<uint32_t>(tid));
                DeleteThread(
                    env.rdram.data(), &deleteCtx, &env.runtime);
                return getRegS32(deleteCtx, 2);
            };
            auto restoreMainPriority = [&]()
            {
                R5900Context restoreCtx{};
                setRegU32(restoreCtx, 4, 0u);
                setRegU32(restoreCtx, 5, 40u);
                ChangeThreadPriority(
                    env.rdram.data(),
                    &restoreCtx,
                    &env.runtime);
                return getRegS32(restoreCtx, 2);
            };

            const int32_t rawPriorityTid =
                createWorker(
                    kMarkerEntry, 0x00311000u, 50u);
            writeGuestU32(
                env.rdram.data(), K_RESCHEDULE_STAGE_ADDR, 0u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                t.Equals(
                    startWorker(rawPriorityTid),
                    rawPriorityTid,
                    "raw-priority worker should start READY");

                R5900Context rawPriorityCtx{};
                setRegU32(
                    rawPriorityCtx,
                    4,
                    static_cast<uint32_t>(rawPriorityTid));
                setRegU32(rawPriorityCtx, 5, 30u);
                iChangeThreadPriority(
                    env.rdram.data(),
                    &rawPriorityCtx,
                    &env.runtime);
                t.Equals(
                    getRegS32(rawPriorityCtx, 2),
                    50,
                    "raw priority change should return the previous priority");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    0u,
                    "raw priority change should not dispatch its promoted target");

                EeThreadStatus promotedStatus{};
                referThread(rawPriorityTid, promotedStatus);
                t.Equals(
                    promotedStatus.current_priority,
                    30,
                    "raw priority change should publish the new priority synchronously");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    getRegS32(env.ctx, 2),
                    40,
                    "ordinary main priority change should return its old priority");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    1u,
                    "the next ordinary boundary should dispatch the raw-promoted target");
            }
            t.IsTrue(
                waitForDormant(rawPriorityTid),
                "raw-promoted target should finish dormant");
            t.Equals(
                restoreMainPriority(),
                60,
                "main priority should restore after raw promotion");
            t.Equals(
                deleteWorker(rawPriorityTid),
                rawPriorityTid,
                "raw-promoted target should be deletable");

            const int32_t ordinaryPriorityTid =
                createWorker(
                    kMarkerEntry, 0x00312000u, 50u);
            writeGuestU32(
                env.rdram.data(), K_RESCHEDULE_STAGE_ADDR, 0u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                t.Equals(
                    startWorker(ordinaryPriorityTid),
                    ordinaryPriorityTid,
                    "ordinary-priority worker should start READY");

                R5900Context ordinaryPriorityCtx{};
                setRegU32(
                    ordinaryPriorityCtx,
                    4,
                    static_cast<uint32_t>(
                        ordinaryPriorityTid));
                setRegU32(ordinaryPriorityCtx, 5, 30u);
                ChangeThreadPriority(
                    env.rdram.data(),
                    &ordinaryPriorityCtx,
                    &env.runtime);
                t.Equals(
                    getRegS32(ordinaryPriorityCtx, 2),
                    50,
                    "ordinary priority change should return the previous priority");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    1u,
                    "ordinary priority change should dispatch its promoted target before returning");
            }
            t.IsTrue(
                waitForDormant(ordinaryPriorityTid),
                "ordinary-promoted target should finish dormant");
            t.Equals(
                deleteWorker(ordinaryPriorityTid),
                ordinaryPriorityTid,
                "ordinary-promoted target should be deletable");

            auto waitForSuspended = [&](int32_t tid)
            {
                return waitUntil(
                    [&]()
                    {
                        EeThreadStatus status{};
                        return referThread(tid, status) ==
                                   THS_SUSPEND &&
                               status.status == THS_SUSPEND;
                    },
                    std::chrono::milliseconds(200));
            };
            auto suspendAndPromote =
                [&](int32_t tid)
                {
                    R5900Context suspendCtx{};
                    setRegU32(
                        suspendCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    SuspendThread(
                        env.rdram.data(),
                        &suspendCtx,
                        &env.runtime);
                    t.Equals(
                        getRegS32(suspendCtx, 2),
                        tid,
                        "SuspendThread should return its target id");
                    t.IsTrue(
                        waitForSuspended(tid),
                        "target should settle in SUSPEND");

                    R5900Context priorityCtx{};
                    setRegU32(
                        priorityCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(priorityCtx, 5, 30u);
                    iChangeThreadPriority(
                        env.rdram.data(),
                        &priorityCtx,
                        &env.runtime);
                    t.Equals(
                        getRegS32(priorityCtx, 2),
                        50,
                        "suspended target promotion should return priority 50");
                };

            const int32_t rawResumeTid =
                createWorker(
                    kMarkerEntry, 0x00313000u, 50u);
            writeGuestU32(
                env.rdram.data(), K_RESCHEDULE_STAGE_ADDR, 0u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                t.Equals(
                    startWorker(rawResumeTid),
                    rawResumeTid,
                    "raw-resume worker should start");
                suspendAndPromote(rawResumeTid);

                R5900Context resumeCtx{};
                setRegU32(
                    resumeCtx,
                    4,
                    static_cast<uint32_t>(rawResumeTid));
                iResumeThread(
                    env.rdram.data(), &resumeCtx, &env.runtime);
                t.Equals(
                    getRegS32(resumeCtx, 2),
                    rawResumeTid,
                    "raw resume should return its target id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    0u,
                    "raw resume should not dispatch its promoted target");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    1u,
                    "the next ordinary boundary should dispatch the raw-resumed target");
            }
            t.IsTrue(
                waitForDormant(rawResumeTid),
                "raw-resumed target should finish dormant");
            t.Equals(
                restoreMainPriority(),
                60,
                "main priority should restore after raw resume");
            t.Equals(
                deleteWorker(rawResumeTid),
                rawResumeTid,
                "raw-resumed target should be deletable");

            const int32_t ordinaryResumeTid =
                createWorker(
                    kMarkerEntry, 0x00314000u, 50u);
            writeGuestU32(
                env.rdram.data(), K_RESCHEDULE_STAGE_ADDR, 0u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                t.Equals(
                    startWorker(ordinaryResumeTid),
                    ordinaryResumeTid,
                    "ordinary-resume worker should start");
                suspendAndPromote(ordinaryResumeTid);

                R5900Context resumeCtx{};
                setRegU32(
                    resumeCtx,
                    4,
                    static_cast<uint32_t>(
                        ordinaryResumeTid));
                ResumeThread(
                    env.rdram.data(), &resumeCtx, &env.runtime);
                t.Equals(
                    getRegS32(resumeCtx, 2),
                    ordinaryResumeTid,
                    "ordinary resume should return its target id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    1u,
                    "ordinary resume should dispatch its promoted target before returning");
            }
            t.IsTrue(
                waitForDormant(ordinaryResumeTid),
                "ordinary-resumed target should finish dormant");
            t.Equals(
                deleteWorker(ordinaryResumeTid),
                ordinaryResumeTid,
                "ordinary-resumed target should be deletable");

            const int32_t sleeperTid =
                createWorker(
                    kSleeperEntry, 0x00315000u, 30u);
            writeGuestU32(
                env.rdram.data(), K_WAITSUSPEND_STAGE_ADDR, 0u);
            t.Equals(
                startWorker(sleeperTid),
                sleeperTid,
                "raw-suspend candidate should start");
            t.IsTrue(
                waitUntil(
                    [&]()
                    {
                        EeThreadStatus status{};
                        return readGuestU32(
                                   env.rdram.data(),
                                   K_WAITSUSPEND_STAGE_ADDR) ==
                                   1u &&
                               referThread(sleeperTid, status) ==
                                   THS_WAIT &&
                               status.waitType == TSW_SLEEP;
                    },
                    std::chrono::milliseconds(200)),
                "raw-suspend candidate should block in SleepThread");

            const int32_t rawSuspendTid =
                createWorker(
                    kMarkerEntry, 0x00316000u, 50u);
            writeGuestU32(
                env.rdram.data(), K_RESCHEDULE_STAGE_ADDR, 0u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                t.Equals(
                    startWorker(rawSuspendTid),
                    rawSuspendTid,
                    "raw-suspend target should start");

                R5900Context wakeCtx{};
                setRegU32(
                    wakeCtx,
                    4,
                    static_cast<uint32_t>(sleeperTid));
                iWakeupThread(
                    env.rdram.data(), &wakeCtx, &env.runtime);
                t.Equals(
                    getRegS32(wakeCtx, 2),
                    sleeperTid,
                    "raw wake should make the scheduling candidate READY");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_WAITSUSPEND_STAGE_ADDR),
                    1u,
                    "raw wake should defer the scheduling candidate");

                R5900Context suspendCtx{};
                setRegU32(
                    suspendCtx,
                    4,
                    static_cast<uint32_t>(rawSuspendTid));
                t.IsTrue(
                    callSyscall(
                        static_cast<uint32_t>(-0x38),
                        env.rdram.data(),
                        &suspendCtx,
                        &env.runtime),
                    "raw suspend syscall should dispatch");
                t.Equals(
                    getRegS32(suspendCtx, 2),
                    rawSuspendTid,
                    "raw suspend should return its target id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_WAITSUSPEND_STAGE_ADDR),
                    1u,
                    "raw suspend should not dispatch an unrelated READY candidate");
                t.IsTrue(
                    waitForSuspended(rawSuspendTid),
                    "raw suspend target should settle in SUSPEND");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 60u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_WAITSUSPEND_STAGE_ADDR),
                    2u,
                    "the next ordinary boundary should dispatch the raw-woken candidate");

                R5900Context resumeCtx{};
                setRegU32(
                    resumeCtx,
                    4,
                    static_cast<uint32_t>(rawSuspendTid));
                iResumeThread(
                    env.rdram.data(), &resumeCtx, &env.runtime);
                t.Equals(
                    getRegS32(resumeCtx, 2),
                    rawSuspendTid,
                    "raw resume should return the raw-suspended target");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    0u,
                    "raw resume should keep its own dispatch deferred");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 70u);
                ChangeThreadPriority(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_RESCHEDULE_STAGE_ADDR),
                    1u,
                    "the next ordinary boundary should dispatch the raw-resumed target");
            }
            t.IsTrue(
                waitForDormant(sleeperTid),
                "raw-woken candidate should finish dormant");
            t.IsTrue(
                waitForDormant(rawSuspendTid),
                "raw-suspended and resumed target should finish dormant");
            t.Equals(
                restoreMainPriority(),
                70,
                "main priority should restore after raw suspend");
            t.Equals(
                deleteWorker(sleeperTid),
                sleeperTid,
                "raw-woken candidate should be deletable");
            t.Equals(
                deleteWorker(rawSuspendTid),
                rawSuspendTid,
                "raw-suspended target should be deletable");
        });

        tc.Run("semaphore EE layout covers poll, signal overflow, and status", [](TestCase &t)
        {
            TestEnv env;

            const uint32_t semaParam[6] = {
                0u,          // count (unused by runtime decode)
                2u,          // max_count
                1u,          // init_count
                0u,          // wait_threads
                0x11u,       // attr
                0x00202020u  // option
            };

            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t sid = getRegS32(env.ctx, 2);
            t.IsTrue(sid >= 0, "CreateSema should return a valid nonnegative semaphore id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferSemaStatus(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "ReferSemaStatus should succeed for valid semaphore");

            EeSemaStatus semaStatus{};
            std::memcpy(&semaStatus, env.rdram.data() + K_STATUS_ADDR, sizeof(semaStatus));
            t.Equals(semaStatus.count, 1, "initial semaphore count should match init_count");
            t.Equals(semaStatus.max_count, 2, "max_count should match CreateSema params");
            t.Equals(semaStatus.init_count, 1, "init_count should be preserved");
            t.Equals(semaStatus.attr, semaParam[4], "attr should be preserved");
            t.Equals(semaStatus.option, semaParam[5], "option should be preserved");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            PollSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), sid, "PollSema should return sid when consuming one available token");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            PollSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_SEMA_ZERO, "PollSema should fail when count is zero");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), sid, "SignalSema should return sid when incrementing count below max");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), sid, "SignalSema should return sid when incrementing up to max");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_SEMA_OVF, "SignalSema should report overflow at max_count");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            DeleteSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), sid, "DeleteSema should return sid for existing semaphore");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            PollSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_SEMID, "deleted semaphore id should be rejected");
        });

        tc.Run("semaphore legacy layout decode remains supported", [](TestCase &t)
        {
            TestEnv env;

            const uint32_t legacyParam[6] = {
                0x7u,        // attr
                0x1234u,     // legacy option / ee max_count
                3u,          // init
                4u,          // max
                0u,          // ee attr (ignored if legacy selected)
                0x1FFFFFFFu  // ee option (invalid guest pointer to bias decode toward legacy)
            };
            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, legacyParam, std::size(legacyParam));

            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t sid = getRegS32(env.ctx, 2);
            t.IsTrue(sid >= 0, "CreateSema should still accept legacy-style parameter blocks");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            setRegU32(env.ctx, 5, K_STATUS_ADDR);
            ReferSemaStatus(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "ReferSemaStatus should succeed for legacy-decoded semaphore");

            EeSemaStatus semaStatus{};
            std::memcpy(&semaStatus, env.rdram.data() + K_STATUS_ADDR, sizeof(semaStatus));
            t.Equals(semaStatus.count, 3, "legacy init_count should map to runtime count");
            t.Equals(semaStatus.max_count, 4, "legacy max_count should map to runtime max");
            t.Equals(semaStatus.attr, 0x7u, "legacy attr should be preserved");
            t.Equals(semaStatus.option, 0x1234u, "legacy option should be preserved");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            DeleteSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), sid, "DeleteSema should return sid for legacy-decoded semaphore");
        });

        tc.Run("semaphore syscalls return sid on success (EE BIOS convention)", [](TestCase &t)
        {
            // Sub-case A: CreateSema returns a valid nonnegative id.
            {
                TestEnv env;
                const uint32_t semaParam[6] = { 0u, 2u, 1u, 0u, 0x11u, 0u };
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int32_t sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "CreateSema should return a valid semaphore id");
            }

            // Sub-case B: PollSema success returns sid
            {
                TestEnv env;
                const uint32_t semaParam[6] = { 0u, 2u, 1u, 0u, 0u, 0u };
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int32_t sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "CreateSema should return a valid id for PollSema test");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                PollSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid, "PollSema success should return sid");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                PollSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_SEMA_ZERO, "PollSema should return KE_SEMA_ZERO when count exhausted");
            }

            // Sub-case C: SignalSema success returns sid + overflow returns KE_SEMA_OVF
            {
                TestEnv env;
                const uint32_t semaParam[6] = { 0u, 1u, 0u, 0u, 0u, 0u };
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int32_t sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "CreateSema should return a valid id for SignalSema test");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid, "SignalSema success should return sid (count 0->1)");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_SEMA_OVF, "SignalSema should return KE_SEMA_OVF when count at max=1");
            }

            // Sub-case D: WaitSema success returns sid AND decrements count
            {
                TestEnv env;
                const uint32_t semaParam[6] = { 0u, 2u, 1u, 0u, 0u, 0u };
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int32_t sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "CreateSema should return a valid id for WaitSema test");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                WaitSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid, "WaitSema success should return sid");

                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(sid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferSemaStatus(env.rdram.data(), &statusCtx, &env.runtime);
                EeSemaStatus semaStatus{};
                std::memcpy(&semaStatus, env.rdram.data() + K_STATUS_ADDR, sizeof(semaStatus));
                t.Equals(semaStatus.count, 0, "WaitSema should decrement count to 0");
            }

            // Sub-case E: deleting a semaphore makes its waiter return generic -1.
            {
                TestEnv env;
                const uint32_t semaParam[6] = { 0u, 1u, 0u, 0u, 0u, 0u };
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int32_t sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "CreateSema should return a valid id for delete-while-waiting test");

                int32_t workerRet = 0;
                writeGuestU32(env.rdram.data(), K_SEMA_WAIT_READY_ADDR, 0u);

                std::thread worker([&]()
                {
                    R5900Context wctx{};
                    setRegU32(wctx, 4, static_cast<uint32_t>(sid));
                    writeGuestU32(env.rdram.data(), K_SEMA_WAIT_READY_ADDR, 1u);
                    WaitSema(env.rdram.data(), &wctx, &env.runtime);
                    workerRet = getRegS32(wctx, 2);
                });

                // Wait until the waiter has incremented waiter count (count=0, so it must block)
                const bool waiterBlocking = waitUntil([&]()
                {
                    R5900Context statusCtx{};
                    setRegU32(statusCtx, 4, static_cast<uint32_t>(sid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferSemaStatus(env.rdram.data(), &statusCtx, &env.runtime);
                    EeSemaStatus st{};
                    std::memcpy(&st, env.rdram.data() + K_STATUS_ADDR, sizeof(st));
                    return st.wait_threads >= 1;
                }, std::chrono::milliseconds(500));
                t.IsTrue(waiterBlocking, "worker thread should be blocking on WaitSema");

                // Delete the semaphore while worker is waiting
                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                DeleteSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid, "DeleteSema should return sid while thread is waiting");

                worker.join();
                t.Equals(workerRet, KE_ERROR, "WaitSema should return generic -1 when its semaphore is deleted");
            }

            // Sub-case F: DeleteSema success returns sid
            {
                TestEnv env;
                const uint32_t semaParam[6] = { 0u, 1u, 0u, 0u, 0u, 0u };
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int32_t sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "CreateSema should return a valid id for DeleteSema test");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                DeleteSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid, "DeleteSema success should return sid");
            }

            // Sub-case G: Invalid sid returns KE_UNKNOWN_SEMID for all four syscalls
            {
                TestEnv env;
                constexpr uint32_t kBadSid = 0x7FFFu;

                setRegU32(env.ctx, 4, kBadSid);
                PollSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_SEMID, "PollSema should return KE_UNKNOWN_SEMID for invalid sid");

                setRegU32(env.ctx, 4, kBadSid);
                SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_SEMID, "SignalSema should return KE_UNKNOWN_SEMID for invalid sid");

                setRegU32(env.ctx, 4, kBadSid);
                WaitSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_SEMID, "WaitSema should return KE_UNKNOWN_SEMID for invalid sid");

                setRegU32(env.ctx, 4, kBadSid);
                DeleteSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_SEMID, "DeleteSema should return KE_UNKNOWN_SEMID for invalid sid");
            }

            // Sub-case H: WaitSema force-released via ReleaseWaitThread returns generic -1
            // and the ret >= 0 guard must NOT consume a token (count stays 0, not -1).
            {
                TestEnv env;
                const uint32_t semaParam[6] = {0u, 2u, 0u, 0u, 0u, 0u};
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, 6);
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "sub-case H: CreateSema must return a valid sid");

                constexpr uint32_t kWorkerEntry = 0x00260600u;
                env.runtime.registerFunction(
                    kWorkerEntry, &semaWaitOracleHandler);
                writeGuestU32(
                    env.rdram.data(),
                    K_SEMA_ORACLE_STAGE_ADDR,
                    0u);
                writeGuestU32(
                    env.rdram.data(),
                    K_SEMA_ORACLE_RETURN_ADDR,
                    0u);
                const uint32_t threadParam[9] = {
                    0u,
                    kWorkerEntry,
                    0x0031A000u,
                    0x00000800u,
                    0u,
                    30u,
                    0u,
                    0u,
                    0u,
                };
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    threadParam,
                    std::size(threadParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(),
                    &createCtx,
                    &env.runtime);
                const int32_t workerTid =
                    getRegS32(createCtx, 2);
                t.IsTrue(
                    workerTid >= 2,
                    "sub-case H: CreateThread must return a worker id");

                R5900Context startCtx{};
                setRegU32(
                    startCtx,
                    4,
                    static_cast<uint32_t>(workerTid));
                setRegU32(
                    startCtx,
                    5,
                    static_cast<uint32_t>(sid));
                StartThread(
                    env.rdram.data(),
                    &startCtx,
                    &env.runtime);
                t.Equals(
                    getRegS32(startCtx, 2),
                    workerTid,
                    "sub-case H: StartThread must launch the waiter");

                // Wait until the worker is confirmed blocking in WaitSema.
                const bool waiterBlocking = waitUntil([&]() {
                    R5900Context statusCtx{};
                    setRegU32(statusCtx, 4, static_cast<uint32_t>(sid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferSemaStatus(env.rdram.data(), &statusCtx, &env.runtime);
                    EeSemaStatus st{};
                    std::memcpy(&st, env.rdram.data() + K_STATUS_ADDR, sizeof(st));
                    return st.wait_threads >= 1;
                }, std::chrono::milliseconds(500));
                t.IsTrue(waiterBlocking, "sub-case H: worker must be blocking in WaitSema before force-release");

                // Force-release the worker via ReleaseWaitThread.
                setRegU32(
                    env.ctx,
                    4,
                    static_cast<uint32_t>(workerTid));
                ReleaseWaitThread(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), workerTid,
                         "sub-case H: ReleaseWaitThread must return the waiter id");

                const bool workerFinished = waitUntil([&]()
                {
                    return readGuestU32(
                               env.rdram.data(),
                               K_SEMA_ORACLE_STAGE_ADDR) == 2u;
                }, std::chrono::milliseconds(500));
                t.IsTrue(
                    workerFinished,
                    "sub-case H: force-released waiter must finish");
                const int32_t workerRet =
                    static_cast<int32_t>(readGuestU32(
                        env.rdram.data(),
                        K_SEMA_ORACLE_RETURN_ADDR));
                t.Equals(workerRet, KE_ERROR,
                         "sub-case H: force-released WaitSema must return generic -1");

                // Assert the count was NOT decremented (core guard check: ret < 0 skips decrement).
                {
                    R5900Context statusCtx{};
                    setRegU32(statusCtx, 4, static_cast<uint32_t>(sid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferSemaStatus(env.rdram.data(), &statusCtx, &env.runtime);
                    EeSemaStatus st{};
                    std::memcpy(&st, env.rdram.data() + K_STATUS_ADDR, sizeof(st));
                    t.Equals(st.count, 0, "sub-case H: force-released WaitSema must NOT consume a token (count must stay 0, not -1)");
                }

                // Prove token accounting is intact: signal once, poll twice (one token, clean).
                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid,
                         "sub-case H: SignalSema after force-release must return sid (count 0->1)");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                PollSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid,
                         "sub-case H: PollSema must consume the one token after force-release");

                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                PollSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), KE_SEMA_ZERO,
                         "sub-case H: count must be exactly 0 after single token consumed (not -1)");
            }

            // Sub-case I: blocking WaitSema woken by SignalSema returns sid (the DQ8 scenario).
            // init=0 forces the worker to block; SignalSema uses cv.notify_one().
            {
                TestEnv env;
                const uint32_t semaParam[6] = {0u, 1u, 0u, 0u, 0u, 0u};
                writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, 6);
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateSema(env.rdram.data(), &env.ctx, &env.runtime);
                const int sid = getRegS32(env.ctx, 2);
                t.IsTrue(sid >= 0, "sub-case I: CreateSema must return a valid sid");

                writeGuestU32(env.rdram.data(), K_SEMA_WAIT_READY_ADDR, 0u);
                int32_t workerRet = 0;

                std::thread worker([&]() {
                    R5900Context wctx{};
                    setRegU32(wctx, 4, static_cast<uint32_t>(sid));
                    writeGuestU32(env.rdram.data(), K_SEMA_WAIT_READY_ADDR, 1u);
                    WaitSema(env.rdram.data(), &wctx, &env.runtime);
                    workerRet = getRegS32(wctx, 2);
                });

                // Confirm the worker is actually blocking (count==0 forces a block).
                const bool waiterBlocking = waitUntil([&]() {
                    R5900Context statusCtx{};
                    setRegU32(statusCtx, 4, static_cast<uint32_t>(sid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferSemaStatus(env.rdram.data(), &statusCtx, &env.runtime);
                    EeSemaStatus st{};
                    std::memcpy(&st, env.rdram.data() + K_STATUS_ADDR, sizeof(st));
                    return st.wait_threads >= 1;
                }, std::chrono::milliseconds(500));
                t.IsTrue(waiterBlocking, "sub-case I: worker must be blocking in WaitSema before signal");

                // Wake the worker; success path must return sid, not KE_OK.
                setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
                SignalSema(env.rdram.data(), &env.ctx, &env.runtime);
                t.Equals(getRegS32(env.ctx, 2), sid,
                         "sub-case I: SignalSema that wakes a waiter must return sid (count 0->1)");

                worker.join();
                t.Equals(workerRet, sid,
                         "sub-case I: blocking WaitSema woken by signal must return sid, not KE_OK");

                // Signal incremented to 1, the woken wait consumed it back to 0.
                {
                    R5900Context statusCtx{};
                    setRegU32(statusCtx, 4, static_cast<uint32_t>(sid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferSemaStatus(env.rdram.data(), &statusCtx, &env.runtime);
                    EeSemaStatus st{};
                    std::memcpy(&st, env.rdram.data() + K_STATUS_ADDR, sizeof(st));
                    t.Equals(st.count, 0,
                             "sub-case I: woken WaitSema must consume the signaled token (count back to 0)");
                }
            }

            // Reset all global sema/thread state so no entries (e.g. the 0x7FFE ThreadInfo
            // from sub-case H) leak into subsequent test cases.
            notifyRuntimeStop();
        });

        tc.Run("WaitEventFlag preserves waitsuspend state when a suspended thread blocks", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kEventParamAddr = 0x1600u;
            constexpr uint32_t kWaitThreadEntry = 0x00260000u;

            const uint32_t eventParam[3] = {
                0u,
                0u,
                0u
            };
            std::memcpy(env.rdram.data() + kEventParamAddr, eventParam, sizeof(eventParam));
            writeGuestU32(env.rdram.data(), K_EVENT_WAIT_READY_ADDR, 0u);
            writeGuestU32(env.rdram.data(), K_EVENT_WAIT_GATE_ADDR, 0u);

            R5900Context createEventCtx{};
            setRegU32(createEventCtx, 4, kEventParamAddr);
            CreateEventFlag(env.rdram.data(), &createEventCtx, &env.runtime);
            const int32_t eid = getRegS32(createEventCtx, 2);
            t.IsTrue(eid > 0, "CreateEventFlag should return a valid event id");

            env.runtime.registerFunction(kWaitThreadEntry, &waitEventAfterSuspendHandler);

            const uint32_t threadParam[7] = {
                0u,
                kWaitThreadEntry,
                0x00310000u,
                0x00000800u,
                0x00120000u,
                6u,
                0u
            };

            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, threadParam, std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(tid >= 2, "CreateThread should return a valid worker thread id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, static_cast<uint32_t>(eid));
            StartThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "StartThread should return the event waiter id");

            const bool ready = waitUntil([&]()
            {
                return readGuestU32(env.rdram.data(), K_EVENT_WAIT_READY_ADDR) == 1u;
            }, std::chrono::milliseconds(200));
            t.IsTrue(ready, "waiter thread should reach the suspend gate before blocking");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            SuspendThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "SuspendThread should return the running waiter's id");

            writeGuestU32(env.rdram.data(), K_EVENT_WAIT_GATE_ADDR, 1u);

            const bool waiting = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(&status, env.rdram.data() + K_STATUS_ADDR, sizeof(status));
                return getRegS32(statusCtx, 2) == status.status &&
                       status.waitType == TSW_EVENT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(waiting, "waiter thread should block on the event flag");

            EeThreadStatus waitingStatus{};
            std::memcpy(&waitingStatus, env.rdram.data() + K_STATUS_ADDR, sizeof(waitingStatus));
            t.Equals(waitingStatus.status, THS_WAITSUSPEND,
                     "event-flag wait should report THS_WAITSUSPEND when the thread is already suspended");

            R5900Context signalCtx{};
            setRegU32(signalCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(signalCtx, 5, 0x4u);
            SetEventFlag(env.rdram.data(), &signalCtx, &env.runtime);
            t.Equals(getRegS32(signalCtx, 2), KE_OK, "SetEventFlag should wake the waiting thread");

            const bool suspended = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(&status, env.rdram.data() + K_STATUS_ADDR, sizeof(status));
                return getRegS32(statusCtx, 2) == status.status &&
                       status.status == THS_SUSPEND &&
                       status.waitType == 0u;
            }, std::chrono::milliseconds(200));
            t.IsTrue(suspended, "after wake, a still-suspended waiter should move to THS_SUSPEND");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            ResumeThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "ResumeThread should return the released waiter's id");

            const bool dormant = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(&status, env.rdram.data() + K_STATUS_ADDR, sizeof(status));
                return getRegS32(statusCtx, 2) == status.status &&
                       status.status == THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(dormant, "waiter thread should return to dormant after the event is signaled and resumed");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(eid));
            DeleteEventFlag(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "DeleteEventFlag should clean up the test event flag");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "DeleteThread should return the waiter thread id");
        });

        tc.Run("terminate variants preserve the EE raw scheduling boundary", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kCandidateEntry = 0x00260800u;
            constexpr uint32_t kTargetEntry = 0x00260900u;
            env.runtime.registerFunction(
                kCandidateEntry,
                &terminateCandidateSleepHandler);
            env.runtime.registerFunction(
                kTargetEntry,
                &terminateReadyTargetHandler);

            auto createThread = [&](uint32_t entry,
                                    uint32_t stack,
                                    uint32_t priority)
            {
                const uint32_t threadParam[9] = {
                    0u,
                    entry,
                    stack,
                    0x00001000u,
                    0u,
                    priority,
                    0u,
                    0u,
                    0u
                };
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    threadParam,
                    std::size(threadParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto startThread = [&](int32_t tid)
            {
                R5900Context startCtx{};
                setRegU32(
                    startCtx, 4, static_cast<uint32_t>(tid));
                StartThread(
                    env.rdram.data(), &startCtx, &env.runtime);
                return getRegS32(startCtx, 2);
            };
            auto referThread = [&](int32_t tid, EeThreadStatus &status)
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return getRegS32(statusCtx, 2);
            };
            auto deleteThread = [&](int32_t tid)
            {
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx, 4, static_cast<uint32_t>(tid));
                DeleteThread(
                    env.rdram.data(), &deleteCtx, &env.runtime);
                return getRegS32(deleteCtx, 2);
            };

            auto runCase = [&](bool raw,
                               uint32_t candidateStack,
                               uint32_t targetStack)
            {
                writeGuestU32(
                    env.rdram.data(),
                    K_TERMINATE_CANDIDATE_STAGE_ADDR,
                    0u);
                writeGuestU32(
                    env.rdram.data(),
                    K_TERMINATE_TARGET_STAGE_ADDR,
                    0u);

                const int32_t candidateTid =
                    createThread(
                        kCandidateEntry,
                        candidateStack,
                        30u);
                t.IsTrue(
                    candidateTid >= 2,
                    "candidate creation should return a guest thread id");
                t.Equals(
                    startThread(candidateTid),
                    candidateTid,
                    "candidate start should return its thread id");

                const bool candidateSleeping = waitUntil([&]()
                {
                    EeThreadStatus status{};
                    return readGuestU32(
                               env.rdram.data(),
                               K_TERMINATE_CANDIDATE_STAGE_ADDR) == 1u &&
                           referThread(candidateTid, status) == THS_WAIT &&
                           status.status == THS_WAIT &&
                           status.waitType == TSW_SLEEP;
                }, std::chrono::milliseconds(200));
                t.IsTrue(
                    candidateSleeping,
                    "the higher-priority candidate should block in SleepThread");

                const int32_t targetTid =
                    createThread(
                        kTargetEntry,
                        targetStack,
                        50u);
                t.IsTrue(
                    targetTid >= 2,
                    "termination target creation should return a guest thread id");

                {
                    PS2Runtime::GuestExecutionScope guestExecution(
                        &env.runtime, &env.ctx);

                    t.Equals(
                        startThread(targetTid),
                        targetTid,
                        "termination target start should return its thread id");

                    R5900Context wakeCtx{};
                    setRegU32(
                        wakeCtx, 4, static_cast<uint32_t>(candidateTid));
                    iWakeupThread(
                        env.rdram.data(), &wakeCtx, &env.runtime);
                    t.Equals(
                        getRegS32(wakeCtx, 2),
                        candidateTid,
                        "raw wake should publish the higher-priority candidate");
                    t.Equals(
                        readGuestU32(
                            env.rdram.data(),
                            K_TERMINATE_CANDIDATE_STAGE_ADDR),
                        1u,
                        "raw wake should not dispatch the candidate");

                    R5900Context terminateCtx{};
                    setRegU32(
                        terminateCtx, 4, static_cast<uint32_t>(targetTid));
                    if (raw)
                    {
                        t.IsTrue(
                            callSyscall(
                                static_cast<uint32_t>(-0x26),
                                env.rdram.data(),
                                &terminateCtx,
                                &env.runtime),
                            "raw terminate syscall should dispatch");
                    }
                    else
                    {
                        TerminateThread(
                            env.rdram.data(),
                            &terminateCtx,
                            &env.runtime);
                    }

                    t.Equals(
                        getRegS32(terminateCtx, 2),
                        targetTid,
                        "successful termination should return the target id");
                    t.Equals(
                        readGuestU32(
                            env.rdram.data(),
                            K_TERMINATE_CANDIDATE_STAGE_ADDR),
                        raw ? 1u : 2u,
                        raw
                            ? "raw termination should not dispatch a READY candidate"
                            : "ordinary termination should dispatch a READY candidate before returning");
                    t.Equals(
                        readGuestU32(
                            env.rdram.data(),
                            K_TERMINATE_TARGET_STAGE_ADDR),
                        0u,
                        "the terminated READY target must not execute its entry");

                    EeThreadStatus targetStatus{};
                    t.Equals(
                        referThread(targetTid, targetStatus),
                        THS_DORMANT,
                        "terminated target status lookup should return DORMANT");
                    t.Equals(
                        targetStatus.status,
                        THS_DORMANT,
                        "terminated target should synchronously publish DORMANT");
                }

                const bool candidateExited = waitUntil([&]()
                {
                    EeThreadStatus status{};
                    return readGuestU32(
                               env.rdram.data(),
                               K_TERMINATE_CANDIDATE_STAGE_ADDR) == 2u &&
                           referThread(candidateTid, status) == THS_DORMANT;
                }, std::chrono::milliseconds(200));
                t.IsTrue(
                    candidateExited,
                    "released scheduling candidate should exit to DORMANT");
                t.Equals(
                    deleteThread(candidateTid),
                    candidateTid,
                    "candidate should be deletable after exit");
                t.Equals(
                    deleteThread(targetTid),
                    targetTid,
                    "terminated target should be immediately deletable");
            };

            runCase(true, 0x00313000u, 0x00314000u);
            runCase(false, 0x00315000u, 0x00316000u);
            notifyRuntimeStop();
        });

        tc.Run("self exit preserves or deletes the EE thread object", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kExitEntry = 0x00260A00u;
            constexpr uint32_t kExitDeleteEntry = 0x00260B00u;
            env.runtime.registerFunction(
                kExitEntry,
                &selfExitThreadHandler);
            env.runtime.registerFunction(
                kExitDeleteEntry,
                &selfExitDeleteThreadHandler);

            auto createThread = [&](uint32_t entry, uint32_t stack)
            {
                const uint32_t threadParam[9] = {
                    0u,
                    entry,
                    stack,
                    0x00001000u,
                    0u,
                    30u,
                    0u,
                    0u,
                    0u
                };
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    threadParam,
                    std::size(threadParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(), &createCtx, &env.runtime);
                return getRegS32(createCtx, 2);
            };
            auto startThread = [&](int32_t tid)
            {
                R5900Context startCtx{};
                setRegU32(
                    startCtx, 4, static_cast<uint32_t>(tid));
                StartThread(
                    env.rdram.data(), &startCtx, &env.runtime);
                return getRegS32(startCtx, 2);
            };
            auto rawReferThread = [&](int32_t tid, EeThreadStatus &status)
            {
                std::memset(
                    env.rdram.data() + K_STATUS_ADDR,
                    0,
                    sizeof(status));
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                return getRegS32(statusCtx, 2);
            };
            auto deleteThread = [&](int32_t tid)
            {
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx, 4, static_cast<uint32_t>(tid));
                DeleteThread(
                    env.rdram.data(), &deleteCtx, &env.runtime);
                return getRegS32(deleteCtx, 2);
            };

            writeGuestU32(
                env.rdram.data(), K_EXIT_THREAD_STAGE_ADDR, 0u);
            writeGuestU32(
                env.rdram.data(), K_EXIT_THREAD_ID_ADDR, 0u);
            const int32_t exitTid =
                createThread(kExitEntry, 0x00317000u);
            t.IsTrue(
                exitTid >= 2,
                "ExitThread worker creation should return a guest id");
            t.Equals(
                startThread(exitTid),
                exitTid,
                "ExitThread worker start should return its id");

            EeThreadStatus exitedStatus{};
            const bool exitedDormant = waitUntil([&]()
            {
                return readGuestU32(
                           env.rdram.data(),
                           K_EXIT_THREAD_STAGE_ADDR) == 1u &&
                       rawReferThread(exitTid, exitedStatus) ==
                           THS_DORMANT;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                exitedDormant,
                "ExitThread should synchronously leave a DORMANT object");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_EXIT_THREAD_ID_ADDR),
                static_cast<uint32_t>(exitTid),
                "ExitThread worker should observe its own guest id");
            t.Equals(
                exitedStatus.status,
                THS_DORMANT,
                "ExitThread status payload should be DORMANT");
            t.Equals(
                deleteThread(exitTid),
                exitTid,
                "an ExitThread object should remain deletable");

            writeGuestU32(
                env.rdram.data(),
                K_EXIT_DELETE_THREAD_STAGE_ADDR,
                0u);
            writeGuestU32(
                env.rdram.data(),
                K_EXIT_DELETE_THREAD_ID_ADDR,
                0u);
            const int32_t exitDeleteTid =
                createThread(kExitDeleteEntry, 0x00318000u);
            t.IsTrue(
                exitDeleteTid >= 2,
                "ExitDeleteThread worker creation should return a guest id");
            t.Equals(
                startThread(exitDeleteTid),
                exitDeleteTid,
                "ExitDeleteThread worker start should return its id");

            int32_t deletedStatusResult = THS_DORMANT;
            EeThreadStatus deletedStatus{};
            const bool exitDeleted = waitUntil([&]()
            {
                if (readGuestU32(
                        env.rdram.data(),
                        K_EXIT_DELETE_THREAD_STAGE_ADDR) != 1u)
                {
                    return false;
                }
                deletedStatusResult =
                    rawReferThread(exitDeleteTid, deletedStatus);
                return deletedStatusResult == KE_OK ||
                       deletedStatusResult == KE_UNKNOWN_THID;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                exitDeleted,
                "ExitDeleteThread should remove its thread object");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_EXIT_DELETE_THREAD_ID_ADDR),
                static_cast<uint32_t>(exitDeleteTid),
                "ExitDeleteThread worker should observe its own guest id");
            t.Equals(
                deletedStatusResult,
                KE_OK,
                "raw status of an ExitDeleteThread id should return zero");
            t.Equals(
                deletedStatus.status,
                0,
                "raw missing-id status should leave the zeroed payload untouched");
            t.Equals(
                deleteThread(exitDeleteTid),
                KE_ERROR,
                "deleting an ExitDeleteThread id should return generic -1");

            notifyRuntimeStop();
        });

        tc.Run("debug controls remain bounded across every EE kernel wait state", [](TestCase &t)
        {
            enum class WaitKind
            {
                Sleep,
                Sema,
                Event,
                Suspend,
                WaitSuspend,
            };

            struct WaitExpectation
            {
                WaitKind kind;
                const char *name;
                int32_t status;
                uint32_t waitType;
            };

            constexpr std::array<WaitExpectation, 5u> expectations{{
                {WaitKind::Sleep, "sleep", THS_WAIT, TSW_SLEEP},
                {WaitKind::Sema, "semaphore", THS_WAIT, TSW_SEMA},
                {WaitKind::Event, "event", THS_WAIT, TSW_EVENT},
                {WaitKind::Suspend, "suspend", THS_SUSPEND, 0u},
                {WaitKind::WaitSuspend, "wait-suspend", THS_WAITSUSPEND, TSW_SLEEP},
            }};

            constexpr uint32_t kSleepEntry = 0x00259000u;
            constexpr uint32_t kSemaEntry = 0x00259100u;
            constexpr uint32_t kEventEntry = 0x00259200u;
            constexpr uint32_t kSuspendEntry = 0x00259300u;

            auto readThreadStatus =
                [&](TestEnv &env, int32_t tid)
                {
                    EeThreadStatus status{};
                    std::memset(
                        env.rdram.data() + K_STATUS_ADDR,
                        0,
                        sizeof(status));
                    R5900Context statusCtx{};
                    setRegU32(
                        statusCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    ReferThreadStatus(
                        env.rdram.data(),
                        &statusCtx,
                        &env.runtime);
                    std::memcpy(
                        &status,
                        env.rdram.data() + K_STATUS_ADDR,
                        sizeof(status));
                    return std::pair{
                        getRegS32(statusCtx, 2),
                        status};
                };

            auto startWaiter =
                [&](TestEnv &env, const WaitExpectation &expectation)
                {
                    env.runtime.registerFunction(
                        kSleepEntry, &controlSleepWaitHandler);
                    env.runtime.registerFunction(
                        kSemaEntry, &controlSemaWaitHandler);
                    env.runtime.registerFunction(
                        kEventEntry, &controlEventWaitHandler);
                    env.runtime.registerFunction(
                        kSuspendEntry, &controlSuspendWaitHandler);

                    R5900Context priorityCtx{};
                    setRegU32(priorityCtx, 4, 0u);
                    setRegU32(priorityCtx, 5, 40u);
                    ChangeThreadPriority(
                        env.rdram.data(),
                        &priorityCtx,
                        &env.runtime);

                    uint32_t entry = kSleepEntry;
                    uint32_t argument = 0u;
                    if (expectation.kind == WaitKind::Sema)
                    {
                        const uint32_t semaParam[6] = {
                            0u,
                            1u,
                            0u,
                            0u,
                            0u,
                            0u,
                        };
                        writeGuestWords(
                            env.rdram.data(),
                            K_CONTROL_SEMA_PARAM_ADDR,
                            semaParam,
                            std::size(semaParam));
                        R5900Context createCtx{};
                        setRegU32(
                            createCtx,
                            4,
                            K_CONTROL_SEMA_PARAM_ADDR);
                        CreateSema(
                            env.rdram.data(),
                            &createCtx,
                            &env.runtime);
                        argument = static_cast<uint32_t>(
                            getRegS32(createCtx, 2));
                        entry = kSemaEntry;
                    }
                    else if (expectation.kind == WaitKind::Event)
                    {
                        const uint32_t eventParam[3] = {
                            0u,
                            0u,
                            0u,
                        };
                        writeGuestWords(
                            env.rdram.data(),
                            K_CONTROL_EVENT_PARAM_ADDR,
                            eventParam,
                            std::size(eventParam));
                        R5900Context createCtx{};
                        setRegU32(
                            createCtx,
                            4,
                            K_CONTROL_EVENT_PARAM_ADDR);
                        CreateEventFlag(
                            env.rdram.data(),
                            &createCtx,
                            &env.runtime);
                        argument = static_cast<uint32_t>(
                            getRegS32(createCtx, 2));
                        entry = kEventEntry;
                    }
                    else if (expectation.kind == WaitKind::Suspend)
                    {
                        entry = kSuspendEntry;
                    }

                    const uint32_t threadParam[9] = {
                        0u,
                        entry,
                        0x00319000u,
                        0x00001000u,
                        0u,
                        30u,
                        0u,
                        0u,
                        0u,
                    };
                    writeGuestWords(
                        env.rdram.data(),
                        K_PARAM_ADDR,
                        threadParam,
                        std::size(threadParam));

                    R5900Context createCtx{};
                    setRegU32(createCtx, 4, K_PARAM_ADDR);
                    CreateThread(
                        env.rdram.data(),
                        &createCtx,
                        &env.runtime);
                    const int32_t tid =
                        getRegS32(createCtx, 2);

                    R5900Context startCtx{};
                    setRegU32(
                        startCtx,
                        4,
                        static_cast<uint32_t>(tid));
                    setRegU32(startCtx, 5, argument);
                    StartThread(
                        env.rdram.data(),
                        &startCtx,
                        &env.runtime);

                    if (expectation.kind ==
                        WaitKind::WaitSuspend)
                    {
                        const bool sleeping = waitUntil(
                            [&]()
                            {
                                const auto [result, status] =
                                    readThreadStatus(env, tid);
                                return result == THS_WAIT &&
                                       status.status == THS_WAIT &&
                                       status.waitType == TSW_SLEEP;
                            },
                            std::chrono::milliseconds(500));
                        t.IsTrue(
                            sleeping,
                            std::string(expectation.name) +
                                " worker should enter WAIT before suspension");
                        if (sleeping)
                        {
                            R5900Context suspendCtx{};
                            setRegU32(
                                suspendCtx,
                                4,
                                static_cast<uint32_t>(tid));
                            SuspendThread(
                                env.rdram.data(),
                                &suspendCtx,
                                &env.runtime);
                        }
                    }
                    return tid;
                };

            auto waitForExpectedState =
                [&](TestEnv &env,
                    int32_t tid,
                    const WaitExpectation &expectation)
                {
                    return waitUntil(
                        [&]()
                        {
                            const auto [result, status] =
                                readThreadStatus(env, tid);
                            return result == expectation.status &&
                                   status.status == expectation.status &&
                                   status.waitType ==
                                       expectation.waitType;
                        },
                        std::chrono::milliseconds(500));
                };

            auto finishWorkers =
                [&](PS2Runtime &runtime, const std::string &label)
                {
                    bool finished = waitUntil(
                        [&runtime]()
                        {
                            return runtime
                                       .activeEeHostThreadCount() ==
                                   0;
                        },
                        std::chrono::milliseconds(500));
                    if (!finished)
                    {
                        runtime.requestStop();
                        notifyRuntimeStop(&runtime);
                        finished = waitUntil(
                            [&runtime]()
                            {
                                return runtime
                                           .activeEeHostThreadCount() ==
                                       0;
                            },
                            std::chrono::milliseconds(500));
                    }
                    t.IsTrue(
                        finished,
                        label +
                            " should retire every guest host worker");
                    if (finished)
                    {
                        joinAllGuestHostThreads(&runtime);
                    }
                    else
                    {
                        detachAllGuestHostThreads(&runtime);
                    }
                };

            for (const WaitExpectation &expectation :
                 expectations)
            {
                notifyRuntimeStop();
                TestEnv env;
                const int32_t tid =
                    startWaiter(env, expectation);
                const bool waiting =
                    waitForExpectedState(
                        env, tid, expectation);
                t.IsTrue(
                    waiting,
                    std::string(expectation.name) +
                        " worker should reach its expected state");

                if (waiting)
                {
                    t.IsTrue(
                        env.runtime.debugPause(
                            std::chrono::milliseconds(200)),
                        std::string("debug pause should quiesce a ") +
                            expectation.name + " waiter");
                    const auto [pausedResult, pausedStatus] =
                        readThreadStatus(env, tid);
                    t.Equals(
                        pausedResult,
                        expectation.status,
                        std::string("debug pause should preserve the ") +
                            expectation.name + " status result");
                    t.Equals(
                        pausedStatus.status,
                        expectation.status,
                        std::string("debug pause should preserve the ") +
                            expectation.name + " state");
                    t.Equals(
                        pausedStatus.waitType,
                        expectation.waitType,
                        std::string("debug pause should preserve the ") +
                            expectation.name + " wait reason");
                    env.runtime.debugResume();
                }

                env.runtime.requestStop();
                t.IsTrue(
                    env.runtime.isStopRequested(),
                    std::string(
                        "runtime stop/debugger shutdown should publish while ") +
                        expectation.name + " is waiting");
                finishWorkers(
                    env.runtime,
                    std::string("runtime stop/debugger shutdown from ") +
                        expectation.name);
            }

            for (const WaitExpectation &expectation :
                 expectations)
            {
                notifyRuntimeStop();
                TestEnv env;
                const int32_t tid =
                    startWaiter(env, expectation);
                const bool waiting =
                    waitForExpectedState(
                        env, tid, expectation);
                t.IsTrue(
                    waiting,
                    std::string(expectation.name) +
                        " reset worker should reach its expected state");

                notifyRuntimeStop(&env.runtime);
                t.IsFalse(
                    env.runtime.isStopRequested(),
                    std::string(
                        "kernel reset should not become runtime stop for ") +
                        expectation.name);
                finishWorkers(
                    env.runtime,
                    std::string("kernel reset from ") +
                        expectation.name);

                const auto [resetResult, resetStatus] =
                    readThreadStatus(env, tid);
                (void)resetStatus;
                t.Equals(
                    resetResult,
                    KE_UNKNOWN_THID,
                    std::string("kernel reset should remove the ") +
                        expectation.name + " thread object");
            }

            notifyRuntimeStop();
        });

        tc.Run("TerminateThread unwinds semaphore wait as a normal thread exit", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kWaitThreadEntry = 0x00261000u;
            const uint32_t semaParam[6] = {
                0u,
                1u,
                0u,
                0u,
                0u,
                0u
            };
            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, semaParam, std::size(semaParam));

            R5900Context createSemaCtx{};
            setRegU32(createSemaCtx, 4, K_PARAM_ADDR);
            CreateSema(env.rdram.data(), &createSemaCtx, &env.runtime);
            const int32_t sid = getRegS32(createSemaCtx, 2);
            t.IsTrue(sid >= 0, "CreateSema should create a zero-count semaphore");

            env.runtime.registerFunction(kWaitThreadEntry, &waitSemaUntilTerminatedHandler);

            const uint32_t threadParam[7] = {
                0u,
                kWaitThreadEntry,
                0x00312000u,
                0x00000800u,
                0x00120000u,
                6u,
                0u
            };

            writeGuestU32(env.rdram.data(), K_TERMINATE_SEMA_WAIT_READY_ADDR, 0u);
            writeGuestWords(env.rdram.data(), K_PARAM_ADDR, threadParam, std::size(threadParam));
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            CreateThread(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t tid = getRegS32(env.ctx, 2);
            t.IsTrue(tid >= 2, "CreateThread should return a valid semaphore waiter thread id");

            std::ostringstream capturedErr;
            std::streambuf *oldErr = std::cerr.rdbuf(capturedErr.rdbuf());

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, static_cast<uint32_t>(sid));
            StartThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "StartThread should return the semaphore waiter id");

            const bool waiting = waitUntil([&]()
            {
                if (readGuestU32(env.rdram.data(), K_TERMINATE_SEMA_WAIT_READY_ADDR) != 1u)
                {
                    return false;
                }

                R5900Context statusCtx{};
                setRegU32(statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                ReferThreadStatus(env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(&status, env.rdram.data() + K_STATUS_ADDR, sizeof(status));
                return getRegS32(statusCtx, 2) == status.status &&
                       status.status == THS_WAIT &&
                       status.waitType == TSW_SEMA;
            }, std::chrono::milliseconds(200));
            t.IsTrue(waiting, "worker should block inside WaitSema before termination");

            R5900Context terminateCtx{};
            setRegU32(terminateCtx, 4, static_cast<uint32_t>(tid));
            TerminateThread(env.rdram.data(), &terminateCtx, &env.runtime);
            t.Equals(
                getRegS32(terminateCtx, 2),
                tid,
                "TerminateThread should return the semaphore waiter id");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_TERMINATE_SEMA_WAIT_READY_ADDR),
                1u,
                "terminated WaitSema must not return into guest code");

            std::cerr.rdbuf(oldErr);
            const std::string errText = capturedErr.str();
            t.IsTrue(errText.find("PS2 Thread Exit") == std::string::npos,
                     "thread-exit exceptions from Sync.cpp should be caught as normal exits");

            R5900Context dormantCtx{};
            setRegU32(dormantCtx, 4, static_cast<uint32_t>(tid));
            setRegU32(dormantCtx, 5, K_STATUS_ADDR);
            ReferThreadStatus(env.rdram.data(), &dormantCtx, &env.runtime);
            t.Equals(
                getRegS32(dormantCtx, 2),
                THS_DORMANT,
                "terminated waiter should return dormant status");

            EeThreadStatus dormantStatus{};
            std::memcpy(&dormantStatus, env.rdram.data() + K_STATUS_ADDR, sizeof(dormantStatus));
            t.Equals(dormantStatus.status, THS_DORMANT, "terminated waiter should become dormant");

            R5900Context semaStatusCtx{};
            setRegU32(semaStatusCtx, 4, static_cast<uint32_t>(sid));
            setRegU32(semaStatusCtx, 5, K_STATUS_ADDR);
            iReferSemaStatus(
                env.rdram.data(), &semaStatusCtx, &env.runtime);
            EeSemaStatus semaStatus{};
            std::memcpy(
                &semaStatus,
                env.rdram.data() + K_STATUS_ADDR,
                sizeof(semaStatus));
            t.Equals(
                getRegS32(semaStatusCtx, 2),
                KE_OK,
                "semaphore should remain queryable after termination");
            t.Equals(
                semaStatus.wait_threads,
                0,
                "termination should synchronously unlink the semaphore waiter");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(tid));
            DeleteThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), tid, "DeleteThread should return the terminated waiter id");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(sid));
            DeleteSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), sid, "DeleteSema should return sid while cleaning up the waiter semaphore");
        });

        tc.Run("setup heap and allocator primitives track end-of-heap", [](TestCase &t)
        {
            TestEnv env;

            setRegU32(env.ctx, 4, 0x00180010u);
            setRegU32(env.ctx, 5, 0x00001000u);
            t.IsTrue(callSyscall(0x3Du, env.rdram.data(), &env.ctx, &env.runtime), "SetupHeap syscall should dispatch");
            const uint32_t heapBase = static_cast<uint32_t>(getRegS32(env.ctx, 2));
            t.Equals(heapBase, 0x00180010u, "SetupHeap should return configured base");

            t.IsTrue(callSyscall(0x3Eu, env.rdram.data(), &env.ctx, &env.runtime), "EndOfHeap syscall should dispatch");
            const uint32_t heapLimit = static_cast<uint32_t>(getRegS32(env.ctx, 2));
            t.Equals(heapLimit, 0x00181010u, "EndOfHeap should report the upper limit of the configured heap");

            const uint32_t alignedAlloc = env.runtime.guestMalloc(0x20u, 64u);
            t.IsTrue(alignedAlloc != 0u, "guestMalloc should allocate inside configured heap");
            t.Equals(alignedAlloc & 0x3Fu, 0u, "guestMalloc should honor 64-byte alignment");

            env.runtime.guestFree(alignedAlloc);

            const uint32_t a = env.runtime.guestMalloc(0x100u, 16u);
            const uint32_t b = env.runtime.guestMalloc(0x100u, 16u);
            t.IsTrue(a != 0u && b != 0u, "guestMalloc should provide two adjacent blocks in this heap window");
            env.runtime.guestFree(b);

            const uint32_t grown = env.runtime.guestRealloc(a, 0x180u, 16u);
            t.Equals(grown, a, "guestRealloc should grow in place when adjacent free space is available");

            env.runtime.guestFree(grown);
            const uint32_t reused = env.runtime.guestMalloc(0x80u, 16u);
            t.Equals(reused, heapBase, "guestFree should make the head block reusable");
        });

        tc.Run("memalign stubs allocate aligned guest memory", [](TestCase &t)
        {
            TestEnv env;

            env.runtime.configureGuestHeap(0x00180010u, 0x00182010u);

            setRegU32(env.ctx, 4, 128u);
            setRegU32(env.ctx, 5, 0x40u);
            ps2_stubs::memalign(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t direct = ::getRegU32(&env.ctx, 2);
            t.IsTrue(direct != 0u, "memalign should return a guest address");
            t.Equals(direct & 0x7Fu, 0u, "memalign should honor 128-byte alignment");

            setRegU32(env.ctx, 5, 64u);
            setRegU32(env.ctx, 6, 0x40u);
            ps2_stubs::memalign_r(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t reent = ::getRegU32(&env.ctx, 2);
            t.IsTrue(reent != 0u, "_memalign_r should return a guest address");
            t.Equals(reent & 0x3Fu, 0u, "_memalign_r should honor 64-byte alignment");
            t.IsTrue(reent != direct, "_memalign_r should allocate a distinct block");
        });

        tc.Run("allocator compatibility stubs use the runtime guest heap", [](TestCase &t)
        {
            TestEnv env;

            env.runtime.configureGuestHeap(0x00180010u, 0x00183010u);

            setRegU32(env.ctx, 5, 0x20u);
            ps2_stubs::malloc_r(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t initial = ::getRegU32(&env.ctx, 2);
            t.IsTrue(initial != 0u, "_malloc_r should allocate guest memory");

            writeGuestU32(env.rdram.data(), initial, 0xAABBCCDDu);

            setRegU32(env.ctx, 5, initial);
            setRegU32(env.ctx, 6, 0x80u);
            ps2_stubs::realloc_r(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t grown = ::getRegU32(&env.ctx, 2);
            t.IsTrue(grown != 0u, "_realloc_r should return a guest block");
            t.Equals(readGuestU32(env.rdram.data(), grown), 0xAABBCCDDu,
                     "_realloc_r should preserve existing guest bytes");

            setRegU32(env.ctx, 5, grown);
            ps2_stubs::free_r(env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 5, 0x100u);
            ps2_stubs::malloc_extend_top(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(::getRegU32(&env.ctx, 2), 0u,
                     "malloc_extend_top should be a safe runtime-owned heap no-op");

            ps2_stubs::__malloc_lock(env.rdram.data(), &env.ctx, &env.runtime);
            ps2_stubs::__malloc_unlock(env.rdram.data(), &env.ctx, &env.runtime);
        });

        tc.Run("libc helper stubs cover memclr and libgcc div", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kBuf = 0x5000u;
            std::memset(env.rdram.data() + kBuf, 0xCD, 16u);
            setRegU32(env.ctx, 4, kBuf);
            setRegU32(env.ctx, 5, 12u);
            ps2_stubs::memclr(env.rdram.data(), &env.ctx, &env.runtime);
            for (uint32_t i = 0; i < 12u; ++i)
            {
                t.Equals(env.rdram[kBuf + i], static_cast<uint8_t>(0),
                         "memclr should zero the requested byte range");
            }
            t.Equals(env.rdram[kBuf + 12u], static_cast<uint8_t>(0xCD),
                     "memclr should not write past the requested byte range");

            SET_GPR_S64(&env.ctx, 4, -9);
            SET_GPR_S64(&env.ctx, 5, 2);
            ps2_stubs::__divdi3(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), -4, "__divdi3 should divide signed 64-bit values");
        });

        tc.Run("ReleaseAlarm aliases CancelAlarm and cache toggles succeed", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kAlarmHandlerAddr = 0x00270000u;
            env.runtime.registerFunction(kAlarmHandlerAddr, &alarmNoopHandler);

            setRegU32(env.ctx, 4, 0xFFFFu);
            setRegU32(env.ctx, 5, kAlarmHandlerAddr);
            setRegU32(env.ctx, 6, 0u);
            SetAlarm(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t alarmId = getRegS32(env.ctx, 2);
            t.IsTrue(alarmId > 0, "SetAlarm should create a cancellable alarm");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(alarmId));
            ReleaseAlarm(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "ReleaseAlarm should cancel active alarms");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(alarmId));
            CancelAlarm(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ERROR,
                     "CancelAlarm should report missing alarms after ReleaseAlarm consumes them");

            EnableCache(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "EnableCache should succeed as a no-op");

            DisableCache(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "DisableCache should succeed as a no-op");
        });

        tc.Run("setup heap and thread invalid ids use documented kernel errors", [](TestCase &t)
        {
            TestEnv env;

            setRegU32(env.ctx, 4, 0u);
            CreateThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ERROR, "CreateThread with null param should fail");

            setRegU32(env.ctx, 4, 0u);
            DeleteThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_ILLEGAL_THID, "DeleteThread(0) should be KE_ILLEGAL_THID");

            setRegU32(env.ctx, 4, 0x7FFFu);
            StartThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_THID, "StartThread should reject unknown thread ids");

            setRegU32(env.ctx, 4, 0x7FFFu);
            WakeupThread(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_THID, "WakeupThread should reject unknown thread ids");

            setRegU32(env.ctx, 4, 0x7FFFu);
            PollSema(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_UNKNOWN_SEMID, "PollSema should reject unknown semaphore ids");

            setRegU32(env.ctx, 4, 0xFFFFFFFFu);
            t.IsTrue(callSyscall(0x3Du, env.rdram.data(), &env.ctx, &env.runtime), "SetupHeap syscall should dispatch");
            const uint32_t clampedBase = static_cast<uint32_t>(getRegS32(env.ctx, 2));
            t.IsTrue(clampedBase < PS2_RAM_SIZE, "SetupHeap should normalize out-of-range base into guest RAM");

            t.IsTrue(callSyscall(0x3Eu, env.rdram.data(), &env.ctx, &env.runtime), "EndOfHeap syscall should dispatch");
            const uint32_t heapEnd = static_cast<uint32_t>(getRegS32(env.ctx, 2));
            t.IsTrue(heapEnd >= clampedBase, "EndOfHeap should be at or above normalized heap base");

            setRegU32(env.ctx, 4, 1u);
            setRegU32(env.ctx, 5, 0u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 29, 0x0010FFF0u);
            t.IsTrue(callSyscall(0x3Cu, env.rdram.data(), &env.ctx, &env.runtime), "SetupThread syscall should dispatch");
            const uint32_t setupSp = static_cast<uint32_t>(getRegS32(env.ctx, 2));
            t.Equals(setupSp & 0xFu, 0u, "SetupThread should always return a 16-byte aligned stack pointer");
        });

        tc.Run("OSD config2 syscalls round-trip extended config", [](TestCase &t)
        {
            TestEnv env;
            constexpr uint32_t kConfig2Addr = 0x00005000u;
            constexpr uint32_t kConfig2OutAddr = 0x00005010u;
            constexpr uint32_t kConfig1OutAddr = 0x00005020u;
            constexpr uint32_t kInitialConfig1 =
                (1u << 0) |  // SPDIF disabled
                (1u << 4) |  // non-Japanese language flag
                (1u << 13) | // OSD2
                (1u << 16);  // English
            constexpr uint32_t kConfig2Raw =
                0xABu |        // format
                (0xB0u << 8) | // daylightSaving=1, timeFormat=1, dateFormat=2
                (2u << 16) |   // extended OSD version
                (10u << 24);   // traditional Chinese

            writeGuestU32(env.rdram.data(), K_PARAM_ADDR, kInitialConfig1);
            setRegU32(env.ctx, 4, K_PARAM_ADDR);
            t.IsTrue(callSyscall(0x4Au, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetOsdConfigParam syscall should dispatch");
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SetOsdConfigParam should seed base OSD state");

            writeGuestU32(env.rdram.data(), kConfig2Addr, kConfig2Raw);
            setRegU32(env.ctx, 4, kConfig2Addr);
            setRegU32(env.ctx, 5, 4u);
            setRegU32(env.ctx, 6, 0u);
            t.IsTrue(callSyscall(0x6Eu, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetOsdConfigParam2 syscall should dispatch");
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SetOsdConfigParam2 should succeed");

            writeGuestU32(env.rdram.data(), kConfig2OutAddr, 0xFFFFFFFFu);
            setRegU32(env.ctx, 4, kConfig2OutAddr);
            setRegU32(env.ctx, 5, 4u);
            setRegU32(env.ctx, 6, 0u);
            t.IsTrue(callSyscall(0x6Fu, env.rdram.data(), &env.ctx, &env.runtime),
                     "GetOsdConfigParam2 syscall should dispatch");
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "GetOsdConfigParam2 should succeed");
            const uint32_t readConfig2 = readGuestU32(env.rdram.data(), kConfig2OutAddr);
            t.Equals(readConfig2, kConfig2Raw, "GetOsdConfigParam2 should round-trip the sanitized Config2Param bytes");
            t.Equals((readConfig2 >> 12) & 1u, 1u, "Config2 daylightSaving should live at bit 12 for libosd callers");

            setRegU32(env.ctx, 4, kConfig1OutAddr);
            t.IsTrue(callSyscall(0x4Bu, env.rdram.data(), &env.ctx, &env.runtime),
                     "GetOsdConfigParam syscall should dispatch after Config2 update");
            const uint32_t readConfig1 = readGuestU32(env.rdram.data(), kConfig1OutAddr);
            t.Equals((readConfig1 >> 13) & 0x7u, 2u, "SetOsdConfigParam2 should sync ConfigParam.version");
            t.Equals((readConfig1 >> 16) & 0x1Fu, 10u, "SetOsdConfigParam2 should sync ConfigParam.language");
        });

        tc.Run("numeric syscall 0x83 finds matching table entry", [](TestCase &t)
        {
            TestEnv env;
            constexpr uint32_t kTableBase = 0x00002000u;
            constexpr uint32_t kValues[] = {
                0x11111111u,
                0x11223344u,
                0x55555555u,
                0x89ABCDEFu
            };

            writeGuestWords(env.rdram.data(), kTableBase, kValues, std::size(kValues));
            setRegU32(env.ctx, 4, kTableBase);
            setRegU32(env.ctx, 5, kTableBase + static_cast<uint32_t>(sizeof(kValues)));
            setRegU32(env.ctx, 6, 0x11223344u);

            t.IsTrue(callSyscall(0x83u, env.rdram.data(), &env.ctx, &env.runtime),
                     "syscall 0x83 should dispatch");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     kTableBase + 4u,
                     "FindAddress should return address of first matching word");
        });

        tc.Run("numeric syscall 0x83 supports KSEG aliases", [](TestCase &t)
        {
            TestEnv env;
            constexpr uint32_t kTableBasePhys = 0x00003000u;
            constexpr uint32_t kTableBaseKseg = 0x80003000u;
            constexpr uint32_t kValues[] = {
                0x00123456u,
                0x8000AAAAu
            };

            writeGuestWords(env.rdram.data(), kTableBasePhys, kValues, std::size(kValues));
            setRegU32(env.ctx, 4, kTableBaseKseg);
            setRegU32(env.ctx, 5, kTableBaseKseg + static_cast<uint32_t>(sizeof(kValues)));
            setRegU32(env.ctx, 6, 0x80123456u); // Alias of first table value

            t.IsTrue(callSyscall(0x83u, env.rdram.data(), &env.ctx, &env.runtime),
                     "syscall 0x83 should dispatch");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     kTableBaseKseg,
                     "FindAddress should match KSEG aliases and preserve guest segment in return value");
        });

        tc.Run("numeric syscall 0x83 returns 0 when entry is absent", [](TestCase &t)
        {
            TestEnv env;
            constexpr uint32_t kTableBase = 0x00004000u;
            constexpr uint32_t kValues[] = {
                0x00000001u,
                0x00000002u,
                0x00000003u
            };

            writeGuestWords(env.rdram.data(), kTableBase, kValues, std::size(kValues));
            setRegU32(env.ctx, 4, kTableBase);
            setRegU32(env.ctx, 5, kTableBase + static_cast<uint32_t>(sizeof(kValues)));
            setRegU32(env.ctx, 6, 0xDEADBEEFu);

            t.IsTrue(callSyscall(0x83u, env.rdram.data(), &env.ctx, &env.runtime),
                     "syscall 0x83 should dispatch");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     0u,
                     "FindAddress should return 0 when no matching word exists");
        });

        tc.Run("SetSyscall mirrors guest kernel table entries into low memory", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            initializeGuestKernelState(env.rdram.data());

            constexpr uint32_t kGuestSyscallTableGuestBase = 0x80011F80u;
            constexpr uint32_t kSyscallIndex = 0x83u;
            constexpr uint32_t kHandler = 0x00383548u;
            constexpr uint32_t kExpectedGuestAddr = kGuestSyscallTableGuestBase + (kSyscallIndex * 4u);
            constexpr uint32_t kExpectedPhysAddr = kExpectedGuestAddr & 0x1FFFFFFFu;

            setRegU32(env.ctx, 4, kSyscallIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            uint32_t mirrored = 0u;
            std::memcpy(&mirrored, env.rdram.data() + kExpectedPhysAddr, sizeof(mirrored));
            t.Equals(mirrored,
                     kHandler,
                     "SetSyscall should mirror handler pointers into the guest kernel syscall table");

            setRegU32(env.ctx, 4, 0x80000000u);
            setRegU32(env.ctx, 5, 0x80080000u);
            setRegU32(env.ctx, 6, kHandler);
            t.IsTrue(callSyscall(0x83u, env.rdram.data(), &env.ctx, &env.runtime),
                     "FindAddress syscall should dispatch");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     kExpectedGuestAddr,
                     "FindAddress should discover mirrored SetSyscall entries in low guest memory");

            notifyRuntimeStop();
        });

        tc.Run("SetSyscall honors signed kernel-table offsets", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            initializeGuestKernelState(env.rdram.data());

            constexpr uint32_t kPatchIndex = 0xFFFFC402u;
            constexpr uint32_t kHandler = 0xDEADBEEFu;
            constexpr uint32_t kExpectedGuestAddr = 0x80002F88u;
            constexpr uint32_t kExpectedPhysAddr = kExpectedGuestAddr & 0x1FFFFFFFu;

            setRegU32(env.ctx, 4, kPatchIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch for signed offsets");

            uint32_t mirrored = 0u;
            std::memcpy(&mirrored, env.rdram.data() + kExpectedPhysAddr, sizeof(mirrored));
            t.Equals(mirrored,
                     kHandler,
                     "SetSyscall should treat the syscall index as a signed offset from the kernel table base");

            notifyRuntimeStop();
        });

        tc.Run("guest kernel syscall mirror resets between runs", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            initializeGuestKernelState(env.rdram.data());

            constexpr uint32_t kGuestSyscallTableGuestBase = 0x80011F80u;
            constexpr uint32_t kGuestSyscallTableProbeBase = 0x000002F0u;
            constexpr uint32_t kSyscallIndex = 0x5Au;
            constexpr uint32_t kHandler = 0x00383510u;
            constexpr uint32_t kEntryPhysAddr = (kGuestSyscallTableGuestBase + (kSyscallIndex * 4u)) & 0x1FFFFFFFu;

            setRegU32(env.ctx, 4, kSyscallIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            notifyRuntimeStop();
            initializeGuestKernelState(env.rdram.data());

            uint32_t mirrored = 1u;
            std::memcpy(&mirrored, env.rdram.data() + kEntryPhysAddr, sizeof(mirrored));
            t.Equals(mirrored,
                     0u,
                     "Initializing guest kernel state should clear stale mirrored syscall entries");

            uint32_t probeHi = 0u;
            uint32_t probeLo = 0u;
            std::memcpy(&probeHi, env.rdram.data() + kGuestSyscallTableProbeBase + 0u, sizeof(probeHi));
            std::memcpy(&probeLo, env.rdram.data() + kGuestSyscallTableProbeBase + 8u, sizeof(probeLo));
            t.Equals(probeHi,
                     kGuestSyscallTableGuestBase >> 16,
                     "Guest kernel initialization should seed the syscall table probe high word");
            t.Equals(probeLo,
                     kGuestSyscallTableGuestBase & 0xFFFFu,
                     "Guest kernel initialization should seed the syscall table probe low word");
        });

        tc.Run("SetSyscall override dispatches guest handlers that return through the sentinel", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            constexpr uint32_t kSyscallIndex = 0x91u;
            constexpr uint32_t kHandler = 0x00200000u;

            env.runtime.registerFunction(kHandler, overrideReturnHandler);
            setRegU32(env.ctx, 4, kSyscallIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            setRegU32(env.ctx, 4, 7u);
            setRegU32(env.ctx, 5, 5u);
            t.IsTrue(callSyscall(kSyscallIndex, env.rdram.data(), &env.ctx, &env.runtime),
                     "Overridden syscall should dispatch through guest handler");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     12u,
                     "Successful override dispatch should propagate guest handler return value");

            notifyRuntimeStop();
        });

        tc.Run("SetSyscall override preserves KSEG argument sign extension", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            constexpr uint32_t kSyscallIndex = 0x92u;
            constexpr uint32_t kHandler = 0x00200030u;

            env.runtime.registerFunction(kHandler, overrideKsegCompareHandler);
            setRegU32(env.ctx, 4, kSyscallIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            setRegU32(env.ctx, 4, 0x80000000u);
            setRegU32(env.ctx, 5, 0x80080000u);
            t.IsTrue(callSyscall(kSyscallIndex, env.rdram.data(), &env.ctx, &env.runtime),
                     "Override syscall should invoke the guest handler");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     0x80000004u,
                     "Override invocation should preserve KSEG ordering after 32-bit guest writes");

            notifyRuntimeStop();
        });

        tc.Run("SetSyscall override preserves upper 64 bits when writing 32-bit args", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            constexpr uint32_t kSyscallIndex = 0x93u;
            constexpr uint32_t kHandler = 0x00200040u;

            env.runtime.registerFunction(kHandler, overridePreserveUpper64Handler);
            setRegU32(env.ctx, 4, kSyscallIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            env.ctx.r[4] = _mm_set_epi64x(static_cast<int64_t>(K_EXPECTED_UPPER64),
                                          static_cast<int64_t>(static_cast<int32_t>(0x80000000u)));
            t.IsTrue(callSyscall(kSyscallIndex, env.rdram.data(), &env.ctx, &env.runtime),
                     "Override syscall should invoke the guest handler");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     1u,
                     "Override invocation should preserve the upper 64 bits of 128-bit GPRs when setting 32-bit args");

            notifyRuntimeStop();
        });

        tc.Run("broken syscall overrides fall back to builtin handlers", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            constexpr uint32_t kHandler = 0x00200010u;
            constexpr uint32_t kTableBase = 0x00002000u;
            constexpr uint32_t kValues[] = {
                0x11111111u,
                0x11223344u,
                0x55555555u
            };

            env.runtime.registerFunction(kHandler, overrideBrokenHandler);
            setRegU32(env.ctx, 4, 0x83u);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            writeGuestWords(env.rdram.data(), kTableBase, kValues, std::size(kValues));
            setRegU32(env.ctx, 4, kTableBase);
            setRegU32(env.ctx, 5, kTableBase + static_cast<uint32_t>(sizeof(kValues)));
            setRegU32(env.ctx, 6, 0x11223344u);
            t.IsTrue(callSyscall(0x83u, env.rdram.data(), &env.ctx, &env.runtime),
                     "Builtin syscall should still dispatch when override exits abnormally");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     kTableBase + 4u,
                     "Abnormal override exits should fall back to the builtin syscall implementation");

            notifyRuntimeStop();
        });

        tc.Run("reentrant syscall overrides fall back to builtin handlers", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            constexpr uint32_t kHandler = 0x00200020u;
            constexpr uint32_t kTableBase = 0x00003000u;
            constexpr uint32_t kValues[] = {
                0xCAFEBABEu,
                0x11223344u,
                0x55667788u
            };

            env.runtime.registerFunction(kHandler, overrideRecursiveFindAddressHandler);
            setRegU32(env.ctx, 4, 0x83u);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            writeGuestWords(env.rdram.data(), kTableBase, kValues, std::size(kValues));
            setRegU32(env.ctx, 4, kTableBase);
            setRegU32(env.ctx, 5, kTableBase + static_cast<uint32_t>(sizeof(kValues)));
            setRegU32(env.ctx, 6, 0x11223344u);
            t.IsTrue(callSyscall(0x83u, env.rdram.data(), &env.ctx, &env.runtime),
                     "Reentrant override should resolve through builtin fallback");
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     kTableBase + 4u,
                     "Reentrant override dispatch should use builtin syscall implementation");

            notifyRuntimeStop();
        });

        tc.Run("Copy syscall (0x5A) performs a memory copy", [](TestCase &t)
        {
            TestEnv env;
            constexpr uint32_t kDestAddr = 0x00005000u;
            constexpr uint32_t kSrcAddr = 0x00006000u;
            constexpr uint32_t kSize = 16u;
            constexpr uint32_t kValues[] = {
                0x11223344u,
                0x55667788u,
                0x99AABBCCu,
                0xDDEEFF00u
            };

            writeGuestWords(env.rdram.data(), kSrcAddr, kValues, std::size(kValues));
            
            setRegU32(env.ctx, 4, kDestAddr);
            setRegU32(env.ctx, 5, kSrcAddr);
            setRegU32(env.ctx, 6, kSize);

            t.IsTrue(callSyscall(0x5Au, env.rdram.data(), &env.ctx, &env.runtime),
                     "Copy syscall should dispatch");
            
            for (size_t i = 0; i < std::size(kValues); ++i)
            {
                uint32_t destVal = readGuestU32(env.rdram.data(), kDestAddr + static_cast<uint32_t>(i * sizeof(uint32_t)));
                t.Equals(destVal, kValues[i], "Copy should correctly transfer bytes");
            }
        });

        tc.Run("GetEntryAddress syscall (0x5B) returns handler from guest table", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            initializeGuestKernelState(env.rdram.data());

            constexpr uint32_t kGuestSyscallTableGuestBase = 0x80011F80u;
            constexpr uint32_t kSyscallIndex = 0x5Au;
            constexpr uint32_t kExpectedHandler = 0x00383548u;
            constexpr uint32_t kEntryPhysAddr = (kGuestSyscallTableGuestBase + (kSyscallIndex * 4u)) & 0x1FFFFFFFu;

            writeGuestU32(env.rdram.data(), kEntryPhysAddr, kExpectedHandler);

            setRegU32(env.ctx, 4, kSyscallIndex);

            t.IsTrue(callSyscall(0x5Bu, env.rdram.data(), &env.ctx, &env.runtime),
                     "GetEntryAddress syscall should dispatch");
            
            t.Equals(static_cast<uint32_t>(getRegS32(env.ctx, 2)),
                     kExpectedHandler,
                     "GetEntryAddress should read and return the handler address from the table");

            notifyRuntimeStop();
        });
    });
}
