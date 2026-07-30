#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2_syscalls.h"
#include "ps2_stubs.h"
#include "Stubs/MPEG.h"

#include <chrono>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
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
    constexpr int KE_RELEASE_WAIT = -418;
    constexpr int KE_WAIT_DELETE = -425;
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
    constexpr uint32_t K_ASYNC_EXIT_DELETE_STAGE_ADDR = 0x18D4u;
    constexpr uint32_t K_ALARM_CALLBACK_STAGE_ADDR = 0x18E0u;
    constexpr uint32_t K_ALARM_CALLBACK_ID_ADDR = 0x18E4u;
    constexpr uint32_t K_ALARM_CALLBACK_TICKS_ADDR = 0x18E8u;
    constexpr uint32_t K_ALARM_CALLBACK_ARG_ADDR = 0x18ECu;
    constexpr uint32_t K_ALARM_CALLBACK_GP_ADDR = 0x18F0u;
    constexpr uint32_t K_ALARM_CALLBACK_SP_ADDR = 0x18F4u;
    constexpr uint32_t K_ALARM_SELF_STOP_STAGE_ADDR = 0x18F8u;
    constexpr uint32_t K_ALARM_SELF_STOP_GATE_ADDR = 0x18FCu;
    constexpr uint32_t K_EXECUTOR_THREAD_PARAM_ADDR =
        0x1910u;
    constexpr uint32_t K_EXECUTOR_MAIN_ENTRY =
        0x259800u;
    constexpr uint32_t K_EXECUTOR_CHILD_ENTRY =
        0x259900u;
    constexpr uint32_t K_EXECUTOR_SLEEP_MAIN_ENTRY =
        0x259a00u;
    constexpr uint32_t K_EXECUTOR_SLEEP_CHILD_ENTRY =
        0x259b00u;
    constexpr uint32_t K_EXECUTOR_EXTERNAL_MAIN_ENTRY =
        0x259c00u;
    constexpr uint32_t K_EXECUTOR_EXTERNAL_CHILD_ENTRY =
        0x259d00u;
    constexpr uint32_t K_EXECUTOR_SEMA_MAIN_ENTRY =
        0x259e00u;
    constexpr uint32_t K_EXECUTOR_SEMA_CHILD_ENTRY =
        0x259f00u;
    constexpr uint32_t K_EXECUTOR_EVENT_MAIN_ENTRY =
        0x25a000u;
    constexpr uint32_t K_EXECUTOR_EVENT_CHILD_ENTRY =
        0x25a100u;
    constexpr uint32_t K_EXECUTOR_EVENT_RESULT_ADDR =
        0x1920u;
    constexpr uint32_t K_EXECUTOR_EVENT_MULTI_MAIN_ENTRY =
        0x25a200u;
    constexpr uint32_t K_EXECUTOR_EVENT_CLEAR_CHILD_ENTRY =
        0x25a300u;
    constexpr uint32_t K_EXECUTOR_EVENT_RETAIN_CHILD_ENTRY =
        0x25a400u;
    constexpr uint32_t K_EXECUTOR_EVENT_CLEAR_RESULT_ADDR =
        0x1930u;
    constexpr uint32_t K_EXECUTOR_EVENT_RETAIN_RESULT_ADDR =
        0x1934u;
    constexpr uint32_t K_EXECUTOR_SUSPEND_MAIN_ENTRY =
        0x25a500u;
    constexpr uint32_t K_EXECUTOR_SUSPEND_CHILD_ENTRY =
        0x25a600u;
    constexpr uint32_t K_EXECUTOR_TERMINATE_MAIN_ENTRY =
        0x25a700u;
    constexpr uint32_t K_EXECUTOR_TERMINATE_CANDIDATE_ENTRY =
        0x25a800u;
    constexpr uint32_t K_EXECUTOR_TERMINATE_TARGET_ENTRY =
        0x25a900u;
    constexpr uint32_t K_EXECUTOR_EXIT_MAIN_ENTRY =
        0x25aa00u;
    constexpr uint32_t K_EXECUTOR_EXIT_THREAD_ENTRY =
        0x25ab00u;
    constexpr uint32_t K_EXECUTOR_EXIT_DELETE_THREAD_ENTRY =
        0x25ac00u;
    constexpr uint32_t K_EXECUTOR_PRIORITY_MAIN_ENTRY =
        0x25ad00u;
    constexpr uint32_t K_EXECUTOR_PRIORITY_MARKER_ENTRY =
        0x25ae00u;
    constexpr uint32_t K_EXECUTOR_DEBUG_SPIN_ENTRY =
        0x25af00u;
    constexpr uint32_t K_EXECUTOR_DEBUG_TARGET_ENTRY =
        0x25b000u;
    constexpr uint32_t K_EXECUTOR_CHECKPOINT_ENTRY =
        0x25b100u;
    constexpr uint32_t K_EXECUTOR_NESTED_CHECKPOINT_ENTRY =
        0x25b200u;
    constexpr uint32_t K_EXECUTOR_NESTED_CHECKPOINT_CHILD =
        0x25b300u;
    constexpr uint32_t K_EXECUTOR_MPEG_MAIN_ENTRY =
        0x25b400u;
    constexpr uint32_t K_EXECUTOR_MPEG_PRODUCER_ENTRY =
        0x25b500u;
    constexpr uint32_t K_EXECUTOR_ALARM_MAIN_ENTRY =
        0x25b600u;
    constexpr uint32_t
        K_EXECUTOR_ALARM_CALLBACK_ENTRY =
            0x25b700u;
    constexpr uint32_t
        K_SETUP_THREAD_ROOT_MAIN_ENTRY =
            0x25b800u;
    constexpr uint32_t
        K_SETUP_THREAD_ROOT_HANDLER_ENTRY =
            0x25b900u;
    constexpr uint32_t
        K_SETUP_THREAD_ROOT_WORKER_ENTRY =
            0x25ba00u;
    constexpr uint32_t K_EXECUTOR_MPEG_ADDR =
        0x00123000u;
    constexpr uint32_t K_EXECUTOR_MPEG_IMAGE_ADDR =
        0x00180000u;

    std::mutex g_guestWordMutex;
    std::mutex g_alarmCallbackThreadMutex;
    std::thread::id g_alarmCallbackThread;
    std::mutex g_executorFixtureMutex;
    std::vector<int> g_executorFixtureOrder;
    std::thread::id g_executorMainHostThread;
    std::thread::id g_executorChildHostThread;
    std::thread::id g_executorAlarmMainHostThread;
    std::thread::id
        g_executorAlarmCallbackHostThread;
    std::atomic<int> g_executorAlarmSetResult{
        KE_ERROR};
    std::atomic<uint32_t>
        g_executorAlarmCallbackHits{0u};
    std::atomic<bool>
        g_executorAlarmMainSleeping{false};
    std::atomic<bool>
        g_executorAlarmMainReturned{false};
    std::atomic<bool>
        g_executorAlarmCallbackRanWhileSleeping{
            false};
    std::atomic<uint32_t>
        g_setupThreadRootObservedRa{0u};
    std::atomic<uint32_t>
        g_setupThreadRootHits{0u};
    std::atomic<int>
        g_setupThreadRootObservedThreadId{0};
    std::atomic<int>
        g_setupThreadRootCreateResult{KE_ERROR};
    std::atomic<int>
        g_setupThreadRootStartResult{KE_ERROR};
    std::atomic<bool>
        g_setupThreadRootMainReturned{false};
    int g_executorChildThreadId = 0;
    std::array<int, 3u> g_executorWakeCountResults{
        KE_ERROR,
        KE_ERROR,
        KE_ERROR,
    };
    std::vector<int> g_executorSleepOrder;
    int g_executorSleepChildThreadId = 0;
    int g_executorSleepResult = KE_ERROR;
    int g_executorSecondSleepResult = KE_ERROR;
    int g_executorWakeResult = KE_ERROR;
    int g_executorRawWakeResult = KE_ERROR;
    bool g_executorMainAttemptedWake = false;
    bool g_executorChildRanInsideWake = false;
    bool g_executorChildRanInsideRawWake = false;
    bool g_executorWakeInProgress = false;
    bool g_executorRawWakeInProgress = false;
    bool g_executorSleepRescued = false;
    int g_executorExternalChildThreadId = 0;
    int g_executorExternalSleepResult = KE_ERROR;
    bool g_executorExternalMainReturned = false;
    bool g_executorExternalChildReturned = false;

    enum class ExecutorSemaOperation
    {
        OrdinarySignal,
        RawSignal,
        RawSignalThenDelete,
        OrdinaryRelease,
        RawRelease,
        OrdinaryDelete,
        RawDelete,
    };

    std::vector<int> g_executorSemaOrder;
    ExecutorSemaOperation g_executorSemaOperation =
        ExecutorSemaOperation::OrdinarySignal;
    int g_executorSemaId = KE_ERROR;
    int g_executorSemaChildThreadId = 0;
    int g_executorSemaWaitResult = KE_ERROR;
    int g_executorSemaSignalResult = KE_ERROR;
    int g_executorSemaRawStatus = KE_ERROR;
    uint32_t g_executorSemaRawWaitType = 0u;
    uint32_t g_executorSemaRawWaitId = 0u;
    bool g_executorSemaRawCompletionPending = false;
    bool g_executorSemaSignalInProgress = false;
    bool g_executorSemaChildRanInsideSignal = false;
    bool g_executorSemaMainAttemptedSignal = false;
    bool g_executorSemaRescued = false;

    enum class ExecutorEventOperation
    {
        OrdinarySet,
        RawSet,
        RawSetThenClear,
        OrdinaryRelease,
        RawRelease,
        Delete,
    };

    std::vector<int> g_executorEventOrder;
    ExecutorEventOperation g_executorEventOperation =
        ExecutorEventOperation::OrdinarySet;
    int g_executorEventId = KE_ERROR;
    int g_executorEventChildThreadId = 0;
    int g_executorEventWaitResult = KE_ERROR;
    int g_executorEventOperationResult = KE_ERROR;
    uint32_t g_executorEventResultBits = 0u;
    int g_executorEventRawStatus = KE_ERROR;
    uint32_t g_executorEventRawWaitType = 0u;
    bool g_executorEventOperationInProgress = false;
    bool g_executorEventChildRanInsideOperation = false;
    bool g_executorEventMainAttemptedOperation = false;
    bool g_executorEventRescued = false;
    std::vector<int> g_executorEventMultiOrder;
    int g_executorEventMultiId = KE_ERROR;
    int g_executorEventClearResult = KE_ERROR;
    int g_executorEventRetainResult = KE_ERROR;
    uint32_t g_executorEventClearBits = 0u;
    uint32_t g_executorEventRetainBits = 0u;
    bool g_executorEventRetainWaitedAfterFirstSet =
        false;

    enum class ExecutorSuspendScenario
    {
        SelfOrdinaryResume,
        SelfRawResume,
        WaitSuspendOrdinary,
        WaitSuspendRaw,
    };

    std::vector<int> g_executorSuspendOrder;
    ExecutorSuspendScenario g_executorSuspendScenario =
        ExecutorSuspendScenario::SelfOrdinaryResume;
    int g_executorSuspendChildThreadId = 0;
    int g_executorSuspendWaitResult = KE_ERROR;
    int g_executorSuspendResult = KE_ERROR;
    int g_executorSuspendWakeResult = KE_ERROR;
    int g_executorSuspendResumeResult = KE_ERROR;
    int g_executorSuspendStatusBeforeWake = KE_ERROR;
    int g_executorSuspendStatusAfterWake = KE_ERROR;

    enum class ExecutorTerminateOperation
    {
        OrdinarySemaphore,
        RawSemaphore,
        OrdinaryEvent,
        RawEvent,
    };

    std::vector<int> g_executorTerminateOrder;
    ExecutorTerminateOperation g_executorTerminateOperation =
        ExecutorTerminateOperation::OrdinarySemaphore;
    int g_executorTerminateObjectId = KE_ERROR;
    int g_executorTerminateCandidateThreadId = 0;
    int g_executorTerminateTargetThreadId = 0;
    int g_executorTerminateResult = KE_ERROR;
    int g_executorTerminateStatus = KE_ERROR;
    int g_executorTerminateWaiters = -1;
    int g_executorTerminateDeleteResult = KE_ERROR;
    bool g_executorTerminateTargetReturned = false;
    bool g_executorExitCompleted = false;
    int g_executorExitThreadId = 0;
    int g_executorExitDeleteThreadId = 0;
    int g_executorExitThreadStartResult = KE_ERROR;
    int g_executorExitDeleteThreadStartResult = KE_ERROR;
    int g_executorExitThreadStatusResult = KE_ERROR;
    int32_t g_executorExitThreadStatusPayload = 0;
    int g_executorExitThreadDeleteResult = KE_ERROR;
    int g_executorExitDeleteThreadStatusResult = KE_ERROR;
    int32_t g_executorExitDeleteThreadStatusPayload = 0;
    int g_executorExitDeleteThreadDeleteResult = KE_ERROR;

    enum class ExecutorPriorityScenario
    {
        RawPriority,
        OrdinaryPriority,
        RawCurrentRotation,
        OrdinaryCurrentRotation,
        NamedRotation,
    };

    ExecutorPriorityScenario g_executorPriorityScenario =
        ExecutorPriorityScenario::RawPriority;
    bool g_executorPriorityCompleted = false;
    std::vector<int> g_executorPriorityOrder;
    int g_executorPrioritySetupResult = KE_ERROR;
    int g_executorPriorityOperationResult = KE_ERROR;
    int g_executorPriorityTriggerResult = KE_ERROR;
    int g_executorPriorityStatusValue = -1;
    int g_executorPriorityFirstThreadId = 0;
    int g_executorPrioritySecondThreadId = 0;
    std::atomic<uint32_t> g_executorDebugSpinCount{0u};
    std::atomic<bool> g_executorDebugFinish{false};
    std::atomic<bool> g_executorDebugTargetRequested{
        false};
    std::atomic<bool> g_executorDebugBoundaryApplied{
        false};
    std::atomic<uint32_t> g_executorCheckpointEntries{
        0u};
    std::atomic<bool> g_executorCheckpointProbeAllowed{
        false};
    std::atomic<bool> g_executorCheckpointReturned{
        false};
    std::atomic<bool> g_executorCheckpointFinish{false};
    std::atomic<uint32_t>
        g_executorNestedCheckpointEntries{0u};
    std::atomic<uint32_t>
        g_executorNestedCheckpointChildEntries{0u};
    std::atomic<bool>
        g_executorNestedCheckpointProbeAllowed{false};
    std::atomic<bool>
        g_executorNestedCheckpointContinued{false};
    std::atomic<uint32_t> g_executorMpegSequence{0u};
    std::atomic<uint32_t> g_executorMpegProducerSequence{
        0u};
    std::atomic<uint32_t> g_executorMpegReturnSequence{
        0u};
    std::atomic<int> g_executorMpegResult{KE_ERROR};
    enum class DedicatedExecutorMpegAction
    {
        RestartStream,
        ReleaseWait,
        Terminate,
    };
    std::atomic<DedicatedExecutorMpegAction>
        g_executorMpegAction{
            DedicatedExecutorMpegAction::
                RestartStream};
    std::atomic<int> g_executorMpegMainThreadId{0};
    std::atomic<int> g_executorMpegControlResult{
        KE_ERROR};
    std::atomic<int> g_executorMpegObservedStatus{0};
    std::atomic<uint32_t>
        g_executorMpegObservedWaitType{0u};
    std::atomic<uint32_t>
        g_executorMpegObservedWaitId{0u};

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

    struct EeEventStatus
    {
        uint32_t attr;
        uint32_t option;
        uint32_t initBits;
        uint32_t currBits;
        int32_t numThreads;
        int32_t reserved1;
        int32_t reserved2;
    };

    static_assert(sizeof(EeThreadStatus) == 0x30u, "Unexpected ee_thread_status_t size.");
    static_assert(sizeof(EeSemaStatus) == 0x18u, "Unexpected ee_sema_t size.");
    static_assert(sizeof(EeEventStatus) == 0x1Cu, "Unexpected ee_event_status_t size.");

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
        // Several scheduler fixtures deliberately use guest words as
        // host/guest stage gates. Serialize only these harness helpers so
        // their publications are defined under TSAN; production guest-memory
        // accesses remain non-atomic and concurrent EE execution is still
        // forbidden.
        std::lock_guard<std::mutex> lock(g_guestWordMutex);
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
        std::lock_guard<std::mutex> lock(g_guestWordMutex);
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

    void asyncExitDeleteThreadHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        writeGuestU32(
            rdram, K_ASYNC_EXIT_DELETE_STAGE_ADDR, 1u);
        SleepThread(rdram, ctx, runtime);
        writeGuestU32(
            rdram, K_ASYNC_EXIT_DELETE_STAGE_ADDR, 2u);
        ExitDeleteThread(rdram, ctx, runtime);
        writeGuestU32(
            rdram, K_ASYNC_EXIT_DELETE_STAGE_ADDR, 99u);
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

    void alarmCaptureHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *)
    {
        writeGuestU32(
            rdram,
            K_ALARM_CALLBACK_ID_ADDR,
            ::getRegU32(ctx, 4));
        writeGuestU32(
            rdram,
            K_ALARM_CALLBACK_TICKS_ADDR,
            ::getRegU32(ctx, 5));
        writeGuestU32(
            rdram,
            K_ALARM_CALLBACK_ARG_ADDR,
            ::getRegU32(ctx, 6));
        writeGuestU32(
            rdram,
            K_ALARM_CALLBACK_GP_ADDR,
            ::getRegU32(ctx, 28));
        writeGuestU32(
            rdram,
            K_ALARM_CALLBACK_SP_ADDR,
            ::getRegU32(ctx, 29));
        writeGuestU32(
            rdram,
            K_ALARM_CALLBACK_STAGE_ADDR,
            1u);
        ctx->pc = 0u;
    }

    void alarmThreadCaptureHandler(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_alarmCallbackThreadMutex);
            g_alarmCallbackThread =
                std::this_thread::get_id();
        }
        ctx->pc = 0u;
    }

    void alarmSelfStopHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        runtime->requestStop();
        writeGuestU32(
            rdram,
            K_ALARM_SELF_STOP_STAGE_ADDR,
            1u);
        while (readGuestU32(
                   rdram,
                   K_ALARM_SELF_STOP_GATE_ADDR) == 0u)
        {
            std::this_thread::yield();
        }
        writeGuestU32(
            rdram,
            K_ALARM_SELF_STOP_STAGE_ADDR,
            2u);
        ctx->pc = 0u;
    }

    void dedicatedExecutorAlarmCallback(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorAlarmCallbackHostThread =
                std::this_thread::get_id();
        }
        g_executorAlarmCallbackRanWhileSleeping.store(
            g_executorAlarmMainSleeping.load(
                std::memory_order_acquire) &&
                !g_executorAlarmMainReturned.load(
                    std::memory_order_acquire),
            std::memory_order_release);
        g_executorAlarmCallbackHits.fetch_add(
            1u, std::memory_order_acq_rel);
        ctx->pc = 0u;
    }

    void dedicatedExecutorAlarmMain(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorAlarmMainHostThread =
                std::this_thread::get_id();
        }

        // One thousand modeled alarm ticks is roughly 64 ms: long enough
        // for this sole continuation to commit its scheduler WAIT before the
        // external timer worker publishes the callback.
        setRegU32(*ctx, 4, 1000u);
        setRegU32(
            *ctx,
            5,
            K_EXECUTOR_ALARM_CALLBACK_ENTRY);
        setRegU32(*ctx, 6, 0u);
        SetAlarm(rdram, ctx, runtime);
        const int alarmResult = getRegS32(*ctx, 2);
        g_executorAlarmSetResult.store(
            alarmResult,
            std::memory_order_release);
        if (alarmResult <= 0)
        {
            ctx->pc = 0u;
            return;
        }

        g_executorAlarmMainSleeping.store(
            true, std::memory_order_release);
        SleepThread(rdram, ctx, runtime);
        g_executorAlarmMainReturned.store(
            true, std::memory_order_release);
        ctx->pc = 0u;
    }

    void dedicatedExecutorChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorChildHostThread =
                std::this_thread::get_id();
            g_executorFixtureOrder.push_back(
                runtime->currentEeThreadId());
        }
        for (int wake = 0; wake < 3; ++wake)
        {
            setRegU32(*ctx, 4, 1u);
            WakeupThread(rdram, ctx, runtime);
        }
        ctx->pc = 0u;
    }

    void setupThreadRootWorkerHandler(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        const uint32_t returnAddress =
            ::getRegU32(ctx, 31);
        g_setupThreadRootObservedRa.store(
            returnAddress,
            std::memory_order_release);
        ctx->pc = returnAddress;
    }

    void setupThreadRootExitHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_setupThreadRootObservedThreadId.store(
            runtime->currentEeThreadId(),
            std::memory_order_release);
        g_setupThreadRootHits.fetch_add(
            1u,
            std::memory_order_acq_rel);
        ExitThread(rdram, ctx, runtime);
    }

    void setupThreadRootMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        setRegU32(*ctx, 4, 0u);
        setRegU32(*ctx, 5, 0u);
        setRegU32(*ctx, 6, 0u);
        setRegU32(*ctx, 7, 0u);
        setRegU32(
            *ctx,
            8,
            K_SETUP_THREAD_ROOT_HANDLER_ENTRY);
        SetupThread(rdram, ctx, runtime);

        setRegU32(*ctx, 4, 0u);
        setRegU32(*ctx, 5, 40u);
        ChangeThreadPriority(rdram, ctx, runtime);

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int threadId = getRegS32(*ctx, 2);
        g_setupThreadRootCreateResult.store(
            threadId,
            std::memory_order_release);

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(threadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);
        g_setupThreadRootStartResult.store(
            getRegS32(*ctx, 2),
            std::memory_order_release);
        g_setupThreadRootMainReturned.store(
            true,
            std::memory_order_release);
        ctx->pc = 0u;
    }

    void dedicatedExecutorMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorMainHostThread =
                std::this_thread::get_id();
        }

        InitThread(rdram, ctx, runtime);

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorChildThreadId =
                childThreadId;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorFixtureOrder.push_back(
                runtime->currentEeThreadId());
        }
        SleepThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorWakeCountResults[0] =
                getRegS32(*ctx, 2);
        }
        SleepThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorWakeCountResults[1] =
                getRegS32(*ctx, 2);
        }
        setRegU32(*ctx, 4, 0u);
        CancelWakeupThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorWakeCountResults[2] =
                getRegS32(*ctx, 2);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorSleepChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSleepOrder.push_back(1);
        }
        SleepThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSleepResult =
                getRegS32(*ctx, 2);
            g_executorChildRanInsideWake =
                g_executorWakeInProgress;
            g_executorSleepOrder.push_back(2);
        }
        SleepThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSecondSleepResult =
                getRegS32(*ctx, 2);
            g_executorChildRanInsideRawWake =
                g_executorRawWakeInProgress;
            g_executorSleepOrder.push_back(4);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorSleepMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSleepChildThreadId =
                childThreadId;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorMainAttemptedWake = true;
            g_executorWakeInProgress = true;
        }
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        WakeupThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorWakeResult =
                getRegS32(*ctx, 2);
            g_executorWakeInProgress = false;
            g_executorRawWakeInProgress = true;
        }
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        iWakeupThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorRawWakeResult =
                getRegS32(*ctx, 2);
            g_executorRawWakeInProgress = false;
            g_executorSleepOrder.push_back(3);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorExternalChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        SleepThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorExternalSleepResult =
                getRegS32(*ctx, 2);
            g_executorExternalChildReturned = true;
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorExternalMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);
        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorExternalChildThreadId =
                childThreadId;
        }
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorExternalMainReturned = true;
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorSemaChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSemaOrder.push_back(1);
        }
        WaitSema(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSemaWaitResult =
                getRegS32(*ctx, 2);
            g_executorSemaChildRanInsideSignal =
                g_executorSemaSignalInProgress;
            g_executorSemaOrder.push_back(2);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorSemaMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);

        const std::array<uint32_t, 6u>
            semaphoreParameters{
                0u,
                1u,
                0u,
                0u,
                0u,
                0u,
            };
        writeGuestWords(
            rdram,
            K_CONTROL_SEMA_PARAM_ADDR,
            semaphoreParameters.data(),
            semaphoreParameters.size());
        setRegU32(
            *ctx,
            4,
            K_CONTROL_SEMA_PARAM_ADDR);
        CreateSema(rdram, ctx, runtime);
        const int semaphoreId =
            getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSemaId = semaphoreId;
        }

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSemaChildThreadId =
                childThreadId;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        setRegU32(
            *ctx,
            5,
            static_cast<uint32_t>(
                semaphoreId));
        StartThread(rdram, ctx, runtime);

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSemaMainAttemptedSignal = true;
            g_executorSemaSignalInProgress = true;
        }

        ExecutorSemaOperation operation =
            ExecutorSemaOperation::OrdinarySignal;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            operation = g_executorSemaOperation;
        }
        const bool raw =
            operation ==
                ExecutorSemaOperation::RawSignal ||
            operation ==
                ExecutorSemaOperation::
                    RawSignalThenDelete ||
            operation ==
                ExecutorSemaOperation::RawRelease ||
            operation ==
                ExecutorSemaOperation::RawDelete;
        switch (operation)
        {
        case ExecutorSemaOperation::OrdinarySignal:
        case ExecutorSemaOperation::RawSignal:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    semaphoreId));
            if (raw)
            {
                iSignalSema(rdram, ctx, runtime);
            }
            else
            {
                SignalSema(rdram, ctx, runtime);
            }
            break;
        case ExecutorSemaOperation::
            RawSignalThenDelete:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    semaphoreId));
            iSignalSema(rdram, ctx, runtime);
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    semaphoreId));
            iDeleteSema(rdram, ctx, runtime);
            break;
        case ExecutorSemaOperation::OrdinaryRelease:
        case ExecutorSemaOperation::RawRelease:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    childThreadId));
            if (raw)
            {
                iReleaseWaitThread(
                    rdram, ctx, runtime);
            }
            else
            {
                ReleaseWaitThread(
                    rdram, ctx, runtime);
            }
            break;
        case ExecutorSemaOperation::OrdinaryDelete:
        case ExecutorSemaOperation::RawDelete:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    semaphoreId));
            if (raw)
            {
                iDeleteSema(rdram, ctx, runtime);
            }
            else
            {
                DeleteSema(rdram, ctx, runtime);
            }
            break;
        }

        int rawStatus = KE_ERROR;
        uint32_t rawWaitType = 0u;
        uint32_t rawWaitId = 0u;
        bool rawCompletionPending = false;
        if (raw)
        {
            R5900Context statusContext{};
            setRegU32(
                statusContext,
                4,
                static_cast<uint32_t>(
                    childThreadId));
            setRegU32(
                statusContext,
                5,
                K_STATUS_ADDR);
            iReferThreadStatus(
                rdram,
                &statusContext,
                runtime);
            EeThreadStatus status{};
            std::memcpy(
                &status,
                rdram + K_STATUS_ADDR,
                sizeof(status));
            rawStatus = status.status;
            rawWaitType = status.waitType;
            rawWaitId = status.waitId;
            for (const auto &snapshot :
                 debugThreadSnapshots(runtime))
            {
                if (snapshot.id == childThreadId)
                {
                    rawCompletionPending =
                        snapshot
                            .waitCompletionPending;
                    break;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSemaSignalResult =
                getRegS32(*ctx, 2);
            g_executorSemaRawStatus = rawStatus;
            g_executorSemaRawWaitType =
                rawWaitType;
            g_executorSemaRawWaitId = rawWaitId;
            g_executorSemaRawCompletionPending =
                rawCompletionPending;
            g_executorSemaSignalInProgress = false;
            g_executorSemaOrder.push_back(3);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorEventChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventOrder.push_back(1);
        }
        const uint32_t eventId =
            ::getRegU32(ctx, 4);
        writeGuestU32(
            rdram,
            K_EXECUTOR_EVENT_RESULT_ADDR,
            0x7f7f7f7fu);
        setRegU32(*ctx, 4, eventId);
        setRegU32(*ctx, 5, 0x4u);
        setRegU32(*ctx, 6, 1u);
        setRegU32(
            *ctx,
            7,
            K_EXECUTOR_EVENT_RESULT_ADDR);
        WaitEventFlag(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventWaitResult =
                getRegS32(*ctx, 2);
            g_executorEventResultBits =
                readGuestU32(
                    rdram,
                    K_EXECUTOR_EVENT_RESULT_ADDR);
            g_executorEventChildRanInsideOperation =
                g_executorEventOperationInProgress;
            g_executorEventOrder.push_back(2);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorEventMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);

        const std::array<uint32_t, 3u>
            eventParameters{
                0x2u,
                0u,
                0u,
            };
        writeGuestWords(
            rdram,
            K_CONTROL_EVENT_PARAM_ADDR,
            eventParameters.data(),
            eventParameters.size());
        setRegU32(
            *ctx,
            4,
            K_CONTROL_EVENT_PARAM_ADDR);
        CreateEventFlag(rdram, ctx, runtime);
        const int eventId = getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventId = eventId;
        }

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventChildThreadId =
                childThreadId;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                childThreadId));
        setRegU32(
            *ctx,
            5,
            static_cast<uint32_t>(
                eventId));
        StartThread(rdram, ctx, runtime);

        ExecutorEventOperation operation =
            ExecutorEventOperation::OrdinarySet;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            operation = g_executorEventOperation;
            g_executorEventMainAttemptedOperation =
                true;
            g_executorEventOperationInProgress =
                true;
        }
        const bool raw =
            operation ==
                ExecutorEventOperation::RawSet ||
            operation ==
                ExecutorEventOperation::
                    RawSetThenClear ||
            operation ==
                ExecutorEventOperation::RawRelease;
        switch (operation)
        {
        case ExecutorEventOperation::OrdinarySet:
        case ExecutorEventOperation::RawSet:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(eventId));
            setRegU32(*ctx, 5, 0x4u);
            if (raw)
            {
                iSetEventFlag(rdram, ctx, runtime);
            }
            else
            {
                SetEventFlag(rdram, ctx, runtime);
            }
            break;
        case ExecutorEventOperation::
            RawSetThenClear:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(eventId));
            setRegU32(*ctx, 5, 0x4u);
            iSetEventFlag(rdram, ctx, runtime);
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(eventId));
            setRegU32(*ctx, 5, 0u);
            iClearEventFlag(rdram, ctx, runtime);
            break;
        case ExecutorEventOperation::OrdinaryRelease:
        case ExecutorEventOperation::RawRelease:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    childThreadId));
            if (raw)
            {
                iReleaseWaitThread(
                    rdram, ctx, runtime);
            }
            else
            {
                ReleaseWaitThread(
                    rdram, ctx, runtime);
            }
            break;
        case ExecutorEventOperation::Delete:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(eventId));
            DeleteEventFlag(rdram, ctx, runtime);
            break;
        }

        int rawStatus = KE_ERROR;
        uint32_t rawWaitType = 0u;
        if (raw)
        {
            R5900Context statusContext{};
            setRegU32(
                statusContext,
                4,
                static_cast<uint32_t>(
                    childThreadId));
            setRegU32(
                statusContext,
                5,
                K_STATUS_ADDR);
            iReferThreadStatus(
                rdram,
                &statusContext,
                runtime);
            EeThreadStatus status{};
            std::memcpy(
                &status,
                rdram + K_STATUS_ADDR,
                sizeof(status));
            rawStatus = status.status;
            rawWaitType = status.waitType;
        }

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventOperationResult =
                getRegS32(*ctx, 2);
            g_executorEventRawStatus = rawStatus;
            g_executorEventRawWaitType =
                rawWaitType;
            g_executorEventOperationInProgress =
                false;
            g_executorEventOrder.push_back(3);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorEventClearChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventMultiOrder.push_back(1);
        }
        const uint32_t eventId =
            ::getRegU32(ctx, 4);
        setRegU32(*ctx, 4, eventId);
        setRegU32(*ctx, 5, 0x4u);
        setRegU32(*ctx, 6, 0x11u);
        setRegU32(
            *ctx,
            7,
            K_EXECUTOR_EVENT_CLEAR_RESULT_ADDR);
        WaitEventFlag(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventClearResult =
                getRegS32(*ctx, 2);
            g_executorEventClearBits =
                readGuestU32(
                    rdram,
                    K_EXECUTOR_EVENT_CLEAR_RESULT_ADDR);
            g_executorEventMultiOrder.push_back(3);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorEventRetainChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventMultiOrder.push_back(2);
        }
        const uint32_t eventId =
            ::getRegU32(ctx, 4);
        setRegU32(*ctx, 4, eventId);
        setRegU32(*ctx, 5, 0x4u);
        setRegU32(*ctx, 6, 0x1u);
        setRegU32(
            *ctx,
            7,
            K_EXECUTOR_EVENT_RETAIN_RESULT_ADDR);
        WaitEventFlag(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventRetainResult =
                getRegS32(*ctx, 2);
            g_executorEventRetainBits =
                readGuestU32(
                    rdram,
                    K_EXECUTOR_EVENT_RETAIN_RESULT_ADDR);
            g_executorEventMultiOrder.push_back(5);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorEventMultiMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);
        const std::array<uint32_t, 3u>
            eventParameters{
                0x2u,
                0u,
                0u,
            };
        writeGuestWords(
            rdram,
            K_CONTROL_EVENT_PARAM_ADDR,
            eventParameters.data(),
            eventParameters.size());
        setRegU32(
            *ctx,
            4,
            K_CONTROL_EVENT_PARAM_ADDR);
        CreateEventFlag(rdram, ctx, runtime);
        const int eventId = getRegS32(*ctx, 2);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventMultiId = eventId;
        }

        const auto createAndStart =
            [&](uint32_t entry, uint32_t stack)
        {
            const std::array<uint32_t, 9u>
                threadParameters{
                    0u,
                    entry,
                    stack,
                    0x800u,
                    0u,
                    0u,
                    0u,
                    0u,
                    0u,
                };
            writeGuestWords(
                rdram,
                K_EXECUTOR_THREAD_PARAM_ADDR,
                threadParameters.data(),
                threadParameters.size());
            setRegU32(
                *ctx,
                4,
                K_EXECUTOR_THREAD_PARAM_ADDR);
            CreateThread(rdram, ctx, runtime);
            const int threadId =
                getRegS32(*ctx, 2);
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    threadId));
            setRegU32(
                *ctx,
                5,
                static_cast<uint32_t>(
                    eventId));
            StartThread(rdram, ctx, runtime);
            return threadId;
        };

        (void)createAndStart(
            K_EXECUTOR_EVENT_CLEAR_CHILD_ENTRY,
            0x00310000u);
        const int retainThreadId =
            createAndStart(
                K_EXECUTOR_EVENT_RETAIN_CHILD_ENTRY,
                0x00311000u);

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(eventId));
        setRegU32(*ctx, 5, 0x4u);
        SetEventFlag(rdram, ctx, runtime);

        R5900Context statusContext{};
        setRegU32(
            statusContext,
            4,
            static_cast<uint32_t>(
                retainThreadId));
        setRegU32(
            statusContext,
            5,
            K_STATUS_ADDR);
        iReferThreadStatus(
            rdram,
            &statusContext,
            runtime);
        EeThreadStatus status{};
        std::memcpy(
            &status,
            rdram + K_STATUS_ADDR,
            sizeof(status));
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventRetainWaitedAfterFirstSet =
                status.status == THS_WAIT &&
                status.waitType == TSW_EVENT;
            g_executorEventMultiOrder.push_back(4);
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(eventId));
        setRegU32(*ctx, 5, 0x4u);
        SetEventFlag(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorEventMultiOrder.push_back(6);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorSuspendChildHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        ExecutorSuspendScenario scenario;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            scenario = g_executorSuspendScenario;
            g_executorSuspendOrder.push_back(1);
        }

        if (scenario ==
                ExecutorSuspendScenario::
                    SelfOrdinaryResume ||
            scenario ==
                ExecutorSuspendScenario::SelfRawResume)
        {
            setRegU32(*ctx, 4, 0u);
            SuspendThread(rdram, ctx, runtime);
        }
        else
        {
            SleepThread(rdram, ctx, runtime);
        }

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSuspendWaitResult =
                getRegS32(*ctx, 2);
            g_executorSuspendOrder.push_back(3);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorSuspendMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        ExecutorSuspendScenario scenario;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSuspendChildThreadId =
                childThreadId;
            scenario = g_executorSuspendScenario;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(childThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSuspendOrder.push_back(2);
        }

        const bool waitSuspend =
            scenario ==
                ExecutorSuspendScenario::
                    WaitSuspendOrdinary ||
            scenario ==
                ExecutorSuspendScenario::WaitSuspendRaw;
        const bool raw =
            scenario ==
                ExecutorSuspendScenario::SelfRawResume ||
            scenario ==
                ExecutorSuspendScenario::WaitSuspendRaw;
        if (waitSuspend)
        {
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(childThreadId));
            if (raw)
            {
                iSuspendThread(rdram, ctx, runtime);
            }
            else
            {
                SuspendThread(rdram, ctx, runtime);
            }
            {
                std::lock_guard<std::mutex> lock(
                    g_executorFixtureMutex);
                g_executorSuspendResult =
                    getRegS32(*ctx, 2);
            }
        }

        R5900Context statusContext{};
        setRegU32(
            statusContext,
            4,
            static_cast<uint32_t>(childThreadId));
        setRegU32(
            statusContext,
            5,
            K_STATUS_ADDR);
        iReferThreadStatus(
            rdram,
            &statusContext,
            runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSuspendStatusBeforeWake =
                getRegS32(statusContext, 2);
        }

        if (waitSuspend)
        {
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(childThreadId));
            if (raw)
            {
                iWakeupThread(rdram, ctx, runtime);
            }
            else
            {
                WakeupThread(rdram, ctx, runtime);
            }
            {
                std::lock_guard<std::mutex> lock(
                    g_executorFixtureMutex);
                g_executorSuspendWakeResult =
                    getRegS32(*ctx, 2);
            }

            setRegU32(
                statusContext,
                4,
                static_cast<uint32_t>(childThreadId));
            setRegU32(
                statusContext,
                5,
                K_STATUS_ADDR);
            iReferThreadStatus(
                rdram,
                &statusContext,
                runtime);
            {
                std::lock_guard<std::mutex> lock(
                    g_executorFixtureMutex);
                g_executorSuspendStatusAfterWake =
                    getRegS32(statusContext, 2);
            }
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(childThreadId));
        if (raw)
        {
            iResumeThread(rdram, ctx, runtime);
        }
        else
        {
            ResumeThread(rdram, ctx, runtime);
        }
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorSuspendResumeResult =
                getRegS32(*ctx, 2);
            g_executorSuspendOrder.push_back(4);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorTerminateCandidateHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateOrder.push_back(1);
        }
        SleepThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateOrder.push_back(4);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorTerminateTargetHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        ExecutorTerminateOperation operation;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            operation = g_executorTerminateOperation;
            g_executorTerminateOrder.push_back(2);
        }
        const bool event =
            operation ==
                ExecutorTerminateOperation::OrdinaryEvent ||
            operation ==
                ExecutorTerminateOperation::RawEvent;
        if (event)
        {
            const uint32_t eventId = ::getRegU32(ctx, 4);
            setRegU32(*ctx, 4, eventId);
            setRegU32(*ctx, 5, 0x4u);
            setRegU32(*ctx, 6, 1u);
            setRegU32(*ctx, 7, 0u);
            WaitEventFlag(rdram, ctx, runtime);
        }
        else
        {
            WaitSema(rdram, ctx, runtime);
        }

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateTargetReturned = true;
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorTerminateMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);

        ExecutorTerminateOperation operation;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            operation = g_executorTerminateOperation;
        }
        const bool event =
            operation ==
                ExecutorTerminateOperation::OrdinaryEvent ||
            operation ==
                ExecutorTerminateOperation::RawEvent;
        const bool raw =
            operation ==
                ExecutorTerminateOperation::RawSemaphore ||
            operation ==
                ExecutorTerminateOperation::RawEvent;

        int objectId = KE_ERROR;
        if (event)
        {
            const std::array<uint32_t, 3u>
                eventParameters{
                    0x2u,
                    0u,
                    0u,
                };
            writeGuestWords(
                rdram,
                K_CONTROL_EVENT_PARAM_ADDR,
                eventParameters.data(),
                eventParameters.size());
            setRegU32(
                *ctx,
                4,
                K_CONTROL_EVENT_PARAM_ADDR);
            CreateEventFlag(rdram, ctx, runtime);
            objectId = getRegS32(*ctx, 2);
        }
        else
        {
            const std::array<uint32_t, 6u>
                semaphoreParameters{
                    0u,
                    1u,
                    0u,
                    0u,
                    0u,
                    0u,
                };
            writeGuestWords(
                rdram,
                K_CONTROL_SEMA_PARAM_ADDR,
                semaphoreParameters.data(),
                semaphoreParameters.size());
            setRegU32(
                *ctx,
                4,
                K_CONTROL_SEMA_PARAM_ADDR);
            CreateSema(rdram, ctx, runtime);
            objectId = getRegS32(*ctx, 2);
        }
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateObjectId = objectId;
        }

        const auto createThread =
            [&](uint32_t entry, uint32_t stack)
            {
                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        entry,
                        stack,
                        0x800u,
                        0u,
                        0u,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                setRegU32(
                    *ctx,
                    4,
                    K_EXECUTOR_THREAD_PARAM_ADDR);
                CreateThread(rdram, ctx, runtime);
                return getRegS32(*ctx, 2);
            };

        const int candidateThreadId =
            createThread(
                K_EXECUTOR_TERMINATE_CANDIDATE_ENTRY,
                0x00310000u);
        const int targetThreadId =
            createThread(
                K_EXECUTOR_TERMINATE_TARGET_ENTRY,
                0x00311000u);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateCandidateThreadId =
                candidateThreadId;
            g_executorTerminateTargetThreadId =
                targetThreadId;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(candidateThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(targetThreadId));
        setRegU32(
            *ctx,
            5,
            static_cast<uint32_t>(objectId));
        StartThread(rdram, ctx, runtime);

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(candidateThreadId));
        iWakeupThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateOrder.push_back(3);
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(targetThreadId));
        if (raw)
        {
            iTerminateThread(rdram, ctx, runtime);
        }
        else
        {
            TerminateThread(rdram, ctx, runtime);
        }
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateResult =
                getRegS32(*ctx, 2);
        }

        R5900Context statusContext{};
        setRegU32(
            statusContext,
            4,
            static_cast<uint32_t>(targetThreadId));
        setRegU32(
            statusContext,
            5,
            K_STATUS_ADDR);
        iReferThreadStatus(
            rdram,
            &statusContext,
            runtime);
        const int targetStatus =
            getRegS32(statusContext, 2);

        setRegU32(
            statusContext,
            4,
            static_cast<uint32_t>(objectId));
        setRegU32(
            statusContext,
            5,
            K_STATUS_ADDR);
        int waiters = -1;
        if (event)
        {
            iReferEventFlagStatus(
                rdram,
                &statusContext,
                runtime);
            EeEventStatus status{};
            std::memcpy(
                &status,
                rdram + K_STATUS_ADDR,
                sizeof(status));
            waiters = status.numThreads;
        }
        else
        {
            iReferSemaStatus(
                rdram,
                &statusContext,
                runtime);
            EeSemaStatus status{};
            std::memcpy(
                &status,
                rdram + K_STATUS_ADDR,
                sizeof(status));
            waiters = status.wait_threads;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(targetThreadId));
        DeleteThread(rdram, ctx, runtime);
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorTerminateStatus =
                targetStatus;
            g_executorTerminateWaiters =
                waiters;
            g_executorTerminateDeleteResult =
                getRegS32(*ctx, 2);
            g_executorTerminateOrder.push_back(5);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorExitMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);

        const auto createAndStart =
            [&](uint32_t entry, uint32_t stack)
            {
                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        entry,
                        stack,
                        0x800u,
                        0u,
                        0u,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                setRegU32(
                    *ctx,
                    4,
                    K_EXECUTOR_THREAD_PARAM_ADDR);
                CreateThread(rdram, ctx, runtime);
                const int threadId =
                    getRegS32(*ctx, 2);
                setRegU32(
                    *ctx,
                    4,
                    static_cast<uint32_t>(threadId));
                setRegU32(*ctx, 5, 0u);
                StartThread(rdram, ctx, runtime);
                return std::pair{
                    threadId,
                    getRegS32(*ctx, 2),
                };
            };

        const auto [exitThreadId, exitThreadStartResult] =
            createAndStart(
                K_EXECUTOR_EXIT_THREAD_ENTRY,
                0x00312000u);

        std::memset(
            rdram + K_STATUS_ADDR,
            0xA5,
            sizeof(EeThreadStatus));
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(exitThreadId));
        setRegU32(*ctx, 5, K_STATUS_ADDR);
        iReferThreadStatus(rdram, ctx, runtime);
        const int exitThreadStatusResult =
            getRegS32(*ctx, 2);
        EeThreadStatus exitThreadStatus{};
        std::memcpy(
            &exitThreadStatus,
            rdram + K_STATUS_ADDR,
            sizeof(exitThreadStatus));

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(exitThreadId));
        DeleteThread(rdram, ctx, runtime);
        const int exitThreadDeleteResult =
            getRegS32(*ctx, 2);

        const auto [
            exitDeleteThreadId,
            exitDeleteThreadStartResult] =
            createAndStart(
                K_EXECUTOR_EXIT_DELETE_THREAD_ENTRY,
                0x00313000u);

        std::memset(
            rdram + K_STATUS_ADDR,
            0x5A,
            sizeof(EeThreadStatus));
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(exitDeleteThreadId));
        setRegU32(*ctx, 5, K_STATUS_ADDR);
        iReferThreadStatus(rdram, ctx, runtime);
        const int exitDeleteThreadStatusResult =
            getRegS32(*ctx, 2);
        EeThreadStatus exitDeleteThreadStatus{};
        std::memcpy(
            &exitDeleteThreadStatus,
            rdram + K_STATUS_ADDR,
            sizeof(exitDeleteThreadStatus));

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                exitDeleteThreadId));
        DeleteThread(rdram, ctx, runtime);
        const int exitDeleteThreadDeleteResult =
            getRegS32(*ctx, 2);

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorExitThreadId = exitThreadId;
            g_executorExitDeleteThreadId =
                exitDeleteThreadId;
            g_executorExitThreadStartResult =
                exitThreadStartResult;
            g_executorExitDeleteThreadStartResult =
                exitDeleteThreadStartResult;
            g_executorExitThreadStatusResult =
                exitThreadStatusResult;
            g_executorExitThreadStatusPayload =
                exitThreadStatus.status;
            g_executorExitThreadDeleteResult =
                exitThreadDeleteResult;
            g_executorExitDeleteThreadStatusResult =
                exitDeleteThreadStatusResult;
            g_executorExitDeleteThreadStatusPayload =
                exitDeleteThreadStatus.status;
            g_executorExitDeleteThreadDeleteResult =
                exitDeleteThreadDeleteResult;
            g_executorExitCompleted = true;
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorPriorityMarkerHandler(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        const int marker =
            static_cast<int>(::getRegU32(ctx, 4));
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorPriorityOrder.push_back(marker);
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorPriorityMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);
        setRegU32(*ctx, 4, 0u);
        setRegU32(*ctx, 5, 40u);
        ChangeThreadPriority(rdram, ctx, runtime);
        const int setupResult = getRegS32(*ctx, 2);

        const auto createAndStart =
            [&](uint32_t stack,
                uint32_t priority,
                uint32_t marker)
            {
                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        K_EXECUTOR_PRIORITY_MARKER_ENTRY,
                        stack,
                        0x800u,
                        0u,
                        priority,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                setRegU32(
                    *ctx,
                    4,
                    K_EXECUTOR_THREAD_PARAM_ADDR);
                CreateThread(rdram, ctx, runtime);
                const int threadId =
                    getRegS32(*ctx, 2);
                setRegU32(
                    *ctx,
                    4,
                    static_cast<uint32_t>(threadId));
                setRegU32(*ctx, 5, marker);
                StartThread(rdram, ctx, runtime);
                return threadId;
            };
        const auto recordMain =
            [](int marker)
            {
                std::lock_guard<std::mutex> lock(
                    g_executorFixtureMutex);
                g_executorPriorityOrder.push_back(
                    marker);
            };

        ExecutorPriorityScenario scenario;
        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            scenario = g_executorPriorityScenario;
        }

        int operationResult = KE_ERROR;
        int triggerResult = KE_ERROR;
        int statusValue = -1;
        int firstThreadId = 0;
        int secondThreadId = 0;

        switch (scenario)
        {
        case ExecutorPriorityScenario::RawPriority:
            firstThreadId =
                createAndStart(
                    0x00314000u, 50u, 2u);
            recordMain(1);
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(firstThreadId));
            setRegU32(*ctx, 5, 30u);
            iChangeThreadPriority(
                rdram, ctx, runtime);
            operationResult = getRegS32(*ctx, 2);
            recordMain(3);
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(firstThreadId));
            setRegU32(*ctx, 5, K_STATUS_ADDR);
            iReferThreadStatus(rdram, ctx, runtime);
            {
                EeThreadStatus status{};
                std::memcpy(
                    &status,
                    rdram + K_STATUS_ADDR,
                    sizeof(status));
                statusValue =
                    status.current_priority;
            }
            setRegU32(*ctx, 4, 0u);
            setRegU32(*ctx, 5, 60u);
            ChangeThreadPriority(
                rdram, ctx, runtime);
            triggerResult = getRegS32(*ctx, 2);
            recordMain(4);
            break;

        case ExecutorPriorityScenario::
            OrdinaryPriority:
            firstThreadId =
                createAndStart(
                    0x00315000u, 50u, 2u);
            recordMain(1);
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(firstThreadId));
            setRegU32(*ctx, 5, 30u);
            ChangeThreadPriority(
                rdram, ctx, runtime);
            operationResult = getRegS32(*ctx, 2);
            recordMain(3);
            break;

        case ExecutorPriorityScenario::
            RawCurrentRotation:
            firstThreadId =
                createAndStart(
                    0x00316000u, 40u, 2u);
            recordMain(1);
            setRegU32(*ctx, 4, 40u);
            iRotateThreadReadyQueue(
                rdram, ctx, runtime);
            operationResult = getRegS32(*ctx, 2);
            recordMain(3);
            setRegU32(*ctx, 4, 0u);
            setRegU32(*ctx, 5, 60u);
            ChangeThreadPriority(
                rdram, ctx, runtime);
            triggerResult = getRegS32(*ctx, 2);
            recordMain(4);
            break;

        case ExecutorPriorityScenario::
            OrdinaryCurrentRotation:
            firstThreadId =
                createAndStart(
                    0x00317000u, 40u, 2u);
            recordMain(1);
            setRegU32(*ctx, 4, 40u);
            RotateThreadReadyQueue(
                rdram, ctx, runtime);
            operationResult = getRegS32(*ctx, 2);
            recordMain(3);
            break;

        case ExecutorPriorityScenario::NamedRotation:
            firstThreadId =
                createAndStart(
                    0x00318000u, 50u, 2u);
            secondThreadId =
                createAndStart(
                    0x00319000u, 50u, 3u);
            recordMain(1);
            setRegU32(*ctx, 4, 50u);
            RotateThreadReadyQueue(
                rdram, ctx, runtime);
            operationResult = getRegS32(*ctx, 2);
            recordMain(4);
            setRegU32(*ctx, 4, 0u);
            setRegU32(*ctx, 5, 60u);
            ChangeThreadPriority(
                rdram, ctx, runtime);
            triggerResult = getRegS32(*ctx, 2);
            recordMain(5);
            break;
        }

        {
            std::lock_guard<std::mutex> lock(
                g_executorFixtureMutex);
            g_executorPrioritySetupResult =
                setupResult;
            g_executorPriorityOperationResult =
                operationResult;
            g_executorPriorityTriggerResult =
                triggerResult;
            g_executorPriorityStatusValue =
                statusValue;
            g_executorPriorityFirstThreadId =
                firstThreadId;
            g_executorPrioritySecondThreadId =
                secondThreadId;
            g_executorPriorityCompleted = true;
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorDebugSpinHandler(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        g_executorDebugSpinCount.fetch_add(
            1u, std::memory_order_relaxed);
        if (g_executorDebugFinish.load(
                std::memory_order_acquire))
        {
            ctx->pc = 0u;
        }
        else if (
            g_executorDebugTargetRequested.exchange(
                false,
                std::memory_order_acq_rel))
        {
            ctx->pc =
                K_EXECUTOR_DEBUG_TARGET_ENTRY;
        }
        else
        {
            ctx->pc =
                K_EXECUTOR_DEBUG_SPIN_ENTRY;
        }
    }

    void dedicatedExecutorDebugTargetHandler(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        g_executorDebugSpinCount.fetch_add(
            1u, std::memory_order_relaxed);
        ctx->pc =
            g_executorDebugFinish.load(
                std::memory_order_acquire)
                ? 0u
                : K_EXECUTOR_DEBUG_SPIN_ENTRY;
    }

    void dedicatedExecutorCheckpointHandler(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_executorCheckpointEntries.fetch_add(
            1u, std::memory_order_relaxed);
        while (!g_executorCheckpointProbeAllowed.load(
            std::memory_order_acquire))
        {
            std::this_thread::yield();
        }

        for (uint32_t probe = 0u;
             probe < 300u;
             ++probe)
        {
            ctx->pc = K_EXECUTOR_CHECKPOINT_ENTRY;
            if (runtime->checkpointGuestExecution(
                    ctx) ==
                PS2GuestCheckpointResult::
                    ExitToDispatcher)
            {
                return;
            }
        }
        g_executorCheckpointReturned.store(
            true, std::memory_order_release);

        while (!g_executorCheckpointFinish.load(
            std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorNestedCheckpointChild(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_executorNestedCheckpointChildEntries
            .fetch_add(
                1u, std::memory_order_relaxed);
        while (!g_executorNestedCheckpointProbeAllowed
                    .load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }

        ctx->pc =
            K_EXECUTOR_NESTED_CHECKPOINT_CHILD;
        if (runtime->checkpointGuestExecution(ctx) ==
            PS2GuestCheckpointResult::
                ExitToDispatcher)
        {
            return;
        }
    }

    void dedicatedExecutorNestedCheckpointHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_executorNestedCheckpointEntries.fetch_add(
            1u, std::memory_order_relaxed);
        ctx->pc =
            K_EXECUTOR_NESTED_CHECKPOINT_CHILD;
        if (!runtime->dispatchGuestBranch(
                rdram,
                ctx,
                K_EXECUTOR_NESTED_CHECKPOINT_CHILD,
                K_EXECUTOR_NESTED_CHECKPOINT_ENTRY,
                K_EXECUTOR_NESTED_CHECKPOINT_ENTRY +
                    8u,
                PS2Runtime::GuestBranchKind::
                    DirectCall,
                "nested-checkpoint"))
        {
            return;
        }

        g_executorNestedCheckpointContinued.store(
            true, std::memory_order_release);
        ctx->pc = 0u;
    }

    void dedicatedExecutorMpegProducerHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_executorMpegProducerSequence.store(
            g_executorMpegSequence.fetch_add(
                1u, std::memory_order_acq_rel) +
            1u,
            std::memory_order_release);

        const int mainThreadId =
            g_executorMpegMainThreadId.load(
                std::memory_order_acquire);
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(mainThreadId));
        setRegU32(*ctx, 5, K_STATUS_ADDR);
        ReferThreadStatus(rdram, ctx, runtime);
        EeThreadStatus status{};
        std::memcpy(
            &status,
            rdram + K_STATUS_ADDR,
            sizeof(status));
        g_executorMpegObservedStatus.store(
            status.status,
            std::memory_order_release);
        g_executorMpegObservedWaitType.store(
            status.waitType,
            std::memory_order_release);
        g_executorMpegObservedWaitId.store(
            status.waitId,
            std::memory_order_release);

        switch (g_executorMpegAction.load(
            std::memory_order_acquire))
        {
        case DedicatedExecutorMpegAction::ReleaseWait:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    mainThreadId));
            ReleaseWaitThread(rdram, ctx, runtime);
            g_executorMpegControlResult.store(
                getRegS32(*ctx, 2),
                std::memory_order_release);
            break;
        case DedicatedExecutorMpegAction::Terminate:
            setRegU32(
                *ctx,
                4,
                static_cast<uint32_t>(
                    mainThreadId));
            TerminateThread(rdram, ctx, runtime);
            g_executorMpegControlResult.store(
                getRegS32(*ctx, 2),
                std::memory_order_release);
            break;
        case DedicatedExecutorMpegAction::
            RestartStream:
            ps2_stubs::notifyMpegCdStreamStart(
                runtime);
            g_executorMpegControlResult.store(
                0,
                std::memory_order_release);
            break;
        }
        ctx->pc = 0u;
    }

    void dedicatedExecutorMpegMainHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);
        g_executorMpegMainThreadId.store(
            runtime->currentEeThreadId(),
            std::memory_order_release);
        ps2_stubs::notifyMpegCdStreamStart(runtime);

        setRegU32(
            *ctx,
            4,
            K_EXECUTOR_THREAD_PARAM_ADDR);
        CreateThread(rdram, ctx, runtime);
        const int producerThreadId =
            getRegS32(*ctx, 2);
        if (producerThreadId <= 1)
        {
            g_executorMpegResult.store(
                producerThreadId,
                std::memory_order_release);
            ctx->pc = 0u;
            return;
        }

        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                producerThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);

        setRegU32(
            *ctx, 4, K_EXECUTOR_MPEG_ADDR);
        setRegU32(
            *ctx, 5, K_EXECUTOR_MPEG_IMAGE_ADDR);
        setRegU32(*ctx, 6, 832u);
        ps2_stubs::sceMpegGetPicture(
            rdram, ctx, runtime);
        g_executorMpegResult.store(
            getRegS32(*ctx, 2),
            std::memory_order_release);
        g_executorMpegReturnSequence.store(
            g_executorMpegSequence.fetch_add(
                1u, std::memory_order_acq_rel) +
                1u,
            std::memory_order_release);
        ctx->pc = 0u;
    }

    struct DedicatedExecutorMpegFixtureResult
    {
        bool memoryInitialized = false;
        bool completedWithoutRescue = false;
        bool completedAfterRescue = false;
        uint32_t producerSequence = 0u;
        uint32_t returnSequence = 0u;
        int pictureResult = KE_ERROR;
        int mainThreadId = 0;
        int controlResult = KE_ERROR;
        int observedStatus = 0;
        uint32_t observedWaitType = 0u;
        uint32_t observedWaitId = 0u;
        size_t managedAfterStop = 0u;
    };

    DedicatedExecutorMpegFixtureResult
    runDedicatedExecutorMpegFixture(
        DedicatedExecutorMpegAction action)
    {
        DedicatedExecutorMpegFixtureResult result{};
        PS2RuntimeConfiguration configuration{};
        configuration.eeExecutionBackend =
            EeExecutionBackendKind::LegacyCppFiber;
        configuration
            .useEeExecutionBackendEnvironment =
            false;
        PS2Runtime runtime(configuration);
        result.memoryInitialized =
            runtime.memory().initialize();
        if (!result.memoryInitialized)
        {
            return result;
        }

        uint8_t *const rdram =
            runtime.memory().getRDRAM();
        const std::array<uint32_t, 9u>
            producerParameters{
                0u,
                K_EXECUTOR_MPEG_PRODUCER_ENTRY,
                0x00310000u,
                0x800u,
                0u,
                2u,
                0u,
                0u,
                0u,
            };
        writeGuestWords(
            rdram,
            K_EXECUTOR_THREAD_PARAM_ADDR,
            producerParameters.data(),
            producerParameters.size());
        runtime.registerFunction(
            K_EXECUTOR_MPEG_MAIN_ENTRY,
            &dedicatedExecutorMpegMainHandler);
        runtime.registerFunction(
            K_EXECUTOR_MPEG_PRODUCER_ENTRY,
            &dedicatedExecutorMpegProducerHandler);
        runtime.cpu().pc =
            K_EXECUTOR_MPEG_MAIN_ENTRY;
        setRegU32(
            runtime.cpu(), 29, 0x00300000u);
        ps2_stubs::resetMpegStubState(&runtime);
        g_executorMpegSequence.store(
            0u, std::memory_order_release);
        g_executorMpegProducerSequence.store(
            0u, std::memory_order_release);
        g_executorMpegReturnSequence.store(
            0u, std::memory_order_release);
        g_executorMpegResult.store(
            KE_ERROR, std::memory_order_release);
        g_executorMpegAction.store(
            action, std::memory_order_release);
        g_executorMpegMainThreadId.store(
            0, std::memory_order_release);
        g_executorMpegControlResult.store(
            KE_ERROR, std::memory_order_release);
        g_executorMpegObservedStatus.store(
            0, std::memory_order_release);
        g_executorMpegObservedWaitType.store(
            0u, std::memory_order_release);
        g_executorMpegObservedWaitId.store(
            0u, std::memory_order_release);

        runtime.startDedicatedEeExecutionForTesting();
        const auto completed =
            [&runtime]()
            {
                return g_executorMpegProducerSequence
                               .load(
                                   std::memory_order_acquire) !=
                           0u &&
                       runtime
                               .managedEeExecutionThreadCountForTesting() ==
                           0u;
            };
        result.completedWithoutRescue =
            waitUntil(
                completed,
                std::chrono::seconds(1));
        if (!result.completedWithoutRescue)
        {
            // Keep a legacy host-condition-variable
            // regression bounded without making the
            // guest controller runnable.
            ps2_stubs::notifyMpegCdStreamStart(
                &runtime);
        }
        result.completedAfterRescue =
            result.completedWithoutRescue ||
            waitUntil(
                completed,
                std::chrono::seconds(2));
        runtime.stopDedicatedEeExecutionForTesting();

        result.producerSequence =
            g_executorMpegProducerSequence.load(
                std::memory_order_acquire);
        result.returnSequence =
            g_executorMpegReturnSequence.load(
                std::memory_order_acquire);
        result.pictureResult =
            g_executorMpegResult.load(
                std::memory_order_acquire);
        result.mainThreadId =
            g_executorMpegMainThreadId.load(
                std::memory_order_acquire);
        result.controlResult =
            g_executorMpegControlResult.load(
                std::memory_order_acquire);
        result.observedStatus =
            g_executorMpegObservedStatus.load(
                std::memory_order_acquire);
        result.observedWaitType =
            g_executorMpegObservedWaitType.load(
                std::memory_order_acquire);
        result.observedWaitId =
            g_executorMpegObservedWaitId.load(
                std::memory_order_acquire);
        result.managedAfterStop =
            runtime
                .managedEeExecutionThreadCountForTesting();
        return result;
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
        tc.Run("runtime selects the legacy host-thread EE backend", [](TestCase &t)
        {
            PS2Runtime runtime;
            const EeExecutionBackendBuildInfo build =
                eeExecutionBackendBuildInfo();
            const std::string diagnostics =
                eeExecutionBackendDiagnostics(
                    EeExecutionBackendKind::
                        LegacyHostThread);
            t.Equals(
                std::string(runtime.eeExecutionBackendName()),
                std::string("legacy-host-thread"),
                "the transitional EE backend should be explicit");
            t.Equals(
                runtime.managedEeExecutionThreadCountForTesting(),
                size_t{0u},
                "a fresh backend should not own guest continuations");
            t.IsTrue(
                diagnostics.find(
                    "selected=legacy-host-thread") !=
                        std::string::npos &&
                    diagnostics.find(
                        "Boost.Context=") !=
                        std::string::npos,
                "backend diagnostics should identify the selected and optional continuation mechanisms");
            if (build.boostContextFcontextAvailable)
            {
                t.IsTrue(
                    build.boostVersion == "1.91.0" &&
                        build.architecture ==
                            "x86_64" &&
                        (build.binaryFormat == "elf" ||
                         build.binaryFormat == "pe") &&
                        (build.abi == "sysv" ||
                         build.abi == "ms") &&
                        build.contextImplementation ==
                            "fcontext" &&
                        diagnostics.find(
                            "Boost=1.91.0") !=
                            std::string::npos &&
                        diagnostics.find(
                            "context-implementation=fcontext") !=
                            std::string::npos,
                    "a built fiber backend should expose the exact pinned Boost fcontext contract");
            }
            else
            {
                t.IsTrue(
                    build.boostVersion == "not-built" &&
                        build.contextImplementation ==
                            "unavailable",
                    "an unsupported or disabled target should report the explicit host-thread fallback");
            }
        });

        tc.Run(
            "fiber runtime owns main and started thread on one executor",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::
                        LegacyCppFiber;
                configuration
                    .useEeExecutionBackendEnvironment =
                    false;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                t.IsTrue(
                    memoryInitialized,
                    "the executor fixture should allocate "
                    "runtime RDRAM");
                if (!memoryInitialized)
                {
                    return;
                }
                uint8_t *const rdram =
                    runtime.memory().getRDRAM();

                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        K_EXECUTOR_CHILD_ENTRY,
                        0x00310000u,
                        0x800u,
                        0u,
                        0u,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                runtime.registerFunction(
                    K_EXECUTOR_MAIN_ENTRY,
                    &dedicatedExecutorMainHandler);
                runtime.registerFunction(
                    K_EXECUTOR_CHILD_ENTRY,
                    &dedicatedExecutorChildHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);

                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    g_executorFixtureOrder.clear();
                    g_executorMainHostThread = {};
                    g_executorChildHostThread = {};
                    g_executorChildThreadId = 0;
                    g_executorWakeCountResults = {
                        KE_ERROR,
                        KE_ERROR,
                        KE_ERROR,
                    };
                }
                const std::thread::id callerThread =
                    std::this_thread::get_id();

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool completed = waitUntil(
                    []()
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        return g_executorFixtureOrder
                                   .size() == 2u;
                    },
                    std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                std::vector<int> order;
                std::thread::id mainHostThread;
                std::thread::id childHostThread;
                int childThreadId = 0;
                std::array<int, 3u> wakeCountResults{};
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    order = g_executorFixtureOrder;
                    mainHostThread =
                        g_executorMainHostThread;
                    childHostThread =
                        g_executorChildHostThread;
                    childThreadId =
                        g_executorChildThreadId;
                    wakeCountResults =
                        g_executorWakeCountResults;
                }

                t.IsTrue(
                    runtime.usesDedicatedEeExecutor() &&
                        completed &&
                        order ==
                            std::vector<int>{
                                childThreadId,
                                1},
                    "priority-zero StartThread should "
                    "dispatch the child before returning "
                    "to the priority-one main thread");
                t.IsTrue(
                    mainHostThread != callerThread &&
                        childHostThread ==
                            mainHostThread,
                    "the main and started guest should "
                    "share the runtime-owned executor "
                    "host thread");
                t.IsTrue(
                    wakeCountResults ==
                        std::array<int, 3u>{1, 1, 1},
                    "three child wakes should let the "
                    "main consume two SleepThread calls "
                    "and cancel the final wake count");
                t.Equals(
                    runtime
                        .managedEeExecutionThreadCountForTesting(),
                    size_t{0u},
                    "executor shutdown should destroy "
                    "every runtime-owned continuation");
            });

        tc.Run(
            "fiber StartThread returns through the SetupThread root",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::
                        LegacyCppFiber;
                configuration
                    .useEeExecutionBackendEnvironment =
                    false;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                t.IsTrue(
                    memoryInitialized,
                    "the SetupThread-root fiber fixture should allocate runtime RDRAM");
                if (!memoryInitialized)
                {
                    return;
                }

                uint8_t *const rdram =
                    runtime.memory().getRDRAM();
                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        K_SETUP_THREAD_ROOT_WORKER_ENTRY,
                        0x00310000u,
                        0x800u,
                        0u,
                        30u,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                runtime.registerFunction(
                    K_SETUP_THREAD_ROOT_MAIN_ENTRY,
                    &setupThreadRootMainHandler);
                runtime.registerFunction(
                    K_SETUP_THREAD_ROOT_HANDLER_ENTRY,
                    &setupThreadRootExitHandler);
                runtime.registerFunction(
                    K_SETUP_THREAD_ROOT_WORKER_ENTRY,
                    &setupThreadRootWorkerHandler);
                runtime.cpu().pc =
                    K_SETUP_THREAD_ROOT_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);

                g_setupThreadRootObservedRa.store(
                    0u,
                    std::memory_order_release);
                g_setupThreadRootHits.store(
                    0u,
                    std::memory_order_release);
                g_setupThreadRootObservedThreadId.store(
                    0,
                    std::memory_order_release);
                g_setupThreadRootCreateResult.store(
                    KE_ERROR,
                    std::memory_order_release);
                g_setupThreadRootStartResult.store(
                    KE_ERROR,
                    std::memory_order_release);
                g_setupThreadRootMainReturned.store(
                    false,
                    std::memory_order_release);

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool completed = waitUntil(
                    [&runtime]()
                    {
                        return g_setupThreadRootMainReturned
                                   .load(
                                       std::memory_order_acquire) &&
                               g_setupThreadRootHits.load(
                                   std::memory_order_acquire) ==
                                   1u &&
                               runtime
                                       .managedEeExecutionThreadCountForTesting() ==
                                   0u;
                    },
                    std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                const int threadId =
                    g_setupThreadRootCreateResult.load(
                        std::memory_order_acquire);
                t.IsTrue(
                    completed,
                    "the fiber worker should return through the configured root before its main continuation finishes");
                t.IsTrue(
                    threadId >= 2 &&
                        g_setupThreadRootStartResult.load(
                            std::memory_order_acquire) ==
                            threadId,
                    "the dedicated executor should create and start the root-return worker");
                t.Equals(
                    g_setupThreadRootObservedRa.load(
                        std::memory_order_acquire),
                    K_SETUP_THREAD_ROOT_HANDLER_ENTRY,
                    "the fiber worker should inherit the SetupThread root in $ra");
                t.Equals(
                    g_setupThreadRootObservedThreadId.load(
                        std::memory_order_acquire),
                    threadId,
                    "the fiber root should retain the returning worker identity");
                t.Equals(
                    runtime
                        .managedEeExecutionThreadCountForTesting(),
                    size_t{0u},
                    "the root-exited worker and main continuation should both be destroyed");
            });

        tc.Run(
            "fiber executor wakes from idle to publish an alarm callback",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::
                        LegacyCppFiber;
                configuration
                    .useEeExecutionBackendEnvironment =
                    false;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                t.IsTrue(
                    memoryInitialized,
                    "the alarm publication fixture should allocate runtime RDRAM");
                if (!memoryInitialized)
                {
                    return;
                }

                runtime.registerFunction(
                    K_EXECUTOR_ALARM_MAIN_ENTRY,
                    &dedicatedExecutorAlarmMain);
                runtime.registerFunction(
                    K_EXECUTOR_ALARM_CALLBACK_ENTRY,
                    &dedicatedExecutorAlarmCallback);
                runtime.cpu().pc =
                    K_EXECUTOR_ALARM_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);
                g_executorAlarmSetResult.store(
                    KE_ERROR,
                    std::memory_order_release);
                g_executorAlarmCallbackHits.store(
                    0u,
                    std::memory_order_release);
                g_executorAlarmMainSleeping.store(
                    false,
                    std::memory_order_release);
                g_executorAlarmMainReturned.store(
                    false,
                    std::memory_order_release);
                g_executorAlarmCallbackRanWhileSleeping
                    .store(
                        false,
                        std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    g_executorAlarmMainHostThread = {};
                    g_executorAlarmCallbackHostThread =
                        {};
                }
                const std::thread::id callerThread =
                    std::this_thread::get_id();

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool callbackRan = waitUntil(
                    []()
                    {
                        return g_executorAlarmCallbackHits
                                   .load(
                                       std::memory_order_acquire) ==
                               1u;
                    },
                    std::chrono::seconds(2));
                const size_t managedBeforeStop =
                    runtime
                        .managedEeExecutionThreadCountForTesting();
                const bool mainReturnedBeforeStop =
                    g_executorAlarmMainReturned.load(
                        std::memory_order_acquire);
                runtime
                    .stopDedicatedEeExecutionForTesting();

                std::thread::id mainHostThread;
                std::thread::id callbackHostThread;
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    mainHostThread =
                        g_executorAlarmMainHostThread;
                    callbackHostThread =
                        g_executorAlarmCallbackHostThread;
                }
                t.IsTrue(
                    callbackRan &&
                        g_executorAlarmSetResult.load(
                            std::memory_order_acquire) >
                            0,
                    "the external alarm worker should publish exactly one callback without another guest boundary");
                t.IsTrue(
                    g_executorAlarmMainSleeping.load(
                        std::memory_order_acquire) &&
                        g_executorAlarmCallbackRanWhileSleeping
                            .load(
                                std::memory_order_acquire) &&
                        !mainReturnedBeforeStop &&
                        managedBeforeStop == 1u,
                    "the callback should run while the sole guest fiber remains scheduler-blocked");
                t.IsTrue(
                    mainHostThread != callerThread &&
                        callbackHostThread ==
                            mainHostThread,
                    "the timer worker must wake the one EE executor rather than borrowing guest execution");
                t.Equals(
                    runtime
                        .managedEeExecutionThreadCountForTesting(),
                    size_t{0u},
                    "stopping the idle fixture should destroy its suspended continuation");
            });

        tc.Run(
            "fiber sleep yields the executor until ordinary wake",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::
                        LegacyCppFiber;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                t.IsTrue(
                    memoryInitialized,
                    "the sleep fixture should allocate "
                    "runtime RDRAM");
                if (!memoryInitialized)
                {
                    return;
                }
                uint8_t *const rdram =
                    runtime.memory().getRDRAM();
                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        K_EXECUTOR_SLEEP_CHILD_ENTRY,
                        0x00310000u,
                        0x800u,
                        0u,
                        0u,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                runtime.registerFunction(
                    K_EXECUTOR_SLEEP_MAIN_ENTRY,
                    &dedicatedExecutorSleepMainHandler);
                runtime.registerFunction(
                    K_EXECUTOR_SLEEP_CHILD_ENTRY,
                    &dedicatedExecutorSleepChildHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_SLEEP_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);

                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    g_executorSleepOrder.clear();
                    g_executorSleepChildThreadId = 0;
                    g_executorSleepResult = KE_ERROR;
                    g_executorSecondSleepResult =
                        KE_ERROR;
                    g_executorWakeResult = KE_ERROR;
                    g_executorRawWakeResult = KE_ERROR;
                    g_executorMainAttemptedWake = false;
                    g_executorChildRanInsideWake = false;
                    g_executorChildRanInsideRawWake =
                        false;
                    g_executorWakeInProgress = false;
                    g_executorRawWakeInProgress = false;
                    g_executorSleepRescued = false;
                }

                std::thread rescue([&runtime]()
                {
                    const bool childCreated = waitUntil(
                        []()
                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            return g_executorSleepChildThreadId >
                                   0;
                        },
                        std::chrono::seconds(1));
                    if (!childCreated)
                    {
                        return;
                    }

                    const bool mainReachedWake = waitUntil(
                        []()
                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            return g_executorMainAttemptedWake;
                        },
                        std::chrono::milliseconds(200));
                    if (mainReachedWake)
                    {
                        return;
                    }

                    int childThreadId = 0;
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        childThreadId =
                            g_executorSleepChildThreadId;
                    }
                    const EeThreadHandle handle =
                        captureEeThreadHandle(
                            &runtime, childThreadId);
                    const bool rescued =
                        publishEeThreadWake(
                            &runtime, handle) ==
                        EeThreadWakeResult::WokeSleeper;
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        g_executorSleepRescued = rescued;
                    }
                });

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool completed = waitUntil(
                    []()
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        return g_executorSleepOrder.size() ==
                               4u;
                    },
                    std::chrono::seconds(2));
                rescue.join();
                runtime
                    .stopDedicatedEeExecutionForTesting();

                std::vector<int> order;
                int childThreadId = 0;
                int sleepResult = KE_ERROR;
                int secondSleepResult = KE_ERROR;
                int wakeResult = KE_ERROR;
                int rawWakeResult = KE_ERROR;
                bool childRanInsideWake = false;
                bool childRanInsideRawWake = false;
                bool rescued = false;
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    order = g_executorSleepOrder;
                    childThreadId =
                        g_executorSleepChildThreadId;
                    sleepResult = g_executorSleepResult;
                    secondSleepResult =
                        g_executorSecondSleepResult;
                    wakeResult = g_executorWakeResult;
                    rawWakeResult =
                        g_executorRawWakeResult;
                    childRanInsideWake =
                        g_executorChildRanInsideWake;
                    childRanInsideRawWake =
                        g_executorChildRanInsideRawWake;
                    rescued = g_executorSleepRescued;
                }

                t.IsTrue(
                    completed && !rescued &&
                        order ==
                            std::vector<int>{1, 2, 3, 4},
                    "SleepThread should yield the sole "
                    "executor without requiring a host "
                    "rescue wake");
                t.IsTrue(
                    sleepResult == childThreadId &&
                        secondSleepResult ==
                            childThreadId &&
                        wakeResult == childThreadId &&
                        rawWakeResult == childThreadId &&
                        childRanInsideWake &&
                        !childRanInsideRawWake,
                    "ordinary WakeupThread should dispatch "
                    "before returning while raw wake "
                    "should return to its caller first");
            });

        tc.Run(
            "external wake publication resumes an idle fiber executor",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::
                        LegacyCppFiber;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                t.IsTrue(
                    memoryInitialized,
                    "the external wake fixture should "
                    "allocate runtime RDRAM");
                if (!memoryInitialized)
                {
                    return;
                }
                uint8_t *const rdram =
                    runtime.memory().getRDRAM();
                const std::array<uint32_t, 9u>
                    threadParameters{
                        0u,
                        K_EXECUTOR_EXTERNAL_CHILD_ENTRY,
                        0x00310000u,
                        0x800u,
                        0u,
                        0u,
                        0u,
                        0u,
                        0u,
                    };
                writeGuestWords(
                    rdram,
                    K_EXECUTOR_THREAD_PARAM_ADDR,
                    threadParameters.data(),
                    threadParameters.size());
                runtime.registerFunction(
                    K_EXECUTOR_EXTERNAL_MAIN_ENTRY,
                    &dedicatedExecutorExternalMainHandler);
                runtime.registerFunction(
                    K_EXECUTOR_EXTERNAL_CHILD_ENTRY,
                    &dedicatedExecutorExternalChildHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_EXTERNAL_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    g_executorExternalChildThreadId = 0;
                    g_executorExternalSleepResult =
                        KE_ERROR;
                    g_executorExternalMainReturned =
                        false;
                    g_executorExternalChildReturned =
                        false;
                }

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool reachedIdleWait = waitUntil(
                    [&runtime]()
                    {
                        int childThreadId = 0;
                        bool mainReturned = false;
                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            childThreadId =
                                g_executorExternalChildThreadId;
                            mainReturned =
                                g_executorExternalMainReturned;
                        }
                        if (!mainReturned ||
                            childThreadId <= 0)
                        {
                            return false;
                        }
                        for (const auto &snapshot :
                             debugThreadSnapshots(
                                 &runtime))
                        {
                            if (snapshot.id ==
                                    childThreadId &&
                                snapshot.status ==
                                    THS_WAIT &&
                                snapshot.waitType ==
                                    TSW_SLEEP)
                            {
                                return true;
                            }
                        }
                        return false;
                    },
                    std::chrono::seconds(2));

                int childThreadId = 0;
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    childThreadId =
                        g_executorExternalChildThreadId;
                }
                EeThreadWakeResult wakeResult =
                    EeThreadWakeResult::InvalidHandle;
                if (reachedIdleWait)
                {
                    wakeResult = publishEeThreadWake(
                        &runtime,
                        captureEeThreadHandle(
                            &runtime,
                            childThreadId));
                }
                const bool childReturned = waitUntil(
                    []()
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        return g_executorExternalChildReturned;
                    },
                    std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                int sleepResult = KE_ERROR;
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    sleepResult =
                        g_executorExternalSleepResult;
                }
                t.IsTrue(
                    reachedIdleWait && childReturned &&
                        wakeResult ==
                            EeThreadWakeResult::
                                WokeSleeper &&
                        sleepResult == childThreadId,
                    "an external wake should cross the "
                    "executor publication queue and "
                    "resume the sleeping continuation");
            });

        tc.Run(
            "fiber semaphore wait preserves signal release and delete boundaries",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct FixtureResult
                {
                    bool memoryInitialized = false;
                    bool completed = false;
                    bool rescued = false;
                    std::vector<int> order;
                    int semaphoreId = KE_ERROR;
                    int childThreadId = 0;
                    int waitResult = KE_ERROR;
                    int operationResult = KE_ERROR;
                    bool childRanInsideOperation = false;
                    int rawStatus = KE_ERROR;
                    uint32_t rawWaitType = 0u;
                    uint32_t rawWaitId = 0u;
                    bool rawCompletionPending = false;
                };

                const auto runFixture =
                    [](ExecutorSemaOperation operation)
                    {
                        FixtureResult result{};
                        PS2RuntimeConfiguration
                            configuration{};
                        configuration.eeExecutionBackend =
                            EeExecutionBackendKind::
                                LegacyCppFiber;
                        PS2Runtime runtime(
                            configuration);
                        result.memoryInitialized =
                            runtime.memory().initialize();
                        if (!result.memoryInitialized)
                        {
                            return result;
                        }
                        uint8_t *const rdram =
                            runtime.memory().getRDRAM();
                        const std::array<uint32_t, 9u>
                            threadParameters{
                                0u,
                                K_EXECUTOR_SEMA_CHILD_ENTRY,
                                0x00310000u,
                                0x800u,
                                0u,
                                0u,
                                0u,
                                0u,
                                0u,
                            };
                        writeGuestWords(
                            rdram,
                            K_EXECUTOR_THREAD_PARAM_ADDR,
                            threadParameters.data(),
                            threadParameters.size());
                        runtime.registerFunction(
                            K_EXECUTOR_SEMA_MAIN_ENTRY,
                            &dedicatedExecutorSemaMainHandler);
                        runtime.registerFunction(
                            K_EXECUTOR_SEMA_CHILD_ENTRY,
                            &dedicatedExecutorSemaChildHandler);
                        runtime.cpu().pc =
                            K_EXECUTOR_SEMA_MAIN_ENTRY;
                        setRegU32(
                            runtime.cpu(),
                            29,
                            0x00300000u);

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            g_executorSemaOperation =
                                operation;
                            g_executorSemaOrder.clear();
                            g_executorSemaId = KE_ERROR;
                            g_executorSemaChildThreadId =
                                0;
                            g_executorSemaWaitResult =
                                KE_ERROR;
                            g_executorSemaSignalResult =
                                KE_ERROR;
                            g_executorSemaRawStatus =
                                KE_ERROR;
                            g_executorSemaRawWaitType =
                                0u;
                            g_executorSemaRawWaitId = 0u;
                            g_executorSemaRawCompletionPending =
                                false;
                            g_executorSemaSignalInProgress =
                                false;
                            g_executorSemaChildRanInsideSignal =
                                false;
                            g_executorSemaMainAttemptedSignal =
                                false;
                            g_executorSemaRescued = false;
                        }

                        std::thread rescue(
                            [&runtime, rdram]()
                            {
                                const bool created =
                                    waitUntil(
                                        []()
                                        {
                                            std::lock_guard<
                                                std::mutex>
                                                lock(
                                                    g_executorFixtureMutex);
                                            return g_executorSemaId >=
                                                   0;
                                        },
                                        std::chrono::
                                            seconds(1));
                                if (!created)
                                {
                                    return;
                                }
                                const bool mainReached =
                                    waitUntil(
                                        []()
                                        {
                                            std::lock_guard<
                                                std::mutex>
                                                lock(
                                                    g_executorFixtureMutex);
                                            return g_executorSemaMainAttemptedSignal;
                                        },
                                        std::chrono::
                                            milliseconds(
                                                200));
                                if (mainReached)
                                {
                                    return;
                                }

                                int semaphoreId =
                                    KE_ERROR;
                                {
                                    std::lock_guard<
                                        std::mutex>
                                        lock(
                                            g_executorFixtureMutex);
                                    semaphoreId =
                                        g_executorSemaId;
                                }
                                R5900Context
                                    signalContext{};
                                setRegU32(
                                    signalContext,
                                    4,
                                    static_cast<uint32_t>(
                                        semaphoreId));
                                iSignalSema(
                                    rdram,
                                    &signalContext,
                                    &runtime);
                                {
                                    std::lock_guard<
                                        std::mutex>
                                        lock(
                                            g_executorFixtureMutex);
                                    g_executorSemaRescued =
                                        true;
                                }
                            });

                        runtime
                            .startDedicatedEeExecutionForTesting();
                        result.completed = waitUntil(
                            []()
                            {
                                std::lock_guard<
                                    std::mutex>
                                    lock(
                                        g_executorFixtureMutex);
                                return g_executorSemaOrder
                                           .size() == 3u;
                            },
                            std::chrono::seconds(2));
                        rescue.join();
                        runtime
                            .stopDedicatedEeExecutionForTesting();

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            result.order =
                                g_executorSemaOrder;
                            result.semaphoreId =
                                g_executorSemaId;
                            result.childThreadId =
                                g_executorSemaChildThreadId;
                            result.waitResult =
                                g_executorSemaWaitResult;
                            result.operationResult =
                                g_executorSemaSignalResult;
                            result.childRanInsideOperation =
                                g_executorSemaChildRanInsideSignal;
                            result.rawStatus =
                                g_executorSemaRawStatus;
                            result.rawWaitType =
                                g_executorSemaRawWaitType;
                            result.rawWaitId =
                                g_executorSemaRawWaitId;
                            result.rawCompletionPending =
                                g_executorSemaRawCompletionPending;
                            result.rescued =
                                g_executorSemaRescued;
                        }
                        return result;
                    };

                const std::array<
                    std::pair<
                        ExecutorSemaOperation,
                        const char *>,
                    7u>
                    operations{{
                        {ExecutorSemaOperation::
                             OrdinarySignal,
                         "ordinary signal"},
                        {ExecutorSemaOperation::
                             RawSignal,
                         "raw signal"},
                        {ExecutorSemaOperation::
                             RawSignalThenDelete,
                         "raw signal then delete"},
                        {ExecutorSemaOperation::
                             OrdinaryRelease,
                         "ordinary release"},
                        {ExecutorSemaOperation::
                             RawRelease,
                         "raw release"},
                        {ExecutorSemaOperation::
                             OrdinaryDelete,
                         "ordinary delete"},
                        {ExecutorSemaOperation::
                             RawDelete,
                         "raw delete"},
                    }};
                for (const auto &[operation, name] :
                     operations)
                {
                    const FixtureResult result =
                        runFixture(operation);
                    const bool raw =
                        operation ==
                            ExecutorSemaOperation::
                                RawSignal ||
                        operation ==
                            ExecutorSemaOperation::
                                RawSignalThenDelete ||
                        operation ==
                            ExecutorSemaOperation::
                                RawRelease ||
                        operation ==
                            ExecutorSemaOperation::
                                RawDelete;
                    const bool signal =
                        operation ==
                            ExecutorSemaOperation::
                                OrdinarySignal ||
                        operation ==
                            ExecutorSemaOperation::
                                RawSignal ||
                        operation ==
                            ExecutorSemaOperation::
                                RawSignalThenDelete;
                    const bool release =
                        operation ==
                            ExecutorSemaOperation::
                                OrdinaryRelease ||
                        operation ==
                            ExecutorSemaOperation::
                                RawRelease;
                    const bool rawDelete =
                        operation ==
                        ExecutorSemaOperation::
                            RawDelete;

                    t.IsTrue(
                        result.memoryInitialized &&
                            result.completed &&
                            !result.rescued &&
                            result.order ==
                                (raw
                                     ? std::vector<int>{
                                           1, 3, 2}
                                     : std::vector<int>{
                                           1, 2, 3}),
                        std::string(name) +
                            " should release the sole "
                            "executor with the retained "
                            "ordinary/raw dispatch order");
                    t.IsTrue(
                        result.semaphoreId >= 0 &&
                            result.waitResult ==
                                (signal
                                     ? result.semaphoreId
                                     : KE_ERROR) &&
                            result.operationResult ==
                                (release
                                     ? result.childThreadId
                                     : result.semaphoreId) &&
                            result.childRanInsideOperation ==
                                !raw,
                        std::string(name) +
                            " should preserve syscall "
                            "results and dispatch timing");
                    if (raw)
                    {
                        t.IsTrue(
                            result.rawStatus ==
                                    THS_READY &&
                                result.rawWaitType ==
                                    (rawDelete
                                         ? TSW_SEMA
                                         : 0u) &&
                                result.rawWaitId ==
                                    (rawDelete
                                         ? static_cast<
                                               uint32_t>(
                                               result
                                                   .semaphoreId)
                                         : 0u) &&
                                result
                                        .rawCompletionPending ==
                                    rawDelete,
                            std::string(name) +
                                " should synchronously "
                                "publish the retained "
                                "pre-dispatch status");
                    }
                }
            });

        tc.Run(
            "fiber event wait preserves set release and delete boundaries",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct FixtureResult
                {
                    bool memoryInitialized = false;
                    bool completed = false;
                    bool rescued = false;
                    std::vector<int> order;
                    int eventId = KE_ERROR;
                    int childThreadId = 0;
                    int waitResult = KE_ERROR;
                    int operationResult = KE_ERROR;
                    uint32_t resultBits = 0u;
                    bool childRanInsideOperation = false;
                    int rawStatus = KE_ERROR;
                    uint32_t rawWaitType = 0u;
                };

                const auto runFixture =
                    [](ExecutorEventOperation operation)
                    {
                        FixtureResult result{};
                        PS2RuntimeConfiguration
                            configuration{};
                        configuration.eeExecutionBackend =
                            EeExecutionBackendKind::
                                LegacyCppFiber;
                        PS2Runtime runtime(
                            configuration);
                        result.memoryInitialized =
                            runtime.memory().initialize();
                        if (!result.memoryInitialized)
                        {
                            return result;
                        }
                        uint8_t *const rdram =
                            runtime.memory().getRDRAM();
                        const std::array<uint32_t, 9u>
                            threadParameters{
                                0u,
                                K_EXECUTOR_EVENT_CHILD_ENTRY,
                                0x00310000u,
                                0x800u,
                                0u,
                                0u,
                                0u,
                                0u,
                                0u,
                            };
                        writeGuestWords(
                            rdram,
                            K_EXECUTOR_THREAD_PARAM_ADDR,
                            threadParameters.data(),
                            threadParameters.size());
                        runtime.registerFunction(
                            K_EXECUTOR_EVENT_MAIN_ENTRY,
                            &dedicatedExecutorEventMainHandler);
                        runtime.registerFunction(
                            K_EXECUTOR_EVENT_CHILD_ENTRY,
                            &dedicatedExecutorEventChildHandler);
                        runtime.cpu().pc =
                            K_EXECUTOR_EVENT_MAIN_ENTRY;
                        setRegU32(
                            runtime.cpu(),
                            29,
                            0x00300000u);

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            g_executorEventOperation =
                                operation;
                            g_executorEventOrder.clear();
                            g_executorEventId = KE_ERROR;
                            g_executorEventChildThreadId =
                                0;
                            g_executorEventWaitResult =
                                KE_ERROR;
                            g_executorEventOperationResult =
                                KE_ERROR;
                            g_executorEventResultBits =
                                0u;
                            g_executorEventRawStatus =
                                KE_ERROR;
                            g_executorEventRawWaitType =
                                0u;
                            g_executorEventOperationInProgress =
                                false;
                            g_executorEventChildRanInsideOperation =
                                false;
                            g_executorEventMainAttemptedOperation =
                                false;
                            g_executorEventRescued = false;
                        }

                        std::thread rescue(
                            [&runtime, rdram]()
                            {
                                const bool created =
                                    waitUntil(
                                        []()
                                        {
                                            std::lock_guard<
                                                std::mutex>
                                                lock(
                                                    g_executorFixtureMutex);
                                            return g_executorEventId >
                                                   0;
                                        },
                                        std::chrono::
                                            seconds(1));
                                if (!created)
                                {
                                    return;
                                }
                                const bool mainReached =
                                    waitUntil(
                                        []()
                                        {
                                            std::lock_guard<
                                                std::mutex>
                                                lock(
                                                    g_executorFixtureMutex);
                                            return g_executorEventMainAttemptedOperation;
                                        },
                                        std::chrono::
                                            milliseconds(
                                                200));
                                if (mainReached)
                                {
                                    return;
                                }

                                int eventId = KE_ERROR;
                                {
                                    std::lock_guard<
                                        std::mutex>
                                        lock(
                                            g_executorFixtureMutex);
                                    eventId =
                                        g_executorEventId;
                                }
                                R5900Context
                                    deleteContext{};
                                setRegU32(
                                    deleteContext,
                                    4,
                                    static_cast<uint32_t>(
                                        eventId));
                                DeleteEventFlag(
                                    rdram,
                                    &deleteContext,
                                    &runtime);
                                {
                                    std::lock_guard<
                                        std::mutex>
                                        lock(
                                            g_executorFixtureMutex);
                                    g_executorEventRescued =
                                        true;
                                }
                            });

                        runtime
                            .startDedicatedEeExecutionForTesting();
                        result.completed = waitUntil(
                            []()
                            {
                                std::lock_guard<
                                    std::mutex>
                                    lock(
                                        g_executorFixtureMutex);
                                return g_executorEventOrder
                                           .size() == 3u;
                            },
                            std::chrono::seconds(2));
                        rescue.join();
                        runtime
                            .stopDedicatedEeExecutionForTesting();

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            result.order =
                                g_executorEventOrder;
                            result.eventId =
                                g_executorEventId;
                            result.childThreadId =
                                g_executorEventChildThreadId;
                            result.waitResult =
                                g_executorEventWaitResult;
                            result.operationResult =
                                g_executorEventOperationResult;
                            result.resultBits =
                                g_executorEventResultBits;
                            result.childRanInsideOperation =
                                g_executorEventChildRanInsideOperation;
                            result.rawStatus =
                                g_executorEventRawStatus;
                            result.rawWaitType =
                                g_executorEventRawWaitType;
                            result.rescued =
                                g_executorEventRescued;
                        }
                        return result;
                    };

                const std::array<
                    std::pair<
                        ExecutorEventOperation,
                        const char *>,
                    6u>
                    operations{{
                        {ExecutorEventOperation::
                             OrdinarySet,
                         "ordinary set"},
                        {ExecutorEventOperation::RawSet,
                         "raw set"},
                        {ExecutorEventOperation::
                             RawSetThenClear,
                         "raw set then clear"},
                        {ExecutorEventOperation::
                             OrdinaryRelease,
                         "ordinary release"},
                        {ExecutorEventOperation::
                             RawRelease,
                         "raw release"},
                        {ExecutorEventOperation::Delete,
                         "delete"},
                    }};
                for (const auto &[operation, name] :
                     operations)
                {
                    const FixtureResult result =
                        runFixture(operation);
                    const bool raw =
                        operation ==
                            ExecutorEventOperation::
                                RawSet ||
                        operation ==
                            ExecutorEventOperation::
                                RawSetThenClear ||
                        operation ==
                            ExecutorEventOperation::
                                RawRelease;
                    const bool set =
                        operation ==
                            ExecutorEventOperation::
                                OrdinarySet ||
                        operation ==
                            ExecutorEventOperation::
                                RawSet ||
                        operation ==
                            ExecutorEventOperation::
                                RawSetThenClear;
                    const bool release =
                        operation ==
                            ExecutorEventOperation::
                                OrdinaryRelease ||
                        operation ==
                            ExecutorEventOperation::
                                RawRelease;

                    t.IsTrue(
                        result.memoryInitialized &&
                            result.completed &&
                            !result.rescued &&
                            result.order ==
                                (raw
                                     ? std::vector<int>{
                                           1, 3, 2}
                                     : std::vector<int>{
                                           1, 2, 3}),
                        std::string(name) +
                            " should release the sole "
                            "executor with ordinary/raw "
                            "dispatch order");
                    t.IsTrue(
                        result.eventId > 0 &&
                            result.waitResult ==
                                (set
                                     ? KE_OK
                                     : (release
                                            ? KE_RELEASE_WAIT
                                            : KE_WAIT_DELETE)) &&
                            result.operationResult ==
                                (release
                                     ? result.childThreadId
                                     : KE_OK) &&
                            result.resultBits ==
                                (set
                                     ? 0x4u
                                     : 0x7f7f7f7fu) &&
                            result.childRanInsideOperation ==
                                !raw,
                        std::string(name) +
                            " should preserve event "
                            "results and dispatch timing");
                    if (raw)
                    {
                        t.IsTrue(
                            result.rawStatus ==
                                    THS_READY &&
                                result.rawWaitType == 0u,
                            std::string(name) +
                                " should synchronously "
                                "publish READY with no "
                                "retained wait reason");
                    }
                }
            });

        tc.Run(
            "fiber event wait applies FIFO clear effects",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::
                        LegacyCppFiber;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                t.IsTrue(
                    memoryInitialized,
                    "the multi-wait event fixture should "
                    "allocate runtime RDRAM");
                if (!memoryInitialized)
                {
                    return;
                }
                runtime.registerFunction(
                    K_EXECUTOR_EVENT_MULTI_MAIN_ENTRY,
                    &dedicatedExecutorEventMultiMainHandler);
                runtime.registerFunction(
                    K_EXECUTOR_EVENT_CLEAR_CHILD_ENTRY,
                    &dedicatedExecutorEventClearChildHandler);
                runtime.registerFunction(
                    K_EXECUTOR_EVENT_RETAIN_CHILD_ENTRY,
                    &dedicatedExecutorEventRetainChildHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_EVENT_MULTI_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    g_executorEventMultiOrder.clear();
                    g_executorEventMultiId = KE_ERROR;
                    g_executorEventClearResult =
                        KE_ERROR;
                    g_executorEventRetainResult =
                        KE_ERROR;
                    g_executorEventClearBits = 0u;
                    g_executorEventRetainBits = 0u;
                    g_executorEventRetainWaitedAfterFirstSet =
                        false;
                }

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool completed = waitUntil(
                    []()
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        return g_executorEventMultiOrder
                                   .size() == 6u;
                    },
                    std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                std::vector<int> order;
                int eventId = KE_ERROR;
                int clearResult = KE_ERROR;
                int retainResult = KE_ERROR;
                uint32_t clearBits = 0u;
                uint32_t retainBits = 0u;
                bool retainedWait = false;
                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    order = g_executorEventMultiOrder;
                    eventId = g_executorEventMultiId;
                    clearResult =
                        g_executorEventClearResult;
                    retainResult =
                        g_executorEventRetainResult;
                    clearBits =
                        g_executorEventClearBits;
                    retainBits =
                        g_executorEventRetainBits;
                    retainedWait =
                        g_executorEventRetainWaitedAfterFirstSet;
                }

                t.IsTrue(
                    completed && eventId > 0 &&
                        order ==
                            std::vector<int>{
                                1, 2, 3, 4, 5, 6},
                    "the first FIFO waiter should run "
                    "inside the first set and the second "
                    "inside the second set");
                t.IsTrue(
                    clearResult == KE_OK &&
                        retainResult == KE_OK &&
                        clearBits == 0x4u &&
                        retainBits == 0x4u &&
                        retainedWait,
                    "the first waiter's clear should "
                    "consume the first publication before "
                    "the second predicate is evaluated");
            });

        tc.Run(
            "fiber suspend and resume preserve executor ownership",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct FixtureResult
                {
                    bool memoryInitialized = false;
                    bool completed = false;
                    std::vector<int> order;
                    int childThreadId = 0;
                    int waitResult = KE_ERROR;
                    int suspendResult = KE_ERROR;
                    int wakeResult = KE_ERROR;
                    int resumeResult = KE_ERROR;
                    int statusBeforeWake = KE_ERROR;
                    int statusAfterWake = KE_ERROR;
                };

                const auto runFixture =
                    [](ExecutorSuspendScenario scenario)
                    {
                        FixtureResult result{};
                        PS2RuntimeConfiguration
                            configuration{};
                        configuration.eeExecutionBackend =
                            EeExecutionBackendKind::
                                LegacyCppFiber;
                        PS2Runtime runtime(configuration);
                        result.memoryInitialized =
                            runtime.memory().initialize();
                        if (!result.memoryInitialized)
                        {
                            return result;
                        }
                        uint8_t *const rdram =
                            runtime.memory().getRDRAM();
                        const std::array<uint32_t, 9u>
                            threadParameters{
                                0u,
                                K_EXECUTOR_SUSPEND_CHILD_ENTRY,
                                0x00310000u,
                                0x800u,
                                0u,
                                0u,
                                0u,
                                0u,
                                0u,
                            };
                        writeGuestWords(
                            rdram,
                            K_EXECUTOR_THREAD_PARAM_ADDR,
                            threadParameters.data(),
                            threadParameters.size());
                        runtime.registerFunction(
                            K_EXECUTOR_SUSPEND_MAIN_ENTRY,
                            &dedicatedExecutorSuspendMainHandler);
                        runtime.registerFunction(
                            K_EXECUTOR_SUSPEND_CHILD_ENTRY,
                            &dedicatedExecutorSuspendChildHandler);
                        runtime.cpu().pc =
                            K_EXECUTOR_SUSPEND_MAIN_ENTRY;
                        setRegU32(
                            runtime.cpu(),
                            29,
                            0x00300000u);

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            g_executorSuspendScenario =
                                scenario;
                            g_executorSuspendOrder.clear();
                            g_executorSuspendChildThreadId =
                                0;
                            g_executorSuspendWaitResult =
                                KE_ERROR;
                            g_executorSuspendResult =
                                KE_ERROR;
                            g_executorSuspendWakeResult =
                                KE_ERROR;
                            g_executorSuspendResumeResult =
                                KE_ERROR;
                            g_executorSuspendStatusBeforeWake =
                                KE_ERROR;
                            g_executorSuspendStatusAfterWake =
                                KE_ERROR;
                        }

                        runtime
                            .startDedicatedEeExecutionForTesting();
                        result.completed = waitUntil(
                            []()
                            {
                                std::lock_guard<
                                    std::mutex>
                                    lock(
                                        g_executorFixtureMutex);
                                return g_executorSuspendOrder
                                           .size() == 4u;
                            },
                            std::chrono::seconds(2));
                        runtime
                            .stopDedicatedEeExecutionForTesting();

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            result.order =
                                g_executorSuspendOrder;
                            result.childThreadId =
                                g_executorSuspendChildThreadId;
                            result.waitResult =
                                g_executorSuspendWaitResult;
                            result.suspendResult =
                                g_executorSuspendResult;
                            result.wakeResult =
                                g_executorSuspendWakeResult;
                            result.resumeResult =
                                g_executorSuspendResumeResult;
                            result.statusBeforeWake =
                                g_executorSuspendStatusBeforeWake;
                            result.statusAfterWake =
                                g_executorSuspendStatusAfterWake;
                        }
                        return result;
                    };

                const std::array<
                    std::pair<
                        ExecutorSuspendScenario,
                        const char *>,
                    4u>
                    scenarios{{
                        {ExecutorSuspendScenario::
                             SelfOrdinaryResume,
                         "self suspend / ordinary resume"},
                        {ExecutorSuspendScenario::
                             SelfRawResume,
                         "self suspend / raw resume"},
                        {ExecutorSuspendScenario::
                             WaitSuspendOrdinary,
                         "wait-suspend / ordinary resume"},
                        {ExecutorSuspendScenario::
                             WaitSuspendRaw,
                         "wait-suspend / raw resume"},
                    }};
                for (const auto &[scenario, name] :
                     scenarios)
                {
                    const FixtureResult result =
                        runFixture(scenario);
                    const bool waitSuspend =
                        scenario ==
                            ExecutorSuspendScenario::
                                WaitSuspendOrdinary ||
                        scenario ==
                            ExecutorSuspendScenario::
                                WaitSuspendRaw;
                    const bool raw =
                        scenario ==
                            ExecutorSuspendScenario::
                                SelfRawResume ||
                        scenario ==
                            ExecutorSuspendScenario::
                                WaitSuspendRaw;
                    const std::vector<int> expectedOrder =
                        raw
                            ? std::vector<int>{1, 2, 4, 3}
                            : std::vector<int>{1, 2, 3, 4};

                    t.IsTrue(
                        result.memoryInitialized &&
                            result.completed &&
                            result.order == expectedOrder,
                        std::string(name) +
                            " should switch only through "
                            "the one executor");
                    t.IsTrue(
                        result.childThreadId > 1 &&
                            result.waitResult ==
                                result.childThreadId &&
                            result.resumeResult ==
                                result.childThreadId &&
                            result.statusBeforeWake ==
                                (waitSuspend
                                     ? THS_WAITSUSPEND
                                     : THS_SUSPEND),
                        std::string(name) +
                            " should preserve syscall "
                            "results and suspended status");
                    if (waitSuspend)
                    {
                        t.IsTrue(
                            result.suspendResult ==
                                    result.childThreadId &&
                                result.wakeResult ==
                                    result.childThreadId &&
                                result.statusAfterWake ==
                                    THS_SUSPEND,
                            std::string(name) +
                                " should move WAIT through "
                                "WAITSUSPEND to SUSPEND");
                    }
                }
            });

        tc.Run(
            "fiber termination retires blocked continuations",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct FixtureResult
                {
                    bool memoryInitialized = false;
                    bool completed = false;
                    std::vector<int> order;
                    int objectId = KE_ERROR;
                    int candidateThreadId = 0;
                    int targetThreadId = 0;
                    int terminateResult = KE_ERROR;
                    int targetStatus = KE_ERROR;
                    int waiters = -1;
                    int deleteResult = KE_ERROR;
                    bool targetReturned = false;
                    bool schedulerTargetPresent = true;
                    size_t managedContinuations = 1u;
                };

                const auto runFixture =
                    [](ExecutorTerminateOperation operation)
                    {
                        FixtureResult result{};
                        PS2RuntimeConfiguration
                            configuration{};
                        configuration.eeExecutionBackend =
                            EeExecutionBackendKind::
                                LegacyCppFiber;
                        PS2Runtime runtime(configuration);
                        result.memoryInitialized =
                            runtime.memory().initialize();
                        if (!result.memoryInitialized)
                        {
                            return result;
                        }
                        uint8_t *const rdram =
                            runtime.memory().getRDRAM();
                        runtime.registerFunction(
                            K_EXECUTOR_TERMINATE_MAIN_ENTRY,
                            &dedicatedExecutorTerminateMainHandler);
                        runtime.registerFunction(
                            K_EXECUTOR_TERMINATE_CANDIDATE_ENTRY,
                            &dedicatedExecutorTerminateCandidateHandler);
                        runtime.registerFunction(
                            K_EXECUTOR_TERMINATE_TARGET_ENTRY,
                            &dedicatedExecutorTerminateTargetHandler);
                        runtime.cpu().pc =
                            K_EXECUTOR_TERMINATE_MAIN_ENTRY;
                        setRegU32(
                            runtime.cpu(),
                            29,
                            0x00300000u);

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            g_executorTerminateOperation =
                                operation;
                            g_executorTerminateOrder.clear();
                            g_executorTerminateObjectId =
                                KE_ERROR;
                            g_executorTerminateCandidateThreadId =
                                0;
                            g_executorTerminateTargetThreadId =
                                0;
                            g_executorTerminateResult =
                                KE_ERROR;
                            g_executorTerminateStatus =
                                KE_ERROR;
                            g_executorTerminateWaiters =
                                -1;
                            g_executorTerminateDeleteResult =
                                KE_ERROR;
                            g_executorTerminateTargetReturned =
                                false;
                        }

                        runtime
                            .startDedicatedEeExecutionForTesting();
                        result.completed = waitUntil(
                            []()
                            {
                                std::lock_guard<
                                    std::mutex>
                                    lock(
                                        g_executorFixtureMutex);
                                return g_executorTerminateOrder
                                           .size() == 5u;
                            },
                            std::chrono::seconds(2));
                        const bool continuationsRetired =
                            waitUntil(
                                [&runtime]()
                                {
                                    return runtime
                                               .managedEeExecutionThreadCountForTesting() ==
                                           0u;
                                },
                                std::chrono::
                                    milliseconds(200));

                        int targetThreadId = 0;
                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            targetThreadId =
                                g_executorTerminateTargetThreadId;
                        }
                        std::optional<
                            ps2x::ee::
                                EeSchedulerThreadSnapshot>
                            schedulerTarget;
                        runtime
                            .invokeEeSchedulerUpdateAtBoundary(
                                [&schedulerTarget,
                                 targetThreadId](
                                    ps2x::ee::
                                        EeThreadScheduler
                                            &scheduler)
                                {
                                    schedulerTarget =
                                        scheduler.thread(
                                            targetThreadId);
                                },
                                ps2x::ee::
                                    EeSchedulerReschedulePolicy::
                                        None);
                        result.schedulerTargetPresent =
                            schedulerTarget.has_value();
                        result.managedContinuations =
                            runtime
                                .managedEeExecutionThreadCountForTesting();
                        runtime
                            .stopDedicatedEeExecutionForTesting();

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            result.order =
                                g_executorTerminateOrder;
                            result.objectId =
                                g_executorTerminateObjectId;
                            result.candidateThreadId =
                                g_executorTerminateCandidateThreadId;
                            result.targetThreadId =
                                g_executorTerminateTargetThreadId;
                            result.terminateResult =
                                g_executorTerminateResult;
                            result.targetStatus =
                                g_executorTerminateStatus;
                            result.waiters =
                                g_executorTerminateWaiters;
                            result.deleteResult =
                                g_executorTerminateDeleteResult;
                            result.targetReturned =
                                g_executorTerminateTargetReturned;
                        }
                        if (!continuationsRetired)
                        {
                            result.managedContinuations =
                                std::max<size_t>(
                                    1u,
                                    result
                                        .managedContinuations);
                        }
                        return result;
                    };

                const std::array<
                    std::pair<
                        ExecutorTerminateOperation,
                        const char *>,
                    4u>
                    operations{{
                        {ExecutorTerminateOperation::
                             OrdinarySemaphore,
                         "ordinary semaphore termination"},
                        {ExecutorTerminateOperation::
                             RawSemaphore,
                         "raw semaphore termination"},
                        {ExecutorTerminateOperation::
                             OrdinaryEvent,
                         "ordinary event termination"},
                        {ExecutorTerminateOperation::
                             RawEvent,
                         "raw event termination"},
                    }};
                for (const auto &[operation, name] :
                     operations)
                {
                    const FixtureResult result =
                        runFixture(operation);
                    const bool raw =
                        operation ==
                            ExecutorTerminateOperation::
                                RawSemaphore ||
                        operation ==
                            ExecutorTerminateOperation::
                                RawEvent;
                    const std::vector<int> expectedOrder =
                        raw
                            ? std::vector<int>{
                                  1, 2, 3, 5, 4}
                            : std::vector<int>{
                                  1, 2, 3, 4, 5};

                    t.IsTrue(
                        result.memoryInitialized &&
                            result.completed &&
                            result.order == expectedOrder,
                        std::string(name) +
                            " should preserve the "
                            "ordinary/raw boundary");
                    t.IsTrue(
                        result.objectId >= 0 &&
                            result.candidateThreadId > 1 &&
                            result.targetThreadId > 1 &&
                            result.terminateResult ==
                                result.targetThreadId &&
                            result.targetStatus ==
                                THS_DORMANT &&
                            result.waiters == 0 &&
                            !result.targetReturned,
                        std::string(name) +
                            " should synchronously publish "
                            "DORMANT and unlink its wait");
                    t.IsTrue(
                        result.deleteResult ==
                                result.targetThreadId &&
                            !result.schedulerTargetPresent &&
                            result.managedContinuations ==
                                0u,
                        std::string(name) +
                            " should delete scheduler "
                            "ownership and its fiber");
                }
            });

        tc.Run(
            "fiber self exit preserves or removes scheduler objects",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct FixtureResult
                {
                    bool memoryInitialized = false;
                    bool completed = false;
                    int exitThreadId = 0;
                    int exitDeleteThreadId = 0;
                    int exitThreadStartResult = KE_ERROR;
                    int exitDeleteThreadStartResult =
                        KE_ERROR;
                    int exitThreadStatusResult = KE_ERROR;
                    int32_t exitThreadStatusPayload = 0;
                    int exitThreadDeleteResult = KE_ERROR;
                    int exitDeleteThreadStatusResult =
                        KE_ERROR;
                    int32_t exitDeleteThreadStatusPayload =
                        0;
                    int exitDeleteThreadDeleteResult =
                        KE_ERROR;
                    uint32_t exitThreadStage = 0u;
                    uint32_t exitDeleteThreadStage = 0u;
                    uint32_t exitThreadObservedId = 0u;
                    uint32_t exitDeleteThreadObservedId =
                        0u;
                    bool schedulerExitThreadPresent = true;
                    bool schedulerExitDeleteThreadPresent =
                        true;
                    size_t managedContinuations = 1u;
                };

                FixtureResult result{};
                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::LegacyCppFiber;
                PS2Runtime runtime(configuration);
                result.memoryInitialized =
                    runtime.memory().initialize();
                if (!result.memoryInitialized)
                {
                    t.IsTrue(
                        false,
                        "fiber self-exit fixture memory should initialize");
                    return;
                }

                uint8_t *const rdram =
                    runtime.memory().getRDRAM();
                runtime.registerFunction(
                    K_EXECUTOR_EXIT_MAIN_ENTRY,
                    &dedicatedExecutorExitMainHandler);
                runtime.registerFunction(
                    K_EXECUTOR_EXIT_THREAD_ENTRY,
                    &selfExitThreadHandler);
                runtime.registerFunction(
                    K_EXECUTOR_EXIT_DELETE_THREAD_ENTRY,
                    &selfExitDeleteThreadHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_EXIT_MAIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);

                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    g_executorExitCompleted = false;
                    g_executorExitThreadId = 0;
                    g_executorExitDeleteThreadId = 0;
                    g_executorExitThreadStartResult =
                        KE_ERROR;
                    g_executorExitDeleteThreadStartResult =
                        KE_ERROR;
                    g_executorExitThreadStatusResult =
                        KE_ERROR;
                    g_executorExitThreadStatusPayload = 0;
                    g_executorExitThreadDeleteResult =
                        KE_ERROR;
                    g_executorExitDeleteThreadStatusResult =
                        KE_ERROR;
                    g_executorExitDeleteThreadStatusPayload =
                        0;
                    g_executorExitDeleteThreadDeleteResult =
                        KE_ERROR;
                }

                runtime
                    .startDedicatedEeExecutionForTesting();
                result.completed = waitUntil(
                    []()
                    {
                        std::lock_guard<std::mutex> lock(
                            g_executorFixtureMutex);
                        return g_executorExitCompleted;
                    },
                    std::chrono::seconds(2));
                const bool continuationsRetired =
                    waitUntil(
                        [&runtime]()
                        {
                            return runtime
                                       .managedEeExecutionThreadCountForTesting() ==
                                   0u;
                        },
                        std::chrono::milliseconds(200));

                {
                    std::lock_guard<std::mutex> lock(
                        g_executorFixtureMutex);
                    result.exitThreadId =
                        g_executorExitThreadId;
                    result.exitDeleteThreadId =
                        g_executorExitDeleteThreadId;
                    result.exitThreadStartResult =
                        g_executorExitThreadStartResult;
                    result.exitDeleteThreadStartResult =
                        g_executorExitDeleteThreadStartResult;
                    result.exitThreadStatusResult =
                        g_executorExitThreadStatusResult;
                    result.exitThreadStatusPayload =
                        g_executorExitThreadStatusPayload;
                    result.exitThreadDeleteResult =
                        g_executorExitThreadDeleteResult;
                    result.exitDeleteThreadStatusResult =
                        g_executorExitDeleteThreadStatusResult;
                    result.exitDeleteThreadStatusPayload =
                        g_executorExitDeleteThreadStatusPayload;
                    result.exitDeleteThreadDeleteResult =
                        g_executorExitDeleteThreadDeleteResult;
                }

                std::optional<
                    ps2x::ee::EeSchedulerThreadSnapshot>
                    schedulerExitThread;
                std::optional<
                    ps2x::ee::EeSchedulerThreadSnapshot>
                    schedulerExitDeleteThread;
                runtime.invokeEeSchedulerUpdateAtBoundary(
                    [&schedulerExitThread,
                     &schedulerExitDeleteThread,
                     &result](
                        ps2x::ee::EeThreadScheduler
                            &scheduler)
                    {
                        schedulerExitThread =
                            scheduler.thread(
                                result.exitThreadId);
                        schedulerExitDeleteThread =
                            scheduler.thread(
                                result.exitDeleteThreadId);
                    },
                    ps2x::ee::
                        EeSchedulerReschedulePolicy::None);
                result.schedulerExitThreadPresent =
                    schedulerExitThread.has_value();
                result.schedulerExitDeleteThreadPresent =
                    schedulerExitDeleteThread.has_value();
                result.managedContinuations =
                    runtime
                        .managedEeExecutionThreadCountForTesting();
                runtime
                    .stopDedicatedEeExecutionForTesting();

                result.exitThreadStage =
                    readGuestU32(
                        rdram,
                        K_EXIT_THREAD_STAGE_ADDR);
                result.exitDeleteThreadStage =
                    readGuestU32(
                        rdram,
                        K_EXIT_DELETE_THREAD_STAGE_ADDR);
                result.exitThreadObservedId =
                    readGuestU32(
                        rdram,
                        K_EXIT_THREAD_ID_ADDR);
                result.exitDeleteThreadObservedId =
                    readGuestU32(
                        rdram,
                        K_EXIT_DELETE_THREAD_ID_ADDR);
                if (!continuationsRetired)
                {
                    result.managedContinuations =
                        std::max<size_t>(
                            1u,
                            result.managedContinuations);
                }

                t.IsTrue(
                    result.memoryInitialized &&
                        result.completed &&
                        result.exitThreadId > 1 &&
                        result.exitDeleteThreadId > 1 &&
                        result.exitThreadId !=
                            result.exitDeleteThreadId &&
                        result.exitThreadStartResult ==
                            result.exitThreadId &&
                        result.exitDeleteThreadStartResult ==
                            result.exitDeleteThreadId &&
                        result.exitThreadStage == 1u &&
                        result.exitDeleteThreadStage == 1u &&
                        result.exitThreadObservedId ==
                            static_cast<uint32_t>(
                                result.exitThreadId) &&
                        result.exitDeleteThreadObservedId ==
                            static_cast<uint32_t>(
                                result.exitDeleteThreadId),
                    "self-exit workers should stop at their syscalls and return control to the executor");
                t.IsTrue(
                    result.exitThreadStatusResult ==
                            THS_DORMANT &&
                        result.exitThreadStatusPayload ==
                            THS_DORMANT &&
                        result.exitThreadDeleteResult ==
                            result.exitThreadId &&
                        result.exitDeleteThreadStatusResult ==
                            KE_OK &&
                        result.exitDeleteThreadStatusPayload ==
                            0x5A5A5A5A &&
                        result.exitDeleteThreadDeleteResult ==
                            KE_ERROR,
                    "self exit should preserve the guest object while exit-delete should remove it");
                t.IsTrue(
                    !result.schedulerExitThreadPresent &&
                        !result
                             .schedulerExitDeleteThreadPresent &&
                        result.managedContinuations == 0u,
                    "self-exit lifecycle should retire scheduler and continuation ownership exactly once");
            });

        tc.Run(
            "fiber priority and rotation preserve ordinary and raw boundaries",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                struct FixtureResult
                {
                    bool memoryInitialized = false;
                    bool completed = false;
                    std::vector<int> order;
                    int setupResult = KE_ERROR;
                    int operationResult = KE_ERROR;
                    int triggerResult = KE_ERROR;
                    int statusValue = -1;
                    int firstThreadId = 0;
                    int secondThreadId = 0;
                    int schedulerMainPriority = -1;
                    int schedulerFirstPriority = -1;
                    int schedulerSecondPriority = -1;
                    bool schedulerValid = false;
                    size_t managedContinuations = 1u;
                };

                const auto runFixture =
                    [](ExecutorPriorityScenario scenario)
                    {
                        FixtureResult result{};
                        PS2RuntimeConfiguration
                            configuration{};
                        configuration.eeExecutionBackend =
                            EeExecutionBackendKind::
                                LegacyCppFiber;
                        PS2Runtime runtime(configuration);
                        result.memoryInitialized =
                            runtime.memory().initialize();
                        if (!result.memoryInitialized)
                        {
                            return result;
                        }

                        runtime.registerFunction(
                            K_EXECUTOR_PRIORITY_MAIN_ENTRY,
                            &dedicatedExecutorPriorityMainHandler);
                        runtime.registerFunction(
                            K_EXECUTOR_PRIORITY_MARKER_ENTRY,
                            &dedicatedExecutorPriorityMarkerHandler);
                        runtime.cpu().pc =
                            K_EXECUTOR_PRIORITY_MAIN_ENTRY;
                        setRegU32(
                            runtime.cpu(),
                            29,
                            0x00300000u);
                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            g_executorPriorityScenario =
                                scenario;
                            g_executorPriorityCompleted =
                                false;
                            g_executorPriorityOrder.clear();
                            g_executorPrioritySetupResult =
                                KE_ERROR;
                            g_executorPriorityOperationResult =
                                KE_ERROR;
                            g_executorPriorityTriggerResult =
                                KE_ERROR;
                            g_executorPriorityStatusValue =
                                -1;
                            g_executorPriorityFirstThreadId =
                                0;
                            g_executorPrioritySecondThreadId =
                                0;
                        }

                        runtime
                            .startDedicatedEeExecutionForTesting();
                        result.completed = waitUntil(
                            []()
                            {
                                std::lock_guard<
                                    std::mutex>
                                    lock(
                                        g_executorFixtureMutex);
                                return g_executorPriorityCompleted;
                            },
                            std::chrono::seconds(2));
                        const bool continuationsRetired =
                            waitUntil(
                                [&runtime]()
                                {
                                    return runtime
                                               .managedEeExecutionThreadCountForTesting() ==
                                           0u;
                                },
                                std::chrono::
                                    milliseconds(200));

                        {
                            std::lock_guard<std::mutex>
                                lock(
                                    g_executorFixtureMutex);
                            result.order =
                                g_executorPriorityOrder;
                            result.setupResult =
                                g_executorPrioritySetupResult;
                            result.operationResult =
                                g_executorPriorityOperationResult;
                            result.triggerResult =
                                g_executorPriorityTriggerResult;
                            result.statusValue =
                                g_executorPriorityStatusValue;
                            result.firstThreadId =
                                g_executorPriorityFirstThreadId;
                            result.secondThreadId =
                                g_executorPrioritySecondThreadId;
                        }

                        runtime
                            .invokeEeSchedulerUpdateAtBoundary(
                                [&result](
                                    ps2x::ee::
                                        EeThreadScheduler
                                            &scheduler)
                                {
                                    const auto main =
                                        scheduler.thread(1);
                                    const auto first =
                                        scheduler.thread(
                                            result
                                                .firstThreadId);
                                    const auto second =
                                        scheduler.thread(
                                            result
                                                .secondThreadId);
                                    result.schedulerMainPriority =
                                        main.has_value()
                                            ? main->priority
                                            : -1;
                                    result.schedulerFirstPriority =
                                        first.has_value()
                                            ? first->priority
                                            : -1;
                                    result.schedulerSecondPriority =
                                        second.has_value()
                                            ? second->priority
                                            : -1;
                                    result.schedulerValid =
                                        scheduler.validate();
                                },
                                ps2x::ee::
                                    EeSchedulerReschedulePolicy::
                                        None);
                        result.managedContinuations =
                            runtime
                                .managedEeExecutionThreadCountForTesting();
                        runtime
                            .stopDedicatedEeExecutionForTesting();
                        if (!continuationsRetired)
                        {
                            result.managedContinuations =
                                std::max<size_t>(
                                    1u,
                                    result
                                        .managedContinuations);
                        }
                        return result;
                    };

                struct ScenarioExpectation
                {
                    ExecutorPriorityScenario scenario;
                    const char *name;
                    std::vector<int> order;
                    int operationResult;
                    int triggerResult;
                    int statusValue;
                    int mainPriority;
                    int firstPriority;
                    int secondPriority;
                };
                const std::array<
                    ScenarioExpectation,
                    5u>
                    expectations{{
                        {
                            ExecutorPriorityScenario::
                                RawPriority,
                            "raw priority promotion",
                            {1, 3, 2, 4},
                            50,
                            40,
                            30,
                            60,
                            30,
                            -1,
                        },
                        {
                            ExecutorPriorityScenario::
                                OrdinaryPriority,
                            "ordinary priority promotion",
                            {1, 2, 3},
                            50,
                            KE_ERROR,
                            -1,
                            40,
                            30,
                            -1,
                        },
                        {
                            ExecutorPriorityScenario::
                                RawCurrentRotation,
                            "raw current-priority rotation",
                            {1, 3, 2, 4},
                            40,
                            40,
                            -1,
                            60,
                            40,
                            -1,
                        },
                        {
                            ExecutorPriorityScenario::
                                OrdinaryCurrentRotation,
                            "ordinary current-priority rotation",
                            {1, 2, 3},
                            40,
                            KE_ERROR,
                            -1,
                            40,
                            40,
                            -1,
                        },
                        {
                            ExecutorPriorityScenario::
                                NamedRotation,
                            "named lower-priority rotation",
                            {1, 4, 3, 2, 5},
                            50,
                            40,
                            -1,
                            60,
                            50,
                            50,
                        },
                    }};

                for (const auto &expectation :
                     expectations)
                {
                    const FixtureResult result =
                        runFixture(
                            expectation.scenario);
                    t.IsTrue(
                        result.memoryInitialized &&
                            result.completed &&
                            result.setupResult == 1 &&
                            result.firstThreadId > 1 &&
                            (expectation.secondPriority < 0 ||
                             result.secondThreadId > 1) &&
                            result.order ==
                                expectation.order,
                        std::string(expectation.name) +
                            " should preserve the PCSX2 "
                            "ordinary/raw order");
                    t.IsTrue(
                        result.operationResult ==
                                expectation
                                    .operationResult &&
                            result.triggerResult ==
                                expectation.triggerResult &&
                            result.statusValue ==
                                expectation.statusValue,
                        std::string(expectation.name) +
                            " should preserve syscall "
                            "results and synchronous "
                            "guest status");
                    t.IsTrue(
                        result.schedulerMainPriority ==
                                expectation.mainPriority &&
                            result.schedulerFirstPriority ==
                                expectation
                                    .firstPriority &&
                            result.schedulerSecondPriority ==
                                expectation
                                    .secondPriority &&
                            result.schedulerValid &&
                            result.managedContinuations ==
                                0u,
                        std::string(expectation.name) +
                            " should keep scheduler "
                            "priority and ownership "
                            "authoritative");
                }
            });

        tc.Run(
            "fiber debugger pause leaves the executor responsive",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::LegacyCppFiber;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                if (!memoryInitialized)
                {
                    t.IsTrue(
                        false,
                        "fiber debugger fixture memory should initialize");
                    return;
                }

                runtime.registerFunction(
                    K_EXECUTOR_DEBUG_SPIN_ENTRY,
                    &dedicatedExecutorDebugSpinHandler);
                runtime.registerFunction(
                    K_EXECUTOR_DEBUG_TARGET_ENTRY,
                    &dedicatedExecutorDebugTargetHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_DEBUG_SPIN_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);
                g_executorDebugSpinCount.store(
                    0u, std::memory_order_release);
                g_executorDebugFinish.store(
                    false, std::memory_order_release);
                g_executorDebugTargetRequested.store(
                    false, std::memory_order_release);
                g_executorDebugBoundaryApplied.store(
                    false, std::memory_order_release);

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool guestStarted = waitUntil(
                    []()
                    {
                        return g_executorDebugSpinCount
                                   .load(
                                       std::memory_order_acquire) >=
                               8u;
                    },
                    std::chrono::seconds(2));
                const bool paused =
                    guestStarted &&
                    runtime.debugPause(
                        std::chrono::milliseconds(500));
                const bool guestQuiesced =
                    paused &&
                    waitUntil(
                        []()
                        {
                            const uint32_t before =
                                g_executorDebugSpinCount
                                    .load(
                                        std::memory_order_acquire);
                            std::this_thread::sleep_for(
                                std::chrono::
                                    milliseconds(2));
                            return before ==
                                   g_executorDebugSpinCount
                                       .load(
                                           std::memory_order_acquire);
                        },
                        std::chrono::
                            milliseconds(200));
                const bool publicationAccepted =
                    guestQuiesced &&
                    runtime.publishEeSchedulerUpdate(
                        [](
                            ps2x::ee::
                                EeThreadScheduler &)
                        {
                            g_executorDebugBoundaryApplied
                                .store(
                                    true,
                                    std::memory_order_release);
                        },
                        ps2x::ee::
                            EeSchedulerReschedulePolicy::
                                None);
                const bool appliedWhilePaused =
                    publicationAccepted &&
                    waitUntil(
                        []()
                        {
                            return g_executorDebugBoundaryApplied
                                .load(
                                    std::memory_order_acquire);
                        },
                        std::chrono::
                            milliseconds(100));

                const auto step =
                    runtime.debugStepDispatches(
                        1u,
                        std::chrono::seconds(1));
                g_executorDebugTargetRequested.store(
                    true, std::memory_order_release);
                const auto target =
                    runtime.debugRunUntilPc(
                        K_EXECUTOR_DEBUG_TARGET_ENTRY,
                        std::chrono::seconds(1));

                g_executorDebugFinish.store(
                    true, std::memory_order_release);
                runtime.debugResume();
                const bool continuationRetired =
                    waitUntil(
                        [&runtime]()
                        {
                            return runtime
                                       .managedEeExecutionThreadCountForTesting() ==
                                   0u;
                        },
                        std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                t.IsTrue(
                    memoryInitialized && guestStarted,
                    "debugger fixture should start its running fiber");
                t.IsTrue(
                    paused,
                    "debug pause should reach an acknowledged executor boundary");
                t.IsTrue(
                    guestQuiesced,
                    "debug pause should quiesce a running fiber at a guest boundary");
                t.IsTrue(
                    publicationAccepted &&
                        appliedWhilePaused,
                    "a paused guest fiber should leave the executor free to apply boundary commands");
                t.IsTrue(
                    step.completed &&
                        step.reason == "step" &&
                        step.pc ==
                            K_EXECUTOR_DEBUG_SPIN_ENTRY,
                    "dispatch stepping should resume one fiber step and stop on the executor again");
                t.IsTrue(
                    target.completed &&
                        target.reason == "predicate" &&
                        target.pc ==
                            K_EXECUTOR_DEBUG_TARGET_ENTRY,
                    "run-until should stop before its target handler while leaving the executor responsive");
                t.IsTrue(
                    continuationRetired &&
                        g_executorDebugBoundaryApplied
                            .load(
                                std::memory_order_acquire),
                    "debug resume should finish the guest and retain the paused boundary command");
            });

        tc.Run(
            "fiber backedge checkpoint preserves its native continuation",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::LegacyCppFiber;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                if (!memoryInitialized)
                {
                    t.IsTrue(
                        false,
                        "checkpoint fixture memory should initialize");
                    return;
                }

                runtime.registerFunction(
                    K_EXECUTOR_CHECKPOINT_ENTRY,
                    &dedicatedExecutorCheckpointHandler);
                runtime.cpu().pc =
                    K_EXECUTOR_CHECKPOINT_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);
                g_executorCheckpointEntries.store(
                    0u, std::memory_order_release);
                g_executorCheckpointProbeAllowed.store(
                    false, std::memory_order_release);
                g_executorCheckpointReturned.store(
                    false, std::memory_order_release);
                g_executorCheckpointFinish.store(
                    false, std::memory_order_release);

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool entered = waitUntil(
                    []()
                    {
                        return g_executorCheckpointEntries
                                   .load(
                                       std::memory_order_acquire) >=
                               1u;
                    },
                    std::chrono::seconds(2));
                runtime.requestGuestPreemption();
                g_executorCheckpointProbeAllowed.store(
                    true, std::memory_order_release);
                const bool checkpointed = waitUntil(
                    []()
                    {
                        return g_executorCheckpointReturned
                            .load(
                                std::memory_order_acquire);
                    },
                    std::chrono::seconds(2));
                const uint32_t entriesBeforeFinish =
                    g_executorCheckpointEntries.load(
                        std::memory_order_acquire);
                g_executorCheckpointFinish.store(
                    true, std::memory_order_release);
                const bool continuationRetired =
                    waitUntil(
                        [&runtime]()
                        {
                            return runtime
                                       .managedEeExecutionThreadCountForTesting() ==
                                   0u;
                        },
                        std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                t.IsTrue(
                    memoryInitialized && entered &&
                        checkpointed &&
                        continuationRetired,
                    "checkpoint fixture should run and retire its production fiber");
                t.Equals(
                    entriesBeforeFinish,
                    1u,
                    "a stackful backedge checkpoint should resume at the call instead of re-entering the generated function");
            });

        tc.Run(
            "fiber checkpoint preserves nested generated callers",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                PS2RuntimeConfiguration configuration{};
                configuration.eeExecutionBackend =
                    EeExecutionBackendKind::LegacyCppFiber;
                PS2Runtime runtime(configuration);
                const bool memoryInitialized =
                    runtime.memory().initialize();
                if (!memoryInitialized)
                {
                    t.IsTrue(
                        false,
                        "nested checkpoint fixture memory should initialize");
                    return;
                }

                runtime.registerFunction(
                    K_EXECUTOR_NESTED_CHECKPOINT_ENTRY,
                    &dedicatedExecutorNestedCheckpointHandler);
                runtime.registerFunction(
                    K_EXECUTOR_NESTED_CHECKPOINT_CHILD,
                    &dedicatedExecutorNestedCheckpointChild);
                runtime.cpu().pc =
                    K_EXECUTOR_NESTED_CHECKPOINT_ENTRY;
                setRegU32(
                    runtime.cpu(),
                    29,
                    0x00300000u);
                g_executorNestedCheckpointEntries.store(
                    0u, std::memory_order_release);
                g_executorNestedCheckpointChildEntries
                    .store(
                        0u, std::memory_order_release);
                g_executorNestedCheckpointProbeAllowed
                    .store(
                        false,
                        std::memory_order_release);
                g_executorNestedCheckpointContinued
                    .store(
                        false,
                        std::memory_order_release);

                runtime
                    .startDedicatedEeExecutionForTesting();
                const bool childEntered = waitUntil(
                    []()
                    {
                        return g_executorNestedCheckpointChildEntries
                                   .load(
                                       std::memory_order_acquire) >=
                               1u;
                    },
                    std::chrono::seconds(2));
                runtime.requestGuestPreemption();
                g_executorNestedCheckpointProbeAllowed
                    .store(
                        true,
                        std::memory_order_release);
                const bool callerContinued = waitUntil(
                    []()
                    {
                        return g_executorNestedCheckpointContinued
                            .load(
                                std::memory_order_acquire);
                    },
                    std::chrono::seconds(2));
                const bool continuationRetired =
                    callerContinued &&
                    waitUntil(
                        [&runtime]()
                        {
                            return runtime
                                       .managedEeExecutionThreadCountForTesting() ==
                                   0u;
                        },
                        std::chrono::seconds(2));
                runtime
                    .stopDedicatedEeExecutionForTesting();

                t.IsTrue(
                    memoryInitialized && childEntered &&
                        callerContinued &&
                        continuationRetired,
                    "a nested checkpoint should resume through its native generated callers and retire");
                t.IsTrue(
                    g_executorNestedCheckpointEntries
                                .load(
                                    std::memory_order_acquire) ==
                            1u &&
                        g_executorNestedCheckpointChildEntries
                                .load(
                                    std::memory_order_acquire) ==
                            1u,
                    "a stackful checkpoint should not re-enter either generated function");
            });

        tc.Run(
            "fiber MPEG picture wait leaves the executor available to its guest producer",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                const auto result =
                    runDedicatedExecutorMpegFixture(
                        DedicatedExecutorMpegAction::
                            RestartStream);

                t.IsTrue(
                    result.memoryInitialized &&
                        result.completedAfterRescue,
                    "the bounded MPEG fixture should retire both continuations");
                t.IsTrue(
                    result.completedWithoutRescue,
                    "sceMpegGetPicture must not park the sole EE executor on its host condition variable");
                t.IsTrue(
                    result.producerSequence == 1u &&
                        result.returnSequence == 2u,
                    "the lower-priority guest producer should release the picture wait before it returns");
                t.Equals(
                    result.pictureResult,
                    -1,
                    "a new CD stream generation should interrupt the prior picture wait");
                t.IsTrue(
                    result.observedStatus == THS_WAIT &&
                        result.observedWaitType ==
                            TSW_SEMA &&
                        result.observedWaitId ==
                            K_EXECUTOR_MPEG_ADDR,
                    "the picture wait should expose the "
                    "retail internal-semaphore status");
                t.Equals(
                    result.managedAfterStop,
                    size_t{0u},
                    "the producer wake should leave no "
                    "managed fiber");
            });

        tc.Run(
            "ReleaseWaitThread interrupts a fiber MPEG picture wait",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                const auto result =
                    runDedicatedExecutorMpegFixture(
                        DedicatedExecutorMpegAction::
                            ReleaseWait);
                t.IsTrue(
                    result.memoryInitialized &&
                        result.completedWithoutRescue,
                    "forced release should complete without "
                    "a host rescue");
                t.IsTrue(
                    result.observedStatus == THS_WAIT &&
                        result.observedWaitType ==
                            TSW_SEMA &&
                        result.observedWaitId ==
                            K_EXECUTOR_MPEG_ADDR,
                    "ReleaseWaitThread should observe the "
                    "HLE semaphore wait");
                t.Equals(
                    result.controlResult,
                    result.mainThreadId,
                    "ReleaseWaitThread should return the "
                    "picture waiter's thread id");
                t.IsTrue(
                    result.producerSequence == 1u &&
                        result.returnSequence == 2u &&
                        result.pictureResult == -1,
                    "forced release should resume and "
                    "interrupt the suspended native call");
                t.Equals(
                    result.managedAfterStop,
                    size_t{0u},
                    "forced release should leave no managed "
                    "fiber");
            });

        tc.Run(
            "TerminateThread unwinds a fiber MPEG picture wait",
            [](TestCase &t)
            {
                if (!eeExecutionBackendBuildInfo()
                         .boostContextFcontextAvailable)
                {
                    return;
                }

                const auto result =
                    runDedicatedExecutorMpegFixture(
                        DedicatedExecutorMpegAction::
                            Terminate);
                t.IsTrue(
                    result.memoryInitialized &&
                        result.completedWithoutRescue,
                    "termination should complete without a "
                    "host rescue");
                t.IsTrue(
                    result.observedStatus == THS_WAIT &&
                        result.observedWaitType ==
                            TSW_SEMA &&
                        result.observedWaitId ==
                            K_EXECUTOR_MPEG_ADDR,
                    "TerminateThread should observe the HLE "
                    "semaphore wait");
                t.Equals(
                    result.controlResult,
                    result.mainThreadId,
                    "TerminateThread should return the "
                    "picture waiter's thread id");
                t.IsTrue(
                    result.producerSequence == 1u &&
                        result.returnSequence == 0u &&
                        result.pictureResult == KE_ERROR,
                    "termination must unwind the suspended "
                    "native call without returning through it");
                t.Equals(
                    result.managedAfterStop,
                    size_t{0u},
                    "termination should destroy the picture "
                    "waiter's fiber");
            });

        tc.Run("host presentation upload state is isolated per runtime", [](TestCase &t)
        {
            PS2Runtime first;
            PS2Runtime second;

            t.IsTrue(
                first
                        .hostPresentationUploadStateIdentityForTesting() !=
                    second
                        .hostPresentationUploadStateIdentityForTesting(),
                "runtime restart must not inherit another texture's upload latch");
        });

#ifndef NDEBUG
        tc.Run("DECI2 sessions are isolated per runtime", [](TestCase &t)
        {
            TestEnv first;
            TestEnv second;
            constexpr uint32_t kDeci2ArgsAddr = 0x1A00u;

            const auto deci2Call =
                [&](TestEnv &env,
                    int32_t code,
                    const std::array<uint32_t, 4u> &args)
            {
                writeGuestWords(
                    env.rdram.data(),
                    kDeci2ArgsAddr,
                    args.data(),
                    args.size());
                R5900Context ctx{};
                setRegU32(
                    ctx,
                    4,
                    static_cast<uint32_t>(code));
                setRegU32(ctx, 5, kDeci2ArgsAddr);
                callSyscall(
                    0x7Cu,
                    env.rdram.data(),
                    &ctx,
                    &env.runtime);
                return getRegS32(ctx, 2);
            };
            const auto open =
                [&](TestEnv &env)
            {
                return deci2Call(
                    env,
                    1,
                    {0x1234u, 0u, 0u, 0u});
            };
            const auto close =
                [&](TestEnv &env, int32_t socket)
            {
                return deci2Call(
                    env,
                    2,
                    {static_cast<uint32_t>(socket),
                     0u,
                     0u,
                     0u});
            };
            const auto lock =
                [&](TestEnv &env, int32_t socket)
            {
                return deci2Call(
                    env,
                    -8,
                    {static_cast<uint32_t>(socket),
                     0u,
                     0u,
                     0u});
            };

            const int32_t firstSocket = open(first);
            const int32_t secondSocket = open(second);
            t.Equals(
                firstSocket,
                1,
                "the first runtime should allocate its first DECI2 socket");
            t.Equals(
                secondSocket,
                1,
                "the second runtime should allocate its own first DECI2 socket");
            t.Equals(
                close(first, firstSocket),
                KE_OK,
                "the first runtime should close its DECI2 socket");
            t.Equals(
                lock(second, secondSocket),
                KE_OK,
                "closing one runtime's socket must preserve its peer's session");

            const int32_t restartedFirstSocket = open(first);
            const int32_t restartedSecondSocket = open(second);
            notifyRuntimeStop(&first.runtime);
            t.Equals(
                close(first, restartedFirstSocket),
                KE_ERROR,
                "stopping a runtime should discard its DECI2 sessions");
            t.Equals(
                lock(second, restartedSecondSocket),
                KE_OK,
                "stopping one runtime must preserve its peer's DECI2 sessions");
        });
#endif

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

        tc.Run("repeated runtime construction and targeted reset restore EE ownership roots", [](TestCase &t)
        {
            struct OwnershipIds
            {
                int32_t thread = -1;
                int32_t sema = -1;
                int32_t event = -1;
                int32_t alarm = -1;
            };

            constexpr uint32_t kAlarmHandlerAddr = 0x00202380u;
            constexpr uint32_t kThreadEntry = 0x00202400u;
            constexpr uint32_t kThreadStack = 0x00302400u;
            constexpr int kRuntimeLifetimes = 4;

            for (int lifetime = 0;
                 lifetime < kRuntimeLifetimes;
                 ++lifetime)
            {
                auto env = std::make_unique<TestEnv>();
                env->runtime.registerFunction(
                    kAlarmHandlerAddr, &alarmNoopHandler);

                const auto label =
                    [&](const char *epoch,
                        const char *expectation)
                {
                    std::ostringstream stream;
                    stream << "runtime lifetime "
                           << lifetime
                           << ", "
                           << epoch
                           << ": "
                           << expectation;
                    return stream.str();
                };

                const auto populateOwnershipRoots =
                    [&](const char *epoch)
                {
                    initializeGuestKernelState(
                        env->rdram.data(),
                        &env->runtime);

                    R5900Context mainStatusCtx{};
                    setRegU32(mainStatusCtx, 4, 0u);
                    setRegU32(
                        mainStatusCtx,
                        5,
                        K_STATUS_ADDR);
                    ReferThreadStatus(
                        env->rdram.data(),
                        &mainStatusCtx,
                        &env->runtime);
                    t.Equals(
                        getRegS32(mainStatusCtx, 2),
                        1,
                        label(
                            epoch,
                            "the main thread should be selected as RUN"));

                    const uint32_t threadParam[9] = {
                        0u,
                        kThreadEntry,
                        kThreadStack,
                        0x00000800u,
                        0u,
                        40u,
                        0u,
                        0u,
                        0u,
                    };
                    writeGuestWords(
                        env->rdram.data(),
                        K_PARAM_ADDR,
                        threadParam,
                        std::size(threadParam));
                    R5900Context createThreadCtx{};
                    setRegU32(
                        createThreadCtx,
                        4,
                        K_PARAM_ADDR);
                    CreateThread(
                        env->rdram.data(),
                        &createThreadCtx,
                        &env->runtime);

                    const uint32_t semaParam[6] = {
                        0u,
                        3u,
                        2u,
                        0u,
                        0x11u,
                        0x22u,
                    };
                    writeGuestWords(
                        env->rdram.data(),
                        K_PARAM_ADDR,
                        semaParam,
                        std::size(semaParam));
                    R5900Context createSemaCtx{};
                    setRegU32(
                        createSemaCtx,
                        4,
                        K_PARAM_ADDR);
                    CreateSema(
                        env->rdram.data(),
                        &createSemaCtx,
                        &env->runtime);

                    const uint32_t eventParam[3] = {
                        0x33u,
                        0x44u,
                        0x55u,
                    };
                    writeGuestWords(
                        env->rdram.data(),
                        K_PARAM_ADDR,
                        eventParam,
                        std::size(eventParam));
                    R5900Context createEventCtx{};
                    setRegU32(
                        createEventCtx,
                        4,
                        K_PARAM_ADDR);
                    CreateEventFlag(
                        env->rdram.data(),
                        &createEventCtx,
                        &env->runtime);

                    R5900Context setAlarmCtx{};
                    setRegU32(setAlarmCtx, 4, 0xFFFFu);
                    setRegU32(
                        setAlarmCtx,
                        5,
                        kAlarmHandlerAddr);
                    SetAlarm(
                        env->rdram.data(),
                        &setAlarmCtx,
                        &env->runtime);

                    env->ctx.cop0_config =
                        0x00073443u;
                    env->ctx.advanceEeCycleTicks(16u);
                    env->runtime
                        .serviceEeEventsAtBlockBoundary(
                            env->rdram.data(),
                            &env->ctx);

                    OwnershipIds ids{
                        getRegS32(createThreadCtx, 2),
                        getRegS32(createSemaCtx, 2),
                        getRegS32(createEventCtx, 2),
                        getRegS32(setAlarmCtx, 2),
                    };
                    t.Equals(
                        ids.thread,
                        2,
                        label(
                            epoch,
                            "thread allocation should restart at id 2"));
                    t.Equals(
                        ids.sema,
                        0,
                        label(
                            epoch,
                            "semaphore allocation should restart at id 0"));
                    t.Equals(
                        ids.event,
                        1,
                        label(
                            epoch,
                            "event allocation should restart at id 1"));
                    t.Equals(
                        ids.alarm,
                        1,
                        label(
                            epoch,
                            "alarm allocation should restart at id 1"));
                    t.Equals(
                        env->runtime.currentEeThreadId(),
                        1,
                        label(
                            epoch,
                            "the runtime should retain main-thread ownership"));
                    t.Equals(
                        env->runtime.currentEeTick().raw(),
                        uint64_t{16u},
                        label(
                            epoch,
                            "the canonical timeline should accept local work"));
                    t.Equals(
                        env->runtime
                            .pendingAlarmCallbackCountForTesting(),
                        size_t{0u},
                        label(
                            epoch,
                            "the long alarm should remain with its publisher"));
                    return ids;
                };

                const auto resetOwnershipRoots =
                    [&](const OwnershipIds &oldIds,
                        const char *epoch)
                {
                    notifyRuntimeStop(&env->runtime);
                    env->runtime.resetEeTiming(&env->ctx);
                    initializeGuestKernelState(
                        env->rdram.data(),
                        &env->runtime);

                    t.Equals(
                        env->runtime.currentEeThreadId(),
                        1,
                        label(
                            epoch,
                            "targeted reset should restore main-thread selection"));
                    t.Equals(
                        env->runtime.currentEeTick().raw(),
                        uint64_t{0u},
                        label(
                            epoch,
                            "targeted timing reset should clear canonical time"));
                    t.Equals(
                        env->runtime
                            .managedEeExecutionThreadCountForTesting(),
                        size_t{0u},
                        label(
                            epoch,
                            "targeted reset should leave no managed continuation"));
                    t.Equals(
                        env->runtime
                            .pendingAlarmCallbackCountForTesting(),
                        size_t{0u},
                        label(
                            epoch,
                            "targeted reset should discard alarm publications"));
                    t.Equals(
                        debugThreadSnapshots(
                            &env->runtime)
                            .size(),
                        size_t{0u},
                        label(
                            epoch,
                            "targeted reset should clear the thread registry"));

                    R5900Context staleThreadCtx{};
                    setRegU32(
                        staleThreadCtx,
                        4,
                        static_cast<uint32_t>(
                            oldIds.thread));
                    setRegU32(
                        staleThreadCtx,
                        5,
                        K_STATUS_ADDR);
                    ReferThreadStatus(
                        env->rdram.data(),
                        &staleThreadCtx,
                        &env->runtime);
                    t.Equals(
                        getRegS32(staleThreadCtx, 2),
                        KE_UNKNOWN_THID,
                        label(
                            epoch,
                            "a pre-reset thread id should be stale"));

                    R5900Context staleSemaCtx{};
                    setRegU32(
                        staleSemaCtx,
                        4,
                        static_cast<uint32_t>(
                            oldIds.sema));
                    setRegU32(
                        staleSemaCtx,
                        5,
                        K_STATUS_ADDR);
                    ReferSemaStatus(
                        env->rdram.data(),
                        &staleSemaCtx,
                        &env->runtime);
                    t.Equals(
                        getRegS32(staleSemaCtx, 2),
                        KE_UNKNOWN_SEMID,
                        label(
                            epoch,
                            "a pre-reset semaphore id should be stale"));

                    R5900Context staleAlarmCtx{};
                    setRegU32(
                        staleAlarmCtx,
                        4,
                        static_cast<uint32_t>(
                            oldIds.alarm));
                    CancelAlarm(
                        env->rdram.data(),
                        &staleAlarmCtx,
                        &env->runtime);
                    t.Equals(
                        getRegS32(staleAlarmCtx, 2),
                        KE_ERROR,
                        label(
                            epoch,
                            "a pre-reset alarm id should be stale"));
                };

                const OwnershipIds firstIds =
                    populateOwnershipRoots(
                        "first epoch");
                resetOwnershipRoots(
                    firstIds,
                    "first reset");

                const OwnershipIds secondIds =
                    populateOwnershipRoots(
                        "second epoch");
                resetOwnershipRoots(
                    secondIds,
                    "second reset");

                env.reset();
            }

            notifyRuntimeStop();
        });

        tc.Run("debug thread snapshots expose authoritative valid state", [](TestCase &t)
        {
            TestEnv env;

            R5900Context mainStatusCtx{};
            setRegU32(mainStatusCtx, 4, 0u);
            setRegU32(mainStatusCtx, 5, K_STATUS_ADDR);
            ReferThreadStatus(
                env.rdram.data(),
                &mainStatusCtx,
                &env.runtime);
            t.Equals(
                getRegS32(mainStatusCtx, 2),
                1,
                "the initialized main thread should report RUN");

            constexpr uint32_t kDormantEntry = 0x00202480u;
            const uint32_t threadParam[9] = {
                0u,
                kDormantEntry,
                0x00302480u,
                0x00000800u,
                0u,
                37u,
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
            const int32_t dormantTid =
                getRegS32(createCtx, 2);

            const auto snapshots =
                debugThreadSnapshots(&env.runtime);
            const GuestThreadDebugSnapshot *mainSnapshot = nullptr;
            const GuestThreadDebugSnapshot *dormantSnapshot =
                nullptr;
            for (const GuestThreadDebugSnapshot &snapshot :
                 snapshots)
            {
                if (snapshot.id == 1)
                {
                    mainSnapshot = &snapshot;
                }
                else if (snapshot.id == dormantTid)
                {
                    dormantSnapshot = &snapshot;
                }
            }

            t.IsTrue(
                mainSnapshot != nullptr,
                "debug enumeration should include the runtime main thread");
            if (mainSnapshot)
            {
                t.IsTrue(
                    mainSnapshot->stateValid,
                    "the debug main-thread state should be valid");
                t.Equals(
                    mainSnapshot->status,
                    1,
                    "debug enumeration should use the main thread's authoritative RUN state");
                t.Equals(
                    mainSnapshot->waitQueue,
                    0,
                    "a RUN thread should have no wait-queue membership");
            }

            t.IsTrue(
                dormantSnapshot != nullptr,
                "debug enumeration should include a newly created thread");
            if (dormantSnapshot)
            {
                t.IsTrue(
                    dormantSnapshot->stateValid,
                    "the debug dormant-thread state should be valid");
                t.Equals(
                    dormantSnapshot->status,
                    THS_DORMANT,
                    "a newly created debug thread should be DORMANT");
                t.IsFalse(
                    dormantSnapshot->started,
                    "DORMANT state should authoritatively own the not-started condition");
                t.Equals(
                    dormantSnapshot->currentPriority,
                    37,
                    "debug enumeration should share authoritative priority state with ReferThreadStatus");
            }
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

        tc.Run(
            "SetupThread root handles a normal started-thread return",
            [](TestCase &t)
            {
                TestEnv env;
                env.runtime.registerFunction(
                    K_SETUP_THREAD_ROOT_HANDLER_ENTRY,
                    &setupThreadRootExitHandler);
                env.runtime.registerFunction(
                    K_SETUP_THREAD_ROOT_WORKER_ENTRY,
                    &setupThreadRootWorkerHandler);
                g_setupThreadRootObservedRa.store(
                    0u,
                    std::memory_order_release);
                g_setupThreadRootHits.store(
                    0u,
                    std::memory_order_release);
                g_setupThreadRootObservedThreadId.store(
                    0,
                    std::memory_order_release);

                setRegU32(env.ctx, 29, 0x00300000u);
                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 0u);
                setRegU32(env.ctx, 6, 0u);
                setRegU32(env.ctx, 7, 0u);
                setRegU32(
                    env.ctx,
                    8,
                    K_SETUP_THREAD_ROOT_HANDLER_ENTRY);
                SetupThread(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                t.Equals(
                    getRegU32(&env.ctx, 2),
                    0x00300000u,
                    "SetupThread should preserve an aligned current stack when no replacement is supplied");

                setRegU32(env.ctx, 4, 0u);
                setRegU32(env.ctx, 5, 40u);
                ChangeThreadPriority(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);

                const uint32_t threadParam[9] = {
                    0u,
                    K_SETUP_THREAD_ROOT_WORKER_ENTRY,
                    0x00310000u,
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
                setRegU32(env.ctx, 4, K_PARAM_ADDR);
                CreateThread(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                const int32_t threadId =
                    getRegS32(env.ctx, 2);
                t.IsTrue(
                    threadId >= 2,
                    "the normal-return worker should be created");

                setRegU32(
                    env.ctx,
                    4,
                    static_cast<uint32_t>(threadId));
                setRegU32(env.ctx, 5, 0u);
                StartThread(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                t.Equals(
                    getRegS32(env.ctx, 2),
                    threadId,
                    "the normal-return worker should start");

                const bool exitedThroughRoot =
                    waitUntil(
                        [&]()
                        {
                            R5900Context statusCtx{};
                            setRegU32(
                                statusCtx,
                                4,
                                static_cast<uint32_t>(
                                    threadId));
                            setRegU32(
                                statusCtx,
                                5,
                                K_STATUS_ADDR);
                            ReferThreadStatus(
                                env.rdram.data(),
                                &statusCtx,
                                &env.runtime);
                            return g_setupThreadRootHits
                                           .load(
                                               std::memory_order_acquire) ==
                                       1u &&
                                   getRegS32(statusCtx, 2) ==
                                       THS_DORMANT;
                        },
                        std::chrono::milliseconds(500));
                t.IsTrue(
                    exitedThroughRoot,
                    "a normal worker return should dispatch the SetupThread root and become dormant");
                t.Equals(
                    g_setupThreadRootObservedRa.load(
                        std::memory_order_acquire),
                    K_SETUP_THREAD_ROOT_HANDLER_ENTRY,
                    "StartThread should initialize $ra from the SetupThread root");
                t.Equals(
                    g_setupThreadRootObservedThreadId.load(
                        std::memory_order_acquire),
                    threadId,
                    "the SetupThread root should execute as the returning worker");

                setRegU32(
                    env.ctx,
                    4,
                    static_cast<uint32_t>(threadId));
                DeleteThread(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                t.Equals(
                    getRegS32(env.ctx, 2),
                    threadId,
                    "the root-exited worker should remain deletable");
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

        tc.Run("asynchronous EE wake handles reject deleted and aliased threads", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00255800u;
            constexpr uint32_t kStackAddr = 0x0030C000u;
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

            const auto configureEnv = [&](TestEnv &target)
            {
                target.runtime.registerFunction(
                    kThreadEntry, &idReuseSleepHandler);
                writeGuestU32(
                    target.rdram.data(), K_ID_REUSE_STAGE_ADDR, 0u);
                writeGuestU32(
                    target.rdram.data(), K_ID_REUSE_RETURN_ADDR, 0u);
                writeGuestWords(
                    target.rdram.data(),
                    K_PARAM_ADDR,
                    threadParam,
                    std::size(threadParam));

                setRegU32(target.ctx, 4, 0u);
                setRegU32(target.ctx, 5, 40u);
                ChangeThreadPriority(
                    target.rdram.data(),
                    &target.ctx,
                    &target.runtime);
            };
            const auto createThread = [&](TestEnv &target)
            {
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateThread(
                    target.rdram.data(),
                    &createCtx,
                    &target.runtime);
                return getRegS32(createCtx, 2);
            };
            const auto deleteThread =
                [&](TestEnv &target, int32_t tid)
            {
                R5900Context deleteCtx{};
                setRegU32(
                    deleteCtx, 4, static_cast<uint32_t>(tid));
                DeleteThread(
                    target.rdram.data(),
                    &deleteCtx,
                    &target.runtime);
                return getRegS32(deleteCtx, 2);
            };
            const auto startSleepingThread =
                [&](TestEnv &target, int32_t tid)
            {
                setRegU32(
                    target.ctx, 4, static_cast<uint32_t>(tid));
                setRegU32(target.ctx, 5, 0u);
                StartThread(
                    target.rdram.data(),
                    &target.ctx,
                    &target.runtime);
                if (getRegS32(target.ctx, 2) != tid)
                {
                    return false;
                }

                return waitUntil([&]()
                {
                    R5900Context statusCtx{};
                    setRegU32(
                        statusCtx, 4, static_cast<uint32_t>(tid));
                    setRegU32(statusCtx, 5, K_STATUS_ADDR);
                    iReferThreadStatus(
                        target.rdram.data(),
                        &statusCtx,
                        &target.runtime);
                    EeThreadStatus status{};
                    std::memcpy(
                        &status,
                        target.rdram.data() + K_STATUS_ADDR,
                        sizeof(status));
                    return readGuestU32(
                               target.rdram.data(),
                               K_ID_REUSE_STAGE_ADDR) == 1u &&
                           getRegS32(statusCtx, 2) == THS_WAIT &&
                           status.status == THS_WAIT &&
                           status.waitType == TSW_SLEEP;
                }, std::chrono::milliseconds(200));
            };
            const auto statusOf =
                [&](TestEnv &target, int32_t tid)
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    target.rdram.data(),
                    &statusCtx,
                    &target.runtime);
                return getRegS32(statusCtx, 2);
            };
            const auto finishReadyThread =
                [&](TestEnv &target, int32_t tid)
            {
                setRegU32(target.ctx, 4, 0u);
                setRegU32(target.ctx, 5, 60u);
                ChangeThreadPriority(
                    target.rdram.data(),
                    &target.ctx,
                    &target.runtime);
                return waitUntil([&]()
                {
                    return statusOf(target, tid) == THS_DORMANT;
                }, std::chrono::milliseconds(200));
            };

            configureEnv(env);
            const int32_t firstTid = createThread(env);
            t.Equals(
                firstTid,
                2,
                "the first asynchronous handle should target id 2");
            const EeThreadHandle staleHandle =
                captureEeThreadHandle(&env.runtime, firstTid);
            t.IsTrue(
                static_cast<bool>(staleHandle),
                "a registered thread should produce a valid host handle");
            t.Equals(
                staleHandle.threadId,
                firstTid,
                "the host handle should retain the numeric thread id");
            t.Equals(
                deleteThread(env, firstTid),
                firstTid,
                "the captured dormant thread should be deletable");
            t.IsTrue(
                publishEeThreadWake(
                    &env.runtime, staleHandle) ==
                    EeThreadWakeResult::StaleHandle,
                "a deleted thread must reject a delayed host wake");

            {
                TestEnv other;
                configureEnv(other);
                const int32_t aliasTid = createThread(other);
                t.Equals(
                    aliasTid,
                    firstTid,
                    "a second runtime should reuse the same numeric id");
                const EeThreadHandle aliasHandle =
                    captureEeThreadHandle(&other.runtime, aliasTid);
                t.IsTrue(
                    static_cast<bool>(aliasHandle),
                    "the second runtime should produce a valid host handle");
                t.Equals(
                    aliasHandle.threadGeneration,
                    staleHandle.threadGeneration,
                    "fresh runtimes should deliberately alias the per-thread generation");
                t.IsTrue(
                    aliasHandle != staleHandle,
                    "runtime identity must distinguish otherwise identical handles");
                t.IsTrue(
                    startSleepingThread(other, aliasTid),
                    "the aliased second-runtime thread should sleep");

                t.IsTrue(
                    publishEeThreadWake(
                        &other.runtime, staleHandle) ==
                        EeThreadWakeResult::StaleHandle,
                    "a handle from another runtime must reject publication");
                t.Equals(
                    statusOf(other, aliasTid),
                    THS_WAIT,
                    "a cross-runtime stale wake must not change the alias");
                t.Equals(
                    readGuestU32(
                        other.rdram.data(), K_ID_REUSE_STAGE_ADDR),
                    1u,
                    "a cross-runtime stale wake must not run guest code");

                t.IsTrue(
                    publishEeThreadWake(
                        &other.runtime, aliasHandle) ==
                        EeThreadWakeResult::WokeSleeper,
                    "the matching host handle should publish the wake");
                t.Equals(
                    statusOf(other, aliasTid),
                    THS_READY,
                    "host wake publication should make the sleeper READY");
                t.Equals(
                    readGuestU32(
                        other.rdram.data(), K_ID_REUSE_STAGE_ADDR),
                    1u,
                    "host wake publication must not dispatch guest code");
                t.IsTrue(
                    finishReadyThread(other, aliasTid),
                    "a later ordinary boundary should run the published thread");
                t.Equals(
                    deleteThread(other, aliasTid),
                    aliasTid,
                    "the second-runtime thread should remain cleanly deletable");
            }

            std::vector<int32_t> dormantIds;
            dormantIds.reserve(253u);
            for (int32_t expectedTid = 3;
                 expectedTid <= 0xFF;
                 ++expectedTid)
            {
                const int32_t tid = createThread(env);
                t.Equals(
                    tid,
                    expectedTid,
                    "the allocator should advance to the wrap boundary");
                if (tid > 0)
                {
                    dormantIds.push_back(tid);
                }
            }

            const int32_t replacementTid = createThread(env);
            t.Equals(
                replacementTid,
                firstTid,
                "the allocator should reuse the captured numeric id");
            const EeThreadHandle replacementHandle =
                captureEeThreadHandle(&env.runtime, replacementTid);
            t.IsTrue(
                static_cast<bool>(replacementHandle),
                "the replacement should produce a valid host handle");
            t.IsTrue(
                replacementHandle != staleHandle,
                "ID reuse must advance the non-guest-visible generation");
            t.IsTrue(
                startSleepingThread(env, replacementTid),
                "the replacement thread should sleep");

            t.IsTrue(
                publishEeThreadWake(
                    &env.runtime, staleHandle) ==
                    EeThreadWakeResult::StaleHandle,
                "a pre-delete handle must reject the recycled numeric id");
            t.Equals(
                statusOf(env, replacementTid),
                THS_WAIT,
                "a stale host wake must leave the replacement waiting");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), K_ID_REUSE_STAGE_ADDR),
                1u,
                "a stale host wake must not run the replacement");

            t.IsTrue(
                publishEeThreadWake(
                    &env.runtime, replacementHandle) ==
                    EeThreadWakeResult::WokeSleeper,
                "the replacement's matching handle should publish a wake");
            t.Equals(
                statusOf(env, replacementTid),
                THS_READY,
                "matching host publication should make the replacement READY");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), K_ID_REUSE_STAGE_ADDR),
                1u,
                "matching host publication must still defer guest execution");
            t.IsTrue(
                finishReadyThread(env, replacementTid),
                "a later ordinary boundary should run the replacement");
            t.Equals(
                deleteThread(env, replacementTid),
                replacementTid,
                "the replacement should remain cleanly deletable");

            for (const int32_t tid : dormantIds)
            {
                t.Equals(
                    deleteThread(env, tid),
                    tid,
                    "each dormant ring entry should remain independently deletable");
            }
        });

        tc.Run("asynchronous wake publication linearizes with exit-delete cleanup", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kThreadEntry = 0x00255900u;
            env.runtime.registerFunction(
                kThreadEntry, &asyncExitDeleteThreadHandler);
            writeGuestU32(
                env.rdram.data(),
                K_ASYNC_EXIT_DELETE_STAGE_ADDR,
                0u);

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 40u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);

            const uint32_t threadParam[9] = {
                0u,
                kThreadEntry,
                0u, // Let StartThread allocate and own the guest stack.
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
            const int32_t tid = getRegS32(createCtx, 2);
            t.Equals(
                tid,
                2,
                "the exit-delete race should create worker id 2");

            const EeThreadHandle handle =
                captureEeThreadHandle(&env.runtime, tid);
            t.IsTrue(
                static_cast<bool>(handle),
                "the exit-delete worker should produce a host handle");

            setRegU32(
                env.ctx, 4, static_cast<uint32_t>(tid));
            setRegU32(env.ctx, 5, 0u);
            StartThread(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2),
                tid,
                "the auto-stack worker should start");

            uint32_t autoStackAddr = 0u;
            const bool sleeping = waitUntil([&]()
            {
                R5900Context statusCtx{};
                setRegU32(
                    statusCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(statusCtx, 5, K_STATUS_ADDR);
                iReferThreadStatus(
                    env.rdram.data(), &statusCtx, &env.runtime);
                EeThreadStatus status{};
                std::memcpy(
                    &status,
                    env.rdram.data() + K_STATUS_ADDR,
                    sizeof(status));
                if (getRegS32(statusCtx, 2) == THS_WAIT &&
                    status.status == THS_WAIT &&
                    status.waitType == TSW_SLEEP)
                {
                    autoStackAddr = status.stack;
                    return readGuestU32(
                               env.rdram.data(),
                               K_ASYNC_EXIT_DELETE_STAGE_ADDR) == 1u;
                }
                return false;
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                sleeping,
                "the auto-stack worker should reach its sleep boundary");
            t.IsTrue(
                autoStackAddr != 0u,
                "StartThread should publish its owned guest stack");

            std::atomic<bool> publishedWake{false};
            std::atomic<bool> observedStale{false};
            std::atomic<uint32_t> unexpectedResults{0u};
            std::thread publisher([&]()
            {
                for (uint32_t attempt = 0;
                     attempt < 100000u;
                     ++attempt)
                {
                    const EeThreadWakeResult result =
                        publishEeThreadWake(
                            &env.runtime, handle);
                    switch (result)
                    {
                    case EeThreadWakeResult::WokeSleeper:
                        publishedWake.store(
                            true, std::memory_order_release);
                        break;
                    case EeThreadWakeResult::WakeupCounted:
                    case EeThreadWakeResult::Dormant:
                        break;
                    case EeThreadWakeResult::StaleHandle:
                        observedStale.store(
                            true, std::memory_order_release);
                        return;
                    case EeThreadWakeResult::InvalidHandle:
                        unexpectedResults.fetch_add(
                            1u, std::memory_order_relaxed);
                        return;
                    }
                    std::this_thread::yield();
                }
            });

            const bool wakePublished = waitUntil([&]()
            {
                return publishedWake.load(
                    std::memory_order_acquire);
            }, std::chrono::milliseconds(200));
            t.IsTrue(
                wakePublished,
                "the host publisher should release the sleeping worker");

            setRegU32(env.ctx, 4, 0u);
            setRegU32(env.ctx, 5, 60u);
            ChangeThreadPriority(
                env.rdram.data(), &env.ctx, &env.runtime);

            const bool deleted = waitUntil([&]()
            {
                return observedStale.load(
                           std::memory_order_acquire) &&
                       readGuestU32(
                           env.rdram.data(),
                           K_ASYNC_EXIT_DELETE_STAGE_ADDR) == 2u;
            }, std::chrono::milliseconds(500));
            if (!deleted)
            {
                notifyRuntimeStop(&env.runtime);
            }
            publisher.join();
            joinAllGuestHostThreads(&env.runtime);

            t.IsTrue(
                deleted,
                "publication should become stale after ExitDeleteThread");
            t.Equals(
                unexpectedResults.load(
                    std::memory_order_relaxed),
                0u,
                "the publication race should return only modeled outcomes");
            t.IsTrue(
                !captureEeThreadHandle(&env.runtime, tid),
                "ExitDeleteThread should remove the captured generation");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_ASYNC_EXIT_DELETE_STAGE_ADDR),
                2u,
                "guest code must not continue after ExitDeleteThread");

            const uint32_t reclaimed =
                env.runtime.guestMalloc(0x1000u, 16u);
            t.Equals(
                reclaimed,
                autoStackAddr,
                "exit-delete cleanup should reclaim the owned guest stack");
            env.runtime.guestFree(reclaimed);
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

                EeThreadStatus rawSignalStatus{};
                t.Equals(
                    referThread(rawSignalTid, rawSignalStatus),
                    THS_READY,
                    "raw signal should synchronously publish READY");
                t.Equals(
                    rawSignalStatus.status,
                    THS_READY,
                    "raw-signaled waiter should be READY before dispatch");
                t.Equals(
                    rawSignalStatus.waitType,
                    0u,
                    "raw signal should synchronously clear the semaphore wait reason");
                const auto snapshots =
                    debugThreadSnapshots(&env.runtime);
                bool foundRawSignalSnapshot = false;
                for (const GuestThreadDebugSnapshot &snapshot :
                     snapshots)
                {
                    if (snapshot.id != rawSignalTid)
                    {
                        continue;
                    }
                    foundRawSignalSnapshot = true;
                    t.IsTrue(
                        snapshot.stateValid,
                        "raw signal should publish one valid authoritative state");
                    t.Equals(
                        snapshot.waitQueue,
                        0,
                        "raw signal should synchronously detach semaphore queue membership");
                    t.IsFalse(
                        snapshot.waitCompletionPending,
                        "raw signal should not retain a pending wait completion");
                }
                t.IsTrue(
                    foundRawSignalSnapshot,
                    "debug enumeration should observe the raw-signaled waiter");

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

                const auto snapshots =
                    debugThreadSnapshots(&env.runtime);
                const GuestThreadDebugSnapshot *rawDeleteSnapshot =
                    nullptr;
                for (const GuestThreadDebugSnapshot &snapshot :
                     snapshots)
                {
                    if (snapshot.id == rawDeleteTid)
                    {
                        rawDeleteSnapshot = &snapshot;
                        break;
                    }
                }
                t.IsTrue(
                    rawDeleteSnapshot != nullptr,
                    "debug enumeration should observe the raw-deleted waiter");
                if (rawDeleteSnapshot)
                {
                    t.IsTrue(
                        rawDeleteSnapshot->stateValid,
                        "raw semaphore deletion should publish one valid authoritative state");
                    t.Equals(
                        rawDeleteSnapshot->waitQueue,
                        0,
                        "raw semaphore deletion should synchronously unlink wait-queue membership");
                    t.Equals(
                        rawDeleteSnapshot->waitQueueId,
                        0,
                        "an unlinked raw-delete waiter should expose no queue id");
                    t.IsTrue(
                        rawDeleteSnapshot
                            ->waitCompletionPending,
                        "the retained raw-delete wait reason should be an explicit pending completion");
                }

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

        tc.Run("semaphore and event registries are isolated per runtime", [](TestCase &t)
        {
            notifyRuntimeStop();
            auto first = std::make_unique<TestEnv>();
            auto second = std::make_unique<TestEnv>();

            const uint32_t semaParam[6] = {
                0u,
                3u,
                2u,
                0u,
                0x11u,
                0x22u
            };
            const uint32_t eventParam[3] = {
                0x33u,
                0x44u,
                0x55u
            };

            const auto createSema = [&](TestEnv &env)
            {
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    semaParam,
                    std::size(semaParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateSema(
                    env.rdram.data(),
                    &createCtx,
                    &env.runtime);
                return getRegS32(createCtx, 2);
            };
            const auto createEvent = [&](TestEnv &env)
            {
                writeGuestWords(
                    env.rdram.data(),
                    K_PARAM_ADDR,
                    eventParam,
                    std::size(eventParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, K_PARAM_ADDR);
                CreateEventFlag(
                    env.rdram.data(),
                    &createCtx,
                    &env.runtime);
                return getRegS32(createCtx, 2);
            };

            const int32_t firstSema = createSema(*first);
            const int32_t secondSema = createSema(*second);
            const int32_t firstEvent = createEvent(*first);
            const int32_t secondEvent = createEvent(*second);
            t.Equals(
                firstSema,
                0,
                "the first runtime should allocate semaphore id 0");
            t.Equals(
                secondSema,
                0,
                "a second runtime should independently allocate semaphore id 0");
            t.Equals(
                firstEvent,
                1,
                "the first runtime should allocate event id 1");
            t.Equals(
                secondEvent,
                1,
                "a second runtime should independently allocate event id 1");

            second.reset();

            R5900Context semaStatusCtx{};
            setRegU32(
                semaStatusCtx,
                4,
                static_cast<uint32_t>(firstSema));
            setRegU32(semaStatusCtx, 5, K_STATUS_ADDR);
            ReferSemaStatus(
                first->rdram.data(),
                &semaStatusCtx,
                &first->runtime);
            t.Equals(
                getRegS32(semaStatusCtx, 2),
                KE_OK,
                "destroying a second runtime must not erase the first runtime semaphore");
            EeSemaStatus semaStatus{};
            std::memcpy(
                &semaStatus,
                first->rdram.data() + K_STATUS_ADDR,
                sizeof(semaStatus));
            t.Equals(
                semaStatus.count,
                2,
                "the surviving runtime should retain its semaphore state");

            struct EventStatus
            {
                uint32_t attr;
                uint32_t option;
                uint32_t initBits;
                uint32_t currBits;
                int32_t numThreads;
                int32_t reserved1;
                int32_t reserved2;
            };
            R5900Context eventStatusCtx{};
            setRegU32(
                eventStatusCtx,
                4,
                static_cast<uint32_t>(firstEvent));
            setRegU32(eventStatusCtx, 5, K_STATUS_ADDR);
            ReferEventFlagStatus(
                first->rdram.data(),
                &eventStatusCtx,
                &first->runtime);
            t.Equals(
                getRegS32(eventStatusCtx, 2),
                KE_OK,
                "destroying a second runtime must not erase the first runtime event flag");
            EventStatus eventStatus{};
            std::memcpy(
                &eventStatus,
                first->rdram.data() + K_STATUS_ADDR,
                sizeof(eventStatus));
            t.Equals(
                eventStatus.currBits,
                0x55u,
                "the surviving runtime should retain its event bits");
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
                    const auto snapshots =
                        debugThreadSnapshots(&env.runtime);
                    const GuestThreadDebugSnapshot *waitSnapshot =
                        nullptr;
                    for (const GuestThreadDebugSnapshot &snapshot :
                         snapshots)
                    {
                        if (snapshot.id == tid)
                        {
                            waitSnapshot = &snapshot;
                            break;
                        }
                    }
                    t.IsTrue(
                        waitSnapshot != nullptr,
                        std::string("debug enumeration should include the ") +
                            expectation.name + " waiter");
                    if (waitSnapshot)
                    {
                        int expectedQueue = 0;
                        if (expectation.kind == WaitKind::Sleep ||
                            expectation.kind ==
                                WaitKind::WaitSuspend)
                        {
                            expectedQueue = 1;
                        }
                        else if (
                            expectation.kind == WaitKind::Sema)
                        {
                            expectedQueue = 2;
                        }
                        else if (
                            expectation.kind == WaitKind::Event)
                        {
                            expectedQueue = 3;
                        }
                        t.IsTrue(
                            waitSnapshot->stateValid,
                            std::string("the ") +
                                expectation.name +
                                " waiter should expose a valid authoritative state");
                        t.Equals(
                            waitSnapshot->waitQueue,
                            expectedQueue,
                            std::string("the ") +
                                expectation.name +
                                " state should own its exact wait-queue membership");
                        t.IsFalse(
                            waitSnapshot
                                ->waitCompletionPending,
                            std::string("an actively queued ") +
                                expectation.name +
                                " waiter should not expose a pending completion");
                    }

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
                if (readGuestU32(
                        env.rdram.data(),
                        K_TERMINATE_SEMA_WAIT_READY_ADDR) !=
                    1u)
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

        tc.Run("alarm registries and cancellation are isolated per runtime", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv first;
            TestEnv second;

            constexpr uint32_t kAlarmHandlerAddr = 0x00270100u;
            first.runtime.registerFunction(
                kAlarmHandlerAddr, &alarmNoopHandler);
            second.runtime.registerFunction(
                kAlarmHandlerAddr, &alarmNoopHandler);

            const auto setLongAlarm = [&](TestEnv &target)
            {
                setRegU32(target.ctx, 4, 0xFFFFu);
                setRegU32(target.ctx, 5, kAlarmHandlerAddr);
                setRegU32(target.ctx, 6, 0u);
                SetAlarm(
                    target.rdram.data(),
                    &target.ctx,
                    &target.runtime);
                return getRegS32(target.ctx, 2);
            };
            const auto cancelAlarm =
                [&](TestEnv &target, int32_t alarmId)
            {
                R5900Context cancelCtx{};
                setRegU32(
                    cancelCtx,
                    4,
                    static_cast<uint32_t>(alarmId));
                CancelAlarm(
                    target.rdram.data(),
                    &cancelCtx,
                    &target.runtime);
                return getRegS32(cancelCtx, 2);
            };

            const int32_t firstAlarm = setLongAlarm(first);
            const int32_t secondAlarm = setLongAlarm(second);
            t.Equals(
                firstAlarm,
                1,
                "the first runtime should allocate alarm id 1");
            t.Equals(
                secondAlarm,
                1,
                "a second runtime should independently allocate alarm id 1");

            t.Equals(
                cancelAlarm(first, firstAlarm),
                KE_OK,
                "the first runtime should cancel its own alarm");
            t.Equals(
                cancelAlarm(first, firstAlarm),
                KE_ERROR,
                "the first runtime should no longer contain that alarm");
            t.Equals(
                cancelAlarm(second, secondAlarm),
                KE_OK,
                "first-runtime cancellation must not consume the second alarm");
        });

        tc.Run("alarm shutdown discards a published callback without a guest-worker join", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kAlarmHandlerAddr = 0x00270200u;
            env.runtime.registerFunction(
                kAlarmHandlerAddr, &alarmCaptureHandler);
            writeGuestU32(
                env.rdram.data(),
                K_ALARM_CALLBACK_STAGE_ADDR,
                0u);

            auto guestExecution =
                std::make_unique<PS2Runtime::GuestExecutionScope>(
                    &env.runtime, &env.ctx);

            setRegU32(env.ctx, 4, 1u);
            setRegU32(env.ctx, 5, kAlarmHandlerAddr);
            setRegU32(env.ctx, 6, 0u);
            SetAlarm(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(
                getRegS32(env.ctx, 2) > 0,
                "the shutdown fixture should create a short alarm");

            const bool callbackPublished = waitUntil(
                [&]()
                {
                    return env.runtime
                               .pendingAlarmCallbackCountForTesting() >
                           0u;
                },
                std::chrono::milliseconds(1000));
            t.IsTrue(
                callbackPublished,
                "the expired alarm should publish its callback while the guest boundary is held");

            std::atomic<bool> stopReturned{false};
            std::thread stopThread;
            if (callbackPublished)
            {
                stopThread = std::thread(
                    [&]()
                    {
                        env.runtime.requestStop();
                        stopReturned.store(
                            true, std::memory_order_release);
                    });
            }

            const bool returnedWhileGuestHeld =
                callbackPublished &&
                waitUntil(
                    [&]()
                    {
                        return stopReturned.load(
                            std::memory_order_acquire);
                    },
                    std::chrono::milliseconds(100));

            guestExecution.reset();
            if (stopThread.joinable())
            {
                stopThread.join();
            }

            t.IsTrue(
                returnedWhileGuestHeld,
                "requestStop should join only the alarm publication worker");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_ALARM_CALLBACK_STAGE_ADDR),
                0u,
                "shutdown should discard a published callback before guest execution");
        });

        tc.Run("alarm callbacks use runtime-owned context and ABI arguments", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kAlarmHandlerAddr = 0x00270300u;
            constexpr uint32_t kAlarmArgument = 0x4A17C0DEu;
            constexpr uint32_t kAlarmGp = 0x00126000u;
            env.runtime.registerFunction(
                kAlarmHandlerAddr, &alarmCaptureHandler);

            writeGuestU32(
                env.rdram.data(),
                K_ALARM_CALLBACK_STAGE_ADDR,
                0u);
            setRegU32(env.ctx, 4, 1u);
            setRegU32(env.ctx, 5, kAlarmHandlerAddr);
            setRegU32(env.ctx, 6, kAlarmArgument);
            setRegU32(env.ctx, 28, kAlarmGp);
            SetAlarm(
                env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t alarmId = getRegS32(env.ctx, 2);
            t.Equals(
                alarmId,
                1,
                "a fresh runtime should allocate alarm id 1");

            const bool callbackRan = waitUntil(
                [&]()
                {
                    {
                        PS2Runtime::GuestExecutionScope
                            guestExecution(
                                &env.runtime,
                                &env.ctx);
                        env.runtime.executeGuestStep(
                            env.rdram.data(),
                            &env.ctx,
                            &alarmNoopHandler);
                    }
                    return readGuestU32(
                               env.rdram.data(),
                               K_ALARM_CALLBACK_STAGE_ADDR) ==
                           1u;
                },
                std::chrono::milliseconds(1000));
            t.IsTrue(
                callbackRan,
                "the short alarm should dispatch its guest callback");
            if (callbackRan)
            {
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_ALARM_CALLBACK_ID_ADDR),
                    static_cast<uint32_t>(alarmId),
                    "alarm callback a0 should contain its id");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_ALARM_CALLBACK_TICKS_ADDR),
                    1u,
                    "alarm callback a1 should contain the requested ticks");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_ALARM_CALLBACK_ARG_ADDR),
                    kAlarmArgument,
                    "alarm callback a2 should contain the common argument");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        K_ALARM_CALLBACK_GP_ADDR),
                    kAlarmGp,
                    "alarm callback should retain the registering guest gp");
                const uint32_t callbackSp =
                    readGuestU32(
                        env.rdram.data(),
                        K_ALARM_CALLBACK_SP_ADDR);
                t.IsTrue(
                    callbackSp != 0u &&
                        (callbackSp & 0xFu) == 0u,
                    "alarm callback should use a nonzero aligned runtime-owned stack");
            }

            R5900Context cancelCtx{};
            setRegU32(
                cancelCtx,
                4,
                static_cast<uint32_t>(alarmId));
            CancelAlarm(
                env.rdram.data(),
                &cancelCtx,
                &env.runtime);
            t.Equals(
                getRegS32(cancelCtx, 2),
                KE_ERROR,
                "a dispatched alarm should no longer be cancellable");
        });

        tc.Run("alarm workers publish callbacks to an EE execution boundary", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kAlarmHandlerAddr = 0x00270380u;
            env.runtime.registerFunction(
                kAlarmHandlerAddr,
                &alarmThreadCaptureHandler);
            {
                std::lock_guard<std::mutex> lock(
                    g_alarmCallbackThreadMutex);
                g_alarmCallbackThread = {};
            }

            setRegU32(env.ctx, 4, 1u);
            setRegU32(env.ctx, 5, kAlarmHandlerAddr);
            setRegU32(env.ctx, 6, 0u);
            SetAlarm(
                env.rdram.data(),
                &env.ctx,
                &env.runtime);
            t.IsTrue(
                getRegS32(env.ctx, 2) > 0,
                "the executor-dispatch fixture should create a short alarm");

            const std::thread::id boundaryThread =
                std::this_thread::get_id();
            std::thread::id callbackThread;
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(1);
            while (std::chrono::steady_clock::now() <
                   deadline)
            {
                {
                    PS2Runtime::GuestExecutionScope
                        guestExecution(
                            &env.runtime,
                            &env.ctx);
                    env.runtime.executeGuestStep(
                        env.rdram.data(),
                        &env.ctx,
                        &alarmNoopHandler);
                }
                {
                    std::lock_guard<std::mutex> lock(
                        g_alarmCallbackThreadMutex);
                    callbackThread =
                        g_alarmCallbackThread;
                }
                if (callbackThread != std::thread::id{})
                {
                    break;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));
            }

            t.IsTrue(
                callbackThread != std::thread::id{},
                "the short alarm should reach an EE execution boundary");
            t.IsTrue(
                callbackThread == boundaryThread,
                "the alarm timer worker must publish work instead of executing guest code");
        });

        tc.Run("alarm self-stop leaves callback lifetime with the EE boundary", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kAlarmHandlerAddr = 0x00270400u;
            env.runtime.registerFunction(
                kAlarmHandlerAddr, &alarmSelfStopHandler);
            writeGuestU32(
                env.rdram.data(),
                K_ALARM_SELF_STOP_STAGE_ADDR,
                0u);
            writeGuestU32(
                env.rdram.data(),
                K_ALARM_SELF_STOP_GATE_ADDR,
                0u);

            setRegU32(env.ctx, 4, 1u);
            setRegU32(env.ctx, 5, kAlarmHandlerAddr);
            setRegU32(env.ctx, 6, 0u);
            SetAlarm(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(
                getRegS32(env.ctx, 2) > 0,
                "the self-stop fixture should create a short alarm");

            std::atomic<bool> executorReturned{false};
            std::thread executorThread(
                [&]()
                {
                    const auto deadline =
                        std::chrono::steady_clock::now() +
                        std::chrono::seconds(1);
                    while (
                        readGuestU32(
                            env.rdram.data(),
                            K_ALARM_SELF_STOP_STAGE_ADDR) == 0u &&
                        std::chrono::steady_clock::now() <
                            deadline)
                    {
                        {
                            PS2Runtime::GuestExecutionScope
                                guestExecution(
                                    &env.runtime,
                                    &env.ctx);
                            env.runtime.executeGuestStep(
                                env.rdram.data(),
                                &env.ctx,
                                &alarmNoopHandler);
                        }
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1));
                    }
                    executorReturned.store(
                        true, std::memory_order_release);
                });

            const bool selfStopReturned = waitUntil(
                [&]()
                {
                    return readGuestU32(
                               env.rdram.data(),
                               K_ALARM_SELF_STOP_STAGE_ADDR) ==
                           1u;
                },
                std::chrono::milliseconds(1000));
            t.IsTrue(
                selfStopReturned,
                "the callback should return from its own stop request");

            std::atomic<bool> externalStopReturned{false};
            std::thread externalStopThread;
            if (selfStopReturned)
            {
                externalStopThread = std::thread(
                    [&]()
                    {
                        env.runtime.requestStop();
                        externalStopReturned.store(
                            true, std::memory_order_release);
                    });
            }

            const bool returnedWhileCallbackHeld =
                selfStopReturned &&
                waitUntil(
                    [&]()
                    {
                        return externalStopReturned.load(
                            std::memory_order_acquire);
                    },
                    std::chrono::milliseconds(100));

            writeGuestU32(
                env.rdram.data(),
                K_ALARM_SELF_STOP_GATE_ADDR,
                1u);
            if (externalStopThread.joinable())
            {
                externalStopThread.join();
            }
            if (executorThread.joinable())
            {
                executorThread.join();
            }

            t.IsTrue(
                returnedWhileCallbackHeld,
                "an external stop should join only the publication worker");
            t.IsTrue(
                externalStopReturned.load(
                    std::memory_order_acquire),
                "external stop should return while the EE callback is held");
            t.IsTrue(
                executorReturned.load(
                    std::memory_order_acquire),
                "the EE boundary should return after the callback gate opens");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    K_ALARM_SELF_STOP_STAGE_ADDR),
                2u,
                "the callback should exit after its gate is released");
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

        tc.Run(
            "SetupThread reserves the retail EE thread context",
            [](TestCase &t)
            {
                TestEnv env;

                // The strict retail-BIOS oracle returns 0x001202D0 for this
                // exact 0x00100570 base and 0x20000-byte stack.
                setRegU32(env.ctx, 5, 0x00100570u);
                setRegU32(env.ctx, 6, 0x00020000u);
                t.IsTrue(
                    callSyscall(
                        0x3Cu,
                        env.rdram.data(),
                        &env.ctx,
                        &env.runtime),
                    "SetupThread with an explicit stack should dispatch");
                t.Equals(
                    getRegU32(&env.ctx, 2),
                    0x001202D0u,
                    "SetupThread should reserve the retail EE thread context at the stack top");
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

        tc.Run("guest kernel state is isolated per runtime", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv first;
            TestEnv second;

            constexpr uint32_t kFirstConfigAddr = 0x00005000u;
            constexpr uint32_t kSecondConfigAddr = 0x00005010u;
            constexpr uint32_t kFirstConfigOutAddr = 0x00005020u;
            constexpr uint32_t kFirstConfig =
                (1u << 0) |
                (1u << 4) |
                (1u << 13) |
                (1u << 16);
            constexpr uint32_t kSecondConfig =
                (1u << 0) |
                (1u << 4) |
                (2u << 13) |
                (10u << 16);

            writeGuestU32(
                first.rdram.data(),
                kFirstConfigAddr,
                kFirstConfig);
            setRegU32(first.ctx, 4, kFirstConfigAddr);
            SetOsdConfigParam(
                first.rdram.data(),
                &first.ctx,
                &first.runtime);

            writeGuestU32(
                second.rdram.data(),
                kSecondConfigAddr,
                kSecondConfig);
            setRegU32(second.ctx, 4, kSecondConfigAddr);
            SetOsdConfigParam(
                second.rdram.data(),
                &second.ctx,
                &second.runtime);

            setRegU32(first.ctx, 4, kFirstConfigOutAddr);
            GetOsdConfigParam(
                first.rdram.data(),
                &first.ctx,
                &first.runtime);
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kFirstConfigOutAddr),
                kFirstConfig,
                "the second runtime must not replace the first runtime's OSD configuration");

            setRegU32(first.ctx, 4, 3u);
            QueryBootMode(
                first.rdram.data(),
                &first.ctx,
                &first.runtime);
            const uint32_t firstBootMode =
                ::getRegU32(&first.ctx, 2);

            setRegU32(second.ctx, 4, 3u);
            QueryBootMode(
                second.rdram.data(),
                &second.ctx,
                &second.runtime);
            const uint32_t secondBootMode =
                ::getRegU32(&second.ctx, 2);

            t.Equals(
                secondBootMode,
                firstBootMode,
                "each runtime should independently allocate the same guest boot-mode address");
            t.IsTrue(
                firstBootMode != 0u &&
                    readGuestU32(
                        first.rdram.data(),
                        firstBootMode) != 0u,
                "the first runtime should populate its boot-mode table");
            t.Equals(
                readGuestU32(
                    second.rdram.data(),
                    secondBootMode),
                readGuestU32(
                    first.rdram.data(),
                    firstBootMode),
                "the second runtime should populate its own boot-mode table");

            GetThreadTLS(
                first.rdram.data(),
                &first.ctx,
                &first.runtime);
            const uint32_t firstTls =
                ::getRegU32(&first.ctx, 2);
            GetThreadTLS(
                second.rdram.data(),
                &second.ctx,
                &second.runtime);
            const uint32_t secondTls =
                ::getRegU32(&second.ctx, 2);
            t.Equals(
                secondTls,
                firstTls,
                "each runtime should independently allocate the first TLS slot");

            constexpr uint32_t kSyscallIndex = 0x91u;
            constexpr uint32_t kFirstHandler = 0x00200000u;
            constexpr uint32_t kSecondHandler = 0x00200010u;
            first.runtime.registerFunction(
                kFirstHandler,
                overrideReturnHandler);
            second.runtime.registerFunction(
                kSecondHandler,
                overrideBrokenHandler);

            setRegU32(first.ctx, 4, kSyscallIndex);
            setRegU32(first.ctx, 5, kFirstHandler);
            SetSyscall(
                first.rdram.data(),
                &first.ctx,
                &first.runtime);
            setRegU32(second.ctx, 4, kSyscallIndex);
            setRegU32(second.ctx, 5, kSecondHandler);
            SetSyscall(
                second.rdram.data(),
                &second.ctx,
                &second.runtime);

            setRegU32(first.ctx, 4, 7u);
            setRegU32(first.ctx, 5, 5u);
            t.IsTrue(
                callSyscall(
                    kSyscallIndex,
                    first.rdram.data(),
                    &first.ctx,
                    &first.runtime),
                "the first runtime's syscall override should dispatch");
            t.Equals(
                static_cast<uint32_t>(
                    getRegS32(first.ctx, 2)),
                12u,
                "the second runtime must not replace the first runtime's syscall override");

            notifyRuntimeStop(&second.runtime);
            setRegU32(first.ctx, 4, 9u);
            setRegU32(first.ctx, 5, 4u);
            t.IsTrue(
                callSyscall(
                    kSyscallIndex,
                    first.rdram.data(),
                    &first.ctx,
                    &first.runtime),
                "the first runtime's syscall override should survive stopping the second");
            t.Equals(
                static_cast<uint32_t>(
                    getRegS32(first.ctx, 2)),
                13u,
                "stopping the second runtime must not erase the first runtime's syscall override");

            notifyRuntimeStop(&first.runtime);
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
            initializeGuestKernelState(
                env.rdram.data(), &env.runtime);

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
            initializeGuestKernelState(
                env.rdram.data(), &env.runtime);

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
            initializeGuestKernelState(
                env.rdram.data(), &env.runtime);

            constexpr uint32_t kGuestSyscallTableGuestBase = 0x80011F80u;
            constexpr uint32_t kGuestSyscallTableProbeBase = 0x000002F0u;
            constexpr uint32_t kSyscallIndex = 0x5Au;
            constexpr uint32_t kHandler = 0x00383510u;
            constexpr uint32_t kEntryPhysAddr = (kGuestSyscallTableGuestBase + (kSyscallIndex * 4u)) & 0x1FFFFFFFu;

            setRegU32(env.ctx, 4, kSyscallIndex);
            setRegU32(env.ctx, 5, kHandler);
            t.IsTrue(callSyscall(0x74u, env.rdram.data(), &env.ctx, &env.runtime),
                     "SetSyscall syscall should dispatch");

            notifyRuntimeStop(&env.runtime);
            initializeGuestKernelState(
                env.rdram.data(), &env.runtime);

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
            initializeGuestKernelState(
                env.rdram.data(), &env.runtime);

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
