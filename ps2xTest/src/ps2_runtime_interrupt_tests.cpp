#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_syscalls.h"
#include "Stubs/DMA.h"
#include "Stubs/GS.h"
#include "runtime/ps2_gs_gpu.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

using namespace ps2_syscalls;

namespace
{
    constexpr int KE_OK = 0;
    constexpr int KE_EVF_COND = -421;

    constexpr uint32_t WEF_OR = 1u;
    constexpr uint32_t WEF_CLEAR = 0x10u;
    constexpr uint32_t WEF_CLEAR_ALL = 0x20u;
    constexpr uint32_t kInterruptIsolationHitAddr = 0x1180u;
    constexpr uint32_t kInterruptIsolationSpAddr = 0x1184u;
    constexpr uint32_t kInterruptIsolationPcAddr = 0x1188u;
    constexpr uint32_t
        kExecutorBoundaryThreadParamAddr = 0x1200u;
    constexpr uint32_t
        kExecutorBoundaryResumeAddress = 0x00ABC3C0u;

    enum class ExecutorBoundaryInterruptKind
    {
        VSync,
        Dmac,
        Counter,
    };

    struct Ps2EventFlagInfo
    {
        uint32_t attr;
        uint32_t option;
        uint32_t initBits;
        uint32_t currBits;
        int32_t numThreads;
        int32_t reserved1;
        int32_t reserved2;
    };

    static_assert(sizeof(Ps2EventFlagInfo) == 28u, "Unexpected Ps2EventFlagInfo layout.");

    struct TestEnv
    {
        std::vector<uint8_t> rdram;
        PS2Runtime runtime;

        TestEnv() : rdram(PS2_RAM_SIZE, 0u)
        {
        }
    };

    std::atomic<uint32_t> g_vblankStartHits{0u};
    std::atomic<uint32_t> g_vblankEndHits{0u};
    std::atomic<uint32_t> g_vsyncCallbackHits{0u};
    std::atomic<uint32_t> g_vsyncCallbackLastTick{0u};
    std::atomic<uint32_t> g_lastIntcArg{0u};
    std::atomic<uint32_t> g_dmacSendHits{0u};
    std::atomic<uint32_t> g_dmacSendLastCause{0u};
    std::atomic<uint32_t> g_dmacSendLastChcr{0u};
    std::atomic<uint32_t> g_dmacPublishedAddress{0u};
    std::atomic<uint32_t> g_dmacObservedPublishedValue{0u};
    std::atomic<uint32_t> g_dmacHandlerResubmissions{0u};
    std::atomic<uint32_t> g_eeCounterHits{0u};
    std::atomic<uint32_t> g_eeCounterLastCause{0u};
    std::atomic<uint32_t> g_eeCounterModeAtHandler{0u};
    std::atomic<uint32_t> g_eeCounterIntcAtHandler{0u};
    std::atomic<uint32_t>
        g_executorBoundaryCallbackHits{0u};
    std::atomic<ExecutorBoundaryInterruptKind>
        g_executorBoundaryInterruptKind{
            ExecutorBoundaryInterruptKind::VSync};
    std::atomic<bool>
        g_executorBoundaryMainWaiting{false};
    std::atomic<bool>
        g_executorBoundaryCallbackRanWhileWaiting{
            false};
    std::atomic<bool>
        g_executorBoundaryMainCompleted{false};
    std::atomic<int>
        g_executorBoundaryChildThreadId{0};
    std::atomic<int>
        g_executorBoundaryWakeResult{-1};
    std::atomic<bool>
        g_executorBoundaryCallbackActive{false};
    std::atomic<bool>
        g_executorBoundaryChildWaiting{false};
    std::atomic<bool>
        g_executorBoundaryChildCompleted{false};
    std::atomic<bool>
        g_executorBoundaryChildRanInsideCallback{
            false};
    std::atomic<bool>
        g_executorBoundaryMainResumedBeforeChild{
            false};
    std::atomic<bool>
        g_interruptIsolationBlockNext{false};
    std::atomic<bool>
        g_interruptIsolationBlockEntered{false};
    std::atomic<bool>
        g_interruptIsolationBlockRelease{false};

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    int32_t getRegS32(const R5900Context &ctx, int reg)
    {
        return static_cast<int32_t>(::getRegU32(&ctx, reg));
    }

    bool callSyscall(uint32_t syscallNumber, uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        return dispatchNumericSyscall(syscallNumber, rdram, ctx, runtime);
    }

    void writeGuestU32(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    void writeGuestU64(uint8_t *rdram, uint32_t addr, uint64_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    uint64_t makeDmaTag(uint16_t qwc, uint8_t id, uint32_t addr, bool irq = false)
    {
        return static_cast<uint64_t>(qwc) |
               (static_cast<uint64_t>(id & 0x7u) << 28) |
               (irq ? (1ull << 31) : 0ull) |
               (static_cast<uint64_t>(addr & 0x7FFFFFFFu) << 32);
    }

    void writeDmaTag(uint8_t *rdram, uint32_t tagAddr, uint64_t tagLo)
    {
        std::memset(rdram + tagAddr, 0, 16);
        std::memcpy(rdram + tagAddr, &tagLo, sizeof(tagLo));
    }

    uint32_t readGuestU32(const uint8_t *rdram, uint32_t addr)
    {
        uint32_t value = 0;
        std::memcpy(&value, rdram + addr, sizeof(value));
        return value;
    }

    uint64_t readGuestU64(const uint8_t *rdram, uint32_t addr)
    {
        uint64_t value = 0;
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

    void cleanupRuntime(TestEnv &env)
    {
        env.runtime.requestStop();
        notifyRuntimeStop();
    }

    bool serviceNextScheduledBoundary(
        TestEnv &env, R5900Context &context)
    {
        const PS2Runtime::DebugEeScheduler scheduler =
            env.runtime.debugEeSchedulerSnapshot();
        if (!scheduler.hasNextDeadline)
        {
            return false;
        }

        const uint64_t now =
            env.runtime.currentEeTick().raw();
        const uint64_t elapsed =
            scheduler.nextDeadlineTick > now
                ? scheduler.nextDeadlineTick - now
                : 0u;
        if (elapsed >
            std::numeric_limits<uint32_t>::max())
        {
            return false;
        }

        {
            PS2Runtime::GuestExecutionScope guestExecution(
                &env.runtime, &context);
            context.cop0_config |= 1u << 18u;
            context.advanceEeCycleTicks(
                static_cast<uint32_t>(elapsed));
            env.runtime.serviceEeEventsAtBlockBoundary(
                env.rdram.data(), &context);
        }
        return true;
    }

    bool serviceAllScheduledBoundaries(
        TestEnv &env,
        R5900Context &context,
        size_t maximumBoundaries = 64u)
    {
        for (size_t boundary = 0u;
             boundary < maximumBoundaries;
             ++boundary)
        {
            if (!env.runtime
                     .debugEeSchedulerSnapshot()
                     .hasNextDeadline)
            {
                return true;
            }
            if (!serviceNextScheduledBoundary(
                    env, context))
            {
                return false;
            }
        }
        return !env.runtime
                    .debugEeSchedulerSnapshot()
                    .hasNextDeadline;
    }

    void testIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        const uint32_t cause = getRegU32(ctx, 4);
        const uint32_t arg = getRegU32(ctx, 5);
        g_lastIntcArg.store(arg, std::memory_order_relaxed);

        if (cause == 2u)
        {
            g_vblankStartHits.fetch_add(1u, std::memory_order_relaxed);
        }
        else if (cause == 3u)
        {
            g_vblankEndHits.fetch_add(1u, std::memory_order_relaxed);
        }

        ctx->pc = 0u;
    }

    void testInterruptIsolationHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        (void)runtime;
        if (g_interruptIsolationBlockNext.exchange(
                false, std::memory_order_acq_rel))
        {
            g_interruptIsolationBlockEntered.store(
                true, std::memory_order_release);
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(2);
            while (!g_interruptIsolationBlockRelease.load(
                       std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() <
                       deadline)
            {
                std::this_thread::yield();
            }
        }
        writeGuestU32(
            rdram,
            kInterruptIsolationHitAddr,
            readGuestU32(
                rdram,
                kInterruptIsolationHitAddr) +
                1u);
        writeGuestU32(
            rdram,
            kInterruptIsolationSpAddr,
            getRegU32(ctx, 29));
        writeGuestU32(
            rdram,
            kInterruptIsolationPcAddr,
            ctx->pc);
        ctx->pc = 0u;
    }

    void testDmacSendHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        g_dmacSendHits.fetch_add(1u, std::memory_order_relaxed);
        g_dmacSendLastCause.store(cause, std::memory_order_relaxed);

        const uint32_t publishedAddress = g_dmacPublishedAddress.load(std::memory_order_relaxed);
        if (rdram && publishedAddress != 0u)
        {
            g_dmacObservedPublishedValue.store(readGuestU32(rdram, publishedAddress), std::memory_order_relaxed);
        }

        uint32_t channelBase = 0u;
        if (cause == 0u)
        {
            channelBase = 0x10008000u;
        }
        else if (cause == 1u)
        {
            channelBase = 0x10009000u;
        }
        else if (cause == 2u)
        {
            channelBase = 0x1000A000u;
        }

        if (runtime && channelBase != 0u)
        {
            g_dmacSendLastChcr.store(runtime->memory().readIORegister(channelBase + 0x00u), std::memory_order_relaxed);
        }

        ctx->pc = 0u;
    }

    void testDmacResubmittingHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        (void)rdram;
        g_dmacSendHits.fetch_add(1u, std::memory_order_relaxed);

        if (runtime &&
            g_dmacHandlerResubmissions.exchange(
                0u, std::memory_order_relaxed) != 0u)
        {
            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kVif1StatusBit = 1u << 1u;

            // A handler can acknowledge the current completion, submit the
            // next chain, and reach an event-test boundary before returning.
            (void)runtime->memory().writeIORegister(
                kDStat, kVif1StatusBit);
            const DmacTransferToken next =
                runtime->memory().beginDmacTransfer(
                    DmacChannel::Vif1);
            (void)runtime->memory().requestDmacCompletion(next);
            runtime->serviceEeEventsAtBlockBoundary(
                runtime->memory().getRDRAM(), ctx);
        }

        ctx->pc = 0u;
    }

    void testEeCounterHandler(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        (void)rdram;
        g_eeCounterHits.fetch_add(
            1u, std::memory_order_relaxed);
        g_eeCounterLastCause.store(
            getRegU32(ctx, 4),
            std::memory_order_relaxed);
        if (runtime)
        {
            g_eeCounterModeAtHandler.store(
                runtime->memory().readIORegister(
                    0x10000010u),
                std::memory_order_relaxed);
            g_eeCounterIntcAtHandler.store(
                runtime->memory().readIORegister(
                    0x1000F000u),
                std::memory_order_relaxed);
        }
        ctx->pc = 0u;
    }

    void testVSyncCallback(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        g_vsyncCallbackHits.fetch_add(
            1u, std::memory_order_relaxed);
        g_vsyncCallbackLastTick.store(
            getRegU32(ctx, 4), std::memory_order_relaxed);
        ctx->pc = 0u;
    }

    void testExecutorBoundaryWakeCallback(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_executorBoundaryCallbackHits.fetch_add(
            1u, std::memory_order_acq_rel);
        if (g_executorBoundaryMainWaiting.load(
                std::memory_order_acquire))
        {
            g_executorBoundaryCallbackRanWhileWaiting
                .store(
                    true,
                    std::memory_order_release);
        }
        g_executorBoundaryCallbackActive.store(
            true, std::memory_order_release);
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(
                g_executorBoundaryChildThreadId.load(
                    std::memory_order_acquire)));
        iWakeupThread(rdram, ctx, runtime);
        g_executorBoundaryWakeResult.store(
            getRegS32(*ctx, 2),
            std::memory_order_release);
        g_executorBoundaryCallbackActive.store(
            false, std::memory_order_release);
        ctx->pc = 0u;
    }

    void testExecutorBoundaryChild(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        g_executorBoundaryChildWaiting.store(
            true, std::memory_order_release);
        SleepThread(rdram, ctx, runtime);
        g_executorBoundaryChildRanInsideCallback.store(
            g_executorBoundaryCallbackActive.load(
                std::memory_order_acquire),
            std::memory_order_release);
        g_executorBoundaryChildCompleted.store(
            true, std::memory_order_release);
        ctx->pc = 0u;
    }

    void testExecutorBoundaryMainResume(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        g_executorBoundaryMainWaiting.store(
            false, std::memory_order_release);
        g_executorBoundaryMainResumedBeforeChild.store(
            !g_executorBoundaryChildCompleted.load(
                std::memory_order_acquire),
            std::memory_order_release);
        g_executorBoundaryMainCompleted.store(
            true, std::memory_order_release);
        ctx->pc = 0u;
    }

    void testExecutorBoundaryMain(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        InitThread(rdram, ctx, runtime);
        setRegU32(
            *ctx, 4,
            kExecutorBoundaryThreadParamAddr);
        CreateThread(rdram, ctx, runtime);
        const int childThreadId =
            getRegS32(*ctx, 2);
        g_executorBoundaryChildThreadId.store(
            childThreadId,
            std::memory_order_release);
        if (childThreadId <= 1)
        {
            g_executorBoundaryMainCompleted.store(
                true, std::memory_order_release);
            ctx->pc = 0u;
            return;
        }
        setRegU32(
            *ctx,
            4,
            static_cast<uint32_t>(childThreadId));
        setRegU32(*ctx, 5, 0u);
        StartThread(rdram, ctx, runtime);

        switch (g_executorBoundaryInterruptKind.load(
            std::memory_order_acquire))
        {
        case ExecutorBoundaryInterruptKind::VSync:
            break;
        case ExecutorBoundaryInterruptKind::Dmac:
        {
            runtime->memory().writeIORegister(
                0x10009020u, 3u);
            const DmacTransferToken transfer =
                runtime->memory().beginDmacTransfer(
                    DmacChannel::Vif1);
            (void)runtime->memory().requestDmacCompletion(
                transfer);
            break;
        }
        case ExecutorBoundaryInterruptKind::Counter:
            runtime->Store32(
                rdram,
                ctx,
                0x10000020u,
                3u);
            runtime->Store32(
                rdram,
                ctx,
                0x10000000u,
                0u);
            runtime->Store32(
                rdram,
                ctx,
                0x10000010u,
                0x1c0u);
            break;
        }

        const PS2Runtime::DebugEeScheduler scheduler =
            runtime->debugEeSchedulerSnapshot();
        if (!scheduler.hasNextDeadline)
        {
            g_executorBoundaryMainCompleted.store(
                true, std::memory_order_release);
            ctx->pc = 0u;
            return;
        }

        const uint64_t now =
            runtime->currentEeTick().raw();
        const uint64_t elapsed =
            scheduler.nextDeadlineTick > now
                ? scheduler.nextDeadlineTick - now
                : 0u;
        if (elapsed >
            std::numeric_limits<uint32_t>::max())
        {
            g_executorBoundaryMainCompleted.store(
                true, std::memory_order_release);
            ctx->pc = 0u;
            return;
        }

        g_executorBoundaryMainWaiting.store(
            true, std::memory_order_release);
        ctx->cop0_config |= 1u << 18u;
        ctx->advanceEeCycleTicks(
            static_cast<uint32_t>(elapsed));
        runtime->serviceEeEventsAtBlockBoundary(
            rdram, ctx);
        ctx->pc = kExecutorBoundaryResumeAddress;
        if (runtime->checkpointGuestExecution(ctx) ==
            PS2GuestCheckpointResult::ExitToDispatcher)
        {
            return;
        }
        testExecutorBoundaryMainResume(
            rdram, ctx, runtime);
    }
}

void register_ps2_runtime_interrupt_tests()
{
    MiniTest::Case("PS2RuntimeInterrupt", [](TestCase &tc)
    {
        tc.Run("event VSync follows emulated NTSC phases and defers guest callbacks", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            ps2_stubs::resetGsSyncVCallbackState(
                &env.runtime);
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");
            env.runtime.configureEeVSyncVideoMode(
                0x02u, true);

            constexpr uint64_t kTicksPerEeCycle = 8u;
            constexpr uint64_t kRenderCycles = 4'498'396u;
            constexpr uint64_t kGsBlankCycles = 65'601u;
            constexpr uint64_t kBlankCycles = 421'724u;
            constexpr uint64_t kPeriodCycles = 4'920'120u;
            constexpr uint64_t kGsCsrFieldMask = 0x2000ull;
            constexpr uint32_t kFlagAddr = 0x10A0u;
            constexpr uint32_t kTickAddr = 0x10B0u;
            constexpr uint32_t kIntcHandlerAddr = 0x00ABC240u;
            constexpr uint32_t kGsCallbackAddr = 0x00ABC280u;

            g_vblankStartHits.store(
                0u, std::memory_order_relaxed);
            g_vblankEndHits.store(
                0u, std::memory_order_relaxed);
            g_vsyncCallbackHits.store(
                0u, std::memory_order_relaxed);
            g_vsyncCallbackLastTick.store(
                0u, std::memory_order_relaxed);
            env.runtime.registerFunction(
                kIntcHandlerAddr, &testIntcHandler);
            env.runtime.registerFunction(
                kGsCallbackAddr, &testVSyncCallback);

            for (const uint32_t cause : {2u, 3u})
            {
                R5900Context addCtx{};
                setRegU32(addCtx, 4, cause);
                setRegU32(addCtx, 5, kIntcHandlerAddr);
                setRegU32(addCtx, 6, 0u);
                setRegU32(addCtx, 7, cause);
                AddIntcHandler(
                    env.rdram.data(), &addCtx,
                    &env.runtime);
                t.IsTrue(
                    getRegS32(addCtx, 2) > 0,
                    "event VSync fixture should register both INTC edges");
            }

            R5900Context callbackCtx{};
            setRegU32(
                callbackCtx, 4, kGsCallbackAddr);
            ps2_stubs::sceGsSyncVCallback(
                env.rdram.data(), &callbackCtx,
                &env.runtime);

            R5900Context flagCtx{};
            setRegU32(flagCtx, 4, kFlagAddr);
            setRegU32(flagCtx, 5, kTickAddr);
            SetVSyncFlag(
                env.rdram.data(), &flagCtx,
                &env.runtime);

            const uint64_t initialVSyncTick =
                GetCurrentVSyncTick(&env.runtime);
            const PS2Runtime::DebugEeScheduler initial =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &initialSlot =
                initial.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.IsTrue(
                initialSlot.pending,
                "event mode should retain one scheduled VSync source");
            t.Equals(
                initialSlot.deadlineTick,
                kRenderCycles * kTicksPerEeCycle,
                "the first NTSC VBlank start should follow the render interval");
            t.IsTrue(
                initialSlot.device.kind ==
                    PS2Runtime::DebugEeEventDeviceKind::
                        VSync,
                "scheduler status should expose typed VSync phase state");
            t.Equals(
                initialSlot.device.phase, 0u,
                "the initial VSync phase should be start");

            std::this_thread::sleep_for(
                std::chrono::milliseconds(25));
            t.Equals(
                GetCurrentVSyncTick(&env.runtime),
                initialVSyncTick,
                "host wall time must not service event-mode VSync");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), kFlagAddr),
                0u,
                "the one-shot flag must remain clear without emulated progress");

            env.runtime.memory().gs().csr =
                kGsCsrFieldMask | 0x3ull;
            env.runtime.debugStartEeEventTrace(
                3u, std::nullopt, true);
            R5900Context timingCtx{};
            timingCtx.cop0_config = 1u << 18u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        initialSlot.deadlineTick -
                        kTicksPerEeCycle));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
                t.Equals(
                    GetCurrentVSyncTick(&env.runtime),
                    initialVSyncTick,
                    "VSync must remain pending immediately before its deadline");

                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        kTicksPerEeCycle));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
                t.Equals(
                    g_vblankStartHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "VBlank INTC start must wait for the outer guest boundary");
                t.Equals(
                    g_vsyncCallbackHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "GS VSync callback must wait for the outer guest boundary");
            }

            const uint64_t firstVSyncTick =
                initialVSyncTick + 1u;
            t.Equals(
                GetCurrentVSyncTick(&env.runtime),
                firstVSyncTick,
                "VBlank start should publish exactly one VSync tick");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), kFlagAddr),
                1u,
                "VBlank start should publish the one-shot flag");
            t.Equals(
                readGuestU64(
                    env.rdram.data(), kTickAddr),
                firstVSyncTick,
                "VBlank start should publish the matching tick value");
            t.Equals(
                g_vblankStartHits.load(
                    std::memory_order_relaxed),
                1u,
                "the outer guest boundary should deliver VBlank start");
            t.Equals(
                g_vsyncCallbackHits.load(
                    std::memory_order_relaxed),
                1u,
                "the outer guest boundary should deliver the GS callback");
            t.Equals(
                g_vsyncCallbackLastTick.load(
                    std::memory_order_relaxed),
                static_cast<uint32_t>(
                    firstVSyncTick),
                "the GS callback should receive the published tick");
            t.Equals(
                env.runtime.memory().gs().csr.load() &
                    kGsCsrFieldMask,
                kGsCsrFieldMask,
                "GS CSR.FIELD should not change at VBlank start");

            PS2Runtime::DebugEeScheduler afterStart =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &gsBlankSlot =
                afterStart.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                gsBlankSlot.deadlineTick,
                (kRenderCycles + kGsBlankCycles) *
                    kTicksPerEeCycle,
                "GS field publication should follow the reference scanline delay");
            t.Equals(
                gsBlankSlot.device.phase, 1u,
                "the source should retain its GS-blank phase");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        gsBlankSlot.deadlineTick -
                        env.runtime.currentEeTick().raw()));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
            }
            t.Equals(
                env.runtime.memory().gs().csr.load() &
                    kGsCsrFieldMask,
                0ull,
                "the delayed GS phase should swap interlaced FIELD");
            t.Equals(
                g_vblankEndHits.load(
                    std::memory_order_relaxed),
                0u,
                "GS field publication must precede VBlank end");

            PS2Runtime::DebugEeScheduler afterGsBlank =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &endSlot =
                afterGsBlank.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                endSlot.deadlineTick,
                (kRenderCycles + kBlankCycles) *
                    kTicksPerEeCycle,
                "VBlank end should retain the reference blank duration");
            t.Equals(
                endSlot.device.phase, 2u,
                "the source should retain its VBlank-end phase");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        endSlot.deadlineTick -
                        env.runtime.currentEeTick().raw()));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
                t.Equals(
                    g_vblankEndHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "VBlank end must remain deferred inside guest execution");
            }
            t.Equals(
                g_vblankEndHits.load(
                    std::memory_order_relaxed),
                1u,
                "the outer boundary should deliver VBlank end");

            const PS2Runtime::DebugEeScheduler afterEnd =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &nextStartSlot =
                afterEnd.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                nextStartSlot.deadlineTick,
                (kRenderCycles + kPeriodCycles) *
                    kTicksPerEeCycle,
                "the next VBlank start should retain field cadence without rebasing");
            t.Equals(
                nextStartSlot.device.phase, 0u,
                "VBlank end should return the source to start phase");

            const PS2Runtime::DebugEeEventTrace trace =
                env.runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(3u),
                "the three VSync phases should be independently observable");
            for (const auto &entry : trace.entries)
            {
                t.IsTrue(
                    entry.source ==
                        ps2x::timing::EeEventSource::
                            VSync,
                    "every phased entry should retain VSync source ownership");
            }

            cleanupRuntime(env);
            ps2_stubs::resetGsSyncVCallbackState(
                &env.runtime);
        });

        tc.Run(
            "legacy EE backends agree on due VSync DMAC and counter callback boundaries",
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
                    std::string backendName;
                    uint32_t callbackHits = 0u;
                    bool callbackRanWhileWaiting = false;
                    int childThreadId = 0;
                    int wakeResult = -1;
                    bool childWaiting = false;
                    bool childRanInsideCallback = false;
                    bool mainResumedBeforeChild = false;
                    bool executionQuiescent = false;
                    bool memoryCaptured = false;
                    std::vector<
                        PS2Runtime::
                            EeThreadContextImageForTesting>
                        contexts;
                    std::vector<uint8_t> memory;
                    std::vector<
                        ps2x::timing::EeEventSource>
                        eventSources;
                    uint64_t eventCount = 0u;
                    uint64_t modeledTick = 0u;
                    size_t managedAfterStop = 0u;
                };

                constexpr uint32_t kMainAddress =
                    0x00ABC300u;
                constexpr uint32_t kCallbackAddress =
                    0x00ABC340u;
                constexpr uint32_t kChildAddress =
                    0x00ABC380u;
                const auto runFixture =
                    [](ExecutorBoundaryInterruptKind
                           interruptKind,
                       EeExecutionBackendKind backendKind)
                    {
                        FixtureResult result{};
                        PS2RuntimeConfiguration
                            configuration{};
                        configuration.eeExecutionBackend =
                            backendKind;
                        configuration
                            .useEeExecutionBackendEnvironment =
                            false;
                        PS2Runtime runtime(configuration);
                        result.backendName =
                            runtime.eeExecutionBackendName();
                        result.memoryInitialized =
                            runtime.memory().initialize();
                        if (!result.memoryInitialized)
                        {
                            return result;
                        }
                        uint8_t *const rdram =
                            runtime.memory().getRDRAM();

                        runtime.registerFunction(
                            kMainAddress,
                            &testExecutorBoundaryMain);
                        runtime.registerFunction(
                            kCallbackAddress,
                            &testExecutorBoundaryWakeCallback);
                        runtime.registerFunction(
                            kChildAddress,
                            &testExecutorBoundaryChild);
                        runtime.registerFunction(
                            kExecutorBoundaryResumeAddress,
                            &testExecutorBoundaryMainResume);
                        const std::array<uint32_t, 9u>
                            childParameters{
                                0u,
                                kChildAddress,
                                0x00310000u,
                                0x800u,
                                0u,
                                0u,
                                0u,
                                0u,
                                0u,
                            };
                        for (size_t index = 0u;
                             index < childParameters.size();
                             ++index)
                        {
                            writeGuestU32(
                                rdram,
                                kExecutorBoundaryThreadParamAddr +
                                    static_cast<uint32_t>(
                                        index *
                                        sizeof(uint32_t)),
                                childParameters[index]);
                        }
                        switch (interruptKind)
                        {
                        case ExecutorBoundaryInterruptKind::
                            VSync:
                        {
                            ps2_stubs::
                                resetGsSyncVCallbackState(
                                    &runtime);
                            runtime
                                .configureEeVSyncVideoMode(
                                    0x02u, true);
                            R5900Context registration{};
                            setRegU32(
                                registration,
                                4,
                                kCallbackAddress);
                            ps2_stubs::sceGsSyncVCallback(
                                rdram,
                                &registration,
                                &runtime);
                            EnsureVSyncScheduled(
                                rdram, &runtime);
                            break;
                        }
                        case ExecutorBoundaryInterruptKind::
                            Dmac:
                        {
                            R5900Context registration{};
                            setRegU32(
                                registration, 4, 1u);
                            setRegU32(
                                registration,
                                5,
                                kCallbackAddress);
                            AddDmacHandler(
                                rdram,
                                &registration,
                                &runtime);
                            R5900Context enable{};
                            setRegU32(enable, 4, 1u);
                            EnableDmac(
                                rdram,
                                &enable,
                                &runtime);
                            break;
                        }
                        case ExecutorBoundaryInterruptKind::
                            Counter:
                        {
                            R5900Context registration{};
                            setRegU32(
                                registration, 4, 9u);
                            setRegU32(
                                registration,
                                5,
                                kCallbackAddress);
                            AddIntcHandler(
                                rdram,
                                &registration,
                                &runtime);
                            R5900Context enable{};
                            setRegU32(enable, 4, 9u);
                            EnableIntc(
                                rdram,
                                &enable,
                                &runtime);
                            break;
                        }
                        }

                        g_executorBoundaryInterruptKind.store(
                            interruptKind,
                            std::memory_order_release);
                        g_executorBoundaryCallbackHits.store(
                            0u, std::memory_order_release);
                        g_executorBoundaryMainWaiting.store(
                            false, std::memory_order_release);
                        g_executorBoundaryCallbackRanWhileWaiting
                            .store(
                                false,
                                std::memory_order_release);
                        g_executorBoundaryMainCompleted.store(
                            false, std::memory_order_release);
                        g_executorBoundaryChildThreadId.store(
                            0, std::memory_order_release);
                        g_executorBoundaryWakeResult.store(
                            -1, std::memory_order_release);
                        g_executorBoundaryCallbackActive.store(
                            false, std::memory_order_release);
                        g_executorBoundaryChildWaiting.store(
                            false, std::memory_order_release);
                        g_executorBoundaryChildCompleted.store(
                            false, std::memory_order_release);
                        g_executorBoundaryChildRanInsideCallback
                            .store(
                                false,
                                std::memory_order_release);
                        g_executorBoundaryMainResumedBeforeChild
                            .store(
                                false,
                                std::memory_order_release);
                        runtime.cpu().pc = kMainAddress;
                        setRegU32(
                            runtime.cpu(), 29,
                            0x00300000u);
                        runtime.debugStartEeEventTrace(
                            8u);

                        runtime
                            .startEeExecutionForTesting();
                        result.completed = waitUntil(
                            [&runtime]()
                            {
                                return g_executorBoundaryMainCompleted
                                           .load(
                                               std::memory_order_acquire) &&
                                       g_executorBoundaryChildCompleted
                                           .load(
                                               std::memory_order_acquire);
                            },
                            std::chrono::seconds(2));
                        result.executionQuiescent =
                            waitUntil(
                                [&runtime]()
                                {
                                    return runtime
                                        .eeExecutionQuiescentForTesting();
                                },
                                std::chrono::seconds(2));
                        result.contexts =
                            runtime
                                .eeThreadContextsForTesting();
                        result.memoryCaptured =
                            runtime.debugReadRdram(
                                0u,
                                PS2_RAM_SIZE,
                                result.memory);
                        const PS2Runtime::
                            DebugEeEventTrace eventTrace =
                                runtime
                                    .debugEeEventTraceSnapshot(
                                        true);
                        result.eventCount =
                            eventTrace.totalEntries;
                        for (const auto &entry :
                             eventTrace.entries)
                        {
                            result.eventSources.push_back(
                                entry.source);
                        }
                        result.modeledTick =
                            runtime.currentEeTick().raw();
                        runtime
                            .stopEeExecutionForTesting();

                        result.callbackHits =
                            g_executorBoundaryCallbackHits.load(
                                std::memory_order_acquire);
                        result.callbackRanWhileWaiting =
                            g_executorBoundaryCallbackRanWhileWaiting
                                .load(
                                    std::memory_order_acquire);
                        result.childThreadId =
                            g_executorBoundaryChildThreadId.load(
                                std::memory_order_acquire);
                        result.wakeResult =
                            g_executorBoundaryWakeResult.load(
                                std::memory_order_acquire);
                        result.childWaiting =
                            g_executorBoundaryChildWaiting.load(
                                std::memory_order_acquire);
                        result.childRanInsideCallback =
                            g_executorBoundaryChildRanInsideCallback
                                .load(
                                    std::memory_order_acquire);
                        result.mainResumedBeforeChild =
                            g_executorBoundaryMainResumedBeforeChild
                                .load(
                                    std::memory_order_acquire);
                        result.managedAfterStop =
                            runtime
                                .managedEeExecutionThreadCountForTesting();
                        if (interruptKind ==
                            ExecutorBoundaryInterruptKind::
                                VSync)
                        {
                            ps2_stubs::
                                resetGsSyncVCallbackState(
                                    &runtime);
                        }
                        return result;
                    };

                const auto describe =
                    [](const FixtureResult &result)
                    {
                        return result.backendName +
                               "{memory=" +
                               std::to_string(
                                   result.memoryInitialized) +
                               ",completed=" +
                               std::to_string(
                                   result.completed) +
                               ",hits=" +
                               std::to_string(
                                   result.callbackHits) +
                               ",callback_wait=" +
                               std::to_string(
                                   result.callbackRanWhileWaiting) +
                               ",child_wait=" +
                               std::to_string(
                                   result.childWaiting) +
                               ",child=" +
                               std::to_string(
                                   result.childThreadId) +
                               ",wake=" +
                               std::to_string(
                                   result.wakeResult) +
                               ",main_first=" +
                               std::to_string(
                                   result.mainResumedBeforeChild) +
                               ",child_in_callback=" +
                               std::to_string(
                                   result.childRanInsideCallback) +
                               ",events=" +
                               std::to_string(
                                   result.eventCount) +
                               ",tick=" +
                               std::to_string(
                                   result.modeledTick) +
                               ",managed=" +
                               std::to_string(
                                   result.managedAfterStop) +
                               "}";
                    };
                struct BoundaryExpectation
                {
                    ExecutorBoundaryInterruptKind kind;
                    const char *name;
                    ps2x::timing::EeEventSource source;
                };
                constexpr std::array<
                    BoundaryExpectation,
                    3u>
                    expectations{{
                        {
                            ExecutorBoundaryInterruptKind::
                                VSync,
                            "VSync",
                            ps2x::timing::EeEventSource::
                                VSync,
                        },
                        {
                            ExecutorBoundaryInterruptKind::
                                Dmac,
                            "DMAC completion",
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                        },
                        {
                            ExecutorBoundaryInterruptKind::
                                Counter,
                            "EE counter",
                            ps2x::timing::EeEventSource::
                                EeCounters,
                        },
                    }};

                for (const auto &expectation :
                     expectations)
                {
                    const FixtureResult host =
                        runFixture(
                            expectation.kind,
                            EeExecutionBackendKind::
                                LegacyHostThread);
                    const FixtureResult fiber =
                        runFixture(
                            expectation.kind,
                            EeExecutionBackendKind::
                                LegacyCppFiber);
                    const std::string
                        differentialSummary =
                            describe(host) + " " +
                            describe(fiber);
                    const std::string prefix =
                        std::string(expectation.name) +
                        " boundary";

                    t.IsTrue(
                        host.memoryInitialized &&
                            fiber.memoryInitialized &&
                            host.completed &&
                            fiber.completed &&
                            host.backendName ==
                                "legacy-host-thread" &&
                            fiber.backendName ==
                                "legacy-cpp-fiber",
                        prefix +
                            " should complete under both "
                            "production backends: " +
                            differentialSummary);
                    t.IsTrue(
                        host.callbackHits == 1u &&
                            fiber.callbackHits ==
                                host.callbackHits &&
                            host.callbackRanWhileWaiting &&
                            fiber
                                    .callbackRanWhileWaiting ==
                                host
                                    .callbackRanWhileWaiting &&
                            host.childWaiting &&
                            fiber.childWaiting ==
                                host.childWaiting,
                        prefix +
                            " should publish one deferred "
                            "callback while the main "
                            "continuation remains at its "
                            "boundary: " +
                            differentialSummary);
                    t.IsTrue(
                        host.childThreadId > 1 &&
                            fiber.childThreadId ==
                                host.childThreadId &&
                            host.wakeResult ==
                                host.childThreadId &&
                            fiber.wakeResult ==
                                host.wakeResult,
                        prefix +
                            " raw wake should return the "
                            "same child ID under both "
                            "backends");
                    t.IsTrue(
                        host.mainResumedBeforeChild &&
                            fiber
                                    .mainResumedBeforeChild ==
                                host
                                    .mainResumedBeforeChild &&
                            !host.childRanInsideCallback &&
                            fiber.childRanInsideCallback ==
                                host
                                    .childRanInsideCallback,
                        prefix +
                            " raw wake should make the "
                            "child READY without recursive "
                            "dispatch");
                    t.IsTrue(
                        host.eventSources ==
                                std::vector<
                                    ps2x::timing::
                                        EeEventSource>{
                                    expectation.source} &&
                            fiber.eventSources ==
                                host.eventSources &&
                            host.eventCount ==
                                fiber.eventCount &&
                            host.eventCount == 1u &&
                            host.modeledTick ==
                                fiber.modeledTick &&
                            host.executionQuiescent &&
                            fiber.executionQuiescent &&
                            host.memoryCaptured &&
                            fiber.memoryCaptured &&
                            host.contexts ==
                                fiber.contexts &&
                            host.memory == fiber.memory &&
                            host.managedAfterStop == 0u &&
                            fiber.managedAfterStop == 0u,
                        prefix +
                            " should retain the same event "
                            "trace, final contexts, full "
                            "RDRAM, modeled ticks, and "
                            "teardown");
                }
            });

        tc.Run("event VSync preserves the active phase when SetGsCrt selects PAL", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint64_t kTicksPerEeCycle = 8u;
            constexpr uint64_t kInitialRenderCycles =
                4'493'898u;
            constexpr uint64_t kPalRenderCycles =
                5'435'818u;
            constexpr uint64_t kPalGsBlankCycles =
                56'623u;

            EnsureVSyncScheduled(
                env.rdram.data(), &env.runtime);
            const auto initial =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &initialSlot =
                initial.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                initialSlot.deadlineTick,
                kInitialRenderCycles * kTicksPerEeCycle,
                "VSync should begin with PCSX2's temporary 60 Hz render phase");

            R5900Context crtCtx{};
            setRegU32(crtCtx, 4, 1u);
            setRegU32(crtCtx, 5, 0x03u);
            setRegU32(crtCtx, 6, 1u);
            GsSetCrt(
                env.rdram.data(), &crtCtx,
                &env.runtime);

            const auto configured =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &configuredSlot =
                configured.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                configuredSlot.deadlineTick,
                initialSlot.deadlineTick,
                "SetGsCrt should preserve the already-pending phase endpoint");

            R5900Context timingCtx{};
            timingCtx.cop0_config = 1u << 18u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        initialSlot.deadlineTick));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
            }

            const auto afterStart =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &palGsBlankSlot =
                afterStart.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                palGsBlankSlot.deadlineTick,
                (kInitialRenderCycles +
                 kPalGsBlankCycles) *
                    kTicksPerEeCycle,
                "the next phase should use PAL scanline timing");

            env.runtime.resetEeTiming(&timingCtx);
            const auto reset =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &resetSlot =
                reset.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                resetSlot.deadlineTick,
                kPalRenderCycles * kTicksPerEeCycle,
                "timing reset should retain the configured PAL standard");

            cleanupRuntime(env);
        });

        tc.Run("event VSync normalizes aliases and retains SDTV 480p timing", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint64_t kGsCsrFieldMask = 0x2000ull;
            constexpr uint64_t kTicksPerEeCycle = 8u;
            constexpr uint64_t kRenderCycles = 4'489'024u;
            constexpr uint64_t kGsBlankCycles = 74'973u;
            constexpr uint64_t kBlankCycles = 431'096u;
            constexpr uint64_t kPeriodCycles = 4'920'120u;

            env.runtime.configureEeVSyncVideoMode(
                0x00u, true);
            env.runtime.memory().gs().csr.fetch_and(
                ~kGsCsrFieldMask,
                std::memory_order_relaxed);
            env.runtime.configureEeVSyncVideoMode(
                0x02u, true);
            t.Equals(
                env.runtime.memory().gs().csr.load(
                    std::memory_order_relaxed) &
                    kGsCsrFieldMask,
                0ull,
                "BIOS and libgs NTSC aliases should retain one normalized mode class");

            env.runtime.configureEeVSyncVideoMode(
                0x50u, false);
            t.Equals(
                env.runtime.memory().gs().csr.load(
                    std::memory_order_relaxed) &
                    kGsCsrFieldMask,
                kGsCsrFieldMask,
                "changing to SDTV 480p should initialize FIELD high");

            EnsureVSyncScheduled(
                env.rdram.data(), &env.runtime);
            const auto initial =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &startSlot =
                initial.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                startSlot.deadlineTick,
                kRenderCycles * kTicksPerEeCycle,
                "SDTV 480p should use PCSX2's distinct 59.94 Hz render duration");

            R5900Context timingCtx{};
            timingCtx.cop0_config = 1u << 18u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        startSlot.deadlineTick));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
            }

            const auto afterStart =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &gsBlankSlot =
                afterStart.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                gsBlankSlot.deadlineTick,
                (kRenderCycles + kGsBlankCycles) *
                    kTicksPerEeCycle,
                "SDTV 480p should retain its reference GS-blank offset");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        gsBlankSlot.deadlineTick -
                        env.runtime.currentEeTick().raw()));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
            }
            t.Equals(
                env.runtime.memory().gs().csr.load(
                    std::memory_order_relaxed) &
                    kGsCsrFieldMask,
                kGsCsrFieldMask,
                "a progressive GS-blank should keep FIELD high");

            const auto afterGsBlank =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &endSlot =
                afterGsBlank.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                endSlot.deadlineTick,
                (kRenderCycles + kBlankCycles) *
                    kTicksPerEeCycle,
                "SDTV 480p should retain its reference blank duration");

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &timingCtx);
                timingCtx.advanceEeCycleTicks(
                    static_cast<uint32_t>(
                        endSlot.deadlineTick -
                        env.runtime.currentEeTick().raw()));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &timingCtx);
            }
            const auto afterEnd =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &nextStartSlot =
                afterEnd.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                nextStartSlot.deadlineTick,
                (kRenderCycles + kPeriodCycles) *
                    kTicksPerEeCycle,
                "SDTV 480p should retain its 59.94 Hz field cadence");

            cleanupRuntime(env);
        });

        tc.Run("event VSync wait fast-forwards only emulated deadlines and survives reset", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");
            env.runtime.configureEeVSyncVideoMode(
                0x02u, true);

            constexpr uint64_t kTicksPerEeCycle = 8u;
            constexpr uint64_t kRenderCycles = 4'498'396u;
            constexpr uint64_t kPeriodCycles = 4'920'120u;

            R5900Context ctx{};
            ctx.cop0_config = 1u << 18u;
            const uint64_t tickBefore =
                GetCurrentVSyncTick(&env.runtime);
            const auto hostStart =
                std::chrono::steady_clock::now();
            uint64_t firstTick = 0u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &ctx);
                firstTick = WaitForNextVSyncTick(
                    env.rdram.data(), &env.runtime,
                    &ctx);
            }
            const auto hostElapsed =
                std::chrono::steady_clock::now() -
                hostStart;

            t.Equals(
                firstTick, tickBefore + 1u,
                "the last runnable guest should publish the next VSync");
            t.Equals(
                env.runtime.currentEeTick().raw(),
                kRenderCycles * kTicksPerEeCycle,
                "idle advancement should stop at the exact first VSync deadline");
            t.IsTrue(
                hostElapsed < std::chrono::milliseconds(100),
                "event VSync wait should not consume a host-frame delay");

            uint64_t secondTick = 0u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &ctx);
                secondTick = WaitForNextVSyncTick(
                    env.rdram.data(), &env.runtime,
                    &ctx);
            }
            t.Equals(
                secondTick, firstTick + 1u,
                "the next wait should cross GS-blank and end phases to the next start");
            t.Equals(
                env.runtime.currentEeTick().raw(),
                (kRenderCycles + kPeriodCycles) *
                    kTicksPerEeCycle,
                "successive waits should preserve the fixed emulated field cadence");

            env.runtime.resetEeTiming(&ctx);
            const PS2Runtime::DebugEeScheduler reset =
                env.runtime.debugEeSchedulerSnapshot();
            const auto &resetSlot =
                reset.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VSync)];
            t.Equals(
                env.runtime.currentEeTick().raw(), 0u,
                "timing reset should restore canonical EE time");
            t.IsTrue(
                resetSlot.pending,
                "timing reset should re-arm an enabled VSync source");
            t.Equals(
                resetSlot.deadlineTick,
                kRenderCycles * kTicksPerEeCycle,
                "timing reset should restore the deterministic initial phase");

            uint64_t postResetTick = 0u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &ctx);
                postResetTick = WaitForNextVSyncTick(
                    env.rdram.data(), &env.runtime,
                    &ctx);
            }
            t.Equals(
                postResetTick, secondTick + 1u,
                "reset should preserve the monotonic guest-visible VSync count");
            t.Equals(
                env.runtime.currentEeTick().raw(),
                kRenderCycles * kTicksPerEeCycle,
                "post-reset wait should reach the same emulated deadline");

            cleanupRuntime(env);
        });

        tc.Run("interactive VSync pacing bounds consecutive idle advances", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");
            env.runtime.configureEeVSyncVideoMode(
                0x02u, true);
            env.runtime.setRealtimeVSyncPacingEnabled(
                true);

            R5900Context ctx{};
            ctx.cop0_config = 1u << 18u;
            const uint64_t tickBefore =
                GetCurrentVSyncTick(&env.runtime);
            const auto hostStart =
                std::chrono::steady_clock::now();
            uint64_t secondTick = 0u;
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &ctx);
                (void)WaitForNextVSyncTick(
                    env.rdram.data(), &env.runtime,
                    &ctx);
                secondTick = WaitForNextVSyncTick(
                    env.rdram.data(), &env.runtime,
                    &ctx);
            }
            const auto hostElapsed =
                std::chrono::steady_clock::now() -
                hostStart;

            t.Equals(
                secondTick, tickBefore + 2u,
                "two idle waits should still publish two exact emulated VSync starts");
            t.IsTrue(
                hostElapsed >=
                    std::chrono::milliseconds(20),
                "interactive pacing should retain consecutive VSync waits in host time");
            t.IsTrue(
                hostElapsed <
                    std::chrono::milliseconds(500),
                "interactive pacing should not introduce an unbounded host delay");

            cleanupRuntime(env);
        });

        tc.Run("event VSync wait shares one idle advance and yields to queued guest work", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");
            env.runtime.configureEeVSyncVideoMode(
                0x02u, true);
            EnsureVSyncScheduled(
                env.rdram.data(), &env.runtime);

            constexpr uint64_t kTicksPerEeCycle = 8u;
            constexpr uint64_t kRenderCycles = 4'498'396u;
            const uint64_t observedTick =
                GetCurrentVSyncTick(&env.runtime);

            R5900Context firstWaitCtx{};
            R5900Context secondWaitCtx{};
            uint64_t firstResult = 0u;
            uint64_t secondResult = 0u;
            std::thread firstWaiter(
                [&]()
                {
                    firstResult =
                        env.runtime.waitForNextScheduledVSync(
                            env.rdram.data(),
                            &firstWaitCtx,
                            observedTick);
                });
            std::thread secondWaiter(
                [&]()
                {
                    secondResult =
                        env.runtime.waitForNextScheduledVSync(
                            env.rdram.data(),
                            &secondWaitCtx,
                            observedTick);
                });
            firstWaiter.join();
            secondWaiter.join();

            t.Equals(
                firstResult, observedTick + 1u,
                "the first idle waiter should observe the next shared tick");
            t.Equals(
                secondResult, observedTick + 1u,
                "concurrent idle waiters should share one VSync advance");
            t.Equals(
                env.runtime.currentEeTick().raw(),
                kRenderCycles * kTicksPerEeCycle,
                "two waiters must not advance through two VSync fields");

            const uint64_t secondObservedTick =
                GetCurrentVSyncTick(&env.runtime);
            std::atomic<bool> ordinaryEntered{false};
            std::atomic<bool> allowOrdinaryExit{false};
            std::atomic<bool> thirdWaitDone{false};
            uint64_t thirdResult = 0u;
            R5900Context ownerCtx{};
            R5900Context ordinaryCtx{};
            R5900Context thirdWaitCtx{};
            std::thread ordinaryThread;
            std::thread thirdWaiter;
            {
                PS2Runtime::GuestExecutionScope owner(
                    &env.runtime, &ownerCtx);
                ordinaryThread = std::thread(
                    [&]()
                    {
                        PS2Runtime::GuestExecutionScope guest(
                            &env.runtime, &ordinaryCtx);
                        ordinaryEntered.store(
                            true, std::memory_order_release);
                        while (!allowOrdinaryExit.load(
                            std::memory_order_acquire) &&
                               !env.runtime.isStopRequested())
                        {
                            std::this_thread::yield();
                        }
                    });
                t.IsTrue(
                    waitUntil(
                        [&]()
                        {
                            return env.runtime
                                       .guestExecutionWaiterCountForTesting() >=
                                   1u;
                        },
                        std::chrono::milliseconds(100)),
                    "ordinary guest work should queue behind the owner");

                thirdWaiter = std::thread(
                    [&]()
                    {
                        thirdResult =
                            env.runtime.waitForNextScheduledVSync(
                                env.rdram.data(),
                                &thirdWaitCtx,
                                secondObservedTick);
                        thirdWaitDone.store(
                            true, std::memory_order_release);
                    });
                t.IsTrue(
                    waitUntil(
                        [&]()
                        {
                            return env.runtime
                                       .guestExecutionWaiterCountForTesting() >=
                                   2u;
                        },
                        std::chrono::milliseconds(100)),
                    "the idle arbiter should also queue behind the owner");
            }

            const bool ordinaryReceivedToken =
                waitUntil(
                    [&]()
                    {
                        return ordinaryEntered.load(
                            std::memory_order_acquire);
                    },
                    std::chrono::milliseconds(100));
            t.IsTrue(
                ordinaryReceivedToken,
                "queued ordinary guest work should receive the token first");
            t.IsFalse(
                thirdWaitDone.load(
                    std::memory_order_acquire),
                "VSync must not skip emulated time while ordinary guest work is runnable");
            t.Equals(
                GetCurrentVSyncTick(&env.runtime),
                secondObservedTick,
                "the next VSync should remain pending while ordinary work owns the token");

            allowOrdinaryExit.store(
                true, std::memory_order_release);
            if (ordinaryThread.joinable())
            {
                ordinaryThread.join();
            }
            if (thirdWaiter.joinable())
            {
                thirdWaiter.join();
            }
            t.Equals(
                thirdResult, secondObservedTick + 1u,
                "idle advancement should resume after queued guest work yields");

            cleanupRuntime(env);
        });

        // Regression test for the GS CSR data race: a two-writer word-level
        // lost-update guard. Pre-fix, every CSR update was a plain (non-atomic)
        // 64-bit load-modify-store of the WHOLE word, so two threads that own
        // logically disjoint bits could still clobber each other: thread A's
        // read-modify-write of the word can overwrite thread B's bit with the
        // stale value A loaded before B's update landed.
        //
        // Two racer threads with disjoint bit ownership run concurrently:
        //   - racer A owns SIGNAL (bit 0): sets it via the GIF register path
        //     (GS_REG_SIGNAL) then W1C-clears ONLY bit 0 via the MMIO write path;
        //   - racer B owns FINISH (bit 1): same protocol with GS_REG_FINISH and
        //     a W1C write of only bit 1.
        // Each racer checks only its own bit after each half-op. With the fix
        // (std::atomic CSR, every update a single atomic RMW) each racer is the
        // sole writer of its bit, so its bit deterministically reflects its own
        // last operation: zero anomalies are possible. Pre-fix, the racers'
        // whole-word W1C RMWs constantly interleave and lose each other's
        // set/clear, lighting up the anomaly counters.
        //
        // Why racer-vs-racer instead of racer-vs-vsync: the vsync worker (which
        // motivated the fix) writes CSR only once per ~16.7ms tick, a window far
        // too narrow to hit deterministically in a bounded test. The corrupting
        // mechanism -- a non-atomic whole-word RMW clobbering a concurrently
        // written disjoint bit -- is identical, so guarding it with two
        // high-frequency writers also guards the vsync FIELD interleaving. The
        // real vsync worker still runs throughout (started via the same
        // SetVSyncFlag syscall production uses) and its FIELD (bit 13) toggling
        // is asserted when at least two ticks were observed.
        tc.Run("Disjoint-bit GS CSR writers (SIGNAL vs FINISH vs vsync FIELD) never lose word-level updates", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(env.runtime.memory().initialize(), "runtime memory initialize should succeed");
            t.IsTrue(env.runtime.syncCoreSubsystems(), "runtime subsystems should bind");

            constexpr uint32_t kFlagAddr = 0x1180u;
            constexpr uint32_t kTickAddr = 0x1190u;
            constexpr uint64_t kGsCsrFieldMask = 0x2000ull;
            constexpr uint32_t kCsrAddr = PS2_GS_PRIV_REG_BASE + 0x1000u;
            constexpr uint32_t kIterations = 80000u;

            GS gs;
            gs.init(env.runtime.memory().getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE),
                    &env.runtime.memory().gs());

            // Drive the real vsync worker via the same syscall path production
            // code uses; it runs on its own thread and toggles CSR.FIELD once
            // per tick via updateGsCsrFieldForVSync.
            R5900Context ctx{};
            setRegU32(ctx, 4, kFlagAddr);
            setRegU32(ctx, 5, kTickAddr);
            t.IsTrue(callSyscall(0x73u, env.rdram.data(), &ctx, &env.runtime), "SetVSyncFlag syscall should dispatch");
            const uint64_t tickBefore =
                GetCurrentVSyncTick(&env.runtime);

            std::atomic<uint32_t> setAnomaliesA{0u}, clearAnomaliesA{0u};
            std::atomic<uint32_t> setAnomaliesB{0u}, clearAnomaliesB{0u};
            std::atomic<uint32_t> racersDone{0u};

            // ownBit: the single CSR status bit this racer exclusively owns.
            // Each iteration: raise the bit via the GIF register-write path,
            // verify it reads back set, W1C-clear only that bit via the guest
            // MMIO path, verify it reads back clear. The other racer and the
            // vsync worker never touch this bit, so under atomic RMWs both
            // checks are exact -- any anomaly is a lost word-level update.
            auto racerBody = [&](uint8_t gifReg, uint64_t gifValue, uint64_t ownBit,
                                 std::atomic<uint32_t> &setAnomalies, std::atomic<uint32_t> &clearAnomalies) {
                for (uint32_t i = 0; i < kIterations; ++i)
                {
                    gs.writeRegister(gifReg, gifValue);
                    if ((env.runtime.memory().gs().csr.load() & ownBit) == 0ull)
                    {
                        setAnomalies.fetch_add(1u, std::memory_order_relaxed);
                    }

                    env.runtime.memory().write64(kCsrAddr, ownBit);
                    if ((env.runtime.memory().gs().csr.load() & ownBit) != 0ull)
                    {
                        clearAnomalies.fetch_add(1u, std::memory_order_relaxed);
                    }
                }
                racersDone.fetch_add(1u, std::memory_order_relaxed);
            };

            const uint64_t signalValue = (0xFFFFFFFFull << 32) | 0x11223344ull;
            std::thread racerA(racerBody, GS_REG_SIGNAL, signalValue, 0x1ull,
                               std::ref(setAnomaliesA), std::ref(clearAnomaliesA));
            std::thread racerB(racerBody, GS_REG_FINISH, 0ull, 0x2ull,
                               std::ref(setAnomaliesB), std::ref(clearAnomaliesB));

            // While the racers hammer bits 0..1, watch for CSR.FIELD (bit 13)
            // flips from the vsync worker. Polling ends when both racers finish,
            // so this adds no fixed wall-clock cost.
            const uint64_t initialField = env.runtime.memory().gs().csr.load() & kGsCsrFieldMask;
            bool fieldFlipped = false;
            while (racersDone.load(std::memory_order_relaxed) < 2u)
            {
                if ((env.runtime.memory().gs().csr.load() & kGsCsrFieldMask) != initialField)
                {
                    fieldFlipped = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            racerA.join();
            racerB.join();
            const uint64_t ticksElapsed =
                GetCurrentVSyncTick(&env.runtime) -
                tickBefore;

            t.Equals(setAnomaliesA.load(), 0u, "racer A: SIGNAL set must never be lost to a concurrent whole-word CSR RMW");
            t.Equals(clearAnomaliesA.load(), 0u, "racer A: SIGNAL W1C-clear must never be lost to a concurrent whole-word CSR RMW");
            t.Equals(setAnomaliesB.load(), 0u, "racer B: FINISH set must never be lost to a concurrent whole-word CSR RMW");
            t.Equals(clearAnomaliesB.load(), 0u, "racer B: FINISH W1C-clear must never be lost to a concurrent whole-word CSR RMW");
            t.Equals(env.runtime.memory().gs().csr.load() & 0x3ull, 0x0ull,
                     "final CSR status bits must match both racers' ledgers (last op on each bit was a clear)");
            if (ticksElapsed >= 2u)
            {
                t.IsTrue(fieldFlipped, "VSync worker should toggle GS CSR FIELD while the racers run");
            }

            cleanupRuntime(env);
        });

        tc.Run("interrupt handler registries and masks are isolated per runtime", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv first;
            TestEnv second;
            t.IsTrue(
                first.runtime.memory().initialize(),
                "first runtime memory initialize should succeed");
            t.IsTrue(
                second.runtime.memory().initialize(),
                "second runtime memory initialize should succeed");

            constexpr uint32_t kIntcCause = 7u;
            constexpr uint32_t kDmacCause = 5u;
            constexpr uint32_t kHandlerAddr =
                0x00ABC0C0u;
            first.runtime.registerFunction(
                kHandlerAddr,
                &testInterruptIsolationHandler);
            second.runtime.registerFunction(
                kHandlerAddr,
                &testInterruptIsolationHandler);

            auto addIntc = [&](TestEnv &env)
            {
                R5900Context context{};
                setRegU32(context, 4, kIntcCause);
                setRegU32(context, 5, kHandlerAddr);
                AddIntcHandler(
                    env.rdram.data(),
                    &context,
                    &env.runtime);
                return getRegS32(context, 2);
            };
            auto addDmac = [&](TestEnv &env)
            {
                R5900Context context{};
                setRegU32(context, 4, kDmacCause);
                setRegU32(context, 5, kHandlerAddr);
                AddDmacHandler(
                    env.rdram.data(),
                    &context,
                    &env.runtime);
                return getRegS32(context, 2);
            };

            t.Equals(
                addIntc(first),
                1,
                "first runtime should allocate INTC handler ID 1");
            t.Equals(
                addIntc(second),
                1,
                "second runtime should independently allocate INTC handler ID 1");
            t.Equals(
                addDmac(first),
                1,
                "first runtime should allocate DMAC handler ID 1");
            t.Equals(
                addDmac(second),
                1,
                "second runtime should independently allocate DMAC handler ID 1");

            R5900Context disable{};
            setRegU32(disable, 4, kIntcCause);
            DisableIntc(
                second.rdram.data(),
                &disable,
                &second.runtime);
            dispatchIntcHandlersForCause(
                first.rdram.data(),
                &first.runtime,
                kIntcCause);
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationHitAddr),
                1u,
                "masking one runtime must not suppress another runtime's INTC handler");

            R5900Context enable{};
            setRegU32(enable, 4, kIntcCause);
            EnableIntc(
                first.rdram.data(),
                &enable,
                &first.runtime);
            dispatchDmacHandlersForCause(
                first.rdram.data(),
                &first.runtime,
                kDmacCause);
            const uint32_t beforeSecondStop =
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationHitAddr);

            second.runtime.requestStop();
            dispatchIntcHandlersForCause(
                first.rdram.data(),
                &first.runtime,
                kIntcCause);
            dispatchDmacHandlersForCause(
                first.rdram.data(),
                &first.runtime,
                kDmacCause);
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationHitAddr),
                beforeSecondStop + 2u,
                "stopping one runtime must not erase another runtime's interrupt handlers");

            cleanupRuntime(first);
        });

        tc.Run("VSync publication and GS callbacks are isolated per runtime", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv first;
            TestEnv second;
            ps2_stubs::resetGsSyncVCallbackState(
                &first.runtime);
            ps2_stubs::resetGsSyncVCallbackState(
                &second.runtime);
            t.IsTrue(
                first.runtime.memory().initialize(),
                "first runtime memory initialize should succeed");
            t.IsTrue(
                second.runtime.memory().initialize(),
                "second runtime memory initialize should succeed");

            constexpr uint32_t kFirstFlagAddr = 0x11A0u;
            constexpr uint32_t kFirstTickAddr = 0x11B0u;
            constexpr uint32_t kSecondFlagAddr = 0x11C0u;
            constexpr uint32_t kSecondTickAddr = 0x11D0u;
            auto registerFlag =
                [](TestEnv &env,
                   uint32_t flagAddr,
                   uint32_t tickAddr)
            {
                R5900Context context{};
                setRegU32(context, 4, flagAddr);
                setRegU32(context, 5, tickAddr);
                SetVSyncFlag(
                    env.rdram.data(),
                    &context,
                    &env.runtime);
                return getRegS32(context, 2);
            };
            t.Equals(
                registerFlag(
                    first,
                    kFirstFlagAddr,
                    kFirstTickAddr),
                KE_OK,
                "first runtime should register its VSync flag");
            t.Equals(
                registerFlag(
                    second,
                    kSecondFlagAddr,
                    kSecondTickAddr),
                KE_OK,
                "second runtime should register its VSync flag");

            const uint64_t firstTick =
                PublishVSyncStart(
                    first.rdram.data(),
                    &first.runtime);
            t.Equals(
                firstTick,
                1ull,
                "first runtime should publish VSync tick 1");
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kFirstFlagAddr),
                1u,
                "first publication should consume the first runtime's flag registration");
            t.Equals(
                readGuestU64(
                    first.rdram.data(),
                    kFirstTickAddr),
                1ull,
                "first publication should write the first runtime's tick");
            t.Equals(
                readGuestU32(
                    second.rdram.data(),
                    kSecondFlagAddr),
                0u,
                "first publication must not consume the second runtime's flag registration");

            const uint64_t secondTick =
                PublishVSyncStart(
                    second.rdram.data(),
                    &second.runtime);
            t.Equals(
                secondTick,
                1ull,
                "second runtime should independently publish VSync tick 1");
            t.Equals(
                readGuestU32(
                    second.rdram.data(),
                    kSecondFlagAddr),
                1u,
                "second publication should consume only its own flag registration");
            t.Equals(
                readGuestU64(
                    second.rdram.data(),
                    kSecondTickAddr),
                1ull,
                "second publication should write its independent tick");

            constexpr uint32_t kFirstHandlerAddr =
                0x00ABC100u;
            constexpr uint32_t kSecondHandlerAddr =
                0x00ABC140u;
            first.runtime.registerFunction(
                kFirstHandlerAddr,
                &testInterruptIsolationHandler);
            second.runtime.registerFunction(
                kSecondHandlerAddr,
                &testInterruptIsolationHandler);
            auto registerCallback =
                [](TestEnv &env, uint32_t handler)
            {
                R5900Context context{};
                setRegU32(context, 4, handler);
                setRegU32(context, 28, 0x12340000u);
                setRegU32(context, 29, 0x001FFFE0u);
                ps2_stubs::sceGsSyncVCallback(
                    env.rdram.data(),
                    &context,
                    &env.runtime);
                return getRegU32(&context, 2);
            };

            t.Equals(
                registerCallback(
                    first, kFirstHandlerAddr),
                0u,
                "first runtime should replace an empty GS callback");
            ps2_stubs::dispatchGsSyncVCallback(
                first.rdram.data(),
                &first.runtime,
                firstTick);
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationHitAddr),
                1u,
                "first runtime should dispatch its GS callback");
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationPcAddr),
                kFirstHandlerAddr,
                "first runtime should execute its registered callback address");

            t.Equals(
                registerCallback(
                    second, kSecondHandlerAddr),
                0u,
                "second runtime should independently replace an empty GS callback");
            ps2_stubs::dispatchGsSyncVCallback(
                second.rdram.data(),
                &second.runtime,
                secondTick);
            t.Equals(
                readGuestU32(
                    second.rdram.data(),
                    kInterruptIsolationHitAddr),
                1u,
                "second runtime should dispatch its GS callback");
            t.Equals(
                readGuestU32(
                    second.rdram.data(),
                    kInterruptIsolationPcAddr),
                kSecondHandlerAddr,
                "second runtime should execute its registered callback address");
            const uint32_t secondCallbackStack =
                readGuestU32(
                    second.rdram.data(),
                    kInterruptIsolationSpAddr);
            const uint32_t secondNextStack =
                second.runtime.reserveAsyncCallbackStack(
                    0x4000u, 16u);
            t.IsTrue(
                secondCallbackStack != 0u &&
                    secondNextStack !=
                        secondCallbackStack,
                "the second callback stack should be reserved by the second runtime");

            auto resetGsVideo =
                [](TestEnv &env, uint32_t interlace)
            {
                R5900Context context{};
                setRegU32(context, 4, 0u);
                setRegU32(context, 5, interlace);
                setRegU32(context, 6, 2u);
                setRegU32(context, 7, 1u);
                ps2_stubs::sceGsResetGraph(
                    env.rdram.data(),
                    &context,
                    &env.runtime);
            };
            resetGsVideo(first, 0u);
            resetGsVideo(second, 1u);
            R5900Context firstParameters{};
            ps2_stubs::sceGsGetGParam(
                first.rdram.data(),
                &firstParameters,
                &first.runtime);
            t.Equals(
                static_cast<uint32_t>(
                    first.runtime.memory()
                        .getScratchpad()[0x100u]),
                0u,
                "the second runtime's interlace mode must not replace the first runtime's GS parameters");

            ps2_stubs::dispatchGsSyncVCallback(
                first.rdram.data(),
                &first.runtime,
                firstTick);
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationHitAddr),
                2u,
                "registering the second runtime must not replace the first callback");
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationPcAddr),
                kFirstHandlerAddr,
                "the first runtime should retain its callback address");

            second.runtime.requestStop();
            const uint64_t firstTickAfterSecondStop =
                PublishVSyncStart(
                    first.rdram.data(),
                    &first.runtime);
            t.Equals(
                firstTickAfterSecondStop,
                2ull,
                "stopping the second runtime must not rewind the first runtime's VSync tick");
            ps2_stubs::dispatchGsSyncVCallback(
                first.rdram.data(),
                &first.runtime,
                firstTickAfterSecondStop);
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationHitAddr),
                3u,
                "stopping the second runtime must not clear the first GS callback");
            t.Equals(
                readGuestU32(
                    first.rdram.data(),
                    kInterruptIsolationPcAddr),
                kFirstHandlerAddr,
                "stopping the second runtime must not replace the first callback address");

            cleanupRuntime(first);
            ps2_stubs::resetGsSyncVCallbackState(
                &first.runtime);
        });

        tc.Run("interrupt callback stack belongs to the runtime rather than the host thread", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint32_t kCause = 7u;
            constexpr uint32_t kHandlerAddr =
                0x00ABC0C0u;
            env.runtime.registerFunction(
                kHandlerAddr,
                &testInterruptIsolationHandler);

            R5900Context add{};
            setRegU32(add, 4, kCause);
            setRegU32(add, 5, kHandlerAddr);
            AddIntcHandler(
                env.rdram.data(),
                &add,
                &env.runtime);
            t.Equals(
                getRegS32(add, 2),
                1,
                "fixture should allocate INTC handler ID 1");

            auto dispatch = [&]()
            {
                dispatchIntcHandlersForCause(
                    env.rdram.data(),
                    &env.runtime,
                    kCause);
            };
            std::thread firstDispatch(dispatch);
            firstDispatch.join();
            const uint32_t firstStack =
                readGuestU32(
                    env.rdram.data(),
                    kInterruptIsolationSpAddr);

            std::thread secondDispatch(dispatch);
            secondDispatch.join();
            const uint32_t secondStack =
                readGuestU32(
                    env.rdram.data(),
                    kInterruptIsolationSpAddr);

            t.IsTrue(
                firstStack != 0u &&
                    (firstStack & 0xFu) == 0u,
                "interrupt callback stack should be nonzero and 16-byte aligned");
            t.Equals(
                secondStack,
                firstStack,
                "changing host threads must reuse the runtime-owned callback stack");

            const uint32_t beforeConcurrent =
                readGuestU32(
                    env.rdram.data(),
                    kInterruptIsolationHitAddr);
            g_interruptIsolationBlockEntered.store(
                false, std::memory_order_relaxed);
            g_interruptIsolationBlockRelease.store(
                false, std::memory_order_relaxed);
            g_interruptIsolationBlockNext.store(
                true, std::memory_order_release);
            std::thread blockedDispatch(dispatch);
            const bool entered = waitUntil(
                []()
                {
                    return g_interruptIsolationBlockEntered.load(
                        std::memory_order_acquire);
                },
                std::chrono::seconds(2));
            t.IsTrue(
                entered,
                "first concurrent callback should enter its bounded gate");

            std::thread waitingDispatch(dispatch);
            const bool waiting = waitUntil(
                [&]()
                {
                    return env.runtime
                               .guestExecutionWaiterCountForTesting() !=
                           0u;
                },
                std::chrono::seconds(2));
            g_interruptIsolationBlockRelease.store(
                true, std::memory_order_release);
            blockedDispatch.join();
            waitingDispatch.join();
            t.IsTrue(
                waiting,
                "second callback should wait behind the active guest boundary");
            t.Equals(
                readGuestU32(
                    env.rdram.data(),
                    kInterruptIsolationHitAddr),
                beforeConcurrent + 2u,
                "concurrent host delivery should complete both callbacks");

            cleanupRuntime(env);
        });

        tc.Run("INTC VBLANK handlers respect EnableIntc and DisableIntc masks", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            g_vblankStartHits.store(0u, std::memory_order_relaxed);
            g_vblankEndHits.store(0u, std::memory_order_relaxed);
            g_lastIntcArg.store(0u, std::memory_order_relaxed);

            constexpr uint32_t kFlagAddr = 0x1100u;
            constexpr uint32_t kTickAddr = 0x1110u;
            constexpr uint32_t kHandlerAddr = 0x00ABC100u;

            env.runtime.registerFunction(kHandlerAddr, &testIntcHandler);

            R5900Context addStart{};
            setRegU32(addStart, 4, 2u); // VBLANK start
            setRegU32(addStart, 5, kHandlerAddr);
            setRegU32(addStart, 6, 0u);
            setRegU32(addStart, 7, 0xCAFE0002u);
            setRegU32(addStart, 28, 0x12340000u);
            setRegU32(addStart, 29, 0x001FFFE0u);
            t.IsTrue(callSyscall(0x10u, env.rdram.data(), &addStart, &env.runtime), "AddIntcHandler syscall should dispatch");
            t.IsTrue(getRegS32(addStart, 2) > 0, "AddIntcHandler for cause 2 should return handler id");

            R5900Context addEnd{};
            setRegU32(addEnd, 4, 3u); // VBLANK end
            setRegU32(addEnd, 5, kHandlerAddr);
            setRegU32(addEnd, 6, 0u);
            setRegU32(addEnd, 7, 0xCAFE0003u);
            setRegU32(addEnd, 28, 0x12340000u);
            setRegU32(addEnd, 29, 0x001FFFE0u);
            t.IsTrue(callSyscall(0x10u, env.rdram.data(), &addEnd, &env.runtime), "AddIntcHandler syscall should dispatch");
            t.IsTrue(getRegS32(addEnd, 2) > 0, "AddIntcHandler for cause 3 should return handler id");

            R5900Context vsyncCtx{};
            setRegU32(vsyncCtx, 4, kFlagAddr);
            setRegU32(vsyncCtx, 5, kTickAddr);
            t.IsTrue(callSyscall(0x73u, env.rdram.data(), &vsyncCtx, &env.runtime), "SetVSyncFlag syscall should dispatch");
            t.Equals(getRegS32(vsyncCtx, 2), KE_OK, "SetVSyncFlag should succeed");

            R5900Context timingCtx{};
            t.IsTrue(
                serviceNextScheduledBoundary(env, timingCtx) &&
                    serviceNextScheduledBoundary(env, timingCtx) &&
                    serviceNextScheduledBoundary(env, timingCtx),
                "one complete scheduled VBlank should service all three phases");
            t.Equals(
                g_vblankStartHits.load(std::memory_order_relaxed),
                1u,
                "VBLANK start handler should fire while cause 2 is enabled");
            t.Equals(
                g_vblankEndHits.load(std::memory_order_relaxed),
                1u,
                "VBLANK end handler should fire while cause 3 is enabled");

            R5900Context disableStart{};
            setRegU32(disableStart, 4, 2u);
            t.IsTrue(callSyscall(0x15u, env.rdram.data(), &disableStart, &env.runtime), "DisableIntc syscall should dispatch");
            t.Equals(getRegS32(disableStart, 2), KE_OK, "DisableIntc should return KE_OK");
            t.IsTrue(
                (env.runtime.memory().intcMask() &
                 (1u << 2u)) == 0u,
                "DisableIntc should clear the matching hardware INTC mask bit");

            const uint32_t startAfterDisable = g_vblankStartHits.load(std::memory_order_relaxed);
            const uint32_t endAfterDisable = g_vblankEndHits.load(std::memory_order_relaxed);

            t.IsTrue(
                serviceNextScheduledBoundary(env, timingCtx) &&
                    serviceNextScheduledBoundary(env, timingCtx) &&
                    serviceNextScheduledBoundary(env, timingCtx),
                "a second scheduled VBlank should service all three phases");
            const uint32_t startLater = g_vblankStartHits.load(std::memory_order_relaxed);
            const uint32_t endLater = g_vblankEndHits.load(std::memory_order_relaxed);

            t.Equals(startLater, startAfterDisable, "cause 2 handler count should stop increasing while cause 2 is disabled");
            t.IsTrue(endLater > endAfterDisable, "cause 3 handler should keep firing while still enabled");

            R5900Context enableStart{};
            setRegU32(enableStart, 4, 2u);
            t.IsTrue(callSyscall(0x14u, env.rdram.data(), &enableStart, &env.runtime), "EnableIntc syscall should dispatch");
            t.Equals(getRegS32(enableStart, 2), KE_OK, "EnableIntc should return KE_OK");
            t.IsTrue(
                (env.runtime.memory().intcMask() &
                 (1u << 2u)) != 0u,
                "EnableIntc should set the matching hardware INTC mask bit");

            t.IsTrue(
                serviceNextScheduledBoundary(env, timingCtx),
                "the next scheduled start phase should service");
            t.IsTrue(
                g_vblankStartHits.load(std::memory_order_relaxed) >
                    startLater,
                "cause 2 handler should resume after re-enable");

            const uint32_t lastArg = g_lastIntcArg.load(std::memory_order_relaxed);
            t.IsTrue(lastArg == 0xCAFE0002u || lastArg == 0xCAFE0003u,
                     "handler should receive configured argument value");

            cleanupRuntime(env);
        });

        tc.Run("sceDmaSend preserves configured VIF tag transfer", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(env.runtime.memory().initialize(), "runtime memory initialize should succeed");
            t.IsTrue(env.runtime.syncCoreSubsystems(), "runtime subsystems should bind");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag = 0x00027F00u;
            uint8_t *rdram = env.runtime.memory().getRDRAM();
            writeDmaTag(rdram, kTag, makeDmaTag(0u, 7u, 0u, false)); // END
            writeGuestU32(rdram, kTag + 0x0Cu, 0x04000055u); // ITOP 0x55 in tag high data.

            R5900Context sendCtx{};
            setRegU32(sendCtx, 4, kVif1Ch);
            setRegU32(sendCtx, 5, kTag);

            t.IsTrue(env.runtime.memory().writeIORegister(kVif1Ch + 0x00u, 0u),
                     "VIF1 channel should accept TTE clear");
            ps2_stubs::sceDmaSend(rdram, &sendCtx, &env.runtime);
            R5900Context timingCtx{};
            t.IsTrue(
                serviceAllScheduledBoundaries(env, timingCtx),
                "TTE-clear VIF1 chain should reach its scheduled completion");
            t.Equals(env.runtime.memory().vif1_regs.itops, 0u,
                     "sceDmaSend must not infer TTE from the VIF channel");

            t.IsTrue(env.runtime.memory().writeIORegister(kVif1Ch + 0x00u, 0x40u),
                     "VIF1 channel should accept TTE set");
            ps2_stubs::sceDmaSend(rdram, &sendCtx, &env.runtime);
            t.IsTrue(
                serviceAllScheduledBoundaries(env, timingCtx),
                "TTE-set VIF1 chain should reach its scheduled completion");
            t.Equals(env.runtime.memory().vif1_regs.itops, 0x55u,
                     "sceDmaSend should preserve configured TTE");

            cleanupRuntime(env);
        });

        tc.Run("sceDmaSend defers VIF1 DMAC completion until guest state is published", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(env.runtime.memory().initialize(), "runtime memory initialize should succeed");
            t.IsTrue(env.runtime.syncCoreSubsystems(), "runtime subsystems should bind");

            constexpr uint32_t kHandlerAddr = 0x00ABD100u;
            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag0 = 0x00028000u;
            constexpr uint32_t kTag1 = kTag0 + 0x20u;
            constexpr uint32_t kPublishedAddr = 0x00028100u;
            constexpr uint32_t kPublishedValue = 0xA55A1234u;

            uint8_t *rdram = env.runtime.memory().getRDRAM();
            writeDmaTag(rdram, kTag0, makeDmaTag(1u, 1u, 0u, false)); // CNT
            const uint32_t itopCmd = 0x04000066u;
            writeGuestU32(rdram, kTag0 + 0x0Cu, itopCmd);
            writeGuestU64(rdram, kTag0 + 0x10u, 0u);
            writeGuestU64(rdram, kTag0 + 0x18u, 0u);
            writeDmaTag(rdram, kTag1, makeDmaTag(0u, 7u, 0u, false)); // END

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            g_dmacSendLastCause.store(0u, std::memory_order_relaxed);
            g_dmacSendLastChcr.store(0u, std::memory_order_relaxed);
            g_dmacPublishedAddress.store(kPublishedAddr, std::memory_order_relaxed);
            g_dmacObservedPublishedValue.store(0u, std::memory_order_relaxed);
            env.runtime.registerFunction(kHandlerAddr, &testDmacSendHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 1u);
            setRegU32(addCtx, 5, kHandlerAddr);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, 0u);
            ps2_syscalls::AddDmacHandler(rdram, &addCtx, &env.runtime);
            t.IsTrue(getRegS32(addCtx, 2) > 0, "AddDmacHandler should register VIF1 handler");

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 1u);
            ps2_syscalls::EnableDmac(rdram, &enableCtx, &env.runtime);
            t.Equals(getRegS32(enableCtx, 2), KE_OK, "EnableDmac should enable VIF1 cause");

            t.IsTrue(env.runtime.memory().writeIORegister(kVif1Ch + 0x00u, 0x40u),
                     "VIF1 channel should accept preconfigured TTE");

            R5900Context sendCtx{};
            setRegU32(sendCtx, 4, kVif1Ch);
            setRegU32(sendCtx, 5, kTag0);
            R5900Context timingCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(&env.runtime);
                ps2_stubs::sceDmaSend(rdram, &sendCtx, &env.runtime);
                t.IsTrue(
                    serviceAllScheduledBoundaries(
                        env, timingCtx),
                    "scheduled VIF1 work should complete before the outer safe point");
                t.Equals(g_dmacSendHits.load(std::memory_order_relaxed), 0u,
                         "sceDmaSend must not re-enter the guest DMAC handler before returning");
                writeGuestU32(rdram, kPublishedAddr, kPublishedValue);
            }

            t.Equals(getRegS32(sendCtx, 2), 0, "sceDmaSend should succeed");
            t.Equals(env.runtime.memory().vif1_regs.itops, 0x66u,
                     "sceDmaSend should preserve VIF tag transfer for source chains");
            t.Equals(g_dmacSendHits.load(std::memory_order_relaxed), 1u, "the outer guest safe point should dispatch the VIF1 DMAC handler");
            t.Equals(g_dmacSendLastCause.load(std::memory_order_relaxed), 1u, "DMAC handler should observe VIF1 cause");
            t.Equals(g_dmacSendLastChcr.load(std::memory_order_relaxed) & 0x100u, 0u, "handler should see VIF1 STR cleared");
            t.Equals(g_dmacSendLastChcr.load(std::memory_order_relaxed) & 0x70000000u, 0x70000000u, "handler should see the latched END tag id");
            t.Equals(g_dmacObservedPublishedValue.load(std::memory_order_relaxed), kPublishedValue,
                     "DMAC handler should observe state published after sceDmaSend returned");

            cleanupRuntime(env);
        });


        tc.Run("MMIO VIF1 completion waits for the outer guest safe point", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(env.runtime.memory().initialize(), "runtime memory initialize should succeed");
            t.IsTrue(env.runtime.syncCoreSubsystems(), "runtime subsystems should bind");

            constexpr uint32_t kHandlerAddr = 0x00ABD180u;
            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag0 = 0x00028200u;
            constexpr uint32_t kTag1 = kTag0 + 0x20u;

            uint8_t *rdram = env.runtime.memory().getRDRAM();
            writeDmaTag(rdram, kTag0, makeDmaTag(1u, 1u, 0u, false)); // CNT
            writeGuestU64(rdram, kTag0 + 0x10u, 0u);
            writeGuestU64(rdram, kTag0 + 0x18u, 0u);
            writeDmaTag(rdram, kTag1, makeDmaTag(0u, 7u, 0u, false)); // END

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            g_dmacSendLastCause.store(0u, std::memory_order_relaxed);
            g_dmacSendLastChcr.store(0u, std::memory_order_relaxed);
            env.runtime.registerFunction(kHandlerAddr, &testDmacSendHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 1u);
            setRegU32(addCtx, 5, kHandlerAddr);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, 0u);
            ps2_syscalls::AddDmacHandler(rdram, &addCtx, &env.runtime);
            t.IsTrue(getRegS32(addCtx, 2) > 0, "AddDmacHandler should register VIF1 handler");

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 1u);
            ps2_syscalls::EnableDmac(rdram, &enableCtx, &env.runtime);
            t.Equals(getRegS32(enableCtx, 2), KE_OK, "EnableDmac should enable VIF1 cause");

            R5900Context storeCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(&env.runtime);
                env.runtime.Store32(rdram, &storeCtx, kVif1Ch + 0x30u, kTag0);
                env.runtime.Store32(rdram, &storeCtx, kVif1Ch + 0x00u, 0x185u);
                t.IsTrue(
                    serviceAllScheduledBoundaries(
                        env, storeCtx),
                    "MMIO VIF1 work should complete before the outer safe point");
                t.Equals(g_dmacSendHits.load(std::memory_order_relaxed), 0u,
                         "CHCR write must not synchronously re-enter the guest DMAC handler");
            }

            t.Equals(g_dmacSendHits.load(std::memory_order_relaxed), 1u, "safe point should dispatch the VIF1 DMAC handler");
            t.Equals(g_dmacSendLastCause.load(std::memory_order_relaxed), 1u, "DMAC handler should observe VIF1 cause");
            t.Equals(g_dmacSendLastChcr.load(std::memory_order_relaxed) & 0x100u, 0u, "handler should see VIF1 STR cleared");
            t.Equals(g_dmacSendLastChcr.load(std::memory_order_relaxed) & 0x70000000u, 0x70000000u, "handler should see the latched END tag id");

            cleanupRuntime(env);
        });

        tc.Run("DMAC handler resubmission remains pending for the next safe point", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint32_t kHandlerAddr = 0x00ABD1A0u;
            uint8_t *const rdram =
                env.runtime.memory().getRDRAM();

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            g_dmacHandlerResubmissions.store(
                1u, std::memory_order_relaxed);
            env.runtime.registerFunction(
                kHandlerAddr, &testDmacResubmittingHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 1u);
            setRegU32(addCtx, 5, kHandlerAddr);
            ps2_syscalls::AddDmacHandler(
                rdram, &addCtx, &env.runtime);

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 1u);
            ps2_syscalls::EnableDmac(
                rdram, &enableCtx, &env.runtime);

            R5900Context eventCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &eventCtx);
                const DmacTransferToken first =
                    env.runtime.memory().beginDmacTransfer(
                        DmacChannel::Vif1);
                t.IsTrue(
                    env.runtime.memory().requestDmacCompletion(first),
                    "initial VIF1 completion should queue");
                env.runtime.serviceEeEventsAtBlockBoundary(
                    rdram, &eventCtx);
            }

            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                1u,
                "the first safe point should deliver the original completion");

            {
                PS2Runtime::GuestExecutionScope nextSafePoint(
                    &env.runtime, &eventCtx);
            }
            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                2u,
                "a completion published by the handler must survive its active drain");

            cleanupRuntime(env);
        });

        tc.Run("native GIF DMA MMIO kick defers its completed DMAC handler", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(env.runtime.memory().initialize(), "runtime memory initialize should succeed");
            t.IsTrue(env.runtime.syncCoreSubsystems(), "runtime subsystems should bind");

            constexpr uint32_t kHandlerAddr = 0x00ABD1C0u;
            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kDPcr = 0x1000E020u;
            constexpr uint32_t kTag0 = 0x00028400u;

            uint8_t *rdram = env.runtime.memory().getRDRAM();
            writeDmaTag(rdram, kTag0, makeDmaTag(1u, 7u, 0u, false)); // END
            writeGuestU64(rdram, kTag0 + 0x10u, 0x1122334455667788ull);
            writeGuestU64(rdram, kTag0 + 0x18u, 0x99AABBCCDDEEFF00ull);

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            g_dmacSendLastCause.store(0u, std::memory_order_relaxed);
            g_dmacSendLastChcr.store(0u, std::memory_order_relaxed);
            env.runtime.registerFunction(kHandlerAddr, &testDmacSendHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 2u);
            setRegU32(addCtx, 5, kHandlerAddr);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, 0u);
            ps2_syscalls::AddDmacHandler(rdram, &addCtx, &env.runtime);
            t.IsTrue(getRegS32(addCtx, 2) > 0, "AddDmacHandler should register GIF handler");

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 2u);
            ps2_syscalls::EnableDmac(rdram, &enableCtx, &env.runtime);
            t.Equals(getRegS32(enableCtx, 2), KE_OK, "EnableDmac should enable GIF cause");

            R5900Context kickCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(&env.runtime);
                env.runtime.kickGifDmaChainFromMMIO(rdram, &kickCtx, 4u, 4u, kTag0, 0x105u);
                t.IsTrue(
                    serviceAllScheduledBoundaries(
                        env, kickCtx),
                    "native GIF work should complete before the outer safe point");
                t.Equals(g_dmacSendHits.load(std::memory_order_relaxed), 0u,
                         "native GIF kick must not synchronously re-enter the guest DMAC handler");
            }

            t.Equals(env.runtime.memory().readIORegister(kDPcr), 4u, "native GIF kick should preserve D_PCR write");
            t.IsTrue((env.runtime.memory().readIORegister(kDStat) & (1u << 2)) != 0u,
                     "native GIF kick should raise D_STAT GIF completion status");
            t.Equals(g_dmacSendHits.load(std::memory_order_relaxed), 1u,
                     "safe point should dispatch the GIF DMAC handler");
            t.Equals(g_dmacSendLastCause.load(std::memory_order_relaxed), 2u,
                     "DMAC handler should observe GIF cause");
            t.Equals(g_dmacSendLastChcr.load(std::memory_order_relaxed) & 0x100u, 0u,
                     "handler should see GIF STR cleared");
            t.Equals(g_dmacSendLastChcr.load(std::memory_order_relaxed) & 0x70000000u, 0x70000000u,
                     "handler should see the latched END tag id");

            cleanupRuntime(env);
        });

        tc.Run("event service publishes DMAC state before safe handler delivery", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint32_t kHandlerAddr = 0x00ABD200u;
            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kDStat = 0x1000E010u;
            uint8_t *rdram =
                env.runtime.memory().getRDRAM();

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            g_dmacSendLastCause.store(0u, std::memory_order_relaxed);
            g_dmacSendLastChcr.store(0u, std::memory_order_relaxed);
            env.runtime.registerFunction(
                kHandlerAddr, &testDmacSendHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 1u);
            setRegU32(addCtx, 5, kHandlerAddr);
            ps2_syscalls::AddDmacHandler(
                rdram, &addCtx, &env.runtime);

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 1u);
            ps2_syscalls::EnableDmac(
                rdram, &enableCtx, &env.runtime);

            env.runtime.debugStartEeEventTrace(4u);
            R5900Context eventCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &eventCtx);
                t.IsTrue(
                    env.runtime.memory().writeIORegister(
                        kVif1Ch + 0x20u, 3u),
                    "VIF1 QWC setup should succeed");
                const DmacTransferToken transfer =
                    env.runtime.memory().beginDmacTransfer(
                        DmacChannel::Vif1);
                t.IsTrue(
                    env.runtime.memory().requestDmacCompletion(
                        transfer),
                    "device work should request a completion event");

                t.IsTrue(
                    (env.runtime.memory().readIORegister(
                         kVif1Ch + 0x00u) &
                     0x100u) != 0u,
                    "VIF1 should remain busy before event service");
                t.IsTrue(
                    (env.runtime.memory().readIORegister(
                         kDStat) &
                     (1u << 1u)) == 0u,
                    "D_STAT should remain clear before event service");

                env.runtime.serviceEeEventsAtBlockBoundary(
                    rdram, &eventCtx);

                t.IsTrue(
                    (env.runtime.memory().readIORegister(
                         kVif1Ch + 0x00u) &
                     0x100u) == 0u,
                    "event service should clear VIF1 busy state");
                t.IsTrue(
                    (env.runtime.memory().readIORegister(
                         kDStat) &
                     (1u << 1u)) != 0u,
                    "event service should publish D_STAT");
                t.Equals(
                    g_dmacSendHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "event service must not recursively enter guest handlers");
                t.IsTrue(
                    env.runtime.guestPreemptionRequestedForTesting(),
                    "an unmasked event should request cooperative preemption");
            }

            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                1u,
                "the outer safe boundary should deliver the handler");
            t.Equals(
                g_dmacSendLastCause.load(
                    std::memory_order_relaxed),
                1u,
                "the handler should receive the VIF1 cause");
            {
                PS2Runtime::GuestExecutionScope nextSafeBoundary(
                    &env.runtime, &eventCtx);
            }
            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                1u,
                "later safe boundaries must not redeliver the completion");

            const PS2Runtime::DebugEeEventTrace trace =
                env.runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(), static_cast<size_t>(1u),
                "the completion should produce one scheduler trace entry");
            if (!trace.entries.empty())
            {
                t.IsTrue(
                    trace.entries[0].source ==
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    "the event trace should identify DMAC completion");
            }
            cleanupRuntime(env);
        });

        tc.Run("masked DMAC completion delivers after later enable", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint32_t kHandlerAddr = 0x00ABD240u;
            constexpr uint32_t kDStat = 0x1000E010u;
            uint8_t *rdram =
                env.runtime.memory().getRDRAM();

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            env.runtime.registerFunction(
                kHandlerAddr, &testDmacSendHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 1u);
            setRegU32(addCtx, 5, kHandlerAddr);
            ps2_syscalls::AddDmacHandler(
                rdram, &addCtx, &env.runtime);

            R5900Context disableCtx{};
            setRegU32(disableCtx, 4, 1u);
            ps2_syscalls::DisableDmac(
                rdram, &disableCtx, &env.runtime);

            R5900Context eventCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &eventCtx);
                const DmacTransferToken transfer =
                    env.runtime.memory().beginDmacTransfer(
                        DmacChannel::Vif1);
                t.IsTrue(
                    env.runtime.memory().requestDmacCompletion(
                        transfer),
                    "masked VIF1 completion should queue");
                env.runtime.serviceEeEventsAtBlockBoundary(
                    rdram, &eventCtx);
                t.IsFalse(
                    env.runtime.guestPreemptionRequestedForTesting(),
                    "a masked completion should not request preemption");
            }

            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                0u,
                "masked completion should remain pending");
            t.IsTrue(
                (env.runtime.memory().readIORegister(kDStat) &
                 (1u << 1u)) != 0u,
                "masked completion should still latch D_STAT");

            const uint64_t maskedDrainPasses =
                env.runtime.dmacInterruptDrainPassesForTesting();
            for (uint32_t index = 0u; index < 8u; ++index)
            {
                PS2Runtime::GuestExecutionScope unchangedSafeBoundary(
                    &env.runtime, &eventCtx);
            }
            t.Equals(
                env.runtime.dmacInterruptDrainPassesForTesting(),
                maskedDrainPasses,
                "unchanged safe boundaries should not rescan a retained masked completion");

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 1u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &enableCtx);
                ps2_syscalls::EnableDmac(
                    rdram, &enableCtx, &env.runtime);
                t.Equals(
                    g_dmacSendHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "enable should not enter a handler recursively");
                t.IsTrue(
                    env.runtime.guestPreemptionRequestedForTesting(),
                    "unmasking a latched cause should request preemption");
            }
            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                1u,
                "the next safe boundary should deliver the unmasked cause");
            t.Equals(
                env.runtime.dmacInterruptDrainPassesForTesting(),
                maskedDrainPasses + 1u,
                "unmasking should schedule exactly one new delivery scan");
            cleanupRuntime(env);
        });

        tc.Run("repeated masked DMAC completions coalesce per latched cause", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint32_t kHandlerAddr = 0x00ABD250u;
            constexpr uint32_t kCompletionCount = 256u;
            uint8_t *const rdram =
                env.runtime.memory().getRDRAM();

            g_dmacSendHits.store(0u, std::memory_order_relaxed);
            g_dmacPublishedAddress.store(
                0u, std::memory_order_relaxed);
            env.runtime.registerFunction(
                kHandlerAddr, &testDmacSendHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 1u);
            setRegU32(addCtx, 5, kHandlerAddr);
            ps2_syscalls::AddDmacHandler(
                rdram, &addCtx, &env.runtime);

            R5900Context disableCtx{};
            setRegU32(disableCtx, 4, 1u);
            ps2_syscalls::DisableDmac(
                rdram, &disableCtx, &env.runtime);

            R5900Context eventCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &eventCtx);
                for (uint32_t index = 0u;
                     index < kCompletionCount;
                     ++index)
                {
                    const DmacTransferToken transfer =
                        env.runtime.memory().beginDmacTransfer(
                            DmacChannel::Vif1);
                    t.IsTrue(
                        env.runtime.memory().requestDmacCompletion(
                            transfer),
                        "masked VIF1 completion should queue");
                    env.runtime.serviceEeEventsAtBlockBoundary(
                        rdram, &eventCtx);
                }
            }

            t.Equals(
                env.runtime.pendingDmacInterruptCountForTesting(),
                static_cast<size_t>(1u),
                "one latched channel cause should retain one delivery record");
            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                0u,
                "masked duplicate completions must not enter the handler");

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 1u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &enableCtx);
                ps2_syscalls::EnableDmac(
                    rdram, &enableCtx, &env.runtime);
            }

            t.Equals(
                g_dmacSendHits.load(std::memory_order_relaxed),
                1u,
                "unmasking one latched cause should deliver one handler call");
            t.Equals(
                env.runtime.pendingDmacInterruptCountForTesting(),
                static_cast<size_t>(0u),
                "delivering the coalesced cause should clear deferred state");
            cleanupRuntime(env);
        });

        tc.Run("masked EE counter cause publishes before deferred handler delivery", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;
            t.IsTrue(
                env.runtime.memory().initialize(),
                "runtime memory initialize should succeed");

            constexpr uint32_t kHandlerAddr = 0x00ABD260u;
            constexpr uint32_t kCount = 0x10000000u;
            constexpr uint32_t kMode = 0x10000010u;
            constexpr uint32_t kTarget = 0x10000020u;
            constexpr uint32_t kIntcStat = 0x1000F000u;
            uint8_t *const rdram =
                env.runtime.memory().getRDRAM();

            g_eeCounterHits.store(
                0u, std::memory_order_relaxed);
            g_eeCounterLastCause.store(
                0u, std::memory_order_relaxed);
            g_eeCounterModeAtHandler.store(
                0u, std::memory_order_relaxed);
            g_eeCounterIntcAtHandler.store(
                0u, std::memory_order_relaxed);
            env.runtime.registerFunction(
                kHandlerAddr, &testEeCounterHandler);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 9u);
            setRegU32(addCtx, 5, kHandlerAddr);
            ps2_syscalls::AddIntcHandler(
                rdram, &addCtx, &env.runtime);
            t.IsTrue(
                getRegS32(addCtx, 2) > 0,
                "counter0 handler should register");

            R5900Context disableCtx{};
            setRegU32(disableCtx, 4, 9u);
            ps2_syscalls::DisableIntc(
                rdram, &disableCtx, &env.runtime);

            R5900Context eventCtx{};
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &eventCtx);
                env.runtime.Store32(
                    rdram, &eventCtx, kTarget, 3u);
                env.runtime.Store32(
                    rdram, &eventCtx, kCount, 0u);
                env.runtime.Store32(
                    rdram, &eventCtx, kMode, 0x1c0u);
                eventCtx.advanceEeCycleTicks(48u);
                env.runtime.serviceEeEventsAtBlockBoundary(
                    rdram, &eventCtx);

                t.IsTrue(
                    (env.runtime.Load32(
                         rdram, &eventCtx, kMode) &
                     0x400u) != 0u,
                    "counter target state should publish at event service");
                t.IsTrue(
                    (env.runtime.Load32(
                         rdram, &eventCtx, kIntcStat) &
                     (1u << 9u)) != 0u,
                    "INTC status should publish before handler delivery");
                t.Equals(
                    g_eeCounterHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "masked service must not enter the handler");
                t.IsFalse(
                    env.runtime.guestPreemptionRequestedForTesting(),
                    "masked timer cause should not request preemption");
            }
            t.Equals(
                g_eeCounterHits.load(
                    std::memory_order_relaxed),
                0u,
                "masked timer cause should remain pending");

            R5900Context enableCtx{};
            setRegU32(enableCtx, 4, 9u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &enableCtx);
                ps2_syscalls::EnableIntc(
                    rdram, &enableCtx, &env.runtime);
                t.Equals(
                    g_eeCounterHits.load(
                        std::memory_order_relaxed),
                    0u,
                    "unmask must not recursively enter the handler");
                t.IsTrue(
                    env.runtime.guestPreemptionRequestedForTesting(),
                    "unmasking a latched timer cause should request preemption");
            }

            t.Equals(
                g_eeCounterHits.load(
                    std::memory_order_relaxed),
                1u,
                "the outer safe boundary should deliver the timer handler");
            t.Equals(
                g_eeCounterLastCause.load(
                    std::memory_order_relaxed),
                9u,
                "handler should receive counter0 cause 9");
            t.IsTrue(
                (g_eeCounterModeAtHandler.load(
                     std::memory_order_relaxed) &
                 0x400u) != 0u,
                "handler should observe the latched target flag");
            t.IsTrue(
                (g_eeCounterIntcAtHandler.load(
                     std::memory_order_relaxed) &
                 (1u << 9u)) != 0u,
                "handler should observe INTC status after device publication");
            cleanupRuntime(env);
        });

        tc.Run("interrupt-safe EE syscall ids follow the retail sign table", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kEventParamAddr = 0x1200u;
            constexpr uint32_t kStatusAddr = 0x1210u;

            const uint32_t eventParam[3] = {
                0u,
                0u,
                0u
            };
            std::memcpy(env.rdram.data() + kEventParamAddr, eventParam, sizeof(eventParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kEventParamAddr);
            CreateEventFlag(env.rdram.data(), &createCtx, &env.runtime);
            const int32_t eid = getRegS32(createCtx, 2);
            t.IsTrue(eid > 0, "CreateEventFlag should return a valid event id");

            R5900Context disableIntcCtx{};
            setRegU32(disableIntcCtx, 4, 2u);
            t.IsTrue(callSyscall(static_cast<uint32_t>(-0x1B), env.rdram.data(), &disableIntcCtx, &env.runtime),
                     "negative iDisableIntc syscall id should dispatch");
            t.Equals(getRegS32(disableIntcCtx, 2), KE_OK, "negative iDisableIntc should return KE_OK");

            R5900Context enableIntcCtx{};
            setRegU32(enableIntcCtx, 4, 2u);
            t.IsTrue(callSyscall(static_cast<uint32_t>(-0x1A), env.rdram.data(), &enableIntcCtx, &env.runtime),
                     "negative iEnableIntc syscall id should dispatch");
            t.Equals(getRegS32(enableIntcCtx, 2), KE_OK, "negative iEnableIntc should return KE_OK");

            R5900Context disableDmacCtx{};
            setRegU32(disableDmacCtx, 4, 5u);
            t.IsTrue(callSyscall(static_cast<uint32_t>(-0x1D), env.rdram.data(), &disableDmacCtx, &env.runtime),
                     "negative iDisableDmac syscall id should dispatch");
            t.Equals(getRegS32(disableDmacCtx, 2), KE_OK, "negative iDisableDmac should return KE_OK");

            R5900Context enableDmacCtx{};
            setRegU32(enableDmacCtx, 4, 5u);
            t.IsTrue(callSyscall(static_cast<uint32_t>(-0x1C), env.rdram.data(), &enableDmacCtx, &env.runtime),
                     "negative iEnableDmac syscall id should dispatch");
            t.Equals(getRegS32(enableDmacCtx, 2), KE_OK, "negative iEnableDmac should return KE_OK");

            R5900Context setEventFlagCtx{};
            setRegU32(setEventFlagCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(setEventFlagCtx, 5, 0x6u);
            t.IsTrue(callSyscall(0x53u, env.rdram.data(), &setEventFlagCtx, &env.runtime),
                     "positive retail iSetEventFlag syscall id should dispatch");
            t.Equals(getRegS32(setEventFlagCtx, 2), KE_OK, "retail iSetEventFlag should return KE_OK");

            R5900Context referCtx{};
            setRegU32(referCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(referCtx, 5, kStatusAddr);
            ReferEventFlagStatus(env.rdram.data(), &referCtx, &env.runtime);
            t.Equals(getRegS32(referCtx, 2), KE_OK, "ReferEventFlagStatus should succeed after iSetEventFlag");
            t.Equals(readGuestU32(env.rdram.data(), kStatusAddr + 12u), 0x6u,
                     "retail iSetEventFlag should publish the requested bits");

            R5900Context obsoleteSetCtx{};
            setRegU32(obsoleteSetCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(obsoleteSetCtx, 5, 0x8u);
            t.IsFalse(callSyscall(static_cast<uint32_t>(-0x53), env.rdram.data(), &obsoleteSetCtx, &env.runtime),
                      "obsolete negative iSetEventFlag id must not alias the retail syscall");

            R5900Context deleteCtx{};
            setRegU32(deleteCtx, 4, static_cast<uint32_t>(eid));
            DeleteEventFlag(env.rdram.data(), &deleteCtx, &env.runtime);

            cleanupRuntime(env);
        });

        tc.Run("retail TLB syscall ids do not dispatch obsolete event-flag handlers", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kRetailNonEventSyscalls[] = {
                0x54u,
                static_cast<uint32_t>(-0x55),
                0x56u,
                0x57u,
                static_cast<uint32_t>(-0x58),
                0x59u,
                static_cast<uint32_t>(-0x5A)
            };

            for (const uint32_t syscallNumber : kRetailNonEventSyscalls)
            {
                R5900Context syscallCtx{};
                t.IsFalse(callSyscall(syscallNumber, env.rdram.data(), &syscallCtx, &env.runtime),
                          "unimplemented retail xlaunch/TLB ids must not fall through to legacy event handlers");
            }

            cleanupRuntime(env);
        });

        tc.Run("WaitEventFlag blocks and wakes when SetEventFlag publishes bits", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kParamAddr = 0x1200u;
            constexpr uint32_t kResBitsAddr = 0x1300u;

            const uint32_t eventParam[3] = {
                0u, // attr
                0u, // option
                0u  // init bits
            };
            std::memcpy(env.rdram.data() + kParamAddr, eventParam, sizeof(eventParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kParamAddr);
            CreateEventFlag(env.rdram.data(), &createCtx, &env.runtime);
            const int32_t eid = getRegS32(createCtx, 2);
            t.IsTrue(eid > 0, "CreateEventFlag should return a valid id");

            writeGuestU32(env.rdram.data(), kResBitsAddr, 0u);

            std::atomic<bool> waiterDone{false};
            std::atomic<bool> waiterThrew{false};
            std::atomic<int32_t> waiterRet{0x7FFFFFFF};
            std::atomic<uint32_t> waiterResBits{0u};

            std::thread waiter([&]()
            {
                try
                {
                    R5900Context waitCtx{};
                    setRegU32(waitCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(waitCtx, 5, 0x4u);      // wait bits
                    setRegU32(waitCtx, 6, WEF_OR);    // OR mode
                    setRegU32(waitCtx, 7, kResBitsAddr);
                    WaitEventFlag(env.rdram.data(), &waitCtx, &env.runtime);
                    waiterRet.store(getRegS32(waitCtx, 2), std::memory_order_relaxed);
                    waiterResBits.store(readGuestU32(env.rdram.data(), kResBitsAddr), std::memory_order_relaxed);
                }
                catch (...)
                {
                    waiterThrew.store(true, std::memory_order_release);
                }

                waiterDone.store(true, std::memory_order_release);
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            t.IsFalse(waiterDone.load(std::memory_order_acquire), "WaitEventFlag should block before matching bits are set");

            R5900Context signalCtx{};
            setRegU32(signalCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(signalCtx, 5, 0x4u);
            SetEventFlag(env.rdram.data(), &signalCtx, &env.runtime);
            t.Equals(getRegS32(signalCtx, 2), KE_OK, "SetEventFlag should succeed");

            const bool woke = waitUntil([&]() {
                return waiterDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(300));
            if (!woke)
            {
                // Force unblock for deterministic test cleanup.
                R5900Context deleteCtx{};
                setRegU32(deleteCtx, 4, static_cast<uint32_t>(eid));
                DeleteEventFlag(env.rdram.data(), &deleteCtx, &env.runtime);
            }

            if (waiter.joinable())
            {
                waiter.join();
            }

            t.IsFalse(waiterThrew.load(std::memory_order_acquire),
                      "WaitEventFlag waiter thread should not throw");
            t.IsTrue(woke, "WaitEventFlag should wake after SetEventFlag publishes matching bits");
            t.Equals(waiterRet.load(std::memory_order_relaxed), KE_OK, "waiter should return KE_OK");
            t.IsTrue((waiterResBits.load(std::memory_order_relaxed) & 0x4u) != 0u,
                     "waiter result bits should include published bit");

            R5900Context deleteCtx{};
            setRegU32(deleteCtx, 4, static_cast<uint32_t>(eid));
            DeleteEventFlag(env.rdram.data(), &deleteCtx, &env.runtime);

            cleanupRuntime(env);
        });

        tc.Run("PollEventFlag WEF_CLEAR clears only matched bits", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            constexpr uint32_t kParamAddr = 0x1400u;
            constexpr uint32_t kResBitsAddr = 0x1410u;
            constexpr uint32_t kStatusAddr = 0x1420u;

            const uint32_t eventParam[3] = {
                0u, // attr
                0u, // option
                0x7u // init bits: 0b111
            };
            std::memcpy(env.rdram.data() + kParamAddr, eventParam, sizeof(eventParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kParamAddr);
            CreateEventFlag(env.rdram.data(), &createCtx, &env.runtime);
            const int32_t eid = getRegS32(createCtx, 2);
            t.IsTrue(eid > 0, "CreateEventFlag should return a valid id");

            R5900Context pollCtx{};
            setRegU32(pollCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(pollCtx, 5, 0x1u);
            setRegU32(pollCtx, 6, WEF_OR | WEF_CLEAR);
            setRegU32(pollCtx, 7, kResBitsAddr);
            PollEventFlag(env.rdram.data(), &pollCtx, &env.runtime);
            t.Equals(getRegS32(pollCtx, 2), KE_OK, "PollEventFlag should succeed when condition is met");
            t.Equals(readGuestU32(env.rdram.data(), kResBitsAddr), 0x7u, "PollEventFlag should report bits before clear");

            R5900Context referCtx{};
            setRegU32(referCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(referCtx, 5, kStatusAddr);
            ReferEventFlagStatus(env.rdram.data(), &referCtx, &env.runtime);
            t.Equals(getRegS32(referCtx, 2), KE_OK, "ReferEventFlagStatus should succeed");

            Ps2EventFlagInfo info{};
            std::memcpy(&info, env.rdram.data() + kStatusAddr, sizeof(info));
            t.Equals(info.currBits, 0x6u, "WEF_CLEAR should clear only requested bits, not all bits");

            R5900Context pollMissCtx{};
            setRegU32(pollMissCtx, 4, static_cast<uint32_t>(eid));
            setRegU32(pollMissCtx, 5, 0x1u);
            setRegU32(pollMissCtx, 6, WEF_OR);
            setRegU32(pollMissCtx, 7, 0u);
            PollEventFlag(env.rdram.data(), &pollMissCtx, &env.runtime);
            t.Equals(getRegS32(pollMissCtx, 2), KE_EVF_COND,
                     "after clearing bit 0, polling for bit 0 should fail condition");

            R5900Context deleteCtx{};
            setRegU32(deleteCtx, 4, static_cast<uint32_t>(eid));
            DeleteEventFlag(env.rdram.data(), &deleteCtx, &env.runtime);
            t.Equals(getRegS32(deleteCtx, 2), KE_OK, "DeleteEventFlag should succeed");

            cleanupRuntime(env);
        });

        tc.Run("WaitVSyncTick returns when runtime stop is requested", [](TestCase &t)
        {
            notifyRuntimeStop();
            TestEnv env;

            std::atomic<bool> waiterDone{false};
            std::atomic<bool> waiterThrew{false};
            std::thread waiter([&]()
            {
                try
                {
                    WaitVSyncTick(env.rdram.data(), &env.runtime);
                }
                catch (...)
                {
                    waiterThrew.store(true, std::memory_order_release);
                }
                waiterDone.store(true, std::memory_order_release);
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            env.runtime.requestStop();

            bool wokeOnStop = waitUntil([&]() {
                return waiterDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(80));

            if (!wokeOnStop)
            {
                // Fallback wake-up for deterministic cleanup: one extra tick on fresh runtime.
                TestEnv wakeEnv;
                R5900Context setCtx{};
                constexpr uint32_t kWakeFlagAddr = 0x1500u;
                constexpr uint32_t kWakeTickAddr = 0x1510u;
                setRegU32(setCtx, 4, kWakeFlagAddr);
                setRegU32(setCtx, 5, kWakeTickAddr);
                (void)callSyscall(0x73u, wakeEnv.rdram.data(), &setCtx, &wakeEnv.runtime);
                (void)waitUntil([&]() {
                    return readGuestU64(wakeEnv.rdram.data(), kWakeTickAddr) > 0u;
                }, std::chrono::milliseconds(300));
                wakeEnv.runtime.requestStop();
                wokeOnStop = waitUntil([&]() {
                    return waiterDone.load(std::memory_order_acquire);
                }, std::chrono::milliseconds(80));
            }

            if (waiter.joinable())
            {
                waiter.join();
            }

            t.IsFalse(waiterThrew.load(std::memory_order_acquire),
                      "WaitVSyncTick waiter thread should not throw");
            t.IsTrue(wokeOnStop, "WaitVSyncTick waiter should unblock when runtime is stopping");

            cleanupRuntime(env);
        });
    });
}
