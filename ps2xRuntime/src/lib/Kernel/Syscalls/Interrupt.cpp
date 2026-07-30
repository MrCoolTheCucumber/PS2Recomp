#include "Common.h"
#include "Interrupt.h"
#include "Helpers/InterruptRuntimeState.h"
#include "ps2_log.h"
#include "Stubs/GS.h"

namespace ps2_syscalls
{
    namespace interrupt_state
    {
        constexpr uint32_t kIntcVblankStart = 2u;
        constexpr uint32_t kIntcVblankEnd = 3u;
        constexpr uint32_t kMaxIrqHandlerSteps = 4096u;

        std::mutex g_vsync_flag_mutex;
        uint64_t g_vsync_tick_counter = 0u;
        VSyncFlagRegistration g_vsync_registration{};
    }

    using namespace interrupt_state;

    static void writeGuestU32NoThrow(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        if (addr == 0u)
        {
            return;
        }

        uint8_t *dst = getMemPtr(rdram, addr);
        if (!dst)
        {
            return;
        }
        std::memcpy(dst, &value, sizeof(value));
    }

    static void writeGuestU64NoThrow(uint8_t *rdram, uint32_t addr, uint64_t value)
    {
        if (addr == 0u)
        {
            return;
        }

        uint8_t *dst = getMemPtr(rdram, addr);
        if (!dst)
        {
            return;
        }
        std::memcpy(dst, &value, sizeof(value));
    }

    static uint32_t readGuestU32NoThrow(uint8_t *rdram, uint32_t addr)
    {
        if (addr == 0u)
        {
            return 0u;
        }

        uint8_t *src = getMemPtr(rdram, addr);
        if (!src)
        {
            return 0u;
        }

        uint32_t value = 0u;
        std::memcpy(&value, src, sizeof(value));
        return value;
    }

    class InterruptCallbackContextLease
    {
    public:
        InterruptCallbackContextLease(
            EeInterruptRuntimeState &state,
            PS2Runtime &runtime)
            : m_state(state),
              m_slot(state.acquireCallbackContext(runtime))
        {
        }

        ~InterruptCallbackContextLease()
        {
            m_state.releaseCallbackContext(m_slot);
        }

        R5900Context &context()
        {
            return m_slot->context;
        }

        uint32_t stackTop() const
        {
            return m_slot->stackTop != 0u
                       ? m_slot->stackTop
                       : (PS2_RAM_SIZE - 0x10u);
        }

        InterruptCallbackContextLease(
            const InterruptCallbackContextLease &) = delete;
        InterruptCallbackContextLease &operator=(
            const InterruptCallbackContextLease &) = delete;

    private:
        EeInterruptRuntimeState &m_state;
        EeInterruptCallbackContext *m_slot = nullptr;
    };

    void dispatchIntcHandlersForCause(uint8_t *rdram, PS2Runtime *runtime, uint32_t cause)
    {
        if (!rdram || !runtime)
        {
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        std::vector<IrqHandlerInfo> handlers;
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            if (cause < 32u &&
                (state.enabledIntcMask &
                 (1u << cause)) == 0u)
            {
                return;
            }

            handlers.reserve(state.intcHandlers.size());
            for (const auto &[id, info] :
                 state.intcHandlers)
            {
                (void)id;
                if (!info.enabled)
                {
                    continue;
                }
                if (info.cause != cause)
                {
                    continue;
                }
                if (info.handler == 0u)
                {
                    continue;
                }
                handlers.push_back(info);
            }
            std::sort(handlers.begin(), handlers.end(), [](const IrqHandlerInfo &a, const IrqHandlerInfo &b)
                      { return a.order < b.order; });
        }

        for (const IrqHandlerInfo &info : handlers)
        {
            if (!runtime->hasFunction(info.handler))
            {
                if (cause == kIntcVblankStart)
                {
                    PS2_IF_AGRESSIVE_LOGS({
                        static std::atomic<uint32_t> s_missingHandlerLogCount{0u};
                        const uint32_t logIndex = s_missingHandlerLogCount.fetch_add(1u, std::memory_order_relaxed);
                        if (logIndex < 32u)
                        {
                            auto flags = std::cout.flags();
                            std::cout << "[INTC:missing] cause=" << cause
                                      << " handler=0x" << std::hex << info.handler
                                      << std::dec
                                      << " id=" << info.id
                                      << std::endl;
                            std::cout.flags(flags);
                        }
                    });
                }
                continue;
            }

            try
            {
                InterruptCallbackContextLease callback(
                    state, *runtime);
                R5900Context &irqCtx =
                    callback.context();
                SET_GPR_U32(&irqCtx, 28, info.gp);
                SET_GPR_U32(
                    &irqCtx, 29, callback.stackTop());
                SET_GPR_U32(&irqCtx, 31, 0u);
                SET_GPR_U32(&irqCtx, 4, cause);
                SET_GPR_U32(&irqCtx, 5, info.arg);
                SET_GPR_U32(&irqCtx, 6, 0u);
                SET_GPR_U32(&irqCtx, 7, 0u);
                irqCtx.pc = info.handler;

                bool reschedulePending = false;
                uint64_t handoffBaseline = 0u;
                uint32_t stepCount = 0u;
                {
                    PS2Runtime::GuestExecutionScope guestExecution(
                        runtime, &irqCtx);
                    PS2Runtime::DeferredGuestYieldScope deferYield(reschedulePending);

                    while (irqCtx.pc != 0u && runtime && !runtime->isStopRequested() && stepCount < kMaxIrqHandlerSteps)
                    {
                        PS2Runtime::RecompiledFunction step = runtime->lookupFunction(irqCtx.pc);
                        if (!step)
                        {
                            break;
                        }
                        runtime->executeGuestStep(rdram, &irqCtx, step);
                        ++stepCount;
                    }
                    handoffBaseline = runtime->guestExecutionHandoffEpochSnapshot();
                }
                if (stepCount >= kMaxIrqHandlerSteps)
                {
                    static uint32_t s_stepLimitLogCount = 0u;
                    if (s_stepLimitLogCount < 16u)
                    {
                        std::cerr << "[INTC:step-limit] handler=0x" << std::hex << info.handler << " pc=0x" << irqCtx.pc << std::dec << std::endl;
                        ++s_stepLimitLogCount;
                    }
                }
                if (reschedulePending && !runtime->isStopRequested())
                {
                    runtime->waitForGuestExecutionHandoff(handoffBaseline);
                }
                runtime->drainCompletedDmacHandlers(rdram);
            }
            catch (const ThreadExitException &)
            {
            }
            catch (const std::exception &e)
            {
                static uint32_t warnCount = 0;
                if (warnCount < 8u)
                {
                    std::cerr << "[INTC] handler 0x" << std::hex << info.handler
                              << " threw exception: " << e.what() << std::dec << std::endl;
                    ++warnCount;
                }
            }
        }
    }

    bool isIntcCauseEnabled(
        PS2Runtime *runtime, uint32_t cause)
    {
        if (!runtime)
        {
            return false;
        }
        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.handlerMutex);
        return cause < 32u &&
               (state.enabledIntcMask &
                (1u << cause)) != 0u;
    }

    void dispatchDmacHandlersForCause(uint8_t *rdram, PS2Runtime *runtime, uint32_t cause)
    {
        if (!rdram || !runtime)
        {
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        std::vector<IrqHandlerInfo> handlers;
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            if (cause < 32u &&
                (state.enabledDmacMask &
                 (1u << cause)) == 0u)
            {
                return;
            }

            handlers.reserve(state.dmacHandlers.size());
            for (const auto &[id, info] :
                 state.dmacHandlers)
            {
                (void)id;
                if (!info.enabled)
                {
                    continue;
                }
                if (info.cause != cause)
                {
                    continue;
                }
                if (info.handler == 0u)
                {
                    continue;
                }
                handlers.push_back(info);
            }
            std::sort(handlers.begin(), handlers.end(), [](const IrqHandlerInfo &a, const IrqHandlerInfo &b)
                      { return a.order < b.order; });
        }

        for (const IrqHandlerInfo &info : handlers)
        {
            if (!runtime->hasFunction(info.handler))
            {
                continue;
            }

            try
            {
                InterruptCallbackContextLease callback(
                    state, *runtime);
                R5900Context &irqCtx =
                    callback.context();
                SET_GPR_U32(&irqCtx, 28, info.gp);
                SET_GPR_U32(
                    &irqCtx, 29, callback.stackTop());
                SET_GPR_U32(&irqCtx, 31, 0u);
                SET_GPR_U32(&irqCtx, 4, cause);
                SET_GPR_U32(&irqCtx, 5, info.arg);
                SET_GPR_U32(&irqCtx, 6, 0u);
                SET_GPR_U32(&irqCtx, 7, 0u);
                irqCtx.pc = info.handler;

                bool reschedulePending = false;
                uint64_t handoffBaseline = 0u;
                uint32_t steps = 0u;
                {
                    PS2Runtime::GuestExecutionScope guestExecution(
                        runtime, &irqCtx);
                    PS2Runtime::DeferredGuestYieldScope deferYield(reschedulePending);

                    while (irqCtx.pc != 0u && runtime && !runtime->isStopRequested() &&
                           steps < kMaxIrqHandlerSteps)
                    {
                        PS2Runtime::RecompiledFunction step = runtime->lookupFunction(irqCtx.pc);
                        if (!step)
                        {
                            break;
                        }
                        runtime->executeGuestStep(rdram, &irqCtx, step);
                        ++steps;
                    }
                    handoffBaseline = runtime->guestExecutionHandoffEpochSnapshot();
                }
                if (steps >= kMaxIrqHandlerSteps)
                {
                    static uint32_t s_stepLimitLogCount = 0u;
                    if (s_stepLimitLogCount < 16u)
                    {
                        std::cerr << "[DMAC:step-limit] handler=0x" << std::hex << info.handler
                                  << " pc=0x" << irqCtx.pc << std::dec << std::endl;
                        ++s_stepLimitLogCount;
                    }
                }
                if (reschedulePending && !runtime->isStopRequested())
                {
                    runtime->waitForGuestExecutionHandoff(handoffBaseline);
                }
            }
            catch (const ThreadExitException &)
            {
            }
            catch (const std::exception &e)
            {
                static uint32_t warnCount = 0;
                if (warnCount < 8u)
                {
                    std::cerr << "[DMAC] handler 0x" << std::hex << info.handler
                              << " threw exception: " << e.what() << std::dec << std::endl;
                    ++warnCount;
                }
            }
        }
    }

    bool isDmacCauseEnabled(
        PS2Runtime *runtime, uint32_t cause)
    {
        if (!runtime || cause >= 32u)
        {
            return false;
        }
        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.handlerMutex);
        return (state.enabledDmacMask &
                (1u << cause)) != 0u;
    }

    static void updateGsCsrFieldForVSync(PS2Runtime *runtime, uint64_t tickValue)
    {
        if (!runtime)
        {
            return;
        }

        constexpr uint64_t kGsCsrFieldMask = 0x2000ull;
        std::atomic<uint64_t> &csr = runtime->memory().gs().csr;
        if (tickValue & 1ull)
        {
            csr.fetch_or(kGsCsrFieldMask);
        }
        else
        {
            csr.fetch_and(~kGsCsrFieldMask);
        }
    }

    static uint64_t signalVSyncFlag(uint8_t *rdram, PS2Runtime *runtime)
    {
        VSyncFlagRegistration reg{};
        uint64_t tickValue = 0u;
        {
            std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
            reg = g_vsync_registration;
            g_vsync_registration = {};
            tickValue = ++g_vsync_tick_counter;
        }

        if (reg.flagAddr != 0u)
        {
            writeGuestU32NoThrow(rdram, reg.flagAddr, 1u);
        }
        if (reg.tickAddr != 0u)
        {
            writeGuestU64NoThrow(rdram, reg.tickAddr, tickValue);
        }
        return tickValue;
    }

    uint64_t PublishVSyncStart(
        uint8_t *rdram, PS2Runtime *runtime)
    {
        return signalVSyncFlag(rdram, runtime);
    }

    void PublishVSyncField(
        PS2Runtime *runtime, uint64_t tick)
    {
        updateGsCsrFieldForVSync(runtime, tick);
    }

    void DispatchVSyncStartHandlers(
        uint8_t *rdram,
        PS2Runtime *runtime,
        uint64_t tick)
    {
        ps2_stubs::dispatchGsSyncVCallback(
            rdram, runtime, tick);
        dispatchIntcHandlersForCause(
            rdram, runtime, kIntcVblankStart);
    }

    void DispatchVSyncEndHandlers(
        uint8_t *rdram, PS2Runtime *runtime)
    {
        dispatchIntcHandlersForCause(
            rdram, runtime, kIntcVblankEnd);
    }

    void EnsureVSyncScheduled(uint8_t *rdram, PS2Runtime *runtime)
    {
        if (rdram && runtime)
        {
            runtime->ensureEeVSyncScheduled();
        }
    }

    uint64_t GetCurrentVSyncTick()
    {
        std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
        return g_vsync_tick_counter;
    }

    uint64_t WaitForNextVSyncTick(
        uint8_t *rdram,
        PS2Runtime *runtime,
        R5900Context *ctx)
    {
        EnsureVSyncScheduled(rdram, runtime);
        if (!runtime)
        {
            return GetCurrentVSyncTick();
        }

        uint64_t current = 0u;
        {
            std::lock_guard<std::mutex> lock(
                g_vsync_flag_mutex);
            current = g_vsync_tick_counter;
        }
        return runtime->waitForNextScheduledVSync(
            rdram, ctx, current);
    }

    void WaitVSyncTick(uint8_t *rdram, PS2Runtime *runtime)
    {
        (void)WaitForNextVSyncTick(rdram, runtime, nullptr);
    }

    void SetVSyncFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t flagAddr = getRegU32(ctx, 4);
        const uint32_t tickAddr = getRegU32(ctx, 5);

        {
            std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
            g_vsync_registration.flagAddr = flagAddr;
            g_vsync_registration.tickAddr = tickAddr;
        }

        writeGuestU32NoThrow(rdram, flagAddr, 0u);
        writeGuestU64NoThrow(rdram, tickAddr, 0u);
        EnsureVSyncScheduled(rdram, runtime);
        setReturnS32(ctx, KE_OK);
    }

    void EnableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            state.enabledIntcMask |=
                (1u << cause);
        }
        runtime->memory().setIntcInterruptMask(
            cause, true);
        if (cause == kIntcVblankStart || cause == kIntcVblankEnd)
        {
            PS2_IF_AGRESSIVE_LOGS({
                static std::atomic<uint32_t> s_enableLogCount{0u};
                const uint32_t logIndex = s_enableLogCount.fetch_add(1u, std::memory_order_relaxed);
                if (logIndex < 32u)
                {
                    RUNTIME_LOG("[EnableIntc] cause=" << cause);
                }
            });
        }
        setReturnS32(ctx, KE_OK);
    }

    void iEnableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EnableIntc(rdram, ctx, runtime);
    }

    void DisableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            state.enabledIntcMask &=
                ~(1u << cause);
        }
        runtime->memory().setIntcInterruptMask(
            cause, false);
        if (cause == kIntcVblankStart || cause == kIntcVblankEnd)
        {
            PS2_IF_AGRESSIVE_LOGS({
                static std::atomic<uint32_t> s_disableLogCount{0u};
                const uint32_t logIndex = s_disableLogCount.fetch_add(1u, std::memory_order_relaxed);
                if (logIndex < 32u)
                {
                    RUNTIME_LOG("[DisableIntc] cause=" << cause);
                }
            });
        }
        setReturnS32(ctx, KE_OK);
    }

    void iDisableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        DisableIntc(rdram, ctx, runtime);
    }

    void AddIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        IrqHandlerInfo info{};
        info.cause = getRegU32(ctx, 4);
        info.handler = getRegU32(ctx, 5);
        uint32_t next = getRegU32(ctx, 6);
        info.arg = getRegU32(ctx, 7);
        info.gp = getRegU32(ctx, 28);
        info.sp = getRegU32(ctx, 29);
        info.enabled = true;

        int handlerId = 0;
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            info.order =
                (next == 0)
                    ? --state.intcHeadOrder
                    : ++state.intcTailOrder;
            handlerId = state.nextIntcHandlerId++;
            info.id = handlerId;
            state.intcHandlers[handlerId] = info;
        }

        if (info.cause == kIntcVblankStart)
        {
            PS2_IF_AGRESSIVE_LOGS({
                static std::atomic<uint32_t> s_addHandlerLogCount{0u};
                const uint32_t logIndex = s_addHandlerLogCount.fetch_add(1u, std::memory_order_relaxed);
                if (logIndex < 32u)
                {
                    auto flags = std::cout.flags();
                    std::cout << "[AddIntcHandler] cause=" << info.cause
                              << " handler=0x" << std::hex << info.handler
                              << " arg=0x" << info.arg
                              << " gp=0x" << info.gp
                              << " sp=0x" << info.sp
                              << std::dec
                              << " id=" << handlerId
                              << std::endl;
                    std::cout.flags(flags);
                }
            });
        }

        EnsureVSyncScheduled(rdram, runtime);
        setReturnS32(ctx, handlerId);
    }

    void AddIntcHandler2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        AddIntcHandler(rdram, ctx, runtime);
    }

    void RemoveIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const uint32_t cause = getRegU32(ctx, 4);
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        if (handlerId > 0)
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            auto it = state.intcHandlers.find(handlerId);
            if (it != state.intcHandlers.end() &&
                it->second.cause == cause)
            {
                state.intcHandlers.erase(it);
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void AddDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        IrqHandlerInfo info{};
        info.cause = getRegU32(ctx, 4);
        info.handler = getRegU32(ctx, 5);
        uint32_t next = getRegU32(ctx, 6);
        info.arg = getRegU32(ctx, 7);
        info.gp = getRegU32(ctx, 28);
        info.sp = getRegU32(ctx, 29);
        info.enabled = true;

        int handlerId = 0;
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            info.order =
                (next == 0)
                    ? --state.dmacHeadOrder
                    : ++state.dmacTailOrder;
            handlerId = state.nextDmacHandlerId++;
            info.id = handlerId;
            state.dmacHandlers[handlerId] = info;
        }
        setReturnS32(ctx, handlerId);
    }

    void AddDmacHandler2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        AddDmacHandler(rdram, ctx, runtime);
    }

    void RemoveDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const uint32_t cause = getRegU32(ctx, 4);
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        if (handlerId > 0)
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            auto it = state.dmacHandlers.find(handlerId);
            if (it != state.dmacHandlers.end() &&
                it->second.cause == cause)
            {
                state.dmacHandlers.erase(it);
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void EnableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            if (auto it =
                    state.intcHandlers.find(handlerId);
                it != state.intcHandlers.end())
            {
                it->second.enabled = true;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void DisableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            if (auto it =
                    state.intcHandlers.find(handlerId);
                it != state.intcHandlers.end())
            {
                it->second.enabled = false;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void EnableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            if (auto it =
                    state.dmacHandlers.find(handlerId);
                it != state.dmacHandlers.end())
            {
                it->second.enabled = true;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void DisableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            if (auto it =
                    state.dmacHandlers.find(handlerId);
                it != state.dmacHandlers.end())
            {
                it->second.enabled = false;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void EnableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            state.enabledDmacMask |= (1u << cause);
        }
        if (cause < PS2_DMAC_CHANNEL_COUNT)
        {
            runtime->memory().setDmacInterruptMask(
                static_cast<DmacChannel>(cause), true);
        }
        setReturnS32(ctx, KE_OK);
    }

    void iEnableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EnableDmac(rdram, ctx, runtime);
    }

    void DisableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeInterruptRuntimeState &state =
            runtime->eeInterruptRuntimeState();
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            state.enabledDmacMask &= ~(1u << cause);
        }
        if (cause < PS2_DMAC_CHANNEL_COUNT)
        {
            runtime->memory().setDmacInterruptMask(
                static_cast<DmacChannel>(cause), false);
        }
        setReturnS32(ctx, KE_OK);
    }

    void iDisableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        DisableDmac(rdram, ctx, runtime);
    }
}
