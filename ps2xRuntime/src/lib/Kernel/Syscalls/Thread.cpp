#include "Common.h"
#include "Thread.h"

namespace ps2_syscalls
{
    struct EeThreadWakePublication
    {
        EeThreadWakeResult result =
            EeThreadWakeResult::InvalidHandle;
        int status = THS_DORMANT;
        int wakeupCount = 0;
    };

    // The caller retains threadMapMutex so deletion, reset, and publication
    // have one ordering boundary. This function acquires only the selected
    // thread's state mutex.
    static EeThreadWakePublication publishRegisteredEeThreadWake(
        const std::shared_ptr<ThreadInfo> &info)
    {
        EeThreadWakePublication publication{};
        std::lock_guard<std::mutex> lock(info->m);
        if (info->guestState.isDormant())
        {
            publication.result = EeThreadWakeResult::Dormant;
        }
        else if (info->guestState.wakeSleeping())
        {
            info->cv.notify_one();
            publication.result =
                EeThreadWakeResult::WokeSleeper;
        }
        else
        {
            info->guestState.incrementWakeup();
            publication.result =
                EeThreadWakeResult::WakeupCounted;
        }

        const EeThreadGuestStateSnapshot &guest =
            info->guestState.snapshot();
        publication.status = guest.status;
        publication.wakeupCount = guest.wakeupCount;
        return publication;
    }

    static EeThreadWakePublication publishEeThreadWakeById(
        PS2Runtime *runtime,
        int threadId)
    {
        if (!runtime || threadId <= 0)
        {
            return {};
        }

        EeThreadRuntimeState &state =
            runtime->eeThreadRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.threadMapMutex);
        const auto it = state.threads.find(threadId);
        if (it == state.threads.end() || !it->second)
        {
            EeThreadWakePublication publication{};
            publication.result =
                EeThreadWakeResult::StaleHandle;
            return publication;
        }
        return publishRegisteredEeThreadWake(it->second);
    }

    EeThreadHandle captureEeThreadHandle(
        PS2Runtime *runtime,
        int threadId)
    {
        if (!runtime || threadId <= 0)
        {
            return {};
        }

        EeThreadRuntimeState &state =
            runtime->eeThreadRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.threadMapMutex);
        const auto it = state.threads.find(threadId);
        if (it == state.threads.end() || !it->second)
        {
            return {};
        }

        return {
            state.runtimeGeneration,
            it->second->generation,
            threadId,
        };
    }

    EeThreadWakeResult publishEeThreadWake(
        PS2Runtime *runtime,
        EeThreadHandle handle)
    {
        if (!runtime || !handle)
        {
            return EeThreadWakeResult::InvalidHandle;
        }

        EeThreadRuntimeState &state =
            runtime->eeThreadRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.threadMapMutex);
        if (handle.runtimeGeneration !=
            state.runtimeGeneration)
        {
            return EeThreadWakeResult::StaleHandle;
        }

        const auto it = state.threads.find(handle.threadId);
        if (it == state.threads.end() ||
            !it->second ||
            it->second->generation !=
                handle.threadGeneration)
        {
            return EeThreadWakeResult::StaleHandle;
        }

        return publishRegisteredEeThreadWake(
                   it->second)
            .result;
    }

    static void eraseThreadContextMappingsLocked(
        EeThreadRuntimeState &state,
        int tid)
    {
        for (auto it = state.contextThreadIds.begin();
             it != state.contextThreadIds.end();)
        {
            if (it->second == tid)
            {
                it = state.contextThreadIds.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    std::vector<GuestThreadDebugSnapshot> debugThreadSnapshots(
        PS2Runtime *runtime)
    {
        if (!runtime)
        {
            return {};
        }

        EeThreadRuntimeState &state =
            runtime->eeThreadRuntimeState();
        std::vector<std::pair<int, std::shared_ptr<ThreadInfo>>> threads;
        {
            std::lock_guard<std::mutex> lock(
                state.threadMapMutex);
            threads.reserve(state.threads.size());
            for (const auto &[id, info] : state.threads)
            {
                threads.emplace_back(id, info);
            }
        }

        std::sort(threads.begin(), threads.end(),
                  [](const auto &lhs, const auto &rhs)
                  {
                      return lhs.first < rhs.first;
                  });

        std::vector<GuestThreadDebugSnapshot> snapshots;
        snapshots.reserve(threads.size());
        for (const auto &[id, info] : threads)
        {
            if (!info)
            {
                continue;
            }

            GuestThreadDebugSnapshot snapshot{};
            snapshot.id = id;
            {
                std::lock_guard<std::mutex> lock(info->m);
                const EeThreadGuestStateSnapshot &guest =
                    info->guestState.snapshot();
                snapshot.entry = info->entry;
                snapshot.stack = info->stack;
                snapshot.stackSize = info->stackSize;
                snapshot.gp = info->gp;
                snapshot.status = guest.status;
                snapshot.waitType = guest.waitType;
                snapshot.waitId = guest.waitId;
                snapshot.wakeupCount = guest.wakeupCount;
                snapshot.currentPriority = guest.currentPriority;
                snapshot.suspendCount = guest.suspendCount;
                snapshot.waitQueue =
                    static_cast<int>(guest.waitQueue);
                snapshot.waitQueueId = guest.waitQueueId;
                snapshot.stateRevision = guest.revision;
                snapshot.waitCompletionPending =
                    guest.waitCompletionPending;
                snapshot.stateValid = info->guestState.isValid();
                snapshot.started = guest.started;
            }
            snapshot.pc = info->currentPc.load(std::memory_order_relaxed);
            snapshot.forceRelease =
                info->forceRelease.load(std::memory_order_relaxed);
            snapshot.terminated =
                info->terminated.load(std::memory_order_relaxed);
            snapshots.push_back(snapshot);
        }
        return snapshots;
    }

    static void notifyThreadWaitObject(
        PS2Runtime *runtime,
        int waitType,
        int waitId)
    {
        if (waitType == TSW_SEMA)
        {
            auto sema = lookupSemaInfo(runtime, waitId);
            if (sema)
            {
                sema->cv.notify_all();
            }
        }
        else if (waitType == TSW_EVENT)
        {
            auto eventFlag =
                lookupEventFlagInfo(runtime, waitId);
            if (eventFlag)
            {
                eventFlag->cv.notify_all();
            }
        }
    }

    static bool transitionReleasedWaitLocked(
        ThreadInfo &info,
        int expectedWaitType,
        int expectedWaitId)
    {
        const EeThreadWaitQueue queue =
            expectedWaitType == TSW_SEMA
                ? EeThreadWaitQueue::Semaphore
                : (expectedWaitType == TSW_EVENT
                       ? EeThreadWaitQueue::EventFlag
                       : EeThreadWaitQueue::Sleep);
        if (!info.guestState.releaseWait(
                queue,
                expectedWaitType,
                expectedWaitId,
                false))
        {
            return false;
        }

        info.forceRelease = true;
        return true;
    }

    static void runExitHandlersForThread(int tid, uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime || !ctx)
            return;

        std::vector<ExitHandlerEntry> handlers;
        EeKernelRuntimeState &kernelState =
            runtime->eeKernelRuntimeState();
        {
            std::lock_guard<std::mutex> lock(
                kernelState.exitHandlerMutex);
            auto it =
                kernelState.exitHandlers.find(tid);
            if (it == kernelState.exitHandlers.end())
                return;
            handlers = std::move(it->second);
            kernelState.exitHandlers.erase(it);
        }

        for (const auto &handler : handlers)
        {
            if (!handler.func)
                continue;
            try
            {
                rpcInvokeFunction(rdram, ctx, runtime, handler.func, handler.arg, 0, 0, 0, nullptr);
            }
            catch (const ThreadExitException &)
            {
                // ignore
            }
            catch (const std::exception &)
            {
            }
        }
    }

    void FlushCache(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, KE_OK);
    }

    void iFlushCache(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        FlushCache(rdram, ctx, runtime);
    }

    void EnableCache(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, KE_OK);
    }

    void DisableCache(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, KE_OK);
    }

    void ResetEE(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::cerr << "Syscall: ResetEE - requesting runtime stop" << std::endl;
        // runtime->requestStop();
        setReturnS32(ctx, KE_OK);
    }

    void SetMemoryMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, KE_OK);
    }

    void InitThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // This is a common ps2sdk helper that some games link against.
        // The EE bootstrap thread begins at priority 0. The linked helper
        // normally creates its priority-0 top thread and promotes the caller
        // to 1; preserve that caller-visible transition when collapsing it.
        auto info = ensureCurrentThreadInfo(runtime, ctx);
        {
            std::lock_guard<std::mutex> lock(info->m);
            info->guestState.setCurrentPriority(1);
        }
        setReturnS32(ctx, 1);
    }

    void CreateThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t paramAddr = getRegU32(ctx, 4); // $a0 points to ThreadParam
        if (paramAddr == 0u)
        {
            std::cerr << "CreateThread error: null ThreadParam pointer" << std::endl;
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        const uint32_t *param = reinterpret_cast<const uint32_t *>(getConstMemPtr(rdram, paramAddr));

        if (!param)
        {
            std::cerr << "CreateThread error: invalid ThreadParam address 0x" << std::hex << paramAddr << std::dec << std::endl;
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeThreadRuntimeState &state =
            runtime->eeThreadRuntimeState();
        auto info = std::make_shared<ThreadInfo>();
        info->entry = param[1];
        info->stack = param[2];
        info->stackSize = param[3];
        info->gp = param[4];
        info->priority = param[5];
        // The EE BIOS ignores the input status, current-priority, attr, and
        // option fields. ReferThreadStatus reports zero for attr and option.
        info->attr = 0;
        info->option = 0;
        if (info->priority >= 128)
        {
            info->priority = 127;
        }
        info->guestState.setCurrentPriority(
            static_cast<int>(info->priority));

        int id = 0;
        {
            std::lock_guard<std::mutex> lock(
                state.threadMapMutex);
            info->generation =
                state.allocateThreadGenerationLocked();
            info->boundContext = &info->context;
            // Keep IDs in the classic low range used by patched libkernel helpers.
            for (int attempts = 0; attempts < 0xFE; ++attempts)
            {
                if (state.nextThreadId < 2 ||
                    state.nextThreadId > 0xFF)
                {
                    state.nextThreadId = 2;
                }

                const int candidate = state.nextThreadId;
                state.nextThreadId =
                    (state.nextThreadId >= 0xFF)
                        ? 2
                        : (state.nextThreadId + 1);

                if (state.threads.find(candidate) ==
                    state.threads.end())
                {
                    id = candidate;
                    break;
                }
            }

            if (id == 0)
            {
                setReturnS32(ctx, KE_ERROR);
                return;
            }

            state.threads[id] = info;
            state.contextThreadIds[info->boundContext] = id;
        }

        RUNTIME_LOG("[CreateThread] id=" << id
                                         << " entry=0x" << std::hex << info->entry
                                         << " stack=0x" << info->stack
                                         << " size=0x" << info->stackSize
                                         << " gp=0x" << info->gp
                                         << " prio=" << std::dec << info->priority << std::endl);

        setReturnS32(ctx, id);
    }

    void DeleteThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4)); // $a0
        if (tid == 0)
        {
            setReturnS32(ctx, KE_ILLEGAL_THID);
            return;
        }

        auto info = lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        uint32_t autoStackToFree = 0;
        {
            std::lock_guard<std::mutex> lock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (guest.started ||
                guest.status != THS_DORMANT)
            {
                setReturnS32(ctx, KE_NOT_DORMANT);
                return;
            }

            if (info->ownsStack && info->stack != 0)
            {
                autoStackToFree = info->stack;
                info->stack = 0;
                info->stackSize = 0;
                info->ownsStack = false;
            }
        }

        {
            EeThreadRuntimeState &state =
                runtime->eeThreadRuntimeState();
            std::lock_guard<std::mutex> lock(
                state.threadMapMutex);
            eraseThreadContextMappingsLocked(state, tid);
            state.threads.erase(tid);
        }

        {
            EeKernelRuntimeState &kernelState =
                runtime->eeKernelRuntimeState();
            std::lock_guard<std::mutex> lock(
                kernelState.exitHandlerMutex);
            kernelState.exitHandlers.erase(tid);
        }

        if (runtime && autoStackToFree != 0)
        {
            runtime->guestFree(autoStackToFree);
        }

        setReturnS32(ctx, tid);
    }

    void StartThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4)); // $a0 = thread id
        uint32_t arg = getRegU32(ctx, 5);              // $a1 = user arg
        if (tid == 0)
        {
            setReturnS32(ctx, KE_ILLEGAL_THID);
            return;
        }

        auto info = lookupThreadInfo(runtime, tid);
        if (!info)
        {
            std::cerr << "StartThread error: unknown thread id " << tid << std::endl;
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        if (!runtime || !runtime->hasFunction(info->entry))
        {
            std::cerr << "[StartThread] entry 0x" << std::hex << info->entry << std::dec << " is not registered" << std::endl;
            setReturnS32(ctx, KE_ERROR);
            return;
        }
        if (runtime->isStopRequested())
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        joinHostThreadById(runtime, tid);

        const uint32_t callerSp = getRegU32(ctx, 29);
        const uint32_t callerGp = getRegU32(ctx, 28);

        {
            std::lock_guard<std::mutex> lock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (guest.started ||
                guest.status != THS_DORMANT)
            {
                setReturnS32(ctx, KE_NOT_DORMANT);
                return;
            }

            info->guestState.start();
            info->arg = arg;
            info->terminated = false;
            info->forceRelease = false;
            if (info->stack == 0 && info->stackSize != 0)
            {
                const uint32_t autoStack = runtime->guestMalloc(info->stackSize, 16u);
                if (autoStack != 0)
                {
                    info->stack = autoStack;
                    info->ownsStack = true;
                    RUNTIME_LOG("[StartThread] id=" << tid
                                                    << " auto-stack=0x" << std::hex << autoStack
                                                    << " size=0x" << info->stackSize << std::dec << std::endl);
                }
            }

            if (info->stack != 0 && info->stackSize == 0)
            {
                // Some games leave size zero in the thread param even though a stack
                // buffer is supplied; use a conservative default instead of caller SP.
                info->stackSize = 0x800u;
            }
        }

        EeThreadRuntimeState &threadState =
            runtime->eeThreadRuntimeState();
        threadState.activeHostThreads.fetch_add(
            1, std::memory_order_relaxed);
        try
        {
            std::thread worker([=]() mutable
                               {
            {
                std::string name = "PS2Thread_" + std::to_string(tid);
                ThreadNaming::SetCurrentThreadName(name);
            }
            R5900Context *threadCtx = &info->context;
            *threadCtx = R5900Context{};

            {
                std::lock_guard<std::mutex> lock(info->m);
                if (info->terminated.load(
                        std::memory_order_relaxed))
                {
                    info->guestState.makeDormant();
                }
            }

            uint32_t threadSp = callerSp;
            if (info->stack)
            {
                const uint32_t stackSize = (info->stackSize != 0) ? info->stackSize : 0x800u;
                threadSp = (info->stack + stackSize) & ~0xFu;
            }
            uint32_t threadGp = info->gp;
            const uint32_t normalizedGp = threadGp & 0x1FFFFFFFu;
            if (threadGp == 0 || normalizedGp < 0x10000u || normalizedGp >= PS2_RAM_SIZE)
            {
                threadGp = callerGp;
            }

            SET_GPR_U32(threadCtx, 29, threadSp);
            SET_GPR_U32(threadCtx, 28, threadGp);
            SET_GPR_U32(threadCtx, 4, info->arg);
            SET_GPR_U32(threadCtx, 31, 0);
            threadCtx->pc = info->entry;

            RUNTIME_LOG("[StartThread] id=" << tid
                      << " entry=0x" << std::hex << info->entry
                      << " sp=0x" << GPR_U32(threadCtx, 29)
                      << " gp=0x" << GPR_U32(threadCtx, 28)
                      << " arg=0x" << info->arg << std::dec << std::endl);

            bool exited = false;
            try
            {
                uint32_t lastPc = 0xFFFFFFFFu;
                uint32_t samePcCount = 0;
                constexpr uint32_t kSamePcYieldMask = 0xFFu;
                constexpr uint32_t kSamePcWarnInterval = 0x20000u;
                uint64_t stepCount = 0u;

                while (runtime && !runtime->isStopRequested())
                {
                    ++stepCount;
                    if (info->terminated.load(std::memory_order_relaxed))
                    {
                        throw ThreadExitException();
                    }

                    waitWhileSuspended(info, runtime);

                    const uint32_t pc = threadCtx->pc;
                    info->currentPc.store(pc, std::memory_order_relaxed);
                    if (pc == 0u)
                    {
                        break;
                    }

                    if ((stepCount & 0x1FFFFFu) == 0u)
                    {
                        RUNTIME_LOG("[StartThread] id=" << tid
                                  << " heartbeat pc=0x" << std::hex << pc
                                  << " ra=0x" << GPR_U32(threadCtx, 31)
                                  << " sp=0x" << GPR_U32(threadCtx, 29)
                                  << " gp=0x" << GPR_U32(threadCtx, 28)
                                  << std::dec << std::endl);
                    }

                    if (pc == lastPc)
                    {
                        ++samePcCount;
                        if ((samePcCount & kSamePcYieldMask) == 0u)
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        if (samePcCount > kSamePcWarnInterval)
                        {
                            // If a thread is spinning for an extremely long time (e.g. idle thread),
                            // force a 1ms sleep to prevent host CPU starvation.
                            if ((samePcCount % (kSamePcWarnInterval * 8u)) == 0u)
                            {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                            else if ((samePcCount % (kSamePcWarnInterval)) == 0u)
                            {
                                std::this_thread::yield();
                            }
                        }
                    }
                    else
                    {
                        samePcCount = 0;
                        lastPc = pc;
                    }

                    PS2Runtime::RecompiledFunction step = runtime->lookupFunction(pc);
                    if (!step)
                    {
                        std::cerr << "[StartThread] id=" << tid << " missing function for pc=0x"
                                  << std::hex << pc << std::dec << std::endl;
                        throw ThreadExitException();
                    }
                    uint64_t handoffBaseline = 0u;
                    bool suspendedAtBoundary = false;
                    {
                        PS2Runtime::GuestExecutionScope guestExecution(
                            runtime, threadCtx);
                        suspendedAtBoundary =
                            !publishRunningAtGuestBoundary(info);
                        if (!suspendedAtBoundary)
                        {
                            runtime->executeGuestStep(
                                rdram, threadCtx, step);
                            handoffBaseline =
                                runtime
                                    ->guestExecutionHandoffEpochSnapshot();
                        }
                    }
                    if (suspendedAtBoundary)
                    {
                        continue;
                    }
                    runtime->waitForGuestExecutionHandoff(handoffBaseline);
                }
            }
            catch (const ThreadExitException &)
            {
                exited = true;
            }
            catch (const std::exception &e)
            {
                std::cerr << "[StartThread] id=" << tid << " exception: " << e.what() << std::endl;
            }

            if (!exited)
            {
                RUNTIME_LOG("[StartThread] id=" << tid << " returned (pc=0x"
                          << std::hex << threadCtx->pc << std::dec << ")" << std::endl);
            }

            runExitHandlersForThread(tid, rdram, threadCtx, runtime);

            uint32_t detachedAutoStack = 0;
            {
                std::lock_guard<std::mutex> lock(info->m);
                info->guestState.makeDormant();
                info->forceRelease = false;
                info->terminated = false;
            }

            bool stillRegistered = false;
            {
                EeThreadRuntimeState &state =
                    runtime->eeThreadRuntimeState();
                std::lock_guard<std::mutex> lock(
                    state.threadMapMutex);
                stillRegistered =
                    state.threads.find(tid) !=
                    state.threads.end();
            }
            if (!stillRegistered)
            {
                // ExitDeleteThread removes the record immediately; reclaim auto stack here.
                std::lock_guard<std::mutex> lock(info->m);
                if (info->ownsStack && info->stack != 0)
                {
                    detachedAutoStack = info->stack;
                    info->stack = 0;
                    info->stackSize = 0;
                    info->ownsStack = false;
                }
            }

            if (detachedAutoStack != 0 && runtime)
            {
                runtime->guestFree(detachedAutoStack);
            }

            // Notify anybody waiting for termination (like TerminateThread)
            info->cv.notify_all();

            runtime->eeThreadRuntimeState()
                .activeHostThreads.fetch_sub(
                1, std::memory_order_relaxed); });
            registerHostThread(
                runtime, tid, std::move(worker));
        }
        catch (const std::exception &e)
        {
            std::cerr << "[StartThread] failed to spawn host thread for tid=" << tid << ": " << e.what() << std::endl;
            runtime->eeThreadRuntimeState()
                .activeHostThreads.fetch_sub(
                1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(info->m);
            info->guestState.makeDormant();
            info->forceRelease = false;
            info->terminated = false;
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        setReturnS32(ctx, tid);
    }

    void ExitThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        RUNTIME_LOG("[ExitThread] Game requested thread exit! PC=0x" << std::hex << ctx->pc
                                                                     << " RA=0x" << getRegU32(ctx, 31) << std::dec << " tid=" << getCurrentThreadId(runtime) << std::endl);

        runExitHandlersForThread(getCurrentThreadId(runtime), rdram, ctx, runtime);
        auto info = ensureCurrentThreadInfo(runtime, ctx);
        if (info)
        {
            std::lock_guard<std::mutex> lock(info->m);
            info->terminated = true;
            info->forceRelease = true;
            info->guestState.makeDormant();
        }
        if (info)
        {
            info->cv.notify_all();
        }
        throw ThreadExitException();
    }

    void ExitDeleteThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int tid = getCurrentThreadId(runtime);
        RUNTIME_LOG("[ExitDeleteThread] Game requested thread exit & delete! PC=0x" << std::hex << ctx->pc
                                                                                    << " RA=0x" << getRegU32(ctx, 31) << std::dec << " tid=" << tid << std::endl);

        runExitHandlersForThread(tid, rdram, ctx, runtime);
        auto info = ensureCurrentThreadInfo(runtime, ctx);
        if (info)
        {
            std::lock_guard<std::mutex> lock(info->m);
            info->terminated = true;
            info->forceRelease = true;
            info->guestState.makeDormant();
        }
        if (info)
        {
            info->cv.notify_all();
        }
        {
            EeThreadRuntimeState &state =
                runtime->eeThreadRuntimeState();
            std::lock_guard<std::mutex> lock(
                state.threadMapMutex);
            eraseThreadContextMappingsLocked(state, tid);
            state.threads.erase(tid);
        }
        throw ThreadExitException();
    }

    static void terminateThread(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0)
            tid = getCurrentThreadId(runtime);

        auto info = (tid == getCurrentThreadId(runtime))
                        ? ensureCurrentThreadInfo(runtime, ctx)
                        : lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        // Preserve the existing self-termination path. The external-thread
        // path below models the BIOS's synchronous scheduler transition
        // without waiting for the legacy host thread to unwind.
        if (tid == getCurrentThreadId(runtime))
        {
            {
                std::lock_guard<std::mutex> lock(info->m);
                if (info->guestState.isDormant())
                {
                    setReturnS32(ctx, KE_DORMANT);
                    return;
                }
                info->terminated = true;
                info->forceRelease = true;
                info->guestState.makeDormant();
            }
            info->cv.notify_all();
            runExitHandlersForThread(tid, rdram, ctx, runtime);
            throw ThreadExitException();
        }

        int waitType = TSW_NONE;
        int waitId = 0;
        {
            std::lock_guard<std::mutex> lock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (guest.status == THS_DORMANT)
            {
                setReturnS32(ctx, KE_DORMANT);
                return;
            }
            waitType = guest.waitType;
            waitId = guest.waitId;
        }

        bool transitioned = false;
        if (waitType == TSW_SEMA)
        {
            auto sema = lookupSemaInfo(runtime, waitId);
            if (sema)
            {
                std::lock_guard<std::mutex> semaLock(sema->m);
                std::lock_guard<std::mutex> threadLock(info->m);
                if (!info->guestState.isDormant())
                {
                    if (info->guestState.isLinkedTo(
                            EeThreadWaitQueue::Semaphore,
                            waitId) &&
                        sema->waiters > 0)
                    {
                        sema->waiters--;
                    }
                    info->terminated = true;
                    info->forceRelease = true;
                    info->guestState.makeDormant();
                    transitioned = true;
                }
            }
        }
        else if (waitType == TSW_EVENT)
        {
            auto eventFlag =
                lookupEventFlagInfo(runtime, waitId);
            if (eventFlag)
            {
                std::lock_guard<std::mutex> eventLock(
                    eventFlag->m);
                std::lock_guard<std::mutex> threadLock(
                    info->m);
                if (!info->guestState.isDormant())
                {
                    if (info->guestState.isLinkedTo(
                            EeThreadWaitQueue::EventFlag,
                            waitId) &&
                        eventFlag->waiters > 0)
                    {
                        --eventFlag->waiters;
                    }
                    info->terminated = true;
                    info->forceRelease = true;
                    info->guestState.makeDormant();
                    transitioned = true;
                }
            }
        }

        if (!transitioned)
        {
            std::lock_guard<std::mutex> lock(info->m);
            if (info->guestState.isDormant())
            {
                setReturnS32(ctx, KE_DORMANT);
                return;
            }
            info->terminated = true;
            info->forceRelease = true;
            info->guestState.makeDormant();
        }

        info->cv.notify_all();
        notifyThreadWaitObject(runtime, waitType, waitId);
        setReturnS32(ctx, tid);

        if (reschedule)
        {
            yieldGuestExecutionAtBoundary(runtime);
        }
    }

    void TerminateThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        terminateThread(rdram, ctx, runtime, true);
    }

    void iTerminateThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        terminateThread(rdram, ctx, runtime, false);
    }

    static void suspendThread(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0)
            tid = getCurrentThreadId(runtime);
        const bool suspendingCurrentThread =
            tid == getCurrentThreadId(runtime);

        auto info = suspendingCurrentThread
                        ? ensureCurrentThreadInfo(runtime, ctx)
                        : lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(info->m);
            if (info->guestState.isDormant())
            {
                setReturnS32(ctx, KE_DORMANT);
                return;
            }
            info->guestState.suspend();
        }
        info->cv.notify_all();

        if (suspendingCurrentThread)
        {
            std::unique_lock<std::mutex> lock(info->m);
            bool terminated = false;
            waitWithGuestExecutionReleasedUntilUnlocked(
                runtime,
                lock,
                [&]()
                {
                    info->cv.wait(lock, [&]()
                                  {
                                      return info->guestState.snapshot().suspendCount == 0 ||
                                             info->terminated.load();
                                  });
                },
                [&]()
                {
                    terminated = info->terminated.load();
                });

            if (terminated)
            {
                throw ThreadExitException();
            }
            (void)publishRunningAtGuestBoundary(info);
        }

        setReturnS32(ctx, tid);
        if (!suspendingCurrentThread && reschedule)
        {
            yieldGuestExecutionAtBoundary(runtime);
        }
    }

    void SuspendThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        suspendThread(rdram, ctx, runtime, true);
    }

    void iSuspendThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        suspendThread(rdram, ctx, runtime, false);
    }

    static void resumeThread(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0)
            tid = getCurrentThreadId(runtime);

        auto info = (tid == getCurrentThreadId(runtime))
                        ? ensureCurrentThreadInfo(runtime, ctx)
                        : lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        bool becameRunnable = false;
        {
            std::lock_guard<std::mutex> lock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (guest.status == THS_DORMANT)
            {
                setReturnS32(ctx, KE_DORMANT);
                return;
            }
            if (guest.suspendCount <= 0)
            {
                setReturnS32(ctx, KE_NOT_SUSPEND);
                return;
            }
            info->guestState.resume(
                tid == getCurrentThreadId(runtime),
                becameRunnable);
        }
        info->cv.notify_all();
        setReturnS32(ctx, tid);
        if (becameRunnable && reschedule)
        {
            yieldGuestExecutionAfterWake(runtime);
        }
    }

    void ResumeThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        resumeThread(rdram, ctx, runtime, true);
    }

    void iResumeThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        resumeThread(rdram, ctx, runtime, false);
    }

    void GetThreadId(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, getCurrentThreadId(runtime));
    }

    static void referThreadStatus(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        int unknownThreadResult)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t statusAddr = getRegU32(ctx, 5);

        if (tid == 0) // TH_SELF
        {
            tid = getCurrentThreadId(runtime);
        }

        auto info = (tid == getCurrentThreadId(runtime))
                        ? ensureCurrentThreadInfo(runtime, ctx)
                        : lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, unknownThreadResult);
            return;
        }

        ee_thread_status_t *status = reinterpret_cast<ee_thread_status_t *>(getMemPtr(rdram, statusAddr));
        if (!status)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        std::lock_guard<std::mutex> lock(info->m);
        const EeThreadGuestStateSnapshot &guest =
            info->guestState.snapshot();
        status->status = guest.status;
        status->func = info->entry;
        status->stack = info->stack;
        status->stack_size = info->stackSize;
        status->gp_reg = info->gp;
        status->initial_priority = info->priority;
        status->current_priority = guest.currentPriority;
        status->attr = info->attr;
        status->option = info->option;
        status->waitType = guest.waitType;
        status->waitId = guest.waitId;
        status->wakeupCount = guest.wakeupCount;
        setReturnS32(ctx, guest.status);
    }

    void ReferThreadStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        referThreadStatus(rdram, ctx, runtime, KE_UNKNOWN_THID);
    }

    void iReferThreadStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // The retail BIOS exposes status 0 for an ID removed by
        // ExitDeleteThread through the raw query and leaves the output
        // buffer untouched.
        referThreadStatus(rdram, ctx, runtime, KE_OK);
    }

    void SleepThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        auto info = ensureCurrentThreadInfo(runtime, ctx);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        throwIfTerminated(info);

        int ret = 0;
        int wakeupCountAfter = 0;
        bool terminated = false;
        std::unique_lock<std::mutex> lock(info->m);

        if (info->guestState.consumeWakeup())
        {
            ret = getCurrentThreadId(runtime);
            wakeupCountAfter =
                info->guestState.snapshot().wakeupCount;
        }
        else
        {
            static std::atomic<uint32_t> s_sleepBlockLogs{0};
            const uint32_t sleepBlockLog = s_sleepBlockLogs.fetch_add(1, std::memory_order_relaxed);
            if (sleepBlockLog < 256u)
            {
                RUNTIME_LOG("[SleepThread:block] tid=" << getCurrentThreadId(runtime)
                                                       << " pc=0x" << std::hex << ctx->pc
                                                       << " ra=0x" << getRegU32(ctx, 31)
                                                       << std::dec << std::endl);
            }

            info->guestState.block(
                EeThreadWaitQueue::Sleep,
                TSW_SLEEP,
                0);
            info->forceRelease = false;

            waitWithGuestExecutionReleasedUntilUnlocked(
                runtime,
                lock,
                [&]()
                {
                    info->cv.wait(lock, [&]()
                                  {
                                      return !info->guestState.isWaitingOn(
                                                 TSW_SLEEP, 0) ||
                                             info->forceRelease.load() ||
                                             info->terminated.load();
                                  });
                },
                [&]()
                {
                    terminated = info->terminated.load();
                    if (terminated)
                    {
                        return;
                    }

                    info->guestState.finishWait(
                        EeThreadWaitQueue::Sleep,
                        TSW_SLEEP,
                        0,
                        false);

                    if (info->forceRelease.load())
                    {
                        info->forceRelease = false;
                        ret = KE_RELEASE_WAIT;
                    }
                    else
                    {
                        ret = getCurrentThreadId(runtime);
                    }
                    wakeupCountAfter =
                        info->guestState.snapshot().wakeupCount;
                });
        }

        if (terminated)
        {
            throw ThreadExitException();
        }

        static std::atomic<uint32_t> s_sleepWakeLogs{0};
        const uint32_t sleepWakeLog = s_sleepWakeLogs.fetch_add(1, std::memory_order_relaxed);
        if (sleepWakeLog < 256u)
        {
            RUNTIME_LOG("[SleepThread:wake] tid=" << getCurrentThreadId(runtime)
                                                  << " ret=" << ret
                                                  << " wakeupCount=" << wakeupCountAfter
                                                  << std::endl);
        }

        if (lock.owns_lock())
        {
            lock.unlock();
        }
        {
            std::lock_guard<std::mutex> stateLock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (!info->terminated.load() &&
                guest.suspendCount == 0 &&
                guest.status == THS_READY)
            {
                info->guestState.publishRunning();
            }
        }
        waitWhileSuspended(info, runtime);
        (void)publishRunningAtGuestBoundary(info);
        setReturnS32(ctx, ret);
    }

    static void wakeupThread(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0)
        {
            setReturnS32(ctx, KE_ILLEGAL_THID);
            return;
        }
        if (tid == getCurrentThreadId(runtime))
        {
            setReturnS32(ctx, KE_ILLEGAL_THID);
            return;
        }

        const EeThreadWakePublication publication =
            publishEeThreadWakeById(runtime, tid);
        if (publication.result ==
                EeThreadWakeResult::InvalidHandle ||
            publication.result ==
                EeThreadWakeResult::StaleHandle)
        {
            // The raw EE wake syscall collapses a deleted ID to generic -1;
            // the ordinary syscall retains the kernel's unknown-ID result.
            setReturnS32(
                ctx,
                reschedule ? KE_UNKNOWN_THID : KE_ERROR);
            return;
        }

        if (publication.result ==
            EeThreadWakeResult::Dormant)
        {
            setReturnS32(ctx, KE_DORMANT);
            return;
        }

        static std::atomic<uint32_t> s_wakeupLogs{0};
        const uint32_t wakeupLog = s_wakeupLogs.fetch_add(1, std::memory_order_relaxed);
        if (wakeupLog < 256u)
        {
            RUNTIME_LOG("[WakeupThread] tid=" << getCurrentThreadId(runtime)
                                              << " target=" << tid
                                              << " status=" << publication.status
                                              << " wakeupCount=" << publication.wakeupCount
                                              << std::endl);
        }
        setReturnS32(ctx, tid);
        if (publication.result ==
                EeThreadWakeResult::WokeSleeper &&
            reschedule)
        {
            yieldGuestExecutionAfterWake(runtime);
        }
    }

    void WakeupThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        wakeupThread(rdram, ctx, runtime, true);
    }

    void iWakeupThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        wakeupThread(rdram, ctx, runtime, false);
    }

    void CancelWakeupThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0)
            tid = getCurrentThreadId(runtime);

        auto info = (tid == getCurrentThreadId(runtime))
                        ? ensureCurrentThreadInfo(runtime, ctx)
                        : lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        int previous = 0;
        {
            std::lock_guard<std::mutex> lock(info->m);
            previous = info->guestState.cancelWakeups();
        }
        setReturnS32(ctx, previous);
    }

    void iCancelWakeupThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0)
        {
            setReturnS32(ctx, KE_ILLEGAL_THID);
            return;
        }

        auto info = lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        int previous = 0;
        {
            std::lock_guard<std::mutex> lock(info->m);
            previous = info->guestState.cancelWakeups();
        }
        setReturnS32(ctx, previous);
    }

    static void changeThreadPriority(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        int newPrio = static_cast<int>(getRegU32(ctx, 5));
        int previousPriority = 0;

        if (tid == 0)
            tid = getCurrentThreadId(runtime);

        auto info = (tid == getCurrentThreadId(runtime))
                        ? ensureCurrentThreadInfo(runtime, ctx)
                        : lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(info->m);
            if (info->guestState.isDormant())
            {
                setReturnS32(ctx, KE_DORMANT);
                return;
            }

            if (newPrio < 0 || newPrio >= 128)
            {
                setReturnS32(ctx, KE_ILLEGAL_PRIORITY);
                return;
            }

            previousPriority =
                info->guestState.changeCurrentPriority(newPrio);
        }

        setReturnS32(ctx, previousPriority);
        if (reschedule)
        {
            yieldGuestExecutionAtBoundary(runtime);
        }
    }

    void ChangeThreadPriority(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        changeThreadPriority(rdram, ctx, runtime, true);
    }

    void iChangeThreadPriority(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        changeThreadPriority(rdram, ctx, runtime, false);
    }

    static void rotateThreadReadyQueue(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        static int logCount = 0;
        int prio = static_cast<int>(getRegU32(ctx, 4));
        const int requestedPriority = prio;
        if (logCount < 16)
        {
            RUNTIME_LOG("[RotateThreadReadyQueue] prio=" << prio);
            ++logCount;
        }
        if (prio < 0 || prio >= 128)
        {
            runtime->recordEeThreadQueueRotation(
                getCurrentThreadId(runtime),
                requestedPriority,
                prio,
                false);
            setReturnS32(ctx, KE_ILLEGAL_PRIORITY);
            return;
        }

        runtime->recordEeThreadQueueRotation(
            getCurrentThreadId(runtime),
            requestedPriority,
            prio,
            true);
        setReturnS32(ctx, prio);
        if (reschedule)
        {
            yieldGuestExecutionAtBoundary(runtime);
        }
    }

    void RotateThreadReadyQueue(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        rotateThreadReadyQueue(rdram, ctx, runtime, true);
    }

    void iRotateThreadReadyQueue(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        rotateThreadReadyQueue(rdram, ctx, runtime, false);
    }

    static void releaseWaitThread(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int tid = static_cast<int>(getRegU32(ctx, 4));
        if (tid == 0 || tid == getCurrentThreadId(runtime))
        {
            setReturnS32(ctx, KE_ILLEGAL_THID);
            return;
        }

        auto info = lookupThreadInfo(runtime, tid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_THID);
            return;
        }

        bool wasWaiting = false;
        int waitType = 0;
        int waitId = 0;

        {
            std::lock_guard<std::mutex> lock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (guest.status == THS_WAIT ||
                guest.status == THS_WAITSUSPEND)
            {
                waitType = guest.waitType;
                waitId = guest.waitId;
            }
        }

        if (waitType == TSW_SEMA)
        {
            auto sema = lookupSemaInfo(runtime, waitId);
            if (sema)
            {
                std::lock_guard<std::mutex> semaLock(sema->m);
                std::lock_guard<std::mutex> threadLock(info->m);
                wasWaiting = transitionReleasedWaitLocked(
                    *info, waitType, waitId);
                if (wasWaiting &&
                    sema->waiters > 0)
                {
                    sema->waiters--;
                }
            }
        }
        else if (waitType == TSW_EVENT)
        {
            auto eventFlag =
                lookupEventFlagInfo(runtime, waitId);
            if (eventFlag)
            {
                std::lock_guard<std::mutex> eventLock(
                    eventFlag->m);
                std::lock_guard<std::mutex> threadLock(
                    info->m);
                wasWaiting = transitionReleasedWaitLocked(
                    *info, waitType, waitId);
                if (wasWaiting &&
                    eventFlag->waiters > 0)
                {
                    --eventFlag->waiters;
                }
            }
        }
        else
        {
            std::lock_guard<std::mutex> lock(info->m);
            wasWaiting = transitionReleasedWaitLocked(
                *info, waitType, waitId);
        }

        if (!wasWaiting)
        {
            setReturnS32(ctx, KE_NOT_WAIT);
            return;
        }

        info->cv.notify_all();
        notifyThreadWaitObject(runtime, waitType, waitId);
        setReturnS32(ctx, tid);
        if (reschedule)
        {
            yieldGuestExecutionAfterWake(runtime);
        }
    }

    void ReleaseWaitThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        releaseWaitThread(rdram, ctx, runtime, true);
    }

    void iReleaseWaitThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        releaseWaitThread(rdram, ctx, runtime, false);
    }
}
