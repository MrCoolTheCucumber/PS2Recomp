#include "Common.h"
#include "FileIO.h"
#include "Interrupt.h"
#include "Lifecycle.h"
#include "System.h"
#include "../Stubs/Audio.h"
#include "../Stubs/CD.h"
#include "../Stubs/DMA.h"
#include "../Stubs/LibC.h"
#include "../Stubs/MemoryCard.h"
#include "../Stubs/Pad.h"
#include "Helpers/InterruptRuntimeState.h"
#include "Helpers/KernelRuntimeState.h"
#include "Helpers/RpcRuntimeState.h"

namespace ps2_syscalls
{
    void notifyRuntimeStop(PS2Runtime *runtime)
    {
        if (runtime)
        {
            EeInterruptRuntimeState &state =
                runtime->eeInterruptRuntimeState();
            std::lock_guard<std::mutex> lock(
                state.handlerMutex);
            state.intcHandlers.clear();
            state.dmacHandlers.clear();
            state.nextIntcHandlerId = 1;
            state.nextDmacHandlerId = 1;
            state.intcHeadOrder = 0;
            state.intcTailOrder = 1000;
            state.dmacHeadOrder = 0;
            state.dmacTailOrder = 1000;
            state.enabledIntcMask = 0xFFFFFFFFu;
            state.enabledDmacMask = 0xFFFFFFFFu;
        }
        if (runtime)
        {
            EeInterruptRuntimeState &state =
                runtime->eeInterruptRuntimeState();
            {
                std::lock_guard<std::mutex> lock(
                    state.vsyncMutex);
                state.vsyncRegistration = {};
                state.vsyncTick = 0u;
                state.gsSyncVBaseTick = 0u;
                state.gsVideoParameters = {};
                state.gsSyncVCallback = 0u;
                state.gsSyncVCallbackGp = 0u;
                state.gsSyncVCallbackSp = 0u;
            }
            state.gsSyncVCallbackSetLogCount.store(
                0u, std::memory_order_relaxed);
            state.gsSyncVCallbackMissingLogCount.store(
                0u, std::memory_order_relaxed);
            state.gsSyncVCallbackDispatchLogCount.store(
                0u, std::memory_order_relaxed);
            state.gsSyncVCallbackBadPcLogCount.store(
                0u, std::memory_order_relaxed);
            state.gsSyncVCallbackWarningCount.store(
                0u, std::memory_order_relaxed);
        }

        std::vector<std::shared_ptr<ThreadInfo>> threads;
        threads.reserve(32);
        if (runtime)
        {
            EeThreadRuntimeState &state =
                runtime->eeThreadRuntimeState();
            {
                std::lock_guard<std::mutex> lock(
                    state.threadMapMutex);
                for (const auto &entry : state.threads)
                {
                    if (entry.second)
                    {
                        threads.push_back(
                            entry.second);
                    }
                }
                state.threads.clear();
                state.contextThreadIds.clear();
                state.contextThreadIds.emplace(
                    &runtime->cpu(), 1);
                state.nextThreadId = 2;
                state.currentThreadId.store(
                    1, std::memory_order_release);
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.legacyReadyMutex);
                for (auto &queue :
                     state.legacyReadyQueues)
                {
                    queue.clear();
                }
            }
        }
        g_currentThreadRuntime = runtime;
        g_currentThreadId = 1;

        for (const auto &threadInfo : threads)
        {
            {
                std::lock_guard<std::mutex> lock(threadInfo->m);
                threadInfo->forceRelease = true;
                threadInfo->terminated = true;
            }
            threadInfo->cv.notify_all();
        }

        std::vector<std::shared_ptr<SemaInfo>> semas;
        if (runtime)
        {
            EeSyncRuntimeState &state =
                runtime->eeSyncRuntimeState();
            std::lock_guard<std::mutex> lock(
                state.semaMapMutex);
            semas.reserve(state.semas.size());
            for (const auto &entry : state.semas)
            {
                if (entry.second)
                {
                    semas.push_back(entry.second);
                }
            }
            state.semas.clear();
            state.nextSemaId = 0;
        }
        for (const auto &sema : semas)
        {
            sema->cv.notify_all();
        }

        std::vector<std::shared_ptr<EventFlagInfo>> eventFlags;
        if (runtime)
        {
            EeSyncRuntimeState &state =
                runtime->eeSyncRuntimeState();
            std::lock_guard<std::mutex> lock(
                state.eventFlagMapMutex);
            eventFlags.reserve(state.eventFlags.size());
            for (const auto &entry : state.eventFlags)
            {
                if (entry.second)
                {
                    eventFlags.push_back(entry.second);
                }
            }
            state.eventFlags.clear();
            state.nextEventFlagId = 1;
        }
        for (const auto &eventFlag : eventFlags)
        {
            eventFlag->cv.notify_all();
        }

        stopAlarmWorker(runtime);

        if (runtime)
        {
            EeKernelRuntimeState &state =
                runtime->eeKernelRuntimeState();
            {
                std::lock_guard<std::mutex> lock(
                    state.exitHandlerMutex);
                state.exitHandlers.clear();
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.bootModeMutex);
                state.bootModeInitialized = false;
                state.bootModePoolOffset = 0u;
                state.bootModeAddresses.clear();
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.syscallOverrideMutex);
                state.syscallOverrides.clear();
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.activeSyscallOverrideMutex);
                state.activeSyscallOverrides.clear();
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.tlsMutex);
                state.tlsIndex = 0u;
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.osdMutex);
                state.osdConfigInitialized = false;
                state.osdConfigRaw = 0u;
                state.osdConfig2Raw = 0u;
            }
        }

        if (runtime)
        {
            EeRpcRuntimeState &state =
                runtime->eeRpcRuntimeState();
            {
                std::lock_guard<std::mutex> lock(
                    state.rpcMutex);
                state.initialized = false;
                state.servers.clear();
                state.clients.clear();
                state.debugHistory.fill(
                    SifRpcDebugEvent{});
                state.nextDebugSequence = 0u;
                state.nextRequestId = 1u;
                state.packetIndex = 0u;
                state.serverIndex = 0u;
                state.activeQueue = 0u;
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.moduleMutex);
                state.modulesById.clear();
                state.moduleIdByPath.clear();
                state.nextModuleId = 1;
                state.moduleLogCount = 0u;
            }
            {
                std::lock_guard<std::mutex> lock(
                    state.traceMutex);
                state.loggedTraceSignatures.clear();
            }
        }

        resetFileIoState(runtime);
        resetDeci2State(runtime);
        ps2_stubs::resetAudioStubState(runtime);
        ps2_stubs::resetCdState(runtime);
        ps2_stubs::resetDmaState(runtime);
        ps2_stubs::resetLibCState(runtime);
        ps2_stubs::resetMemoryCardState(runtime);
        ps2_stubs::resetPadState(runtime);
        ps2_stubs::resetSifState(runtime);
    }

    void joinAllGuestHostThreads(PS2Runtime *runtime)
    {
        joinAllHostThreads(runtime);
    }

    void detachAllGuestHostThreads(PS2Runtime *runtime)
    {
        detachAllHostThreads(runtime);
    }
}
