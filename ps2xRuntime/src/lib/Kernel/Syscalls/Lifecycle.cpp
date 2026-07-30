#include "Common.h"
#include "Interrupt.h"
#include "Lifecycle.h"

namespace ps2_syscalls
{
    using namespace interrupt_state;

    void notifyRuntimeStop(PS2Runtime *runtime)
    {
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            g_intcHandlers.clear();
            g_dmacHandlers.clear();
            g_nextIntcHandlerId = 1;
            g_nextDmacHandlerId = 1;
            g_enabled_intc_mask = 0xFFFFFFFFu;
            g_enabled_dmac_mask = 0xFFFFFFFFu;
        }
        {
            std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
            g_vsync_registration = {};
            g_vsync_tick_counter = 0u;
        }

        std::vector<std::shared_ptr<ThreadInfo>> threads;
        threads.reserve(32);
        if (runtime)
        {
            EeThreadRuntimeState &state =
                runtime->eeThreadRuntimeState();
            std::lock_guard<std::mutex> lock(
                state.threadMapMutex);
            for (const auto &entry : state.threads)
            {
                if (entry.second)
                {
                    threads.push_back(entry.second);
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

        {
            std::lock_guard<std::mutex> lock(g_exit_handler_mutex);
            g_exit_handlers.clear();
        }
        {
            std::lock_guard<std::mutex> lock(g_syscall_override_mutex);
            g_syscall_overrides.clear();
        }
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
