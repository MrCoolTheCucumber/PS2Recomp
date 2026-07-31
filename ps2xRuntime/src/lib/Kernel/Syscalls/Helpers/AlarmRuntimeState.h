#pragma once

#include "ps2_runtime.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

struct AlarmInfo
{
    int id = 0;
    uint16_t ticks = 0;
    uint32_t handler = 0;
    uint32_t commonArg = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
    uint8_t *rdram = nullptr;
    std::chrono::steady_clock::time_point dueAt;
};

// Runtime-owned alarm registry, worker continuation, and guest callback
// context. The worker is joined synchronously by notifyRuntimeStop; the
// destructor is a final safety net for partially constructed runtimes.
struct EeAlarmRuntimeState
{
    ~EeAlarmRuntimeState()
    {
        std::thread workerToJoin;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopRequested.store(
                true, std::memory_order_release);
            alarms.clear();
            pendingCallbacks.clear();
            pendingCallbackPublished.store(
                false, std::memory_order_release);
            workerToJoin = std::move(worker);
        }
        cv.notify_all();
        if (workerToJoin.joinable())
        {
            if (workerToJoin.get_id() ==
                std::this_thread::get_id())
            {
                // Destroying the runtime from its own callback cannot be
                // made safe: the callback still refers to runtime-owned
                // state. Keep the no-detach ownership contract explicit.
                std::terminate();
            }
            workerToJoin.join();
        }
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<int, std::shared_ptr<AlarmInfo>> alarms;
    std::deque<std::shared_ptr<AlarmInfo>> pendingCallbacks;
    std::atomic<bool> pendingCallbackPublished{false};
    int nextAlarmId = 1;
    std::atomic<bool> stopRequested{false};
    std::thread worker;
    bool workerRunning = false;

    R5900Context callbackContext{};
    uint32_t callbackStackTop = 0u;
    uint32_t exceptionLogCount = 0u;
    uint32_t setLogCount = 0u;
};
