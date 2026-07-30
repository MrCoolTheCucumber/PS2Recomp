#pragma once

#include "ps2_runtime.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

struct ThreadInfo
{
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    uint32_t priority = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t arg = 0;
    uint32_t generation = 0;
    bool started = false;
    bool ownsStack = false;
    uint32_t tlsBase = 0;

    // Guest-visible thread state.
    int status = 0x10; // THS_DORMANT
    int waitType = 0;  // TSW_NONE
    int waitId = 0;
    int wakeupCount = 0;
    int currentPriority = 0;
    int suspendCount = 0;

    // The legacy host-thread backend executes directly from this
    // runtime-owned context. Main thread ID 1 continues to use
    // PS2Runtime::cpu() as its runtime-owned context.
    R5900Context context{};
    R5900Context *boundContext = nullptr;
    std::atomic<uint32_t> currentPc{0};

    std::mutex m;
    std::condition_variable cv;
    std::atomic<bool> forceRelease{false};
    std::atomic<bool> terminated{false};
    std::atomic<bool> semaWaitLinked{false};
};

struct EeThreadRuntimeState
{
    std::mutex threadMapMutex;
    std::unordered_map<int, std::shared_ptr<ThreadInfo>> threads;
    std::unordered_map<const R5900Context *, int> contextThreadIds;
    int nextThreadId = 2;
    uint32_t nextGeneration = 1u;

    std::mutex hostThreadMutex;
    std::unordered_map<int, std::thread> hostThreads;
    std::atomic<int> activeHostThreads{0};

    // This is the runtime authority. The host TLS ID remains a checked
    // adapter until all legacy direct-call sites bind through the executor.
    std::atomic<int> currentThreadId{1};
    std::atomic<uint64_t> legacyAdapterMismatches{0u};
};

// Compatibility view for legacy code that still observes a host TLS slot.
// Runtime context binding owns the value; syscall code must use
// PS2Runtime::currentEeThreadId() instead.
inline thread_local PS2Runtime *g_currentThreadRuntime = nullptr;
inline thread_local int g_currentThreadId = 1;
