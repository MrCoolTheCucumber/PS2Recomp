#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

struct Deci2Session
{
    uint16_t protocol = 0u;
    uint32_t opt = 0u;
    uint32_t handler = 0u;
    uint32_t userArea = 0u;
    bool locked = false;
};

struct Deci2RuntimeState
{
    void reset()
    {
        {
            std::lock_guard<std::mutex> lock(sessionMutex);
            sessions.clear();
            nextSocket = 1;
        }
        textLogCount.store(0u, std::memory_order_relaxed);
        unknownLogCount.store(0u, std::memory_order_relaxed);
    }

    std::mutex sessionMutex;
    std::unordered_map<int32_t, Deci2Session> sessions;
    int32_t nextSocket = 1;
    std::atomic<uint32_t> textLogCount{0u};
    std::atomic<uint32_t> unknownLogCount{0u};
};
