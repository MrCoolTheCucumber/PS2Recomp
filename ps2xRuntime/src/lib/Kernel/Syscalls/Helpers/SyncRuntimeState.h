#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

struct SemaInfo
{
    int count = 0;
    int maxCount = 0;
    int initCount = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    int waiters = 0;
    bool deleted = false;
    std::mutex m;
    std::condition_variable cv;
};

struct EventFlagInfo
{
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t initBits = 0;
    uint32_t bits = 0;
    int waiters = 0;
    bool deleted = false;
    std::mutex m;
    std::condition_variable cv;
};

// Guest-visible synchronization objects and recyclable IDs belong to one
// runtime. Object-local mutexes protect wait state; these registry mutexes
// protect identity/allocation and are never held while entering guest code.
struct EeSyncRuntimeState
{
    std::mutex semaMapMutex;
    std::unordered_map<int, std::shared_ptr<SemaInfo>> semas;
    int nextSemaId = 0;

    std::mutex eventFlagMapMutex;
    std::unordered_map<int, std::shared_ptr<EventFlagInfo>> eventFlags;
    int nextEventFlagId = 1;
};
