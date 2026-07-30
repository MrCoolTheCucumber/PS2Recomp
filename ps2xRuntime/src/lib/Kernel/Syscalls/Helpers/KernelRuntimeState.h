#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ExitHandlerEntry
{
    uint32_t func = 0u;
    uint32_t arg = 0u;
};

// Guest-visible EE kernel configuration and allocation cursors belong to one
// runtime. These locks protect host publication and lookup only; guest code
// must never run while one is held.
struct EeKernelRuntimeState
{
    std::mutex exitHandlerMutex;
    std::unordered_map<
        int,
        std::vector<ExitHandlerEntry>>
        exitHandlers;

    std::mutex bootModeMutex;
    bool bootModeInitialized = false;
    uint32_t bootModePoolOffset = 0u;
    std::unordered_map<uint8_t, uint32_t>
        bootModeAddresses;

    std::mutex syscallOverrideMutex;
    std::unordered_map<uint32_t, uint32_t>
        syscallOverrides;
    std::unordered_set<uint32_t>
        syscallMirrorAddresses;
    std::mutex activeSyscallOverrideMutex;
    std::vector<uint32_t> activeSyscallOverrides;

    std::mutex tlsMutex;
    uint32_t tlsIndex = 0u;

    std::mutex osdMutex;
    bool osdConfigInitialized = false;
    uint32_t osdConfigRaw = 0u;
    uint32_t osdConfig2Raw = 0u;
};
