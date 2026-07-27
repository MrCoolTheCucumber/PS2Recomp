#ifndef PS2_SYSCALLS_H
#define PS2_SYSCALLS_H

#include "ps2_runtime.h"
#include "ps2_call_list.h"
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

std::string translatePs2Path(const char *ps2Path);

extern std::atomic<int> g_activeThreads;

inline std::mutex g_sys_fd_mutex;

namespace ps2_syscalls
{
    struct GuestThreadDebugSnapshot
    {
        int id = 0;
        uint32_t entry = 0u;
        uint32_t stack = 0u;
        uint32_t stackSize = 0u;
        uint32_t gp = 0u;
        uint32_t pc = 0u;
        int status = 0;
        int waitType = 0;
        int waitId = 0;
        int wakeupCount = 0;
        int currentPriority = 0;
        int suspendCount = 0;
        bool started = false;
        bool forceRelease = false;
        bool terminated = false;
    };

#define PS2_DECLARE_SYSCALL(name) void name(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    PS2_SYSCALL_LIST(PS2_DECLARE_SYSCALL)
#undef PS2_DECLARE_SYSCALL

    void iDeleteSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void EnableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void DisableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void EnableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void DisableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);

    bool dispatchNumericSyscall(uint32_t syscallNumber, uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void dispatchDmacHandlersForCause(uint8_t *rdram, PS2Runtime *runtime, uint32_t cause);
    bool isDmacCauseEnabled(uint32_t cause);
    void initializeGuestKernelState(uint8_t *rdram);
    void TODO(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t encodedSyscallId);
    void notifyRuntimeStop();
    void joinAllGuestHostThreads();
    void detachAllGuestHostThreads();
    void EnsureVSyncWorkerRunning(uint8_t *rdram, PS2Runtime *runtime);
    uint64_t GetCurrentVSyncTick();
    uint64_t PublishVSyncStart(uint8_t *rdram, PS2Runtime *runtime);
    void PublishVSyncField(PS2Runtime *runtime, uint64_t tick);
    void DispatchVSyncStartHandlers(
        uint8_t *rdram, PS2Runtime *runtime, uint64_t tick);
    void DispatchVSyncEndHandlers(
        uint8_t *rdram, PS2Runtime *runtime);
    uint64_t WaitForNextVSyncTick(
        uint8_t *rdram,
        PS2Runtime *runtime,
        R5900Context *ctx = nullptr);
    void WaitVSyncTick(uint8_t *rdram, PS2Runtime *runtime);
    std::vector<GuestThreadDebugSnapshot> debugThreadSnapshots();
}

#endif // PS2_SYSCALLS_H
