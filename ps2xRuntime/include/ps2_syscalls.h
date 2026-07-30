#ifndef PS2_SYSCALLS_H
#define PS2_SYSCALLS_H

#include "ps2_runtime.h"
#include "ps2_call_list.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

std::string translatePs2Path(
    const char *ps2Path,
    PS2Runtime *runtime);

namespace ps2_syscalls
{
    // Host-side reference for delayed EE-thread work. Guest syscalls continue
    // to use recyclable numeric IDs; queued host work must retain and validate
    // both generations before publishing into authoritative thread state.
    struct EeThreadHandle
    {
        uint64_t runtimeGeneration = 0u;
        uint32_t threadGeneration = 0u;
        int threadId = 0;

        constexpr explicit operator bool() const noexcept
        {
            return runtimeGeneration != 0u &&
                   threadGeneration != 0u &&
                   threadId > 0;
        }

        friend constexpr bool operator==(
            EeThreadHandle lhs,
            EeThreadHandle rhs) noexcept
        {
            return lhs.runtimeGeneration == rhs.runtimeGeneration &&
                   lhs.threadGeneration == rhs.threadGeneration &&
                   lhs.threadId == rhs.threadId;
        }

        friend constexpr bool operator!=(
            EeThreadHandle lhs,
            EeThreadHandle rhs) noexcept
        {
            return !(lhs == rhs);
        }
    };

    enum class EeThreadWakeResult : uint8_t
    {
        InvalidHandle,
        StaleHandle,
        Dormant,
        WakeupCounted,
        WokeSleeper,
    };

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
        int waitQueue = 0;
        int waitQueueId = 0;
        uint64_t stateRevision = 0u;
        bool waitCompletionPending = false;
        bool stateValid = false;
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
    void dispatchIntcHandlersForCause(
        uint8_t *rdram, PS2Runtime *runtime, uint32_t cause);
    bool isDmacCauseEnabled(
        PS2Runtime *runtime, uint32_t cause);
    bool isIntcCauseEnabled(
        PS2Runtime *runtime, uint32_t cause);
    void initializeGuestKernelState(
        uint8_t *rdram, PS2Runtime *runtime);
    void TODO(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t encodedSyscallId);
    void notifyRuntimeStop(PS2Runtime *runtime = nullptr);
    void joinAllGuestHostThreads(PS2Runtime *runtime);
    void detachAllGuestHostThreads(PS2Runtime *runtime);
    void EnsureVSyncScheduled(uint8_t *rdram, PS2Runtime *runtime);
    uint64_t GetCurrentVSyncTick(
        PS2Runtime *runtime);
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
    EeThreadHandle captureEeThreadHandle(
        PS2Runtime *runtime,
        int threadId);
    // Publishes only the wake transition and host notification. It never
    // dispatches a guest thread or runs a guest callback.
    EeThreadWakeResult publishEeThreadWake(
        PS2Runtime *runtime,
        EeThreadHandle handle);
    std::vector<GuestThreadDebugSnapshot> debugThreadSnapshots(
        PS2Runtime *runtime);
}

#endif // PS2_SYSCALLS_H
