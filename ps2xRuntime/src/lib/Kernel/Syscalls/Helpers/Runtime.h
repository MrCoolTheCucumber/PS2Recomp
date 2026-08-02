#include "InterruptRuntimeState.h"
#include "runtime/ee_guest_exceptions.h"

static void throwIfTerminated(const std::shared_ptr<ThreadInfo> &info)
{
    if (info && info->terminated.load())
    {
        throw ThreadExitException();
    }
}

// A legacy host worker may observe READY, SUSPEND, or a retained wait
// completion before it acquires the one guest-execution boundary. Publish RUN
// only after that boundary is actually owned, and re-check suspension there
// so an external suspend cannot race a worker's earlier host-side test.
static bool publishRunningAtGuestBoundary(
    const std::shared_ptr<ThreadInfo> &info)
{
    if (!info)
    {
        return true;
    }

    std::lock_guard<std::mutex> lock(info->m);
    if (info->terminated.load())
    {
        throw ThreadExitException();
    }

    const EeThreadGuestStateSnapshot &guest =
        info->guestState.snapshot();
    if (guest.suspendCount != 0 ||
        guest.status == EeThreadGuestState::kSuspend ||
        guest.status == EeThreadGuestState::kWaitSuspend ||
        guest.status == EeThreadGuestState::kWait ||
        guest.status == EeThreadGuestState::kDormant)
    {
        return false;
    }
    if (guest.status == EeThreadGuestState::kReady)
    {
        info->guestState.publishRunning();
    }
    return true;
}

// Condition-variable waits in the EE runtime must release the global guest
// execution mutex, but must not reacquire it while still holding the local
// wait-object mutex. Reacquiring guest execution while holding a semaphore,
// thread, event-flag, or vsync mutex can create an ABBA deadlock:
//
//   awakened thread: local wait mutex -> waiting for GuestExecutionScope
//   running thread:  GuestExecutionScope -> waiting for local wait mutex
//
// This helper keeps guest execution released for the whole host wait and for
// any post-wake bookkeeping that needs the local mutex, then unlocks the local
// mutex before the GuestExecutionReleaseScope destructor reacquires guest code.
template <typename Lock, typename WaitFn, typename FinishFn>
static void waitWithGuestExecutionReleasedUntilUnlocked(PS2Runtime *runtime,
                                                        Lock &lock,
                                                        WaitFn waitFn,
                                                        FinishFn finishFn)
{
    auto releaseGuestExecution = std::make_unique<PS2Runtime::GuestExecutionReleaseScope>(runtime);

    waitFn();
    finishFn();

    if (lock.owns_lock())
    {
        lock.unlock();
    }

    releaseGuestExecution.reset();
}

template <typename Lock, typename WaitFn>
static void waitWithGuestExecutionReleasedUntilUnlocked(PS2Runtime *runtime, Lock &lock, WaitFn waitFn)
{
    waitWithGuestExecutionReleasedUntilUnlocked(runtime, lock, waitFn, []() {});
}

static void waitWhileSuspended(const std::shared_ptr<ThreadInfo> &info, PS2Runtime *runtime = nullptr)
{
    if (!info)
        return;

    if (runtime && runtime->usesDedicatedEeExecutor())
    {
        std::lock_guard<std::mutex> lock(info->m);
        if (info->terminated.load())
        {
            throw ThreadExitException();
        }
        const EeThreadGuestStateSnapshot &guest =
            info->guestState.snapshot();
        if (guest.suspendCount != 0 ||
            guest.status == EeThreadGuestState::kSuspend ||
            guest.status == EeThreadGuestState::kWaitSuspend)
        {
            throw std::logic_error(
                "the EE scheduler resumed a suspended "
                "dedicated continuation");
        }
        return;
    }

    std::unique_lock<std::mutex> lock(info->m);
    if (info->guestState.snapshot().suspendCount > 0)
    {
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
    }
}

static std::shared_ptr<ThreadInfo> lookupThreadInfo(
    PS2Runtime *runtime,
    int tid)
{
    if (!runtime)
    {
        return nullptr;
    }

    EeThreadRuntimeState &state =
        runtime->eeThreadRuntimeState();
    std::lock_guard<std::mutex> lock(
        state.threadMapMutex);
    auto it = state.threads.find(tid);
    if (it != state.threads.end())
    {
        return it->second;
    }
    return nullptr;
}

static int getCurrentThreadId(PS2Runtime *runtime)
{
    return runtime
               ? runtime->currentEeThreadId()
               : g_currentThreadId;
}

static std::shared_ptr<ThreadInfo> ensureCurrentThreadInfo(
    PS2Runtime *runtime,
    R5900Context *ctx)
{
    if (!runtime)
    {
        return nullptr;
    }

    const int tid = getCurrentThreadId(runtime);
    EeThreadRuntimeState &state =
        runtime->eeThreadRuntimeState();
    std::lock_guard<std::mutex> lock(
        state.threadMapMutex);
    auto it = state.threads.find(tid);
    if (it != state.threads.end())
    {
        if (ctx && !it->second->boundContext)
        {
            it->second->boundContext =
                tid == 1 ? &runtime->cpu() : ctx;
            state.contextThreadIds[
                it->second->boundContext] = tid;
        }
        return it->second;
    }

    auto info = std::make_shared<ThreadInfo>();
    info->generation =
        state.allocateThreadGenerationLocked();
    info->priority = 0u;
    info->guestState.initializeRunning(
        static_cast<int>(info->priority));
    if (ctx)
    {
        info->entry = ctx->pc;
        info->stack = getRegU32(ctx, 29);
        info->gp = getRegU32(ctx, 28);
        info->boundContext =
            tid == 1 ? &runtime->cpu() : ctx;
        state.contextThreadIds[
            info->boundContext] = tid;
    }

    state.threads.emplace(tid, info);
    return info;
}

static std::shared_ptr<SemaInfo> lookupSemaInfo(
    PS2Runtime *runtime,
    int sid)
{
    if (!runtime)
    {
        return nullptr;
    }

    EeSyncRuntimeState &state =
        runtime->eeSyncRuntimeState();
    std::lock_guard<std::mutex> lock(
        state.semaMapMutex);
    auto it = state.semas.find(sid);
    if (it != state.semas.end())
    {
        return it->second;
    }
    return nullptr;
}

static std::shared_ptr<EventFlagInfo> lookupEventFlagInfo(
    PS2Runtime *runtime,
    int eid)
{
    if (!runtime)
    {
        return nullptr;
    }

    EeSyncRuntimeState &state =
        runtime->eeSyncRuntimeState();
    std::lock_guard<std::mutex> lock(
        state.eventFlagMapMutex);
    auto it = state.eventFlags.find(eid);
    if (it != state.eventFlags.end())
    {
        return it->second;
    }
    return nullptr;
}

static void setRegU32(R5900Context *ctx, int reg, uint32_t value)
{
    if (!ctx || reg < 0 || reg > 31)
        return;
    SET_GPR_U32(ctx, reg, value);
}

static std::chrono::microseconds alarmTicksToDuration(uint16_t ticks)
{
    constexpr uint64_t kAlarmTickUsec = 64u; // Approximate EE H-SYNC tick period.
    const uint64_t clampedTicks = (ticks == 0u) ? 1u : static_cast<uint64_t>(ticks);
    return std::chrono::microseconds(clampedTicks * kAlarmTickUsec);
}

static bool ensureAlarmWorkerRunning(PS2Runtime *runtime)
{
    if (!runtime)
    {
        return false;
    }

    EeAlarmRuntimeState &state =
        runtime->eeAlarmRuntimeState();
    std::unique_lock<std::mutex> lifecycleLock(state.mutex);
    if (state.stopRequested.load(
            std::memory_order_acquire))
    {
        return false;
    }
    if (state.worker.joinable())
    {
        if (state.workerRunning)
        {
            return true;
        }

        // Reap a completed publication worker before a later run starts a
        // replacement.
        std::thread stoppedWorker =
            std::move(state.worker);
        lifecycleLock.unlock();
        stoppedWorker.join();
        lifecycleLock.lock();

        if (state.stopRequested.load(
                std::memory_order_acquire))
        {
            return false;
        }
        if (state.worker.joinable())
        {
            return state.workerRunning;
        }
    }

    state.workerRunning = true;
    try
    {
        state.worker = std::thread([runtime, &state]()
        {
            struct WorkerExit
            {
                EeAlarmRuntimeState &state;

                ~WorkerExit()
                {
                    {
                        std::lock_guard<std::mutex> lock(
                            state.mutex);
                        state.workerRunning = false;
                    }
                    state.cv.notify_all();
                }
            } workerExit{state};

            for (;;)
            {
                std::shared_ptr<AlarmInfo> readyAlarm;
                {
                    std::unique_lock<std::mutex> lock(state.mutex);
                    while (!state.stopRequested.load(
                               std::memory_order_acquire) &&
                           !readyAlarm)
                    {
                        if (state.alarms.empty())
                        {
                            state.cv.wait(lock);
                            continue;
                        }

                        const auto nextIt = std::min_element(
                            state.alarms.begin(),
                            state.alarms.end(),
                            [](const auto &a, const auto &b)
                            {
                                return a.second->dueAt <
                                       b.second->dueAt;
                            });
                        if (nextIt == state.alarms.end())
                        {
                            state.cv.wait(lock);
                            continue;
                        }

                        const auto now =
                            std::chrono::steady_clock::now();
                        const auto dueAt =
                            nextIt->second->dueAt;
                        if (dueAt > now)
                        {
                            // Keep the deadline by value. Cancellation may
                            // erase the final shared owner as soon as
                            // wait_until unlocks.
                            state.cv.wait_until(lock, dueAt);
                            continue;
                        }

                        readyAlarm = nextIt->second;
                        state.alarms.erase(nextIt);
                    }

                    if (state.stopRequested.load(
                            std::memory_order_acquire))
                    {
                        return;
                    }
                }

                if (!readyAlarm ||
                    runtime->isStopRequested())
                {
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(
                        state.mutex);
                    if (state.stopRequested.load(
                            std::memory_order_acquire))
                    {
                        return;
                    }
                    state.pendingCallbacks.push_back(
                        std::move(readyAlarm));
                    state.pendingCallbackPublished.store(
                        true, std::memory_order_release);
                }
                runtime->requestGuestPreemption();
            }
        });
    }
    catch (...)
    {
        state.workerRunning = false;
        return false;
    }
    return true;
}

static void stopAlarmWorker(PS2Runtime *runtime)
{
    if (!runtime)
    {
        return;
    }

    EeAlarmRuntimeState &state =
        runtime->eeAlarmRuntimeState();
    std::thread workerToJoin;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.stopRequested.store(
            true, std::memory_order_release);
        state.alarms.clear();
        state.pendingCallbacks.clear();
        state.pendingCallbackPublished.store(
            false, std::memory_order_release);
        state.nextAlarmId = 1;
        state.setLogCount = 0u;
        if (state.worker.joinable() &&
            state.worker.get_id() !=
                 std::this_thread::get_id())
        {
            workerToJoin = std::move(state.worker);
        }
    }
    state.cv.notify_all();

    if (workerToJoin.joinable())
    {
        workerToJoin.join();
    }

    // Callback continuation state belongs to the EE execution boundary, not
    // the timer worker. It may still be live when an external stop joins this
    // publication worker, so leave it intact until the runtime is destroyed
    // or a later callback initializes it.
}

static void rpcCopyToRdram(uint8_t *rdram, uint32_t dst, uint32_t src, size_t size)
{
    if (!rdram || size == 0)
        return;

    ps2TraceGuestRangeWrite(rdram, dst, static_cast<uint32_t>(size), "rpcCopyToRdram", nullptr);

    constexpr size_t kMaxRpcTransferBytes = 1u * 1024u * 1024u;
    const size_t clampedSize = std::min(size, kMaxRpcTransferBytes);
    if (clampedSize != size)
    {
        static uint32_t warnCount = 0;
        if (warnCount < 8)
        {
            std::cerr << "[SifCallRpc] clamping copy size from " << size
                      << " to " << clampedSize
                      << " bytes (dst=0x" << std::hex << dst
                      << " src=0x" << src << std::dec << ")" << std::endl;
            ++warnCount;
        }
    }

    for (size_t i = 0; i < clampedSize; ++i)
    {
        const uint32_t dstAddr = dst + static_cast<uint32_t>(i);
        const uint32_t srcAddr = src + static_cast<uint32_t>(i);
        uint8_t *dstPtr = getMemPtr(rdram, dstAddr);
        const uint8_t *srcPtr = getConstMemPtr(rdram, srcAddr);
        if (!dstPtr || !srcPtr)
        {
            break;
        }
        *dstPtr = *srcPtr;
    }
}

static void rpcZeroRdram(uint8_t *rdram, uint32_t dst, size_t size)
{
    if (!rdram || size == 0)
        return;

    ps2TraceGuestRangeWrite(rdram, dst, static_cast<uint32_t>(size), "rpcZeroRdram", nullptr);

    constexpr size_t kMaxRpcTransferBytes = 1u * 1024u * 1024u;
    const size_t clampedSize = std::min(size, kMaxRpcTransferBytes);
    if (clampedSize != size)
    {
        static uint32_t warnCount = 0;
        if (warnCount < 8)
        {
            std::cerr << "[SifCallRpc] clamping zero size from " << size
                      << " to " << clampedSize
                      << " bytes (dst=0x" << std::hex << dst << std::dec << ")" << std::endl;
            ++warnCount;
        }
    }

    for (size_t i = 0; i < clampedSize; ++i)
    {
        const uint32_t dstAddr = dst + static_cast<uint32_t>(i);
        uint8_t *dstPtr = getMemPtr(rdram, dstAddr);
        if (!dstPtr)
        {
            break;
        }
        *dstPtr = 0;
    }
}

static bool readStackU32(uint8_t *rdram, uint32_t sp, uint32_t offset, uint32_t &out)
{
    uint8_t *ptr = getMemPtr(rdram, sp + offset);
    if (!ptr)
        return false;
    out = *reinterpret_cast<uint32_t *>(ptr);
    return true;
}

enum class RpcInvokeExitReason
{
    Returned,
    NullPc,
    MissingFunction,
    StepLimit,
    SamePcLimit
};

static const char *rpcInvokeExitReasonName(RpcInvokeExitReason reason)
{
    switch (reason)
    {
    case RpcInvokeExitReason::Returned:
        return "returned";
    case RpcInvokeExitReason::NullPc:
        return "null-pc";
    case RpcInvokeExitReason::MissingFunction:
        return "missing-function";
    case RpcInvokeExitReason::StepLimit:
        return "step-limit";
    case RpcInvokeExitReason::SamePcLimit:
        return "same-pc-limit";
    default:
        return "unknown";
    }
}

static bool rpcInvokeFunction(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
                              uint32_t funcAddr, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t *outV0)
{
    if (!runtime || !ctx || !funcAddr || !runtime->hasFunction(funcAddr))
        return false;

    constexpr uint32_t kRpcInvokeReturnSentinel = 0x00FFF000u;
    constexpr uint32_t kRpcInvokeMaxSteps = 0x8000u;

    EeAsyncCallbackContextLease callbackContext(
        runtime->eeInterruptRuntimeState());
    R5900Context &tmp = callbackContext.context();
    tmp = *ctx;
    setRegU32(&tmp, 4, a0);
    setRegU32(&tmp, 5, a1);
    setRegU32(&tmp, 6, a2);
    setRegU32(&tmp, 7, a3);

    uint32_t steps = 0u;
    uint32_t lastPc = 0xFFFFFFFFu;
    uint32_t samePcCount = 0u;
    RpcInvokeExitReason exitReason = RpcInvokeExitReason::MissingFunction;
    const int continuationThreadId =
        runtime->currentEeThreadId();
    {
        PS2Runtime::AsyncCallbackInvocationScope
            callbackExecution(
                runtime,
                &tmp,
                ctx,
                continuationThreadId);
        setRegU32(
            &tmp, 29,
            callbackExecution.stackTop());
        setRegU32(&tmp, 31, kRpcInvokeReturnSentinel);
        tmp.pc = funcAddr;

        while (tmp.pc != 0u &&
               tmp.pc != kRpcInvokeReturnSentinel &&
               runtime->hasFunction(tmp.pc) &&
               steps < kRpcInvokeMaxSteps)
        {
            const uint32_t pc = tmp.pc;
            if (pc == lastPc)
            {
                ++samePcCount;
                if (samePcCount > 0x2000u)
                {
                    exitReason = RpcInvokeExitReason::SamePcLimit;
                    break;
                }
            }
            else
            {
                lastPc = pc;
                samePcCount = 0u;
            }

            PS2Runtime::RecompiledFunction func =
                runtime->lookupFunction(pc);
            runtime->executeGuestStep(rdram, &tmp, func);
            ++steps;
        }
    }

    if (outV0)
    {
        *outV0 = getRegU32(&tmp, 2);
    }

    if (tmp.pc == kRpcInvokeReturnSentinel)
    {
        return true;
    }

    if (tmp.pc == 0u)
    {
        exitReason = RpcInvokeExitReason::NullPc;
    }
    else if (steps >= kRpcInvokeMaxSteps)
    {
        exitReason = RpcInvokeExitReason::StepLimit;
    }
    else if (!runtime->hasFunction(tmp.pc))
    {
        exitReason = RpcInvokeExitReason::MissingFunction;
    }

    static std::atomic<uint32_t> s_rpcInvokeFailureLogs{0u};
    constexpr uint32_t kMaxRpcInvokeFailureLogs = 64u;
    const uint32_t logIndex = s_rpcInvokeFailureLogs.fetch_add(1u, std::memory_order_relaxed);
    if (logIndex < kMaxRpcInvokeFailureLogs)
    {
        PS2_IF_AGRESSIVE_LOGS({
            std::cerr << "[SyscallOverride:invoke-failed]"
                      << " func=0x" << std::hex << funcAddr
                      << " exitPc=0x" << tmp.pc
                      << " ra=0x" << getRegU32(&tmp, 31)
                      << std::dec
                      << " steps=" << steps
                      << " reason=" << rpcInvokeExitReasonName(exitReason)
                      << std::endl;
        });
    }

    return false;
}

static uint32_t rpcAllocPacketHandle(PS2Runtime *runtime)
{
    if (!runtime || kRpcPacketHandleCount == 0)
        return 0;

    EeRpcRuntimeState &state =
        runtime->eeRpcRuntimeState();
    std::lock_guard<std::mutex> lock(state.rpcMutex);
    uint32_t slot =
        state.packetIndex++ % kRpcPacketHandleCount;
    return kRpcPacketHandleBase + (slot * kRpcPacketSize);
}

static uint32_t rpcAllocServerHandle(PS2Runtime *runtime)
{
    if (!runtime || kRpcServerHandleCount == 0)
        return 0;

    EeRpcRuntimeState &state =
        runtime->eeRpcRuntimeState();
    std::lock_guard<std::mutex> lock(state.rpcMutex);
    uint32_t slot =
        state.serverIndex++ % kRpcServerHandleCount;
    return kRpcServerHandleBase + (slot * kRpcServerStride);
}

inline std::string translatePs2Path(
    const char *ps2Path,
    PS2Runtime *runtime)
{
    if (!runtime || !ps2Path || !*ps2Path)
    {
        return {};
    }

    std::string pathStr(ps2Path);
    std::string lower = toLowerAscii(pathStr);
    const GuestPathResolverSnapshot resolver =
        guestPathResolverSnapshot(runtime);

    auto resolveWithBase = [&](const std::filesystem::path &base, const std::string &suffix) -> std::string
    {
        const std::string normalizedSuffix = normalizePs2PathSuffix(suffix);
        std::filesystem::path resolved = base;
        if (!normalizedSuffix.empty())
        {
            resolved /= std::filesystem::path(normalizedSuffix);
        }
        return resolved.lexically_normal().string();
    };
    auto resolveDevicePath =
        [&](const std::filesystem::path &root,
            const std::filesystem::path &cwd,
            const std::string &suffix) -> std::string
    {
        return resolveWithBase(
            guestPathSuffixIsAbsolute(suffix)
                ? root
                : cwd,
            suffix);
    };

    if (lower.rfind("host0:", 0) == 0 || lower.rfind("host:", 0) == 0)
    {
        const std::size_t prefixLength = (lower.rfind("host0:", 0) == 0) ? 6 : 5;
        return resolveDevicePath(
            resolver.hostBase,
            resolver.hostCwd,
            pathStr.substr(prefixLength));
    }

    if (lower.rfind("cdrom0:", 0) == 0 || lower.rfind("cdrom:", 0) == 0)
    {
        const std::size_t prefixLength = (lower.rfind("cdrom0:", 0) == 0) ? 7 : 6;
        return resolveDevicePath(
            resolver.cdromBase,
            resolver.cdromCwd,
            pathStr.substr(prefixLength));
    }

    if (lower.rfind(kMc0Prefix, 0) == 0)
    {
        const std::size_t prefixLength = sizeof(kMc0Prefix) - 1;
        return resolveDevicePath(
            resolver.mcBase,
            resolver.mcCwd,
            pathStr.substr(prefixLength));
    }

    if (!pathStr.empty() && (pathStr.front() == '/' || pathStr.front() == '\\'))
    {
        if (resolver.cwdDevice == "host0")
        {
            return resolveWithBase(
                resolver.hostBase, pathStr);
        }
        if (resolver.cwdDevice == "mc0")
        {
            return resolveWithBase(
                resolver.mcBase, pathStr);
        }
        return resolveWithBase(
            resolver.cdromBase, pathStr);
    }

    if (pathStr.size() > 1 && pathStr[1] == ':')
    {
        return pathStr;
    }

    if (resolver.cwdDevice == "host0")
    {
        return resolveWithBase(
            resolver.hostCwd, pathStr);
    }
    if (resolver.cwdDevice == "mc0")
    {
        return resolveWithBase(
            resolver.mcCwd, pathStr);
    }
    return resolveWithBase(
        resolver.cdromCwd, pathStr);
}

static bool localtimeSafe(const std::time_t *t, std::tm *out)
{
#ifdef _WIN32
    return localtime_s(out, t) == 0;
#else
    return localtime_r(t, out) != nullptr;
#endif
}

static void encodePs2Time(std::time_t t, uint8_t out[8])
{
    std::tm tm{};
    if (!localtimeSafe(&t, &tm))
    {
        std::memset(out, 0, 8);
        return;
    }

    uint16_t year = static_cast<uint16_t>(tm.tm_year + 1900);
    out[0] = 0;
    out[1] = static_cast<uint8_t>(tm.tm_sec);
    out[2] = static_cast<uint8_t>(tm.tm_min);
    out[3] = static_cast<uint8_t>(tm.tm_hour);
    out[4] = static_cast<uint8_t>(tm.tm_mday);
    out[5] = static_cast<uint8_t>(tm.tm_mon + 1);
    out[6] = static_cast<uint8_t>(year & 0xFF);
    out[7] = static_cast<uint8_t>((year >> 8) & 0xFF);
}

static std::time_t fileTimeToTimeT(std::filesystem::file_time_type ft)
{
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ft - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
    return std::chrono::system_clock::to_time_t(sctp);
}

static bool gmtimeSafe(const std::time_t *t, std::tm *out)
{
#ifdef _WIN32
    return gmtime_s(out, t) == 0;
#else
    return gmtime_r(t, out) != nullptr;
#endif
}

static int getTimezoneOffsetMinutes()
{
    std::time_t now = std::time(nullptr);
    std::tm local{};
    std::tm gmt{};
    if (!localtimeSafe(&now, &local) || !gmtimeSafe(&now, &gmt))
        return 0;

    std::time_t localTime = std::mktime(&local);
    std::time_t gmtTime = std::mktime(&gmt);
    if (localTime == static_cast<std::time_t>(-1) || gmtTime == static_cast<std::time_t>(-1))
        return 0;

    double diff = std::difftime(localTime, gmtTime);
    return static_cast<int>(diff / 60.0);
}

static uint32_t packOsdConfig(uint32_t spdifMode, uint32_t screenType, uint32_t videoOutput,
                              uint32_t japLanguage, uint32_t ps1drvConfig, uint32_t version,
                              uint32_t language, int timezoneOffset)
{
    uint32_t raw = 0;
    raw |= (spdifMode & 0x1) << 0;
    raw |= (screenType & 0x3) << 1;
    raw |= (videoOutput & 0x1) << 3;
    raw |= (japLanguage & 0x1) << 4;
    raw |= (ps1drvConfig & 0xFF) << 5;
    raw |= (version & 0x7) << 13;
    raw |= (language & 0x1F) << 16;
    raw |= (static_cast<uint32_t>(timezoneOffset) & 0x7FF) << 21;
    return raw;
}

static uint32_t packOsdConfig2(uint32_t format, uint32_t daylightSaving, uint32_t timeFormat,
                               uint32_t dateFormat, uint32_t version, uint32_t language)
{
    const uint32_t flags =
        ((daylightSaving & 0x1u) << 4) |
        ((timeFormat & 0x1u) << 5) |
        ((dateFormat & 0x3u) << 6);

    return (format & 0xFFu) |
           ((flags & 0xFFu) << 8) |
           ((version & 0xFFu) << 16) |
           ((language & 0xFFu) << 24);
}

static int decodeTimezoneOffset(uint32_t raw)
{
    int tz = static_cast<int>((raw >> 21) & 0x7FF);
    if (tz & 0x400)
        tz |= ~0x7FF;
    return tz;
}

static int clampTimezoneOffset(int tz)
{
    if (tz < -1024)
        return -1024;
    if (tz > 1023)
        return 1023;
    return tz;
}

static uint32_t sanitizeOsdConfigRaw(uint32_t raw)
{
    uint32_t spdifMode = raw & 0x1;
    uint32_t screenType = (raw >> 1) & 0x3;
    if (screenType > 2)
        screenType = 0;
    uint32_t videoOutput = (raw >> 3) & 0x1;
    uint32_t japLanguage = (raw >> 4) & 0x1;
    uint32_t ps1drvConfig = (raw >> 5) & 0xFF;
    uint32_t version = (raw >> 13) & 0x7;
    if (version > 2)
        version = 1;
    uint32_t language = (raw >> 16) & 0x1F;
    int tz = clampTimezoneOffset(decodeTimezoneOffset(raw));
    return packOsdConfig(spdifMode, screenType, videoOutput, japLanguage, ps1drvConfig, version, language, tz);
}

static uint32_t sanitizeOsdConfig2Raw(uint32_t raw)
{
    const uint32_t format = raw & 0xFFu;
    const uint32_t flags = (raw >> 8) & 0xFFu;
    const uint32_t daylightSaving = (flags >> 4) & 0x1u;
    const uint32_t timeFormat = (flags >> 5) & 0x1u;
    uint32_t dateFormat = (flags >> 6) & 0x3u;
    if (dateFormat > 2u)
        dateFormat = 0u;

    uint32_t version = (raw >> 16) & 0xFFu;
    if (version > 2u)
        version = 1u;

    const uint32_t language = (raw >> 24) & 0xFFu;
    return packOsdConfig2(format, daylightSaving, timeFormat, dateFormat, version, language);
}

static uint32_t syncOsdConfigRawVersionLanguage(uint32_t raw, uint32_t version, uint32_t language)
{
    raw &= ~((0x7u << 13) | (0x1Fu << 16));
    raw |= (version & 0x7u) << 13;
    raw |= (language & 0x1Fu) << 16;
    return sanitizeOsdConfigRaw(raw);
}

static uint32_t makeReadableOsdConfig2RawLocked(
    const EeKernelRuntimeState &state)
{
    const uint32_t version =
        (state.osdConfigRaw >> 13) & 0x7u;
    const uint32_t language =
        (state.osdConfigRaw >> 16) & 0x1Fu;
    uint32_t raw =
        state.osdConfig2Raw & 0x0000FFFFu;
    raw |= (version & 0xFFu) << 16;
    raw |= (language & 0xFFu) << 24;
    return sanitizeOsdConfig2Raw(raw);
}

static bool ensureOsdConfigInitialized(
    PS2Runtime *runtime)
{
    if (!runtime)
    {
        return false;
    }

    EeKernelRuntimeState &state =
        runtime->eeKernelRuntimeState();
    std::lock_guard<std::mutex> lock(state.osdMutex);
    if (state.osdConfigInitialized)
    {
        return true;
    }

    int tz = clampTimezoneOffset(getTimezoneOffsetMinutes());
    uint32_t spdifMode = 1;   // disabled
    uint32_t screenType = 0;  // 4:3
    uint32_t videoOutput = 0; // RGB
    uint32_t japLanguage = 1; // non-japanese
    uint32_t ps1drvConfig = 0;
    uint32_t version = 1;  // OSD2
    uint32_t language = 1; // English
    state.osdConfigRaw = packOsdConfig(
        spdifMode, screenType, videoOutput,
        japLanguage, ps1drvConfig, version,
        language, tz);
    state.osdConfig2Raw = packOsdConfig2(
        0, 0, 0, 0, version, language);
    state.osdConfigInitialized = true;
    return true;
}

static uint32_t allocTlsAddr(
    uint8_t *rdram, PS2Runtime *runtime)
{
    if (!rdram || !runtime || kTlsPoolCount == 0)
        return 0;

    EeKernelRuntimeState &state =
        runtime->eeKernelRuntimeState();
    std::lock_guard<std::mutex> lock(state.tlsMutex);
    uint32_t slot =
        state.tlsIndex++ % kTlsPoolCount;
    uint32_t addr = kTlsPoolBase + (slot * kTlsBlockSize);
    rpcZeroRdram(rdram, addr, kTlsBlockSize);
    return addr;
}

static uint32_t allocBootModeAddr(
    uint8_t *rdram,
    EeKernelRuntimeState &state,
    size_t bytes)
{
    if (!rdram)
        return 0;

    size_t aligned = (bytes + 15u) & ~15u;
    if (state.bootModePoolOffset + aligned >
        kBootModePoolBytes)
        return 0;

    uint32_t addr =
        kBootModePoolBase + state.bootModePoolOffset;
    state.bootModePoolOffset +=
        static_cast<uint32_t>(aligned);
    rpcZeroRdram(rdram, addr, aligned);
    return addr;
}

static uint32_t createBootModeEntry(
    uint8_t *rdram,
    EeKernelRuntimeState &state,
    uint8_t id,
    uint16_t value,
    uint8_t lenField,
    const uint32_t *data,
    uint8_t dataCount)
{
    uint8_t allocCount = (dataCount == 0) ? 1 : dataCount;
    size_t bytes = static_cast<size_t>(1 + allocCount) * sizeof(uint32_t);
    uint32_t addr =
        allocBootModeAddr(rdram, state, bytes);
    if (!addr)
        return 0;

    uint32_t header = (static_cast<uint32_t>(lenField) << 24) |
                      (static_cast<uint32_t>(id) << 16) |
                      (static_cast<uint32_t>(value) & 0xFFFFu);

    uint32_t *dst = reinterpret_cast<uint32_t *>(getMemPtr(rdram, addr));
    if (!dst)
        return 0;

    dst[0] = header;
    for (uint8_t i = 0; i < allocCount; ++i)
    {
        dst[1 + i] = (data && i < dataCount) ? data[i] : 0;
    }

    return addr;
}

static bool ensureBootModeTable(
    uint8_t *rdram, PS2Runtime *runtime)
{
    if (!runtime)
    {
        return false;
    }

    EeKernelRuntimeState &state =
        runtime->eeKernelRuntimeState();
    std::lock_guard<std::mutex> lock(
        state.bootModeMutex);
    if (state.bootModeInitialized)
    {
        return true;
    }

    state.bootModePoolOffset = 0u;
    state.bootModeAddresses.clear();

    const uint32_t boot3Data[1] = {0};
    const uint32_t boot5Data[1] = {0};

    state.bootModeAddresses[1] =
        createBootModeEntry(
            rdram, state, 1, 0, 0, nullptr, 0);
    state.bootModeAddresses[3] =
        createBootModeEntry(
            rdram, state, 3, 0, 1, boot3Data, 1);
    state.bootModeAddresses[4] =
        createBootModeEntry(
            rdram, state, 4, 0, 0, nullptr, 0);
    state.bootModeAddresses[5] =
        createBootModeEntry(
            rdram, state, 5, 0, 1, boot5Data, 1);
    state.bootModeAddresses[6] =
        createBootModeEntry(
            rdram, state, 6, 0, 0, nullptr, 0);
    state.bootModeAddresses[7] =
        createBootModeEntry(
            rdram, state, 7, 0, 0, nullptr, 0);

    state.bootModeInitialized = true;
    return true;
}
