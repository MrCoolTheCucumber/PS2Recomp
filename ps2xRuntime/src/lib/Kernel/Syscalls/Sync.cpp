#include "Common.h"
#include "Sync.h"

namespace ps2_syscalls
{
    static bool looksLikeGuestPointerOrNull(uint32_t value)
    {
        if (value == 0u)
        {
            return true;
        }
        const uint32_t normalized = value & 0x1FFFFFFFu;
        return normalized < PS2_RAM_SIZE;
    }

    static bool readGuestU32Safe(const uint8_t *rdram, uint32_t addr, uint32_t &out)
    {
        const uint8_t *b0 = getConstMemPtr(rdram, addr + 0u);
        const uint8_t *b1 = getConstMemPtr(rdram, addr + 1u);
        const uint8_t *b2 = getConstMemPtr(rdram, addr + 2u);
        const uint8_t *b3 = getConstMemPtr(rdram, addr + 3u);
        if (!b0 || !b1 || !b2 || !b3)
        {
            out = 0u;
            return false;
        }

        out = static_cast<uint32_t>(*b0) |
              (static_cast<uint32_t>(*b1) << 8) |
              (static_cast<uint32_t>(*b2) << 16) |
              (static_cast<uint32_t>(*b3) << 24);
        return true;
    }

    struct DecodedSemaParams
    {
        int init = 0;
        int max = 1;
        uint32_t attr = 0;
        uint32_t option = 0;
    };

    static DecodedSemaParams decodeCreateSemaParams(const uint32_t *param, uint32_t availableWords)
    {
        DecodedSemaParams out{};
        if (!param || availableWords == 0u)
        {
            return out;
        }

        // EE layout (kernel.h):
        // [0]=count [1]=max_count [2]=init_count [3]=wait_threads [4]=attr [5]=option
        const bool hasEeLayout = availableWords >= 3u;
        const int eeMax = hasEeLayout ? static_cast<int>(param[1]) : 1;
        const int eeInit = hasEeLayout ? static_cast<int>(param[2]) : 0;
        const uint32_t eeAttr = (availableWords >= 5u) ? param[4] : 0u;
        const uint32_t eeOption = (availableWords >= 6u) ? param[5] : 0u;

        // Legacy layout (IOP-style):
        // [0]=attr [1]=option [2]=init [3]=max
        const bool hasLegacyLayout = availableWords >= 4u;
        const int legacyMax = hasLegacyLayout ? static_cast<int>(param[3]) : 1;
        const int legacyInit = hasLegacyLayout ? static_cast<int>(param[2]) : 0;
        const uint32_t legacyAttr = hasLegacyLayout ? param[0] : 0u;
        const uint32_t legacyOption = hasLegacyLayout ? param[1] : 0u;

        auto countLooksPlausible = [](int value) -> bool
        {
            return value > 0 && value <= 0x10000;
        };

        bool useLegacyLayout = hasLegacyLayout && !hasEeLayout;
        if (hasLegacyLayout && hasEeLayout && countLooksPlausible(legacyMax) && !countLooksPlausible(eeMax))
        {
            useLegacyLayout = true;
        }
        else if (hasLegacyLayout && hasEeLayout && countLooksPlausible(legacyMax) && countLooksPlausible(eeMax))
        {
            // If both max values look valid, prefer the layout whose option field
            // looks like a pointer/NULL payload.
            const bool eeOptionLooksValid = looksLikeGuestPointerOrNull(eeOption);
            const bool legacyOptionLooksValid = looksLikeGuestPointerOrNull(legacyOption);
            if (!eeOptionLooksValid && legacyOptionLooksValid)
            {
                useLegacyLayout = true;
            }
        }

        if (useLegacyLayout && hasLegacyLayout)
        {
            out.max = legacyMax;
            out.init = legacyInit;
            out.attr = legacyAttr;
            out.option = legacyOption;
        }
        else
        {
            if (!hasEeLayout)
            {
                return out;
            }
            out.max = eeMax;
            out.init = eeInit;
            out.attr = eeAttr;
            out.option = eeOption;
        }

        return out;
    }

    static std::vector<std::shared_ptr<ThreadInfo>>
    snapshotRuntimeThreads(PS2Runtime *runtime)
    {
        if (!runtime)
        {
            return {};
        }

        EeThreadRuntimeState &state =
            runtime->eeThreadRuntimeState();
        std::vector<std::shared_ptr<ThreadInfo>> result;
        std::lock_guard<std::mutex> lock(state.threadMapMutex);
        result.reserve(state.threads.size());
        for (const auto &[id, info] : state.threads)
        {
            (void)id;
            if (info)
            {
                result.push_back(info);
            }
        }
        return result;
    }

    void CreateSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t paramAddr = getRegU32(ctx, 4); // $a0
        if (!runtime || paramAddr == 0u)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        uint32_t rawParams[6] = {};
        uint32_t availableWords = 0u;
        for (uint32_t i = 0; i < 6u; ++i)
        {
            if (!readGuestU32Safe(rdram, paramAddr + (i * 4u), rawParams[i]))
            {
                break;
            }
            availableWords = i + 1u;
        }

        if (availableWords < 3u)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        const DecodedSemaParams decoded = decodeCreateSemaParams(rawParams, availableWords);
        int init = decoded.init;
        int max = decoded.max;
        uint32_t attr = decoded.attr;
        uint32_t option = decoded.option;

        if (max <= 0)
        {
            max = 1;
        }
        if (init < 0)
        {
            init = 0;
        }
        if (init > max)
        {
            init = max;
        }

        int id = -1;
        auto info = std::make_shared<SemaInfo>();
        info->count = init;
        info->maxCount = max;
        info->initCount = init;
        info->attr = attr;
        info->option = option;

        EeSyncRuntimeState &syncState =
            runtime->eeSyncRuntimeState();
        {
            std::lock_guard<std::mutex> lock(
                syncState.semaMapMutex);
            for (int attempts = 0; attempts < 0x7FFF; ++attempts)
            {
                if (syncState.nextSemaId < 0)
                {
                    syncState.nextSemaId = 0;
                }

                const int candidate =
                    syncState.nextSemaId++;
                if (candidate < 0)
                {
                    continue;
                }

                if (syncState.semas.find(candidate) ==
                    syncState.semas.end())
                {
                    id = candidate;
                    break;
                }
            }

            if (id < 0)
            {
                setReturnS32(ctx, KE_ERROR);
                return;
            }

            syncState.semas.emplace(id, info);
        }
        RUNTIME_LOG("[CreateSema] id=" << id << " init=" << init << " max=" << max);
        setReturnS32(ctx, id);
    }

    static void deleteSema(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int sid = static_cast<int>(getRegU32(ctx, 4));
        std::shared_ptr<SemaInfo> sema;

        if (!runtime)
        {
            setReturnS32(ctx, KE_UNKNOWN_SEMID);
            return;
        }

        if (runtime->usesDedicatedEeExecutor())
        {
            sema = lookupSemaInfo(runtime, sid);
            if (!sema)
            {
                setReturnS32(ctx, KE_UNKNOWN_SEMID);
                return;
            }

            struct DeleteTransition
            {
                int result = KE_ERROR;
                bool applied = false;
            };
            const auto transition =
                std::make_shared<DeleteTransition>();
            const bool published =
                runtime->publishEeSchedulerUpdate(
                    [runtime,
                     sema,
                     sid,
                     transition](
                        ps2x::ee::EeThreadScheduler
                            &scheduler)
                    {
                        EeSyncRuntimeState &syncState =
                            runtime
                                ->eeSyncRuntimeState();
                        {
                            std::lock_guard<std::mutex>
                                mapLock(
                                    syncState
                                        .semaMapMutex);
                            const auto it =
                                syncState.semas.find(
                                    sid);
                            if (it ==
                                    syncState.semas.end() ||
                                it->second != sema)
                            {
                                transition->result =
                                    KE_UNKNOWN_SEMID;
                                transition->applied =
                                    true;
                                return;
                            }
                            syncState.semas.erase(it);
                            if (sid <
                                syncState.nextSemaId)
                            {
                                syncState.nextSemaId =
                                    sid;
                            }
                        }

                        std::lock_guard<std::mutex> lock(
                            sema->m);
                        sema->deleted = true;
                        const ps2x::ee::
                            EeSchedulerWaitKey wait{
                                ps2x::ee::
                                    EeSchedulerWaitKind::
                                        Semaphore,
                                sid,
                            };
                        const std::vector<int> waiters =
                            scheduler.waitOrder(wait);
                        for (const int threadId :
                             waiters)
                        {
                            const auto handle =
                                scheduler.threadHandle(
                                    threadId);
                            if (!handle.has_value() ||
                                !scheduler
                                     .releaseWaitThread(
                                         *handle,
                                         ps2x::ee::
                                             EeSchedulerWaitCompletion::
                                                 ObjectDeleted,
                                         true))
                            {
                                throw std::logic_error(
                                    "DeleteSema could not "
                                    "detach an EE scheduler "
                                    "waiter");
                            }

                            const auto waiter =
                                lookupThreadInfo(
                                    runtime,
                                    threadId);
                            if (!waiter)
                            {
                                throw std::logic_error(
                                    "DeleteSema detached a "
                                    "missing EE thread");
                            }
                            std::lock_guard<std::mutex>
                                threadLock(waiter->m);
                            if (!waiter->guestState
                                     .releaseWait(
                                         EeThreadWaitQueue::
                                             Semaphore,
                                         TSW_SEMA,
                                         sid,
                                         true) ||
                                sema->waiters <= 0)
                            {
                                throw std::logic_error(
                                    "DeleteSema scheduler "
                                    "and guest wait queues "
                                    "diverged");
                            }
                            waiter->pendingWaitCompletion =
                                ps2x::ee::
                                    EeSchedulerWaitCompletion::
                                        ObjectDeleted;
                            --sema->waiters;
                        }
                        if (sema->waiters != 0)
                        {
                            throw std::logic_error(
                                "DeleteSema left an "
                                "unmodeled executor waiter");
                        }
                        transition->result = sid;
                        transition->applied = true;
                    },
                    reschedule
                        ? ps2x::ee::
                              EeSchedulerReschedulePolicy::
                                  HigherPriorityOnly
                        : ps2x::ee::
                              EeSchedulerReschedulePolicy::
                                  None);
            if (!published)
            {
                setReturnS32(ctx, KE_ERROR);
                return;
            }

            runtime->yieldEeExecutorCurrent(
                ps2x::ee::EeSchedulerExitReason::
                    Preempted);
            if (!transition->applied)
            {
                throw std::logic_error(
                    "DeleteSema resumed before its "
                    "scheduler transition");
            }
            setReturnS32(ctx, transition->result);
            return;
        }

        EeSyncRuntimeState &syncState =
            runtime->eeSyncRuntimeState();
        {
            std::lock_guard<std::mutex> lock(
                syncState.semaMapMutex);
            auto it = syncState.semas.find(sid);
            if (it == syncState.semas.end())
            {
                setReturnS32(ctx, KE_UNKNOWN_SEMID);
                return;
            }
            sema = it->second;
            syncState.semas.erase(it);
            if (sid < syncState.nextSemaId)
            {
                syncState.nextSemaId = sid;
            }
        }

        const std::vector<std::shared_ptr<ThreadInfo>> threads =
            snapshotRuntimeThreads(runtime);
        bool hadWaiters = false;
        {
            std::lock_guard<std::mutex> lock(sema->m);
            sema->deleted = true;
            hadWaiters = sema->waiters > 0;
            for (const std::shared_ptr<ThreadInfo> &info :
                 threads)
            {
                std::lock_guard<std::mutex> threadLock(info->m);
                if (info->guestState.releaseWait(
                        EeThreadWaitQueue::Semaphore,
                        TSW_SEMA,
                        sid,
                        true) &&
                    sema->waiters > 0)
                {
                    --sema->waiters;
                }
            }
        }
        sema->cv.notify_all();

        // PS2 EE BIOS returns sid on success.
        setReturnS32(ctx, sid);
        if (hadWaiters && reschedule)
        {
            yieldGuestExecutionAfterWake(runtime);
        }
    }

    void DeleteSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        deleteSema(rdram, ctx, runtime, true);
    }

    void iDeleteSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        deleteSema(rdram, ctx, runtime, false);
    }

    static void signalSema(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool reschedule)
    {
        int sid = static_cast<int>(getRegU32(ctx, 4));
        auto sema = lookupSemaInfo(runtime, sid);
        if (!sema)
        {
            setReturnS32(ctx, KE_UNKNOWN_SEMID);
            return;
        }

        if (runtime->usesDedicatedEeExecutor())
        {
            struct SignalTransition
            {
                int result = KE_ERROR;
                bool applied = false;
            };

            const auto transition =
                std::make_shared<SignalTransition>();
            const bool published =
                runtime->publishEeSchedulerUpdate(
                    [runtime, sema, sid, transition](
                        ps2x::ee::EeThreadScheduler
                            &scheduler)
                    {
                        std::lock_guard<std::mutex> lock(
                            sema->m);
                        transition->result = sid;
                        if (sema->deleted)
                        {
                            transition->result =
                                KE_UNKNOWN_SEMID;
                        }
                        else if (
                            sema->count >=
                            sema->maxCount)
                        {
                            transition->result =
                                KE_SEMA_OVF;
                        }
                        else
                        {
                            const auto wokenThreadId =
                                scheduler.wakeOne(
                                    {
                                        ps2x::ee::
                                            EeSchedulerWaitKind::
                                                Semaphore,
                                        sid,
                                    });
                            if (wokenThreadId.has_value())
                            {
                                const auto waiter =
                                    lookupThreadInfo(
                                        runtime,
                                        *wokenThreadId);
                                if (!waiter)
                                {
                                    throw std::logic_error(
                                        "SignalSema woke a "
                                        "missing EE thread");
                                }
                                std::lock_guard<std::mutex>
                                    threadLock(waiter->m);
                                if (!waiter->guestState
                                         .releaseWait(
                                             EeThreadWaitQueue::
                                                 Semaphore,
                                             TSW_SEMA,
                                             sid,
                                             false) ||
                                    sema->waiters <= 0)
                                {
                                    throw std::logic_error(
                                        "SignalSema "
                                        "scheduler and "
                                        "guest wait queues "
                                        "diverged");
                                }
                                waiter->pendingWaitCompletion =
                                    ps2x::ee::
                                        EeSchedulerWaitCompletion::
                                            Satisfied;
                                --sema->waiters;
                            }
                            else
                            {
                                if (sema->waiters != 0)
                                {
                                    throw std::logic_error(
                                        "SignalSema found "
                                        "an unmodeled "
                                        "executor waiter");
                                }
                                ++sema->count;
                            }
                        }
                        transition->applied = true;
                    },
                    reschedule
                        ? ps2x::ee::
                              EeSchedulerReschedulePolicy::
                                  HigherPriorityOnly
                        : ps2x::ee::
                              EeSchedulerReschedulePolicy::
                                  None);
            if (!published)
            {
                setReturnS32(ctx, KE_ERROR);
                return;
            }

            runtime->yieldEeExecutorCurrent(
                ps2x::ee::EeSchedulerExitReason::
                    Preempted);
            if (!transition->applied)
            {
                throw std::logic_error(
                    "SignalSema resumed before its "
                    "scheduler transition");
            }
            setReturnS32(ctx, transition->result);
            return;
        }

        // PS2 EE BIOS returns sid on success; KE_SEMA_OVF overrides on overflow.
        int ret = sid;
        int beforeCount = 0;
        int afterCount = 0;
        bool wokeWaiter = false;
        {
            std::unique_lock<std::mutex> lock(sema->m);
            beforeCount = sema->count;
            if (sema->count >= sema->maxCount)
            {
                ret = KE_SEMA_OVF;
            }
            else
            {
                sema->count++;
                wokeWaiter = sema->waiters > 0;
                const int waitersBeforeWake = sema->waiters;
                sema->cv.notify_one();
                if (wokeWaiter)
                {
                    // Publication is synchronous even for the interrupt-safe
                    // variant. The waiter may update guest state while guest
                    // execution remains owned by this caller, but cannot run
                    // its continuation until a later scheduling boundary.
                    sema->cv.wait(
                        lock,
                        [&]()
                        {
                            return sema->waiters <
                                       waitersBeforeWake ||
                                   sema->deleted;
                        });
                }
            }
            afterCount = sema->count;
        }

        static std::atomic<uint32_t> s_signalSemaLogs{0};
        const uint32_t sigLog = s_signalSemaLogs.fetch_add(1, std::memory_order_relaxed);
        if (sigLog < 256u)
        {
            RUNTIME_LOG("[SignalSema] tid=" << getCurrentThreadId(runtime)
                                            << " sid=" << sid
                                            << " count=" << beforeCount << "->" << afterCount
                                            << " ret=" << ret
                                            << std::endl);
        }

        setReturnS32(ctx, ret);
        if (wokeWaiter && reschedule)
        {
            yieldGuestExecutionAfterWake(runtime);
        }
    }

    void SignalSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        signalSema(rdram, ctx, runtime, true);
    }

    void iSignalSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        signalSema(rdram, ctx, runtime, false);
    }

    void WaitSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int sid = static_cast<int>(getRegU32(ctx, 4));
        auto sema = lookupSemaInfo(runtime, sid);
        if (!sema)
        {
            setReturnS32(ctx, KE_UNKNOWN_SEMID);
            return;
        }

        auto info = ensureCurrentThreadInfo(runtime, ctx);
        throwIfTerminated(info);

        if (runtime->usesDedicatedEeExecutor())
        {
            struct WaitTransition
            {
                bool blocked = false;
                bool applied = false;
            };

            const int threadId =
                getCurrentThreadId(runtime);
            const uint32_t generation =
                info->generation;
            const auto transition =
                std::make_shared<WaitTransition>();
            const bool published =
                runtime->publishEeSchedulerUpdate(
                    [sema,
                     info,
                     sid,
                     threadId,
                     generation,
                     transition](
                        ps2x::ee::EeThreadScheduler
                            &scheduler)
                    {
                        const auto handle =
                            scheduler.threadHandle(
                                threadId);
                        if (!handle.has_value() ||
                            handle->generation !=
                                generation)
                        {
                            throw std::logic_error(
                                "WaitSema lost its EE "
                                "scheduler record");
                        }

                        std::lock_guard<std::mutex> lock(
                            sema->m);
                        {
                            std::lock_guard<std::mutex>
                                threadLock(info->m);
                            info->pendingWaitCompletion =
                                ps2x::ee::
                                    EeSchedulerWaitCompletion::
                                        None;
                        }
                        if (sema->deleted)
                        {
                            transition->applied = true;
                            return;
                        }
                        if (sema->count > 0)
                        {
                            --sema->count;
                            transition->applied = true;
                            return;
                        }

                        (void)scheduler
                            .blockCurrentThread(
                                {
                                    ps2x::ee::
                                        EeSchedulerWaitKind::
                                            Semaphore,
                                    sid,
                                });
                        const auto blocked =
                            scheduler.thread(
                                threadId);
                        if (!blocked.has_value() ||
                            blocked->generation !=
                                generation ||
                            (blocked->state !=
                                 ps2x::ee::
                                     EeSchedulerThreadState::
                                         Waiting &&
                             blocked->state !=
                                 ps2x::ee::
                                     EeSchedulerThreadState::
                                         WaitSuspended))
                        {
                            throw std::logic_error(
                                "WaitSema could not block "
                                "its running scheduler "
                                "owner");
                        }

                        std::lock_guard<std::mutex>
                            threadLock(info->m);
                        if (!info->guestState.block(
                                EeThreadWaitQueue::
                                    Semaphore,
                                TSW_SEMA,
                                sid))
                        {
                            throw std::logic_error(
                                "WaitSema could not mirror "
                                "its scheduler wait");
                        }
                        info->forceRelease = false;
                        ++sema->waiters;
                        transition->blocked = true;
                        transition->applied = true;
                    },
                    ps2x::ee::
                        EeSchedulerReschedulePolicy::
                            None);
            if (!published)
            {
                setReturnS32(ctx, KE_ERROR);
                return;
            }

            runtime->yieldEeExecutorCurrent(
                ps2x::ee::EeSchedulerExitReason::
                    Preempted);
            if (!transition->applied)
            {
                throw std::logic_error(
                    "WaitSema resumed before its "
                    "scheduler transition");
            }

            int ret = sid;
            bool terminated = false;
            if (transition->blocked)
            {
                std::lock_guard<std::mutex> semaLock(
                    sema->m);
                std::lock_guard<std::mutex> threadLock(
                    info->m);
                terminated = info->terminated.load();
                if (!terminated)
                {
                    const auto completion =
                        info->pendingWaitCompletion;
                    info->pendingWaitCompletion =
                        ps2x::ee::
                            EeSchedulerWaitCompletion::
                                None;
                    info->guestState.finishWait(
                        EeThreadWaitQueue::Semaphore,
                        TSW_SEMA,
                        sid,
                        false);
                    if (completion !=
                        ps2x::ee::
                            EeSchedulerWaitCompletion::
                                Satisfied)
                    {
                        info->forceRelease = false;
                        ret = KE_ERROR;
                    }
                }
            }
            else
            {
                std::lock_guard<std::mutex> lock(
                    sema->m);
                if (sema->deleted)
                {
                    ret = KE_ERROR;
                }
            }

            if (terminated)
            {
                throw ThreadExitException();
            }
            waitWhileSuspended(info, runtime);
            (void)publishRunningAtGuestBoundary(info);
            setReturnS32(ctx, ret);
            return;
        }

        std::unique_lock<std::mutex> lock(sema->m);

        // PS2 EE BIOS returns sid on success.
        int ret = sid;
        int countAfter = sema->count;
        bool terminated = false;
        bool consumed = false;

        if (sema->count == 0)
        {
            static std::atomic<uint32_t> s_waitSemaBlockLogs{0};
            const uint32_t blockLog = s_waitSemaBlockLogs.fetch_add(1, std::memory_order_relaxed);
            if (blockLog < 256u)
            {
                RUNTIME_LOG("[WaitSema:block] tid=" << getCurrentThreadId(runtime)
                                                    << " sid=" << sid
                                                    << " pc=0x" << std::hex << ctx->pc
                                                    << " ra=0x" << getRegU32(ctx, 31)
                                                    << std::dec
                                                    << std::endl);
            }

            if (info)
            {
                std::lock_guard<std::mutex> tLock(info->m);
                info->guestState.block(
                    EeThreadWaitQueue::Semaphore,
                    TSW_SEMA,
                    sid);
                info->forceRelease = false;
            }

            sema->waiters++;
            waitWithGuestExecutionReleasedUntilUnlocked(
                runtime,
                lock,
                [&]()
                {
                    sema->cv.wait(lock, [&]()
                                  {
                                      const bool forced = info ? info->forceRelease.load() : false;
                                      const bool isTerminated = info ? info->terminated.load() : false;
                                      return sema->count > 0 || sema->deleted || forced || isTerminated;
                                  });
                },
                [&]()
                {
                    if (sema->deleted)
                    {
                        ret = KE_ERROR;
                    }

                    if (info)
                    {
                        std::lock_guard<std::mutex> tLock(info->m);
                        terminated = info->terminated.load();
                        bool wasLinked =
                            info->guestState.isLinkedTo(
                                EeThreadWaitQueue::Semaphore,
                                sid);
                        if (terminated)
                        {
                            info->guestState.makeDormant();
                        }
                        else
                        {
                            const EeThreadGuestStateSnapshot
                                &guest =
                                    info->guestState.snapshot();
                            const bool retainedCompletion =
                                guest.waitCompletionPending &&
                                guest.waitType == TSW_SEMA &&
                                guest.waitId == sid;
                            if (!retainedCompletion)
                            {
                                wasLinked =
                                    info->guestState.finishWait(
                                        EeThreadWaitQueue::
                                            Semaphore,
                                        TSW_SEMA,
                                        sid,
                                        false);
                            }
                        }
                        if (wasLinked &&
                            sema->waiters > 0)
                        {
                            --sema->waiters;
                            sema->cv.notify_all();
                        }
                        if (info->forceRelease)
                        {
                            info->forceRelease = false;
                            ret = KE_ERROR;
                        }
                    }
                    else if (sema->waiters > 0)
                    {
                        --sema->waiters;
                        sema->cv.notify_all();
                    }

                    if (!terminated && ret == sid && sema->count > 0)
                    {
                        sema->count--;
                        consumed = true;
                    }
                    countAfter = sema->count;
                });
        }

        if (terminated)
        {
            throw ThreadExitException();
        }

        if (info)
        {
            std::lock_guard<std::mutex> tLock(info->m);
            const EeThreadGuestStateSnapshot &guest =
                info->guestState.snapshot();
            if (!info->terminated.load() &&
                guest.suspendCount == 0 &&
                guest.status == THS_READY)
            {
                info->guestState.publishRunning();
            }
        }

        if (!consumed && lock.owns_lock())
        {
            if (ret == sid && sema->count > 0)
            {
                sema->count--;
                consumed = true;
            }
            countAfter = sema->count;
        }

        static std::atomic<uint32_t> s_waitSemaWakeLogs{0};
        const uint32_t wakeLog = s_waitSemaWakeLogs.fetch_add(1, std::memory_order_relaxed);
        if (wakeLog < 256u)
        {
            RUNTIME_LOG("[WaitSema:wake] tid=" << getCurrentThreadId(runtime)
                                               << " sid=" << sid
                                               << " ret=" << ret
                                               << " count=" << countAfter
                                               << std::endl);
        }
        if (lock.owns_lock())
        {
            lock.unlock();
        }
        waitWhileSuspended(info, runtime);
        (void)publishRunningAtGuestBoundary(info);
        setReturnS32(ctx, ret);
    }

    void PollSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int sid = static_cast<int>(getRegU32(ctx, 4));
        auto sema = lookupSemaInfo(runtime, sid);
        if (!sema)
        {
            setReturnS32(ctx, KE_UNKNOWN_SEMID);
            return;
        }

        std::lock_guard<std::mutex> lock(sema->m);
        if (sema->count > 0)
        {
            sema->count--;
            setReturnS32(ctx, sid);  // PS2 EE BIOS returns sid on success.
            return;
        }

        setReturnS32(ctx, KE_SEMA_ZERO);
    }

    void iPollSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        PollSema(rdram, ctx, runtime);
    }

    static void referSemaStatus(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        bool raw)
    {
        int sid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t statusAddr = getRegU32(ctx, 5);

        auto sema = lookupSemaInfo(runtime, sid);
        if (!sema)
        {
            setReturnS32(ctx, raw ? KE_ERROR : KE_UNKNOWN_SEMID);
            return;
        }

        ee_sema_t *status = reinterpret_cast<ee_sema_t *>(getMemPtr(rdram, statusAddr));
        if (!status)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        std::lock_guard<std::mutex> lock(sema->m);
        status->count = sema->count;
        status->max_count = sema->maxCount;
        status->init_count = sema->initCount;
        status->wait_threads = sema->waiters;
        status->attr = sema->attr;
        status->option = sema->option;
        setReturnS32(ctx, KE_OK);
    }

    void ReferSemaStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        referSemaStatus(rdram, ctx, runtime, false);
    }

    void iReferSemaStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        referSemaStatus(rdram, ctx, runtime, true);
    }

    constexpr uint32_t WEF_OR = 1;
    constexpr uint32_t WEF_CLEAR = 0x10;
    constexpr uint32_t WEF_CLEAR_ALL = 0x20;
    constexpr uint32_t WEF_MODE_MASK = WEF_OR | WEF_CLEAR | WEF_CLEAR_ALL;
    constexpr uint32_t EA_MULTI = 0x2;

    void CreateEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        uint32_t paramAddr = getRegU32(ctx, 4); // $a0
        const uint32_t *param = reinterpret_cast<const uint32_t *>(getConstMemPtr(rdram, paramAddr));

        auto info = std::make_shared<EventFlagInfo>();
        if (param)
        {
            info->attr = param[0];
            info->option = param[1];
            info->initBits = param[2];
            info->bits = info->initBits;
        }

        int id = 0;
        EeSyncRuntimeState &syncState =
            runtime->eeSyncRuntimeState();
        {
            std::lock_guard<std::mutex> mapLock(
                syncState.eventFlagMapMutex);
            id = syncState.nextEventFlagId++;
            syncState.eventFlags[id] = info;
        }
        setReturnS32(ctx, id);
    }

    void DeleteEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int eid = static_cast<int>(getRegU32(ctx, 4));
        std::shared_ptr<EventFlagInfo> info;
        if (!runtime)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }
        EeSyncRuntimeState &syncState =
            runtime->eeSyncRuntimeState();
        {
            std::lock_guard<std::mutex> mapLock(
                syncState.eventFlagMapMutex);
            auto it = syncState.eventFlags.find(eid);
            if (it == syncState.eventFlags.end())
            {
                setReturnS32(ctx, KE_UNKNOWN_EVFID);
                return;
            }
            info = it->second;
            syncState.eventFlags.erase(it);
        }

        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(info->m);
            info->deleted = true;
        }
        info->cv.notify_all();
        setReturnS32(ctx, 0);
    }

    void SetEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int eid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t bits = getRegU32(ctx, 5);
        auto info = lookupEventFlagInfo(runtime, eid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }

        if (bits == 0)
        {
            setReturnS32(ctx, KE_OK);
            return;
        }

        uint32_t newBits = 0u;
        bool hadWaiters = false;
        {
            std::lock_guard<std::mutex> lock(info->m);
            info->bits |= bits;
            newBits = info->bits;
            hadWaiters = info->waiters > 0;
        }

        static std::atomic<uint32_t> s_setEventFlagLogs{0};
        const uint32_t setLog = s_setEventFlagLogs.fetch_add(1, std::memory_order_relaxed);
        if (setLog < 256u)
        {
            RUNTIME_LOG("[SetEventFlag] tid=" << getCurrentThreadId(runtime)
                                              << " eid=" << eid
                                              << " bits=0x" << std::hex << bits
                                              << " newBits=0x" << newBits
                                              << std::dec << std::endl);
        }
        info->cv.notify_all();
        setReturnS32(ctx, 0);
        if (hadWaiters)
        {
            yieldGuestExecutionAfterWake(runtime);
        }
    }

    void iSetEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SetEventFlag(rdram, ctx, runtime);
    }

    void ClearEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int eid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t bits = getRegU32(ctx, 5);
        auto info = lookupEventFlagInfo(runtime, eid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(info->m);
            info->bits &= bits;
        }
        info->cv.notify_all();
        setReturnS32(ctx, KE_OK);
    }

    void iClearEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ClearEventFlag(rdram, ctx, runtime);
    }

    void WaitEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int eid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t waitBits = getRegU32(ctx, 5);
        uint32_t mode = getRegU32(ctx, 6);
        uint32_t resBitsAddr = getRegU32(ctx, 7);

        if ((mode & ~WEF_MODE_MASK) != 0)
        {
            setReturnS32(ctx, KE_ILLEGAL_MODE);
            return;
        }

        if (waitBits == 0)
        {
            setReturnS32(ctx, KE_EVF_ILPAT);
            return;
        }

        auto info = lookupEventFlagInfo(runtime, eid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }

        uint32_t *resBitsPtr = resBitsAddr ? reinterpret_cast<uint32_t *>(getMemPtr(rdram, resBitsAddr)) : nullptr;

        std::unique_lock<std::mutex> lock(info->m);
        if ((info->attr & EA_MULTI) == 0 && info->waiters > 0)
        {
            setReturnS32(ctx, KE_EVF_MULTI);
            return;
        }

        auto tInfo = ensureCurrentThreadInfo(runtime, ctx);
        throwIfTerminated(tInfo);
        int ret = KE_OK;

        auto satisfied = [&]()
        {
            if (tInfo && tInfo->forceRelease.load())
                return true;
            if (tInfo && tInfo->terminated.load())
                return true;
            if (info->deleted)
            {
                return true;
            }
            if (mode & WEF_OR)
            {
                return (info->bits & waitBits) != 0;
            }
            return (info->bits & waitBits) == waitBits;
        };

        bool waitedWithGuestRelease = false;
        bool terminated = false;
        uint32_t bitsAfter = info->bits;

        auto finishEventFlagWaitLocked = [&]()
        {
            if (ret == KE_OK && info->deleted)
            {
                ret = KE_WAIT_DELETE;
            }

            if (ret == KE_OK)
            {
                if (resBitsPtr)
                {
                    *resBitsPtr = info->bits;
                }

                if (mode & WEF_CLEAR_ALL)
                {
                    info->bits = 0;
                }
                else if (mode & WEF_CLEAR)
                {
                    info->bits &= ~waitBits;
                }
            }

            bitsAfter = info->bits;
        };

        if (!satisfied())
        {
            static std::atomic<uint32_t> s_waitEventBlockLogs{0};
            const uint32_t evBlockLog = s_waitEventBlockLogs.fetch_add(1, std::memory_order_relaxed);
            if (evBlockLog < 256u)
            {
                RUNTIME_LOG("[WaitEventFlag:block] tid=" << getCurrentThreadId(runtime)
                                                         << " eid=" << eid
                                                         << " waitBits=0x" << std::hex << waitBits
                                                         << " mode=0x" << mode
                                                         << " bits=0x" << info->bits
                                                         << " pc=0x" << ctx->pc
                                                         << " ra=0x" << getRegU32(ctx, 31)
                                                         << std::dec
                                                         << std::endl);
            }

            if (tInfo)
            {
                std::lock_guard<std::mutex> tLock(tInfo->m);
                tInfo->guestState.block(
                    EeThreadWaitQueue::EventFlag,
                    TSW_EVENT,
                    eid);
                tInfo->forceRelease = false;
            }

            info->waiters++;
            waitedWithGuestRelease = true;
            waitWithGuestExecutionReleasedUntilUnlocked(
                runtime,
                lock,
                [&]()
                {
                    info->cv.wait(lock, satisfied);
                },
                [&]()
                {
                    bool wasLinked = !tInfo;

                    if (tInfo)
                    {
                        std::lock_guard<std::mutex> tLock(tInfo->m);
                        terminated = tInfo->terminated.load();
                        wasLinked =
                            tInfo->guestState.isLinkedTo(
                                EeThreadWaitQueue::EventFlag,
                                eid);
                        if (terminated)
                        {
                            tInfo->guestState.makeDormant();
                        }
                        else
                        {
                            wasLinked =
                                tInfo->guestState.finishWait(
                                    EeThreadWaitQueue::EventFlag,
                                    TSW_EVENT,
                                    eid,
                                    false);
                        }
                        if (tInfo->forceRelease)
                        {
                            tInfo->forceRelease = false;
                            ret = KE_RELEASE_WAIT;
                        }
                    }
                    if (wasLinked && info->waiters > 0)
                    {
                        --info->waiters;
                    }

                    if (!terminated)
                    {
                        finishEventFlagWaitLocked();
                    }
                    else
                    {
                        bitsAfter = info->bits;
                    }
                });
        }

        if (terminated)
        {
            throw ThreadExitException();
        }

        if (!waitedWithGuestRelease)
        {
            finishEventFlagWaitLocked();
        }

        static std::atomic<uint32_t> s_waitEventWakeLogs{0};
        const uint32_t evWakeLog = s_waitEventWakeLogs.fetch_add(1, std::memory_order_relaxed);
        if (evWakeLog < 256u)
        {
            RUNTIME_LOG("[WaitEventFlag:wake] tid=" << getCurrentThreadId(runtime)
                                                    << " eid=" << eid
                                                    << " ret=" << ret
                                                    << " bits=0x" << std::hex << bitsAfter
                                                    << std::dec
                                                    << std::endl);
        }

        if (lock.owns_lock())
        {
            lock.unlock();
        }
        if (tInfo)
        {
            std::lock_guard<std::mutex> stateLock(tInfo->m);
            const EeThreadGuestStateSnapshot &guest =
                tInfo->guestState.snapshot();
            if (!tInfo->terminated.load() &&
                guest.suspendCount == 0 &&
                guest.status == THS_READY)
            {
                tInfo->guestState.publishRunning();
            }
        }
        waitWhileSuspended(tInfo, runtime);
        (void)publishRunningAtGuestBoundary(tInfo);
        setReturnS32(ctx, ret);
    }

    void PollEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int eid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t waitBits = getRegU32(ctx, 5);
        uint32_t mode = getRegU32(ctx, 6);
        uint32_t resBitsAddr = getRegU32(ctx, 7);

        if ((mode & ~WEF_MODE_MASK) != 0)
        {
            setReturnS32(ctx, KE_ILLEGAL_MODE);
            return;
        }

        if (waitBits == 0)
        {
            setReturnS32(ctx, KE_EVF_ILPAT);
            return;
        }

        auto info = lookupEventFlagInfo(runtime, eid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }

        uint32_t *resBitsPtr = resBitsAddr ? reinterpret_cast<uint32_t *>(getMemPtr(rdram, resBitsAddr)) : nullptr;

        std::lock_guard<std::mutex> lock(info->m);
        if ((info->attr & EA_MULTI) == 0 && info->waiters > 0)
        {
            setReturnS32(ctx, KE_EVF_MULTI);
            return;
        }

        bool ok = false;
        if (mode & WEF_OR)
        {
            ok = (info->bits & waitBits) != 0;
        }
        else
        {
            ok = (info->bits & waitBits) == waitBits;
        }

        if (!ok)
        {
            setReturnS32(ctx, KE_EVF_COND);
            return;
        }

        if (resBitsPtr)
        {
            *resBitsPtr = info->bits;
        }

        if (mode & WEF_CLEAR_ALL)
        {
            info->bits = 0;
        }
        else if (mode & WEF_CLEAR)
        {
            info->bits &= ~waitBits;
        }

        setReturnS32(ctx, KE_OK);
    }

    void iPollEventFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        PollEventFlag(rdram, ctx, runtime);
    }

    void ReferEventFlagStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int eid = static_cast<int>(getRegU32(ctx, 4));
        uint32_t infoAddr = getRegU32(ctx, 5);

        struct Ps2EventFlagInfo
        {
            uint32_t attr;
            uint32_t option;
            uint32_t initBits;
            uint32_t currBits;
            int32_t numThreads;
            int32_t reserved1;
            int32_t reserved2;
        };

        auto info = lookupEventFlagInfo(runtime, eid);
        if (!info)
        {
            setReturnS32(ctx, KE_UNKNOWN_EVFID);
            return;
        }

        Ps2EventFlagInfo *out = infoAddr ? reinterpret_cast<Ps2EventFlagInfo *>(getMemPtr(rdram, infoAddr)) : nullptr;
        if (!out)
        {
            setReturnS32(ctx, -1);
            return;
        }

        std::lock_guard<std::mutex> lock(info->m);
        out->attr = info->attr;
        out->option = info->option;
        out->initBits = info->initBits;
        out->currBits = info->bits;
        out->numThreads = info->waiters;
        out->reserved1 = 0;
        out->reserved2 = 0;
        setReturnS32(ctx, 0);
    }

    void iReferEventFlagStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ReferEventFlagStatus(rdram, ctx, runtime);
    }

    void SetAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint16_t ticks = static_cast<uint16_t>(getRegU32(ctx, 4) & 0xFFFFu);
        uint32_t handler = getRegU32(ctx, 5);
        uint32_t arg = getRegU32(ctx, 6);

        if (!runtime || !handler || !runtime->hasFunction(handler))
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        auto info = std::make_shared<AlarmInfo>();
        info->ticks = ticks;
        info->handler = handler;
        info->commonArg = arg;
        info->gp = getRegU32(ctx, 28);
        info->sp = getRegU32(ctx, 29);
        info->rdram = rdram;
        info->dueAt = std::chrono::steady_clock::now() + alarmTicksToDuration(ticks);

        EeAlarmRuntimeState &alarmState =
            runtime->eeAlarmRuntimeState();
        int alarmId = 0;
        {
            std::lock_guard<std::mutex> lock(
                alarmState.mutex);
            if (alarmState.stopRequested.load(
                    std::memory_order_acquire))
            {
                if (runtime->isStopRequested())
                {
                    setReturnS32(ctx, KE_ERROR);
                    return;
                }
                // requestStop joins synchronously. A later run or direct
                // test reuse may start a fresh worker and ID namespace.
                alarmState.stopRequested.store(
                    false, std::memory_order_release);
            }

            if (alarmState.setLogCount < 5u)
            {
                RUNTIME_LOG("[SetAlarm] ticks=" << ticks
                                                << " handler=0x" << std::hex << handler
                                                << " arg=0x" << arg << std::dec << std::endl);
                ++alarmState.setLogCount;
            }

            alarmId = alarmState.nextAlarmId++;
            if (alarmState.nextAlarmId <= 0)
            {
                alarmState.nextAlarmId = 1;
            }
            info->id = alarmId;
            alarmState.alarms[alarmId] = info;
        }

        if (!ensureAlarmWorkerRunning(runtime))
        {
            std::lock_guard<std::mutex> lock(
                alarmState.mutex);
            alarmState.alarms.erase(alarmId);
            setReturnS32(ctx, KE_ERROR);
            return;
        }
        alarmState.cv.notify_all();
        setReturnS32(ctx, alarmId);
    }

    void InitAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void iSetAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SetAlarm(rdram, ctx, runtime);
    }

    void CancelAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int alarmId = static_cast<int>(getRegU32(ctx, 4));
        if (!runtime || alarmId <= 0)
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }

        EeAlarmRuntimeState &alarmState =
            runtime->eeAlarmRuntimeState();
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(
                alarmState.mutex);
            removed =
                alarmState.alarms.erase(alarmId) != 0;
        }

        if (removed)
        {
            alarmState.cv.notify_all();
            setReturnS32(ctx, KE_OK);
            return;
        }

        setReturnS32(ctx, KE_ERROR);
    }

    void iCancelAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        CancelAlarm(rdram, ctx, runtime);
    }

    void ReleaseAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        CancelAlarm(rdram, ctx, runtime);
    }

    void iReleaseAlarm(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        iCancelAlarm(rdram, ctx, runtime);
    }
}
