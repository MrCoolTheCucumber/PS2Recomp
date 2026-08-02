#pragma once

#include "ps2_runtime.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct IrqHandlerInfo
{
    int id = 0;
    uint32_t cause = 0;
    uint32_t handler = 0;
    uint32_t arg = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
    bool enabled = true;
    int order = 0;
};

struct EeInterruptCallbackContext
{
    R5900Context context{};
    bool active = false;
};

struct EeVSyncFlagRegistration
{
    uint32_t flagAddr = 0u;
    uint32_t tickAddr = 0u;
};

struct EeGsVideoParameters
{
    uint8_t interlace = 1u;
    uint8_t outputMode = 2u;
    uint8_t frameMode = 1u;
    uint8_t version = 3u;
};

// Guest interrupt registration, masks, and callback continuations belong to
// one runtime. Handler metadata is copied while handlerMutex is held, then
// guest code runs without the registry lock. Callback slots are pooled so
// concurrent or nested delivery never shares an R5900 context. The runtime's
// owned interrupt frame supplies continuation identity and handler stack.
struct EeInterruptRuntimeState
{
    EeInterruptCallbackContext *acquireCallbackContext()
    {
        std::lock_guard<std::mutex> lock(
            callbackContextMutex);
        EeInterruptCallbackContext *slot = nullptr;
        for (const auto &candidate : callbackContexts)
        {
            if (!candidate->active)
            {
                slot = candidate.get();
                break;
            }
        }
        if (!slot)
        {
            auto candidate =
                std::make_unique<EeInterruptCallbackContext>();
            slot = candidate.get();
            callbackContexts.push_back(std::move(candidate));
        }

        slot->context = {};
        slot->active = true;
        return slot;
    }

    void releaseCallbackContext(
        EeInterruptCallbackContext *slot)
    {
        if (!slot)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(
            callbackContextMutex);
        slot->context = {};
        slot->active = false;
    }

    std::mutex handlerMutex;
    std::unordered_map<int, IrqHandlerInfo> intcHandlers;
    std::unordered_map<int, IrqHandlerInfo> dmacHandlers;
    int nextIntcHandlerId = 1;
    int nextDmacHandlerId = 1;
    int intcHeadOrder = 0;
    int intcTailOrder = 1000;
    int dmacHeadOrder = 0;
    int dmacTailOrder = 1000;
    uint32_t enabledIntcMask = 0xFFFFFFFFu;
    uint32_t enabledDmacMask = 0xFFFFFFFFu;

    std::mutex vsyncMutex;
    EeVSyncFlagRegistration vsyncRegistration{};
    uint64_t vsyncTick = 0u;
    uint64_t gsSyncVBaseTick = 0u;
    EeGsVideoParameters gsVideoParameters{};
    uint32_t gsSyncVCallback = 0u;
    uint32_t gsSyncVCallbackGp = 0u;
    uint32_t gsSyncVCallbackSp = 0u;
    std::atomic<uint32_t> gsSyncVCallbackSetLogCount{0u};
    std::atomic<uint32_t> gsSyncVCallbackMissingLogCount{0u};
    std::atomic<uint32_t> gsSyncVCallbackDispatchLogCount{0u};
    std::atomic<uint32_t> gsSyncVCallbackBadPcLogCount{0u};
    std::atomic<uint32_t> gsSyncVCallbackWarningCount{0u};

    std::mutex callbackContextMutex;
    std::vector<
        std::unique_ptr<EeInterruptCallbackContext>>
        callbackContexts;
};

class EeAsyncCallbackContextLease
{
public:
    EeAsyncCallbackContextLease(
        EeInterruptRuntimeState &state)
        : m_state(state),
          m_slot(state.acquireCallbackContext())
    {
    }

    ~EeAsyncCallbackContextLease()
    {
        m_state.releaseCallbackContext(m_slot);
    }

    R5900Context &context()
    {
        return m_slot->context;
    }

    EeAsyncCallbackContextLease(
        const EeAsyncCallbackContextLease &) = delete;
    EeAsyncCallbackContextLease &operator=(
        const EeAsyncCallbackContextLease &) = delete;

private:
    EeInterruptRuntimeState &m_state;
    EeInterruptCallbackContext *m_slot = nullptr;
};
