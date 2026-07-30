#pragma once

#include "ps2_runtime.h"

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
    uint32_t stackTop = 0u;
    bool active = false;
};

// Guest interrupt registration, masks, and callback continuations belong to
// one runtime. Handler metadata is copied while handlerMutex is held, then
// guest code runs without the registry lock. Callback slots are pooled so
// concurrent or nested delivery never shares an R5900 context or guest stack.
struct EeInterruptRuntimeState
{
    EeInterruptCallbackContext *acquireCallbackContext(
        PS2Runtime &runtime)
    {
        constexpr uint32_t kCallbackStackSize = 0x4000u;

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
        if (slot->stackTop == 0u)
        {
            slot->stackTop =
                runtime.reserveAsyncCallbackStack(
                    kCallbackStackSize, 16u);
        }
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

    std::mutex callbackContextMutex;
    std::vector<
        std::unique_ptr<EeInterruptCallbackContext>>
        callbackContexts;
};
