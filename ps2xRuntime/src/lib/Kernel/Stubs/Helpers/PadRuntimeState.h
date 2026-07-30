#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace ps2_stubs
{
    inline constexpr size_t kPadRuntimePortCount = 2u;
    inline constexpr size_t kPadRuntimeSlotCount = 1u;
    inline constexpr size_t kPadRuntimeDataSize = 32u;

    struct PadInputState
    {
        uint16_t buttons = 0xFFFFu;
        uint8_t rx = 0x80u;
        uint8_t ry = 0x80u;
        uint8_t lx = 0x80u;
        uint8_t ly = 0x80u;
    };

    struct PadPortState
    {
        bool open = false;
        bool analogMode = false;
        bool pressureEnabled = false;
        bool lastUsedOverride = false;
        bool lastUsedBackend = false;
        bool lastReadOk = false;
        uint16_t buttonMask = 0xFFFFu;
        uint32_t dmaAddr = 0u;
        uint32_t reqState = 0u;
        uint32_t transientState = 0u;
        PadInputState lastInput{};
        std::array<uint8_t, kPadRuntimeDataSize>
            lastData{};
        uint32_t readCount = 0u;
        uint32_t lastReadDataAddr = 0u;
    };

    struct PadRuntimeState
    {
        void resetGuestStateLocked()
        {
            for (PadPortState &port : ports)
            {
                port = {};
            }
        }

        void reset()
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                resetGuestStateLocked();
                overrideEnabled = false;
                overrideInput = {};
                readLogCount = 0;
            }
            frameCount.store(
                0u, std::memory_order_relaxed);
        }

        std::mutex mutex;
        bool overrideEnabled = false;
        PadInputState overrideInput{};
        std::array<PadPortState, kPadRuntimePortCount>
            ports{};
        int readLogCount = 0;
        std::atomic<uint32_t> frameCount{0u};
    };
}
