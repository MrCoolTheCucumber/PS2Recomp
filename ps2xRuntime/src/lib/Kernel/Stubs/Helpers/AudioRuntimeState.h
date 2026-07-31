#pragma once

#include <array>
#include <cstdint>
#include <mutex>

namespace ps2_stubs
{
    struct AudioVoiceTransferState
    {
        uint32_t sourceAddress = 0u;
        uint32_t destinationAddress = 0u;
        uint32_t size = 0u;
        uint16_t mode = 0u;
        bool completed = true;
    };

    struct AudioBlockTransferState
    {
        uint32_t base = 0u;
        uint32_t size = 0u;
        uint32_t pauseBase = 0u;
        uint32_t offset = 0u;
        uint32_t statusTraceCount = 0u;
        uint16_t mode = 0u;
        bool active = false;
        bool loop = false;
    };

    struct AudioRuntimeState
    {
        void resetLocked()
        {
            initialized = false;
            voiceTransfers = {};
            blockTransfers = {};
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(mutex);
            resetLocked();
        }

        std::mutex mutex;
        bool initialized = false;
        std::array<AudioVoiceTransferState, 2u>
            voiceTransfers{};
        std::array<AudioBlockTransferState, 2u>
            blockTransfers{};
    };
}
