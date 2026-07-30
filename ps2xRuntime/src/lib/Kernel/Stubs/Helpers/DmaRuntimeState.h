#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace ps2_stubs
{
    struct SceDmaEnv
    {
        uint8_t sts = 0u;
        uint8_t std = 0u;
        uint8_t mfd = 0u;
        uint8_t rele = 0u;
        uint32_t pcr = 0u;
        uint32_t sqwc = 0u;
        uint32_t rbor = 0u;
        uint32_t rbsr = 0u;
    };

    static_assert(
        sizeof(SceDmaEnv) == 0x14u,
        "sceDmaEnv must match the guest ABI");

    struct DmaRuntimeState
    {
        void reset()
        {
            {
                std::lock_guard<std::mutex> lock(
                    environmentMutex);
                currentEnvironment = {};
            }
            {
                std::lock_guard<std::mutex> lock(
                    stubMutex);
                pendingPolls.clear();
                stubLogCount = 0u;
            }
        }

        std::mutex environmentMutex;
        SceDmaEnv currentEnvironment{};

        std::mutex stubMutex;
        std::unordered_map<uint32_t, uint32_t>
            pendingPolls;
        uint32_t stubLogCount = 0u;
    };
}
