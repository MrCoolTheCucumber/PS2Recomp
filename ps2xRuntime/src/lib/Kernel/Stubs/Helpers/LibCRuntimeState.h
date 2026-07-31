#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace ps2_stubs
{
    struct LibCRuntimeState
    {
        ~LibCRuntimeState()
        {
            reset();
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto &[handle, file] : files)
            {
                (void)handle;
                if (file)
                {
                    std::fclose(file);
                }
            }
            files.clear();
            nextFileHandle = 1u;
            randomSeed = 1u;
        }

        std::mutex mutex;
        std::unordered_map<uint32_t, FILE *> files;
        uint32_t nextFileHandle = 1u;
        uint32_t randomSeed = 1u;
    };
}
