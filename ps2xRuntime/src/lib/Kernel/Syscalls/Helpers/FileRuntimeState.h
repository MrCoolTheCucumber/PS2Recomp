#pragma once

#include <cstdio>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

struct EeFileVagAccumEntry
{
    std::vector<uint8_t> data;
    uint32_t firstBufAddr = 0u;
};

struct EeFileRuntimeState
{
    ~EeFileRuntimeState()
    {
        closeAll();
    }

    void closeAll()
    {
        {
            std::lock_guard<std::mutex> lock(
                descriptorMutex);
            for (const auto &[fd, file] : descriptors)
            {
                (void)fd;
                if (file)
                {
                    std::fclose(file);
                }
            }
            descriptors.clear();
            nextDescriptor = 3;
        }
        {
            std::lock_guard<std::mutex> lock(
                vagMutex);
            vagAccum.clear();
        }
    }

    std::mutex descriptorMutex;
    std::unordered_map<int, FILE *> descriptors;
    int nextDescriptor = 3;

    std::mutex vagMutex;
    std::unordered_map<int, EeFileVagAccumEntry>
        vagAccum;
};
