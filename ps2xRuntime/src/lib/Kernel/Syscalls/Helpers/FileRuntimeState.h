#pragma once

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
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
        {
            std::lock_guard<std::mutex> lock(
                pathMutex);
            pathsInitialized = false;
            hostBase.clear();
            cdromBase.clear();
            mcBase.clear();
            hostCwd.clear();
            cdromCwd.clear();
            mcCwd.clear();
            cwdDevice = "cdrom0";
        }
    }

    std::mutex descriptorMutex;
    std::unordered_map<int, FILE *> descriptors;
    int nextDescriptor = 3;

    std::mutex vagMutex;
    std::unordered_map<int, EeFileVagAccumEntry>
        vagAccum;

    std::mutex pathMutex;
    bool pathsInitialized = false;
    std::filesystem::path hostBase;
    std::filesystem::path cdromBase;
    std::filesystem::path mcBase;
    std::filesystem::path hostCwd;
    std::filesystem::path cdromCwd;
    std::filesystem::path mcCwd;
    std::string cwdDevice = "cdrom0";
};
