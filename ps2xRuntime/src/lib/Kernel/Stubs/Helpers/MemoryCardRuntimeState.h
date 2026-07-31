#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ps2_stubs
{
    struct MemoryCardOpenFile
    {
        FILE *file = nullptr;
        int32_t port = 0;
        std::filesystem::path hostPath;
    };

    struct MemoryCardPortState
    {
        std::string currentDir = "/";
        bool formatted = true;
    };

    struct MemoryCardRuntimeState
    {
        ~MemoryCardRuntimeState()
        {
            reset();
        }

        void closeFilesLocked()
        {
            for (auto &[fd, openFile] : files)
            {
                (void)fd;
                if (openFile.file)
                {
                    std::fclose(openFile.file);
                    openFile.file = nullptr;
                }
            }
            files.clear();
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(mutex);
            closeFilesLocked();
            nextFd = 1;
            lastCmd = 0;
            commandPending = false;
            lastResult = 0;
            ports = {};
            cvFileCursor = 0;
        }

        std::mutex mutex;
        int32_t nextFd = 1;
        int32_t lastCmd = 0;
        bool commandPending = false;
        int32_t lastResult = 0;
        std::unordered_map<int32_t, MemoryCardOpenFile>
            files;
        std::array<MemoryCardPortState, 2> ports{};
        int32_t cvFileCursor = 0;
    };
}
