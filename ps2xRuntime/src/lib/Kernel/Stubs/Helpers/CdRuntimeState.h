#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ps2_stubs
{
    inline constexpr uint32_t kCdSectorSize = 2048u;
    inline constexpr uint32_t kCdPseudoLbnStart =
        0x00100000u;

    struct CdFileEntry
    {
        std::filesystem::path hostPath;
        uint32_t sizeBytes = 0u;
        uint32_t baseLbn = 0u;
        uint32_t sectors = 0u;
    };

    struct CdRuntimeState
    {
        void resetLocked()
        {
            filesByKey.clear();
            leafIndex.clear();
            loosePathIndex.clear();
            leafIndexRoot.clear();
            leafIndexBuilt = false;
            nextPseudoLbn = kCdPseudoLbnStart;
            imageSizePath.clear();
            imageSizeBytes = 0u;
            imageSizeValid = false;
            lastError = 0;
            mode = 0u;
            streamingLbn = 0u;
            streamingEndLbn = 0xFFFFFFFFu;
            initialized = false;
            stReadTraceCount = 0u;
            readRecoverLogCount = 0u;
            unresolvedReadLogCount = 0u;
            searchTraceCount = 0u;
            emptyPathCount = 0u;
            emptyNormalizedCount = 0u;
            lastFailedPath.clear();
            samePathFailCount = 0u;
        }

        std::mutex mutex;
        std::unordered_map<std::string, CdFileEntry>
            filesByKey;
        std::unordered_map<
            std::string,
            std::filesystem::path>
            leafIndex;
        std::unordered_map<
            std::string,
            std::filesystem::path>
            loosePathIndex;
        std::filesystem::path leafIndexRoot;
        bool leafIndexBuilt = false;
        uint32_t nextPseudoLbn = kCdPseudoLbnStart;
        std::filesystem::path imageSizePath;
        uint64_t imageSizeBytes = 0u;
        bool imageSizeValid = false;
        int32_t lastError = 0;
        uint32_t mode = 0u;
        uint32_t streamingLbn = 0u;
        uint32_t streamingEndLbn = 0xFFFFFFFFu;
        bool initialized = false;

        uint32_t stReadTraceCount = 0u;
        uint32_t readRecoverLogCount = 0u;
        uint32_t unresolvedReadLogCount = 0u;
        uint32_t searchTraceCount = 0u;
        uint32_t emptyPathCount = 0u;
        uint32_t emptyNormalizedCount = 0u;
        std::string lastFailedPath;
        uint32_t samePathFailCount = 0u;
    };
}
