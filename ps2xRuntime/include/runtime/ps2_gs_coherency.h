#pragma once

#include "runtime/ps2_gs_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>

enum class GsVramPageOwnership : uint8_t
{
    Synchronized,
    CpuNewer,
    GpuNewer,
};

struct GsVramPageVersion
{
    uint64_t cpuGeneration = 0u;
    uint64_t gpuGeneration = 0u;

    [[nodiscard]] GsVramPageOwnership ownership() const noexcept;

    bool operator==(const GsVramPageVersion &) const noexcept = default;
};

struct GsVramCoherencySummary
{
    size_t synchronizedPages = GS_VRAM_PAGE_COUNT;
    size_t cpuNewerPages = 0u;
    size_t gpuNewerPages = 0u;
};

struct GsVramCoherencyStatistics
{
    uint64_t cpuWriteOperations = 0u;
    uint64_t cpuWritePages = 0u;
    uint64_t gpuWriteOperations = 0u;
    uint64_t gpuWritePages = 0u;
    uint64_t cpuToGpuOperations = 0u;
    uint64_t cpuToGpuPages = 0u;
    uint64_t gpuToCpuOperations = 0u;
    uint64_t gpuToCpuPages = 0u;
    uint64_t rejectedTransitions = 0u;
};

// Tracks which copy of every physical 8 KiB GS page contains the newest
// bytes. A write is legal only after the opposite-newer pages in its complete
// affected mask have been synchronized. Every mutating operation preflights
// its entire mask, so a rejected transition cannot leave a partially updated
// map.
class GsVramCoherency final
{
public:
    GsVramCoherency() = default;

    void reset() noexcept;
    void resetStatistics() noexcept;

    [[nodiscard]] const GsVramPageVersion &page(
        size_t pageIndex) const;
    [[nodiscard]] const std::array<
        GsVramPageVersion, GS_VRAM_PAGE_COUNT> &pages() const noexcept;

    [[nodiscard]] GsVramPageMask cpuNewerPages(
        const GsVramPageMask &within) const noexcept;
    [[nodiscard]] GsVramPageMask gpuNewerPages(
        const GsVramPageMask &within) const noexcept;
    [[nodiscard]] GsVramCoherencySummary summary() const noexcept;
    [[nodiscard]] bool invariantHolds() const noexcept;

    void noteCpuWrite(const GsVramPageMask &pages);
    void noteGpuWrite(const GsVramPageMask &pages);
    // Applies one already ordered resident GPU batch as a single ownership
    // transition. operationCount and pageTouches retain the per-command
    // diagnostics without rescanning all 512 pages for every draw.
    void noteGpuWriteBatch(
        const GsVramPageMask &pages,
        uint64_t operationCount,
        uint64_t pageTouches);
    void completeCpuToGpu(const GsVramPageMask &pages);
    void completeGpuToCpu(const GsVramPageMask &pages);

    [[nodiscard]] const GsVramCoherencyStatistics &statistics() const noexcept;

private:
    [[nodiscard]] uint64_t allocateGeneration();
    [[noreturn]] void reject(const char *message);

    std::array<GsVramPageVersion, GS_VRAM_PAGE_COUNT> m_pages{};
    GsVramCoherencyStatistics m_statistics{};
    uint64_t m_nextGeneration = 1u;
};
