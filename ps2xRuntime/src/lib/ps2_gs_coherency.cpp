#include "runtime/ps2_gs_coherency.h"

#include <limits>
#include <stdexcept>

GsVramPageOwnership GsVramPageVersion::ownership() const noexcept
{
    if (cpuGeneration == gpuGeneration)
        return GsVramPageOwnership::Synchronized;
    return cpuGeneration > gpuGeneration
        ? GsVramPageOwnership::CpuNewer
        : GsVramPageOwnership::GpuNewer;
}

void GsVramCoherency::reset() noexcept
{
    m_pages = {};
    m_statistics = {};
    m_nextGeneration = 1u;
}

void GsVramCoherency::resetStatistics() noexcept
{
    m_statistics = {};
}

const GsVramPageVersion &GsVramCoherency::page(
    size_t pageIndex) const
{
    if (pageIndex >= m_pages.size())
        throw std::out_of_range("GS VRAM coherency page is out of range");
    return m_pages[pageIndex];
}

const std::array<GsVramPageVersion, GS_VRAM_PAGE_COUNT> &
GsVramCoherency::pages() const noexcept
{
    return m_pages;
}

GsVramPageMask GsVramCoherency::cpuNewerPages(
    const GsVramPageMask &within) const noexcept
{
    GsVramPageMask result;
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (within.test(pageIndex) &&
            m_pages[pageIndex].ownership() ==
                GsVramPageOwnership::CpuNewer)
        {
            result.set(pageIndex);
        }
    }
    return result;
}

GsVramPageMask GsVramCoherency::gpuNewerPages(
    const GsVramPageMask &within) const noexcept
{
    GsVramPageMask result;
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (within.test(pageIndex) &&
            m_pages[pageIndex].ownership() ==
                GsVramPageOwnership::GpuNewer)
        {
            result.set(pageIndex);
        }
    }
    return result;
}

GsVramCoherencySummary GsVramCoherency::summary() const noexcept
{
    GsVramCoherencySummary result{};
    result.synchronizedPages = 0u;
    for (const GsVramPageVersion &pageVersion : m_pages)
    {
        switch (pageVersion.ownership())
        {
        case GsVramPageOwnership::Synchronized:
            ++result.synchronizedPages;
            break;
        case GsVramPageOwnership::CpuNewer:
            ++result.cpuNewerPages;
            break;
        case GsVramPageOwnership::GpuNewer:
            ++result.gpuNewerPages;
            break;
        }
    }
    return result;
}

bool GsVramCoherency::invariantHolds() const noexcept
{
    if (m_nextGeneration == 0u)
        return false;
    for (const GsVramPageVersion &pageVersion : m_pages)
    {
        if (pageVersion.cpuGeneration >= m_nextGeneration ||
            pageVersion.gpuGeneration >= m_nextGeneration)
        {
            return false;
        }
    }
    return true;
}

void GsVramCoherency::noteCpuWrite(
    const GsVramPageMask &pages)
{
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (pages.test(pageIndex) &&
            m_pages[pageIndex].ownership() ==
                GsVramPageOwnership::GpuNewer)
        {
            reject("CPU write overlaps a GPU-newer GS VRAM page");
        }
    }
    if (!pages.any())
        return;

    const uint64_t generation = allocateGeneration();
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (pages.test(pageIndex))
            m_pages[pageIndex].cpuGeneration = generation;
    }
    ++m_statistics.cpuWriteOperations;
    m_statistics.cpuWritePages += pages.count();
}

void GsVramCoherency::noteGpuWrite(
    const GsVramPageMask &pages)
{
    noteGpuWriteBatch(pages, 1u, pages.count());
}

void GsVramCoherency::noteGpuWriteBatch(
    const GsVramPageMask &pages,
    uint64_t operationCount,
    uint64_t pageTouches)
{
    const size_t uniquePages = pages.count();
    if (uniquePages == 0u)
        return;
    if (operationCount == 0u || pageTouches < uniquePages)
    {
        throw std::invalid_argument(
            "GPU write batch diagnostics do not describe its page mask");
    }
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (pages.test(pageIndex) &&
            m_pages[pageIndex].ownership() ==
                GsVramPageOwnership::CpuNewer)
        {
            reject("GPU write overlaps a CPU-newer GS VRAM page");
        }
    }
    const uint64_t generation = allocateGeneration();
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (pages.test(pageIndex))
            m_pages[pageIndex].gpuGeneration = generation;
    }
    m_statistics.gpuWriteOperations += operationCount;
    m_statistics.gpuWritePages += pageTouches;
}

void GsVramCoherency::completeCpuToGpu(
    const GsVramPageMask &pages)
{
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (pages.test(pageIndex) &&
            m_pages[pageIndex].ownership() ==
                GsVramPageOwnership::GpuNewer)
        {
            reject("CPU-to-GPU copy would overwrite a GPU-newer GS VRAM page");
        }
    }
    if (!pages.any())
        return;

    uint64_t copiedPages = 0u;
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        GsVramPageVersion &pageVersion = m_pages[pageIndex];
        if (!pages.test(pageIndex) ||
            pageVersion.cpuGeneration == pageVersion.gpuGeneration)
        {
            continue;
        }
        pageVersion.gpuGeneration = pageVersion.cpuGeneration;
        ++copiedPages;
    }
    ++m_statistics.cpuToGpuOperations;
    m_statistics.cpuToGpuPages += copiedPages;
}

void GsVramCoherency::completeGpuToCpu(
    const GsVramPageMask &pages)
{
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        if (pages.test(pageIndex) &&
            m_pages[pageIndex].ownership() ==
                GsVramPageOwnership::CpuNewer)
        {
            reject("GPU-to-CPU copy would overwrite a CPU-newer GS VRAM page");
        }
    }
    if (!pages.any())
        return;

    uint64_t copiedPages = 0u;
    for (size_t pageIndex = 0u; pageIndex < m_pages.size(); ++pageIndex)
    {
        GsVramPageVersion &pageVersion = m_pages[pageIndex];
        if (!pages.test(pageIndex) ||
            pageVersion.cpuGeneration == pageVersion.gpuGeneration)
        {
            continue;
        }
        pageVersion.cpuGeneration = pageVersion.gpuGeneration;
        ++copiedPages;
    }
    ++m_statistics.gpuToCpuOperations;
    m_statistics.gpuToCpuPages += copiedPages;
}

const GsVramCoherencyStatistics &
GsVramCoherency::statistics() const noexcept
{
    return m_statistics;
}

uint64_t GsVramCoherency::allocateGeneration()
{
    if (m_nextGeneration == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("GS VRAM coherency generation exhausted");
    return m_nextGeneration++;
}

void GsVramCoherency::reject(const char *message)
{
    ++m_statistics.rejectedTransitions;
    throw std::logic_error(message);
}
