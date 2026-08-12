#include "MiniTest.h"
#include "runtime/ps2_gs_coherency.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{
    uint32_t nextRandom(uint32_t &state)
    {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }

    GsVramPageMask makePageMask(
        uint32_t &randomState, size_t maximumPages = 32u)
    {
        GsVramPageMask mask;
        const size_t pageCount =
            1u + nextRandom(randomState) % maximumPages;
        const size_t first =
            nextRandom(randomState) % GS_VRAM_PAGE_COUNT;
        const size_t stride =
            (nextRandom(randomState) | 1u) % GS_VRAM_PAGE_COUNT;
        for (size_t index = 0u; index < pageCount; ++index)
            mask.set((first + index * stride) % GS_VRAM_PAGE_COUNT);
        return mask;
    }

    GsVramPageMask ownershipMask(
        const std::array<
            GsVramPageOwnership, GS_VRAM_PAGE_COUNT> &model,
        GsVramPageOwnership ownership,
        const GsVramPageMask &within)
    {
        GsVramPageMask mask;
        for (size_t page = 0u; page < model.size(); ++page)
        {
            if (within.test(page) && model[page] == ownership)
                mask.set(page);
        }
        return mask;
    }

    bool throwsLogicError(const auto &operation)
    {
        try
        {
            operation();
        }
        catch (const std::logic_error &)
        {
            return true;
        }
        return false;
    }
}

void register_ps2_gs_coherency_tests()
{
    MiniTest::Case("PS2GSCoherency", [](TestCase &tc)
    {
        tc.Run("page ownership transitions are atomic and fail fast", [](TestCase &t)
        {
            GsVramCoherency coherency;
            GsVramPageMask cpuPages;
            cpuPages.set(0u);
            cpuPages.set(1u);
            cpuPages.set(511u);
            coherency.noteCpuWrite(cpuPages);

            t.Equals(coherency.summary().cpuNewerPages,
                     static_cast<size_t>(3u),
                     "a wrapped CPU write should own all named pages");
            t.Equals(coherency.page(0u).cpuGeneration,
                     coherency.page(511u).cpuGeneration,
                     "one write operation should publish one generation");
            t.IsTrue(coherency.cpuNewerPages(cpuPages) == cpuPages,
                     "the CPU upload query should return the exact stale-device pages");

            const auto beforeRejectedGpuWrite = coherency.pages();
            t.IsTrue(throwsLogicError([&]()
                     {
                         coherency.noteGpuWrite(cpuPages);
                     }),
                     "an unmerged GPU writer should fail immediately");
            t.IsTrue(coherency.pages() == beforeRejectedGpuWrite,
                     "a rejected writer must preserve the complete map");

            GsVramPageMask twoPages;
            twoPages.set(0u);
            twoPages.set(511u);
            coherency.completeCpuToGpu(twoPages);
            t.Equals(coherency.page(0u).ownership(),
                     GsVramPageOwnership::Synchronized,
                     "an uploaded page should become synchronized");
            t.Equals(coherency.page(1u).ownership(),
                     GsVramPageOwnership::CpuNewer,
                     "an omitted CPU page should remain newer");

            coherency.noteGpuWrite(twoPages);
            t.Equals(coherency.summary().gpuNewerPages,
                     static_cast<size_t>(2u),
                     "a synchronized subset should accept a GPU writer");
            const auto beforeRejectedDownload = coherency.pages();
            GsVramPageMask mixedCopy;
            mixedCopy.set(0u);
            mixedCopy.set(1u);
            t.IsTrue(throwsLogicError([&]()
                     {
                         coherency.completeGpuToCpu(mixedCopy);
                     }),
                     "a mixed stale-source download should reject its whole span");
            t.IsTrue(coherency.pages() == beforeRejectedDownload,
                     "a rejected mixed download must not synchronize an earlier page");

            GsVramPageMask gpuPage;
            gpuPage.set(0u);
            const auto beforeRejectedUpload = coherency.pages();
            t.IsTrue(throwsLogicError([&]()
                     {
                         coherency.completeCpuToGpu(gpuPage);
                     }),
                     "a stale CPU upload should fail before overwriting GPU ownership");
            t.IsTrue(coherency.pages() == beforeRejectedUpload,
                     "a rejected stale upload must preserve every version");

            coherency.completeGpuToCpu(twoPages);
            GsVramPageMask remainingCpuPage;
            remainingCpuPage.set(1u);
            coherency.completeCpuToGpu(remainingCpuPage);
            const GsVramCoherencySummary summary = coherency.summary();
            t.Equals(summary.synchronizedPages,
                     static_cast<size_t>(GS_VRAM_PAGE_COUNT),
                     "opposite disjoint copies should restore a synchronized image");
            t.Equals(summary.cpuNewerPages, static_cast<size_t>(0u),
                     "no CPU-newer page should remain after the forced boundary");
            t.Equals(summary.gpuNewerPages, static_cast<size_t>(0u),
                     "no GPU-newer page should remain after the forced boundary");
            t.IsTrue(coherency.invariantHolds(),
                     "all accepted and rejected transitions should preserve invariants");

            const GsVramCoherencyStatistics statistics =
                coherency.statistics();
            t.Equals(statistics.cpuWriteOperations, 1ull,
                     "the initial CPU write should count once");
            t.Equals(statistics.gpuWriteOperations, 1ull,
                     "the accepted GPU write should count once");
            t.Equals(statistics.cpuToGpuPages, 3ull,
                     "uploads should count only pages whose versions changed");
            t.Equals(statistics.gpuToCpuPages, 2ull,
                     "downloads should count only GPU-newer pages");
            t.Equals(statistics.rejectedTransitions, 3ull,
                     "each conflicting whole-span transition should be counted");

            t.IsTrue(throwsLogicError([&]()
                     {
                         GsVramPageMask conflict;
                         conflict.set(0u);
                         coherency.noteCpuWrite(conflict);
                         coherency.noteGpuWrite(conflict);
                     }),
                     "a second writer should remain illegal after returning to synchronization");
            t.IsTrue(coherency.invariantHolds(),
                     "the final rejected writer should retain a valid map");
        });

        tc.Run("ordered GPU batches retain per-command diagnostics", [](TestCase &t)
        {
            GsVramCoherency coherency;
            GsVramPageMask pages;
            pages.set(7u);
            pages.set(8u);
            pages.set(511u);

            coherency.noteGpuWriteBatch(pages, 19u, 43u);

            t.IsTrue(coherency.gpuNewerPages(pages) == pages,
                     "one batch transition should own the complete write union");
            t.Equals(coherency.page(7u).gpuGeneration,
                     coherency.page(511u).gpuGeneration,
                     "an ordered batch should publish one final GPU generation");
            const GsVramCoherencyStatistics statistics =
                coherency.statistics();
            t.Equals(statistics.gpuWriteOperations, 19ull,
                     "batch ownership should retain its architectural draw count");
            t.Equals(statistics.gpuWritePages, 43ull,
                     "batch ownership should retain repeated per-draw page touches");
            t.IsTrue(coherency.invariantHolds(),
                     "the compact batch transition should preserve invariants");
        });

        tc.Run("fixed-seed access streams match an independent ownership model", [](TestCase &t)
        {
            GsVramCoherency coherency;
            std::array<
                GsVramPageOwnership, GS_VRAM_PAGE_COUNT> model{};
            model.fill(GsVramPageOwnership::Synchronized);
            GsVramPageMask allPages;
            allPages.setAll();
            uint32_t randomState = 0x50414745u;

            constexpr size_t operationCount = 20000u;
            for (size_t operation = 0u;
                 operation < operationCount; ++operation)
            {
                const uint32_t kind = nextRandom(randomState) % 6u;
                const GsVramPageMask pages =
                    makePageMask(randomState);
                if (kind == 0u || kind == 4u)
                {
                    const GsVramPageMask downloads = ownershipMask(
                        model, GsVramPageOwnership::GpuNewer, pages);
                    coherency.completeGpuToCpu(downloads);
                    for (size_t page = 0u; page < model.size(); ++page)
                    {
                        if (downloads.test(page))
                            model[page] =
                                GsVramPageOwnership::Synchronized;
                    }
                    coherency.noteCpuWrite(pages);
                    for (size_t page = 0u; page < model.size(); ++page)
                    {
                        if (pages.test(page))
                            model[page] = GsVramPageOwnership::CpuNewer;
                    }
                }
                else if (kind == 1u)
                {
                    const GsVramPageMask uploads = ownershipMask(
                        model, GsVramPageOwnership::CpuNewer, pages);
                    coherency.completeCpuToGpu(uploads);
                    for (size_t page = 0u; page < model.size(); ++page)
                    {
                        if (uploads.test(page))
                            model[page] =
                                GsVramPageOwnership::Synchronized;
                    }
                    coherency.noteGpuWrite(pages);
                    for (size_t page = 0u; page < model.size(); ++page)
                    {
                        if (pages.test(page))
                            model[page] = GsVramPageOwnership::GpuNewer;
                    }
                }
                else if (kind == 2u)
                {
                    const GsVramPageMask downloads = ownershipMask(
                        model, GsVramPageOwnership::GpuNewer, pages);
                    coherency.completeGpuToCpu(downloads);
                    for (size_t page = 0u; page < model.size(); ++page)
                    {
                        if (downloads.test(page))
                            model[page] =
                                GsVramPageOwnership::Synchronized;
                    }
                }
                else if (kind == 3u)
                {
                    const GsVramPageMask uploads = ownershipMask(
                        model, GsVramPageOwnership::CpuNewer, pages);
                    coherency.completeCpuToGpu(uploads);
                    for (size_t page = 0u; page < model.size(); ++page)
                    {
                        if (uploads.test(page))
                            model[page] =
                                GsVramPageOwnership::Synchronized;
                    }
                }
                else
                {
                    const GsVramPageMask downloads = ownershipMask(
                        model, GsVramPageOwnership::GpuNewer, allPages);
                    const GsVramPageMask uploads = ownershipMask(
                        model, GsVramPageOwnership::CpuNewer, allPages);
                    coherency.completeGpuToCpu(downloads);
                    coherency.completeCpuToGpu(uploads);
                    model.fill(GsVramPageOwnership::Synchronized);
                }

                if (!coherency.invariantHolds())
                {
                    t.Fail("coherency invariant failed after operation " +
                           std::to_string(operation));
                    return;
                }
                for (size_t page = 0u; page < model.size(); ++page)
                {
                    const GsVramPageVersion &version =
                        coherency.page(page);
                    if (version.ownership() != model[page])
                    {
                        t.Fail("ownership model diverged after operation " +
                               std::to_string(operation) +
                               " at page " + std::to_string(page));
                        return;
                    }
                    if ((model[page] ==
                             GsVramPageOwnership::Synchronized &&
                         version.cpuGeneration != version.gpuGeneration) ||
                        (model[page] == GsVramPageOwnership::CpuNewer &&
                         version.cpuGeneration <= version.gpuGeneration) ||
                        (model[page] == GsVramPageOwnership::GpuNewer &&
                         version.gpuGeneration <= version.cpuGeneration))
                    {
                        t.Fail("generation ordering diverged after operation " +
                               std::to_string(operation) +
                               " at page " + std::to_string(page));
                        return;
                    }
                }

                const GsVramPageMask expectedCpu = ownershipMask(
                    model, GsVramPageOwnership::CpuNewer, allPages);
                const GsVramPageMask expectedGpu = ownershipMask(
                    model, GsVramPageOwnership::GpuNewer, allPages);
                if (coherency.cpuNewerPages(allPages) != expectedCpu ||
                    coherency.gpuNewerPages(allPages) != expectedGpu)
                {
                    t.Fail("transfer query diverged after operation " +
                           std::to_string(operation));
                    return;
                }
            }

            const GsVramCoherencySummary summary = coherency.summary();
            t.Equals(summary.synchronizedPages +
                         summary.cpuNewerPages +
                         summary.gpuNewerPages,
                     static_cast<size_t>(GS_VRAM_PAGE_COUNT),
                     "the final ownership partition should cover all pages");
            t.Equals(coherency.statistics().rejectedTransitions, 0ull,
                     "the model should synchronize every conflicting writer first");
        });
    });
}
