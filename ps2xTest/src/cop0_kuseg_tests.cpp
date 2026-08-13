#include "MiniTest.h"

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <cstring>
#include <cstdint>

namespace
{
    constexpr uint32_t kStatusErl = 0x00000004u;
    constexpr uint32_t kStatusUser = 0x00000010u;
    constexpr uint32_t kCauseExcCodeMask = 0x0000007cu;

    constexpr uint32_t makeEntryLo(
        uint32_t pfn,
        bool dirty,
        bool valid,
        bool global)
    {
        return ((pfn & 0x000fffffu) << 6u) |
               (2u << 3u) |
               (dirty ? 0x4u : 0u) |
               (valid ? 0x2u : 0u) |
               (global ? 0x1u : 0u);
    }

    bool raisesGuestException(const auto &operation)
    {
        try
        {
            operation();
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }

    void clearTlb(
        TestCase &t,
        PS2Runtime &runtime,
        const char *message)
    {
        for (uint32_t index = 0u;
             index < runtime.memory().tlbEntryCount();
             ++index)
        {
            t.IsTrue(
                runtime.memory().tlbWrite(
                    index, EeTlbEntry{}),
                message);
        }
    }

    void generatedKusegLoad(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        SET_GPR_U32(ctx, 2u, READ32(0x00123000u));
    }

    void generatedKusegPairAccess(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        SET_GPR_U32(ctx, 2u, READ32(0x00400040u));
        SET_GPR_U32(ctx, 3u, READ32(0x00401040u));
        WRITE32(0x00401044u, 0xa5b6c7d8u);
    }
}

void register_cop0_kuseg_tests()
{
    MiniTest::Case("COP0 kuseg translation", [](TestCase &tc)
    {
        tc.Run("ordinary kuseg misses enter the TLB refill path", [](TestCase &t)
        {
            constexpr uint32_t virtualAddress = 0x00123000u;
            constexpr uint32_t physicalSentinel = 0xdecafbad;
            constexpr uint32_t asid = 0x35u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            clearTlb(
                t,
                runtime,
                "the refill fixture should clear every TLB entry");
            std::memcpy(
                runtime.memory().getRDRAM() + virtualAddress,
                &physicalSentinel,
                sizeof(physicalSentinel));

            R5900Context ctx{};
            ctx.pc = 0x00150000u;
            ctx.cop0_entryhi = asid;
            const bool raised = raisesGuestException([&]()
            {
                generatedKusegLoad(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    &runtime);
            });

            t.IsTrue(
                raised,
                "an unmapped ordinary kuseg load should raise a refill");
            t.Equals(
                ctx.pc,
                0x80000000u,
                "a first-level kuseg miss should enter the refill vector");
            t.Equals(
                ctx.cop0_cause & kCauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD)
                 << 2u) &
                    kCauseExcCodeMask,
                "the miss should publish the TLBL cause code");
            t.Equals(
                ctx.cop0_badvaddr,
                virtualAddress,
                "the refill should publish the original kuseg address");
            t.Equals(
                ctx.cop0_entryhi,
                (virtualAddress & 0xffffe000u) | asid,
                "the refill should publish VPN2 while preserving ASID");
            t.Equals(
                GPR_U32((&ctx), 2u),
                0u,
                "the failed load should not expose identity-mapped RDRAM");
        });

        tc.Run("mapped kuseg uses paired PFNs and the live ASID", [](TestCase &t)
        {
            constexpr uint32_t virtualBase = 0x00400000u;
            constexpr uint32_t asid = 0x46u;
            constexpr uint32_t evenPfn = 0x00800u;
            constexpr uint32_t oddPfn = 0x00900u;
            constexpr uint32_t evenValue = 0x11223344u;
            constexpr uint32_t oddValue = 0x55667788u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            clearTlb(
                t,
                runtime,
                "the paired mapping fixture should clear every TLB entry");
            t.IsTrue(
                runtime.memory().tlbWrite(
                    12u,
                    EeTlbEntry{
                        0u,
                        virtualBase | asid,
                        makeEntryLo(
                            evenPfn, true, true, false),
                        makeEntryLo(
                            oddPfn, true, true, false),
                    }),
                "the paired kuseg entry should be writable");
            std::memcpy(
                runtime.memory().getRDRAM() +
                    (evenPfn << 12u) + 0x40u,
                &evenValue,
                sizeof(evenValue));
            std::memcpy(
                runtime.memory().getRDRAM() +
                    (oddPfn << 12u) + 0x40u,
                &oddValue,
                sizeof(oddValue));

            R5900Context ctx{};
            ctx.cop0_status = kStatusUser;
            ctx.cop0_entryhi = asid;
            runtime.memory().setTlbTranslationCacheDiagnosticsEnabled(
                true);
            runtime.memory().resetTlbTranslationCacheStats();
            const bool mappedFault = raisesGuestException([&]()
            {
                generatedKusegPairAccess(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    &runtime);
            });
            t.IsFalse(
                mappedFault,
                "a valid user-mode kuseg pair should translate");
            t.Equals(
                GPR_U32((&ctx), 2u),
                evenValue,
                "the even kuseg page should select EntryLo0.PFN");
            t.Equals(
                GPR_U32((&ctx), 3u),
                oddValue,
                "the odd kuseg page should select EntryLo1.PFN");
            uint32_t storedValue = 0u;
            std::memcpy(
                &storedValue,
                runtime.memory().getRDRAM() +
                    (oddPfn << 12u) + 0x44u,
                sizeof(storedValue));
            t.Equals(
                storedValue,
                0xa5b6c7d8u,
                "the mapped kuseg store should target the odd PFN");

            uint32_t cachedPhysicalOffset = 0u;
            t.IsTrue(
                Ps2ResolveFastGuestRdramAccess(
                    &runtime,
                    &ctx,
                    virtualBase + 0x1044u,
                    sizeof(uint32_t),
                    true,
                    cachedPhysicalOffset),
                "a successful mapped access should expose its cached RDRAM offset");
            t.Equals(
                cachedPhysicalOffset,
                (oddPfn << 12u) + 0x44u,
                "the mapped fast resolver must return the non-identity odd PFN");

            runtime.memory().setTlbTranslationCacheDiagnosticsEnabled(
                false);
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &ctx, virtualBase + 0x40u);
            runtime.memory().setTlbTranslationCacheDiagnosticsEnabled(
                true);
            runtime.memory().resetTlbTranslationCacheStats();
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &ctx, virtualBase + 0x44u);
            const EeTlbTranslationCacheStats fetchStats =
                runtime.memory().tlbTranslationCacheStats();
            t.Equals(
                fetchStats.hits,
                1ull,
                "fast instruction-fetch validation should use the mapped cache");
            t.Equals(
                fetchStats.misses,
                0ull,
                "a warm mapped instruction fetch should avoid the translator");

            R5900Context sharedContext{};
            sharedContext.cop0_status = kStatusUser;
            sharedContext.cop0_entryhi = asid;
            uint32_t sharedPhysicalOffset = 0u;
            t.IsTrue(
                Ps2ResolveFastGuestRdramAccess(
                    &runtime,
                    &sharedContext,
                    virtualBase + 0x40u,
                    sizeof(uint32_t),
                    false,
                    sharedPhysicalOffset),
                "a second guest context should share the processor TLB cache");
            t.Equals(
                sharedPhysicalOffset,
                (evenPfn << 12u) + 0x40u,
                "shared contexts should resolve the same processor-global PFN");

            R5900Context wrongAsid{};
            wrongAsid.pc = 0x00150040u;
            wrongAsid.cop0_status = kStatusUser;
            wrongAsid.cop0_entryhi = asid ^ 1u;
            const bool asidMiss = raisesGuestException([&]()
            {
                (void)runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &wrongAsid,
                    virtualBase);
            });
            t.IsTrue(
                asidMiss,
                "ordinary kuseg should not bypass a mismatching ASID");
            t.Equals(
                wrongAsid.pc,
                0x80000000u,
                "an ASID mismatch should use the refill vector");
        });

        tc.Run("mapped fetch cache preserves page-end and stale-fault boundaries", [](TestCase &t)
        {
            constexpr uint32_t virtualBase = 0x01200000u;
            constexpr uint32_t physicalPfn = 0x00500u;
            constexpr uint32_t asid = 0x2au;
            constexpr uint32_t pageEndFetch =
                virtualBase + 0x0ffcu;
            constexpr uint32_t crossingFetch =
                virtualBase + 0x0ffeu;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the fetch-boundary fixture should allocate RDRAM");
            clearTlb(
                t,
                runtime,
                "the fetch-boundary fixture should clear every TLB entry");
            t.IsTrue(
                runtime.memory().tlbWrite(
                    14u,
                    EeTlbEntry{
                        0u,
                        virtualBase | asid,
                        makeEntryLo(
                            physicalPfn, true, true, false),
                        makeEntryLo(
                            physicalPfn + 1u, true, true, false),
                    }),
                "the fetch-boundary fixture should install its mapping");

            R5900Context ctx{};
            ctx.cop0_status = kStatusUser;
            ctx.cop0_entryhi = asid;
            (void)runtime.Load32(
                runtime.memory().getRDRAM(),
                &ctx,
                pageEndFetch);
            runtime.memory().setTlbTranslationCacheDiagnosticsEnabled(
                true);
            runtime.memory().resetTlbTranslationCacheStats();
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &ctx, pageEndFetch);
            t.Equals(
                runtime.memory().tlbTranslationCacheStats().hits,
                1ull,
                "an aligned fetch ending at the cached page boundary should hit");

            R5900Context crossingCtx{};
            crossingCtx.cop0_status = kStatusUser;
            crossingCtx.cop0_entryhi = asid;
            const bool crossingRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &crossingCtx, crossingFetch);
            });
            t.IsTrue(
                crossingRaised,
                "a page-crossing unaligned fetch should retain its address fault");
            t.Equals(
                crossingCtx.cop0_badvaddr,
                crossingFetch,
                "the fetch address fault should retain the original virtual address");
            t.Equals(
                crossingCtx.cop0_cause & kCauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_ADDRESS_ERROR_LOAD)
                 << 2u) &
                    kCauseExcCodeMask,
                "the crossing fetch should publish AdEL rather than use a partial hit");

            t.IsTrue(
                runtime.memory().tlbWrite(
                    14u,
                    EeTlbEntry{
                        0u,
                        virtualBase | asid,
                        makeEntryLo(
                            physicalPfn, true, false, false),
                        makeEntryLo(
                            physicalPfn + 1u, true, true, false),
                    }),
                "the fetch-boundary fixture should replace the cached page with invalid");
            R5900Context invalidCtx{};
            invalidCtx.cop0_status = kStatusUser;
            invalidCtx.cop0_entryhi = asid;
            const bool invalidRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &invalidCtx, pageEndFetch);
            });
            t.IsTrue(
                invalidRaised,
                "an invalid replacement should fault after a prior fetch hit");
            t.Equals(
                invalidCtx.cop0_badvaddr,
                pageEndFetch,
                "the stale fetch fault should retain the original virtual address");
            t.Equals(
                invalidCtx.cop0_cause & kCauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD)
                 << 2u) &
                    kCauseExcCodeMask,
                "the invalid fetch should publish the architectural TLBL code");
        });

        tc.Run("instruction fetch page cache tracks page ASID and TLB generation", [](TestCase &t)
        {
            constexpr uint32_t virtualBase = 0x01400000u;
            constexpr uint32_t physicalPfn = 0x00600u;
            constexpr uint32_t asid = 0x3cu;
            constexpr uint32_t evenFetch =
                virtualBase + 0x40u;
            constexpr uint32_t oddFetch =
                virtualBase + 0x1040u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the fetch-page fixture should allocate RDRAM");
            clearTlb(
                t,
                runtime,
                "the fetch-page fixture should clear every TLB entry");
            t.IsTrue(
                runtime.memory().tlbWrite(
                    15u,
                    EeTlbEntry{
                        0u,
                        virtualBase | asid,
                        makeEntryLo(
                            physicalPfn, true, true, false),
                        makeEntryLo(
                            physicalPfn + 1u, true, false, false),
                    }),
                "the fetch-page fixture should install one valid page");

            R5900Context asidCtx{};
            asidCtx.cop0_status = kStatusUser;
            asidCtx.cop0_entryhi = asid;
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &asidCtx, evenFetch);
            asidCtx.cop0_entryhi = asid ^ 1u;
            const bool asidRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &asidCtx, evenFetch + 4u);
            });
            t.IsTrue(
                asidRaised,
                "an ASID change must reject a cached non-global page");
            t.Equals(
                asidCtx.cop0_badvaddr,
                evenFetch + 4u,
                "the ASID fault should retain the requested fetch address");
            t.Equals(
                asidCtx.cop0_cause & kCauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD)
                 << 2u) &
                    kCauseExcCodeMask,
                "the ASID fault should publish TLBL");

            R5900Context pageCtx{};
            pageCtx.cop0_status = kStatusUser;
            pageCtx.cop0_entryhi = asid;
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &pageCtx, evenFetch);
            const bool pageRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &pageCtx, oddFetch);
            });
            t.IsTrue(
                pageRaised,
                "an adjacent invalid page must not reuse the cached page");
            t.Equals(
                pageCtx.cop0_badvaddr,
                oddFetch,
                "the adjacent-page fault should retain its fetch address");
            t.Equals(
                pageCtx.cop0_cause & kCauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD)
                 << 2u) &
                    kCauseExcCodeMask,
                "the adjacent-page fault should publish TLBL");

            R5900Context generationCtx{};
            generationCtx.cop0_status = kStatusUser;
            generationCtx.cop0_entryhi = asid;
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &generationCtx, evenFetch);
            t.IsTrue(
                runtime.memory().tlbWrite(
                    15u,
                    EeTlbEntry{
                        0u,
                        virtualBase | asid,
                        makeEntryLo(
                            physicalPfn, true, false, false),
                        makeEntryLo(
                            physicalPfn + 1u, true, false, false),
                    }),
                "the fetch-page fixture should invalidate the cached mapping");
            const bool generationRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &generationCtx, evenFetch + 8u);
            });
            t.IsTrue(
                generationRaised,
                "a TLB write must invalidate a same-context fetch page");
            t.Equals(
                generationCtx.cop0_badvaddr,
                evenFetch + 8u,
                "the generation fault should retain its fetch address");
            t.Equals(
                generationCtx.cop0_cause & kCauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD)
                 << 2u) &
                    kCauseExcCodeMask,
                "the generation fault should publish TLBL");
        });

        tc.Run("ERL and explicit compatibility aliases remain direct", [](TestCase &t)
        {
            constexpr uint32_t erlAddress = 0x00102000u;
            constexpr uint32_t uncachedAddress = 0x20002000u;
            constexpr uint32_t acceleratedAddress = 0x30103000u;
            constexpr uint32_t deviceAddress = 0x10000000u;
            constexpr uint32_t scratchAddress = 0x70000100u;
            constexpr uint32_t erlValue = 0x01020304u;
            constexpr uint32_t uncachedValue = 0x11223344u;
            constexpr uint32_t acceleratedValue = 0x55667788u;
            constexpr uint32_t scratchValue = 0x99aabbccu;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            std::memcpy(
                runtime.memory().getRDRAM() + erlAddress,
                &erlValue,
                sizeof(erlValue));
            std::memcpy(
                runtime.memory().getRDRAM() + 0x2000u,
                &uncachedValue,
                sizeof(uncachedValue));
            std::memcpy(
                runtime.memory().getRDRAM() + 0x00103000u,
                &acceleratedValue,
                sizeof(acceleratedValue));
            std::memcpy(
                runtime.memory().getScratchpad() + 0x100u,
                &scratchValue,
                sizeof(scratchValue));

            R5900Context erlCtx{};
            erlCtx.cop0_status = kStatusErl;
            t.Equals(
                runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &erlCtx,
                    erlAddress),
                erlValue,
                "Status.ERL should make ordinary kuseg unmapped");

            R5900Context normalCtx{};
            t.Equals(
                runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &normalCtx,
                    uncachedAddress),
                uncachedValue,
                "the explicit 0x2000 RDRAM mirror should remain direct");
            t.Equals(
                runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &normalCtx,
                    acceleratedAddress),
                acceleratedValue,
                "the explicit 0x3010 RDRAM mirror should remain direct");
            t.Equals(
                runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &normalCtx,
                    scratchAddress),
                scratchValue,
                "the explicit scratchpad address should remain direct");
            t.Equals(
                runtime.memory().translateAddress(erlAddress),
                erlAddress,
                "unchecked host-internal translation should remain physical-style");
            t.Equals(
                runtime.memory().translateAddress(
                    deviceAddress,
                    EeAddressTranslationContext::fromCop0Status(
                        normalCtx.cop0_status)),
                deviceAddress,
                "the explicit physical device alias should remain direct");

            uint32_t physicalOffset = 0u;
            t.IsFalse(
                Ps2ResolveFastGuestRdramAccess(
                    &runtime,
                    &normalCtx,
                    erlAddress,
                    4u,
                    false,
                    physicalOffset),
                "normal kuseg must not use the generated RDRAM fast path");
            t.IsTrue(
                Ps2ResolveFastGuestRdramAccess(
                    &runtime,
                    &erlCtx,
                    erlAddress,
                    4u,
                    false,
                    physicalOffset),
                "ERL kuseg may use the direct RDRAM fast path");
            t.Equals(
                physicalOffset,
                erlAddress,
                "the ERL direct path should return its physical RDRAM offset");
            t.IsTrue(
                Ps2ResolveFastGuestRdramAccess(
                    &runtime,
                    &normalCtx,
                    uncachedAddress,
                    4u,
                    false,
                    physicalOffset),
                "the explicit uncached mirror may retain its fast path");
            t.Equals(
                physicalOffset,
                0x2000u,
                "the uncached mirror should return a physical RDRAM offset");
            t.IsTrue(
                Ps2ResolveFastGuestRdramAccess(
                    &runtime,
                    &normalCtx,
                    acceleratedAddress,
                    4u,
                    false,
                    physicalOffset),
                "the explicit accelerated mirror may retain its fast path");
            t.Equals(
                physicalOffset,
                0x00103000u,
                "the accelerated mirror should return a physical RDRAM offset");
        });
    });
}
