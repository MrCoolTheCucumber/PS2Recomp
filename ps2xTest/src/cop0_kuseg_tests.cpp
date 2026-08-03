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

            t.IsFalse(
                Ps2CanUseFastGuestRdramAccess(
                    &normalCtx, erlAddress, 4u),
                "normal kuseg must not use the generated RDRAM fast path");
            t.IsTrue(
                Ps2CanUseFastGuestRdramAccess(
                    &erlCtx, erlAddress, 4u),
                "ERL kuseg may use the direct RDRAM fast path");
            t.IsTrue(
                Ps2CanUseFastGuestRdramAccess(
                    &normalCtx, uncachedAddress, 4u),
                "the explicit uncached mirror may retain its fast path");
            t.IsTrue(
                Ps2CanUseFastGuestRdramAccess(
                    &normalCtx, acceleratedAddress, 4u),
                "the explicit accelerated mirror may retain its fast path");
        });
    });
}
