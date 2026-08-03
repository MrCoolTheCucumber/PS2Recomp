#include "MiniTest.h"

#include "ps2_runtime.h"

#include <array>
#include <cstring>
#include <cstdint>
#include <string>

namespace
{
    constexpr uint32_t kCop0CauseExcCodeMask = 0x0000007cu;

    constexpr uint32_t makeEntryLo(
        uint32_t pfn,
        uint32_t cacheMode,
        bool dirty,
        bool valid,
        bool global,
        bool scratchpad = false)
    {
        return (scratchpad ? 0x80000000u : 0u) |
               ((pfn & 0x000fffffu) << 6u) |
               ((cacheMode & 0x7u) << 3u) |
               (dirty ? 0x4u : 0u) |
               (valid ? 0x2u : 0u) |
               (global ? 0x1u : 0u);
    }

    void writeIndexedEntry(
        PS2Runtime &runtime,
        R5900Context &ctx,
        uint32_t index,
        uint32_t entryHi,
        uint32_t entryLo0,
        uint32_t entryLo1,
        uint32_t pageMask = 0u)
    {
        ctx.cop0_index = index;
        ctx.cop0_entryhi = entryHi;
        ctx.cop0_entrylo0 = entryLo0;
        ctx.cop0_entrylo1 = entryLo1;
        ctx.cop0_pagemask = pageMask;
        runtime.handleTLBWI(
            runtime.memory().getRDRAM(), &ctx);
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
}

void register_cop0_tlb_tests()
{
    MiniTest::Case("COP0 TLB entries", [](TestCase &tc)
    {
        tc.Run("indexed write preserves an even and odd page pair", [](TestCase &t)
        {
            constexpr uint32_t index = 7u;
            constexpr uint32_t entryHi = 0xC100002Au;
            constexpr uint32_t entryLo0 =
                makeEntryLo(0x00100u, 2u, true, true, true);
            constexpr uint32_t entryLo1 =
                makeEntryLo(0x00200u, 3u, true, true, true);

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            uint8_t *const rdram = runtime.memory().getRDRAM();

            R5900Context ctx{};
            writeIndexedEntry(
                runtime,
                ctx,
                index,
                entryHi,
                entryLo0,
                entryLo1);

            ctx.cop0_entryhi = 0u;
            ctx.cop0_entrylo0 = 0u;
            ctx.cop0_entrylo1 = 0u;
            ctx.cop0_pagemask = 0xffffffffu;
            runtime.handleTLBR(rdram, &ctx);

            t.Equals(
                ctx.cop0_entryhi,
                entryHi,
                "TLBR should restore VPN2 and ASID");
            t.Equals(
                ctx.cop0_entrylo0,
                entryLo0,
                "TLBR should restore the even-page fields");
            t.Equals(
                ctx.cop0_entrylo1,
                entryLo1,
                "TLBR should restore the distinct odd-page fields");
            t.Equals(
                ctx.cop0_pagemask,
                0u,
                "TLBR should restore PageMask");
        });

        tc.Run("even and odd virtual pages select distinct PFNs", [](TestCase &t)
        {
            constexpr uint32_t entryHi = 0xC1000000u;
            constexpr uint32_t entryLo0 =
                makeEntryLo(0x00100u, 2u, true, true, true);
            constexpr uint32_t entryLo1 =
                makeEntryLo(0x00200u, 2u, true, true, true);

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            writeIndexedEntry(
                runtime,
                ctx,
                11u,
                entryHi,
                entryLo0,
                entryLo1);

            bool evenTranslated = false;
            bool oddTranslated = false;
            uint32_t evenPhysical = 0u;
            uint32_t oddPhysical = 0u;
            try
            {
                evenPhysical = runtime.memory().translateAddress(
                    entryHi + 0x0123u);
                evenTranslated = true;
                oddPhysical = runtime.memory().translateAddress(
                    entryHi + 0x1456u);
                oddTranslated = true;
            }
            catch (const PS2TlbMissException &)
            {
            }

            t.IsTrue(
                evenTranslated,
                "the even page should match the pair VPN2");
            t.IsTrue(
                oddTranslated,
                "the odd page should match the same pair VPN2");
            t.Equals(
                evenPhysical,
                0x00100123u,
                "the even page should select EntryLo0.PFN");
            t.Equals(
                oddPhysical,
                0x00200456u,
                "the odd page should select EntryLo1.PFN");
        });

        tc.Run("PageMask selects large even and odd pages", [](TestCase &t)
        {
            constexpr uint32_t entryHi = 0xC4000000u;
            constexpr uint32_t pageMask = 0x00006000u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            writeIndexedEntry(
                runtime,
                ctx,
                20u,
                entryHi,
                makeEntryLo(
                    0x00123u, 2u, true, true, true),
                makeEntryLo(
                    0x00257u, 2u, true, true, true),
                pageMask);

            bool translated = false;
            uint32_t evenPhysical = 0u;
            uint32_t oddPhysical = 0u;
            try
            {
                evenPhysical = runtime.memory().translateAddress(
                    entryHi + 0x2abcu);
                oddPhysical = runtime.memory().translateAddress(
                    entryHi + 0x5abcu);
                translated = true;
            }
            catch (const PS2TlbMissException &)
            {
            }

            t.IsTrue(
                translated,
                "both 16 KiB pages should match one masked VPN2");
            t.Equals(
                evenPhysical,
                0x00122abcu,
                "the even mapping should mask PFN low bits and retain its page offset");
            t.Equals(
                oddPhysical,
                0x00255abcu,
                "the odd mapping should select EntryLo1 at the page-size boundary");
        });

        tc.Run("all defined PageMask sizes preserve page boundaries", [](TestCase &t)
        {
            constexpr std::array<uint32_t, 7u> maskFields{
                0x000u,
                0x003u,
                0x00fu,
                0x03fu,
                0x0ffu,
                0x3ffu,
                0xfffu,
            };
            constexpr uint32_t virtualBase = 0xCA000000u;
            constexpr uint32_t evenPfn = 0x54321u;
            constexpr uint32_t oddPfn = 0xabcdeu;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};

            for (const uint32_t maskField : maskFields)
            {
                const uint32_t pageMask =
                    maskField << 13u;
                const uint32_t pageSize =
                    (maskField + 1u) << 12u;
                const uint32_t pageOffset =
                    pageSize - 4u;
                writeIndexedEntry(
                    runtime,
                    ctx,
                    30u,
                    virtualBase,
                    makeEntryLo(
                        evenPfn, 2u, true, true, true),
                    makeEntryLo(
                        oddPfn, 2u, true, true, true),
                    pageMask);

                const uint32_t evenPhysical =
                    runtime.memory().translateAddress(
                        virtualBase + pageOffset);
                const uint32_t oddPhysical =
                    runtime.memory().translateAddress(
                        virtualBase + pageSize + pageOffset);
                const std::string suffix =
                    " for PageMask field " +
                    std::to_string(maskField);
                t.Equals(
                    evenPhysical,
                    ((evenPfn & ~maskField) << 12u) |
                        pageOffset,
                    "EntryLo0 should map the final even-page byte" +
                        suffix);
                t.Equals(
                    oddPhysical,
                    ((oddPfn & ~maskField) << 12u) |
                        pageOffset,
                    "EntryLo1 should map the final odd-page byte" +
                        suffix);
            }
        });

        tc.Run("indexed writes mask VPN2 and combine Global bits", [](TestCase &t)
        {
            constexpr uint32_t index = 21u;
            constexpr uint32_t pageMask = 0x00006000u;
            constexpr uint32_t entryHi = 0xC480605Au;
            constexpr uint32_t entryLo0 =
                makeEntryLo(0x00321u, 3u, true, true, true);
            constexpr uint32_t entryLo1 =
                makeEntryLo(0x00432u, 7u, true, true, false);

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            writeIndexedEntry(
                runtime,
                ctx,
                index,
                entryHi,
                entryLo0,
                entryLo1,
                pageMask);

            ctx.cop0_entryhi = 0u;
            ctx.cop0_entrylo0 = 0u;
            ctx.cop0_entrylo1 = 0u;
            ctx.cop0_pagemask = 0u;
            runtime.handleTLBR(
                runtime.memory().getRDRAM(), &ctx);

            t.Equals(
                ctx.cop0_entryhi,
                entryHi & ~pageMask,
                "TLB writes should clear VPN2 bits selected by PageMask");
            t.Equals(
                ctx.cop0_pagemask,
                pageMask,
                "TLBR should return the stored page-size mask");
            t.Equals(
                ctx.cop0_entrylo0,
                entryLo0 & ~1u,
                "one-sided EntryLo0.G should read back as the combined Global bit");
            t.Equals(
                ctx.cop0_entrylo1,
                entryLo1 & ~1u,
                "one-sided Global state should clear both readback G bits");

            ctx.cop0_index = 0x80000000u;
            ctx.cop0_entryhi = entryHi;
            runtime.handleTLBP(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_index & 0x8000003fu,
                index,
                "TLBP should mask the probe VPN2 with the entry PageMask");
        });

        tc.Run("probe matches VPN2 ASID Global and invalid entries", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};

            constexpr uint32_t localIndex = 22u;
            constexpr uint32_t localEntryHi = 0xC5000042u;
            writeIndexedEntry(
                runtime,
                ctx,
                localIndex,
                localEntryHi,
                makeEntryLo(0x00500u, 2u, true, true, false),
                makeEntryLo(0x00501u, 2u, true, true, false));

            ctx.cop0_index = 0u;
            ctx.cop0_entryhi = localEntryHi ^ 0x1u;
            runtime.handleTLBP(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_index & 0x80000000u,
                0x80000000u,
                "a non-global entry should reject a different ASID");

            ctx.cop0_index = 0x80000000u;
            ctx.cop0_entryhi = localEntryHi;
            runtime.handleTLBP(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_index & 0x8000003fu,
                localIndex,
                "a non-global entry should match its own ASID");

            constexpr uint32_t invalidIndex = 23u;
            constexpr uint32_t invalidEntryHi = 0xC5200054u;
            writeIndexedEntry(
                runtime,
                ctx,
                invalidIndex,
                invalidEntryHi,
                makeEntryLo(0x00520u, 2u, true, false, false),
                makeEntryLo(0x00521u, 2u, true, false, false));
            ctx.cop0_index = 0x80000000u;
            ctx.cop0_entryhi = invalidEntryHi;
            runtime.handleTLBP(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_index & 0x8000003fu,
                invalidIndex,
                "TLBP should match VPN2 and ASID without consulting V");

            constexpr uint32_t globalIndex = 24u;
            constexpr uint32_t globalEntryHi = 0xC5400066u;
            writeIndexedEntry(
                runtime,
                ctx,
                globalIndex,
                globalEntryHi,
                makeEntryLo(0x00540u, 2u, true, true, true),
                makeEntryLo(0x00541u, 2u, true, true, true));
            ctx.cop0_index = 0x80000000u;
            ctx.cop0_entryhi = globalEntryHi ^ 0xffu;
            runtime.handleTLBP(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_index & 0x8000003fu,
                globalIndex,
                "two set G bits should make the entry ignore ASID");
        });

        tc.Run("loads select non-global entries by the live ASID", [](TestCase &t)
        {
            constexpr uint32_t virtualBase = 0xC5800000u;
            constexpr uint32_t firstAsid = 0x11u;
            constexpr uint32_t secondAsid = 0x22u;
            constexpr uint32_t firstPfn = 0x00800u;
            constexpr uint32_t secondPfn = 0x00900u;
            constexpr uint32_t firstValue = 0x11112222u;
            constexpr uint32_t secondValue = 0x33334444u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            writeIndexedEntry(
                runtime,
                ctx,
                28u,
                virtualBase | firstAsid,
                makeEntryLo(firstPfn, 2u, true, true, false),
                makeEntryLo(firstPfn + 1u, 2u, true, true, false));
            writeIndexedEntry(
                runtime,
                ctx,
                29u,
                virtualBase | secondAsid,
                makeEntryLo(secondPfn, 2u, true, true, false),
                makeEntryLo(secondPfn + 1u, 2u, true, true, false));
            std::memcpy(
                runtime.memory().getRDRAM() +
                    (firstPfn << 12u),
                &firstValue,
                sizeof(firstValue));
            std::memcpy(
                runtime.memory().getRDRAM() +
                    (secondPfn << 12u),
                &secondValue,
                sizeof(secondValue));

            ctx.cop0_entryhi = secondAsid;
            const uint32_t selected = runtime.Load32(
                runtime.memory().getRDRAM(),
                &ctx,
                virtualBase);
            t.Equals(
                selected,
                secondValue,
                "translation should use EntryHi.ASID to choose between equal VPN2 values");

            ctx.pc = 0x00140080u;
            ctx.cop0_entryhi = 0x33u;
            const bool missed = raisesGuestException([&]()
            {
                (void)runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    virtualBase);
            });
            t.IsTrue(
                missed,
                "a VPN2 match with no matching ASID should raise a refill");
            t.Equals(
                ctx.pc,
                0x80000000u,
                "an ASID miss should use the refill vector");
        });

        tc.Run("random writes preserve the complete selected entry", [](TestCase &t)
        {
            constexpr uint32_t random = 31u;
            constexpr uint32_t pageMask = 0x0001e000u;
            constexpr uint32_t entryHi = 0xCC01E04Bu;
            constexpr uint32_t entryLo0 =
                makeEntryLo(0x00a55u, 3u, true, true, true);
            constexpr uint32_t entryLo1 =
                makeEntryLo(0x00b66u, 7u, true, true, true);

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            ctx.cop0_random = random;
            ctx.cop0_entryhi = entryHi;
            ctx.cop0_entrylo0 = entryLo0;
            ctx.cop0_entrylo1 = entryLo1;
            ctx.cop0_pagemask = pageMask;
            runtime.handleTLBWR(
                runtime.memory().getRDRAM(), &ctx);

            ctx.cop0_index = random;
            ctx.cop0_entryhi = 0u;
            ctx.cop0_entrylo0 = 0u;
            ctx.cop0_entrylo1 = 0u;
            ctx.cop0_pagemask = 0u;
            runtime.handleTLBR(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_pagemask,
                pageMask,
                "TLBWR should retain PageMask");
            t.Equals(
                ctx.cop0_entryhi,
                entryHi & ~pageMask,
                "TLBWR should retain masked VPN2 and ASID");
            t.Equals(
                ctx.cop0_entrylo0,
                entryLo0,
                "TLBWR should retain the complete even-page half");
            t.Equals(
                ctx.cop0_entrylo1,
                entryLo1,
                "TLBWR should retain the complete odd-page half");
        });

        tc.Run("invalid and clean matches raise their distinct exceptions", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");

            R5900Context invalidCtx{};
            invalidCtx.pc = 0x00140000u;
            constexpr uint32_t invalidBase = 0xC6000000u;
            constexpr uint32_t invalidAddress =
                invalidBase + 0x1120u;
            writeIndexedEntry(
                runtime,
                invalidCtx,
                25u,
                invalidBase,
                makeEntryLo(0x00600u, 2u, true, true, true),
                makeEntryLo(0x00601u, 2u, true, false, true));
            const bool invalidRaised = raisesGuestException([&]()
            {
                (void)runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &invalidCtx,
                    invalidAddress);
            });
            t.IsTrue(
                invalidRaised,
                "a matching entry with V clear should raise TLB Invalid");
            t.Equals(
                invalidCtx.pc,
                0x80000180u,
                "TLB Invalid should use the common exception vector");
            t.Equals(
                invalidCtx.cop0_cause &
                    kCop0CauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD)
                 << 2u) &
                    kCop0CauseExcCodeMask,
                "TLB Invalid load should publish the TLBL cause code");
            t.Equals(
                invalidCtx.cop0_badvaddr,
                invalidAddress,
                "TLB Invalid should publish BadVAddr");

            R5900Context cleanCtx{};
            cleanCtx.pc = 0x00140040u;
            constexpr uint32_t cleanBase = 0xC6200000u;
            constexpr uint32_t cleanAddress =
                cleanBase + 0x1080u;
            writeIndexedEntry(
                runtime,
                cleanCtx,
                26u,
                cleanBase,
                makeEntryLo(0x00620u, 2u, true, true, true),
                makeEntryLo(0x00621u, 2u, false, true, true));
            const bool evenStored = !raisesGuestException([&]()
            {
                runtime.Store32(
                    runtime.memory().getRDRAM(),
                    &cleanCtx,
                    cleanBase + 0x80u,
                    0x87654321u);
            });
            t.IsTrue(
                evenStored,
                "EntryLo0.D should permit a store to the even page");
            const bool modifiedRaised = raisesGuestException([&]()
            {
                runtime.Store32(
                    runtime.memory().getRDRAM(),
                    &cleanCtx,
                    cleanAddress,
                    0x12345678u);
            });
            t.IsTrue(
                modifiedRaised,
                "a store through a valid clean page should raise TLB Modified");
            t.Equals(
                cleanCtx.pc,
                0x80000180u,
                "TLB Modified should use the common exception vector");
            t.Equals(
                cleanCtx.cop0_cause &
                    kCop0CauseExcCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_MODIFIED)
                 << 2u) &
                    kCop0CauseExcCodeMask,
                "a clean-page store should publish the Mod cause code");
            t.Equals(
                cleanCtx.cop0_badvaddr,
                cleanAddress,
                "TLB Modified should publish BadVAddr");
        });

        tc.Run("the S bit maps one aligned 16 KiB scratchpad window", [](TestCase &t)
        {
            constexpr uint32_t index = 27u;
            constexpr uint32_t virtualBase = 0xC7000000u;
            constexpr uint32_t offset = 0x3120u;
            constexpr uint32_t value = 0x89abcdefu;
            constexpr uint32_t entryLo0 = makeEntryLo(
                0x00700u, 3u, true, true, true, true);
            constexpr uint32_t entryLo1 = makeEntryLo(
                0x00701u, 7u, true, true, true);

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            writeIndexedEntry(
                runtime,
                ctx,
                index,
                virtualBase,
                entryLo0,
                entryLo1,
                0x00006000u);

            ctx.cop0_entrylo0 = 0u;
            ctx.cop0_entrylo1 = 0u;
            ctx.cop0_pagemask = 0xffffffffu;
            runtime.handleTLBR(
                runtime.memory().getRDRAM(), &ctx);
            t.Equals(
                ctx.cop0_pagemask,
                0u,
                "an S entry should force PageMask to zero on write");
            t.Equals(
                (ctx.cop0_entrylo0 >> 3u) & 0x7u,
                2u,
                "SPRAM EntryLo0 should read back as uncached");
            t.Equals(
                (ctx.cop0_entrylo1 >> 3u) & 0x7u,
                2u,
                "SPRAM EntryLo1 should read back as uncached");
            t.Equals(
                ctx.cop0_entrylo0 & 0x6u,
                entryLo0 & 0x6u,
                "SPRAM should retain its shared dirty and valid state");
            t.Equals(
                ctx.cop0_entrylo1 & 0x6u,
                entryLo0 & 0x6u,
                "both SPRAM halves should expose the same dirty and valid state");

            const bool stored = !raisesGuestException([&]()
            {
                runtime.Store32(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    virtualBase + offset,
                    value);
            });
            uint32_t scratchValue = 0u;
            std::memcpy(
                &scratchValue,
                runtime.memory().getScratchpad() + offset,
                sizeof(scratchValue));
            t.IsTrue(
                stored,
                "all four KiB quarters of an S mapping should be accessible");
            t.Equals(
                scratchValue,
                value,
                "an S mapping should target SPRAM rather than EntryLo0.PFN");
        });
    });
}
