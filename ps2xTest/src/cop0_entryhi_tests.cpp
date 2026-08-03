#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <cstdint>
#include <string>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kEntryHiVpn2Mask = 0xFFFFE000u;
    constexpr uint32_t kEntryHiAsidMask = 0x000000FFu;
    constexpr uint32_t kEntryHiWritableMask =
        kEntryHiVpn2Mask | kEntryHiAsidMask;

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

void register_cop0_entryhi_tests()
{
    MiniTest::Case("COP0 EntryHi fields", [](TestCase &tc)
    {
        tc.Run("MTC0 retains VPN2 and ASID only", [](TestCase &t)
        {
            CodeGenerator generator({}, {});
            Instruction mtc0{};
            mtc0.opcode = OPCODE_COP0;
            mtc0.rs = COP0_MT;
            mtc0.rt = 7u;
            mtc0.rd = COP0_REG_ENTRYHI;

            const std::string generatedMtc0 =
                generator.translateInstruction(mtc0);

            t.IsTrue(
                generatedMtc0.find(
                    "GPR_U32(ctx, 7) & 0xFFFFE0FF") !=
                    std::string::npos,
                "MTC0 EntryHi should retain all of VPN2 and ASID");
            t.IsFalse(
                generatedMtc0.find("0xC00000FF") !=
                    std::string::npos,
                "MTC0 EntryHi should not discard VPN2 bits 29:13");

            Instruction mfc0 = mtc0;
            mfc0.rs = COP0_MF;
            mfc0.rt = 8u;
            const std::string generatedMfc0 =
                generator.translateInstruction(mfc0);

            t.IsTrue(
                generatedMfc0.find(
                    "SET_GPR_S32(ctx, 8, (int32_t)ctx->cop0_entryhi)") !=
                    std::string::npos,
                "MFC0 EntryHi should return the normalized register value");
        });

        tc.Run("TLB faults replace VPN2 and retain ASID only", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            constexpr uint32_t original = 0x81235BA7u;
            constexpr uint32_t badVirtualAddress = 0xC7654321u;
            ctx.pc = 0x00124000u;
            ctx.cop0_entryhi = original;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalMemoryException(
                    &ctx,
                    EXCEPTION_TLB_REFILL_LOAD,
                    badVirtualAddress);
            });

            const uint32_t expected =
                (badVirtualAddress & kEntryHiVpn2Mask) |
                (original & kEntryHiAsidMask);
            t.IsTrue(
                raised,
                "a TLB refill should unwind guest execution");
            t.Equals(
                ctx.cop0_entryhi,
                expected,
                "a TLB refill should replace VPN2, preserve ASID, and clear bits 12:8");
            t.Equals(
                ctx.cop0_entryhi & ~kEntryHiWritableMask,
                0u,
                "EntryHi reserved bits 12:8 should remain zero");
        });

        tc.Run("indexed write and probe use complete VPN2", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            uint8_t *const rdram = runtime.memory().getRDRAM();

            constexpr uint32_t index = 19u;
            constexpr uint32_t entryHi = 0x5A2460A5u;
            constexpr uint32_t legacyProjection =
                entryHi & 0xC00000FFu;
            R5900Context ctx{};
            ctx.cop0_index = index;
            ctx.cop0_entryhi = entryHi;
            ctx.cop0_entrylo0 = (0x12345u << 6u) | 0x2u;
            runtime.handleTLBWI(rdram, &ctx);

            ctx.cop0_index = 0x80000000u;
            ctx.cop0_entryhi = entryHi;
            runtime.handleTLBP(rdram, &ctx);
            t.Equals(
                ctx.cop0_index & 0x8000003Fu,
                index,
                "TLBP should find a TLBWI entry using VPN2 bits 29:13");

            ctx.cop0_index = 0u;
            ctx.cop0_entryhi = legacyProjection;
            runtime.handleTLBP(rdram, &ctx);
            t.Equals(
                ctx.cop0_index & 0x80000000u,
                0x80000000u,
                "the legacy truncated VPN2 must not alias the complete VPN2");
        });
    });
}
