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
    constexpr uint32_t kContextPteBaseMask = 0xFF800000u;
    constexpr uint32_t kContextBadVpn2Mask = 0x007FFFF0u;

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

void register_cop0_context_tests()
{
    MiniTest::Case("COP0 Context fields", [](TestCase &tc)
    {
        tc.Run("MTC0 replaces only PTEBase", [](TestCase &t)
        {
            CodeGenerator generator({}, {});
            Instruction mtc0{};
            mtc0.opcode = OPCODE_COP0;
            mtc0.rs = COP0_MT;
            mtc0.rt = 7u;
            mtc0.rd = COP0_REG_CONTEXT;

            const std::string generated =
                generator.translateInstruction(mtc0);

            t.IsTrue(
                generated.find(
                    "ctx->cop0_context & 0x007FFFF0") !=
                    std::string::npos,
                "MTC0 Context should preserve hardware-maintained BadVPN2");
            t.IsTrue(
                generated.find(
                    "GPR_U32(ctx, 7) & 0xFF800000") !=
                    std::string::npos,
                "MTC0 Context should take only PTEBase from the GPR");
            t.IsFalse(
                generated.find("0x7FFFFF") != std::string::npos,
                "MTC0 Context should not write BadVPN2 or reserved low bits");
        });

        tc.Run("TLB faults replace only BadVPN2", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            constexpr uint32_t original = 0x5AB4567Fu;
            constexpr uint32_t badVirtualAddress = 0xC7654321u;
            ctx.pc = 0x00124000u;
            ctx.cop0_context = original;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalMemoryException(
                    &ctx,
                    EXCEPTION_TLB_REFILL_LOAD,
                    badVirtualAddress);
            });

            const uint32_t expected =
                (original & kContextPteBaseMask) |
                ((badVirtualAddress >> 9u) & kContextBadVpn2Mask);
            t.IsTrue(
                raised,
                "a TLB refill should unwind guest execution");
            t.Equals(
                ctx.cop0_context,
                expected,
                "a TLB refill should preserve PTEBase, replace BadVPN2, and clear bits 3:0");
            t.Equals(
                ctx.cop0_context & 0xFu,
                0u,
                "Context bits 3:0 should remain zero");
        });
    });
}
