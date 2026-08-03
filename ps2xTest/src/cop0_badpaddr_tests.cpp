#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kBadPAddrWritableMask = 0xFFFFFFF0u;
    constexpr uint32_t kStatusBem = 0x00001000u;

    std::string translateBadPAddrMove(uint8_t operation, uint8_t gpr)
    {
        Instruction instruction{};
        instruction.opcode = OPCODE_COP0;
        instruction.rs = operation;
        instruction.rt = gpr;
        instruction.rd = COP0_REG_BADPADDR;
        return CodeGenerator({}, {}).translateInstruction(instruction);
    }

    std::optional<uint32_t> emittedHexAfter(
        const std::string &generated,
        const std::string &marker)
    {
        const std::size_t position = generated.find(marker);
        if (position == std::string::npos)
        {
            return std::nullopt;
        }

        std::size_t parsedCharacters = 0u;
        const unsigned long parsed = std::stoul(
            generated.substr(position + marker.size()),
            &parsedCharacters,
            16);
        if (parsedCharacters == 0u)
        {
            return std::nullopt;
        }
        return static_cast<uint32_t>(parsed);
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

void register_cop0_badpaddr_tests()
{
    MiniTest::Case("COP0 BadPAddr fields", [](TestCase &tc)
    {
        tc.Run("MTC0 writes BdPAddr and clears the zero field", [](TestCase &t)
        {
            const std::string generated = translateBadPAddrMove(COP0_MT, 7u);
            const uint32_t emittedWritableMask = emittedHexAfter(
                generated,
                "GPR_U32(ctx, 7) & 0x").value_or(0u);

            t.Equals(
                emittedWritableMask,
                kBadPAddrWritableMask,
                "MTC0 BadPAddr should accept bits 31:4 only");
            t.IsTrue(
                generated.find("ctx->cop0_badpaddr =") != std::string::npos,
                "MTC0 BadPAddr should update architectural state");
            t.Equals(
                0x89ABCDEFu & emittedWritableMask,
                0x89ABCDE0u,
                "a software write should normalize the fixed low field");

            const std::string readback = translateBadPAddrMove(COP0_MF, 8u);
            t.IsTrue(
                readback.find(
                    "SET_GPR_S32(ctx, 8, (int32_t)ctx->cop0_badpaddr)") !=
                    std::string::npos,
                "MFC0 BadPAddr should return the normalized software value");
        });

        tc.Run("an unmasked bus error replaces software BadPAddr state", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00124000u;
            ctx.cop0_badpaddr = 0x89ABCDE0u;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalBusError(
                    &ctx,
                    PS2BusErrorAccess::DataLoad,
                    0x2200123Fu);
            });

            t.IsTrue(
                raised,
                "an unmasked bus error should enter the guest exception path");
            t.Equals(
                ctx.cop0_badpaddr,
                0x22001230u,
                "processor capture should replace the software-written field");
            t.IsTrue(
                (ctx.cop0_status & kStatusBem) != 0u,
                "processor capture should set Status.BEM");
        });
    });
}
