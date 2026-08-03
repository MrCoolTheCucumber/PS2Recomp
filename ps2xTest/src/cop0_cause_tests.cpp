#include "MiniTest.h"
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
    constexpr uint32_t kCauseHardwareState =
        (uint32_t{1} << 31u) |
        (uint32_t{1} << 30u) |
        (uint32_t{2} << 28u) |
        (uint32_t{3} << 16u) |
        (uint32_t{1} << 15u) |
        (uint32_t{1} << 11u) |
        (uint32_t{1} << 10u) |
        (uint32_t{11} << 2u);

    static_assert(kCauseHardwareState == 0xE0038C2Cu);

    std::string translateCop0Move(
        uint8_t operation,
        uint8_t gpr,
        uint8_t cop0Register)
    {
        Instruction instruction{};
        instruction.opcode = OPCODE_COP0;
        instruction.rs = operation;
        instruction.rt = gpr;
        instruction.rd = cop0Register;
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

    uint32_t applyEmittedWrite(
        uint32_t oldCause,
        uint32_t value,
        uint32_t writableMask)
    {
        return
            (oldCause & ~writableMask) |
            (value & writableMask);
    }
}

void register_cop0_cause_tests()
{
    MiniTest::Case("COP0 Cause fields", [](TestCase &tc)
    {
        tc.Run("MTC0 cannot modify processor-owned Cause state", [](TestCase &t)
        {
            const std::string generated = translateCop0Move(
                COP0_MT,
                7u,
                COP0_REG_CAUSE);
            const uint32_t emittedWritableMask = emittedHexAfter(
                generated,
                "GPR_U32(ctx, 7) & 0x").value_or(0u);

            t.Equals(
                emittedWritableMask,
                0u,
                "EE Cause should have no software-writable bits");
            t.IsTrue(
                generated.find("ctx->cop0_cause =") == std::string::npos,
                "MTC0 Cause should not assign processor-owned state");
            t.Equals(
                applyEmittedWrite(
                    kCauseHardwareState,
                    0xFFFFFFFFu,
                    emittedWritableMask),
                kCauseHardwareState,
                "an all-ones write should preserve pending interrupts and exception fields");
            t.Equals(
                applyEmittedWrite(
                    kCauseHardwareState,
                    0u,
                    emittedWritableMask),
                kCauseHardwareState,
                "a zero write should preserve pending interrupts and exception fields");

            bool everySingleBitIsIgnored = true;
            for (uint32_t bit = 0u; bit < 32u; ++bit)
            {
                const uint32_t value = uint32_t{1} << bit;
                if (applyEmittedWrite(
                        kCauseHardwareState,
                        value,
                        emittedWritableMask) !=
                    kCauseHardwareState)
                {
                    everySingleBitIsIgnored = false;
                    break;
                }
            }
            t.IsTrue(
                everySingleBitIsIgnored,
                "MTC0 Cause should ignore every GPR input bit");
        });

        tc.Run("MFC0 returns pending Cause state", [](TestCase &t)
        {
            const std::string generated = translateCop0Move(
                COP0_MF,
                8u,
                COP0_REG_CAUSE);
            t.IsTrue(
                generated.find(
                    "SET_GPR_S32(ctx, 8, (int32_t)ctx->cop0_cause)") !=
                    std::string::npos,
                "MFC0 Cause should return processor-maintained state");
        });
    });
}
