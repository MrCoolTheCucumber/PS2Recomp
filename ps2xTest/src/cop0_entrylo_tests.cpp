#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <cstddef>
#include <cstdint>
#include <string>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kEntryLo0WritableMask = 0x83FFFFFFu;
    constexpr uint32_t kEntryLo1WritableMask = 0x03FFFFFFu;

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

    uint32_t emittedAndMask(const std::string &generated)
    {
        const std::string marker = " & 0x";
        const std::size_t position = generated.find(marker);
        if (position == std::string::npos)
        {
            return 0u;
        }
        return static_cast<uint32_t>(
            std::stoul(generated.substr(position + marker.size()), nullptr, 16));
    }

    bool everySingleBitMatches(uint32_t actualMask, uint32_t expectedMask)
    {
        for (uint32_t bit = 0u; bit < 32u; ++bit)
        {
            const uint32_t value = uint32_t{1} << bit;
            if ((value & actualMask) != (value & expectedMask))
            {
                return false;
            }
        }
        return true;
    }
}

void register_cop0_entrylo_tests()
{
    MiniTest::Case("COP0 EntryLo fields", [](TestCase &tc)
    {
        tc.Run("EntryLo0 retains S and clears only reserved bits", [](TestCase &t)
        {
            const std::string generatedMtc0 = translateCop0Move(
                COP0_MT,
                7u,
                COP0_REG_ENTRYLO0);
            const uint32_t mask = emittedAndMask(generatedMtc0);

            t.Equals(
                mask,
                kEntryLo0WritableMask,
                "MTC0 EntryLo0 should retain S, PFN, C, D, V, and G");
            t.IsTrue(
                everySingleBitMatches(mask, kEntryLo0WritableMask),
                "MTC0 EntryLo0 should classify every writable and reserved bit correctly");
            t.Equals(
                0xFFFFFFFFu & mask,
                kEntryLo0WritableMask,
                "an all-ones EntryLo0 write should clear bits 30:26 only");
            t.Equals(
                0x8000001Eu & mask,
                0x8000001Eu,
                "the manual's valid dirty SPRAM descriptor should retain S");

            const std::string generatedMfc0 = translateCop0Move(
                COP0_MF,
                8u,
                COP0_REG_ENTRYLO0);
            t.IsTrue(
                generatedMfc0.find(
                    "SET_GPR_S32(ctx, 8, (int32_t)ctx->cop0_entrylo0)") !=
                    std::string::npos,
                "MFC0 EntryLo0 should return the normalized register value");
        });

        tc.Run("EntryLo1 clears its complete reserved upper field", [](TestCase &t)
        {
            const std::string generatedMtc0 = translateCop0Move(
                COP0_MT,
                9u,
                COP0_REG_ENTRYLO1);
            const uint32_t mask = emittedAndMask(generatedMtc0);

            t.Equals(
                mask,
                kEntryLo1WritableMask,
                "MTC0 EntryLo1 should retain PFN, C, D, V, and G only");
            t.IsTrue(
                everySingleBitMatches(mask, kEntryLo1WritableMask),
                "MTC0 EntryLo1 should classify every writable and reserved bit correctly");
            t.Equals(
                0xFFFFFFFFu & mask,
                kEntryLo1WritableMask,
                "an all-ones EntryLo1 write should clear bits 31:26");

            const std::string generatedMfc0 = translateCop0Move(
                COP0_MF,
                10u,
                COP0_REG_ENTRYLO1);
            t.IsTrue(
                generatedMfc0.find(
                    "SET_GPR_S32(ctx, 10, (int32_t)ctx->cop0_entrylo1)") !=
                    std::string::npos,
                "MFC0 EntryLo1 should return the normalized register value");
        });
    });
}
