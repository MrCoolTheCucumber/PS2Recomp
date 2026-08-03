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
    constexpr uint32_t kConfigWritableMask =
        (uint32_t{1} << 18u) |
        (uint32_t{1} << 17u) |
        (uint32_t{1} << 16u) |
        (uint32_t{1} << 13u) |
        (uint32_t{1} << 12u) |
        0x7u;
    constexpr uint32_t kConfigHardwareMask =
        (0x7u << 28u) |
        (0x7u << 9u) |
        (0x7u << 6u);
    constexpr uint32_t kConfigReservedMask =
        ~(kConfigWritableMask | kConfigHardwareMask);
    constexpr uint32_t kSyntheticHardwareState =
        (0x5u << 28u) |
        (0x2u << 9u) |
        (0x1u << 6u);

    static_assert(kConfigWritableMask == 0x00073007u);
    static_assert(kConfigHardwareMask == 0x70000FC0u);

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

    uint32_t applyEmittedMasks(
        uint32_t oldConfig,
        uint32_t value,
        uint32_t hardwareMask,
        uint32_t writableMask)
    {
        return
            (oldConfig & hardwareMask) |
            (value & writableMask);
    }
}

void register_cop0_config_tests()
{
    MiniTest::Case("COP0 Config fields", [](TestCase &tc)
    {
        tc.Run("MTC0 changes only software-owned Config fields", [](TestCase &t)
        {
            const std::string generated = translateCop0Move(
                COP0_MT,
                7u,
                COP0_REG_CONFIG);
            const uint32_t emittedHardwareMask = emittedHexAfter(
                generated,
                "ctx->cop0_config & 0x").value_or(0u);
            const uint32_t emittedWritableMask = emittedHexAfter(
                generated,
                "GPR_U32(ctx, 7) & 0x").value_or(0u);

            t.Equals(
                emittedHardwareMask,
                kConfigHardwareMask,
                "MTC0 Config should preserve EC, IC, and DC only");
            t.Equals(
                emittedWritableMask,
                kConfigWritableMask,
                "MTC0 Config should accept DIE, ICE, DCE, NBE, BPE, and K0 only");

            const uint32_t pollutedOldState =
                kSyntheticHardwareState |
                kConfigWritableMask |
                kConfigReservedMask;
            t.Equals(
                applyEmittedMasks(
                    pollutedOldState,
                    0u,
                    emittedHardwareMask,
                    emittedWritableMask),
                kSyntheticHardwareState,
                "a zero write should retain hardware fields and clear software and reserved fields");
            t.Equals(
                applyEmittedMasks(
                    pollutedOldState,
                    0xFFFFFFFFu,
                    emittedHardwareMask,
                    emittedWritableMask),
                kSyntheticHardwareState | kConfigWritableMask,
                "an all-ones write should retain hardware fields and set only writable fields");

            bool everySingleBitMatches = true;
            for (uint32_t bit = 0u; bit < 32u; ++bit)
            {
                const uint32_t value = uint32_t{1} << bit;
                const uint32_t actual = applyEmittedMasks(
                    pollutedOldState,
                    value,
                    emittedHardwareMask,
                    emittedWritableMask);
                const uint32_t expected =
                    kSyntheticHardwareState |
                    (value & kConfigWritableMask);
                if (actual != expected)
                {
                    everySingleBitMatches = false;
                    break;
                }
            }
            t.IsTrue(
                everySingleBitMatches,
                "MTC0 Config should classify every writable, hardware, and reserved bit correctly");
        });

        tc.Run("MFC0 returns normalized Config state", [](TestCase &t)
        {
            const std::string generated = translateCop0Move(
                COP0_MF,
                8u,
                COP0_REG_CONFIG);
            t.IsTrue(
                generated.find(
                    "SET_GPR_S32(ctx, 8, (int32_t)ctx->cop0_config)") !=
                    std::string::npos,
                "MFC0 Config should return the normalized register value");
        });
    });
}
