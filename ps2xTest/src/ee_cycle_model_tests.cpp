#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ee_cycle_model.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kAddress = 0x00140000u;
    constexpr uint32_t kJumpTicks = 9u;
    constexpr uint32_t kConditionalBranchTicks = 11u;

    struct ControlTransferCase
    {
        const char *name;
        uint32_t raw;
        uint32_t expectedTicks;
    };

    uint32_t encodeBranch(uint32_t opcode)
    {
        return (opcode << 26u) | (1u << 21u) | 1u;
    }

    uint32_t encodeRegimmBranch(uint32_t condition)
    {
        return (OPCODE_REGIMM << 26u) |
               (1u << 21u) |
               (condition << 16u) |
               1u;
    }

    uint32_t encodeCoprocessorBranch(
        uint32_t opcode,
        uint32_t format,
        uint32_t condition)
    {
        return (opcode << 26u) |
               (format << 21u) |
               (condition << 16u) |
               1u;
    }

    uint32_t sumGeneratedCycleTicks(const std::string &generated)
    {
        constexpr std::string_view marker =
            "ctx->advanceEeCycleTicks(";
        uint32_t total = 0u;
        size_t offset = 0u;
        while ((offset = generated.find(marker, offset)) !=
               std::string::npos)
        {
            offset += marker.size();
            const size_t end = generated.find('u', offset);
            if (end == std::string::npos)
            {
                break;
            }
            total += static_cast<uint32_t>(
                std::stoul(generated.substr(offset, end - offset)));
            offset = end + 1u;
        }
        return total;
    }

    uint32_t generatedControlBlockTicks(uint32_t raw)
    {
        R5900Decoder decoder;
        const Instruction control =
            decoder.decodeInstruction(kAddress, raw, false);
        const Instruction delay =
            decoder.decodeInstruction(kAddress + 4u, 0u, false);

        Function function{};
        function.name = "control_transfer_timing";
        function.start = kAddress;
        function.end = kAddress + 8u;
        function.isRecompiled = true;

        const std::string generated =
            CodeGenerator({}, {}).generateFunction(
                function,
                {control, delay},
                false);
        return sumGeneratedCycleTicks(generated);
    }
}

void register_ee_cycle_model_tests()
{
    MiniTest::Case("EE cycle model", [](TestCase &tc)
    {
        tc.Run("decoded control transfers use their complete timing classes", [](TestCase &t)
        {
            const std::vector<ControlTransferCase> cases = {
                {"J", (OPCODE_J << 26u) | 0x100u, kJumpTicks},
                {"JAL", (OPCODE_JAL << 26u) | 0x100u, kJumpTicks},
                {"JR", (1u << 21u) | SPECIAL_JR, kJumpTicks},
                {"JALR", (1u << 21u) | (31u << 11u) | SPECIAL_JALR, kJumpTicks},

                {"BEQ", encodeBranch(OPCODE_BEQ), kConditionalBranchTicks},
                {"BNE", encodeBranch(OPCODE_BNE), kConditionalBranchTicks},
                {"BLEZ", encodeBranch(OPCODE_BLEZ), kConditionalBranchTicks},
                {"BGTZ", encodeBranch(OPCODE_BGTZ), kConditionalBranchTicks},
                {"BEQL", encodeBranch(OPCODE_BEQL), kConditionalBranchTicks},
                {"BNEL", encodeBranch(OPCODE_BNEL), kConditionalBranchTicks},
                {"BLEZL", encodeBranch(OPCODE_BLEZL), kConditionalBranchTicks},
                {"BGTZL", encodeBranch(OPCODE_BGTZL), kConditionalBranchTicks},

                {"BLTZ", encodeRegimmBranch(REGIMM_BLTZ), kConditionalBranchTicks},
                {"BGEZ", encodeRegimmBranch(REGIMM_BGEZ), kConditionalBranchTicks},
                {"BLTZL", encodeRegimmBranch(REGIMM_BLTZL), kConditionalBranchTicks},
                {"BGEZL", encodeRegimmBranch(REGIMM_BGEZL), kConditionalBranchTicks},
                {"BLTZAL", encodeRegimmBranch(REGIMM_BLTZAL), kConditionalBranchTicks},
                {"BGEZAL", encodeRegimmBranch(REGIMM_BGEZAL), kConditionalBranchTicks},
                {"BLTZALL", encodeRegimmBranch(REGIMM_BLTZALL), kConditionalBranchTicks},
                {"BGEZALL", encodeRegimmBranch(REGIMM_BGEZALL), kConditionalBranchTicks},
            };

            std::vector<ControlTransferCase> allCases = cases;
            for (const auto &[name, opcode, format] :
                 std::array{
                     std::array<uint32_t, 3u>{0u, OPCODE_COP0, COP0_BC},
                     std::array<uint32_t, 3u>{1u, OPCODE_COP1, COP1_BC},
                     std::array<uint32_t, 3u>{2u, OPCODE_COP2, COP2_BC},
                 })
            {
                for (uint32_t condition = 0u; condition < 4u; ++condition)
                {
                    static constexpr std::array<const char *, 12u> names = {
                        "BC0F", "BC0T", "BC0FL", "BC0TL",
                        "BC1F", "BC1T", "BC1FL", "BC1TL",
                        "BC2F", "BC2T", "BC2FL", "BC2TL",
                    };
                    allCases.push_back({
                        names[name * 4u + condition],
                        encodeCoprocessorBranch(opcode, format, condition),
                        kConditionalBranchTicks,
                    });
                }
            }

            R5900Decoder decoder;
            for (const ControlTransferCase &test : allCases)
            {
                const Instruction instruction =
                    decoder.decodeInstruction(kAddress, test.raw, false);
                const std::string prefix = std::string(test.name) + ": ";
                t.IsTrue(
                    instruction.hasDelaySlot,
                    prefix + "decoded control transfer should have a delay slot");
                t.IsTrue(
                    instruction.isJump || instruction.isBranch,
                    prefix + "decoder should identify jump/branch control flow");
                t.Equals(
                    eeInstructionCycleTicks(instruction),
                    test.expectedTicks,
                    prefix + "cycle model should use the selected issue class");
            }
        });

        tc.Run("generated jump and BC2 blocks accumulate selected ticks", [](TestCase &t)
        {
            const uint32_t jumpRaw =
                (OPCODE_J << 26u) |
                (((kAddress + 0x100u) >> 2u) & 0x03FFFFFFu);
            const uint32_t bc2Raw = encodeCoprocessorBranch(
                OPCODE_COP2,
                COP2_BC,
                COP2_BC_BCF);

            t.Equals(
                generatedControlBlockTicks(jumpRaw),
                kJumpTicks + 9u,
                "J plus its NOP delay slot should accumulate 18 ticks");
            t.Equals(
                generatedControlBlockTicks(bc2Raw),
                kConditionalBranchTicks + 9u,
                "BC2 plus its NOP delay slot should accumulate 20 ticks");
        });
    });
}
