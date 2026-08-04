#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

using namespace ps2recomp;

namespace
{
    struct BreakpointRegisterCase
    {
        uint16_t selector;
        const char *contextMember;
        uint32_t value;
    };

    constexpr std::array<BreakpointRegisterCase, 7> kBreakpointRegisters{{
        {0u, "cop0_bpc", 0x80010001u},
        {2u, "cop0_iab", 0x00123450u},
        {3u, "cop0_iabm", 0xFFFFFFFCu},
        {4u, "cop0_dab", 0x89ABCDE0u},
        {5u, "cop0_dabm", 0xFFFFFFF0u},
        {6u, "cop0_dvb", 0x13579BDFu},
        {7u, "cop0_dvbm", 0xFFFF00FFu},
    }};

    uint32_t encodeBreakpointMove(
        uint8_t operation,
        uint8_t gpr,
        uint16_t selector)
    {
        return
            (OPCODE_COP0 << 26u) |
            (static_cast<uint32_t>(operation) << 21u) |
            (static_cast<uint32_t>(gpr) << 16u) |
            (COP0_REG_DEBUG << 11u) |
            selector;
    }

    Instruction decodeBreakpointMove(
        uint8_t operation,
        uint8_t gpr,
        uint16_t selector)
    {
        return R5900Decoder{}.decodeInstruction(
            0x00100000u,
            encodeBreakpointMove(operation, gpr, selector),
            false);
    }

    std::optional<std::string> emittedContextMember(
        const std::string &generated)
    {
        constexpr const char *marker = "ctx->";
        const std::size_t markerPosition = generated.find(marker);
        if (markerPosition == std::string::npos)
        {
            return std::nullopt;
        }

        const std::size_t start = markerPosition + 5u;
        std::size_t end = start;
        while (end < generated.size())
        {
            const unsigned char character =
                static_cast<unsigned char>(generated[end]);
            if (std::isalnum(character) == 0 && character != '_')
            {
                break;
            }
            ++end;
        }
        if (end == start)
        {
            return std::nullopt;
        }
        return generated.substr(start, end - start);
    }

    uint32_t breakpointRegisterValue(
        const R5900Context &context,
        uint16_t selector)
    {
        switch (selector)
        {
        case 0u:
            return context.cop0_bpc;
        case 2u:
            return context.cop0_iab;
        case 3u:
            return context.cop0_iabm;
        case 4u:
            return context.cop0_dab;
        case 5u:
            return context.cop0_dabm;
        case 6u:
            return context.cop0_dvb;
        case 7u:
            return context.cop0_dvbm;
        default:
            return 0u;
        }
    }
}

void register_cop0_breakpoint_register_tests()
{
    MiniTest::Case("COP0 breakpoint register selectors", [](TestCase &tc)
    {
        tc.Run("raw move selectors round-trip seven independent registers", [](TestCase &t)
        {
            CodeGenerator generator({}, {});
            PS2Runtime runtime;
            R5900Context context{};

            for (std::size_t index = 0u;
                 index < kBreakpointRegisters.size();
                 ++index)
            {
                const BreakpointRegisterCase &registerCase =
                    kBreakpointRegisters[index];
                const uint8_t gpr = static_cast<uint8_t>(8u + index);
                const Instruction moveTo = decodeBreakpointMove(
                    COP0_MT,
                    gpr,
                    registerCase.selector);

                t.Equals(
                    moveTo.rd,
                    static_cast<uint32_t>(COP0_REG_DEBUG),
                    "every breakpoint move should decode COP0 register 24");
                t.Equals(
                    moveTo.raw & 0x7FFu,
                    static_cast<uint32_t>(registerCase.selector),
                    "the decoder should retain the complete breakpoint selector");

                const std::string generated =
                    generator.translateInstruction(moveTo);
                t.IsTrue(
                    generated.find(
                        "runtime->writeCop0Breakpoint(ctx, " +
                        std::to_string(registerCase.selector) +
                        "u, GPR_U32(ctx, " +
                        std::to_string(gpr) + "));") !=
                        std::string::npos,
                    "each move-to selector should pass through the runtime breakpoint-write seam");
                runtime.writeCop0Breakpoint(
                    &context,
                    registerCase.selector,
                    registerCase.value);
                t.Equals(
                    breakpointRegisterValue(
                        context, registerCase.selector),
                    registerCase.value,
                    "the runtime helper should update the selected architectural register");
            }

            for (std::size_t index = 0u;
                 index < kBreakpointRegisters.size();
                 ++index)
            {
                const BreakpointRegisterCase &registerCase =
                    kBreakpointRegisters[index];
                const uint8_t gpr = static_cast<uint8_t>(16u + index);
                const Instruction moveFrom = decodeBreakpointMove(
                    COP0_MF,
                    gpr,
                    registerCase.selector);
                const std::string generated =
                    generator.translateInstruction(moveFrom);
                const std::optional<std::string> member =
                    emittedContextMember(generated);
                t.IsTrue(
                    member.has_value(),
                    "each move-from variant should read context state");
                if (!member.has_value())
                {
                    continue;
                }

                t.Equals(
                    *member,
                    std::string(registerCase.contextMember),
                    "each move-from selector should target its named breakpoint register");
                t.Equals(
                    breakpointRegisterValue(
                        context, registerCase.selector),
                    registerCase.value,
                    "seven distinct writes should round-trip through their matching selectors");
            }

            const uint32_t originalBpc = context.cop0_bpc;
            runtime.writeCop0Breakpoint(
                &context, 0x7ffu, 0xffffffffu);
            t.Equals(
                context.cop0_bpc,
                originalBpc,
                "unknown register-24 selectors should retain their previous no-op behavior");
        });
    });
}
