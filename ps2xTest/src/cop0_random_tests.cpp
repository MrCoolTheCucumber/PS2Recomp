#include "MiniTest.h"

#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace ps2recomp;

namespace
{
    size_t countOccurrences(
        const std::string &value,
        const std::string &needle)
    {
        size_t count = 0u;
        size_t offset = 0u;
        while ((offset = value.find(
                    needle, offset)) !=
               std::string::npos)
        {
            ++count;
            offset += needle.size();
        }
        return count;
    }

    Instruction makeNop(uint32_t address)
    {
        Instruction instruction{};
        instruction.address = address;
        instruction.opcode = OPCODE_ADDIU;
        return instruction;
    }
}

void register_cop0_random_tests()
{
    MiniTest::Case("COP0 Random", [](TestCase &tc)
    {
        tc.Run("generated instructions publish Random retirement boundaries", [](TestCase &t)
        {
            Function function{};
            function.name = "cop0_random_sequence";
            function.start = 0x00140000u;
            function.end = 0x0014000Cu;
            function.isRecompiled = true;

            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    function,
                    {
                        makeNop(0x00140000u),
                        makeNop(0x00140004u),
                        makeNop(0x00140008u),
                    },
                    false);

            t.Equals(
                countOccurrences(
                    generated,
                    "ctx->beginEeInstruction();"),
                static_cast<size_t>(3u),
                "every executed instruction should enter an architectural retirement boundary");
            t.Equals(
                countOccurrences(
                    generated,
                    "++ctx->insn_count;"),
                static_cast<size_t>(0u),
                "instruction accounting should not bypass the retirement boundary");
        });

        tc.Run("TLBWR consumes Random without advancing it itself", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            uint8_t *const rdram =
                runtime.memory().getRDRAM();

            R5900Context ctx{};
            ctx.cop0_random = 19u;
            ctx.cop0_wired = 5u;
            ctx.cop0_entryhi = 0x12345000u;
            ctx.cop0_entrylo0 =
                (0x12345u << 6u) | 0x2u;
            runtime.handleTLBWR(rdram, &ctx);

            ctx.cop0_entryhi = 0x23456000u;
            ctx.cop0_entrylo0 =
                (0x23456u << 6u) | 0x2u;
            runtime.handleTLBWR(rdram, &ctx);

            uint32_t vpn = 0u;
            uint32_t pfn = 0u;
            uint32_t mask = 0u;
            bool valid = false;
            t.IsTrue(
                runtime.memory().tlbRead(
                    19u, vpn, pfn, mask, valid),
                "the current Random slot should be readable");
            t.Equals(
                vpn,
                0x23456000u,
                "repeated TLBWR effects should consume the same current Random value");
            t.Equals(
                pfn,
                0x23456u,
                "TLBWR should write the payload to the current Random slot");
            t.IsTrue(
                valid,
                "TLBWR should preserve the entry validity payload");
            t.Equals(
                ctx.cop0_random,
                19u,
                "TLBWR should leave automatic Random retirement to the instruction boundary");
        });

        tc.Run("Random decrements after each instruction and wraps at Wired", [](TestCase &t)
        {
            R5900Context ctx{};
            ctx.cop0_wired = 5u;
            ctx.cop0_random = 47u;

            ctx.beginEeInstruction();
            t.Equals(
                ctx.cop0_random,
                47u,
                "an instruction should consume Random before its own retirement");
            ctx.beginEeInstruction();
            t.Equals(
                ctx.cop0_random,
                46u,
                "starting the next instruction should retire the previous one");
            ctx.finishEeInstruction();
            t.Equals(
                ctx.cop0_random,
                45u,
                "a block boundary should retire its final instruction");

            const auto retireInstruction = [&ctx]()
            {
                ctx.beginEeInstruction();
                ctx.finishEeInstruction();
            };
            for (uint32_t count = 0u; count < 40u; ++count)
            {
                retireInstruction();
            }
            t.Equals(
                ctx.cop0_random,
                5u,
                "Random should include the Wired lower bound");
            retireInstruction();
            t.Equals(
                ctx.cop0_random,
                47u,
                "retiring at Wired should wrap Random to entry 47");

            ctx.cop0_wired = 0u;
            ctx.cop0_random = 47u;
            for (uint32_t count = 0u; count < 47u; ++count)
            {
                retireInstruction();
            }
            t.Equals(
                ctx.cop0_random,
                0u,
                "Wired zero should permit Random entry zero");
            retireInstruction();
            t.Equals(
                ctx.cop0_random,
                47u,
                "entry zero should wrap to the upper bound");

            ctx.cop0_wired = 47u;
            ctx.cop0_random = 47u;
            retireInstruction();
            t.Equals(
                ctx.cop0_random,
                47u,
                "Wired 47 should leave a one-entry Random interval");

            PS2Runtime runtime;
            R5900Context boundaryCtx{};
            boundaryCtx.cop0_wired = 10u;
            boundaryCtx.cop0_random = 12u;
            boundaryCtx.beginEeInstruction();
            runtime.serviceEeEventsAtBlockBoundary(
                nullptr, &boundaryCtx);
            t.Equals(
                boundaryCtx.cop0_random,
                11u,
                "an EE block boundary should retire its pending instruction");

            R5900Context faultCtx{};
            faultCtx.pc = 0x00140000u;
            faultCtx.cop0_wired = 10u;
            faultCtx.cop0_random = 12u;
            faultCtx.beginEeInstruction();
            bool raised = false;
            try
            {
                runtime.SignalException(
                    &faultCtx,
                    EXCEPTION_RESERVED_INSTRUCTION);
            }
            catch (const PS2GuestException &)
            {
                raised = true;
            }
            t.IsTrue(
                raised,
                "the synthetic fault should unwind guest execution");
            t.Equals(
                faultCtx.cop0_random,
                11u,
                "a faulting instruction should retire Random before exception delivery");
        });
    });
}
