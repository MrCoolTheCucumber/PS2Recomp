#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kCauseBd = 0x80000000u;
    constexpr uint32_t kCauseExcCodeMask = 0x0000007Cu;
    constexpr uint32_t kStatusExl = 0x00000002u;
    constexpr uint32_t kGeneralExceptionVector = 0x80000180u;
    constexpr uint32_t kEncodedSyscallId = 0x12345u;

    std::string contextAssignment(
        const std::string &field,
        uint32_t value)
    {
        std::ostringstream stream;
        stream << "ctx->" << field << " = 0x"
               << std::uppercase << std::hex << value << "u;";
        return stream.str();
    }

    Instruction decode(uint32_t address, uint32_t raw)
    {
        return R5900Decoder{}.decodeInstruction(address, raw, false);
    }

    Instruction nop(uint32_t address)
    {
        return decode(address, 0u);
    }

    std::string generateBranchWithSyscall(
        uint32_t branchAddress,
        uint32_t branchRaw,
        const std::string &name)
    {
        Function function{};
        function.name = name;
        function.start = branchAddress;
        function.end = branchAddress + 0x10u;
        function.isRecompiled = true;

        const std::vector<Instruction> instructions = {
            decode(branchAddress, branchRaw),
            decode(
                branchAddress + 4u,
                (kEncodedSyscallId << 6u) | SPECIAL_SYSCALL),
            nop(branchAddress + 8u),
            nop(branchAddress + 12u),
        };

        return CodeGenerator({}, {}).generateFunction(
            function, instructions, false);
    }

    void expectGeneratedDelayState(
        TestCase &t,
        const std::string &generated,
        uint32_t branchAddress,
        const std::string &prefix)
    {
        const size_t delayState = generated.find(
            "ctx->in_delay_slot = true;");
        const size_t branchPc = generated.find(
            contextAssignment("branch_pc", branchAddress),
            delayState);
        const size_t syscallPc = generated.rfind(
            contextAssignment("pc", branchAddress + 4u),
            delayState);
        const size_t handler = generated.find(
            "runtime->handleSyscall(rdram, ctx, 0x12345u);",
            delayState);

        t.IsTrue(
            handler != std::string::npos,
            prefix + "fixture should emit the SYSCALL runtime boundary");
        t.IsTrue(
            delayState != std::string::npos &&
                handler != std::string::npos &&
                delayState < handler,
            prefix + "delay state must precede SYSCALL");
        t.IsTrue(
            branchPc != std::string::npos &&
                handler != std::string::npos &&
                branchPc < handler,
            prefix + "owning branch PC must precede SYSCALL");
        t.IsTrue(
            syscallPc != std::string::npos &&
                handler != std::string::npos &&
                syscallPc < handler,
            prefix + "SYSCALL PC must precede its runtime boundary");
    }

    void runDelaySyscallCase(
        TestCase &t,
        uint32_t branchAddress,
        const std::string &prefix)
    {
        PS2Runtime runtime;
        std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
        R5900Context ctx{};
        ctx.pc = branchAddress + 4u;
        ctx.branch_pc = branchAddress;
        ctx.in_delay_slot = true;
        ctx.cop0_cause = 0x00000100u;

        bool guestException = false;
        bool hostException = false;
        try
        {
            runtime.handleSyscall(
                rdram.data(), &ctx, kEncodedSyscallId);
        }
        catch (const PS2GuestException &)
        {
            guestException = true;
        }
        catch (...)
        {
            hostException = true;
        }

        t.IsTrue(
            guestException,
            prefix + "SYSCALL should unwind as a guest exception");
        t.IsFalse(
            hostException,
            prefix + "SYSCALL must not escape as a host exception");
        t.Equals(
            ctx.cop0_epc,
            branchAddress,
            prefix + "EPC should identify the owning branch");
        t.Equals(
            ctx.cop0_cause & kCauseExcCodeMask,
            (static_cast<uint32_t>(EXCEPTION_SYSCALL) << 2u) &
                kCauseExcCodeMask,
            prefix + "Cause.ExcCode should identify System Call");
        t.IsTrue(
            (ctx.cop0_cause & kCauseBd) != 0u,
            prefix + "Cause.BD should identify delay-slot execution");
        t.IsTrue(
            (ctx.cop0_status & kStatusExl) != 0u,
            prefix + "exception delivery should set Status.EXL");
        t.Equals(
            ctx.pc,
            kGeneralExceptionVector,
            prefix + "SYSCALL should enter V_COMMON");
        t.IsFalse(
            ctx.in_delay_slot,
            prefix + "delivered exception should clear runtime delay state");
    }
}

void register_ee_syscall_delay_tests()
{
    MiniTest::Case("EE SYSCALL delay slot", [](TestCase &tc)
    {
        tc.Run("generated branch owners publish precise delay state", [](TestCase &t)
        {
            constexpr uint32_t takenAddress = 0x00150000u;
            constexpr uint32_t notTakenAddress = 0x00150100u;
            constexpr uint32_t indirectAddress = 0x00150200u;

            expectGeneratedDelayState(
                t,
                generateBranchWithSyscall(
                    takenAddress,
                    (OPCODE_BEQ << 26u) | 2u,
                    "syscall_taken"),
                takenAddress,
                "taken: ");
            expectGeneratedDelayState(
                t,
                generateBranchWithSyscall(
                    notTakenAddress,
                    (OPCODE_BNE << 26u) | 2u,
                    "syscall_not_taken"),
                notTakenAddress,
                "not taken: ");
            expectGeneratedDelayState(
                t,
                generateBranchWithSyscall(
                    indirectAddress,
                    (8u << 21u) | SPECIAL_JR,
                    "syscall_indirect"),
                indirectAddress,
                "indirect: ");
        });

        tc.Run("taken branch delivers a guest System Call exception", [](TestCase &t)
        {
            runDelaySyscallCase(t, 0x00150000u, "taken: ");
        });

        tc.Run("not-taken branch still owns its executed delay slot", [](TestCase &t)
        {
            runDelaySyscallCase(t, 0x00150100u, "not taken: ");
        });

        tc.Run("indirect jump delivers a guest System Call exception", [](TestCase &t)
        {
            runDelaySyscallCase(t, 0x00150200u, "indirect: ");
        });
    });
}
