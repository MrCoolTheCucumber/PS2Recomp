#include "MiniTest.h"

#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kBpcIae = 1u << 31u;
    constexpr uint32_t kBpcDre = 1u << 30u;
    constexpr uint32_t kBpcDwe = 1u << 29u;
    constexpr uint32_t kBpcDve = 1u << 28u;
    constexpr uint32_t kBpcIue = 1u << 26u;
    constexpr uint32_t kBpcIke = 1u << 24u;
    constexpr uint32_t kBpcIxe = 1u << 23u;
    constexpr uint32_t kBpcDue = 1u << 21u;
    constexpr uint32_t kBpcDke = 1u << 19u;
    constexpr uint32_t kBpcDxe = 1u << 18u;
    constexpr uint32_t kBpcBed = 1u << 15u;
    constexpr uint32_t kBpcDwb = 1u << 2u;
    constexpr uint32_t kBpcDrb = 1u << 1u;
    constexpr uint32_t kBpcIab = 1u << 0u;

    constexpr uint32_t kStatusErl = 1u << 2u;
    constexpr uint32_t kStatusUser = 2u << 3u;
    constexpr uint32_t kStatusDev = 1u << 23u;
    constexpr uint32_t kCauseExc2Mask = 0x00070000u;
    constexpr uint32_t kCauseExc2Debug = 0x00030000u;
    constexpr uint32_t kCauseBd2 = 1u << 30u;

    struct BreakpointInvocation
    {
        bool apiAvailable = false;
        bool valuePhaseRequested = false;
        bool raised = false;
    };

    template <typename Runtime>
    BreakpointInvocation invokeInstructionBreakpoint(
        Runtime &runtime,
        R5900Context &ctx,
        uint32_t address)
    {
        BreakpointInvocation result{};
        if constexpr (requires(Runtime &candidate, R5900Context *context)
                      {
                          candidate.CheckEeInstructionBreakpoint(context, address);
                      })
        {
            result.apiAvailable = true;
            try
            {
                runtime.CheckEeInstructionBreakpoint(&ctx, address);
            }
            catch (const PS2GuestException &)
            {
                result.raised = true;
            }
        }
        return result;
    }

    template <typename Runtime>
    BreakpointInvocation invokeDataBreakpoint(
        Runtime &runtime,
        R5900Context &ctx,
        uint32_t address,
        uint32_t value,
        bool write)
    {
        BreakpointInvocation result{};
        if constexpr (requires(Runtime &candidate, R5900Context *context)
                      {
                          {
                              candidate.CheckEeDataAddressBreakpoint(
                                  context, address, write)
                          } -> std::convertible_to<bool>;
                          candidate.CheckEeDataValueBreakpoint(
                              context, address, value, write);
                      })
        {
            result.apiAvailable = true;
            try
            {
                result.valuePhaseRequested =
                    runtime.CheckEeDataAddressBreakpoint(
                        &ctx, address, write);
                if (result.valuePhaseRequested)
                {
                    runtime.CheckEeDataValueBreakpoint(
                        &ctx, address, value, write);
                }
            }
            catch (const PS2GuestException &)
            {
                result.raised = true;
            }
        }
        return result;
    }

    Instruction decode(uint32_t address, uint32_t raw)
    {
        return R5900Decoder{}.decodeInstruction(
            address, raw, false);
    }

    Instruction decodeIType(
        uint32_t address,
        uint32_t opcode,
        uint32_t rs,
        uint32_t rt,
        uint16_t immediate)
    {
        return decode(
            address,
            (opcode << 26u) |
                (rs << 21u) |
                (rt << 16u) |
                immediate);
    }

    std::size_t countOccurrences(
        const std::string &text,
        const std::string &needle)
    {
        std::size_t count = 0u;
        std::size_t position = 0u;
        while ((position = text.find(needle, position)) !=
               std::string::npos)
        {
            ++count;
            position += needle.size();
        }
        return count;
    }

    Function makeFunction(
        const char *name,
        uint32_t start,
        uint32_t end)
    {
        Function function{};
        function.name = name;
        function.start = start;
        function.end = end;
        function.isRecompiled = true;
        return function;
    }
}

void register_cop0_hardware_breakpoint_tests()
{
    MiniTest::Case("COP0 hardware breakpoints", [](TestCase &tc)
    {
        tc.Run("generated instructions check before execution including delay NOPs", [](TestCase &t)
        {
            const Instruction ordinary = decode(0x1000u, 0u);
            const Instruction branch = decodeIType(
                0x1004u, OPCODE_BEQ, 1u, 1u, 1u);
            const Instruction delayNop = decode(0x1008u, 0u);
            const Instruction target = decode(0x100Cu, 0u);
            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    makeFunction("breakpoint_sites", 0x1000u, 0x1010u),
                    {ordinary, branch, delayNop, target},
                    false);

            const std::size_t ordinaryPc =
                generated.find("ctx->pc = 0x1000u;");
            const std::size_t ordinaryCheck =
                generated.find(
                    "runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);",
                    ordinaryPc);
            const std::size_t ordinaryRetire =
                generated.find(
                    "ctx->beginEeInstruction();",
                    ordinaryPc);
            t.IsTrue(
                ordinaryPc != std::string::npos &&
                    ordinaryCheck != std::string::npos &&
                    ordinaryCheck < ordinaryRetire,
                "an instruction breakpoint should be tested before the instruction retires");
            t.IsTrue(
                countOccurrences(
                    generated,
                    "runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);") >= 4u,
                "ordinary, branch, target, and executed delay-slot instructions should all be checked");

            const std::size_t delayState =
                generated.find("ctx->in_delay_slot = true;");
            const std::size_t delayCheck =
                generated.find(
                    "runtime->CheckEeInstructionBreakpoint(ctx, ctx->pc);",
                    delayState);
            t.IsTrue(
                delayState != std::string::npos &&
                    delayCheck != std::string::npos,
                "an elided NOP delay slot should still check its virtual instruction address");
        });

        tc.Run("generated mode writes exit only after a resumable instruction boundary", [](TestCase &t)
        {
            const auto mtc0 =
                [](uint32_t address,
                   uint32_t rd,
                   uint32_t selector)
                {
                    return decode(
                        address,
                        (OPCODE_COP0 << 26u) |
                            (COP0_MT << 21u) |
                            (8u << 16u) |
                            (rd << 11u) |
                            selector);
                };

            CodeGenerator generator({}, {});
            const Instruction bpcWrite =
                mtc0(0x4000u, COP0_REG_DEBUG,
                     COP0_BREAKPOINT_BPC);
            const std::string ordinary =
                generator.generateFunction(
                    makeFunction(
                        "observation_transition",
                        0x4000u, 0x4008u),
                    {bpcWrite, decode(0x4004u, 0u)},
                    false);

            const std::size_t modeBefore = ordinary.find(
                "eeObservationModeBefore_4000 = "
                "PS2Runtime::architecturalObservationMode(ctx);");
            const std::size_t write = ordinary.find(
                "runtime->writeCop0Breakpoint(ctx, 0u");
            const std::size_t completion = ordinary.find(
                "runtime->recordEeInstructionCompletion",
                write);
            const std::size_t nextPc = ordinary.find(
                "ctx->pc = 0x4004u;", completion);
            const std::size_t transition = ordinary.find(
                "completeEeObservationModeTransition(",
                nextPc);
            t.IsTrue(
                modeBefore != std::string::npos &&
                    write != std::string::npos &&
                    completion != std::string::npos &&
                    modeBefore < write &&
                    write < completion &&
                    completion < nextPc &&
                    nextPc < transition,
                "a BPC write should compare its pre-write mode only after completing and resolving the following instruction boundary");
            t.IsTrue(
                ordinary.find(
                    "rdram, ctx, eeObservationModeBefore_4000))") !=
                    std::string::npos,
                "the transition detector should not assume which shared specialization entry invoked the generated body");
            t.IsTrue(
                ordinary.find(
                    "case 0x4004u: goto label_4004;") !=
                    std::string::npos,
                "the following instruction should be a registered owner resume label");

            const Instruction pccrWrite =
                mtc0(0x4100u, COP0_REG_PERF, 0u);
            const std::string performance =
                generator.generateFunction(
                    makeFunction(
                        "performance_transition",
                        0x4100u, 0x4108u),
                    {pccrWrite, decode(0x4104u, 0u)},
                    false);
            t.IsTrue(
                performance.find(
                    "runtime->writeCop0Performance(rdram, ctx, 0u") !=
                        std::string::npos &&
                    performance.find(
                        "completeEeObservationModeTransition(") !=
                        std::string::npos,
                "PCCR writes should share the same post-completion transition boundary");
            const std::size_t performanceWrite = performance.find(
                "runtime->writeCop0Performance(rdram, ctx, 0u");
            const std::size_t performanceCompletion = performance.find(
                "runtime->recordEeInstructionCompletion",
                performanceWrite);
            const std::size_t performanceTransition = performance.find(
                "completeEeObservationModeTransition(",
                performanceCompletion);
            t.IsTrue(
                performanceWrite < performanceCompletion &&
                    performanceCompletion < performanceTransition,
                "PCCR transition detection should occur only after the MTC0 completion event has been counted");

            const Instruction branch = decodeIType(
                0x5000u, OPCODE_BEQ, 1u, 1u, 2u);
            const Instruction delayWrite =
                mtc0(0x5004u, COP0_REG_DEBUG,
                     COP0_BREAKPOINT_BPC);
            const std::string delayed =
                generator.generateFunction(
                    makeFunction(
                        "delay_transition",
                        0x5000u, 0x5010u),
                    {branch, delayWrite,
                     decode(0x5008u, 0u),
                     decode(0x500Cu, 0u)},
                    false);
            const std::size_t delayCleared = delayed.find(
                "ctx->in_delay_slot = false;");
            const std::size_t delayedTransition = delayed.find(
                "completeEeObservationModeTransition(",
                delayCleared);
            t.IsTrue(
                delayCleared != std::string::npos &&
                    delayedTransition != std::string::npos &&
                    delayCleared < delayedTransition,
                "a delay-slot mode write should not exit until branch-delay ownership is cleared");
            t.IsTrue(
                delayed.find(
                    "case 0x5008u: goto label_5008;") !=
                        std::string::npos &&
                    delayed.find(
                        "case 0x500cu: goto label_500c;") !=
                        std::string::npos,
                "both resolved conditional continuations should be resumable after a delay-slot transition");

            const Instruction likelyBranch = decodeIType(
                0x5100u, OPCODE_BEQL, 1u, 1u, 2u);
            const std::string likely =
                generator.generateFunction(
                    makeFunction(
                        "likely_delay_transition",
                        0x5100u, 0x5110u),
                    {likelyBranch,
                     mtc0(0x5104u, COP0_REG_DEBUG,
                          COP0_BREAKPOINT_BPC),
                     decode(0x5108u, 0u),
                     decode(0x510Cu, 0u)},
                    false);
            t.Equals(
                countOccurrences(
                    likely,
                    "runtime->writeCop0Breakpoint(ctx, 0u"),
                static_cast<std::size_t>(1u),
                "a likely branch should emit the mode-changing delay effect only on its taken path");
            t.Equals(
                countOccurrences(
                    likely,
                    "completeEeObservationModeTransition("),
                static_cast<std::size_t>(1u),
                "an annulled likely delay slot should not test a transition that never occurred");
            t.IsTrue(
                likely.find(
                    "case 0x5108u: goto label_5108;") !=
                        std::string::npos &&
                    likely.find(
                        "case 0x510cu: goto label_510c;") !=
                        std::string::npos,
                "likely annulment and taken execution should retain distinct resumable continuations");

            const Instruction returnBranch = decode(
                0x6000u,
                (OPCODE_SPECIAL << 26u) |
                    (31u << 21u) |
                    SPECIAL_JR);
            const std::string returning =
                generator.generateFunction(
                    makeFunction(
                        "return_transition",
                        0x6000u, 0x6008u),
                    {returnBranch,
                     mtc0(0x6004u, COP0_REG_DEBUG,
                          COP0_BREAKPOINT_BPC)},
                    false);
            const std::size_t returnTransition = returning.find(
                "completeEeObservationModeTransitionAtGuestBranch(");
            const std::size_t returnValidation = returning.find(
                "ValidateGuestBranchTarget(");
            t.IsTrue(
                returnTransition != std::string::npos &&
                    returnValidation != std::string::npos &&
                    returnTransition < returnValidation,
                "return transitions should preserve callback-sentinel validation inside the transition boundary");

            CodeGenerator relocatedGenerator({}, {});
            relocatedGenerator.setRelocationCallNames(
                {{0x6100u, "ReleaseAlarm"}});
            const uint32_t relocatedTarget = 0x7000u;
            const std::string relocated =
                relocatedGenerator.generateFunction(
                    makeFunction(
                        "relocated_transition",
                        0x6100u, 0x6108u),
                    {decode(
                         0x6100u,
                         (OPCODE_JAL << 26u) |
                             ((relocatedTarget >> 2u) &
                              0x03ffffffu)),
                     mtc0(0x6104u, COP0_REG_DEBUG,
                          COP0_BREAKPOINT_BPC)},
                    false);
            const std::size_t deferredTransition = relocated.find(
                "const bool eeObservationModeChangedAfter_6104 = "
                "runtime->completeEeObservationModeTransition(");
            const std::size_t replacement = relocated.find(
                "ps2_syscalls::ReleaseAlarm(rdram, ctx, runtime);");
            const std::size_t deferredExit = relocated.find(
                "if (eeObservationModeChangedAfter_6104) { return; }",
                replacement);
            t.IsTrue(
                deferredTransition != std::string::npos &&
                    replacement != std::string::npos &&
                    deferredExit != std::string::npos &&
                    deferredTransition < replacement &&
                    replacement < deferredExit,
                "a relocated call should finish its replacement handler before unwinding the old specialization");
        });

        tc.Run("instruction match enters the normal level two debug vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00123450u;
            ctx.cop0_bpc = kBpcIae | kBpcIke;
            ctx.cop0_iab = 0x00123400u;
            ctx.cop0_iabm = 0xFFFFFF00u;

            const BreakpointInvocation result =
                invokeInstructionBreakpoint(
                    runtime, ctx, ctx.pc);
            t.IsTrue(result.apiAvailable,
                     "the runtime should expose an architectural instruction-breakpoint check");
            t.IsTrue(result.raised,
                     "a matching enabled kernel instruction breakpoint should unwind execution");
            t.IsTrue((ctx.cop0_bpc & kBpcIab) != 0u,
                     "instruction matches should set BPC.IAB");
            t.Equals(ctx.cop0_errorepc, 0x00123450u,
                     "ErrorEPC should identify the matching instruction");
            t.Equals(ctx.cop0_cause & kCauseExc2Mask, kCauseExc2Debug,
                     "Cause.EXC2 should identify a debug exception");
            t.IsTrue((ctx.cop0_status & kStatusErl) != 0u,
                     "debug exception entry should set Status.ERL");
            t.Equals(ctx.pc, 0x80000100u,
                     "Status.DEV zero should select the normal debug vector");
        });

        tc.Run("delay match records BD2 and selects the bootstrap debug vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00104004u;
            ctx.branch_pc = 0x00104000u;
            ctx.in_delay_slot = true;
            ctx.cop0_status = kStatusDev;
            ctx.cop0_bpc = kBpcIae | kBpcIke;
            ctx.cop0_iab = ctx.pc;
            ctx.cop0_iabm = 0xFFFFFFFCu;

            const BreakpointInvocation result =
                invokeInstructionBreakpoint(runtime, ctx, ctx.pc);
            t.IsTrue(result.apiAvailable && result.raised,
                     "a delay-slot instruction match should enter the debug handler");
            t.Equals(ctx.cop0_errorepc, 0x00104000u,
                     "a delay-slot debug exception should restart at its branch");
            t.IsTrue((ctx.cop0_cause & kCauseBd2) != 0u,
                     "a delay-slot debug exception should set Cause.BD2");
            t.Equals(ctx.pc, 0xBFC00300u,
                     "Status.DEV should select the bootstrap debug vector");
            t.IsFalse(ctx.in_delay_slot,
                      "exception entry should leave delay-slot execution");
        });

        tc.Run("instruction mode gates BED and ERL follow BPC", [](TestCase &t)
        {
            constexpr uint32_t address = 0x00125000u;

            PS2Runtime kernelOnlyRuntime;
            R5900Context kernelOnly{};
            kernelOnly.pc = address;
            kernelOnly.cop0_status = kStatusUser;
            kernelOnly.cop0_bpc = kBpcIae | kBpcIke;
            kernelOnly.cop0_iab = address;
            kernelOnly.cop0_iabm = 0xFFFFFFFCu;
            const BreakpointInvocation kernelOnlyResult =
                invokeInstructionBreakpoint(
                    kernelOnlyRuntime, kernelOnly, address);
            t.IsTrue(kernelOnlyResult.apiAvailable,
                     "the runtime instruction check should be available");
            t.IsFalse(kernelOnlyResult.raised,
                      "a kernel-only breakpoint should not fire in user mode");
            t.IsFalse((kernelOnly.cop0_bpc & kBpcIab) != 0u,
                      "a mode-disabled comparison should not establish IAB");

            PS2Runtime userRuntime;
            R5900Context user{};
            user.pc = address;
            user.cop0_status = kStatusUser;
            user.cop0_bpc = kBpcIae | kBpcIue;
            user.cop0_iab = address;
            user.cop0_iabm = 0xFFFFFFFCu;
            const BreakpointInvocation userResult =
                invokeInstructionBreakpoint(
                    userRuntime, user, address);
            t.IsTrue(userResult.raised,
                     "an IUE-enabled instruction breakpoint should fire in user mode");

            PS2Runtime bedRuntime;
            R5900Context bed{};
            bed.pc = address;
            bed.cop0_bpc = kBpcIae | kBpcIke | kBpcBed;
            bed.cop0_iab = address;
            bed.cop0_iabm = 0xFFFFFFFCu;
            const BreakpointInvocation bedResult =
                invokeInstructionBreakpoint(
                    bedRuntime, bed, address);
            t.IsFalse(bedResult.raised,
                      "BPC.BED should suppress debug exception generation");
            t.IsTrue((bed.cop0_bpc & kBpcIab) != 0u,
                     "BED should not suppress breakpoint establishment flags");

            PS2Runtime erlRuntime;
            R5900Context erl{};
            erl.pc = address;
            erl.cop0_status = kStatusErl;
            erl.cop0_bpc = kBpcIae | kBpcIxe;
            erl.cop0_iab = address;
            erl.cop0_iabm = 0xFFFFFFFCu;
            const BreakpointInvocation erlResult =
                invokeInstructionBreakpoint(
                    erlRuntime, erl, address);
            t.IsFalse(erlResult.raised,
                      "Status.ERL should mask debug exceptions");
            t.IsFalse((erl.cop0_bpc & kBpcIab) != 0u,
                      "level-two mode should bypass breakpoint establishment");
        });

        tc.Run("data read and write matches set independent direction flags", [](TestCase &t)
        {
            constexpr uint32_t address = 0x00126034u;

            PS2Runtime readRuntime;
            R5900Context read{};
            read.pc = 0x00110000u;
            read.cop0_bpc = kBpcDre | kBpcDke;
            read.cop0_dab = 0x00126000u;
            read.cop0_dabm = 0xFFFFFF00u;
            const BreakpointInvocation readResult =
                invokeDataBreakpoint(
                    readRuntime, read, address, 0u, false);
            t.IsTrue(readResult.apiAvailable && readResult.raised,
                     "a matching enabled data read should enter the debug handler");
            t.IsTrue((read.cop0_bpc & kBpcDrb) != 0u,
                     "a read match should set BPC.DRB");
            t.IsFalse((read.cop0_bpc & kBpcDwb) != 0u,
                      "a read match should not set BPC.DWB");

            PS2Runtime disabledWriteRuntime;
            R5900Context disabledWrite{};
            disabledWrite.cop0_bpc = kBpcDre | kBpcDke;
            disabledWrite.cop0_dab = address;
            disabledWrite.cop0_dabm = 0xFFFFFFFFu;
            const BreakpointInvocation disabledWriteResult =
                invokeDataBreakpoint(
                    disabledWriteRuntime,
                    disabledWrite,
                    address,
                    0x12345678u,
                    true);
            t.IsFalse(disabledWriteResult.raised,
                      "DRE alone should not enable write breakpoints");
            t.IsFalse((disabledWrite.cop0_bpc & kBpcDwb) != 0u,
                      "a direction-disabled write should not establish DWB");

            PS2Runtime writeRuntime;
            R5900Context write{};
            write.pc = 0x00110004u;
            write.cop0_bpc = kBpcDwe | kBpcDke;
            write.cop0_dab = address;
            write.cop0_dabm = 0xFFFFFFFFu;
            const BreakpointInvocation writeResult =
                invokeDataBreakpoint(
                    writeRuntime,
                    write,
                    address,
                    0x12345678u,
                    true);
            t.IsTrue(writeResult.raised,
                     "a matching enabled data write should enter the debug handler");
            t.IsTrue((write.cop0_bpc & kBpcDwb) != 0u,
                     "a write match should set BPC.DWB");
        });

        tc.Run("data value masking gates breakpoint establishment", [](TestCase &t)
        {
            constexpr uint32_t address = 0x00127000u;
            constexpr uint32_t valueMask = 0xFFFF00FFu;
            constexpr uint32_t breakValue = 0x89AB00EFu;

            PS2Runtime mismatchRuntime;
            R5900Context mismatch{};
            mismatch.cop0_bpc =
                kBpcDre | kBpcDve | kBpcDke;
            mismatch.cop0_dab = address;
            mismatch.cop0_dabm = 0xFFFFFFFFu;
            mismatch.cop0_dvb = breakValue;
            mismatch.cop0_dvbm = valueMask;
            const BreakpointInvocation mismatchResult =
                invokeDataBreakpoint(
                    mismatchRuntime,
                    mismatch,
                    address,
                    0x89AC55EFu,
                    false);
            t.IsTrue(mismatchResult.apiAvailable,
                     "the runtime data breakpoint phases should be available");
            t.IsTrue(mismatchResult.valuePhaseRequested,
                     "DVE should defer establishment until loaded data is available");
            t.IsFalse(mismatchResult.raised,
                      "a masked data-value mismatch should not raise");
            t.IsFalse((mismatch.cop0_bpc & kBpcDrb) != 0u,
                      "a data-value mismatch should not establish DRB");

            PS2Runtime matchRuntime;
            R5900Context match{};
            match.pc = 0x00111000u;
            match.cop0_bpc =
                kBpcDre | kBpcDve | kBpcDke;
            match.cop0_dab = address;
            match.cop0_dabm = 0xFFFFFFFFu;
            match.cop0_dvb = breakValue;
            match.cop0_dvbm = valueMask;
            const BreakpointInvocation matchResult =
                invokeDataBreakpoint(
                    matchRuntime,
                    match,
                    address,
                    0x89AB55EFu,
                    false);
            t.IsTrue(matchResult.valuePhaseRequested && matchResult.raised,
                     "a masked data-value match should raise after the value phase");
            t.IsTrue((match.cop0_bpc & kBpcDrb) != 0u,
                     "a data-value match should establish DRB");
        });

        tc.Run("data mode gates work in user and level one modes", [](TestCase &t)
        {
            constexpr uint32_t address = 0x00128000u;

            PS2Runtime userRuntime;
            R5900Context user{};
            user.pc = 0x00112000u;
            user.cop0_status = kStatusUser;
            user.cop0_bpc = kBpcDre | kBpcDue;
            user.cop0_dab = address;
            user.cop0_dabm = 0xFFFFFFFFu;
            const BreakpointInvocation userResult =
                invokeDataBreakpoint(
                    userRuntime, user, address, 0u, false);
            t.IsTrue(userResult.apiAvailable && userResult.raised,
                     "DUE should enable data breakpoints in user mode");

            PS2Runtime levelOneRuntime;
            R5900Context levelOne{};
            levelOne.pc = 0x00112004u;
            levelOne.cop0_status = 1u << 1u;
            levelOne.cop0_bpc = kBpcDwe | kBpcDxe;
            levelOne.cop0_dab = address;
            levelOne.cop0_dabm = 0xFFFFFFFFu;
            const BreakpointInvocation levelOneResult =
                invokeDataBreakpoint(
                    levelOneRuntime,
                    levelOne,
                    address,
                    0xA5A55A5Au,
                    true);
            t.IsTrue(levelOneResult.raised,
                     "DXE should enable data breakpoints while Status.EXL is set");
        });

        tc.Run("generated memory operations supply architectural data values", [](TestCase &t)
        {
            CodeGenerator generator({}, {});
            const std::string lb = generator.translateInstruction(
                decodeIType(0x2000u, OPCODE_LB, 1u, 2u, 4u));
            const std::string sb = generator.translateInstruction(
                decodeIType(0x2004u, OPCODE_SB, 1u, 2u, 5u));
            const std::string lq = generator.translateInstruction(
                decodeIType(0x2008u, OPCODE_LQ, 1u, 2u, 6u));
            const std::string lwl = generator.translateInstruction(
                decodeIType(0x200Cu, OPCODE_LWL, 1u, 2u, 7u));
            const std::string swl = generator.translateInstruction(
                decodeIType(0x2010u, OPCODE_SWL, 1u, 2u, 8u));

            const std::size_t lbAddressCheck =
                lb.find("CheckEeDataAddressBreakpoint");
            const std::size_t lbRead = lb.find("READ8(");
            const std::size_t lbValueCheck =
                lb.find("CheckEeDataValueBreakpoint");
            t.IsTrue(
                lbAddressCheck < lbRead && lbRead < lbValueCheck,
                "a signed load should check address-only breakpoints before the read and data-value breakpoints afterward");
            t.IsTrue(
                lb.find("(uint32_t)(int32_t)(int8_t)eeBreakpointValue") !=
                    std::string::npos,
                "LB should compare the sign-extended loaded byte");
            t.IsTrue(
                sb.find("CheckEeDataValueBreakpoint") != std::string::npos &&
                    sb.find("GPR_U32(ctx, 2)") != std::string::npos,
                "SB should compare the complete lower 32 bits of its source GPR");
            t.IsTrue(
                lq.find("Ps2ExtractEpi32(eeBreakpointValue, 0)") !=
                    std::string::npos,
                "LQ should compare the low 32 bits of loaded data");
            t.IsTrue(
                lwl.find("CheckEeDataValueBreakpoint") !=
                        std::string::npos &&
                    lwl.find("uint32_t merged =") !=
                        std::string::npos &&
                    lwl.find("CheckEeDataValueBreakpoint") >
                        lwl.find("uint32_t merged ="),
                "LWL should compare the final value after merging with its GPR");
            t.IsTrue(
                swl.find("CheckEeDataValueBreakpoint") !=
                        std::string::npos &&
                    swl.find("GPR_U32(ctx, 2)") !=
                        std::string::npos,
                "SWL should compare the unshifted lower 32 bits of its source GPR");
        });

        tc.Run("GIF DMA coalescing retains an architectural breakpoint path", [](TestCase &t)
        {
            const std::vector<Instruction> instructions{
                decodeIType(0x3000u, OPCODE_ADDIU, 0u, 2u, 4u),
                decodeIType(0x3004u, OPCODE_LUI, 0u, 1u, 0x1000u),
                decodeIType(0x3008u, OPCODE_ORI, 1u, 1u, 0xE020u),
                decodeIType(0x300Cu, OPCODE_SW, 1u, 2u, 0u),
                decodeIType(0x3010u, OPCODE_LUI, 0u, 1u, 0x1000u),
                decodeIType(0x3014u, OPCODE_ORI, 1u, 1u, 0xE010u),
                decodeIType(0x3018u, OPCODE_SW, 1u, 2u, 0u),
                decodeIType(0x301Cu, OPCODE_LUI, 0u, 1u, 0x1000u),
                decodeIType(0x3020u, OPCODE_ORI, 1u, 1u, 0xA030u),
                decodeIType(0x3024u, OPCODE_SW, 1u, 4u, 0u),
                decodeIType(0x3028u, OPCODE_ADDIU, 0u, 5u, 0x0105u),
                decodeIType(0x302Cu, OPCODE_ADDIU, 1u, 1u, 0xFFD0u),
                decodeIType(0x3030u, OPCODE_SW, 1u, 5u, 0u),
            };
            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    makeFunction("breakpoint_gif_dma", 0x3000u, 0x3034u),
                    instructions,
                    false);

            t.IsTrue(
                generated.find("kickGifDmaChainFromMMIO") !=
                    std::string::npos,
                "the fixture should exercise the coalesced GIF-DMA path");
            t.IsTrue(
                generated.find("EeDataBreakpointEnabled") !=
                    std::string::npos,
                "coalescing should select individual stores while architectural data breakpoints are enabled");
            t.IsTrue(
                generated.find("CheckEeDataAddressBreakpoint") !=
                    std::string::npos,
                "the individual-store fallback should retain breakpoint matching");
        });
    });
}
