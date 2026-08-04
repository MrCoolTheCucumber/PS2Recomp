#include "MiniTest.h"

#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ee_performance_model.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "runtime/ee_performance.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace ps2recomp;
using namespace ps2x::performance;

namespace
{
    constexpr uint32_t kCodeAddress = 0x00160000u;
    constexpr uint32_t kPccrCte = 1u << 31u;
    constexpr uint32_t kPccrK0 = 1u << 2u;
    constexpr uint32_t kPccrK1 = 1u << 12u;
    constexpr uint32_t kConfigBpe = 1u << 12u;
    constexpr uint32_t kConfigDie = 1u << 18u;

    uint32_t performanceControl(
        uint32_t event0,
        uint32_t event1)
    {
        return kPccrCte |
               kPccrK0 |
               kPccrK1 |
               (event0 << 5u) |
               (event1 << 15u);
    }

    void configurePerformance(
        PS2Runtime &runtime,
        R5900Context &ctx,
        uint32_t event0,
        uint32_t event1,
        uint32_t pcr0 = 0u,
        uint32_t pcr1 = 0u)
    {
        runtime.writeCop0Performance(
            nullptr, &ctx, 1u, pcr0);
        runtime.writeCop0Performance(
            nullptr, &ctx, 3u, pcr1);
        runtime.writeCop0Performance(
            nullptr, &ctx, 0u,
            performanceControl(event0, event1));
    }

    Instruction decode(
        uint32_t address,
        uint32_t raw)
    {
        return R5900Decoder{}.decodeInstruction(
            address, raw, false);
    }

    std::string generateFunction(
        const std::vector<uint32_t> &rawInstructions)
    {
        std::vector<Instruction> instructions;
        instructions.reserve(rawInstructions.size());
        for (size_t i = 0u;
             i < rawInstructions.size();
             ++i)
        {
            instructions.push_back(decode(
                kCodeAddress +
                    static_cast<uint32_t>(i * 4u),
                rawInstructions[i]));
        }

        Function function{};
        function.name = "performance_events";
        function.start = kCodeAddress;
        function.end = kCodeAddress +
                       static_cast<uint32_t>(
                           rawInstructions.size() * 4u);
        function.isRecompiled = true;
        return CodeGenerator({}, {}).generateFunction(
            function, instructions, false);
    }
}

void register_ee_performance_event_tests()
{
    MiniTest::Case("EE performance events", [](TestCase &tc)
    {
        tc.Run("completion hooks select instruction COP and memory events", [](TestCase &t)
        {
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                configurePerformance(
                    runtime, ctx, 12u, 13u);
                runtime.recordEeInstructionCompletion(
                    &ctx,
                    kInstructionCompleted |
                        kNonBdsInstructionCompleted);
                t.Equals(
                    ctx.cop0_pcr0, 1u,
                    "EVENT0 12 should observe an instruction completion");
                t.Equals(
                    ctx.cop0_pcr1, 1u,
                    "EVENT1 13 should observe a non-BDS completion");
            }

            {
                PS2Runtime runtime;
                R5900Context ctx{};
                configurePerformance(
                    runtime, ctx, 14u, 14u);
                runtime.recordEeInstructionCompletion(
                    &ctx, kCop2InstructionCompleted);
                runtime.recordEeInstructionCompletion(
                    &ctx, kCop1InstructionCompleted);
                t.Equals(
                    ctx.cop0_pcr0, 1u,
                    "EVENT0 14 should select completed COP2 instructions");
                t.Equals(
                    ctx.cop0_pcr1, 1u,
                    "EVENT1 14 should select completed COP1 instructions");
            }

            {
                PS2Runtime runtime;
                R5900Context ctx{};
                configurePerformance(
                    runtime, ctx, 15u, 15u);
                runtime.recordEeInstructionCompletion(
                    &ctx, kLoadCompleted);
                runtime.recordEeInstructionCompletion(
                    &ctx, kStoreCompleted);
                t.Equals(
                    ctx.cop0_pcr0, 1u,
                    "EVENT0 15 should select completed loads");
                t.Equals(
                    ctx.cop0_pcr1, 1u,
                    "EVENT1 15 should select completed stores");
            }
        });

        tc.Run("data bus errors retain only their load or store completion", [](TestCase &t)
        {
            for (const auto access :
                 {PS2BusErrorAccess::DataLoad,
                  PS2BusErrorAccess::DataStore})
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.pc = 0x00103500u;
                configurePerformance(
                    runtime, ctx, 15u, 15u);

                bool raised = false;
                try
                {
                    runtime.SignalBusError(
                        &ctx, access, 0x22001234u);
                }
                catch (const PS2GuestException &)
                {
                    raised = true;
                }

                t.IsTrue(
                    raised,
                    "an unmasked data bus error should enter the guest exception path");
                t.Equals(
                    ctx.cop0_pcr0,
                    access == PS2BusErrorAccess::DataLoad
                        ? 1u
                        : 0u,
                    "a bus-erroring access should count as a load only when it reads");
                t.Equals(
                    ctx.cop0_pcr1,
                    access == PS2BusErrorAccess::DataStore
                        ? 1u
                        : 0u,
                    "a bus-erroring access should count as a store only when it writes");
            }

            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00103600u;
            configurePerformance(
                runtime, ctx, 12u, 13u);
            try
            {
                runtime.SignalBusError(
                    &ctx,
                    PS2BusErrorAccess::DataLoad,
                    0x22005678u);
            }
            catch (const PS2GuestException &)
            {
            }
            t.Equals(
                ctx.cop0_pcr0, 0u,
                "a bus error should cancel the generic instruction completion");
            t.Equals(
                ctx.cop0_pcr1, 0u,
                "a bus error should cancel the generic non-BDS completion");
        });

        tc.Run("issue hooks distinguish dual issue and dependency stalls", [](TestCase &t)
        {
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.cop0_config |= kConfigDie;
                configurePerformance(
                    runtime, ctx, 2u, 2u);

                runtime.recordEeInstructionIssue(
                    &ctx,
                    {kIssuePipe0, 1u << 2u,
                     1u << 3u, 0u});
                runtime.recordEeInstructionIssue(
                    &ctx,
                    {kIssuePipe1, 1u << 4u,
                     1u << 5u, 0u});

                t.Equals(
                    ctx.cop0_pcr0, 0u,
                    "independent instructions on distinct pipes should not count as single issue");
                t.Equals(
                    ctx.cop0_pcr1, 1u,
                    "independent instructions on distinct pipes should count one dual issue slot");
            }

            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.cop0_config |= kConfigDie;
                configurePerformance(
                    runtime, ctx, 2u, 2u);

                runtime.recordEeInstructionIssue(
                    &ctx,
                    {kIssuePipe0, 0u,
                     1u << 6u, 0u});
                runtime.recordEeInstructionIssue(
                    &ctx,
                    {kIssuePipe1, 1u << 6u,
                     1u << 7u, 0u});
                runtime.serviceEeEventsAtBlockBoundary(
                    nullptr, &ctx);

                t.Equals(
                    ctx.cop0_pcr0, 2u,
                    "a GPR RAW dependency should leave both instructions in single issue slots");
                t.Equals(
                    ctx.cop0_pcr1, 0u,
                    "a dependency-stalled pair must not count as dual issue");
            }

            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.cop0_config &= ~kConfigDie;
                configurePerformance(
                    runtime, ctx, 2u, 2u);

                runtime.recordEeInstructionIssue(
                    &ctx, {kIssuePipe0, 0u, 0u, 0u});
                runtime.recordEeInstructionIssue(
                    &ctx, {kIssuePipe1, 0u, 0u, 0u});

                t.Equals(
                    ctx.cop0_pcr0, 2u,
                    "Config.DIE clear should force one single issue event per instruction");
                t.Equals(
                    ctx.cop0_pcr1, 0u,
                    "Config.DIE clear must suppress dual issue events");
            }
        });

        tc.Run("branch hooks count predicted forms and conditional misses", [](TestCase &t)
        {
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.cop0_config |= kConfigBpe;
                configurePerformance(
                    runtime, ctx, 3u, 3u);

                runtime.recordEeConditionalBranch(
                    &ctx, 0x00101000u, true);
                runtime.recordEeConditionalBranch(
                    &ctx, 0x00101000u, true);
                runtime.recordEeConditionalBranch(
                    &ctx, 0x00101000u, true);

                t.Equals(
                    ctx.cop0_pcr0, 3u,
                    "every conditional branch should count as an issued predictive branch");
                t.Equals(
                    ctx.cop0_pcr1, 2u,
                    "a two-bit predictor should miss the first two taken outcomes from strongly not-taken");
            }

            {
                PS2Runtime runtime;
                R5900Context ctx{};
                configurePerformance(
                    runtime, ctx, 16u, 0u);

                runtime.recordEePredictedBranch(
                    &ctx, 0x00102000u);
                runtime.recordEePredictedBranch(
                    &ctx, 0x00102004u);
                runtime.recordEeConditionalBranch(
                    &ctx, 0x00102008u, false);

                t.Equals(
                    ctx.cop0_pcr1, 2u,
                    "EVENT1 0 should count predictive branches only at even instruction positions");
            }

            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.cop0_config &= ~kConfigBpe;
                configurePerformance(
                    runtime, ctx, 16u, 3u);
                runtime.recordEeConditionalBranch(
                    &ctx, 0x00103000u, true);
                t.Equals(
                    ctx.cop0_pcr1, 0u,
                    "Config.BPE clear should not report a branch prediction miss");
            }
        });

        tc.Run("occurrence overflow owns a due performance event", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00104000u;
            configurePerformance(
                runtime, ctx, 12u, 16u,
                0x7fffffffu, 0u);

            runtime.recordEeInstructionCompletion(
                &ctx, kInstructionCompleted);
            const auto scheduled =
                runtime.debugEeSchedulerSnapshot();
            const auto &slot =
                scheduled.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            Cop0Performance)];
            t.IsTrue(
                slot.pending,
                "an occurrence overflow should schedule the level-2 event at the current EE tick");

            bool raised = false;
            try
            {
                runtime.serviceEeEventsAtBlockBoundary(
                    nullptr, &ctx);
            }
            catch (const PS2GuestException &)
            {
                raised = true;
            }
            t.IsTrue(
                raised,
                "the next architectural event boundary should deliver occurrence overflow");
            t.Equals(
                ctx.pc, 0x80000080u,
                "occurrence overflow should use the normal performance vector");
            t.Equals(
                ctx.cop0_cause & 0x00070000u,
                0x00020000u,
                "occurrence overflow should publish Cause.EXC2=performance");
        });

        tc.Run("performance overflow precedes a simultaneous level-one exception", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00105000u;
            configurePerformance(
                runtime, ctx, 12u, 16u,
                0x7fffffffu, 0u);
            runtime.recordEeInstructionCompletion(
                &ctx, kInstructionCompleted);

            try
            {
                runtime.handleBreak(nullptr, &ctx);
            }
            catch (const PS2GuestException &)
            {
            }
            t.Equals(
                ctx.pc, 0x80000080u,
                "performance overflow should outrank BREAK's level-one vector");
            t.Equals(
                ctx.cop0_cause & 0x00070000u,
                0x00020000u,
                "the higher-priority level-two cause should remain visible");
        });

        tc.Run("decoder profiles follow documented logical pipes and GPR hazards", [](TestCase &t)
        {
            const Instruction addiu = decode(
                kCodeAddress,
                (OPCODE_ADDIU << 26u) |
                    (2u << 21u) | (3u << 16u) | 1u);
            const auto addiuInfo =
                eeInstructionPerformanceInfo(addiu);
            t.Equals(
                addiuInfo.issuePipeMask,
                kIssuePipe0 | kIssuePipe1,
                "ordinary ALU instructions should bind to either logical pipe");
            t.Equals(
                addiu.gprReadMask, 1u << 2u,
                "the issue profile should retain the decoded source dependency");
            t.Equals(
                addiu.gprWriteMask, 1u << 3u,
                "the issue profile should retain the decoded destination dependency");

            const Instruction load = decode(
                kCodeAddress,
                (OPCODE_LW << 26u) |
                    (2u << 21u) | (3u << 16u));
            t.Equals(
                eeInstructionPerformanceInfo(load)
                    .issuePipeMask,
                kIssuePipe1,
                "loads should issue through logical pipe 1");

            const Instruction plzcw = decode(
                kCodeAddress,
                (OPCODE_MMI << 26u) |
                    (2u << 21u) | (3u << 11u) |
                    MMI_PLZCW);
            t.Equals(
                eeInstructionPerformanceInfo(plzcw)
                    .issuePipeMask,
                kIssuePipe1,
                "PLZCW should use the pipe-1 LZC unit");

            const Instruction mtsab = decode(
                kCodeAddress,
                (OPCODE_REGIMM << 26u) |
                    (2u << 21u) |
                    (REGIMM_MTSAB << 16u));
            t.Equals(
                eeInstructionPerformanceInfo(mtsab)
                    .issuePipeMask,
                kIssuePipe0,
                "MTSAB should use the pipe-0 SA unit");
        });

        tc.Run("completion profiles preserve counter-specific categories", [](TestCase &t)
        {
            const Instruction branch = decode(
                kCodeAddress,
                (OPCODE_BEQ << 26u) |
                    (1u << 21u) | (2u << 16u) | 1u);
            t.Equals(
                eeInstructionPerformanceInfo(branch)
                    .completionEvents,
                kInstructionCompleted,
                "a branch should not count as a non-BDS completion");

            const Instruction load = decode(
                kCodeAddress,
                (OPCODE_LW << 26u) |
                    (1u << 21u) | (2u << 16u));
            t.Equals(
                eeInstructionPerformanceInfo(load)
                    .completionEvents,
                kInstructionCompleted |
                    kNonBdsInstructionCompleted |
                    kLoadCompleted,
                "a scalar load should expose instruction non-BDS and load completion");

            const Instruction cop1 = decode(
                kCodeAddress,
                (OPCODE_COP1 << 26u) |
                    (COP1_MF << 21u) |
                    (2u << 16u) | (3u << 11u));
            t.Equals(
                eeInstructionPerformanceInfo(cop1)
                    .completionEvents,
                kInstructionCompleted |
                    kNonBdsInstructionCompleted |
                    kCop1InstructionCompleted,
                "a COP1 move should expose the COP1 completion category");

            const Instruction cop2 = decode(
                kCodeAddress,
                (OPCODE_COP2 << 26u) |
                    (COP2_QMFC2 << 21u) |
                    (2u << 16u) | (3u << 11u));
            t.Equals(
                eeInstructionPerformanceInfo(cop2)
                    .completionEvents,
                kInstructionCompleted |
                    kNonBdsInstructionCompleted |
                    kCop2InstructionCompleted,
                "a COP2 move should expose the COP2 completion category");
        });

        tc.Run("generated instructions place issue and completion at precise boundaries", [](TestCase &t)
        {
            const uint32_t addiuRaw =
                (OPCODE_ADDIU << 26u) |
                (2u << 21u) | (3u << 16u) | 1u;
            const std::string ordinary =
                generateFunction({addiuRaw});
            const size_t issue = ordinary.find(
                "observeEeInstructionBeginPolicy<Mode>");
            const size_t effect = ordinary.find(
                "SET_GPR_S32(ctx, 3");
            const size_t completion = ordinary.find(
                "observeEeInstructionCompletePolicy<Mode>");
            t.IsTrue(
                issue < effect && effect < completion,
                "an ordinary instruction should issue before its effect and complete afterward");

            const std::string syscall =
                generateFunction({SPECIAL_SYSCALL});
            t.IsTrue(
                syscall.find(
                    "observeEeInstructionCompletePolicy<Mode>") <
                    syscall.find("runtime->handleSyscall"),
                "SYSCALL's architecturally owned exception should retain its completion event");

            const uint32_t eretRaw =
                (OPCODE_COP0 << 26u) |
                (COP0_CO << 21u) |
                COP0_CO_ERET;
            const std::string eret =
                generateFunction({eretRaw});
            t.IsTrue(
                eret.find(
                    "observeEeInstructionCompletePolicy<Mode>") <
                    eret.find("runtime->handleCop0Eret"),
                "ERET should complete before its generated return transfers control");

            const uint32_t lwRaw =
                (OPCODE_LW << 26u) |
                (2u << 21u) | (3u << 16u);
            const std::string load =
                generateFunction({lwRaw});
            t.IsTrue(
                load.find("READ32(") <
                    load.find(
                        "observeEeInstructionCompletePolicy<Mode>"),
                "a load should complete only after a faulting memory access returns");
        });

        tc.Run("generated control flow records predictive and annulled events", [](TestCase &t)
        {
            const uint32_t conditionalRaw =
                (OPCODE_BEQL << 26u) |
                (1u << 21u) | (2u << 16u) | 2u;
            const uint32_t delayLoadRaw =
                (OPCODE_LW << 26u) |
                (4u << 21u) | (5u << 16u);
            const std::string conditional =
                generateFunction(
                    {conditionalRaw, delayLoadRaw, 0u, 0u});
            t.IsTrue(
                conditional.find(
                    "observeEeConditionalBranchPolicy<Mode>") !=
                    std::string::npos,
                "conditional branches should publish their resolved outcome");
            const size_t annulled = conditional.find(
                "if (!branch_taken_");
            t.IsTrue(
                annulled != std::string::npos &&
                    conditional.find(
                        "observeEeInstructionBeginPolicy<Mode>",
                        annulled) != std::string::npos &&
                    conditional.find(
                        "observeEeInstructionCompletePolicy<Mode>",
                        annulled) != std::string::npos,
                "a nullified likely delay load should still publish issue and completion categories");

            const uint32_t jumpRaw =
                (OPCODE_J << 26u) |
                (((kCodeAddress + 0x100u) >> 2u) &
                 0x03ffffffu);
            const std::string jump =
                generateFunction({jumpRaw, 0u});
            t.IsTrue(
                jump.find("observeEePredictedBranchPolicy<Mode>") !=
                    std::string::npos,
                "J should count as a predictive branch");

            const uint32_t jrRaw =
                (1u << 21u) | SPECIAL_JR;
            const std::string jr =
                generateFunction({jrRaw, 0u});
            t.IsFalse(
                jr.find("observeEePredictedBranchPolicy<Mode>") !=
                    std::string::npos,
                "JR should remain excluded from predictive branch events");
        });
    });
}
