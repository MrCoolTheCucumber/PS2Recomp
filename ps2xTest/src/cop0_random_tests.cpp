#include "MiniTest.h"

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ee_retirement_model.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
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

    uint32_t scalarAdvanceRandom(
        uint32_t random,
        uint32_t wired,
        uint32_t count)
    {
        constexpr uint32_t upperBound = 47u;
        const uint32_t lowerBound =
            std::min(wired & 0x3fu, upperBound);
        while (count-- != 0u)
        {
            random &= 0x3fu;
            random =
                random > upperBound || random <= lowerBound
                    ? upperBound
                    : random - 1u;
        }
        return random;
    }

    void batchedPublicationEntry(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        ctx->retireEeInstructions(5u);
    }
}

void register_cop0_random_tests()
{
    MiniTest::Case("COP0 Random", [](TestCase &tc)
    {
        tc.Run("retirement batching accepts only non-observing instructions", [](TestCase &t)
        {
            Instruction addiu = makeNop(0x0013F000u);
            t.IsTrue(
                eeInstructionCanBatchRetirement(addiu),
                "non-trapping scalar arithmetic should batch");

            Instruction addu{};
            addu.opcode = OPCODE_SPECIAL;
            addu.function = SPECIAL_ADDU;
            t.IsTrue(
                eeInstructionCanBatchRetirement(addu),
                "non-trapping register arithmetic should batch");

            Instruction addi{};
            addi.opcode = OPCODE_ADDI;
            t.IsFalse(
                eeInstructionCanBatchRetirement(addi),
                "overflow-capable arithmetic should remain a scalar boundary");

            Instruction load{};
            load.opcode = OPCODE_LW;
            load.isLoad = true;
            t.IsFalse(
                eeInstructionCanBatchRetirement(load),
                "faultable memory should remain a scalar boundary");
            t.IsTrue(
                eeInstructionCanDeferRetirementMaterialization(load),
                "dynamic memory can pre-retire before a guarded Fast access");

            Instruction mmioLoad = load;
            mmioLoad.isMmio = true;
            t.IsFalse(
                eeInstructionCanDeferRetirementMaterialization(mmioLoad),
                "statically known MMIO should materialize before its helper");

            Instruction vuLoad = load;
            vuLoad.isVU = true;
            t.IsFalse(
                eeInstructionCanDeferRetirementMaterialization(vuLoad),
                "VU memory should materialize before synchronization helpers");

            Instruction controlLoad = load;
            controlLoad.modificationInfo.modifiesControl = true;
            t.IsFalse(
                eeInstructionCanDeferRetirementMaterialization(controlLoad),
                "memory with control effects should retain a scalar boundary");

            Instruction cop0{};
            cop0.opcode = OPCODE_COP0;
            t.IsFalse(
                eeInstructionCanBatchRetirement(cop0),
                "COP0 observation should remain a scalar boundary");

            Instruction branch{};
            branch.opcode = OPCODE_BEQ;
            branch.isBranch = true;
            branch.hasDelaySlot = true;
            t.IsFalse(
                eeInstructionCanBatchRetirement(branch),
                "control flow should remain a scalar boundary");

            Instruction mmi{};
            mmi.opcode = OPCODE_MMI;
            mmi.isMMI = true;
            t.IsFalse(
                eeInstructionCanBatchRetirement(mmi),
                "unclassified multimedia operations should remain conservative");
        });

        tc.Run("Fast direct memory retains Random until a slow path", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the retirement fixture should initialize RDRAM");
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            R5900Context ctx{};
            ctx.cop0_random = 47u;
            ctx.cop0_wired = 0u;
            ctx.retireEeInstructions(3u);

            const uint32_t direct =
                Ps2EeMemoryAccessPolicy<
                    EeArchitecturalObservationMode::Fast>::read32(
                    rdram, &ctx, &runtime, 0x80001000u);
            (void)direct;
            t.Equals(
                ctx.cop0_random,
                47u,
                "a direct RDRAM access should not publish pending Random work");
            t.Equals(
                ctx.cop0_random_pending_retirements,
                uint64_t{3u},
                "a direct RDRAM access should retain the counted run");

            (void)Ps2EeMemoryAccessPolicy<
                EeArchitecturalObservationMode::Fast>::read32(
                rdram, &ctx, &runtime, 0xB000E010u);
            t.Equals(
                ctx.cop0_random,
                44u,
                "a slow MMIO path should materialize all prior retirements");
            t.Equals(
                ctx.cop0_random_pending_retirements,
                uint64_t{0u},
                "a slow MMIO path should publish no pending Random work");
        });

        tc.Run("generated fast arithmetic uses one retirement batch", [](TestCase &t)
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
                "the precise specialization should retain scalar retirement boundaries");
            t.Equals(
                countOccurrences(
                    generated,
                    "EeArchitecturalObservationMode::Precise) { ctx->beginEeInstruction(); }"),
                static_cast<size_t>(3u),
                "scalar arithmetic retirement should exist only in the precise specialization");
            t.Equals(
                countOccurrences(
                    generated,
                    "ctx->retireEeInstructions(3u);"),
                static_cast<size_t>(1u),
                "the fast specialization should retire the complete straight-line run once");
        });

        tc.Run("generated Random observers split fast retirement runs", [](TestCase &t)
        {
            Function function{};
            function.name = "cop0_random_barriers";
            function.start = 0x00141000u;
            function.end = 0x00141020u;
            function.isRecompiled = true;

            Instruction mfc0Random{};
            mfc0Random.address = 0x00141008u;
            mfc0Random.opcode = OPCODE_COP0;
            mfc0Random.rs = COP0_MF;
            mfc0Random.rt = 2u;
            mfc0Random.rd = COP0_REG_RANDOM;

            Instruction tlbwr{};
            tlbwr.address = 0x00141010u;
            tlbwr.opcode = OPCODE_COP0;
            tlbwr.rs = COP0_CO;
            tlbwr.function = COP0_CO_TLBWR;
            tlbwr.raw = COP0_CO_TLBWR;

            Instruction mtc0Wired{};
            mtc0Wired.address = 0x00141018u;
            mtc0Wired.opcode = OPCODE_COP0;
            mtc0Wired.rs = COP0_MT;
            mtc0Wired.rt = 3u;
            mtc0Wired.rd = COP0_REG_WIRED;

            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    function,
                    {
                        makeNop(0x00141000u),
                        makeNop(0x00141004u),
                        mfc0Random,
                        makeNop(0x0014100Cu),
                        tlbwr,
                        makeNop(0x00141014u),
                        mtc0Wired,
                        makeNop(0x0014101Cu),
                    },
                    false);

            const size_t firstBatch = generated.find(
                "ctx->retireEeInstructions(2u);");
            const size_t randomRead = generated.find(
                "(int32_t)ctx->cop0_random");
            const size_t tlbWrite = generated.find(
                "runtime->handleTLBWR(rdram, ctx);");
            const size_t wiredWrite = generated.find(
                "ctx->cop0_wired = GPR_U32(ctx, 3)");
            t.IsTrue(
                firstBatch != std::string::npos &&
                    randomRead != std::string::npos &&
                    firstBatch < randomRead,
                "MFC0 Random should observe all prior fast retirements");
            t.IsTrue(
                tlbWrite != std::string::npos &&
                    generated.rfind(
                        "ctx->retireEeInstructions(1u);",
                        tlbWrite) != std::string::npos,
                "TLBWR should split and follow a materializable fast run");
            t.IsTrue(
                wiredWrite != std::string::npos &&
                    generated.rfind(
                        "ctx->retireEeInstructions(1u);",
                        wiredWrite) != std::string::npos,
                "MTC0 Wired should split and follow a materializable fast run");
            t.Equals(
                countOccurrences(
                    generated,
                    "    ctx->beginEeInstruction();"),
                static_cast<size_t>(3u),
                "Random-observing COP0 instructions should retain scalar begin boundaries");
            t.Equals(
                countOccurrences(
                    generated,
                    "ctx->retireEeInstructions(1u);"),
                static_cast<size_t>(3u),
                "each straight-line singleton around Random barriers should retire once");
        });

        tc.Run("generated resume labels split fast retirement runs", [](TestCase &t)
        {
            Function function{};
            function.name = "cop0_random_resume_split";
            function.start = 0x00143000u;
            function.end = 0x00143010u;
            function.isRecompiled = true;

            Instruction branch{};
            branch.address = 0x00143008u;
            branch.opcode = OPCODE_BEQ;
            branch.rs = 2u;
            branch.rt = 3u;
            branch.simmediate =
                static_cast<uint32_t>(
                    static_cast<int32_t>(-2));
            branch.isBranch = true;
            branch.hasDelaySlot = true;

            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    function,
                    {
                        makeNop(0x00143000u),
                        makeNop(0x00143004u),
                        branch,
                        makeNop(0x0014300Cu),
                    },
                    false);

            const size_t firstBatch = generated.find(
                "ctx->retireEeInstructions(1u);");
            const size_t label = generated.find(
                "label_143004:");
            const size_t secondBatch = generated.find(
                "ctx->retireEeInstructions(1u);",
                firstBatch == std::string::npos
                    ? 0u
                    : firstBatch + 1u);
            t.IsTrue(
                firstBatch != std::string::npos &&
                    label != std::string::npos &&
                    secondBatch != std::string::npos &&
                    firstBatch < label && label < secondBatch,
                "normal fallthrough should flush before a label while direct entry skips the prior run");
            t.Equals(
                countOccurrences(
                    generated,
                    "ctx->retireEeInstructions(2u);"),
                static_cast<size_t>(0u),
                "a fast retirement run must never span a resumable label");
            t.Equals(
                countOccurrences(
                    generated,
                    "    ctx->beginEeInstruction();"),
                static_cast<size_t>(2u),
                "the branch and executed delay slot should retain scalar boundaries");
        });

        tc.Run("generated dynamic branch paths cannot inherit a fast retirement run", [](TestCase &t)
        {
            const auto generateBranch = [](uint32_t opcode,
                                           bool includeDelay)
            {
                Function function{};
                function.name = includeDelay
                                    ? "cop0_random_real_delay"
                                    : "cop0_random_synthetic_delay";
                function.start = 0x00144000u;
                function.end = includeDelay
                                   ? 0x00144010u
                                   : 0x0014400Cu;
                function.isRecompiled = true;

                Instruction branch{};
                branch.address = 0x00144008u;
                branch.opcode = opcode;
                branch.rs = 2u;
                branch.rt = 3u;
                branch.simmediate = 4u;
                branch.isBranch = true;
                branch.hasDelaySlot = true;

                std::vector<Instruction> instructions{
                    makeNop(0x00144000u),
                    makeNop(0x00144004u),
                    branch,
                };
                if (includeDelay)
                {
                    instructions.push_back(
                        makeNop(0x0014400Cu));
                }
                return CodeGenerator({}, {}).generateFunction(
                    function, instructions, false);
            };

            const std::string ordinary =
                generateBranch(OPCODE_BEQ, true);
            const std::string likely =
                generateBranch(OPCODE_BEQL, true);
            const std::string synthetic =
                generateBranch(OPCODE_BEQ, false);
            for (const std::string *generated :
                 {&ordinary, &likely, &synthetic})
            {
                t.Equals(
                    countOccurrences(
                        *generated,
                        "ctx->retireEeInstructions(2u);"),
                    static_cast<size_t>(1u),
                    "a safe prefix should flush before dynamic branch execution");
                t.Equals(
                    countOccurrences(
                        *generated,
                        "    ctx->beginEeInstruction();"),
                    static_cast<size_t>(2u),
                    "the branch and only its executed real or synthetic slot should retire scalarly");
            }
            t.IsTrue(
                likely.find("if (branch_taken_0x144008)") !=
                    std::string::npos,
                "a likely delay slot should remain confined to the taken path");
        });

        tc.Run("TLBWR consumes Random without advancing it itself", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            for (uint32_t index = 0u;
                 index < runtime.memory().tlbEntryCount();
                 ++index)
            {
                t.IsTrue(
                    runtime.memory().tlbWrite(
                        index, EeTlbEntry{}),
                    "the TLBWR fixture should clear every TLB entry");
            }
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

            EeTlbEntry entry{};
            t.IsTrue(
                runtime.memory().tlbRead(
                    19u, entry),
                "the current Random slot should be readable");
            t.Equals(
                entry.entryHi & 0xffffe000u,
                0x23456000u,
                "repeated TLBWR effects should consume the same current Random value");
            t.Equals(
                (entry.entryLo0 >> 6u) &
                    0x000fffffu,
                0x23456u,
                "TLBWR should write the payload to the current Random slot");
            t.IsTrue(
                (entry.entryLo0 & 0x2u) != 0u,
                "TLBWR should preserve the entry validity payload");
            t.Equals(
                ctx.cop0_random,
                19u,
                "TLBWR should leave automatic Random retirement to the instruction boundary");

            R5900Context observed{};
            observed.cop0_random = 19u;
            observed.cop0_wired = 5u;
            observed.cop0_entryhi = 0x34567000u;
            observed.cop0_entrylo0 =
                (0x34567u << 6u) | 0x2u;
            observed.retireEeInstructions(2u);
            observed.beginEeInstruction();
            runtime.handleTLBWR(rdram, &observed);

            EeTlbEntry observedEntry{};
            t.IsTrue(
                runtime.memory().tlbRead(
                    17u, observedEntry),
                "TLBWR should consume Random after materializing its completed prefix");
            t.Equals(
                observedEntry.entryHi & 0xffffe000u,
                0x34566000u,
                "TLBWR should use the prior-retirement Random slot before its own decrement");
            observed.finishEeInstruction();
            t.Equals(
                observed.cop0_random,
                16u,
                "TLBWR itself should decrement Random only at its retirement boundary");
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

            R5900Context generatedCtx{};
            generatedCtx.cop0_wired = 10u;
            generatedCtx.cop0_random = 12u;
            generatedCtx.beginEeInstruction();
            runtime.serviceEeEventsAtGeneratedBlockBoundary(
                nullptr, &generatedCtx);
            t.Equals(
                generatedCtx.cop0_random,
                12u,
                "a generated no-event boundary should defer Random materialization");
            t.Equals(
                generatedCtx.cop0_random_pending_retirements,
                1u,
                "a generated no-event boundary should retain counted retirement work");

            runtime.writeCop0Count(
                nullptr, &generatedCtx, 10u);
            runtime.writeCop0Compare(
                nullptr, &generatedCtx, 13u);
            generatedCtx.retireEeInstructions(3u);
            generatedCtx.advanceEeCycleTicks(3u * 8u);
            runtime.serviceEeEventsAtGeneratedBlockBoundary(
                nullptr, &generatedCtx);
            t.Equals(
                generatedCtx.cop0_random,
                46u,
                "a due generated event should materialize all pending retirements first");
            t.Equals(
                generatedCtx.cop0_random_pending_retirements,
                0u,
                "a due generated event should publish no stale Random work");
            t.IsTrue(
                (generatedCtx.cop0_cause & 0x00008000u) != 0u,
                "the generated boundary should still service the due COP0 timer");

            R5900Context checkpointCtx{};
            checkpointCtx.cop0_wired = 10u;
            checkpointCtx.cop0_random = 12u;
            checkpointCtx.retireEeInstructions(3u);
            runtime.requestGuestPreemption();
            t.Equals(
                runtime.checkpointGuestExecution(&checkpointCtx),
                PS2GuestCheckpointResult::ExitToDispatcher,
                "an explicit checkpoint should unwind generated execution");
            t.Equals(
                checkpointCtx.cop0_random,
                47u,
                "a real checkpoint should materialize pending Random work");
            t.Equals(
                checkpointCtx.cop0_random_pending_retirements,
                0u,
                "a real checkpoint should publish no pending retirements");
            t.IsFalse(
                runtime.guestPreemptionRequestedForTesting(),
                "a checkpoint should consume its explicit request");

            runtime.requestGuestPreemption();
            runtime.requestGuestPreemption();
            t.IsTrue(
                runtime.guestPreemptionRequestedForTesting(),
                "multiple producers should coalesce into pending work");
            t.Equals(
                runtime.checkpointGuestExecution(nullptr),
                PS2GuestCheckpointResult::ExitToDispatcher,
                "one checkpoint should consume coalesced requests");
            t.IsFalse(
                runtime.guestPreemptionRequestedForTesting(),
                "coalesced requests should be fully consumed");

            runtime.requestGuestPreemption();
            t.IsTrue(
                runtime.guestPreemptionRequestedForTesting(),
                "a later request should remain independently visible");
            t.Equals(
                runtime.checkpointGuestExecution(nullptr),
                PS2GuestCheckpointResult::ExitToDispatcher,
                "a later request should trigger another checkpoint");
            t.IsFalse(
                runtime.guestPreemptionRequestedForTesting(),
                "the later request should be consumed exactly once");

            R5900Context continueCtx{};
            continueCtx.cop0_wired = 10u;
            continueCtx.cop0_random = 12u;
            bool observedContinue = false;
            for (uint32_t attempt = 0u;
                 attempt < 2u && !observedContinue;
                 ++attempt)
            {
                continueCtx.retireEeInstructions(1u);
                const uint32_t randomBefore =
                    continueCtx.cop0_random;
                if (runtime.checkpointGuestExecution(
                        &continueCtx) ==
                    PS2GuestCheckpointResult::Continue)
                {
                    observedContinue = true;
                    t.Equals(
                        continueCtx.cop0_random,
                        randomBefore,
                        "an ordinary checkpoint probe should not materialize Random");
                    t.Equals(
                        continueCtx.cop0_random_pending_retirements,
                        1u,
                        "an ordinary checkpoint probe should retain pending work");
                }
            }
            t.IsTrue(
                observedContinue,
                "a bounded checkpoint probe should observe the common continue path");

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

        tc.Run("counted Random advancement matches scalar retirement", [](TestCase &t)
        {
            bool matched = true;
            uint32_t failedRandom = 0u;
            uint32_t failedWired = 0u;
            uint32_t failedCount = 0u;
            uint32_t expected = 0u;
            uint32_t actual = 0u;

            for (uint32_t wired = 0u;
                 wired < 64u && matched; ++wired)
            {
                for (uint32_t random = 0u;
                     random < 64u && matched; ++random)
                {
                    for (uint32_t count = 0u;
                         count <= 192u; ++count)
                    {
                        R5900Context ctx{};
                        ctx.cop0_random = random;
                        ctx.cop0_wired = wired;
                        ctx.advanceCop0Random(count);
                        expected = scalarAdvanceRandom(
                            random, wired, count);
                        actual = ctx.cop0_random;
                        if (actual != expected)
                        {
                            matched = false;
                            failedRandom = random;
                            failedWired = wired;
                            failedCount = count;
                            break;
                        }
                    }
                }
            }

            std::ostringstream message;
            message << "counted advance should match scalar Random for "
                    << "random=" << failedRandom
                    << " wired=" << failedWired
                    << " count=" << failedCount
                    << " expected=" << expected
                    << " actual=" << actual;
            t.IsTrue(matched, message.str());

            struct LargeCase
            {
                uint32_t random;
                uint32_t wired;
                uint64_t count;
                uint32_t expected;
            };
            constexpr std::array<LargeCase, 7u> largeCases{{
                {47u, 0u, std::numeric_limits<uint64_t>::max(), 32u},
                {47u, 5u, std::numeric_limits<uint64_t>::max(), 7u},
                {63u, 5u, std::numeric_limits<uint64_t>::max(), 8u},
                {64u, 0u, std::numeric_limits<uint64_t>::max(), 33u},
                {12u, 10u, 1'000'000'000'000'000'000ull, 30u},
                {3u, 31u, 1'000'000'000'000'000'000ull, 33u},
                {47u, 47u, std::numeric_limits<uint64_t>::max(), 47u},
            }};
            for (const LargeCase &test : largeCases)
            {
                R5900Context ctx{};
                ctx.cop0_random = test.random;
                ctx.cop0_wired = test.wired;
                ctx.advanceCop0Random(test.count);
                t.Equals(
                    ctx.cop0_random,
                    test.expected,
                    "large counted Random advance should remain O(1) and exact");
            }
        });

        tc.Run("batched retirements defer and materialize coherently", [](TestCase &t)
        {
            R5900Context ctx{};
            ctx.cop0_random = 47u;
            ctx.cop0_wired = 5u;

            ctx.retireEeInstructions(0u);
            ctx.retireEeInstructions(3u);
            ctx.retireEeInstructions(4u);
            t.Equals(
                ctx.insn_count,
                7u,
                "batched retirement should publish instruction count immediately");
            t.Equals(
                ctx.cop0_random,
                47u,
                "batched retirement should defer Random until a materialization boundary");
            t.Equals(
                ctx.cop0_random_pending_retirements,
                7u,
                "batched retirement should retain the complete pending count");

            ctx.finishEeInstruction();
            t.Equals(
                ctx.cop0_random,
                40u,
                "one materialization should apply the complete retirement batch");
            t.Equals(
                ctx.cop0_random_pending_retirements,
                0u,
                "materialization should drain the pending retirement count");

            ctx.beginEeInstruction();
            t.Equals(
                ctx.cop0_random,
                40u,
                "the current scalar instruction should observe Random before its own retirement");
            t.Equals(
                ctx.insn_count,
                8u,
                "scalar and batch instruction accounting should compose");
            ctx.beginEeInstruction();
            t.Equals(
                ctx.cop0_random,
                39u,
                "the next scalar instruction should materialize its predecessor");
            ctx.finishEeInstruction();
            t.Equals(
                ctx.cop0_random,
                38u,
                "the scalar tail should materialize at the external boundary");

            R5900Context large{};
            large.cop0_random = 47u;
            large.retireEeInstructions(
                std::numeric_limits<uint64_t>::max());
            t.Equals(
                large.cop0_random_pending_retirements,
                std::numeric_limits<uint64_t>::max(),
                "the pending representation should retain a maximum-size batch");
            large.finishEeInstruction();
            t.Equals(
                large.cop0_random,
                32u,
                "a maximum-size pending batch should materialize exactly");

            R5900Context overflow{};
            overflow.cop0_random = 47u;
            overflow.retireEeInstructions(
                std::numeric_limits<uint64_t>::max());
            overflow.retireEeInstructions(1u);
            t.Equals(
                overflow.cop0_random,
                32u,
                "pending-count overflow should first materialize the complete prefix");
            t.Equals(
                overflow.cop0_random_pending_retirements,
                1u,
                "the post-overflow instruction should remain pending independently");
            overflow.finishEeInstruction();
            t.Equals(
                overflow.cop0_random,
                31u,
                "materializing after pending-count overflow should retain exact composition");

            R5900Context reset{};
            reset.cop0_random = 47u;
            reset.retireEeInstructions(3u);
            reset.resetEeBlockTiming();
            t.Equals(
                reset.cop0_random,
                44u,
                "resetting local block state should publish completed Random retirements");
            t.Equals(
                reset.cop0_random_pending_retirements,
                0u,
                "resetting local block state should leave no unpublished retirement count");

            R5900Context wired{};
            wired.cop0_random = 47u;
            wired.retireEeInstructions(2u);
            wired.beginEeInstruction();
            t.Equals(
                wired.cop0_random,
                45u,
                "MTC0 Wired should first observe all prior retirements");
            wired.cop0_wired = 47u;
            wired.finishEeInstruction();
            t.Equals(
                wired.cop0_random,
                47u,
                "the MTC0 Wired instruction should retire using its newly written interval");
        });

        tc.Run("guest context handoff materializes completed batches", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context first{};
            R5900Context second{};
            first.cop0_random = 47u;
            first.cop0_wired = 5u;
            second.cop0_random = 30u;
            second.cop0_wired = 5u;

            {
                PS2Runtime::GuestExecutionScope firstScope(
                    &runtime, &first);
                first.retireEeInstructions(3u);
                {
                    PS2Runtime::GuestExecutionScope secondScope(
                        &runtime, &second);
                    t.Equals(
                        first.cop0_random,
                        44u,
                        "switching away should publish the previous context's Random state");
                    t.Equals(
                        first.cop0_random_pending_retirements,
                        0u,
                        "a switched-out context should retain no pending retirement count");
                    second.retireEeInstructions(2u);
                }
                t.Equals(
                    second.cop0_random,
                    28u,
                    "restoring the prior context should first publish the nested context");
                t.Equals(
                    second.cop0_random_pending_retirements,
                    0u,
                    "a nested context handoff should drain its retirement count");
            }
        });

        tc.Run("exceptions and outer dispatch materialize completed batches", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the publication fixture should initialize RDRAM");

            R5900Context published{};
            published.pc = 0x80001000u;
            // Outer dispatch canonicalizes KSEG0 to kuseg before fetch
            // validation. ERL keeps that canonical low address on the
            // architecturally unmapped path for this synthetic fixture.
            published.cop0_status = 0x00000004u;
            published.cop0_random = 47u;
            runtime.executeGuestStep(
                runtime.memory().getRDRAM(),
                &published,
                batchedPublicationEntry);
            t.Equals(
                published.insn_count,
                5u,
                "outer dispatch should publish the complete batched instruction count");
            t.Equals(
                published.cop0_random,
                42u,
                "outer dispatch should materialize Random before debugger publication");
            t.Equals(
                published.cop0_random_pending_retirements,
                0u,
                "outer dispatch should expose no unpublished Random work");

            R5900Context fault{};
            fault.pc = 0x00142000u;
            fault.cop0_random = 47u;
            fault.retireEeInstructions(3u);
            fault.beginEeInstruction();
            bool raised = false;
            try
            {
                runtime.SignalException(
                    &fault,
                    EXCEPTION_RESERVED_INSTRUCTION);
            }
            catch (const PS2GuestException &)
            {
                raised = true;
            }
            t.IsTrue(
                raised,
                "the faulting instruction should enter the guest exception path");
            t.Equals(
                fault.insn_count,
                4u,
                "the completed prefix and faulting instruction should both count");
            t.Equals(
                fault.cop0_random,
                43u,
                "exception delivery should materialize the prefix and faulting instruction");
            t.Equals(
                fault.cop0_random_pending_retirements,
                0u,
                "exception publication should drain Random retirement state");
        });
    });
}
