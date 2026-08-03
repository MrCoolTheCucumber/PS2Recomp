#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
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
    constexpr uint32_t kCauseBd = 0x80000000u;
    constexpr uint32_t kCauseCeMask = 0x30000000u;
    constexpr uint32_t kCauseExcCodeMask = 0x0000007Cu;
    constexpr uint32_t kStatusExl = 0x00000002u;
    constexpr uint32_t kStatusErl = 0x00000004u;
    constexpr uint32_t kStatusSupervisor = 0x00000008u;
    constexpr uint32_t kStatusUser = 0x00000010u;
    constexpr uint32_t kStatusCu0 = 0x10000000u;
    constexpr uint32_t kStatusCu1 = 0x20000000u;
    constexpr uint32_t kStatusCu2 = 0x40000000u;
    constexpr uint32_t kGeneralExceptionVector = 0x80000180u;
    constexpr uint32_t kCoprocessorUnusable = 11u;

    template <typename Runtime>
    constexpr bool hasCoprocessorUsabilityCheck =
        requires(Runtime &runtime, R5900Context *ctx, uint32_t coprocessor)
        {
            runtime.RequireCoprocessorUsable(ctx, coprocessor);
        };

    template <typename Runtime, typename Operation>
    void executeCoprocessorOperation(
        Runtime &runtime,
        R5900Context *ctx,
        uint32_t coprocessor,
        Operation &&operation)
    {
        if constexpr (hasCoprocessorUsabilityCheck<Runtime>)
        {
            runtime.RequireCoprocessorUsable(ctx, coprocessor);
        }
        operation();
    }

    Instruction decode(uint32_t address, uint32_t raw)
    {
        return R5900Decoder{}.decodeInstruction(address, raw, false);
    }

    Function makeFunction(
        std::string name,
        uint32_t start,
        uint32_t instructionCount)
    {
        Function function{};
        function.name = std::move(name);
        function.start = start;
        function.end = start + instructionCount * 4u;
        function.isRecompiled = true;
        return function;
    }

    void expectOrdered(
        TestCase &t,
        const std::string &code,
        std::string_view first,
        std::string_view second,
        const std::string &prefix)
    {
        const size_t firstPosition = code.find(first);
        const size_t secondPosition = code.find(second);
        t.IsTrue(
            firstPosition != std::string::npos,
            prefix + "should emit " + std::string(first));
        t.IsTrue(
            secondPosition != std::string::npos,
            prefix + "should emit " + std::string(second));
        t.IsTrue(
            firstPosition != std::string::npos &&
                secondPosition != std::string::npos &&
                firstPosition < secondPosition,
            prefix + "must check usability before its first side effect");
    }
}

void register_ee_coprocessor_unusable_tests()
{
    MiniTest::Case("EE coprocessor usability", [](TestCase &tc)
    {
        tc.Run("disabled units raise precise CpU before side effects", [](TestCase &t)
        {
            t.IsTrue(
                hasCoprocessorUsabilityCheck<PS2Runtime>,
                "the runtime should expose one shared coprocessor-usability boundary");

            for (uint32_t coprocessor = 0u; coprocessor <= 2u; ++coprocessor)
            {
                for (bool inDelaySlot : {false, true})
                {
                    PS2Runtime runtime;
                    R5900Context ctx{};
                    ctx.pc = 0x00134004u + coprocessor * 0x100u;
                    ctx.branch_pc = ctx.pc - 4u;
                    ctx.in_delay_slot = inDelaySlot;
                    ctx.cop0_status =
                        coprocessor == 0u ? kStatusUser : 0u;
                    ctx.cop0_cause = kCauseCeMask | 0x00000100u;
                    uint32_t sideEffect = 0x11223344u;

                    bool raised = false;
                    try
                    {
                        executeCoprocessorOperation(
                            runtime,
                            &ctx,
                            coprocessor,
                            [&]() { sideEffect = 0xAABBCCDDu; });
                    }
                    catch (const PS2GuestException &)
                    {
                        raised = true;
                    }

                    const std::string prefix =
                        "COP" + std::to_string(coprocessor) +
                        (inDelaySlot ? " delay slot: " : " direct: ");
                    t.IsTrue(raised, prefix + "disabled access should unwind");
                    t.Equals(
                        sideEffect,
                        0x11223344u,
                        prefix + "the coprocessor operation must be cancelled");
                    t.Equals(
                        ctx.cop0_cause & kCauseExcCodeMask,
                        (kCoprocessorUnusable << 2u) & kCauseExcCodeMask,
                        prefix + "Cause.ExcCode should be CpU");
                    t.Equals(
                        ctx.cop0_cause & kCauseCeMask,
                        (coprocessor << 28u) & kCauseCeMask,
                        prefix + "Cause.CE should replace the prior coprocessor number");
                    t.Equals(
                        ctx.cop0_epc,
                        inDelaySlot ? ctx.branch_pc :
                                      0x00134004u + coprocessor * 0x100u,
                        prefix + "EPC should identify the restart instruction");
                    t.Equals(
                        (ctx.cop0_cause & kCauseBd) != 0u,
                        inDelaySlot,
                        prefix + "Cause.BD should identify delay-slot execution");
                    t.Equals(
                        ctx.pc,
                        kGeneralExceptionVector,
                        prefix + "CpU should enter V_COMMON");
                    t.IsTrue(
                        (ctx.cop0_status & kStatusExl) != 0u,
                        prefix + "CpU should enter kernel exception mode");
                }
            }
        });

        tc.Run("COP0 privilege and CU enabled controls execute", [](TestCase &t)
        {
            struct Control
            {
                const char *name;
                uint32_t coprocessor;
                uint32_t status;
            };
            const std::array controls = {
                Control{"COP0 kernel KSU", 0u, 0u},
                Control{"COP0 EXL kernel", 0u, kStatusUser | kStatusExl},
                Control{"COP0 ERL kernel", 0u, kStatusUser | kStatusErl},
                Control{"COP0 user CU0", 0u, kStatusUser | kStatusCu0},
                Control{"COP1 CU1", 1u, kStatusCu1},
                Control{"COP2 CU2", 2u, kStatusCu2},
            };

            t.IsTrue(
                hasCoprocessorUsabilityCheck<PS2Runtime>,
                "enabled controls should use the shared usability boundary");
            for (const Control &control : controls)
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.pc = 0x00135000u;
                ctx.cop0_status = control.status;
                bool executed = false;
                bool raised = false;
                try
                {
                    executeCoprocessorOperation(
                        runtime,
                        &ctx,
                        control.coprocessor,
                        [&]() { executed = true; });
                }
                catch (const PS2GuestException &)
                {
                    raised = true;
                }

                t.IsFalse(
                    raised,
                    std::string(control.name) + " should remain usable");
                t.IsTrue(
                    executed,
                    std::string(control.name) + " should execute the operation");
                t.Equals(
                    ctx.pc,
                    0x00135000u,
                    std::string(control.name) + " should not redirect control flow");
            }

            for (const uint32_t status : {kStatusSupervisor, kStatusUser})
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.pc = 0x00135100u;
                ctx.cop0_status = status;
                bool executed = false;
                bool raised = false;
                try
                {
                    executeCoprocessorOperation(
                        runtime,
                        &ctx,
                        0u,
                        [&]() { executed = true; });
                }
                catch (const PS2GuestException &)
                {
                    raised = true;
                }
                t.IsTrue(raised, "COP0 with CU0 clear should fail outside kernel mode");
                t.IsFalse(executed, "a disabled COP0 operation should be cancelled");
            }
        });

        tc.Run("nested CpU replaces CE while preserving the restart context", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00136004u;
            ctx.branch_pc = 0x00136000u;
            ctx.in_delay_slot = true;
            ctx.cop0_status = kStatusExl;
            ctx.cop0_epc = 0x00123450u;
            ctx.cop0_cause = kCauseBd | (1u << 28u) | 0x00000200u;

            bool raised = false;
            try
            {
                executeCoprocessorOperation(
                    runtime,
                    &ctx,
                    2u,
                    []() {});
            }
            catch (const PS2GuestException &)
            {
                raised = true;
            }

            t.IsTrue(raised, "disabled COP2 should fault inside a level-1 handler");
            t.Equals(
                ctx.cop0_cause & kCauseExcCodeMask,
                (kCoprocessorUnusable << 2u) & kCauseExcCodeMask,
                "nested Cause.ExcCode should become CpU");
            t.Equals(
                ctx.cop0_cause & kCauseCeMask,
                2u << 28u,
                "nested CpU should replace Cause.CE");
            t.Equals(
                ctx.cop0_epc,
                0x00123450u,
                "nested CpU should preserve the original EPC");
            t.IsTrue(
                (ctx.cop0_cause & kCauseBd) != 0u,
                "nested CpU should preserve the original Cause.BD");
        });

        tc.Run("all translated coprocessor families guard their first effect", [](TestCase &t)
        {
            struct Encoding
            {
                const char *name;
                uint32_t raw;
                uint32_t coprocessor;
                const char *firstEffect;
            };
            const std::array encodings = {
                Encoding{"MFC0", (OPCODE_COP0 << 26u) | (COP0_MF << 21u) |
                                     (2u << 16u) | (COP0_REG_STATUS << 11u),
                         0u, "SET_GPR"},
                Encoding{"MTC0", (OPCODE_COP0 << 26u) | (COP0_MT << 21u) |
                                     (2u << 16u) | (COP0_REG_STATUS << 11u),
                         0u, "writeCop0Status"},
                Encoding{"TLBWI", (OPCODE_COP0 << 26u) | (COP0_CO << 21u) |
                                      COP0_CO_TLBWI,
                         0u, "handleTLBWI"},
                Encoding{"ERET", (OPCODE_COP0 << 26u) | (COP0_CO << 21u) |
                                     COP0_CO_ERET,
                         0u, "handleCop0Eret"},
                Encoding{"CACHE", (OPCODE_CACHE << 26u) | (2u << 21u) |
                                      (0x1Au << 16u),
                         0u, "handleEeCacheOperation"},
                Encoding{"MFC1", (OPCODE_COP1 << 26u) | (COP1_MF << 21u) |
                                     (2u << 16u) | (3u << 11u),
                         1u, "SET_GPR"},
                Encoding{"ADD.S", (OPCODE_COP1 << 26u) | (COP1_S << 21u) |
                                      (4u << 16u) | (3u << 11u) | (2u << 6u) |
                                      COP1_S_ADD,
                         1u, "ctx->f["},
                Encoding{"QMFC2", (OPCODE_COP2 << 26u) | (COP2_QMFC2 << 21u) |
                                      (2u << 16u) | (3u << 11u),
                         2u, "synchronizeVU0Microprogram"},
                Encoding{"VADD", (OPCODE_COP2 << 26u) | (COP2_CO << 21u) |
                                     (4u << 16u) | (3u << 11u) | (2u << 6u) |
                                     VU0_S1_VADD,
                         2u, "synchronizeVU0Microprogram"},
                Encoding{"LWC1", (OPCODE_LWC1 << 26u) | (3u << 21u) |
                                     (4u << 16u),
                         1u, "READ32("},
                Encoding{"SWC1", (OPCODE_SWC1 << 26u) | (3u << 21u) |
                                     (4u << 16u),
                         1u, "WRITE32("},
                Encoding{"LQC2", (OPCODE_LQC2 << 26u) | (3u << 21u) |
                                     (4u << 16u),
                         2u, "synchronizeVU0Microprogram"},
                Encoding{"SQC2", (OPCODE_SQC2 << 26u) | (3u << 21u) |
                                     (4u << 16u),
                         2u, "synchronizeVU0Microprogram"},
            };

            CodeGenerator generator({}, {});
            uint32_t address = 0x00137000u;
            for (const Encoding &encoding : encodings)
            {
                const Instruction instruction =
                    decode(address, encoding.raw);
                const std::string code =
                    generator.translateInstruction(instruction);
                const std::string guard =
                    "RequireCoprocessorUsable(ctx, " +
                    std::to_string(encoding.coprocessor) + "u)";
                expectOrdered(
                    t,
                    code,
                    guard,
                    encoding.firstEffect,
                    std::string(encoding.name) + ": ");
                address += 4u;
            }
        });

        tc.Run("coprocessor branches guard conditions and delay slots", [](TestCase &t)
        {
            struct Branch
            {
                const char *name;
                uint32_t opcode;
                uint32_t format;
                uint32_t coprocessor;
                const char *condition;
            };
            const std::array branches = {
                Branch{"BC0T", OPCODE_COP0, COP0_BC, 0u, "readCop0Condition0"},
                Branch{"BC1T", OPCODE_COP1, COP1_BC, 1u, "ctx->fcr31"},
                Branch{"BC2T", OPCODE_COP2, COP2_BC, 2u, "readCop2Condition"},
            };

            uint32_t start = 0x00138000u;
            for (const Branch &branch : branches)
            {
                const Instruction owner = decode(
                    start,
                    (branch.opcode << 26u) | (branch.format << 21u) |
                        (1u << 16u) | 1u);
                const Instruction delay = decode(
                    start + 4u,
                    (OPCODE_ADDIU << 26u) | (2u << 21u) |
                        (2u << 16u) | 1u);
                const std::string generated =
                    CodeGenerator({}, {}).generateFunction(
                        makeFunction(branch.name, start, 2u),
                        {owner, delay},
                        false);
                const std::string guard =
                    "RequireCoprocessorUsable(ctx, " +
                    std::to_string(branch.coprocessor) + "u)";
                expectOrdered(
                    t,
                    generated,
                    guard,
                    branch.condition,
                    std::string(branch.name) + " condition: ");
                expectOrdered(
                    t,
                    generated,
                    guard,
                    "SET_GPR_S32(ctx, 2",
                    std::string(branch.name) + " delay slot: ");
                start += 0x100u;
            }
        });

        tc.Run("coprocessor delay-slot operations check after BD ownership", [](TestCase &t)
        {
            struct DelayOperation
            {
                const char *name;
                uint32_t raw;
                uint32_t coprocessor;
                const char *firstEffect;
            };
            const std::array operations = {
                DelayOperation{"MTC0 delay", (OPCODE_COP0 << 26u) |
                                                   (COP0_MT << 21u) |
                                                   (2u << 16u) |
                                                   (COP0_REG_STATUS << 11u),
                               0u, "writeCop0Status"},
                DelayOperation{"MTC1 delay", (OPCODE_COP1 << 26u) |
                                                   (COP1_MT << 21u) |
                                                   (2u << 16u) | (3u << 11u),
                               1u, "ctx->f["},
                DelayOperation{"QMTC2 delay", (OPCODE_COP2 << 26u) |
                                                    (COP2_QMTC2 << 21u) |
                                                    (2u << 16u) | (3u << 11u),
                               2u, "synchronizeVU0Microprogram"},
            };

            uint32_t start = 0x00139000u;
            for (const DelayOperation &operation : operations)
            {
                const Instruction owner = decode(
                    start,
                    (OPCODE_BEQ << 26u) | (1u << 21u) |
                        (1u << 16u) | 1u);
                const Instruction delay = decode(start + 4u, operation.raw);
                const std::string generated =
                    CodeGenerator({}, {}).generateFunction(
                        makeFunction(operation.name, start, 2u),
                        {owner, delay},
                        false);
                const std::string guard =
                    "RequireCoprocessorUsable(ctx, " +
                    std::to_string(operation.coprocessor) + "u)";

                expectOrdered(
                    t,
                    generated,
                    "ctx->in_delay_slot = true",
                    guard,
                    std::string(operation.name) + " ownership: ");
                expectOrdered(
                    t,
                    generated,
                    guard,
                    operation.firstEffect,
                    std::string(operation.name) + " effect: ");
                start += 0x100u;
            }
        });
    });
}
