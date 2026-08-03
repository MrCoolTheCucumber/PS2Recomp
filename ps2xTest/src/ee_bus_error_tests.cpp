#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kCauseExcCodeMask = 0x0000007Cu;
    constexpr uint32_t kCauseBd = 0x80000000u;
    constexpr uint32_t kStatusExl = 0x00000002u;
    constexpr uint32_t kStatusErl = 0x00000004u;
    constexpr uint32_t kStatusBem = 0x00001000u;
    constexpr uint32_t kGeneralExceptionVector = 0x80000180u;

    bool raisesGuestException(const auto &operation)
    {
        try
        {
            operation();
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }

    void expectPhysicalBusFault(
        TestCase &t,
        const auto &operation,
        uint32_t expectedPhysicalAddress,
        const char *prefix)
    {
        bool busFault = false;
        bool otherFault = false;
        try
        {
            operation();
        }
        catch (const PS2BusErrorException &fault)
        {
            busFault = true;
            t.Equals(
                fault.physicalAddress(),
                expectedPhysicalAddress,
                std::string(prefix) + "fault should retain the translated physical address");
        }
        catch (...)
        {
            otherFault = true;
        }
        t.IsTrue(
            busFault,
            std::string(prefix) + "access should raise a typed physical-bus fault");
        t.IsFalse(
            otherFault,
            std::string(prefix) + "access should not use an unrelated host fault");
    }

    void expectDeliveredBusError(
        TestCase &t,
        const R5900Context &ctx,
        uint32_t exceptionCode,
        uint32_t expectedBadPAddr,
        uint32_t originalBadVAddr,
        const char *prefix)
    {
        t.Equals(
            ctx.cop0_cause & kCauseExcCodeMask,
            (exceptionCode << 2u) & kCauseExcCodeMask,
            std::string(prefix) + "Cause.ExcCode should identify the bus error");
        t.Equals(
            ctx.pc,
            kGeneralExceptionVector,
            std::string(prefix) + "bus error should enter V_COMMON");
        t.Equals(
            ctx.cop0_badpaddr,
            expectedBadPAddr,
            std::string(prefix) + "BadPAddr should record the physical 16-byte block");
        t.Equals(
            ctx.cop0_badvaddr,
            originalBadVAddr,
            std::string(prefix) + "bus error must not alter BadVAddr");
        t.IsTrue(
            (ctx.cop0_status & kStatusBem) != 0u,
            std::string(prefix) + "delivered bus error should set Status.BEM");
    }
}

void register_ee_bus_error_tests()
{
    MiniTest::Case("EE bus errors", [](TestCase &tc)
    {
        tc.Run("instruction fetch publishes IBE state", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00124000u;
            ctx.cop0_badvaddr = 0xCAFEBABEu;
            ctx.cop0_badpaddr = 0x11111110u;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalBusError(
                    &ctx,
                    PS2BusErrorAccess::InstructionFetch,
                    0x22001234u);
            });

            t.IsTrue(raised, "instruction bus error should unwind guest execution");
            t.Equals(
                ctx.cop0_epc,
                0x00124000u,
                "instruction fetch: EPC should identify the executing instruction");
            t.IsFalse(
                (ctx.cop0_cause & kCauseBd) != 0u,
                "instruction fetch: direct fault should leave Cause.BD clear");
            expectDeliveredBusError(
                t,
                ctx,
                EXCEPTION_BUS_ERROR_INSTRUCTION,
                0x22001230u,
                0xCAFEBABEu,
                "instruction fetch: ");
        });

        tc.Run("generated data accesses retire before possible delivery", [](TestCase &t)
        {
            constexpr uint32_t start = 0x00124800u;
            R5900Decoder decoder;
            const std::vector<Instruction> instructions = {
                decoder.decodeInstruction(
                    start,
                    (OPCODE_LW << 26u) | (1u << 21u) | (2u << 16u),
                    false),
                decoder.decodeInstruction(
                    start + 4u,
                    (OPCODE_SW << 26u) | (3u << 21u) | (4u << 16u),
                    false),
            };
            Function function{};
            function.name = "bus_error_retirement";
            function.start = start;
            function.end = start + 8u;
            function.isRecompiled = true;

            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    function, instructions, false);
            const size_t loadRetirement = generated.find(
                "ctx->beginEeInstruction();");
            const size_t load = generated.find("READ32(", loadRetirement);
            const size_t storeRetirement = generated.find(
                "ctx->beginEeInstruction();",
                loadRetirement + 1u);
            const size_t store = generated.find("WRITE32(", storeRetirement);

            t.IsTrue(
                loadRetirement != std::string::npos &&
                    load != std::string::npos &&
                    loadRetirement < load,
                "a bus-faultable load should retire before its memory boundary");
            t.IsTrue(
                storeRetirement != std::string::npos &&
                    store != std::string::npos &&
                    storeRetirement < store,
                "a bus-faultable store should retire before its memory boundary");
            t.IsTrue(
                load != std::string::npos &&
                    storeRetirement != std::string::npos &&
                    load < storeRetirement,
                "the two data instructions should retain program order");
        });

        tc.Run("CACHE fills route physical faults through DBE", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            ctx.pc = 0x00124900u;
            ctx.cop0_badvaddr = 0x0BADCAFEu;

            const bool raised = raisesGuestException([&]()
            {
                runtime.handleEeCacheOperation(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    0x0Eu,
                    0x22000000u);
            });

            t.IsTrue(raised, "an instruction-cache fill bus fault should unwind guest execution");
            expectDeliveredBusError(
                t,
                ctx,
                EXCEPTION_BUS_ERROR_DATA,
                0x22000000u,
                0x0BADCAFEu,
                "CACHE fill: ");
        });

        tc.Run("unmapped physical load publishes DBE state", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            ctx.pc = 0x00125000u;
            ctx.cop0_badvaddr = 0xA5A5A5A5u;
            ctx.cop0_badpaddr = 0x22222220u;

            const bool raised = raisesGuestException([&]()
            {
                (void)runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    0x22001234u);
            });

            t.IsTrue(raised, "physical-bus load failure should unwind guest execution");
            t.Equals(
                ctx.cop0_epc,
                0x00125000u,
                "data load: EPC should identify the associated load");
            t.IsFalse(
                (ctx.cop0_cause & kCauseBd) != 0u,
                "data load: direct fault should leave Cause.BD clear");
            expectDeliveredBusError(
                t,
                ctx,
                EXCEPTION_BUS_ERROR_DATA,
                0x22001230u,
                0xA5A5A5A5u,
                "data load: ");
        });

        tc.Run("unmapped physical store publishes DBE state", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            ctx.pc = 0x00126004u;
            ctx.branch_pc = 0x00126000u;
            ctx.in_delay_slot = true;
            ctx.cop0_badvaddr = 0x5A5A5A5Au;
            ctx.cop0_badpaddr = 0x33333330u;

            const bool raised = raisesGuestException([&]()
            {
                runtime.Store32(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    0x22005678u,
                    0x11223344u);
            });

            t.IsTrue(raised, "physical-bus store failure should unwind guest execution");
            t.Equals(
                ctx.cop0_epc,
                0x00126000u,
                "data store: delay-slot fault should restart at the owning branch");
            t.IsTrue(
                (ctx.cop0_cause & kCauseBd) != 0u,
                "data store: delay-slot fault should set Cause.BD");
            expectDeliveredBusError(
                t,
                ctx,
                EXCEPTION_BUS_ERROR_DATA,
                0x22005670u,
                0x5A5A5A5Au,
                "data store: ");
        });

        tc.Run("BEM masks delivery and physical-address capture", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "PS2Memory initialize should succeed");
            R5900Context ctx{};
            ctx.pc = 0x00127000u;
            ctx.cop0_status = kStatusBem;
            ctx.cop0_cause = 0x00000120u;
            ctx.cop0_epc = 0x12345678u;
            ctx.cop0_badvaddr = 0xABCDEF01u;
            ctx.cop0_badpaddr = 0x76543210u;

            bool raised = false;
            uint32_t loaded = 0xFFFFFFFFu;
            try
            {
                runtime.SignalBusError(
                    &ctx,
                    PS2BusErrorAccess::InstructionFetch,
                    0x22001000u);
                loaded = runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    0x22002000u);
                runtime.Store32(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    0x22003000u,
                    0xDEADBEEFu);
            }
            catch (const PS2GuestException &)
            {
                raised = true;
            }

            t.IsFalse(raised, "Status.BEM should suppress fetch, load, and store delivery");
            t.Equals(loaded, 0u, "masked bus load should use the deterministic undefined value");
            t.Equals(ctx.pc, 0x00127000u, "masked bus errors should not redirect control flow");
            t.Equals(ctx.cop0_cause, 0x00000120u, "masked bus errors should not alter Cause");
            t.Equals(ctx.cop0_epc, 0x12345678u, "masked bus errors should not alter EPC");
            t.Equals(ctx.cop0_badvaddr, 0xABCDEF01u, "masked bus errors should not alter BadVAddr");
            t.Equals(ctx.cop0_badpaddr, 0x76543210u, "Status.BEM should suppress BadPAddr capture");
            t.Equals(ctx.cop0_status, kStatusBem, "masked faults should preserve Status.BEM");
        });

        tc.Run("EXL and ERL mask bus errors without setting BEM", [](TestCase &t)
        {
            for (const uint32_t exceptionLevel :
                 std::array<uint32_t, 2u>{kStatusExl, kStatusErl})
            {
                PS2Runtime runtime;
                t.IsTrue(
                    runtime.memory().initialize(),
                    "PS2Memory initialize should succeed");
                R5900Context ctx{};
                ctx.pc = 0x00128000u;
                ctx.cop0_status = exceptionLevel;
                ctx.cop0_cause = 0x00000240u;
                ctx.cop0_badpaddr = 0x13572460u;

                bool raised = false;
                try
                {
                    runtime.SignalBusError(
                        &ctx,
                        PS2BusErrorAccess::InstructionFetch,
                        0x22004000u);
                    (void)runtime.Load32(
                        runtime.memory().getRDRAM(),
                        &ctx,
                        0x22005000u);
                }
                catch (const PS2GuestException &)
                {
                    raised = true;
                }

                t.IsFalse(raised, "active exception levels should suppress bus-error delivery");
                t.Equals(ctx.pc, 0x00128000u, "active exception levels should preserve PC");
                t.Equals(ctx.cop0_cause, 0x00000240u, "active exception levels should preserve Cause");
                t.Equals(ctx.cop0_badpaddr, 0x13572460u, "active exception levels should suppress BadPAddr capture");
                t.Equals(ctx.cop0_status, exceptionLevel, "active exception levels should not set BEM");
            }
        });

        tc.Run("all data widths expose typed physical-bus faults", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(memory.initialize(), "PS2Memory initialize should succeed");
            constexpr uint32_t address = 0x22000000u;

            expectPhysicalBusFault(t, [&]() { (void)memory.read8(address); }, address, "read8: ");
            expectPhysicalBusFault(t, [&]() { (void)memory.read16(address); }, address, "read16: ");
            expectPhysicalBusFault(t, [&]() { (void)memory.read32(address); }, address, "read32: ");
            expectPhysicalBusFault(t, [&]() { (void)memory.read64(address); }, address, "read64: ");
            expectPhysicalBusFault(t, [&]() { (void)memory.read128(address); }, address, "read128: ");
            expectPhysicalBusFault(t, [&]() { memory.write8(address, 1u); }, address, "write8: ");
            expectPhysicalBusFault(t, [&]() { memory.write16(address, 2u); }, address, "write16: ");
            expectPhysicalBusFault(t, [&]() { memory.write32(address, 3u); }, address, "write32: ");
            expectPhysicalBusFault(t, [&]() { memory.write64(address, 4u); }, address, "write64: ");
            expectPhysicalBusFault(t, [&]() { memory.write128(address, _mm_setzero_si128()); }, address, "write128: ");
            expectPhysicalBusFault(t, [&]() { memory.writeMasked32(address, 5u, 0x3u); }, address, "masked32: ");
            expectPhysicalBusFault(t, [&]() { memory.writeMasked64(address, 6u, 0x0Fu); }, address, "masked64: ");
        });

        tc.Run("VU and GS documented BUSERR ranges are distinct from reserved slots", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(memory.initialize(), "PS2Memory initialize should succeed");

            expectPhysicalBusFault(
                t,
                [&]() { (void)memory.read32(0x11010000u); },
                0x11010000u,
                "VU BUSERR: ");
            expectPhysicalBusFault(
                t,
                [&]() { memory.write32(0x12010000u, 1u); },
                0x12010000u,
                "GS BUSERR: ");

            t.Equals(
                memory.read32(0x12000030u),
                0u,
                "ordinary reserved GS slots should retain indeterminate-zero behavior");
            memory.write32(0x12000030u, 0xFFFFFFFFu);
        });
    });
}
