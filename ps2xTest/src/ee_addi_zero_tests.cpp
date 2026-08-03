#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kCauseBd = 0x80000000u;
    constexpr uint32_t kCauseExcCodeMask = 0x0000007Cu;
    constexpr uint32_t kGeneralExceptionVector = 0x80000180u;

    bool isZeroRegister(const R5900Context &ctx)
    {
        alignas(16) std::array<uint8_t, 16> bytes{};
        std::memcpy(bytes.data(), &ctx.r[0], bytes.size());
        for (const uint8_t byte : bytes)
        {
            if (byte != 0u)
            {
                return false;
            }
        }
        return true;
    }

    std::string translateAddiZero(uint32_t address, int16_t immediate)
    {
        const uint32_t raw =
            (OPCODE_ADDI << 26u) |
            (4u << 21u) |
            static_cast<uint16_t>(immediate);
        R5900Decoder decoder;
        CodeGenerator generator({}, {});
        return generator.translateInstruction(
            decoder.decodeInstruction(address, raw));
    }

    bool executeAddiZero(
        PS2Runtime &runtime,
        R5900Context &ctx,
        int16_t immediate)
    {
        try
        {
            R5900Context *ctxPtr = &ctx;
            uint32_t result = 0u;
            bool overflow = false;
            ADD32_OV(
                GPR_U32(ctxPtr, 4),
                static_cast<int32_t>(immediate),
                result,
                overflow);
            if (overflow)
            {
                runtime.SignalException(
                    ctxPtr, EXCEPTION_INTEGER_OVERFLOW);
            }
            else
            {
                SET_GPR_S32(ctxPtr, 0, static_cast<int32_t>(result));
            }
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }

    void setSource(R5900Context &ctx, uint32_t value)
    {
        SET_GPR_S32(&ctx, 4, static_cast<int32_t>(value));
    }

    void expectOverflowState(
        TestCase &t,
        const R5900Context &ctx,
        uint32_t expectedEpc,
        bool expectedBd,
        const std::string &prefix)
    {
        t.Equals(
            ctx.cop0_cause & kCauseExcCodeMask,
            (static_cast<uint32_t>(EXCEPTION_INTEGER_OVERFLOW) << 2u) &
                kCauseExcCodeMask,
            prefix + "Cause.ExcCode should identify Integer Overflow");
        t.Equals(
            ctx.cop0_epc,
            expectedEpc,
            prefix + "EPC should identify the restart instruction");
        t.Equals(
            (ctx.cop0_cause & kCauseBd) != 0u,
            expectedBd,
            prefix + "Cause.BD should match delay-slot execution");
        t.Equals(
            ctx.pc,
            kGeneralExceptionVector,
            prefix + "overflow should enter V_COMMON");
        t.IsTrue(
            isZeroRegister(ctx),
            prefix + "overflow must preserve r0");
    }
}

void register_ee_addi_zero_tests()
{
    MiniTest::Case("EE ADDI to zero", [](TestCase &tc)
    {
        tc.Run("direct overflow is not discarded with the r0 write", [](TestCase &t)
        {
            constexpr uint32_t address = 0x00140000u;
            const std::string generated = translateAddiZero(address, 1);
            t.IsTrue(
                generated.find("ADD32_OV(") != std::string::npos,
                "ADDI to r0 should retain the overflow computation");
            t.IsTrue(
                generated.find("EXCEPTION_INTEGER_OVERFLOW") !=
                    std::string::npos,
                "ADDI to r0 should retain the overflow exception path");

            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = address;
            setSource(ctx, 0x7FFFFFFFu);

            t.IsTrue(
                executeAddiZero(runtime, ctx, 1),
                "0x7fffffff + 1 should unwind through guest overflow");
            expectOverflowState(t, ctx, address, false, "direct: ");
        });

        tc.Run("non-overflowing r0 ADDI completes without a write", [](TestCase &t)
        {
            constexpr uint32_t address = 0x00140100u;
            const std::string generated = translateAddiZero(address, 1);
            t.IsTrue(
                generated.find("ADD32_OV(") != std::string::npos,
                "non-overflowing ADDI to r0 should still perform the add");

            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = address;
            ctx.cop0_cause = 0x00000020u;
            ctx.cop0_epc = 0x00100030u;
            setSource(ctx, 1u);

            t.IsFalse(
                executeAddiZero(runtime, ctx, 1),
                "1 + 1 should not raise Integer Overflow");
            t.Equals(
                ctx.pc,
                address,
                "non-overflowing ADDI should not redirect control flow");
            t.Equals(
                ctx.cop0_cause,
                0x00000020u,
                "non-overflowing ADDI should preserve Cause");
            t.Equals(
                ctx.cop0_epc,
                0x00100030u,
                "non-overflowing ADDI should preserve EPC");
            t.IsTrue(
                isZeroRegister(ctx),
                "non-overflowing ADDI should discard only the r0 result");
        });

        tc.Run("delay-slot overflow restarts at the owning branch", [](TestCase &t)
        {
            constexpr uint32_t branchAddress = 0x00140200u;
            constexpr uint32_t addiAddress = branchAddress + 4u;
            const std::string generated = translateAddiZero(addiAddress, 1);
            t.IsTrue(
                generated.find("EXCEPTION_INTEGER_OVERFLOW") !=
                    std::string::npos,
                "delay-slot ADDI to r0 should retain its exception path");

            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = addiAddress;
            ctx.branch_pc = branchAddress;
            ctx.in_delay_slot = true;
            setSource(ctx, 0x7FFFFFFFu);

            t.IsTrue(
                executeAddiZero(runtime, ctx, 1),
                "delay-slot overflow should unwind through the guest path");
            expectOverflowState(
                t, ctx, branchAddress, true, "delay slot: ");
        });
    });
}
