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
    constexpr uint32_t kCounterCount = 0x10000000u;
    constexpr uint32_t kIpuOutputFifo = 0x10007000u;
    constexpr uint32_t kIpuControl = 0x10002010u;
    constexpr uint32_t kUnmappedKseg2 = 0xC0000000u;
    constexpr uint32_t kCauseExcCodeMask = 0x0000007Cu;

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
}

void register_ee_load_zero_tests()
{
    MiniTest::Case("EE loads to zero", [](TestCase &tc)
    {
        tc.Run("all affected translations retain their memory read", [](TestCase &t)
        {
            constexpr struct
            {
                const char *name;
                uint32_t opcode;
                const char *read;
            } cases[] = {
                {"LB", OPCODE_LB, "READ8("},
                {"LBU", OPCODE_LBU, "READ8("},
                {"LH", OPCODE_LH, "READ16("},
                {"LHU", OPCODE_LHU, "READ16("},
                {"LW", OPCODE_LW, "READ32("},
                {"LWU", OPCODE_LWU, "READ32("},
                {"LD", OPCODE_LD, "READ64("},
                {"LQ", OPCODE_LQ, "READ128("},
            };

            R5900Decoder decoder;
            CodeGenerator generator({}, {});
            for (size_t index = 0u; index < std::size(cases); ++index)
            {
                const uint32_t raw =
                    (cases[index].opcode << 26u) |
                    (4u << 21u) |
                    0x20u;
                const Instruction inst = decoder.decodeInstruction(
                    0x00120000u + static_cast<uint32_t>(index * 4u), raw);
                const std::string generated =
                    generator.translateInstruction(inst);

                t.IsTrue(
                    generated.find(cases[index].read) != std::string::npos,
                    std::string(cases[index].name) +
                        " to r0 should retain its memory read");
            }
        });

        tc.Run("aligned MMIO reads execute before the r0 write is discarded", [](TestCase &t)
        {
            PS2Runtime runtimeStorage;
            t.IsTrue(
                runtimeStorage.memory().initialize(),
                "runtime memory should initialize");
            PS2Runtime *runtime = &runtimeStorage;
            R5900Context context{};
            R5900Context *ctx = &context;
            uint8_t *rdram = runtime->memory().getRDRAM();

            uint32_t counterReadCount = 0u;
            runtime->memory().setEeCounterCycleCallback([&]()
            {
                ++counterReadCount;
                return uint64_t{0u};
            });

            const auto expectCounterRead = [&](const char *name, const auto &operation)
            {
                const uint32_t before = counterReadCount;
                operation();
                t.IsTrue(
                    counterReadCount > before,
                    std::string(name) +
                        " to r0 should perform the counter MMIO read");
                t.IsTrue(
                    isZeroRegister(*ctx),
                    std::string(name) + " should preserve r0");
            };

            expectCounterRead("LB", [&]()
            {
                SET_GPR_S32(ctx, 0, (int8_t)READ8(kCounterCount));
            });
            expectCounterRead("LBU", [&]()
            {
                SET_GPR_U32(ctx, 0, (uint8_t)READ8(kCounterCount));
            });
            expectCounterRead("LH", [&]()
            {
                SET_GPR_S32(ctx, 0, (int16_t)READ16(kCounterCount));
            });
            expectCounterRead("LHU", [&]()
            {
                SET_GPR_U32(ctx, 0, (uint16_t)READ16(kCounterCount));
            });
            expectCounterRead("LW", [&]()
            {
                SET_GPR_S32(ctx, 0, (int32_t)READ32(kCounterCount));
            });
            expectCounterRead("LWU", [&]()
            {
                SET_GPR_U64(ctx, 0, (uint64_t)(uint32_t)READ32(kCounterCount));
            });
            expectCounterRead("LD", [&]()
            {
                SET_GPR_U64(ctx, 0, READ64(kCounterCount));
            });

            constexpr std::array<uint8_t, 16> fifoPayload{{
                0x00u, 0x11u, 0x22u, 0x33u,
                0x44u, 0x55u, 0x66u, 0x77u,
                0x88u, 0x99u, 0xAAu, 0xBBu,
                0xCCu, 0xDDu, 0xEEu, 0xFFu,
            }};
            t.Equals(
                runtime->memory().writeIpuOutputFifo(
                    fifoPayload.data(), 1u),
                1u,
                "IPU output FIFO fixture should accept one qword");
            t.Equals(
                (runtime->memory().readIORegister(kIpuControl) >> 4u) & 0xFu,
                1u,
                "IPU output FIFO should begin with one qword");

            SET_GPR_VEC(ctx, 0, READ128(kIpuOutputFifo));

            t.Equals(
                (runtime->memory().readIORegister(kIpuControl) >> 4u) & 0xFu,
                0u,
                "LQ to r0 should consume the MMIO FIFO qword");
            t.IsTrue(isZeroRegister(*ctx), "LQ should preserve r0");
        });

        tc.Run("TLB faults occur before every r0 load result is discarded", [](TestCase &t)
        {
            PS2Runtime runtimeStorage;
            t.IsTrue(
                runtimeStorage.memory().initialize(),
                "runtime memory should initialize");
            PS2Runtime *runtime = &runtimeStorage;
            uint8_t *rdram = runtime->memory().getRDRAM();

            constexpr const char *names[] = {
                "LB", "LBU", "LH", "LHU", "LW", "LWU", "LD", "LQ"};
            for (size_t index = 0u; index < std::size(names); ++index)
            {
                R5900Context context{};
                context.pc =
                    0x00130000u + static_cast<uint32_t>(index * 4u);
                R5900Context *ctx = &context;

                const bool raised = raisesGuestException([&]()
                {
                    switch (index)
                    {
                    case 0u:
                        SET_GPR_S32(ctx, 0, (int8_t)READ8(kUnmappedKseg2));
                        break;
                    case 1u:
                        SET_GPR_U32(ctx, 0, (uint8_t)READ8(kUnmappedKseg2));
                        break;
                    case 2u:
                        SET_GPR_S32(ctx, 0, (int16_t)READ16(kUnmappedKseg2));
                        break;
                    case 3u:
                        SET_GPR_U32(ctx, 0, (uint16_t)READ16(kUnmappedKseg2));
                        break;
                    case 4u:
                        SET_GPR_S32(ctx, 0, (int32_t)READ32(kUnmappedKseg2));
                        break;
                    case 5u:
                        SET_GPR_U64(
                            ctx, 0,
                            (uint64_t)(uint32_t)READ32(kUnmappedKseg2));
                        break;
                    case 6u:
                        SET_GPR_U64(ctx, 0, READ64(kUnmappedKseg2));
                        break;
                    case 7u:
                        SET_GPR_VEC(ctx, 0, READ128(kUnmappedKseg2));
                        break;
                    }
                });

                const std::string prefix =
                    std::string(names[index]) + ": ";
                t.IsTrue(
                    raised,
                    prefix + "r0 destination must not suppress TLB refill");
                t.Equals(
                    context.cop0_cause & kCauseExcCodeMask,
                    (static_cast<uint32_t>(EXCEPTION_TLB_REFILL_LOAD) << 2u) &
                        kCauseExcCodeMask,
                    prefix + "Cause.ExcCode should identify load refill");
                t.Equals(
                    context.cop0_badvaddr,
                    kUnmappedKseg2,
                    prefix + "BadVAddr should retain the access address");
                t.Equals(
                    context.cop0_epc,
                    0x00130000u + static_cast<uint32_t>(index * 4u),
                    prefix + "EPC should retain the load PC");
                t.IsTrue(
                    isZeroRegister(context),
                    prefix + "fault should preserve r0");
            }
        });
    });
}
