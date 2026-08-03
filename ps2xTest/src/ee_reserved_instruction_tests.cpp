#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ee_cycle_model.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2_runtime.h"

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

    struct ReservedEncoding
    {
        const char *name;
        uint32_t opcode;
    };

    bool raisesGuestException(PS2Runtime &runtime, R5900Context &ctx)
    {
        try
        {
            runtime.SignalException(
                &ctx, EXCEPTION_RESERVED_INSTRUCTION);
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }
}

void register_ee_reserved_instruction_tests()
{
    MiniTest::Case("EE reserved instructions", [](TestCase &tc)
    {
        tc.Run("LL LLD SC and SCD raise RI without data side effects", [](TestCase &t)
        {
            constexpr std::array<ReservedEncoding, 4> encodings{{
                {"LL", OPCODE_LL},
                {"LLD", OPCODE_LLD},
                {"SC", OPCODE_SC},
                {"SCD", OPCODE_SCD},
            }};
            constexpr std::array<uint8_t, 16> destinationSentinel{{
                0x10u, 0x21u, 0x32u, 0x43u,
                0x54u, 0x65u, 0x76u, 0x87u,
                0x98u, 0xA9u, 0xBAu, 0xCBu,
                0xDCu, 0xEDu, 0xFEu, 0x0Fu,
            }};
            constexpr std::array<uint8_t, 8> memorySentinel{{
                0xD4u, 0xC3u, 0xB2u, 0xA1u,
                0x86u, 0x77u, 0x68u, 0x59u,
            }};
            constexpr uint32_t targetAddress = 0x00002000u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "runtime memory should initialize");
            uint8_t *const rdram = runtime.memory().getRDRAM();
            t.IsTrue(rdram != nullptr, "runtime should expose RDRAM");

            R5900Decoder decoder;
            CodeGenerator generator({}, {});

            for (size_t encodingIndex = 0u;
                 encodingIndex < encodings.size();
                 ++encodingIndex)
            {
                const ReservedEncoding &encoding =
                    encodings[encodingIndex];
                const uint32_t address =
                    0x00110000u +
                    static_cast<uint32_t>(encodingIndex * 0x100u);
                const uint32_t raw =
                    (encoding.opcode << 26u) |
                    (4u << 21u) |
                    (5u << 16u) |
                    0x0020u;
                const Instruction inst =
                    decoder.decodeInstruction(address, raw, false);
                const std::string prefix =
                    std::string(encoding.name) + ": ";

                t.Equals(
                    generator.translateInstruction(inst),
                    std::string(
                        "runtime->SignalException(ctx, "
                        "EXCEPTION_RESERVED_INSTRUCTION);"),
                    prefix + "translation should raise guest RI");
                t.IsFalse(
                    inst.isLoad,
                    prefix + "unsupported encoding is not a load");
                t.IsFalse(
                    inst.isStore,
                    prefix + "unsupported encoding is not a store");
                t.IsFalse(
                    inst.modificationInfo.modifiesGPR,
                    prefix + "unsupported encoding cannot write rt");
                t.IsFalse(
                    inst.modificationInfo.modifiesMemory,
                    prefix + "unsupported encoding cannot write memory");
                t.IsTrue(
                    inst.modificationInfo.modifiesControl,
                    prefix + "RI changes exception control state");
                t.Equals(
                    eeInstructionCycleTicks(inst), 9u,
                    prefix + "RI should use the ordinary issue weight");

                for (const bool inDelaySlot : {false, true})
                {
                    R5900Context ctx{};
                    ctx.pc = address;
                    ctx.branch_pc = address - 4u;
                    ctx.in_delay_slot = inDelaySlot;
                    std::memcpy(
                        &ctx.r[inst.rt],
                        destinationSentinel.data(),
                        destinationSentinel.size());
                    std::memcpy(
                        rdram + targetAddress,
                        memorySentinel.data(),
                        memorySentinel.size());

                    const std::string mode =
                        inDelaySlot ? "delay slot" : "direct";
                    t.IsTrue(
                        raisesGuestException(runtime, ctx),
                        prefix + mode + " execution should unwind");
                    t.Equals(
                        ctx.cop0_cause & kCauseExcCodeMask,
                        (static_cast<uint32_t>(
                             EXCEPTION_RESERVED_INSTRUCTION) << 2u) &
                            kCauseExcCodeMask,
                        prefix + mode + " Cause.ExcCode should be RI");
                    t.Equals(
                        ctx.cop0_epc,
                        inDelaySlot ? address - 4u : address,
                        prefix + mode + " EPC should identify restart PC");
                    t.Equals(
                        (ctx.cop0_cause & kCauseBd) != 0u,
                        inDelaySlot,
                        prefix + mode + " Cause.BD should match execution");
                    t.Equals(
                        ctx.pc,
                        kGeneralExceptionVector,
                        prefix + mode + " should enter V_COMMON");

                    std::array<uint8_t, 16> destinationAfter{};
                    std::memcpy(
                        destinationAfter.data(),
                        &ctx.r[inst.rt],
                        destinationAfter.size());
                    t.Equals(
                        destinationAfter,
                        destinationSentinel,
                        prefix + mode + " must preserve destination GPR");

                    std::array<uint8_t, 8> memoryAfter{};
                    std::memcpy(
                        memoryAfter.data(),
                        rdram + targetAddress,
                        memoryAfter.size());
                    t.Equals(
                        memoryAfter,
                        memorySentinel,
                        prefix + mode + " must preserve target memory");
                }
            }
        });
    });
}
