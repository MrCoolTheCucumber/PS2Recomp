#include "MiniTest.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_psmct32.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_clip.h"
#include "runtime/ps2_vu1.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    constexpr uint32_t kVuUpperNop = 0x000002FFu;
    constexpr uint32_t kVuUpperEnd = 1u << 30u;

    struct Vu1Fixture
    {
        PS2Memory mem;
        GS gs;
        uint8_t *code = nullptr;
        uint8_t *data = nullptr;

        bool initialize()
        {
            if (!mem.initialize())
                return false;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());
            code = mem.getVU1Code();
            data = mem.getVU1Data();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            std::memset(data, 0, PS2_VU1_DATA_SIZE);
            return code != nullptr && data != nullptr;
        }
    };

    uint32_t makeVifCmd(uint8_t opcode, uint8_t num, uint16_t imm)
    {
        return (static_cast<uint32_t>(opcode) << 24) |
               (static_cast<uint32_t>(num) << 16) |
               static_cast<uint32_t>(imm);
    }

    uint64_t makeGifTag(uint16_t nloop, uint8_t flg, uint8_t nreg, bool eop = true)
    {
        uint64_t tag = static_cast<uint64_t>(nloop & 0x7FFFu);
        if (eop)
            tag |= (1ull << 15);
        tag |= (static_cast<uint64_t>(flg & 0x3u) << 58);
        tag |= (static_cast<uint64_t>(nreg & 0xFu) << 60);
        return tag;
    }

    uint32_t makeVuLowerSpecial(uint8_t specialOp, uint8_t is, uint8_t it = 0u, uint8_t id = 0u, uint8_t dest = 0u)
    {
        return (0x40u << 25) |
               (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(it & 0x1Fu) << 16) |
               (static_cast<uint32_t>(is & 0x1Fu) << 11) |
               (static_cast<uint32_t>(id & 0x1Fu) << 6) |
               (static_cast<uint32_t>(specialOp & 0x7Cu) << 4) |
               static_cast<uint32_t>(specialOp & 0x3u) |
               0x3Cu;
    }

    uint32_t makeVuLowerDirect(uint8_t funct, uint8_t is, uint8_t it = 0u, uint8_t id = 0u, uint8_t dest = 0u)
    {
        return (0x40u << 25) |
               (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(it & 0x1Fu) << 16) |
               (static_cast<uint32_t>(is & 0x1Fu) << 11) |
               (static_cast<uint32_t>(id & 0x1Fu) << 6) |
               static_cast<uint32_t>(funct & 0x3Fu);
    }

    uint32_t makeVuUpper(uint8_t op, uint8_t dest, uint8_t ft, uint8_t fs, uint8_t fd)
    {
        return (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(ft & 0x1Fu) << 16) |
               (static_cast<uint32_t>(fs & 0x1Fu) << 11) |
               (static_cast<uint32_t>(fd & 0x1Fu) << 6) |
               static_cast<uint32_t>(op & 0x3Fu);
    }

    uint32_t makeVuUpperSpecial(uint8_t specialOp, uint8_t dest, uint8_t ft, uint8_t fs)
    {
        return (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(ft & 0x1Fu) << 16) |
               (static_cast<uint32_t>(fs & 0x1Fu) << 11) |
               (static_cast<uint32_t>(specialOp & 0x7Cu) << 4) |
               static_cast<uint32_t>(specialOp & 0x3u) |
               0x3Cu;
    }

    uint32_t makeVuLq(uint8_t dest, uint8_t targetVf, uint8_t baseVi, int16_t imm)
    {
        return (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(targetVf & 0x1Fu) << 16) |
               (static_cast<uint32_t>(baseVi & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuSq(uint8_t dest, uint8_t sourceVf, uint8_t baseVi, int16_t imm)
    {
        return (0x01u << 25) |
               (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(baseVi & 0xFu) << 16) |
               (static_cast<uint32_t>(sourceVf & 0x1Fu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuIaddiu(uint8_t it, uint8_t is, int16_t imm)
    {
        return (0x08u << 25) |
               (static_cast<uint32_t>(it & 0xFu) << 16) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuBranch(int16_t imm)
    {
        return (0x20u << 25) | (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuIbeq(uint8_t is, uint8_t it, int16_t imm)
    {
        return (0x28u << 25) |
               (static_cast<uint32_t>(it & 0xFu) << 16) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuIbgtz(uint8_t is, int16_t imm)
    {
        return (0x2Du << 25) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuMtir(uint8_t it, uint8_t fs, uint8_t fsf)
    {
        return makeVuLowerSpecial(0x3Cu, fs, it, 0u, fsf & 0x3u);
    }

    uint32_t makeVuDiv(uint8_t fs, uint8_t ft, uint8_t fsf, uint8_t ftf)
    {
        return makeVuLowerSpecial(0x38u, fs, ft, 0u, static_cast<uint8_t>(((ftf & 0x3u) << 2) | (fsf & 0x3u)));
    }

    uint32_t makeVuSqrt(uint8_t ft, uint8_t ftf)
    {
        return makeVuLowerSpecial(0x39u, 0u, ft, 0u, static_cast<uint8_t>((ftf & 0x3u) << 2));
    }

    uint32_t makeVuWaitQ()
    {
        return makeVuLowerSpecial(0x3Bu, 0u);
    }

    void writeVuInstructionPair(uint8_t *code, uint32_t pc, uint32_t lower, uint32_t upper)
    {
        std::memcpy(code + pc, &lower, sizeof(lower));
        std::memcpy(code + pc + sizeof(lower), &upper, sizeof(upper));
    }

    uint64_t packVuInstructionPair(uint32_t lower, uint32_t upper)
    {
        return static_cast<uint64_t>(lower) | (static_cast<uint64_t>(upper) << 32);
    }

    void appendU32(std::vector<uint8_t> &bytes, uint32_t value)
    {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(&value);
        bytes.insert(bytes.end(), src, src + sizeof(value));
    }

    void uploadVu1Mpg(PS2Memory &mem, uint16_t instructionAddress, uint32_t lower, uint32_t upper)
    {
        std::vector<uint8_t> packet;
        appendU32(packet, makeVifCmd(0x4Au, 1u, instructionAddress));
        appendU32(packet, lower);
        appendU32(packet, upper);
        mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));
    }

    void writeVuQword(uint8_t *data, uint32_t qwordIndex, const float values[4])
    {
        std::memcpy(data + qwordIndex * 16u, values, sizeof(float) * 4u);
    }

    void readVuQword(const uint8_t *data, uint32_t qwordIndex, float values[4])
    {
        std::memcpy(values, data + qwordIndex * 16u, sizeof(float) * 4u);
    }

    void setFloatBits(float &value, uint32_t bits)
    {
        std::memcpy(&value, &bits, sizeof(bits));
    }

    uint32_t getFloatBits(float value)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
}

void register_ps2_vu1_tests()
{
    MiniTest::Case("PS2VU1", [](TestCase &tc)
    {
        tc.Run("upper ADD applies the destination mask", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, 0u, makeVuUpper(0x28u, 0xAu, 2u, 1u, 3u)); // ADD.xz vf3, vf1, vf2

            VU1Interpreter vu1;
            vu1.state().vf[1][0] = 1.0f;
            vu1.state().vf[1][1] = 2.0f;
            vu1.state().vf[1][2] = 3.0f;
            vu1.state().vf[1][3] = 4.0f;
            vu1.state().vf[2][0] = 10.0f;
            vu1.state().vf[2][1] = 20.0f;
            vu1.state().vf[2][2] = 30.0f;
            vu1.state().vf[2][3] = 40.0f;
            vu1.state().vf[3][0] = -1.0f;
            vu1.state().vf[3][1] = -2.0f;
            vu1.state().vf[3][2] = -3.0f;
            vu1.state().vf[3][3] = -4.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            t.Equals(vu1.state().vf[3][0], 11.0f, "ADD.x should write x");
            t.Equals(vu1.state().vf[3][1], -2.0f, "ADD.xz should preserve y");
            t.Equals(vu1.state().vf[3][2], 33.0f, "ADD.xz should write z");
            t.Equals(vu1.state().vf[3][3], -4.0f, "ADD.xz should preserve w");
        });

        tc.Run("FMAC flags become architectural after four following instruction pairs", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, 0u,
                makeVuUpper(0x28u, 0xFu, 2u, 1u, 3u)); // ADD.xyzw vf3, vf1, vf2
            for (uint32_t pc = 8u; pc <= 32u; pc += 8u)
                writeVuInstructionPair(fx.code, pc, 0u, kVuUpperNop);

            VU1Interpreter vu1;
            vu1.state().mac = 0x004Au;
            vu1.state().status = 0x0043u;
            const float source[4] = {1.0f, -2.0f, 3.0f, 4.0f};
            std::memcpy(vu1.state().vf[1], source, sizeof(source));

            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            t.Equals(vu1.state().mac, 0x004Au,
                     "issued ADD should leave the architectural MAC flags pending");
            t.Equals(vu1.state().status, 0x0043u,
                     "issued ADD should leave the architectural STATUS flags pending");

            for (uint32_t pair = 1u; pair < 4u; ++pair)
            {
                vu1.continueExecution(
                    fx.code, PS2_VU1_CODE_SIZE,
                    fx.data, PS2_VU1_DATA_SIZE,
                    fx.gs, &fx.mem, 1u);
                t.Equals(vu1.state().mac, 0x004Au,
                         "FMAC flags should remain pending before four cycles elapse");
                t.Equals(vu1.state().status, 0x0043u,
                         "STATUS flags should remain pending before four cycles elapse");
            }

            vu1.continueExecution(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 1u);
            t.Equals(vu1.state().mac, 0x0040u,
                     "negative y should set only the y sign MAC flag");
            t.Equals(vu1.state().status, 0x00C2u,
                     "STATUS should expose sign and retain prior sticky zero/sign flags");
        });

        tc.Run("FMAC flags preserve consecutive partial results across pipeline wraparound", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            constexpr std::array<uint8_t, 8> destinations{
                0x8u, 0x4u, 0x2u, 0x1u,
                0x8u, 0x4u, 0x2u, 0x1u,
            };
            for (uint32_t pair = 0u; pair < destinations.size(); ++pair)
            {
                writeVuInstructionPair(
                    fx.code, pair * 8u, 0u,
                    makeVuUpper(
                        0x28u, destinations[pair], 2u, 1u, 3u));
            }
            for (uint32_t pair = 8u; pair < 12u; ++pair)
                writeVuInstructionPair(
                    fx.code, pair * 8u, 0u, kVuUpperNop);

            const auto initializeState = [](VU1Interpreter &interpreter)
            {
                const float source[4] = {-1.0f, 0.0f, -1.0f, 0.0f};
                std::memcpy(
                    interpreter.state().vf[1], source, sizeof(source));
                interpreter.state().mac = 0x55AAu;
                interpreter.state().status = 0u;
            };

            VU1Interpreter stepped;
            VU1Interpreter batched;
            initializeState(stepped);
            initializeState(batched);

            constexpr std::array<uint32_t, 8> expectedMac{
                0x0080u, 0x0004u, 0x0020u, 0x0001u,
                0x0080u, 0x0004u, 0x0020u, 0x0001u,
            };
            constexpr std::array<uint32_t, 8> expectedStatus{
                0x0082u, 0x00C1u, 0x00C2u, 0x00C1u,
                0x00C2u, 0x00C1u, 0x00C2u, 0x00C1u,
            };

            stepped.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            for (uint32_t pair = 1u; pair < 12u; ++pair)
            {
                stepped.continueExecution(
                    fx.code, PS2_VU1_CODE_SIZE,
                    fx.data, PS2_VU1_DATA_SIZE,
                    fx.gs, &fx.mem, 1u);
                if (pair < 4u)
                {
                    t.Equals(
                        stepped.state().mac, 0x55AAu,
                        "no consecutive FMAC result should mature early");
                    t.Equals(
                        stepped.state().status, 0u,
                        "no consecutive FMAC STATUS should mature early");
                }
                else
                {
                    const uint32_t result = pair - 4u;
                    t.Equals(
                        stepped.state().mac, expectedMac[result],
                        "partial-destination MAC flags should mature in issue order");
                    t.Equals(
                        stepped.state().status, expectedStatus[result],
                        "partial-destination STATUS flags should retain sticky summaries");
                }
            }

            batched.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 12u);
            t.Equals(
                stepped.state().pc, batched.state().pc,
                "split and batched wraparound execution should stop at the same PC");
            t.Equals(
                stepped.state().mac, batched.state().mac,
                "split and batched wraparound execution should retain the same MAC flags");
            t.Equals(
                stepped.state().status, batched.state().status,
                "split and batched wraparound execution should retain the same STATUS flags");
        });

        tc.Run("FMAC reset discards pending flags", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, 0u,
                makeVuUpper(0x28u, 0x8u, 2u, 1u, 3u));
            for (uint32_t pc = 8u; pc <= 48u; pc += 8u)
                writeVuInstructionPair(
                    fx.code, pc, 0u, kVuUpperNop);

            VU1Interpreter vu1;
            vu1.state().vf[1][0] = -1.0f;
            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            vu1.reset();
            vu1.state().mac = 0x1234u;
            vu1.state().status = 0x05A0u;
            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 8u, 0u, 0u, 5u);
            t.Equals(
                vu1.state().mac, 0x1234u,
                "reset should prevent an old pending MAC result from becoming visible");
            t.Equals(
                vu1.state().status, 0x05A0u,
                "reset should prevent an old pending STATUS result from becoming visible");
        });

        tc.Run("E-bit completion flushes pending FMAC flags in issue order", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, 0u,
                makeVuUpper(0x28u, 0x8u, 2u, 1u, 3u) |
                    kVuUpperEnd);
            writeVuInstructionPair(
                fx.code, 8u, 0u,
                makeVuUpper(0x28u, 0x4u, 2u, 1u, 3u));

            VU1Interpreter vu1;
            const float source[4] = {-1.0f, 0.0f, 1.0f, 1.0f};
            std::memcpy(vu1.state().vf[1], source, sizeof(source));
            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 8u);

            t.IsFalse(
                vu1.isActive(),
                "the E-bit delay pair should complete the microprogram");
            t.Equals(
                vu1.state().mac, 0x0004u,
                "the newest pending partial result should be architectural after flush");
            t.Equals(
                vu1.state().status, 0x00C1u,
                "flush should commit older and newer sticky STATUS results in order");
        });

        tc.Run("VU FMAC truncates toward zero and restores the host rounding mode", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            // MADDA.xyzw ACC, vf1, vf5.x. These captured normal operands
            // distinguish the VU's required truncation from host
            // round-to-nearest in every lane.
            writeVuInstructionPair(fx.code, 0u, 0u, 0x01E508BCu);

            VU1Interpreter vu1;
            const uint32_t vf1Bits[4] = {
                0xBED52018u, 0x409A33CEu, 0xBFC2EB57u, 0x3B110CC2u};
            const uint32_t vf5Bits[4] = {
                0xC4454000u, 0x44454000u, 0xC6881A00u, 0x42380000u};
            const uint32_t accBits[4] = {
                0x4A90B980u, 0x4A87EA00u, 0x4B6A2CB4u, 0x4501E920u};
            const uint32_t expectedBits[4] = {
                0x4A90BC10u, 0x4A87CC4Bu, 0x4B6A3165u, 0x4501CD2Fu};
            for (size_t lane = 0u; lane < 4u; ++lane)
            {
                setFloatBits(vu1.state().vf[1][lane], vf1Bits[lane]);
                setFloatBits(vu1.state().vf[5][lane], vf5Bits[lane]);
                setFloatBits(vu1.state().acc[lane], accBits[lane]);
            }

            const int hostRoundingMode = std::fegetround();
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            for (size_t lane = 0u; lane < 4u; ++lane)
            {
                t.Equals(getFloatBits(vu1.state().acc[lane]), expectedBits[lane],
                         "MADDA should truncate its result toward zero");
            }
            t.Equals(std::fegetround(), hostRoundingMode,
                     "VU execution should restore the caller's rounding mode");
        });

        tc.Run("VU MADD retains the product through the truncated accumulator add", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            // MADDy.xyzw ACC, vf2, vf5.y.  With these captured operands,
            // rounding the multiplication before the addition produces
            // 0x4b69e65e; the VU's single truncated product-sum is
            // 0x4b69e65d.
            writeVuInstructionPair(fx.code, 0u, 0u, 0x01E510BDu);

            VU1Interpreter vu1;
            setFloatBits(vu1.state().acc[2], 0x4B6A3856u);
            setFloatBits(vu1.state().vf[2][2], 0x41004B21u);
            setFloatBits(vu1.state().vf[5][1], 0xC5239000u);

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            t.Equals(getFloatBits(vu1.state().acc[2]), 0x4B69E65Du,
                     "MADD should truncate once after the fused product-sum");
        });

        tc.Run("CLIP uses absolute w and architectural flag ordering", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, 0u,
                makeVuUpperSpecial(0x1Fu, 0xEu, 2u, 1u)); // CLIPw.xyz vf1, vf2w

            VU1Interpreter vu1;
            vu1.state().clip = 0x00123456u;
            vu1.state().vf[1][0] = 2.0f;
            vu1.state().vf[1][1] = -3.0f;
            vu1.state().vf[1][2] = 0.5f;
            vu1.state().vf[2][3] = -1.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            const uint32_t expected =
                ((0x00123456u << 6u) | 0x09u) & 0x00FFFFFFu;
            t.Equals(vu1.state().clip, expected,
                     "CLIP should set +x in bit 0 and -y in bit 3 using |w|");

            t.Equals(Ps2VuUpdateClipFlags(
                         0u,
                         0x00800000u,
                         0x80800000u,
                         0x00000001u,
                         0x00000000u),
                     0x09u,
                     "zero w should compare against the largest denormal bit pattern");
        });

        tc.Run("VU FMAC flushes denormal inputs and underflow results to signed zero", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, 0u,
                makeVuUpper(0x2Au, 0xEu, 2u, 1u, 3u)); // MUL.xyz vf3, vf1, vf2

            VU1Interpreter vu1;
            setFloatBits(vu1.state().vf[1][0], 0x00800000u); // smallest positive normal
            setFloatBits(vu1.state().vf[1][1], 0x80800000u); // smallest negative normal
            setFloatBits(vu1.state().vf[1][2], 0x00000001u); // positive denormal input
            setFloatBits(vu1.state().vf[2][0], 0x3F000000u); // 0.5
            setFloatBits(vu1.state().vf[2][1], 0x3F000000u); // 0.5
            setFloatBits(vu1.state().vf[2][2], 0x7F000000u); // would normalize an IEEE denormal

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            t.Equals(getFloatBits(vu1.state().vf[3][0]), 0x00000000u,
                     "positive exponent underflow should clamp to positive zero");
            t.Equals(getFloatBits(vu1.state().vf[3][1]), 0x80000000u,
                     "negative exponent underflow should clamp to negative zero");
            t.Equals(getFloatBits(vu1.state().vf[3][2]), 0x00000000u,
                     "denormal FMAC input should be treated as zero");
        });

        tc.Run("LOI commits the lower immediate after the upper instruction", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            const float newI = 7.0f;
            uint32_t lowerImmediate = 0u;
            std::memcpy(&lowerImmediate, &newI, sizeof(newI));
            const uint32_t upperAddiWithIBit = makeVuUpper(0x22u, 0xFu, 0u, 1u, 2u) | 0x80000000u; // ADDi.xyzw vf2, vf1
            writeVuInstructionPair(fx.code, 0u, lowerImmediate, upperAddiWithIBit);

            VU1Interpreter vu1;
            vu1.state().i = 2.0f;
            vu1.state().vf[1][0] = 1.0f;
            vu1.state().vf[1][1] = 2.0f;
            vu1.state().vf[1][2] = 3.0f;
            vu1.state().vf[1][3] = 4.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            t.Equals(vu1.state().vf[2][0], 3.0f, "ADDi should use old I for x");
            t.Equals(vu1.state().vf[2][1], 4.0f, "ADDi should use old I for y");
            t.Equals(vu1.state().vf[2][2], 5.0f, "ADDi should use old I for z");
            t.Equals(vu1.state().vf[2][3], 6.0f, "ADDi should use old I for w");
            t.Equals(vu1.state().i, 7.0f, "LOI should commit lower immediate into I after upper execution");
        });

        tc.Run("LQ and SQ use VI qword addressing and destination masks", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            const float sourceQw[4] = {10.0f, 20.0f, 30.0f, 40.0f};
            const float destQw[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
            writeVuQword(fx.data, 3u, sourceQw);
            writeVuQword(fx.data, 5u, destQw);
            writeVuInstructionPair(fx.code, 0u, makeVuLq(0x5u, 4u, 1u, 1), kVuUpperNop); // LQ.yw vf4, 1(vi1)
            writeVuInstructionPair(fx.code, 8u, makeVuSq(0xAu, 4u, 2u, 1), kVuUpperNop); // SQ.xz vf4, 1(vi2)

            VU1Interpreter vu1;
            vu1.state().vi[1] = 2;
            vu1.state().vi[2] = 4;
            vu1.state().vf[4][0] = 100.0f;
            vu1.state().vf[4][1] = 200.0f;
            vu1.state().vf[4][2] = 300.0f;
            vu1.state().vf[4][3] = 400.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 2u);

            t.Equals(vu1.state().vf[4][0], 100.0f, "LQ.yw should preserve x");
            t.Equals(vu1.state().vf[4][1], 20.0f, "LQ.yw should load y");
            t.Equals(vu1.state().vf[4][2], 300.0f, "LQ.yw should preserve z");
            t.Equals(vu1.state().vf[4][3], 40.0f, "LQ.yw should load w");

            float stored[4] = {};
            readVuQword(fx.data, 5u, stored);
            t.Equals(stored[0], 100.0f, "SQ.xz should store x");
            t.Equals(stored[1], -2.0f, "SQ.xz should preserve y");
            t.Equals(stored[2], 300.0f, "SQ.xz should store z");
            t.Equals(stored[3], -4.0f, "SQ.xz should preserve w");
        });

        tc.Run("integer lower ops keep VI0 hardwired to zero", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, makeVuIaddiu(2u, 1u, 5), kVuUpperNop);      // IADDIU vi2, vi1, 5
            writeVuInstructionPair(fx.code, 8u, makeVuIaddiu(0u, 2u, 7), kVuUpperNop);      // IADDIU vi0, vi2, 7
            writeVuInstructionPair(fx.code, 16u, makeVuLowerDirect(0x30u, 2u, 1u, 3u), kVuUpperNop); // IADD vi3, vi2, vi1

            VU1Interpreter vu1;
            vu1.state().vi[0] = 99;
            vu1.state().vi[1] = 10;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 3u);

            t.Equals(vu1.state().vi[2], 15, "IADDIU should add signed immediate to VI source");
            t.Equals(vu1.state().vi[3], 25, "IADD should add VI source registers");
            t.Equals(vu1.state().vi[0], 0, "VI0 should remain hardwired to zero");
        });

        tc.Run("XTOP and XITOP expose VIF TOP values to VI registers", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, makeVuLowerSpecial(0x68u, 0u, 2u), kVuUpperNop); // XTOP vi2
            writeVuInstructionPair(fx.code, 8u, makeVuLowerSpecial(0x69u, 0u, 3u), kVuUpperNop); // XITOP vi3

            VU1Interpreter vu1;
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0x123u, 0x2ABu, 2u);

            t.Equals(vu1.state().vi[2], 0x123, "XTOP should move TOP into the target VI register");
            t.Equals(vu1.state().vi[3], 0x2AB, "XITOP should move ITOP into the target VI register");
        });

        tc.Run("lower branch commits after one delay-slot instruction", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, makeVuBranch(2), kVuUpperNop);              // target pc = 24
            writeVuInstructionPair(fx.code, 8u, makeVuIaddiu(1u, 0u, 1), kVuUpperNop);      // delay slot
            writeVuInstructionPair(fx.code, 16u, makeVuIaddiu(2u, 0u, 99), kVuUpperNop);    // skipped
            writeVuInstructionPair(fx.code, 24u, makeVuIaddiu(3u, 0u, 7), kVuUpperNop);     // branch target

            VU1Interpreter vu1;
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 3u);

            t.Equals(vu1.state().vi[1], 1, "branch delay slot should execute");
            t.Equals(vu1.state().vi[2], 0, "instruction between delay slot and target should be skipped");
            t.Equals(vu1.state().vi[3], 7, "branch target should execute after the delay slot");
        });

        tc.Run("LQI post-increment exposes the prior VI value to the next branch", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u,
                makeVuLowerSpecial(0x34u, 1u, 1u, 0u, 0xFu),
                kVuUpperNop); // LQI.xyzw vf1, (vi1++)
            writeVuInstructionPair(fx.code, 8u, makeVuIbeq(1u, 2u, 2), kVuUpperNop);
            writeVuInstructionPair(fx.code, 16u, makeVuIaddiu(3u, 0u, 1), kVuUpperNop);
            writeVuInstructionPair(fx.code, 24u, makeVuIaddiu(4u, 0u, 99), kVuUpperNop);
            writeVuInstructionPair(fx.code, 32u, makeVuIaddiu(5u, 0u, 7), kVuUpperNop);

            VU1Interpreter vu1;
            vu1.state().vi[1] = 5;
            vu1.state().vi[2] = 5;
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 4u);

            t.Equals(vu1.state().vi[1], 6, "LQI should still commit its post-increment");
            t.Equals(vu1.state().vi[3], 1, "branch delay slot should execute");
            t.Equals(vu1.state().vi[4], 0, "branch should compare against the pre-increment value");
            t.Equals(vu1.state().vi[5], 7, "branch target should execute");
        });

        tc.Run("MTIR uses its scalar selector and feeds integer branches", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, makeVuMtir(4u, 7u, 0u), kVuUpperNop);
            writeVuInstructionPair(fx.code, 8u, 0u, kVuUpperNop);
            writeVuInstructionPair(fx.code, 16u, makeVuIbgtz(4u, 2), kVuUpperNop);
            writeVuInstructionPair(fx.code, 24u, makeVuIaddiu(5u, 0u, 1), kVuUpperNop);
            writeVuInstructionPair(fx.code, 32u, makeVuIaddiu(6u, 0u, 99), kVuUpperNop);
            writeVuInstructionPair(fx.code, 40u, makeVuIaddiu(7u, 0u, 7), kVuUpperNop);

            VU1Interpreter vu1;
            const uint32_t lanes[4] = {
                0x00000045u, 0x00000000u, 0x00000000u, 0x00000000u};
            std::memcpy(vu1.state().vf[7], lanes, sizeof(lanes));

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 5u);

            t.Equals(vu1.state().vi[4], 0x45,
                     "MTIR.x should read vf7.x rather than treating zero as an empty mask");
            t.Equals(vu1.state().vi[5], 1,
                     "the branch delay slot should execute");
            t.Equals(vu1.state().vi[6], 0,
                     "IBGTZ should skip the fallthrough instruction");
            t.Equals(vu1.state().vi[7], 7,
                     "IBGTZ should reach its target using the MTIR result");
        });

        tc.Run("lower side sees old VF value when upper writes the same register", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code,
                                   0u,
                                   makeVuSq(0xFu, 1u, 1u, 0),                 // SQ.xyzw vf1, 0(vi1)
                                   makeVuUpper(0x28u, 0xFu, 3u, 2u, 1u));     // ADD.xyzw vf1, vf2, vf3

            VU1Interpreter vu1;
            vu1.state().vi[1] = 6;
            vu1.state().vf[1][0] = 1.0f;
            vu1.state().vf[1][1] = 2.0f;
            vu1.state().vf[1][2] = 3.0f;
            vu1.state().vf[1][3] = 4.0f;
            vu1.state().vf[2][0] = 10.0f;
            vu1.state().vf[2][1] = 20.0f;
            vu1.state().vf[2][2] = 30.0f;
            vu1.state().vf[2][3] = 40.0f;
            vu1.state().vf[3][0] = 100.0f;
            vu1.state().vf[3][1] = 200.0f;
            vu1.state().vf[3][2] = 300.0f;
            vu1.state().vf[3][3] = 400.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            float stored[4] = {};
            readVuQword(fx.data, 6u, stored);
            t.Equals(stored[0], 1.0f, "SQ should observe old VF value for x");
            t.Equals(stored[1], 2.0f, "SQ should observe old VF value for y");
            t.Equals(stored[2], 3.0f, "SQ should observe old VF value for z");
            t.Equals(stored[3], 4.0f, "SQ should observe old VF value for w");
            t.Equals(vu1.state().vf[1][0], 110.0f, "upper ADD should write x after lower read");
            t.Equals(vu1.state().vf[1][1], 220.0f, "upper ADD should write y after lower read");
            t.Equals(vu1.state().vf[1][2], 330.0f, "upper ADD should write z after lower read");
            t.Equals(vu1.state().vf[1][3], 440.0f, "upper ADD should write w after lower read");
        });

        tc.Run("FDIV pipeline delays Q and WAITQ synchronizes its paired upper op", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, makeVuDiv(1u, 2u, 1u, 2u), kVuUpperNop);  // Q = vf1.y / vf2.z
            writeVuInstructionPair(fx.code, 8u, 0u, makeVuUpper(0x1Cu, 0x8u, 0u, 5u, 4u)); // vf4.x = vf5.x * old Q
            writeVuInstructionPair(fx.code, 16u, makeVuWaitQ(), makeVuUpper(0x1Cu, 0x8u, 0u, 5u, 6u)); // vf6.x = vf5.x * DIV Q
            writeVuInstructionPair(fx.code, 24u, makeVuSqrt(3u, 3u), kVuUpperNop);         // Q = sqrt(abs(vf3.w))
            writeVuInstructionPair(fx.code, 32u, makeVuWaitQ(), makeVuUpper(0x1Cu, 0x8u, 0u, 5u, 7u)); // vf7.x = vf5.x * SQRT Q

            VU1Interpreter vu1;
            vu1.state().q = 2.0f;
            vu1.state().vf[1][1] = 18.0f;
            vu1.state().vf[2][2] = 3.0f;
            vu1.state().vf[3][3] = 25.0f;
            vu1.state().vf[5][0] = 2.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            t.Equals(vu1.state().q, 2.0f, "DIV should leave the old Q visible while its result is pending");

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(vu1.state().vf[4][0], 4.0f, "unsynchronized MULq should use the old Q");

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(vu1.state().q, 6.0f, "WAITQ should commit the pending DIV result");
            t.Equals(vu1.state().vf[6][0], 12.0f, "the upper op paired with WAITQ should use the committed DIV result");

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(vu1.state().q, 6.0f, "SQRT should also leave Q pending");

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(vu1.state().q, 5.0f, "WAITQ should commit the pending SQRT result");
            t.Equals(vu1.state().vf[7][0], 10.0f, "the upper op paired with WAITQ should use the committed SQRT result");
        });

        tc.Run("DIV result becomes visible after seven following instruction pairs", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, makeVuDiv(1u, 2u, 0u, 0u), kVuUpperNop);
            for (uint32_t pc = 8u; pc <= 56u; pc += 8u)
                writeVuInstructionPair(fx.code, pc, 0u, kVuUpperNop);

            VU1Interpreter vu1;
            vu1.state().q = 3.0f;
            vu1.state().vf[1][0] = 18.0f;
            vu1.state().vf[2][0] = 3.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 7u);
            t.Equals(vu1.state().q, 3.0f, "Q should remain old through the first six pairs after DIV");

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(vu1.state().q, 6.0f, "Q should commit on the seventh pair after DIV");
        });

        tc.Run("MPG upload invalidates cached VU1 decode before MSCAL", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            VU1Interpreter vu1;
            fx.mem.setVu1MscalCallback([&](uint32_t startPC, uint32_t top, uint32_t itop)
            {
                vu1.execute(fx.code,
                            PS2_VU1_CODE_SIZE,
                            fx.data,
                            PS2_VU1_DATA_SIZE,
                            fx.gs,
                            &fx.mem,
                            startPC,
                            top,
                            itop,
                            1u);
            });

            uploadVu1Mpg(fx.mem, 0u, makeVuIaddiu(1u, 0u, 1), kVuUpperNop);
            const uint32_t firstMscal = makeVifCmd(0x14u, 0u, 0u);
            fx.mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&firstMscal), sizeof(firstMscal));
            t.Equals(vu1.state().vi[1], 1, "first MSCAL should execute the first uploaded program");

            uploadVu1Mpg(fx.mem, 0u, makeVuIaddiu(1u, 0u, 2), kVuUpperNop);
            const uint32_t secondMscal = makeVifCmd(0x14u, 0u, 0u);
            fx.mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&secondMscal), sizeof(secondMscal));
            t.Equals(vu1.state().vi[1], 2, "second MSCAL should see the MPG-updated instruction");
        });

        tc.Run("direct VU1 code writes invalidate cached decode", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            VU1Interpreter vu1;
            fx.mem.write64(PS2_VU1_CODE_BASE, packVuInstructionPair(makeVuIaddiu(1u, 0u, 1), kVuUpperNop));
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            t.Equals(vu1.state().vi[1], 1, "first execution should use the original direct write");

            fx.mem.write64(PS2_VU1_CODE_BASE, packVuInstructionPair(makeVuIaddiu(1u, 0u, 2), kVuUpperNop));
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            t.Equals(vu1.state().vi[1], 2, "second execution should rebuild decode after the direct write");
        });

        tc.Run("XGKICK sends a VU memory GIF packet through PATH1", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            uint8_t *vuCode = mem.getVU1Code();
            uint8_t *vuData = mem.getVU1Data();
            std::memset(vuCode, 0, PS2_VU1_CODE_SIZE);
            std::memset(vuData, 0, PS2_VU1_DATA_SIZE);

            constexpr uint32_t kLastQw = (PS2_VU1_DATA_SIZE / 16u) - 1u;
            const uint32_t tagOffset = kLastQw * 16u;

            const uint64_t imageTag = makeGifTag(1u, GIF_FMT_IMAGE, 0u, true);
            std::memcpy(vuData + tagOffset, &imageTag, sizeof(imageTag));

            for (uint32_t i = 0; i < 16u; ++i)
            {
                vuData[i] = static_cast<uint8_t>(0xC0u + i);
            }

            const uint32_t lower = makeVuLowerSpecial(0x6Cu, 1u);
            std::memcpy(vuCode + 0u, &lower, sizeof(lower));
            const uint32_t upper = kVuUpperEnd;
            std::memcpy(vuCode + 4u, &upper, sizeof(upper));

            VU1Interpreter vu1;
            vu1.state().vi[1] = static_cast<int32_t>(kLastQw);
            vu1.execute(vuCode,
                        PS2_VU1_CODE_SIZE,
                        vuData,
                        PS2_VU1_DATA_SIZE,
                        gs,
                        &mem,
                        0u,
                        0u,
                        0u,
                        2u);

            t.Equals(captured.size(), static_cast<size_t>(1u), "XGKICK should emit one wrapped GIF packet");
            if (!captured.empty())
            {
                t.Equals(captured[0].size(), static_cast<size_t>(32u), "wrapped packet should include tag plus one qword payload");
                bool payloadOk = true;
                for (uint32_t i = 0; i < 16u; ++i)
                {
                    if (captured[0].size() < 32u || captured[0][16u + i] != static_cast<uint8_t>(0xC0u + i))
                    {
                        payloadOk = false;
                        break;
                    }
                }
                t.IsTrue(payloadOk, "wrapped payload should be copied from start of VU1 memory");
            }
        });

        tc.Run("XGKICK streams VU memory after the kick starts", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            std::vector<std::vector<uint8_t>> captured;
            fx.mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            const uint64_t imageTag = makeGifTag(1u, GIF_FMT_IMAGE, 0u, true);
            std::memcpy(fx.data, &imageTag, sizeof(imageTag));
            std::memset(fx.data + 16u, 0x11, 16u);
            writeVuInstructionPair(
                fx.code, 0u, makeVuLowerSpecial(0x6Cu, 0u), kVuUpperNop);

            VU1Interpreter vu1;
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            std::memset(fx.data + 16u, 0xA5, 16u);

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE,
                       fx.data, PS2_VU1_DATA_SIZE,
                       fx.gs, &fx.mem, 0u, 0u, 3u);

            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "streamed XGKICK should complete after enough VU cycles");
            if (!captured.empty())
            {
                bool sawUpdatedPayload = captured[0].size() == 32u;
                for (size_t i = 16u; i < captured[0].size(); ++i)
                    sawUpdatedPayload = sawUpdatedPayload && captured[0][i] == 0xA5u;
                t.IsTrue(sawUpdatedPayload,
                         "XGKICK should read payload as PATH1 reaches it, not snapshot it at kick time");
            }
        });

        tc.Run("split VU1 advances match one equal-total cycle budget", [](TestCase &t)
        {
            Vu1Fixture singleFixture;
            Vu1Fixture splitFixture;
            t.IsTrue(singleFixture.initialize(),
                     "single-batch fixture should initialize");
            t.IsTrue(splitFixture.initialize(),
                     "split-batch fixture should initialize");

            std::vector<std::vector<uint8_t>> singlePackets;
            std::vector<std::vector<uint8_t>> splitPackets;
            singleFixture.mem.setGifPacketCallback(
                [&](const uint8_t *data, uint32_t sizeBytes)
                {
                    singlePackets.emplace_back(
                        data, data + sizeBytes);
                });
            splitFixture.mem.setGifPacketCallback(
                [&](const uint8_t *data, uint32_t sizeBytes)
                {
                    splitPackets.emplace_back(
                        data, data + sizeBytes);
                });

            const auto configure =
                [](Vu1Fixture &fixture)
                {
                    const uint64_t imageTag =
                        makeGifTag(1u, GIF_FMT_IMAGE, 0u);
                    std::memcpy(
                        fixture.data, &imageTag,
                        sizeof(imageTag));
                    for (uint32_t index = 0u;
                         index < 16u; ++index)
                    {
                        fixture.data[16u + index] =
                            static_cast<uint8_t>(0x80u + index);
                    }

                    writeVuInstructionPair(
                        fixture.code, 0u,
                        makeVuLowerSpecial(0x6Cu, 0u),
                        kVuUpperNop);
                    writeVuInstructionPair(
                        fixture.code, 8u,
                        makeVuBranch(2), kVuUpperNop);
                    writeVuInstructionPair(
                        fixture.code, 16u,
                        makeVuIaddiu(1u, 0u, 3),
                        kVuUpperNop);
                    writeVuInstructionPair(
                        fixture.code, 24u,
                        makeVuIaddiu(3u, 0u, 99),
                        kVuUpperNop);
                    writeVuInstructionPair(
                        fixture.code, 32u,
                        makeVuIaddiu(2u, 0u, 7),
                        kVuUpperEnd);
                    writeVuInstructionPair(
                        fixture.code, 40u, 0u,
                        kVuUpperNop);
                    fixture.mem.markVU1CodeModified();
                };
            configure(singleFixture);
            configure(splitFixture);

            VU1Interpreter single;
            VU1Interpreter split;
            single.start(0u, 0x123u, 0x2ABu,
                         &singleFixture.mem);
            split.start(0u, 0x123u, 0x2ABu,
                        &splitFixture.mem);

            const VU1AdvanceResult singleResult =
                single.advance(
                    singleFixture.code, PS2_VU1_CODE_SIZE,
                    singleFixture.data, PS2_VU1_DATA_SIZE,
                    singleFixture.gs, &singleFixture.mem,
                    12u);
            uint32_t splitExecuted = 0u;
            for (const uint32_t budget :
                 std::array<uint32_t, 4>{1u, 2u, 2u, 7u})
            {
                const VU1AdvanceResult result =
                    split.advance(
                        splitFixture.code, PS2_VU1_CODE_SIZE,
                        splitFixture.data, PS2_VU1_DATA_SIZE,
                        splitFixture.gs, &splitFixture.mem,
                        budget);
                splitExecuted += result.executedCycles;
            }

            t.Equals(singleResult.requestedCycles, 12u,
                     "single advance should retain its supplied budget");
            t.Equals(singleResult.executedCycles, 5u,
                     "single advance should report exact E-bit work");
            t.IsTrue(singleResult.completed,
                     "single advance should report completion");
            t.Equals(splitExecuted,
                     singleResult.executedCycles,
                     "split calls should execute the same total work");
            t.IsFalse(single.isActive(),
                      "single-batch interpreter should finish");
            t.IsFalse(split.isActive(),
                      "split-batch interpreter should finish");

            const VU1State &singleState = single.state();
            const VU1State &splitState = split.state();
            t.Equals(splitState.pc, singleState.pc,
                     "split execution should retain the same PC");
            t.Equals(splitState.top, singleState.top,
                     "split execution should retain the same TOP");
            t.Equals(splitState.itop, singleState.itop,
                     "split execution should retain the same ITOP");
            t.Equals(splitState.vi[1], singleState.vi[1],
                     "split execution should retain delay-slot VI state");
            t.Equals(splitState.vi[2], singleState.vi[2],
                     "split execution should retain branch-target VI state");
            t.Equals(splitState.vi[3], singleState.vi[3],
                     "split execution should skip the same fallthrough pair");
            t.Equals(splitState.mac, singleState.mac,
                     "split execution should retain FMAC state");
            t.Equals(splitState.status, singleState.status,
                     "split execution should retain STATUS state");
            t.Equals(splitState.q, singleState.q,
                     "split execution should retain Q state");
            t.Equals(splitState.branchPending,
                     singleState.branchPending,
                     "split execution should retain branch pipeline state");
            t.Equals(splitPackets.size(), singlePackets.size(),
                     "split execution should emit the same packet count");
            if (!singlePackets.empty() && !splitPackets.empty())
            {
                t.Equals(splitPackets[0].size(),
                         singlePackets[0].size(),
                         "split execution should emit the same packet size");
                t.IsTrue(
                    splitPackets[0] == singlePackets[0],
                    "split execution should emit identical PATH1 bytes");
            }
        });

        tc.Run("XGKICK preserves non-EOP tags until PATH1 reaches EOP", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            std::vector<std::vector<uint8_t>> captured;
            fx.mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            const uint64_t firstTag = makeGifTag(0u, GIF_FMT_PACKED, 1u, false);
            const uint64_t lastTag = makeGifTag(0u, GIF_FMT_PACKED, 1u, true);
            std::memcpy(fx.data + 0u, &firstTag, sizeof(firstTag));
            std::memcpy(fx.data + 16u, &lastTag, sizeof(lastTag));
            writeVuInstructionPair(
                fx.code, 0u, makeVuLowerSpecial(0x6Cu, 0u), kVuUpperNop);

            VU1Interpreter vu1;
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 4u);

            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "PATH1 should emit once when the second tag supplies EOP");
            if (!captured.empty())
                t.Equals(captured[0].size(), static_cast<size_t>(32u),
                         "PATH1 packet should include both GIFtags");
        });

        tc.Run("XGKICK rejects a GIFtag spanning the entire VU1 data ring", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            size_t callbackCount = 0u;
            fx.mem.setGifPacketCallback([&](const uint8_t *, uint32_t)
            {
                ++callbackCount;
            });

            const uint64_t ringSizedTag =
                makeGifTag((PS2_VU1_DATA_SIZE / 16u) - 1u,
                           GIF_FMT_IMAGE, 0u, true);
            std::memcpy(fx.data, &ringSizedTag, sizeof(ringSizedTag));
            writeVuInstructionPair(
                fx.code, 0u, makeVuLowerSpecial(0x6Cu, 0u), kVuUpperEnd);

            VU1Interpreter vu1;
            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 2u);

            t.Equals(callbackCount, static_cast<size_t>(0u),
                     "invalid ring-sized PATH1 tag must not be submitted");
        });

        tc.Run("MSCAL can start a VU1 XGKICK program and update GS VRAM", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS gs;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                gs.processGIFPacket(data, sizeBytes);
            });
            mem.setGifArbiter(&arbiter);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(0u) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            uint8_t *vuCode = mem.getVU1Code();
            uint8_t *vuData = mem.getVU1Data();
            std::memset(vuCode, 0, PS2_VU1_CODE_SIZE);
            std::memset(vuData, 0, PS2_VU1_DATA_SIZE);

            const uint32_t lower = makeVuLowerSpecial(0x6Cu, 0u);
            std::memcpy(vuCode + 0u, &lower, sizeof(lower));
            const uint32_t upper = kVuUpperEnd;
            std::memcpy(vuCode + 4u, &upper, sizeof(upper));

            const uint64_t gifTag = makeGifTag(1u, GIF_FMT_IMAGE, 0u, true);
            std::memcpy(vuData + 0u, &gifTag, sizeof(gifTag));
            const uint64_t tagHi = 0u;
            std::memcpy(vuData + 8u, &tagHi, sizeof(tagHi));
            for (uint32_t i = 0; i < 16u; ++i)
            {
                vuData[16u + i] = static_cast<uint8_t>(0x90u + i);
            }

            VU1Interpreter vu1;
            mem.setVu1MscalCallback([&](uint32_t startPC, uint32_t top, uint32_t itop)
            {
                vu1.execute(vuCode,
                            PS2_VU1_CODE_SIZE,
                            vuData,
                            PS2_VU1_DATA_SIZE,
                            gs,
                            &mem,
                            startPC,
                            top,
                            itop,
                            2u);
            });

            const uint32_t mscalCmd = makeVifCmd(0x14u, 0u, 0u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscalCmd), sizeof(mscalCmd));

            const uint8_t *vramOut = mem.getGSVRAM();
            bool imageOk = true;
            for (uint32_t x = 0; x < 4u && imageOk; ++x)
            {
                const uint32_t off = GSPSMCT32::addrPSMCT32(0u, 1u, x, 0u);
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if (vramOut[off + c] != static_cast<uint8_t>(0x90u + x * 4u + c))
                    {
                        imageOk = false;
                        break;
                    }
                }
            }
            t.IsTrue(imageOk, "MSCAL-triggered XGKICK should route PATH1 packet into GS VRAM");
        });
    });
}
