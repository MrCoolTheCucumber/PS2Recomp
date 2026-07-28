#include "MiniTest.h"
#include "ps2_runtime.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_psmct32.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_clip.h"
#include "runtime/ps2_vu_ir.h"
#include "runtime/ps2_vu1.h"

#include <array>
#include <cfenv>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kVuUpperNop = 0x000002FFu;
    constexpr uint32_t kVuUpperEnd = 1u << 30u;

    class ScopedEnvironmentVariable
    {
    public:
        explicit ScopedEnvironmentVariable(const char *name)
            : m_name(name)
        {
            if (const char *const value = std::getenv(name))
                m_original = value;
        }

        ~ScopedEnvironmentVariable()
        {
            if (m_original)
                setenv(m_name.c_str(), m_original->c_str(), 1);
            else
                unsetenv(m_name.c_str());
        }

        void set(const char *value)
        {
            setenv(m_name.c_str(), value, 1);
        }

        ScopedEnvironmentVariable(
            const ScopedEnvironmentVariable &) = delete;
        ScopedEnvironmentVariable &operator=(
            const ScopedEnvironmentVariable &) = delete;

    private:
        std::string m_name;
        std::optional<std::string> m_original;
    };

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

    uint32_t makeVuMfp(uint8_t targetVf, uint8_t dest)
    {
        return makeVuLowerSpecial(0x64u, 0u, targetVf, 0u, dest);
    }

    uint32_t makeVuEleng(uint8_t fs)
    {
        return makeVuLowerSpecial(0x72u, fs);
    }

    uint32_t makeVuErcpr(uint8_t fs, uint8_t fsf)
    {
        return makeVuLowerSpecial(
            0x7Au, fs, 0u, 0u,
            static_cast<uint8_t>(fsf & 3u));
    }

    uint32_t makeVuWaitP()
    {
        return makeVuLowerSpecial(0x7Bu, 0u);
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

    constexpr size_t kVuIrOpcodeCount =
        static_cast<size_t>(VuIrOpcode::Unsupported) + 1u;

    struct VuDifferentialResult
    {
        bool matched = false;
        uint32_t executedPairs = 0u;
        std::string failure;
        VuExecutionState finalState{};
        std::vector<uint8_t> finalData;
        std::vector<std::vector<uint8_t>> path1Packets;
        std::array<uint64_t, kVuIrOpcodeCount> opcodeCounts{};
    };

    VuDifferentialResult runVuDifferential(
        const std::vector<uint8_t> &code,
        const std::vector<uint8_t> &initialData,
        const VuExecutionState &initialState,
        uint32_t maximumPairs)
    {
        VuDifferentialResult outcome;
        VuExecutionState referenceState = initialState;
        VuExecutionState irState = initialState;
        std::vector<uint8_t> referenceData = initialData;
        std::vector<uint8_t> irData = initialData;
        VuTransactionalSideEffectSink referenceEffects;
        VuTransactionalSideEffectSink irEffects;
        VuUnit referenceUnit;
        VuUnit irUnit;
        VuInterpreterBackend reference(referenceUnit);
        VuIrInterpreterBackend ir(irUnit);
        VuExecutionContext referenceContext{
            .state = referenceState,
            .code = code.data(),
            .codeSize = static_cast<uint32_t>(code.size()),
            .data = referenceData.data(),
            .dataSize = static_cast<uint32_t>(referenceData.size()),
            .sideEffects = referenceEffects,
            .enableInstrumentation = false,
        };
        VuExecutionContext irContext{
            .state = irState,
            .code = code.data(),
            .codeSize = static_cast<uint32_t>(code.size()),
            .data = irData.data(),
            .dataSize = static_cast<uint32_t>(irData.size()),
            .sideEffects = irEffects,
            .enableInstrumentation = false,
        };

        for (uint32_t pairIndex = 0u;
             pairIndex < maximumPairs; ++pairIndex)
        {
            const uint32_t issuedPc = referenceState.pc;
            uint32_t lowerWord = 0u;
            uint32_t upperWord = 0u;
            if (issuedPc + 8u <= code.size())
            {
                std::memcpy(
                    &lowerWord, code.data() + issuedPc,
                    sizeof(lowerWord));
                std::memcpy(
                    &upperWord, code.data() + issuedPc + 4u,
                    sizeof(upperWord));
                const VuIrInstructionPair pair =
                    decodeVuIrInstructionPair(
                        issuedPc, lowerWord, upperWord);
                ++outcome.opcodeCounts[
                    static_cast<size_t>(pair.upper.opcode)];
                ++outcome.opcodeCounts[
                    static_cast<size_t>(pair.lower.opcode)];
            }

            const VuRunResult referenceResult =
                reference.run(referenceContext, 1u);
            const VuRunResult irResult = ir.run(irContext, 1u);
            ++outcome.executedPairs;

            const bool resultsMatch =
                referenceResult.requestedCycles ==
                    irResult.requestedCycles &&
                referenceResult.executedCycles ==
                    irResult.executedCycles &&
                referenceResult.reason == irResult.reason &&
                referenceResult.activeBefore ==
                    irResult.activeBefore &&
                referenceResult.activeAfter == irResult.activeAfter &&
                referenceResult.completed == irResult.completed;
            std::string stateDifference;
            const bool statesMatch = vuExecutionStatesEqual(
                referenceState, irState, &stateDifference);
            const bool dataMatches = referenceData == irData;
            const bool effectsMatch =
                referenceEffects.path1Packets() ==
                irEffects.path1Packets();
            if (!resultsMatch || !statesMatch ||
                !dataMatches || !effectsMatch)
            {
                std::ostringstream message;
                message
                    << "pair " << pairIndex
                    << " pc " << issuedPc
                    << " lower " << lowerWord
                    << " upper " << upperWord;
                if (!resultsMatch)
                {
                    message
                        << " result "
                        << vuExitReasonName(referenceResult.reason)
                        << "/"
                        << vuExitReasonName(irResult.reason);
                }
                if (!statesMatch)
                    message << " state " << stateDifference;
                if (!dataMatches)
                    message << " VU data";
                if (!effectsMatch)
                    message << " PATH1 effects";
                outcome.failure = message.str();
                break;
            }
            if (!referenceState.active)
            {
                outcome.matched = true;
                break;
            }
        }

        if (outcome.failure.empty())
            outcome.matched = true;
        outcome.finalState = std::move(referenceState);
        outcome.finalData = std::move(referenceData);
        outcome.path1Packets = referenceEffects.path1Packets();
        return outcome;
    }

    class FixedVuGenerator
    {
    public:
        explicit FixedVuGenerator(uint32_t seed)
            : m_state(seed)
        {
        }

        uint32_t next()
        {
            m_state ^= m_state << 13u;
            m_state ^= m_state >> 17u;
            m_state ^= m_state << 5u;
            return m_state;
        }

        uint8_t registerIndex()
        {
            return static_cast<uint8_t>(2u + next() % 14u);
        }

    private:
        uint32_t m_state;
    };

    uint32_t generatedUpper(FixedVuGenerator &generator)
    {
        const uint8_t ft = generator.registerIndex();
        const uint8_t fs = generator.registerIndex();
        const uint8_t fd = generator.registerIndex();
        switch (generator.next() % 10u)
        {
        case 0u:
            return kVuUpperNop;
        case 1u:
            return makeVuUpper(0x28u, 0xfu, ft, fs, fd);
        case 2u:
            return makeVuUpper(0x2au, 0xfu, ft, fs, fd);
        case 3u:
            return makeVuUpper(
                static_cast<uint8_t>(generator.next() & 3u),
                0xfu, ft, fs, fd);
        case 4u:
            return makeVuUpper(
                static_cast<uint8_t>(0x10u |
                                     (generator.next() & 3u)),
                0xfu, ft, fs, fd);
        case 5u:
            return makeVuUpper(0x1cu, 0xfu, ft, fs, fd);
        case 6u:
            return makeVuUpperSpecial(0x10u, 0xfu, ft, fs);
        case 7u:
            return makeVuUpperSpecial(0x15u, 0xfu, ft, fs);
        case 8u:
            return makeVuUpperSpecial(0x1fu, 0xfu, ft, fs);
        default:
            return makeVuUpper(0x22u, 0xfu, ft, fs, fd) |
                   0x80000000u;
        }
    }

    uint32_t generatedLower(FixedVuGenerator &generator)
    {
        const uint8_t first = generator.registerIndex();
        const uint8_t second = generator.registerIndex();
        const uint8_t third = generator.registerIndex();
        switch (generator.next() % 10u)
        {
        case 0u:
            return 0u;
        case 1u:
            return makeVuLq(0xfu, first, second, 0);
        case 2u:
            return makeVuSq(0xfu, first, second, 0);
        case 3u:
            return makeVuIaddiu(
                first, second,
                static_cast<int16_t>(generator.next() & 7u));
        case 4u:
            return makeVuLowerDirect(
                0x30u, first, second, third);
        case 5u:
            return makeVuLowerSpecial(
                0x30u, first, second, 0u, 0xfu);
        case 6u:
            return makeVuLowerSpecial(
                0x34u, first, second, 0u, 0xfu);
        case 7u:
            return makeVuMtir(
                first, second,
                static_cast<uint8_t>(generator.next() & 3u));
        case 8u:
            return makeVuDiv(
                first, second,
                static_cast<uint8_t>(generator.next() & 3u),
                static_cast<uint8_t>(generator.next() & 3u));
        default:
            return makeVuLowerSpecial(
                0x3du, first, second, 0u, 0xfu);
        }
    }
}

void register_ps2_vu1_tests()
{
    MiniTest::Case("PS2VU1", [](TestCase &tc)
    {
        tc.Run("execution state clones every delayed pipeline", [](TestCase &t)
        {
            VuExecutionState original{};
            original.pc = 0x138u;
            original.branchPending = true;
            original.branchTarget = 0x280u;
            original.branchDelay = 1u;
            original.active = true;
            original.issuedCycles = 27u;
            original.pipeline.xgkick.active = true;
            original.pipeline.xgkick.address = 0x40u;
            original.pipeline.xgkick.tagBytesRemaining = 16u;
            original.pipeline.xgkick.packet = {0x11u, 0x22u, 0x33u};
            original.pipeline.delayedQ = {
                .active = true,
                .result = 3.5f,
                .cyclesRemaining = 5u,
            };
            original.pipeline.delayedP = {
                .active = true,
                .result = 7.25f,
                .cyclesRemaining = 11u,
            };
            original.pipeline.fmacFlags[2] = {
                .active = true,
                .mac = 0x1234u,
                .status = 0x5u,
            };
            original.pipeline.fmacFlagIndex = 1u;
            original.pipeline.workingMac = 0xabcd;

            VuExecutionState clone = original;
            original.pipeline.xgkick.packet[0] = 0xffu;
            original.pipeline.delayedQ.result = -1.0f;
            original.pipeline.delayedP.result = -2.0f;
            original.pipeline.fmacFlags[2].mac = 0u;

            t.Equals(clone.pc, uint32_t{0x138u},
                     "the architectural state should be copied");
            t.IsTrue(clone.branchPending,
                     "the branch delay state should be copied");
            t.IsTrue(clone.active,
                     "the active state should be copied");
            t.Equals(clone.issuedCycles, uint64_t{27u},
                     "issued-cycle bookkeeping should be copied");
            t.Equals(clone.pipeline.xgkick.packet[0], uint8_t{0x11u},
                     "XGKICK packet storage should be deep-copied");
            t.Equals(clone.pipeline.delayedQ.result, 3.5f,
                     "the delayed Q result should be copied");
            t.Equals(clone.pipeline.delayedP.result, 7.25f,
                     "the delayed P result should be copied");
            t.Equals(clone.pipeline.fmacFlags[2].mac, uint32_t{0x1234u},
                     "the fixed FMAC pipeline should be copied");
            t.Equals(clone.pipeline.fmacFlagIndex, uint8_t{1u},
                     "the FMAC pipeline cursor should be copied");
            t.Equals(clone.pipeline.workingMac, uint32_t{0xabcdu},
                     "the pending MAC accumulator should be copied");
        });

        tc.Run("unit reports backend-neutral execution lifecycle", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, makeVuIaddiu(1u, 0u, 3), kVuUpperEnd);
            writeVuInstructionPair(
                fx.code, 8u, makeVuIaddiu(2u, 0u, 5), kVuUpperNop);

            VuUnit unit;
            t.IsFalse(unit.isActive(),
                      "reset state should be inactive");
            t.Equals(unit.state().q, 1.0f,
                     "reset should restore Q");
            t.Equals(unit.state().vf[0][3], 1.0f,
                     "reset should restore VF0.w");

            const VuRunResult inactive = unit.advance(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 1u);
            t.IsTrue(inactive.reason == VuExitReason::Inactive,
                     "advancing an idle unit should report inactive");
            t.Equals(inactive.executedCycles, uint32_t{0u},
                     "an idle unit should execute no pairs");

            std::vector<uint32_t> observedPcs;
            unit.setInstructionObserver(
                [&](uint64_t, uint32_t pc, uint32_t, uint32_t,
                    const VuExecutionState &)
                {
                    observedPcs.push_back(pc);
                });
            unit.setProgressTrackingEnabled(true);
            unit.start(0u, 7u, 9u, &fx.mem);
            t.IsTrue(unit.isActive(),
                     "start should make the unit active");
            t.Equals(unit.state().top, uint32_t{7u},
                     "start should publish TOP");
            t.Equals(unit.state().itop, uint32_t{9u},
                     "start should publish ITOP");

            const VuRunResult budget = unit.advance(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 1u);
            t.IsTrue(budget.reason == VuExitReason::CycleBudget,
                     "a bounded active run should report its cycle budget");
            t.Equals(budget.executedCycles, uint32_t{1u},
                     "the first budget should issue one pair");
            t.IsTrue(budget.activeAfter,
                     "the E-bit delay pair should remain active");
            t.Equals(unit.state().issuedCycles, uint64_t{1u},
                     "issued cycles should be exact after a side exit");

            const VuRunResult completed = unit.advance(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 4u);
            t.IsTrue(completed.reason == VuExitReason::ProgramEnded,
                     "the E-bit delay pair should report completion");
            t.Equals(completed.executedCycles, uint32_t{1u},
                     "completion should execute the one remaining pair");
            t.IsTrue(completed.completed && !completed.activeAfter,
                     "completion should make the unit idle");
            t.Equals(unit.state().issuedCycles, uint64_t{2u},
                     "completion should retain total issued cycles");
            t.Equals(observedPcs.size(), size_t{2u},
                     "the observer should run once before every pair");
            if (observedPcs.size() == 2u)
            {
                t.Equals(observedPcs[0], uint32_t{0u},
                         "the observer should see the entry PC");
                t.Equals(observedPcs[1], uint32_t{8u},
                         "the observer should see the E-bit delay PC");
            }

            const VuProgressSnapshot progress =
                unit.getProgressSnapshot();
            t.IsTrue(progress.enabled,
                     "progress tracking should remain enabled");
            t.IsFalse(progress.active,
                      "completed progress should not report a live host run");
            t.Equals(progress.invocations, uint64_t{2u},
                     "both bounded calls should be counted");
            t.Equals(progress.cycles, uint64_t{2u},
                     "progress should report exact issued pairs");
            t.Equals(progress.pc, uint32_t{16u},
                     "progress should report the completion PC");

            unit.reset();
            t.IsFalse(unit.isActive(),
                      "reset should cancel execution");
            t.Equals(unit.state().issuedCycles, uint64_t{0u},
                     "reset should clear issued-cycle bookkeeping");
            unit.state().pc = 8u;
            unit.state().vi[3] = 0x44;
            unit.state().pipeline.delayedQ = {
                .active = true,
                .result = 2.0f,
                .cyclesRemaining = 3u,
            };
            unit.resumeState(11u, 13u, &fx.mem);
            t.IsTrue(unit.isActive(),
                     "resume should make retained state active");
            t.Equals(unit.state().pc, uint32_t{8u},
                     "resume should retain the current PC");
            t.Equals(unit.state().vi[3], int32_t{0x44},
                     "resume should retain architectural registers");
            t.IsTrue(unit.state().pipeline.delayedQ.active,
                     "resume should retain delayed pipeline state");
            t.Equals(unit.state().top, uint32_t{11u},
                     "resume should update TOP");
            t.Equals(unit.state().itop, uint32_t{13u},
                     "resume should update ITOP");
        });

        tc.Run("interpreter backend executes a supplied clone", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU fixture should initialize");
            writeVuInstructionPair(
                fx.code, 0u, makeVuIaddiu(1u, 0u, 7), kVuUpperEnd);

            struct NoopSideEffectSink final : IVuSideEffectSink
            {
                void submitPath1Packet(
                    const uint8_t *, uint32_t) override
                {
                }
            } sideEffects;

            VuUnit unit;
            VuExecutionState clone = unit.state();
            clone.active = true;
            VuExecutionContext context{
                .state = clone,
                .code = fx.code,
                .codeSize = PS2_VU1_CODE_SIZE,
                .data = fx.data,
                .dataSize = PS2_VU1_DATA_SIZE,
                .sideEffects = sideEffects,
                .memory = &fx.mem,
                .enableInstrumentation = false,
            };
            VuInterpreterBackend backend(unit);
            const VuRunResult result = backend.run(context, 1u);

            t.IsTrue(result.reason == VuExitReason::CycleBudget,
                     "the clone should stop at the requested pair budget");
            t.Equals(result.executedCycles, uint32_t{1u},
                     "the backend should report exact clone work");
            t.Equals(clone.vi[1], int32_t{7},
                     "the supplied clone should receive architectural writes");
            t.Equals(clone.issuedCycles, uint64_t{1u},
                     "the supplied clone should receive cycle bookkeeping");
            t.IsFalse(unit.isActive(),
                      "backend execution should not activate canonical state");
            t.Equals(unit.state().vi[1], int32_t{0},
                     "backend execution should not mutate canonical registers");
        });

        tc.Run("backend requests are independent and reject active switches", [](TestCase &t)
        {
            VuBackendKind parsed = VuBackendKind::Auto;
            t.IsTrue(
                parseVuBackendKind("verify", parsed) &&
                    parsed == VuBackendKind::Verify,
                "backend names should parse");
            t.IsFalse(parseVuBackendKind("jit", parsed),
                      "unknown backend names should be rejected");

            PS2RuntimeConfiguration configuration{};
            configuration.vu0Backend = VuBackendKind::Recompiler;
            configuration.vu1Backend = VuBackendKind::Verify;
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            t.IsTrue(
                runtime.vu0().requestedBackend() ==
                    VuBackendKind::Recompiler,
                "VU0 should retain its independent request");
            t.IsTrue(
                runtime.vu1().requestedBackend() ==
                    VuBackendKind::Verify,
                "VU1 should retain its independent request");
            t.IsTrue(
                runtime.vu0().resolvedBackend() ==
                        VuBackendKind::Interpreter &&
                    runtime.vu1().resolvedBackend() ==
                        VuBackendKind::Interpreter,
                "Phase 1 modes should resolve through the interpreter");

            VuUnit unit;
            unit.start();
            std::string diagnostic;
            t.IsFalse(
                unit.setBackend(
                    VuBackendKind::Recompiler, &diagnostic),
                "an active unit should reject a backend change");
            t.IsFalse(diagnostic.empty(),
                      "a rejected switch should explain why");
            t.IsTrue(
                unit.requestedBackend() == VuBackendKind::Auto,
                "a rejected switch should preserve the request");

            unit.reset();
            t.IsTrue(
                unit.setBackend(
                    VuBackendKind::Recompiler, &diagnostic),
                "an idle unit should accept a backend request");
            t.IsTrue(diagnostic.empty(),
                     "an accepted request should clear the diagnostic");
            t.IsTrue(
                unit.requestedBackend() ==
                        VuBackendKind::Recompiler &&
                    unit.resolvedBackend() ==
                        VuBackendKind::Interpreter &&
                    unit.backendName() == "interpreter",
                "request and resolved backend status should remain explicit");
        });

        tc.Run("runtime backend environment overrides each unit", [](TestCase &t)
        {
            ScopedEnvironmentVariable vu0("PS2X_VU0_BACKEND");
            ScopedEnvironmentVariable vu1("PS2X_VU1_BACKEND");
            vu0.set("interpreter");
            vu1.set("verify");

            PS2Runtime runtime;
            t.IsTrue(
                runtime.vu0().requestedBackend() ==
                    VuBackendKind::Interpreter,
                "the VU0 environment request should be applied");
            t.IsTrue(
                runtime.vu1().requestedBackend() ==
                    VuBackendKind::Verify,
                "the VU1 environment request should be applied independently");

            vu1.set("jit");
            bool rejected = false;
            try
            {
                PS2Runtime invalid;
            }
            catch (const std::invalid_argument &error)
            {
                rejected =
                    std::string(error.what()).find(
                        "PS2X_VU1_BACKEND") != std::string::npos;
            }
            t.IsTrue(
                rejected,
                "an invalid environment request should name the failing unit");
        });

        tc.Run("instruction-pair IR retains hazards and pipeline effects", [](TestCase &t)
        {
            const uint32_t upper =
                makeVuUpper(0x28u, 0xfu, 2u, 1u, 3u);
            const uint32_t lower = makeVuSq(0xfu, 3u, 1u, 0);
            const VuIrInstructionPair pair =
                decodeVuIrInstructionPair(0x120u, lower, upper);

            t.IsTrue(
                pair.upper.opcode == VuIrOpcode::UpperAdd,
                "the upper operation should have a semantic opcode");
            t.IsTrue(
                pair.lower.opcode == VuIrOpcode::LowerSq,
                "the lower operation should have a semantic opcode");
            t.IsTrue(
                (pair.upper.vfReadMask & ((1u << 1u) | (1u << 2u))) ==
                    ((1u << 1u) | (1u << 2u)),
                "upper VF reads should be explicit");
            t.IsTrue(
                (pair.upper.vfWriteMask & (1u << 3u)) != 0u &&
                    (pair.lower.vfReadMask & (1u << 3u)) != 0u,
                "the cross-lane VF hazard should be explicit");
            t.IsTrue(
                pair.order == VuIrPairOrder::LowerThenUpper,
                "the lower reader should observe the pre-upper value");
            t.IsTrue(
                vuIrHasOpFlag(pair.upper, VuIrOpFmac) &&
                    pair.upper.latency == 4u,
                "FMAC pipeline changes should carry their latency");
            t.IsTrue(
                vuIrHasOpFlag(pair.lower, VuIrOpWritesVuData),
                "VU data writes should be explicit");
            t.IsTrue(
                vuIrHasPairFlag(pair, VuIrPairAdvancesPipelines),
                "each issued pair should advance retained pipelines");

            VuIrVerificationError error;
            t.IsTrue(
                verifyVuIrInstructionPair(pair, &error),
                "a decoded pair should pass the IR verifier");
            VuIrInstructionPair malformed = pair;
            malformed.cycles = 2u;
            t.IsFalse(
                verifyVuIrInstructionPair(malformed, &error),
                "a malformed cycle cost should fail verification");
            t.Equals(error.pc, uint32_t{0x120u},
                     "verification diagnostics should retain the VU PC");
            t.IsTrue(!error.message.empty(),
                     "verification diagnostics should explain the failure");
        });

        tc.Run("instruction-pair IR exposes immediates latency and unsupported ops", [](TestCase &t)
        {
            const VuIrInstructionPair immediate =
                decodeVuIrInstructionPair(
                    0u, makeVuBranch(2),
                    makeVuUpper(0x22u, 0xfu, 0u, 1u, 2u) |
                        0x80000000u);
            t.IsTrue(
                immediate.order == VuIrPairOrder::UpperThenLoi &&
                    immediate.lower.opcode == VuIrOpcode::Loi &&
                    vuIrHasPairFlag(immediate, VuIrPairImmediate),
                "the I bit should turn the lower word into LOI");
            t.IsFalse(
                vuIrHasPairFlag(immediate, VuIrPairBranch),
                "LOI data must not retain a decoded lower branch");
            t.IsTrue(
                vuIrHasOpFlag(immediate.lower, VuIrOpWritesI),
                "LOI should expose its scalar write");

            const VuIrInstructionPair qPair =
                decodeVuIrInstructionPair(
                    8u, makeVuDiv(1u, 2u, 0u, 1u),
                    makeVuUpper(0x1cu, 0x8u, 0u, 3u, 4u));
            t.IsTrue(
                qPair.lower.opcode == VuIrOpcode::LowerDiv &&
                    qPair.lower.latency == 7u &&
                    vuIrHasOpFlag(qPair.lower, VuIrOpWritesQ) &&
                    vuIrHasOpFlag(qPair.lower, VuIrOpQBarrier),
                "DIV should expose delayed Q production");
            t.IsTrue(
                qPair.upper.opcode == VuIrOpcode::UpperMulQ &&
                    vuIrHasOpFlag(qPair.upper, VuIrOpReadsQ),
                "MULq should expose Q consumption");

            const VuIrInstructionPair unsupported =
                decodeVuIrInstructionPair(
                    16u, 0x04000000u,
                    makeVuUpper(0x34u, 0xfu, 2u, 1u, 3u));
            t.IsTrue(
                unsupported.upper.opcode == VuIrOpcode::Unsupported &&
                    unsupported.lower.opcode == VuIrOpcode::Unsupported &&
                    vuIrHasPairFlag(
                        unsupported, VuIrPairUnsupported),
                "unimplemented encodings should remain explicit");
            t.Equals(
                std::string(vuIrOpcodeName(unsupported.upper.opcode)),
                std::string("Unsupported"),
                "unsupported diagnostics should have a stable name");
        });

        tc.Run("lower IR distinguishes masks selectors and P barriers", [](TestCase &t)
        {
            const VuIrOperation masked =
                decodeVuIrInstructionPair(
                    0u, makeVuMfp(2u, 0xau),
                    kVuUpperNop).lower;
            t.Equals(
                masked.destinationMask, uint8_t{0xau},
                "MFP should retain its destination mask");
            t.Equals(
                masked.selector, uint8_t{0u},
                "a destination mask must not become a scalar selector");
            t.IsTrue(
                vuIrHasOpFlag(masked, VuIrOpReadsP) &&
                    !vuIrHasOpFlag(masked, VuIrOpPBarrier),
                "MFP should read old P without an automatic dependency");

            const VuIrOperation div =
                decodeVuIrInstructionPair(
                    8u, makeVuDiv(1u, 2u, 2u, 3u),
                    kVuUpperNop).lower;
            t.Equals(
                div.destinationMask, uint8_t{0u},
                "DIV component fields are not a destination mask");
            t.Equals(
                div.selector, uint8_t{0xeu},
                "DIV should retain packed fsf and ftf selectors");

            const VuIrOperation mtir =
                decodeVuIrInstructionPair(
                    16u, makeVuMtir(3u, 4u, 2u),
                    kVuUpperNop).lower;
            t.Equals(
                mtir.destinationMask, uint8_t{0u},
                "MTIR's component field is not a destination mask");
            t.Equals(
                mtir.selector, uint8_t{2u},
                "MTIR should expose its fsf selector");

            const VuIrOperation ercpr =
                decodeVuIrInstructionPair(
                    24u, makeVuErcpr(5u, 1u),
                    kVuUpperNop).lower;
            t.IsTrue(
                ercpr.opcode == VuIrOpcode::LowerErcpr &&
                    ercpr.latency == 12u &&
                    vuIrHasOpFlag(ercpr, VuIrOpWritesP) &&
                    vuIrHasOpFlag(ercpr, VuIrOpPBarrier),
                "ERCPR should expose delayed P production");
            t.Equals(
                ercpr.selector, uint8_t{1u},
                "ERCPR should expose its fsf selector");

            const VuIrOperation eleng =
                decodeVuIrInstructionPair(
                    32u, makeVuEleng(6u),
                    kVuUpperNop).lower;
            t.IsTrue(
                eleng.latency == 18u &&
                    vuIrHasOpFlag(eleng, VuIrOpWritesP) &&
                    vuIrHasOpFlag(eleng, VuIrOpPBarrier),
                "ELENG should expose its documented P latency");

            const VuIrOperation esadd =
                decodeVuIrInstructionPair(
                    40u, makeVuLowerSpecial(0x70u, 7u),
                    kVuUpperNop).lower;
            t.IsTrue(
                esadd.opcode == VuIrOpcode::LowerEsadd &&
                    esadd.latency == 11u &&
                    vuIrHasOpFlag(esadd, VuIrOpUnsupported) &&
                    vuIrHasOpFlag(esadd, VuIrOpWritesP) &&
                    vuIrHasOpFlag(esadd, VuIrOpPBarrier) &&
                    (esadd.vfReadMask & (1u << 7u)) != 0u,
                "recognized EFU placeholders should retain full metadata");

            const VuIrOperation eatan =
                decodeVuIrInstructionPair(
                    48u,
                    makeVuLowerSpecial(
                        0x7du, 8u, 0u, 0u, 3u),
                    kVuUpperNop).lower;
            t.IsTrue(
                eatan.opcode == VuIrOpcode::LowerEatan &&
                    eatan.selector == 3u &&
                    eatan.latency == 54u &&
                    vuIrHasOpFlag(eatan, VuIrOpUnsupported) &&
                    vuIrHasOpFlag(eatan, VuIrOpPBarrier),
                "EATAN should retain its selector and documented latency");

            const uint32_t wideImmediate =
                (0x08u << 25u) | (0xau << 21u) |
                (2u << 16u) | (1u << 11u) | 3u;
            const VuIrOperation iaddiu =
                decodeVuIrInstructionPair(
                    56u, wideImmediate, kVuUpperNop).lower;
            t.Equals(
                iaddiu.destinationMask, uint8_t{0u},
                "IADDIU immediate bits must not masquerade as a mask");

            const VuIrOperation upperMadd =
                decodeVuIrInstructionPair(
                    64u, 0u,
                    makeVuUpper(
                        0x29u, 0xfu, 2u, 1u, 3u)).upper;
            const VuIrOperation upperBroadcast =
                decodeVuIrInstructionPair(
                    72u, 0u,
                    makeVuUpper(
                        0x0bu, 0xfu, 2u, 1u, 3u)).upper;
            t.Equals(
                upperMadd.selector, uint8_t{0u},
                "non-broadcast upper opcodes should have no selector");
            t.Equals(
                upperBroadcast.selector, uint8_t{3u},
                "broadcast upper opcodes should retain their component");
        });

        tc.Run("VU IR blocks retain delay slots and side-exit reasons", [](TestCase &t)
        {
            std::array<uint8_t, 64> code{};
            writeVuInstructionPair(
                code.data(), 0u, makeVuBranch(2), kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 8u, makeVuIaddiu(1u, 0u, 1),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 16u, 0u, kVuUpperNop);

            const VuIrBlock branch = decodeVuIrBlock(
                code.data(), static_cast<uint32_t>(code.size()),
                0u, 8u);
            t.Equals(branch.pairs.size(), size_t{2u},
                     "a branch block should include its delay pair");
            t.IsTrue(
                branch.exit == VuIrBlockExit::BranchBoundary,
                "the block should leave at a resumable branch boundary");
            t.IsTrue(
                verifyVuIrBlock(branch),
                "the branch block should pass structural verification");
            VuIrBlock malformedExit = branch;
            malformedExit.exit = VuIrBlockExit::PairLimit;
            VuIrVerificationError error;
            t.IsFalse(
                verifyVuIrBlock(malformedExit, &error),
                "a block should reject an inconsistent side exit");
            t.Equals(
                error.lowerWord, branch.pairs.back().lowerWord,
                "block verification should report terminal words");

            writeVuInstructionPair(
                code.data(), 0u, 0u, kVuUpperEnd);
            const VuIrBlock ending = decodeVuIrBlock(
                code.data(), static_cast<uint32_t>(code.size()),
                0u, 8u);
            t.Equals(ending.pairs.size(), size_t{2u},
                     "an E-bit block should include its delay pair");
            t.IsTrue(
                ending.exit == VuIrBlockExit::ProgramEndBoundary,
                "the E-bit delay should have a structured block exit");

            writeVuInstructionPair(
                code.data(), 0u,
                makeVuLowerSpecial(0x6cu, 1u), kVuUpperNop);
            const VuIrBlock xgkick = decodeVuIrBlock(
                code.data(), static_cast<uint32_t>(code.size()),
                0u, 8u);
            t.Equals(xgkick.pairs.size(), size_t{1u},
                     "XGKICK should form an explicit helper boundary");
            t.IsTrue(
                xgkick.exit == VuIrBlockExit::XgkickBoundary &&
                    vuIrHasOpFlag(
                        xgkick.pairs[0].lower,
                        VuIrOpExternalEffect),
                "XGKICK should retain its external side effect");

            writeVuInstructionPair(
                code.data(), 0u, makeVuBranch(2), kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 8u, 0x04000000u,
                makeVuUpper(0x34u, 0xfu, 2u, 1u, 3u));
            const VuIrBlock unsupportedDelay = decodeVuIrBlock(
                code.data(), static_cast<uint32_t>(code.size()),
                0u, 8u);
            t.Equals(
                unsupportedDelay.pairs.size(), size_t{2u},
                "an unsupported delay pair should be retained for diagnostics");
            t.IsTrue(
                unsupportedDelay.exit ==
                    VuIrBlockExit::UnsupportedInstruction,
                "unsupported semantics should take priority over a later branch exit");
            t.IsTrue(
                verifyVuIrBlock(unsupportedDelay),
                "an explicit unsupported delay exit should verify");

            const VuIrBlock unalignedCode = decodeVuIrBlock(
                code.data(), 63u, 0u, 8u);
            t.IsTrue(
                unalignedCode.pairs.empty() &&
                    unalignedCode.exit == VuIrBlockExit::CodeBounds &&
                    verifyVuIrBlock(unalignedCode),
                "a non-pair-aligned code extent should side-exit safely");

            VuIrBlock malformedEmpty = unalignedCode;
            malformedEmpty.exit = VuIrBlockExit::PairLimit;
            t.IsFalse(
                verifyVuIrBlock(malformedEmpty, &error),
                "an empty block should reject an exit inconsistent with its entry");
        });

        tc.Run("RAC1 steady-workload encodings all map to explicit IR", [](TestCase &t)
        {
            // Exact raw opcode sets from the retained 100,000,000-pair
            // steady-state profile.
            constexpr std::array<uint8_t, 17> upperPrimary = {
                0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u,
                0x06u, 0x07u, 0x08u, 0x09u, 0x0au, 0x0bu,
                0x10u, 0x11u, 0x15u, 0x16u, 0x18u,
            };
            constexpr std::array<uint8_t, 8> upperPrimaryTail = {
                0x1au, 0x1bu, 0x1cu, 0x22u,
                0x28u, 0x2au, 0x2cu, 0x2fu,
            };
            constexpr std::array<uint8_t, 13> upperSpecial = {
                0x08u, 0x09u, 0x0au, 0x10u, 0x12u,
                0x14u, 0x15u, 0x18u, 0x19u, 0x1au,
                0x1bu, 0x1fu, 0x2fu,
            };
            constexpr std::array<uint8_t, 20> lowerPrimary = {
                0x00u, 0x01u, 0x04u, 0x05u, 0x08u,
                0x09u, 0x11u, 0x12u, 0x16u, 0x1cu,
                0x20u, 0x21u, 0x24u, 0x25u, 0x28u,
                0x29u, 0x2cu, 0x2du, 0x2eu, 0x2fu,
            };
            constexpr std::array<uint8_t, 12> lowerSpecial = {
                0x30u, 0x31u, 0x34u, 0x35u,
                0x38u, 0x3bu, 0x3cu, 0x3du,
                0x3eu, 0x3fu, 0x68u, 0x6cu,
            };

            const auto checkUpper = [&](uint32_t word)
            {
                const VuIrInstructionPair pair =
                    decodeVuIrInstructionPair(
                        0u, 0u, word);
                t.IsFalse(
                    vuIrHasOpFlag(
                        pair.upper, VuIrOpUnsupported),
                    "an observed RAC1 upper encoding was unsupported");
            };
            for (const uint8_t opcode : upperPrimary)
            {
                checkUpper(makeVuUpper(
                    opcode, 0xfu, 2u, 1u, 3u));
            }
            for (const uint8_t opcode : upperPrimaryTail)
            {
                checkUpper(makeVuUpper(
                    opcode, 0xfu, 2u, 1u, 3u));
            }
            for (const uint8_t opcode : upperSpecial)
            {
                checkUpper(makeVuUpperSpecial(
                    opcode, 0xfu, 2u, 1u));
            }

            for (const uint8_t opcode : lowerPrimary)
            {
                const uint32_t word =
                    (static_cast<uint32_t>(opcode) << 25u) |
                    (0xfu << 21u) |
                    (2u << 16u) |
                    (3u << 11u) |
                    1u;
                const VuIrInstructionPair pair =
                    decodeVuIrInstructionPair(
                        0u, word, kVuUpperNop);
                t.IsFalse(
                    vuIrHasOpFlag(
                        pair.lower, VuIrOpUnsupported),
                    "an observed RAC1 lower encoding was unsupported");
            }
            for (const uint8_t opcode : lowerSpecial)
            {
                const VuIrInstructionPair pair =
                    decodeVuIrInstructionPair(
                        0u,
                        makeVuLowerSpecial(
                            opcode, 3u, 2u, 1u, 0xfu),
                        kVuUpperNop);
                t.IsFalse(
                    vuIrHasOpFlag(
                        pair.lower, VuIrOpUnsupported),
                    "an observed RAC1 lower-special encoding was unsupported");
            }
        });

        tc.Run("IR oracle matches pair ordering pipelines and completion", [](TestCase &t)
        {
            std::vector<uint8_t> code(8u * 8u);
            std::vector<uint8_t> data(256u);
            const float oldStoredValue[4] = {
                10.0f, 11.0f, 12.0f, 13.0f};
            std::memcpy(
                data.data() + 16u, oldStoredValue,
                sizeof(oldStoredValue));

            float immediate = 5.0f;
            uint32_t immediateBits = 0u;
            std::memcpy(
                &immediateBits, &immediate,
                sizeof(immediateBits));
            writeVuInstructionPair(
                code.data(), 0u, immediateBits,
                makeVuUpper(0x22u, 0xfu, 0u, 1u, 3u) |
                    0x80000000u);
            writeVuInstructionPair(
                code.data(), 8u,
                makeVuDiv(1u, 2u, 0u, 0u),
                makeVuUpper(0x1eu, 0xfu, 0u, 1u, 4u));
            writeVuInstructionPair(
                code.data(), 16u, makeVuWaitQ(),
                makeVuUpper(0x1cu, 0xfu, 0u, 1u, 5u));
            writeVuInstructionPair(
                code.data(), 24u, makeVuSq(0xfu, 4u, 2u, 0),
                makeVuUpper(0x28u, 0xfu, 2u, 1u, 4u));
            writeVuInstructionPair(
                code.data(), 32u, makeVuBranch(1),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 40u, 0u,
                makeVuUpper(0x2au, 0xfu, 2u, 1u, 6u));
            writeVuInstructionPair(
                code.data(), 48u, 0u,
                kVuUpperNop | 0x40000000u);
            writeVuInstructionPair(
                code.data(), 56u, 0u, kVuUpperNop);

            VuUnit seedUnit;
            VuExecutionState initialState = seedUnit.state();
            initialState.active = true;
            initialState.i = 2.0f;
            initialState.vi[2] = 1;
            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                initialState.vf[1][lane] =
                    static_cast<float>(8u + lane);
                initialState.vf[2][lane] =
                    static_cast<float>(2u + lane);
                initialState.vf[4][lane] =
                    oldStoredValue[lane];
            }

            const VuDifferentialResult result =
                runVuDifferential(
                    code, data, initialState, 32u);
            t.IsTrue(
                result.matched,
                "IR oracle divergence: " + result.failure);
            t.IsFalse(
                result.finalState.active,
                "the E-bit delay pair should complete both engines");
            t.Equals(
                result.executedPairs, uint32_t{8u},
                "the branch delay and E-bit delay should issue exactly");
            t.Equals(
                result.finalState.q, 4.0f,
                "WAITQ should expose the prior DIV result to MULq");
            float stored[4]{};
            std::memcpy(
                stored, result.finalData.data() + 16u,
                sizeof(stored));
            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                t.Equals(
                    stored[lane],
                    initialState.vf[1][lane] * immediate,
                    "lower-before-upper should store the pre-ADD VF value");
            }
        });

        tc.Run("IR oracle matches delayed P barriers and unsynchronized MFP", [](TestCase &t)
        {
            std::vector<uint8_t> code(8u * 8u);
            std::vector<uint8_t> data(256u);
            writeVuInstructionPair(
                code.data(), 0u, makeVuEleng(1u),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 8u, makeVuMfp(2u, 0x8u),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 16u, makeVuErcpr(4u, 1u),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 24u, makeVuMfp(3u, 0x8u),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 32u, makeVuWaitP(),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 40u, makeVuMfp(5u, 0x8u),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 48u, 0u,
                kVuUpperNop | kVuUpperEnd);
            writeVuInstructionPair(
                code.data(), 56u, 0u, kVuUpperNop);

            VuUnit seedUnit;
            VuExecutionState initialState = seedUnit.state();
            initialState.active = true;
            initialState.p = 2.0f;
            initialState.vf[1][0] = 3.0f;
            initialState.vf[1][1] = 4.0f;
            initialState.vf[4][1] = 4.0f;

            const VuDifferentialResult result =
                runVuDifferential(
                    code, data, initialState, 16u);
            t.IsTrue(
                result.matched,
                "IR P-pipeline divergence: " + result.failure);
            t.Equals(
                result.executedPairs, uint32_t{8u},
                "the P fixture should include its E-bit delay pair");
            t.Equals(
                result.finalState.vf[2][0], 2.0f,
                "MFP before synchronization should see old P");
            t.Equals(
                result.finalState.vf[3][0], 5.0f,
                "a new EFU op should publish the prior P result");
            t.Equals(
                result.finalState.vf[5][0], 0.25f,
                "MFP after WAITP should see the ERCPR result");
            t.Equals(
                result.finalState.p, 0.25f,
                "both engines should finish with synchronized P");
            t.IsFalse(
                result.finalState.pipeline.delayedP.active,
                "both engines should drain delayed P");
        });

        tc.Run("IR oracle retains streaming XGKICK transactionally", [](TestCase &t)
        {
            std::vector<uint8_t> code(8u * 6u);
            std::vector<uint8_t> data(256u);
            const uint64_t tag = makeGifTag(1u, 2u, 1u);
            std::memcpy(data.data(), &tag, sizeof(tag));

            writeVuInstructionPair(
                code.data(), 0u,
                makeVuLowerSpecial(0x6cu, 1u),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 8u, 0u, kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 16u,
                makeVuSq(0xfu, 3u, 2u, 0),
                kVuUpperNop);
            writeVuInstructionPair(
                code.data(), 24u, 0u,
                kVuUpperNop | 0x40000000u);
            writeVuInstructionPair(
                code.data(), 32u, 0u, kVuUpperNop);

            VuUnit seedUnit;
            VuExecutionState initialState = seedUnit.state();
            initialState.active = true;
            initialState.vi[1] = 0;
            initialState.vi[2] = 1;
            const float payload[4] = {
                21.0f, 22.0f, 23.0f, 24.0f};
            std::memcpy(
                initialState.vf[3], payload,
                sizeof(payload));

            const VuDifferentialResult result =
                runVuDifferential(
                    code, data, initialState, 16u);
            t.IsTrue(
                result.matched,
                "IR XGKICK divergence: " + result.failure);
            t.Equals(
                result.executedPairs, uint32_t{5u},
                "the retained transfer should cross multiple pairs");
            t.Equals(
                result.path1Packets.size(), size_t{1u},
                "the transactional sink should retain one packet");
            if (result.path1Packets.size() == 1u)
            {
                t.Equals(
                    result.path1Packets[0].size(), size_t{32u},
                    "the complete GIFtag and payload should be retained");
                if (result.path1Packets[0].size() == 32u)
                {
                    t.IsTrue(
                        std::memcmp(
                            result.path1Packets[0].data() + 16u,
                            payload, sizeof(payload)) == 0,
                        "XGKICK should observe the later VU data write");
                }
            }
        });

        tc.Run("fixed-seed VU programs match the IR oracle after every pair", [](TestCase &t)
        {
            constexpr std::array<uint32_t, 4> seeds = {
                0x13579bdfu,
                0x2468ace1u,
                0x6d2b79f5u,
                0xa5a55a5au,
            };
            for (const uint32_t seed : seeds)
            {
                FixedVuGenerator generator(seed);
                std::vector<uint8_t> code(64u * 8u);
                std::vector<uint8_t> data(1024u);
                for (uint8_t &byte : data)
                {
                    byte = static_cast<uint8_t>(
                        generator.next() & 0xffu);
                }

                writeVuInstructionPair(
                    code.data(), 0u,
                    makeVuIaddiu(1u, 0u, 3),
                    kVuUpperNop);
                writeVuInstructionPair(
                    code.data(), 8u,
                    makeVuIaddiu(2u, 0u, 1),
                    kVuUpperNop);
                for (uint32_t index = 2u; index <= 55u; ++index)
                {
                    const uint32_t upper =
                        generatedUpper(generator);
                    const uint32_t lower =
                        (upper & 0x80000000u) != 0u
                            ? generator.next()
                            : generatedLower(generator);
                    writeVuInstructionPair(
                        code.data(), index * 8u,
                        lower, upper);
                }
                writeVuInstructionPair(
                    code.data(), 56u * 8u,
                    makeVuLowerDirect(
                        0x32u, 1u, 1u, 31u),
                    kVuUpperNop);
                writeVuInstructionPair(
                    code.data(), 57u * 8u,
                    makeVuIbgtz(1u, -56),
                    kVuUpperNop);
                writeVuInstructionPair(
                    code.data(), 58u * 8u, 0u, kVuUpperNop);
                writeVuInstructionPair(
                    code.data(), 59u * 8u, 0u,
                    kVuUpperNop | 0x40000000u);
                writeVuInstructionPair(
                    code.data(), 60u * 8u, 0u, kVuUpperNop);

                VuUnit seedUnit;
                VuExecutionState initialState = seedUnit.state();
                initialState.active = true;
                for (uint32_t reg = 1u; reg < 32u; ++reg)
                {
                    for (uint32_t lane = 0u; lane < 4u; ++lane)
                    {
                        const int32_t value =
                            static_cast<int32_t>(
                                generator.next() % 4096u) -
                            2048;
                        initialState.vf[reg][lane] =
                            static_cast<float>(value) / 32.0f;
                    }
                }
                for (uint32_t reg = 1u; reg < 16u; ++reg)
                {
                    initialState.vi[reg] =
                        static_cast<int32_t>(
                            generator.next() & 0x3fu);
                }
                initialState.i = 0.25f;

                const VuDifferentialResult result =
                    runVuDifferential(
                        code, data, initialState, 512u);
                t.IsTrue(
                    result.matched,
                    "fixed-seed IR divergence for seed " +
                        std::to_string(seed) + ": " +
                        result.failure);
                t.IsFalse(
                    result.finalState.active,
                    "the bounded generated loop should terminate");
                t.Equals(
                    result.executedPairs, uint32_t{175u},
                    "the fixed three-iteration loop should be exact");

                size_t observedOpcodes = 0u;
                for (const uint64_t count : result.opcodeCounts)
                {
                    if (count != 0u)
                        ++observedOpcodes;
                }
                t.IsTrue(
                    observedOpcodes >= 12u,
                    "each generated corpus should exercise varied IR");
                t.Equals(
                    result.opcodeCounts[
                        static_cast<size_t>(
                            VuIrOpcode::Unsupported)],
                    uint64_t{0u},
                    "generated operations should all be supported");
            }
        });

        tc.Run("IR oracle side-exits before unsupported semantics", [](TestCase &t)
        {
            std::vector<uint8_t> code(16u);
            std::vector<uint8_t> data(256u, 0x5au);
            writeVuInstructionPair(
                code.data(), 0u, 0x04000000u,
                makeVuUpper(0x34u, 0xfu, 2u, 1u, 3u));

            VuUnit unit;
            VuExecutionState state = unit.state();
            state.active = true;
            const VuExecutionState initialState = state;
            VuTransactionalSideEffectSink effects;
            VuExecutionContext context{
                .state = state,
                .code = code.data(),
                .codeSize = static_cast<uint32_t>(code.size()),
                .data = data.data(),
                .dataSize = static_cast<uint32_t>(data.size()),
                .sideEffects = effects,
                .enableInstrumentation = false,
            };
            VuIrInterpreterBackend oracle(unit);
            const VuRunResult result = oracle.run(context, 1u);

            t.IsTrue(
                result.reason ==
                    VuExitReason::UnsupportedInstruction,
                "unsupported IR should return a structured side exit");
            t.Equals(
                result.executedCycles, uint32_t{0u},
                "unsupported IR should not retire a pair");
            t.Equals(
                state.pc, uint32_t{0u},
                "unsupported IR should remain at the original PC");
            std::string difference;
            t.IsTrue(
                vuExecutionStatesEqual(
                    state, initialState, &difference),
                "unsupported IR mutated " + difference);
            t.IsTrue(
                effects.path1Packets().empty(),
                "unsupported IR should publish no side effects");
        });

        tc.Run("bounded workload profile records exact pairs and identical MPG uploads", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            const std::filesystem::path profilePath =
                std::filesystem::temp_directory_path() /
                ("ps2x-vu1-profile-" +
                 std::to_string(
                     reinterpret_cast<uintptr_t>(&t)) +
                 ".json");
            std::error_code removeError;
            std::filesystem::remove(profilePath, removeError);
            t.IsTrue(
                fx.mem.configureVu1WorkloadProfile(
                    profilePath.string().c_str(), 2u, 3u),
                "workload profile should open its bounded output");

            for (uint32_t pair = 0u; pair < 6u; ++pair)
            {
                const uint32_t upper =
                    kVuUpperNop |
                    (pair == 4u ? kVuUpperEnd : 0u);
                writeVuInstructionPair(
                    fx.code, pair * 8u, 0u, upper);
            }

            const uint32_t uploadedLower =
                makeVuIaddiu(1u, 0u, 7);
            uploadVu1Mpg(
                fx.mem, 0x100u, uploadedLower, kVuUpperNop);
            uploadVu1Mpg(
                fx.mem, 0x100u, uploadedLower, kVuUpperNop);

            VuUnit vu1;
            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 64u);

            const Vu1WorkloadProfileSnapshot snapshot =
                fx.mem.vu1WorkloadProfileSnapshot();
            t.IsFalse(snapshot.enabled,
                      "a completed bounded profile should finalize immediately");
            t.IsTrue(snapshot.measurementComplete,
                     "three measured pairs should complete the bound");
            t.Equals(snapshot.observedPairs, uint64_t{5u},
                     "profile should stop after two warm-up and three measured pairs");
            t.Equals(snapshot.measuredPairs, uint64_t{3u},
                     "profile should retain exactly three post-warmup pairs");
            t.Equals(snapshot.codeUploads, uint64_t{2u},
                     "profile should count both MPG uploads");
            t.Equals(snapshot.identicalCodeUploads, uint64_t{1u},
                     "the second equal MPG upload should be identified");
            t.Equals(snapshot.codeUploadBytes, uint64_t{16u},
                     "two one-pair MPG uploads should total sixteen bytes");
            t.Equals(snapshot.identicalCodeUploadBytes, uint64_t{8u},
                     "the repeated one-pair upload should total eight bytes");

            fx.mem.finishVu1WorkloadProfile();
            std::ifstream profileFile(profilePath);
            const std::string profileText{
                std::istreambuf_iterator<char>(profileFile),
                std::istreambuf_iterator<char>()};
            t.IsTrue(
                profileText.find(
                    "\"measurement_complete\":true") !=
                    std::string::npos,
                "profile output should record the completed bound");
            t.IsTrue(
                profileText.find("\"measured_pairs\":3") !=
                    std::string::npos,
                "profile output should retain the exact measured work");
            t.IsTrue(
                profileText.find("\"identical_count\":1") !=
                    std::string::npos,
                "profile output should retain identical upload frequency");
            std::filesystem::remove(profilePath, removeError);
        });

        tc.Run("upper ADD applies the destination mask", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(fx.code, 0u, 0u, makeVuUpper(0x28u, 0xAu, 2u, 1u, 3u)); // ADD.xz vf3, vf1, vf2

            VuUnit vu1;
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

            VuUnit vu1;
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

            const auto initializeState = [](VuUnit &interpreter)
            {
                const float source[4] = {-1.0f, 0.0f, -1.0f, 0.0f};
                std::memcpy(
                    interpreter.state().vf[1], source, sizeof(source));
                interpreter.state().mac = 0x55AAu;
                interpreter.state().status = 0u;
            };

            VuUnit stepped;
            VuUnit batched;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
            setFloatBits(vu1.state().acc[2], 0x4B6A3856u);
            setFloatBits(vu1.state().vf[2][2], 0x41004B21u);
            setFloatBits(vu1.state().vf[5][1], 0xC5239000u);

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE,
                        fx.data, PS2_VU1_DATA_SIZE,
                        fx.gs, &fx.mem, 0u, 0u, 0u, 1u);

            t.Equals(getFloatBits(vu1.state().acc[2]), 0x4B69E65Du,
                     "MADD should truncate once after the fused product-sum");
        });

        tc.Run("VU conversion instructions saturate and honor round-toward-zero", [](TestCase &t)
        {
            std::vector<uint8_t> code(32u);
            std::vector<uint8_t> data(256u);
            writeVuInstructionPair(
                code.data(), 0u, 0u,
                makeVuUpperSpecial(
                    0x14u, 0xfu, 2u, 1u)); // FTOI0 vf2, vf1
            writeVuInstructionPair(
                code.data(), 8u, 0u,
                makeVuUpperSpecial(
                    0x15u, 0xfu, 4u, 3u)); // FTOI4 vf4, vf3
            writeVuInstructionPair(
                code.data(), 16u, 0u,
                makeVuUpperSpecial(
                    0x10u, 0xfu, 6u, 5u)); // ITOF0 vf6, vf5

            VuUnit unit;
            VuExecutionState state = unit.state();
            state.active = true;
            state.pc = 0u;
            state.vf[1][0] = 123.75f;
            state.vf[1][1] = 2147483648.0f;
            state.vf[1][2] = -2147483904.0f;
            state.vf[1][3] =
                std::numeric_limits<float>::quiet_NaN();
            state.vf[3][0] = 134217728.0f;
            state.vf[3][1] = -134217728.0f;
            state.vf[3][2] = 0.55f;
            state.vf[3][3] = -0.45f;
            setFloatBits(state.vf[5][0], 0x7fffffffu);
            setFloatBits(state.vf[5][1], 0x80000000u);
            setFloatBits(state.vf[5][2], 123u);
            setFloatBits(state.vf[5][3], 0xffffff85u);

            VuTransactionalSideEffectSink effects;
            VuExecutionContext context{
                .state = state,
                .code = code.data(),
                .codeSize =
                    static_cast<uint32_t>(code.size()),
                .data = data.data(),
                .dataSize =
                    static_cast<uint32_t>(data.size()),
                .sideEffects = effects,
                .enableInstrumentation = false,
            };
            VuInterpreterBackend backend(unit);
            const VuRunResult result =
                backend.run(context, 3u);

            t.Equals(
                result.executedCycles, uint32_t{3u},
                "all conversion pairs should execute");
            const std::array<uint32_t, 4u> ftoi0Expected{
                123u, 0x7fffffffu, 0x80000000u,
                0x80000000u};
            const std::array<uint32_t, 4u> ftoi4Expected{
                0x7fffffffu, 0x80000000u, 8u,
                0xfffffff9u};
            const std::array<uint32_t, 4u> itof0Expected{
                0x4effffffu, 0xcf000000u,
                0x42f60000u, 0xc2f60000u};
            for (size_t lane = 0u; lane < 4u; ++lane)
            {
                t.Equals(
                    getFloatBits(state.vf[2][lane]),
                    ftoi0Expected[lane],
                    "FTOI0 lane " +
                        std::to_string(lane));
                t.Equals(
                    getFloatBits(state.vf[4][lane]),
                    ftoi4Expected[lane],
                    "FTOI4 lane " +
                        std::to_string(lane));
                t.Equals(
                    getFloatBits(state.vf[6][lane]),
                    itof0Expected[lane],
                    "ITOF0 lane " +
                        std::to_string(lane));
            }
        });

        tc.Run("CLIP uses absolute w and architectural flag ordering", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, 0u,
                makeVuUpperSpecial(0x1Fu, 0xEu, 2u, 1u)); // CLIPw.xyz vf1, vf2w

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
            vu1.state().q = 3.0f;
            vu1.state().vf[1][0] = 18.0f;
            vu1.state().vf[2][0] = 3.0f;

            vu1.execute(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 0u, 7u);
            t.Equals(vu1.state().q, 3.0f, "Q should remain old through the first six pairs after DIV");

            vu1.resume(fx.code, PS2_VU1_CODE_SIZE, fx.data, PS2_VU1_DATA_SIZE, fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(vu1.state().q, 6.0f, "Q should commit on the seventh pair after DIV");
        });

        tc.Run("EFU results remain delayed until WAITP synchronizes P", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, makeVuEleng(1u), kVuUpperNop);
            writeVuInstructionPair(
                fx.code, 8u, makeVuMfp(2u, 0x8u), kVuUpperNop);
            writeVuInstructionPair(
                fx.code, 16u, makeVuWaitP(), kVuUpperNop);
            writeVuInstructionPair(
                fx.code, 24u, makeVuMfp(3u, 0x8u), kVuUpperNop);

            VuUnit vu1;
            vu1.state().p = 2.0f;
            vu1.state().vf[1][0] = 3.0f;
            vu1.state().vf[1][1] = 4.0f;

            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 1u);
            t.Equals(
                vu1.state().p, 2.0f,
                "ELENG should retain the old architectural P");
            t.IsTrue(
                vu1.state().pipeline.delayedP.active,
                "ELENG should leave its result in the EFU pipeline");
            t.Equals(
                vu1.state().pipeline.delayedP.cyclesRemaining,
                uint32_t{18u},
                "ELENG should schedule the documented eighteen-cycle latency");

            vu1.resume(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(
                vu1.state().vf[2][0], 2.0f,
                "MFP without WAITP should observe the old P");

            vu1.resume(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(
                vu1.state().p, 5.0f,
                "WAITP should commit the pending ELENG result");
            t.IsFalse(
                vu1.state().pipeline.delayedP.active,
                "WAITP should drain the EFU pipeline");

            vu1.resume(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(
                vu1.state().vf[3][0], 5.0f,
                "MFP after WAITP should observe the ELENG result");
        });

        tc.Run("ELENG result becomes visible after eighteen following pairs", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, makeVuEleng(1u), kVuUpperNop);
            for (uint32_t pc = 8u; pc <= 144u; pc += 8u)
            {
                writeVuInstructionPair(
                    fx.code, pc, 0u, kVuUpperNop);
            }

            VuUnit vu1;
            vu1.state().p = 2.0f;
            vu1.state().vf[1][0] = 3.0f;
            vu1.state().vf[1][1] = 4.0f;

            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 18u);
            t.Equals(
                vu1.state().p, 2.0f,
                "P should remain old through seventeen following pairs");
            t.Equals(
                vu1.state().pipeline.delayedP.cyclesRemaining,
                uint32_t{1u},
                "one ELENG latency cycle should remain");

            vu1.resume(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 1u);
            t.Equals(
                vu1.state().p, 5.0f,
                "P should commit on the eighteenth following pair");
            t.IsFalse(
                vu1.state().pipeline.delayedP.active,
                "natural latency completion should drain the EFU pipeline");
        });

        tc.Run("E-bit completion drains a pending EFU result", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, makeVuEleng(1u),
                kVuUpperNop | kVuUpperEnd);
            writeVuInstructionPair(
                fx.code, 8u, 0u, kVuUpperNop);

            VuUnit vu1;
            vu1.state().p = 2.0f;
            vu1.state().vf[1][0] = 3.0f;
            vu1.state().vf[1][1] = 4.0f;

            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 2u);
            t.Equals(
                vu1.state().p, 5.0f,
                "program completion should publish the pending P result");
            t.IsFalse(
                vu1.state().pipeline.delayedP.active,
                "program completion should drain the EFU pipeline");
            t.IsFalse(
                vu1.isActive(),
                "the E-bit delay pair should complete normally");
        });

        tc.Run("EFU lower source observes old VF from its paired upper write", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            writeVuInstructionPair(
                fx.code, 0u, makeVuEleng(1u),
                makeVuUpper(0x28u, 0xfu, 3u, 2u, 1u));
            writeVuInstructionPair(
                fx.code, 8u, makeVuWaitP(), kVuUpperNop);

            VuUnit vu1;
            vu1.state().vf[1][0] = 3.0f;
            vu1.state().vf[1][1] = 4.0f;
            vu1.state().vf[2][0] = 1.0f;
            vu1.state().vf[2][1] = 2.0f;
            vu1.state().vf[3][0] = 5.0f;
            vu1.state().vf[3][1] = 6.0f;

            vu1.execute(
                fx.code, PS2_VU1_CODE_SIZE,
                fx.data, PS2_VU1_DATA_SIZE,
                fx.gs, &fx.mem, 0u, 0u, 0u, 2u);
            t.Equals(
                vu1.state().vf[1][0], 6.0f,
                "the paired upper ADD should publish its x result");
            t.Equals(
                vu1.state().vf[1][1], 8.0f,
                "the paired upper ADD should publish its y result");
            t.Equals(
                vu1.state().p, 5.0f,
                "ELENG should read the pre-upper VF source");
        });

        tc.Run("MPG upload invalidates cached VU1 decode before MSCAL", [](TestCase &t)
        {
            Vu1Fixture fx;
            t.IsTrue(fx.initialize(), "VU1 fixture should initialize");

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit single;
            VuUnit split;
            single.start(0u, 0x123u, 0x2ABu,
                         &singleFixture.mem);
            split.start(0u, 0x123u, 0x2ABu,
                        &splitFixture.mem);

            const VuRunResult singleResult =
                single.advance(
                    singleFixture.code, PS2_VU1_CODE_SIZE,
                    singleFixture.data, PS2_VU1_DATA_SIZE,
                    singleFixture.gs, &singleFixture.mem,
                    12u);
            uint32_t splitExecuted = 0u;
            for (const uint32_t budget :
                 std::array<uint32_t, 4>{1u, 2u, 2u, 7u})
            {
                const VuRunResult result =
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

            const VuExecutionState &singleState = single.state();
            const VuExecutionState &splitState = split.state();
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

            VuUnit vu1;
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

            VuUnit vu1;
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

            VuUnit vu1;
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
