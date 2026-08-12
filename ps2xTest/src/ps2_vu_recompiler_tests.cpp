#include "MiniTest.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_recompiler.h"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace
{
    constexpr uint32_t kVuUpperNop = 0x000002ffu;
    constexpr uint32_t kVuUpperEnd = 1u << 30u;

    class ScopedEnvironmentVariable
    {
    public:
        explicit ScopedEnvironmentVariable(
            const char *name)
            : m_name(name)
        {
            if (const char *const value =
                    std::getenv(name))
            {
                m_original = value;
            }
        }

        ~ScopedEnvironmentVariable()
        {
            if (m_original)
            {
                setenv(
                    m_name.c_str(),
                    m_original->c_str(), 1);
            }
            else
            {
                unsetenv(m_name.c_str());
            }
        }

        void set(const char *value)
        {
            setenv(m_name.c_str(), value, 1);
        }

        void unset()
        {
            unsetenv(m_name.c_str());
        }

        ScopedEnvironmentVariable(
            const ScopedEnvironmentVariable &) = delete;
        ScopedEnvironmentVariable &operator=(
            const ScopedEnvironmentVariable &) = delete;

    private:
        std::string m_name;
        std::optional<std::string> m_original;
    };

    struct VuNativeFixture
    {
        PS2Memory memory;
        uint8_t *code = nullptr;
        uint8_t *data = nullptr;
        uint32_t codeSize = 0u;
        uint32_t dataSize = 0u;
        VuUnitId unit = VuUnitId::Vu1;

        bool initialize(
            VuUnitId requestedUnit = VuUnitId::Vu1)
        {
            if (!memory.initialize())
                return false;
            unit = requestedUnit;
            if (unit == VuUnitId::Vu0)
            {
                code = memory.getVU0Code();
                data = memory.getVU0Data();
                codeSize = PS2_VU0_CODE_SIZE;
                dataSize = PS2_VU0_DATA_SIZE;
            }
            else
            {
                code = memory.getVU1Code();
                data = memory.getVU1Data();
                codeSize = PS2_VU1_CODE_SIZE;
                dataSize = PS2_VU1_DATA_SIZE;
            }
            if (!code || !data)
                return false;
            std::memset(code, 0, codeSize);
            std::memset(data, 0, dataSize);
            for (uint32_t pc = 0u; pc < codeSize; pc += 8u)
            {
                std::memcpy(
                    code + pc + 4u,
                    &kVuUpperNop,
                    sizeof(kVuUpperNop));
            }
            markCodeModified();
            return true;
        }

        void markCodeModified()
        {
            if (unit == VuUnitId::Vu0)
                memory.markVU0CodeModified();
            else
                memory.markVU1CodeModified();
        }
    };

    uint32_t makeVuLowerSpecial(
        uint8_t specialOp, uint8_t is,
        uint8_t it = 0u, uint8_t id = 0u,
        uint8_t dest = 0u)
    {
        return
            (0x40u << 25u) |
            (static_cast<uint32_t>(dest & 0xfu) << 21u) |
            (static_cast<uint32_t>(it & 0x1fu) << 16u) |
            (static_cast<uint32_t>(is & 0x1fu) << 11u) |
            (static_cast<uint32_t>(id & 0x1fu) << 6u) |
            (static_cast<uint32_t>(
                 specialOp & 0x7cu)
             << 4u) |
            static_cast<uint32_t>(specialOp & 3u) |
            0x3cu;
    }

    uint32_t makeVuUpper(
        uint8_t op, uint8_t dest, uint8_t ft,
        uint8_t fs, uint8_t fd)
    {
        return
            (static_cast<uint32_t>(dest & 0xfu) << 21u) |
            (static_cast<uint32_t>(ft & 0x1fu) << 16u) |
            (static_cast<uint32_t>(fs & 0x1fu) << 11u) |
            (static_cast<uint32_t>(fd & 0x1fu) << 6u) |
            static_cast<uint32_t>(op & 0x3fu);
    }

    uint32_t makeVuUpperSpecial(
        uint8_t specialOp, uint8_t dest,
        uint8_t ft, uint8_t fs)
    {
        return
            (static_cast<uint32_t>(dest & 0xfu) << 21u) |
            (static_cast<uint32_t>(ft & 0x1fu) << 16u) |
            (static_cast<uint32_t>(fs & 0x1fu) << 11u) |
            (static_cast<uint32_t>(
                 specialOp & 0x7cu)
             << 4u) |
            static_cast<uint32_t>(specialOp & 3u) |
            0x3cu;
    }

    uint32_t makeVuLq(
        uint8_t dest, uint8_t targetVf,
        uint8_t baseVi, int16_t immediate)
    {
        return
            (static_cast<uint32_t>(dest & 0xfu) << 21u) |
            (static_cast<uint32_t>(
                 targetVf & 0x1fu)
             << 16u) |
            (static_cast<uint32_t>(baseVi & 0xfu) << 11u) |
            (static_cast<uint32_t>(immediate) & 0x7ffu);
    }

    uint32_t makeVuLqi(
        uint8_t dest, uint8_t targetVf,
        uint8_t baseVi)
    {
        return makeVuLowerSpecial(
            0x34u, baseVi, targetVf, 0u, dest);
    }

    uint32_t makeVuSq(
        uint8_t dest, uint8_t sourceVf,
        uint8_t baseVi, int16_t immediate)
    {
        return
            (0x01u << 25u) |
            (static_cast<uint32_t>(dest & 0xfu) << 21u) |
            (static_cast<uint32_t>(baseVi & 0xfu) << 16u) |
            (static_cast<uint32_t>(
                 sourceVf & 0x1fu)
             << 11u) |
            (static_cast<uint32_t>(immediate) & 0x7ffu);
    }

    uint32_t makeVuIaddiu(
        uint8_t it, uint8_t is, int16_t immediate)
    {
        return
            (0x08u << 25u) |
            (static_cast<uint32_t>(it & 0xfu) << 16u) |
            (static_cast<uint32_t>(is & 0xfu) << 11u) |
            (static_cast<uint32_t>(immediate) & 0x7ffu);
    }

    uint32_t makeVuBranch(int16_t immediate)
    {
        return
            (0x20u << 25u) |
            (static_cast<uint32_t>(immediate) & 0x7ffu);
    }

    uint32_t makeVuBranchOperation(
        uint8_t opcode, uint8_t is, uint8_t it,
        int16_t immediate)
    {
        return
            (static_cast<uint32_t>(opcode) << 25u) |
            (static_cast<uint32_t>(it & 0xfu) << 16u) |
            (static_cast<uint32_t>(is & 0xfu) << 11u) |
            (static_cast<uint32_t>(immediate) & 0x7ffu);
    }

    uint32_t makeVuDiv(
        uint8_t fs, uint8_t ft,
        uint8_t fsf, uint8_t ftf)
    {
        return makeVuLowerSpecial(
            0x38u, fs, ft, 0u,
            static_cast<uint8_t>(
                ((ftf & 3u) << 2u) | (fsf & 3u)));
    }

    void writePair(
        uint8_t *code, uint32_t pc,
        uint32_t lower, uint32_t upper)
    {
        std::memcpy(code + pc, &lower, sizeof(lower));
        std::memcpy(
            code + pc + 4u, &upper, sizeof(upper));
    }

    uint64_t makeGifTag(
        uint16_t loops, uint8_t format,
        uint8_t registers)
    {
        return
            static_cast<uint64_t>(loops & 0x7fffu) |
            (1ull << 15u) |
            (static_cast<uint64_t>(format & 3u) << 58u) |
            (static_cast<uint64_t>(registers & 0xfu)
             << 60u);
    }

    void configureSyntheticProgram(uint8_t *code)
    {
        writePair(
            code, 0u,
            makeVuIaddiu(1u, 0u, 2),
            makeVuUpper(
                0x28u, 0xfu, 2u, 1u, 3u));
        writePair(
            code, 8u,
            makeVuSq(0xfu, 3u, 1u, 0),
            makeVuUpper(
                0x2au, 0xfu, 2u, 3u, 4u));
        writePair(
            code, 16u,
            makeVuLq(0xfu, 5u, 1u, 0),
            kVuUpperNop);
        writePair(
            code, 24u, 0u,
            makeVuUpper(
                0x22u, 0xfu, 0u, 5u, 6u));

        const float immediate = 0.5f;
        uint32_t immediateBits = 0u;
        std::memcpy(
            &immediateBits, &immediate,
            sizeof(immediateBits));
        writePair(
            code, 32u, immediateBits,
            makeVuUpper(
                0x22u, 0xfu, 0u, 6u, 7u) |
                0x80000000u);

        writePair(
            code, 40u,
            makeVuDiv(1u, 2u, 0u, 0u),
            makeVuUpper(
                0x1cu, 0xfu, 0u, 7u, 8u));
        writePair(
            code, 48u, 0u,
            makeVuUpper(
                0x1cu, 0xfu, 0u, 8u, 9u));
        writePair(
            code, 56u, makeVuBranch(2),
            makeVuUpper(
                0x28u, 0xfu, 2u, 9u, 10u));
        writePair(
            code, 64u,
            makeVuIaddiu(2u, 0u, 9),
            kVuUpperNop);
        writePair(
            code, 72u,
            makeVuIaddiu(3u, 0u, 99),
            kVuUpperNop);
        writePair(
            code, 80u,
            makeVuIaddiu(4u, 2u, 3),
            makeVuUpper(
                0x2cu, 0xfu, 1u, 10u, 11u) |
                kVuUpperEnd);
        writePair(
            code, 88u, 0u, kVuUpperNop);
    }

    void configureHelperRegisterProgram(uint8_t *code)
    {
        writePair(
            code, 0u, 0u,
            makeVuUpper(
                0x28u, 0xfu, 2u, 1u, 3u));
        writePair(
            code, 8u, 0u,
            makeVuUpper(
                0x28u, 0xfu, 2u, 2u, 2u));
        writePair(
            code, 16u, 0u,
            makeVuUpper(
                0x28u, 0xfu, 2u, 3u, 1u));

        const float immediate = 0.5f;
        uint32_t immediateBits = 0u;
        std::memcpy(
            &immediateBits, &immediate,
            sizeof(immediateBits));
        writePair(
            code, 24u, immediateBits,
            makeVuUpper(
                0x22u, 0x5u, 0u, 1u, 3u) |
                0x80000000u);
        writePair(
            code, 32u, 0u,
            makeVuUpper(
                0x28u, 0xfu, 2u, 3u, 4u));
    }

    VuExecutionState initialSyntheticState()
    {
        VuUnit seed(VuUnitId::Vu1);
        VuExecutionState state = seed.state();
        state.active = true;
        state.pc = 0u;
        state.i = 0.25f;
        state.q = 2.0f;
        const std::array<float, 4u> vf1{
            8.0f, -4.0f, 2.0f, 1.0f};
        const std::array<float, 4u> vf2{
            2.0f, 3.0f, -5.0f, 4.0f};
        std::memcpy(
            state.vf[1], vf1.data(), sizeof(state.vf[1]));
        std::memcpy(
            state.vf[2], vf2.data(), sizeof(state.vf[2]));
        return state;
    }

    bool sameRunResult(
        const VuRunResult &left,
        const VuRunResult &right)
    {
        return
            left.requestedCycles == right.requestedCycles &&
            left.executedCycles == right.executedCycles &&
            left.reason == right.reason &&
            left.activeBefore == right.activeBefore &&
            left.activeAfter == right.activeAfter &&
            left.completed == right.completed;
    }

    VuExecutionContext makeContext(
        VuExecutionState &state,
        VuNativeFixture &fixture,
        IVuSideEffectSink &effects,
        bool instrumentation = false)
    {
        return {
            .state = state,
            .code = fixture.code,
            .codeSize = fixture.codeSize,
            .data = fixture.data,
            .dataSize = fixture.dataSize,
            .sideEffects = effects,
            .memory = &fixture.memory,
            .enableInstrumentation = instrumentation,
        };
    }

    class InvalidatingSideEffectSink final :
        public IVuSideEffectSink
    {
    public:
        explicit InvalidatingSideEffectSink(PS2Memory &memory)
            : m_memory(memory)
        {
        }

        void submitPath1Packet(
            const uint8_t *, uint32_t) override
        {
            ++submissions;
            m_memory.markVU1CodeModified();
        }

        uint32_t submissions = 0u;

    private:
        PS2Memory &m_memory;
    };

#if PS2X_TEST_HAS_VU_RECOMPILER_ABI_PROBE
    extern "C" uint64_t ps2xTestVuNativeProbeCalleeSaved(
        VuNativeBlockEntry entry,
        VuExecutionContext *context,
        uint32_t maximumCycles,
        uint64_t preserved[6]);
#endif
}

void register_ps2_vu_recompiler_tests()
{
    MiniTest::Case("PS2VURecompiler", [](TestCase &tc)
    {
        ScopedEnvironmentVariable linking(
            "PS2X_VU_BLOCK_LINKING");
        linking.set("1");

        tc.Run(
            "every IR opcode has an explicit native disposition",
            [](TestCase &t)
            {
                constexpr size_t kDispositionCount = 3u;
                constexpr uint32_t kOpcodeCount =
                    static_cast<uint32_t>(
                        VuIrOpcode::Unsupported) +
                    1u;
                std::array<uint32_t, kDispositionCount>
                    baselineCounts{};
                std::array<uint32_t, kDispositionCount>
                    avxFmaCounts{};

                for (uint32_t value = 0u;
                     value < kOpcodeCount;
                     ++value)
                {
                    const VuIrOpcode opcode =
                        static_cast<VuIrOpcode>(value);
                    t.IsTrue(
                        vuIrOpcodeName(opcode) != "Unknown",
                        "IR opcode " +
                            std::to_string(value) +
                            " should have a stable name");

                    const auto baseline =
                        VuRecompilerBackend::
                            opcodeDisposition(opcode, 0u);
                    const auto avxFma =
                        VuRecompilerBackend::
                            opcodeDisposition(
                                opcode,
                                std::numeric_limits<
                                    uint64_t>::max());
                    ++baselineCounts[
                        static_cast<size_t>(baseline)];
                    ++avxFmaCounts[
                        static_cast<size_t>(avxFma)];
                }

                t.Equals(
                    kOpcodeCount, uint32_t{123u},
                    "the audit must be updated when IR grows");
                t.Equals(
                    baselineCounts[static_cast<size_t>(
                        VuRecompilerOpcodeDisposition::
                            NativeInline)],
                    uint32_t{71u},
                    "baseline inline opcode count");
                t.Equals(
                    baselineCounts[static_cast<size_t>(
                        VuRecompilerOpcodeDisposition::
                            NativeHelper)],
                    uint32_t{51u},
                    "baseline helper opcode count");
                t.Equals(
                    baselineCounts[static_cast<size_t>(
                        VuRecompilerOpcodeDisposition::
                            InterpreterSideExit)],
                    uint32_t{1u},
                    "only unsupported IR may side-exit");
                t.Equals(
                    avxFmaCounts[static_cast<size_t>(
                        VuRecompilerOpcodeDisposition::
                            NativeInline)],
                    uint32_t{87u},
                    "AVX/FMA inline opcode count");
                t.Equals(
                    avxFmaCounts[static_cast<size_t>(
                        VuRecompilerOpcodeDisposition::
                            NativeHelper)],
                    uint32_t{35u},
                    "AVX/FMA helper opcode count");
                t.Equals(
                    avxFmaCounts[static_cast<size_t>(
                        VuRecompilerOpcodeDisposition::
                            InterpreterSideExit)],
                    uint32_t{1u},
                    "AVX/FMA should retain one side-exit opcode");
                t.IsTrue(
                    VuRecompilerBackend::opcodeDisposition(
                        VuIrOpcode::Unsupported, 0u) ==
                        VuRecompilerOpcodeDisposition::
                            InterpreterSideExit,
                    "unsupported IR must retain its interpreter side exit");
            });

        tc.Run(
            "availability and keys require tracked unit-owned code",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::built())
                {
                    t.IsFalse(
                        VuRecompilerBackend::supported(),
                        "an omitted backend cannot report host support");
                    return;
                }
                t.IsTrue(
                    VuRecompilerBackend::supported(),
                    "a built backend should support this x86-64 test host");
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(
                        state, fixture, effects);

                VuProgramKey key;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context, VuCompilationMode::Normal,
                        key, &diagnostic),
                    "tracked VU1 code should produce a key: " +
                        diagnostic);
                t.IsTrue(
                    key.unit == VuUnitId::Vu1,
                    "the key should identify VU1");
                t.Equals(
                    key.codeGeneration,
                    fixture.memory.getVU1CodeGeneration(),
                    "the key should capture the current generation");
                t.Equals(
                    key.entryPc, uint32_t{0u},
                    "the key should capture the current PC");

                context.code = fixture.memory.getVU0Code();
                context.codeSize = PS2_VU0_CODE_SIZE;
                t.IsFalse(
                    backend.programKey(
                        context, VuCompilationMode::Normal,
                        key, &diagnostic),
                    "VU1 must reject VU0 code");

                std::array<uint8_t, 16u> detached{};
                context.code = detached.data();
                context.codeSize =
                    static_cast<uint32_t>(detached.size());
                t.IsFalse(
                    backend.programKey(
                        context, VuCompilationMode::Normal,
                        key, &diagnostic),
                    "detached code should not bypass generation tracking");
            });

        tc.Run(
            "inline XGKICK applies only to normal VU1 blocks",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable inlineXgkick(
                    "PS2X_VU_INLINE_XGKICK");
                inlineXgkick.set("1");
                for (const VuUnitId unitId : {
                         VuUnitId::Vu0,
                         VuUnitId::Vu1,
                     })
                {
                    VuNativeFixture fixture;
                    t.IsTrue(
                        fixture.initialize(unitId),
                        "VU memory fixture should initialize");
                    VuUnit unit(unitId);
                    VuRecompilerBackend backend(unit);
                    VuExecutionState state =
                        initialSyntheticState();
                    VuTransactionalSideEffectSink effects;
                    VuExecutionContext context =
                        makeContext(
                            state, fixture, effects);
                    VuProgramKey key;
                    std::string diagnostic;
                    t.IsTrue(
                        backend.programKey(
                            context,
                            VuCompilationMode::Normal,
                            key, &diagnostic),
                        "normal key should be valid: " +
                            diagnostic);
                    t.Equals(
                        key.inlineXgkick,
                        unitId == VuUnitId::Vu1,
                        "only VU1 should compile inline XGKICK progression");

                    t.IsTrue(
                        backend.programKey(
                            context,
                            VuCompilationMode::Instrumented,
                            key, &diagnostic),
                        "instrumented key should be valid: " +
                            diagnostic);
                    t.IsFalse(
                        key.inlineXgkick,
                        "instrumented observation should retain helper progression");
                }

                inlineXgkick.set("0");
                VuNativeFixture disabledFixture;
                t.IsTrue(
                    disabledFixture.initialize(),
                    "disabled VU1 fixture should initialize");
                VuUnit disabledUnit(VuUnitId::Vu1);
                VuRecompilerBackend disabledBackend(
                    disabledUnit);
                t.IsFalse(
                    disabledBackend.inlineXgkickEnabled(),
                    "the explicit opt-out should retain helper progression");
            });

        tc.Run(
            "block-local VF automatic policy remains selectable",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable registers(
                    "PS2X_VU_BLOCK_LOCAL_VF_REGISTERS");
                registers.unset();

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                VuProgramKey key;
                std::string diagnostic;

                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        key, &diagnostic),
                    "XGKICK-free code should produce an automatic key: " +
                        diagnostic);
                t.IsTrue(
                    key.blockLocalVfRegisters,
                    "automatic mode should select block-local VF analysis");
                t.IsTrue(
                    key.blockLocalVfRegistersAutomatic,
                    "the default key should identify automatic allocation");

                state.pipeline.xgkick.active = true;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        key, &diagnostic),
                    "active XGKICK should produce an automatic key: " +
                        diagnostic);
                t.IsTrue(
                    key.blockLocalVfRegisters,
                    "automatic policy should be based on block traffic");
                t.IsTrue(
                    key.blockLocalVfRegistersAutomatic,
                    "active XGKICK should not change the automatic policy");

                state.pipeline.xgkick.active = true;
                registers.set("1");
                VuUnit forcedUnit(VuUnitId::Vu1);
                VuRecompilerBackend forcedBackend(
                    forcedUnit);
                t.IsTrue(
                    forcedBackend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        key, &diagnostic),
                    "forced active XGKICK should produce a key: " +
                        diagnostic);
                t.IsTrue(
                    key.blockLocalVfRegisters,
                    "explicit opt-in should still cover active XGKICK");
                t.IsFalse(
                    key.blockLocalVfRegistersAutomatic,
                    "explicit opt-in should identify forced allocation");

                registers.set("0");
                VuUnit disabledUnit(VuUnitId::Vu1);
                VuRecompilerBackend disabledBackend(
                    disabledUnit);
                state.pipeline.xgkick.active = false;
                t.IsTrue(
                    disabledBackend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        key, &diagnostic),
                    "disabled inactive XGKICK should produce a key: " +
                        diagnostic);
                t.IsFalse(
                    key.blockLocalVfRegisters,
                    "explicit opt-out should keep canonical VF state");
                t.IsFalse(
                    key.blockLocalVfRegistersAutomatic,
                    "explicit opt-out should not carry an allocation policy");
            });

        tc.Run("synthetic block matches every cycle-budget cut",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                for (const VuUnitId unitId : {
                         VuUnitId::Vu0,
                         VuUnitId::Vu1,
                     })
                {
                    VuNativeFixture referenceFixture;
                    VuNativeFixture nativeFixture;
                    t.IsTrue(referenceFixture.initialize(unitId),
                        "reference memory should initialize");
                    t.IsTrue(nativeFixture.initialize(unitId),
                        "native memory should initialize");
                    configureSyntheticProgram(referenceFixture.code);
                    configureSyntheticProgram(nativeFixture.code);
                    referenceFixture.markCodeModified();
                    nativeFixture.markCodeModified();

                    VuUnit referenceUnit(unitId);
                    VuUnit nativeUnit(unitId);
                    VuInterpreterBackend reference(referenceUnit);
                    VuRecompilerBackend native(nativeUnit);
                    const std::string unitPrefix =
                        unitId == VuUnitId::Vu0 ? "VU0 " : "VU1 ";

                    for (uint32_t budget = 0u; budget <= 12u; ++budget)
                    {
                        std::memset(referenceFixture.data,
                            0x5a,
                            referenceFixture.dataSize);
                        std::memset(
                            nativeFixture.data, 0x5a, nativeFixture.dataSize);
                        VuExecutionState referenceState =
                            initialSyntheticState();
                        VuExecutionState nativeState = referenceState;
                        VuTransactionalSideEffectSink referenceEffects;
                        VuTransactionalSideEffectSink nativeEffects;
                        VuExecutionContext referenceContext = makeContext(
                            referenceState, referenceFixture, referenceEffects);
                        VuExecutionContext nativeContext = makeContext(
                            nativeState, nativeFixture, nativeEffects);

                        const VuRunResult referenceResult =
                            reference.run(referenceContext, budget);
                        const VuRunResult nativeResult =
                            native.run(nativeContext, budget);
                        const std::string prefix = unitPrefix + "budget " +
                                                   std::to_string(budget) +
                                                   ": ";

                        t.IsTrue(sameRunResult(referenceResult, nativeResult),
                            prefix + "run result differs (" +
                                std::string(
                                    vuExitReasonName(referenceResult.reason)) +
                                "/" +
                                std::string(
                                    vuExitReasonName(nativeResult.reason)) +
                                ")");
                        std::string difference;
                        t.IsTrue(vuExecutionStatesEqual(
                                     referenceState, nativeState, &difference),
                            prefix + "state differs at " + difference);
                        t.IsTrue(std::memcmp(referenceFixture.data,
                                     nativeFixture.data,
                                     referenceFixture.dataSize) == 0,
                            prefix + "VU data differs");
                        t.IsTrue(referenceEffects.path1Packets() ==
                                     nativeEffects.path1Packets(),
                            prefix + "PATH1 effects differ");
                    }

                    const VuProgramCacheDiagnostics cache =
                        nativeUnit.programCache().diagnostics();
                    t.IsTrue(cache.compilations > 0u,
                        "the differential should compile native blocks");
                    t.IsTrue(cache.hits > 0u,
                        "later budget cuts should reuse native blocks");
                    const VuRecompilerDiagnostics diagnostics =
                        native.diagnostics();
                    t.IsTrue(diagnostics.inlinePairs > 0u,
                        "eligible memory/MULq pairs should run inline");
                    t.IsTrue(diagnostics.helperPairs > 0u,
                        "delicate pairs should remain on shared helpers");
                    t.Equals(diagnostics.inlinePairs + diagnostics.helperPairs,
                        diagnostics.nativePairs,
                        "native pair accounting should be complete");
                    t.Equals(
                        diagnostics.fullBlockGuards +
                            diagnostics.preciseTailEntries,
                        diagnostics.nativeBlocks,
                        "every native block should select one budget path");
                    t.Equals(
                        diagnostics.fullGuardPairs +
                            diagnostics.preciseTailPairs,
                        diagnostics.nativePairs,
                        "every native pair should have one budget class");
                    if (native.blockBudgetGuardsEnabled())
                    {
                        t.IsTrue(
                            diagnostics.fullBlockGuards > 0u &&
                                diagnostics.preciseTailEntries > 0u,
                            "the sweep should exercise full and precise paths");
                    }
                    else
                    {
                        t.Equals(
                            diagnostics.fullBlockGuards,
                            uint64_t{0u},
                            "the legacy sweep should emit no full guards");
                    }
                }
            });

        tc.Run(
            "block-budget guard switch retains the precise checked form",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture guardedFixture;
                VuNativeFixture legacyFixture;
                t.IsTrue(
                    guardedFixture.initialize(),
                    "guarded memory should initialize");
                t.IsTrue(
                    legacyFixture.initialize(),
                    "legacy memory should initialize");
                for (uint32_t pair = 0u;
                     pair < 64u; ++pair)
                {
                    writePair(
                        guardedFixture.code,
                        pair * 8u,
                        makeVuIaddiu(1u, 1u, 1),
                        kVuUpperNop);
                    writePair(
                        legacyFixture.code,
                        pair * 8u,
                        makeVuIaddiu(1u, 1u, 1),
                        kVuUpperNop);
                }
                guardedFixture.markCodeModified();
                legacyFixture.markCodeModified();

                ScopedEnvironmentVariable guards(
                    "PS2X_VU_BLOCK_BUDGET_GUARDS");
                guards.unset();
                VuUnit defaultUnit(VuUnitId::Vu1);
                VuRecompilerBackend defaultBackend(
                    defaultUnit);
                t.IsTrue(
                    defaultBackend.blockBudgetGuardsEnabled(),
                    "block-budget guards should default to enabled");
                guards.set("1");
                VuUnit guardedUnit(VuUnitId::Vu1);
                VuRecompilerBackend guarded(guardedUnit);
                guards.set("0");
                VuUnit legacyUnit(VuUnitId::Vu1);
                VuRecompilerBackend legacy(legacyUnit);

                VuExecutionState guardedState =
                    initialSyntheticState();
                VuExecutionState legacyState =
                    guardedState;
                VuTransactionalSideEffectSink guardedEffects;
                VuTransactionalSideEffectSink legacyEffects;
                VuExecutionContext guardedContext =
                    makeContext(
                        guardedState, guardedFixture,
                        guardedEffects);
                VuExecutionContext legacyContext =
                    makeContext(
                        legacyState, legacyFixture,
                        legacyEffects);

                const VuRunResult guardedFull =
                    guarded.run(guardedContext, 64u);
                const VuRunResult legacyFull =
                    legacy.run(legacyContext, 64u);
                t.IsTrue(
                    sameRunResult(
                        guardedFull, legacyFull),
                    "full guarded and checked results should match");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        guardedState, legacyState,
                        &difference),
                    "full guarded and checked states differ at " +
                        difference);
                t.IsTrue(
                    guarded.blockBudgetGuardsEnabled(),
                    "the guarded backend should expose its mode");
                t.IsFalse(
                    legacy.blockBudgetGuardsEnabled(),
                    "the checked backend should expose its mode");

                const VuRecompilerDiagnostics guardedFullStats =
                    guarded.diagnostics();
                const VuRecompilerDiagnostics legacyFullStats =
                    legacy.diagnostics();
                t.Equals(
                    guardedFullStats.fullBlockGuards,
                    uint64_t{1u},
                    "the full block should use one guard");
                t.Equals(
                    guardedFullStats.preciseTailEntries,
                    uint64_t{0u},
                    "the full block should avoid the precise tail");
                t.Equals(
                    guardedFullStats.fullGuardPairs,
                    uint64_t{64u},
                    "the full guard should cover all pairs");
                t.Equals(
                    guardedFullStats.preciseTailPairs,
                    uint64_t{0u},
                    "the full block should retire no precise pairs");
                t.Equals(
                    guardedFullStats.budgetGuardComparisons,
                    uint64_t{2u},
                    "the cold full block should compare at entry and return");
                t.Equals(
                    guardedFullStats.nativeBudgetExits,
                    uint64_t{1u},
                    "the exact block boundary should be a budget exit");

                t.Equals(
                    legacyFullStats.fullBlockGuards,
                    uint64_t{0u},
                    "the switch should emit no full guards");
                t.Equals(
                    legacyFullStats.preciseTailEntries,
                    uint64_t{1u},
                    "the checked form should classify one precise entry");
                t.Equals(
                    legacyFullStats.fullGuardPairs,
                    uint64_t{0u},
                    "the checked form should classify no full pairs");
                t.Equals(
                    legacyFullStats.preciseTailPairs,
                    uint64_t{64u},
                    "the checked form should classify every pair as precise");
                t.Equals(
                    legacyFullStats.budgetGuardComparisons,
                    uint64_t{65u},
                    "the checked form should compare every pair and exit");
                t.Equals(
                    legacyFullStats.nativeBudgetExits,
                    uint64_t{1u},
                    "the checked form should report the boundary exit");

                guardedState = initialSyntheticState();
                guardedEffects.clear();
                const VuRecompilerDiagnostics beforeTail =
                    guarded.diagnostics();
                const VuRunResult guardedTail =
                    guarded.run(guardedContext, 7u);
                const VuRecompilerDiagnostics afterTail =
                    guarded.diagnostics();
                t.IsTrue(
                    guardedTail.reason ==
                        VuExitReason::CycleBudget &&
                        guardedTail.executedCycles == 7u,
                    "the bounded entry should stop at the exact cut");
                t.Equals(
                    afterTail.fullBlockGuards -
                        beforeTail.fullBlockGuards,
                    uint64_t{0u},
                    "the bounded run should not take the full path");
                t.Equals(
                    afterTail.preciseTailEntries -
                        beforeTail.preciseTailEntries,
                    uint64_t{1u},
                    "the bounded run should take one precise tail");
                t.Equals(
                    afterTail.preciseTailPairs -
                        beforeTail.preciseTailPairs,
                    uint64_t{7u},
                    "the precise tail should retire exactly its budget");
                t.Equals(
                    afterTail.budgetGuardComparisons -
                        beforeTail.budgetGuardComparisons,
                    uint64_t{9u},
                    "the tail should include selector and terminal checks");
                t.Equals(
                    afterTail.nativeBudgetExits -
                        beforeTail.nativeBudgetExits,
                    uint64_t{1u},
                    "the precise terminal check should be reported");

                guardedState = initialSyntheticState();
                legacyState = guardedState;
                guardedEffects.clear();
                legacyEffects.clear();
                const VuRecompilerDiagnostics beforeShortSlice =
                    guarded.diagnostics();
                const VuRunResult guardedShortSlice =
                    guarded.run(guardedContext, 255u);
                const VuRunResult legacyShortSlice =
                    legacy.run(legacyContext, 255u);
                const VuRecompilerDiagnostics afterShortSlice =
                    guarded.diagnostics();
                t.IsTrue(
                    sameRunResult(
                        guardedShortSlice,
                        legacyShortSlice),
                    "the last short scheduler slice should match "
                    "the precise checker");
                t.IsTrue(
                    vuExecutionStatesEqual(
                        guardedState, legacyState,
                        &difference),
                    "the last short scheduler slice differs at " +
                        difference);
                t.IsTrue(
                    afterShortSlice.fullBlockGuards >
                        beforeShortSlice.fullBlockGuards,
                    "a 255-cycle slice should use guarded basic blocks");

                guardedState = initialSyntheticState();
                guardedEffects.clear();
                const VuProgramCacheDiagnostics cacheBeforeLongSlice =
                    guardedUnit.programCache().diagnostics();
                for (uint32_t warmup = 0u;
                     warmup < 8u; ++warmup)
                {
                    guardedState = initialSyntheticState();
                    guardedEffects.clear();
                    (void)guarded.run(
                        guardedContext, 256u);
                }
                const VuProgramCacheDiagnostics cacheAfterLongSliceWarmup =
                    guardedUnit.programCache().diagnostics();
                t.IsTrue(
                    cacheAfterLongSliceWarmup.compilations >
                        cacheBeforeLongSlice.compilations,
                    "guarded basic blocks and precise linear traces "
                    "should coexist in the cache");

                guardedState = initialSyntheticState();
                legacyState = guardedState;
                guardedEffects.clear();
                legacyEffects.clear();
                const VuRecompilerDiagnostics beforeLongSlice =
                    guarded.diagnostics();
                const VuRunResult guardedLongSlice =
                    guarded.run(guardedContext, 256u);
                const VuRunResult legacyLongSlice =
                    legacy.run(legacyContext, 256u);
                const VuRecompilerDiagnostics afterLongSlice =
                    guarded.diagnostics();
                t.IsTrue(
                    sameRunResult(
                        guardedLongSlice,
                        legacyLongSlice),
                    "the first long scheduler slice should match "
                    "the precise checker");
                t.IsTrue(
                    vuExecutionStatesEqual(
                        guardedState, legacyState,
                        &difference),
                    "the first long scheduler slice differs at " +
                        difference);
                t.Equals(
                    afterLongSlice.fullBlockGuards -
                        beforeLongSlice.fullBlockGuards,
                    uint64_t{0u},
                    "a 256-cycle slice should retain compact "
                    "precise linear traces");
                t.Equals(
                    afterLongSlice.preciseTailPairs -
                        beforeLongSlice.preciseTailPairs,
                    uint64_t{256u},
                    "the precise linear trace should classify every "
                    "long-slice pair");
                t.Equals(
                    afterLongSlice.preciseTailEntries -
                        beforeLongSlice.preciseTailEntries,
                    afterLongSlice.nativeBlocks -
                        beforeLongSlice.nativeBlocks,
                    "every long-slice block should use the precise form");
            });

        tc.Run(
            "VU0 branch targets wrap within its MicroMem extent",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(
                        VuUnitId::Vu0),
                    "reference VU0 memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(
                        VuUnitId::Vu0),
                    "native VU0 memory should initialize");
                const uint32_t jump =
                    makeVuBranchOperation(
                        0x24u, 1u, 0u, 0);
                writePair(
                    referenceFixture.code, 0u,
                    jump, kVuUpperNop);
                writePair(
                    nativeFixture.code, 0u,
                    jump, kVuUpperNop);
                referenceFixture.markCodeModified();
                nativeFixture.markCodeModified();

                VuExecutionState referenceState =
                    initialSyntheticState();
                referenceState.vi[1] = 0x201;
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState, referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);
                VuUnit referenceUnit(VuUnitId::Vu0);
                VuUnit nativeUnit(VuUnitId::Vu0);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);

                const VuRunResult referenceResult =
                    reference.run(referenceContext, 2u);
                const VuRunResult nativeResult =
                    native.run(nativeContext, 2u);
                t.IsTrue(
                    sameRunResult(
                        referenceResult, nativeResult),
                    "VU0 branch run results should match");
                t.Equals(
                    referenceState.pc, uint32_t{8u},
                    "the interpreter should wrap the VU0 jump target");
                t.Equals(
                    nativeState.pc, uint32_t{8u},
                    "native code should wrap the VU0 jump target");

                referenceState = initialSyntheticState();
                referenceState.vi[1] = 0x201;
                nativeState = referenceState;
                referenceEffects.clear();
                nativeEffects.clear();
                const VuRunResult resolvingReference =
                    reference.run(referenceContext, 4u);
                const VuRunResult resolvingNative =
                    native.run(nativeContext, 4u);
                t.IsTrue(
                    sameRunResult(
                        resolvingReference,
                        resolvingNative),
                    "the resolving VU0 wrap run should match");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "the resolving VU0 wrap state differs at " +
                        difference);

                referenceState = initialSyntheticState();
                referenceState.vi[1] = 0x201;
                nativeState = referenceState;
                referenceEffects.clear();
                nativeEffects.clear();
                const VuRecompilerDiagnostics beforeWarm =
                    native.diagnostics();
                const VuRunResult warmReference =
                    reference.run(referenceContext, 4u);
                const VuRunResult warmNative =
                    native.run(nativeContext, 4u);
                const VuRecompilerDiagnostics afterWarm =
                    native.diagnostics();
                t.IsTrue(
                    sameRunResult(
                        warmReference, warmNative),
                    "the warm linked VU0 wrap run should match");
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "the warm linked VU0 wrap state differs at " +
                        difference);
                t.Equals(
                    afterWarm.nativeEntries -
                        beforeWarm.nativeEntries,
                    uint64_t{1u},
                    "the warm wrapped target should stay native");
                t.Equals(
                    afterWarm.nativeBlocks -
                        beforeWarm.nativeBlocks,
                    uint64_t{2u},
                    "the warm wrap should execute source and target");
                t.Equals(
                    afterWarm.linkedEdges -
                        beforeWarm.linkedEdges,
                    uint64_t{1u},
                    "the wrapped dynamic target should tail-link");
            });

        tc.Run(
            "native conversions match saturated interpreter at every boundary",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");

                const auto configure =
                    [](uint8_t *code)
                    {
                        constexpr std::array<uint8_t, 8u>
                            destinations{
                                0xfu, 0xau, 0x5u, 0x3u,
                                0xcu, 0x9u, 0x6u, 0xfu};
                        for (uint32_t pair = 0u;
                             pair < 8u; ++pair)
                        {
                            uint32_t upper =
                                makeVuUpperSpecial(
                                    static_cast<uint8_t>(
                                        0x10u + pair),
                                    destinations[pair],
                                    static_cast<uint8_t>(
                                        17u + pair),
                                    static_cast<uint8_t>(
                                        1u + pair));
                            if (pair == 7u)
                                upper |= kVuUpperEnd;
                            writePair(
                                code, pair * 8u, 0u,
                                upper);
                        }
                        writePair(
                            code, 64u, 0u,
                            kVuUpperNop);
                    };
                configure(referenceFixture.code);
                configure(nativeFixture.code);
                referenceFixture.memory.markVU1CodeModified();
                nativeFixture.memory.markVU1CodeModified();

                const auto initialState =
                    []()
                    {
                        VuUnit seed(VuUnitId::Vu1);
                        VuExecutionState state =
                            seed.state();
                        state.active = true;

                        constexpr std::array<uint32_t, 4u>
                            integers{
                                0x7fffffffu, 0x80000000u,
                                123u, 0xffffff85u};
                        for (uint32_t reg = 1u;
                             reg <= 4u; ++reg)
                        {
                            std::memcpy(
                                state.vf[reg],
                                integers.data(),
                                sizeof(state.vf[reg]));
                        }
                        const std::array<
                            std::array<float, 4u>, 4u>
                            fixedInputs{{
                                {123.75f, 2147483648.0f,
                                 -2147483904.0f,
                                 std::numeric_limits<
                                     float>::quiet_NaN()},
                                {0.55f, -0.45f,
                                 134217728.0f,
                                 -134217728.0f},
                                {123.45f, -123.45f,
                                 524288.0f, -524289.0f},
                                {0.55f, -0.45f,
                                 65536.0f, -65536.0f},
                            }};
                        for (uint32_t reg = 5u;
                             reg <= 8u; ++reg)
                        {
                            std::memcpy(
                                state.vf[reg],
                                fixedInputs[reg - 5u].data(),
                                sizeof(state.vf[reg]));
                        }
                        for (uint32_t reg = 17u;
                             reg <= 24u; ++reg)
                        {
                            constexpr std::array<
                                uint32_t, 4u>
                                sentinel{
                                    0x11111111u,
                                    0x22222222u,
                                    0x33333333u,
                                    0x44444444u};
                            std::memcpy(
                                state.vf[reg],
                                sentinel.data(),
                                sizeof(state.vf[reg]));
                        }
                        return state;
                    };

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                for (uint32_t budget = 0u;
                     budget <= 9u; ++budget)
                {
                    VuExecutionState referenceState =
                        initialState();
                    VuExecutionState nativeState =
                        referenceState;
                    VuTransactionalSideEffectSink
                        referenceEffects;
                    VuTransactionalSideEffectSink
                        nativeEffects;
                    VuExecutionContext referenceContext =
                        makeContext(
                            referenceState,
                            referenceFixture,
                            referenceEffects);
                    VuExecutionContext nativeContext =
                        makeContext(
                            nativeState, nativeFixture,
                            nativeEffects);

                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext, budget);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext, budget);
                    const std::string prefix =
                        "budget " +
                        std::to_string(budget) + ": ";
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        prefix + "run result differs");
                    std::string difference;
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState, &difference),
                        prefix + "state differs at " +
                            difference);
                }

                const VuRecompilerDiagnostics diagnostics =
                    native.diagnostics();
                t.IsTrue(
                    diagnostics.inlinePairs >= 36u,
                    "all conversion prefixes should execute inline");
                t.IsTrue(
                    diagnostics.helperPairs > 0u,
                    "the E-bit delay pair should use the semantic helper");
            });

        tc.Run(
            "packed native FMAC flags match lane classes and masks",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                constexpr std::array<
                    std::array<uint32_t, 4u>, 3u>
                    sources{{
                        {
                            0x00000000u,
                            0xbf800000u,
                            0x7f800000u,
                            0xff800000u,
                        },
                        {
                            0x80000000u,
                            0x3f800000u,
                            0x7fc12345u,
                            0xffc54321u,
                        },
                        {
                            0x00000001u,
                            0x80000001u,
                            0x7f7fffffu,
                            0xff7fffffu,
                        },
                    }};
                constexpr std::array<
                    std::array<uint32_t, 4u>, 3u>
                    addends{{
                        {0u, 0u, 0u, 0u},
                        {0x80000000u, 0u, 0u, 0u},
                        {0u, 0u, 0u, 0u},
                    }};

                ScopedEnvironmentVariable blockLocal(
                    "PS2X_VU_BLOCK_LOCAL_VF_REGISTERS");
                for (uint32_t unitIndex = 0u;
                     unitIndex < 2u; ++unitIndex)
                {
                    const VuUnitId unit =
                        unitIndex == 0u
                            ? VuUnitId::Vu0
                            : VuUnitId::Vu1;
                    for (uint32_t resident = 0u;
                         resident < 2u; ++resident)
                    {
                        blockLocal.set(
                            resident != 0u ? "1" : "0");

                        VuNativeFixture referenceFixture;
                        VuNativeFixture nativeFixture;
                        t.IsTrue(
                            referenceFixture.initialize(unit),
                            "reference memory should initialize");
                        t.IsTrue(
                            nativeFixture.initialize(unit),
                            "native memory should initialize");
                        VuUnit referenceUnit(unit);
                        VuUnit nativeUnit(unit);
                        VuInterpreterBackend reference(
                            referenceUnit);
                        VuRecompilerBackend native(nativeUnit);

                        for (uint32_t destination = 0u;
                             destination < 16u;
                             ++destination)
                        {
                            const uint32_t add =
                                makeVuUpper(
                                    0x28u,
                                    static_cast<uint8_t>(
                                        destination),
                                    2u, 1u, 3u);
                            const uint32_t consume =
                                makeVuUpper(
                                    0x2bu, 0xfu,
                                    0u, 3u, 4u);
                            writePair(
                                referenceFixture.code,
                                0u, 0u, add);
                            writePair(
                                nativeFixture.code,
                                0u, 0u, add);
                            writePair(
                                referenceFixture.code,
                                8u, 0u, consume);
                            writePair(
                                nativeFixture.code,
                                8u, 0u, consume);
                            referenceFixture.markCodeModified();
                            nativeFixture.markCodeModified();

                            for (size_t pattern = 0u;
                                 pattern < sources.size();
                                 ++pattern)
                            {
                                VuExecutionState referenceState =
                                    initialSyntheticState();
                                VuExecutionState nativeState =
                                    referenceState;
                                std::memcpy(
                                    referenceState.vf[1],
                                    sources[pattern].data(),
                                    sizeof(referenceState.vf[1]));
                                std::memcpy(
                                    referenceState.vf[2],
                                    addends[pattern].data(),
                                    sizeof(referenceState.vf[2]));
                                nativeState = referenceState;
                                referenceState.mac = 0x55aau;
                                referenceState.status = 0x05a3u;
                                referenceState.pipeline.workingMac =
                                    0xa55au;
                                nativeState = referenceState;

                                VuTransactionalSideEffectSink
                                    referenceEffects;
                                VuTransactionalSideEffectSink
                                    nativeEffects;
                                VuExecutionContext referenceContext =
                                    makeContext(
                                        referenceState,
                                        referenceFixture,
                                        referenceEffects);
                                VuExecutionContext nativeContext =
                                    makeContext(
                                        nativeState,
                                        nativeFixture,
                                        nativeEffects);
                                const VuRunResult referenceResult =
                                    reference.run(
                                        referenceContext, 5u);
                                const VuRunResult nativeResult =
                                    native.run(
                                        nativeContext, 5u);
                                const std::string prefix =
                                    "unit " +
                                    std::to_string(unitIndex) +
                                    ", resident " +
                                    std::to_string(resident) +
                                    ", destination " +
                                    std::to_string(destination) +
                                    ", pattern " +
                                    std::to_string(pattern) + ": ";
                                t.IsTrue(
                                    sameRunResult(
                                        referenceResult,
                                        nativeResult),
                                    prefix + "run result differs");
                                std::string difference;
                                t.IsTrue(
                                    vuExecutionStatesEqual(
                                        referenceState,
                                        nativeState,
                                        &difference),
                                    prefix + "state differs at " +
                                        difference);
                            }
                        }
                    }
                }
            });

        tc.Run(
            "native DIV preserves scalar edges Q barriers and pair hazards",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                const uint32_t lower =
                    makeVuDiv(1u, 2u, 0u, 0u);
                const uint32_t upper =
                    makeVuUpper(
                        0x28u, 0xfu,
                        4u, 3u, 1u);
                writePair(
                    referenceFixture.code, 0u,
                    lower, upper);
                writePair(
                    nativeFixture.code, 0u,
                    lower, upper);
                referenceFixture.memory.markVU1CodeModified();
                nativeFixture.memory.markVU1CodeModified();

                struct Inputs
                {
                    float numerator;
                    float denominator;
                };
                const std::array<Inputs, 7u> inputs{{
                    {7.0f, 2.0f},
                    {7.0f, 0.0f},
                    {-7.0f, -0.0f},
                    {std::numeric_limits<
                         float>::quiet_NaN(),
                     0.0f},
                    {std::numeric_limits<
                         float>::infinity(),
                     std::numeric_limits<
                         float>::infinity()},
                    {std::numeric_limits<
                         float>::denorm_min(),
                     1.0f},
                    {1.0f,
                     std::numeric_limits<
                         float>::denorm_min()},
                }};

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                const VuRecompilerDiagnostics before =
                    native.diagnostics();
                for (size_t index = 0u;
                     index < inputs.size(); ++index)
                {
                    VuExecutionState referenceState =
                        initialSyntheticState();
                    VuExecutionState nativeState =
                        referenceState;
                    referenceState.vf[1][0] =
                        inputs[index].numerator;
                    nativeState.vf[1][0] =
                        inputs[index].numerator;
                    referenceState.vf[2][0] =
                        inputs[index].denominator;
                    nativeState.vf[2][0] =
                        inputs[index].denominator;
                    for (uint32_t lane = 0u;
                         lane < 4u; ++lane)
                    {
                        referenceState.vf[3][lane] =
                            10.0f +
                            static_cast<float>(lane);
                        nativeState.vf[3][lane] =
                            referenceState.vf[3][lane];
                        referenceState.vf[4][lane] =
                            20.0f +
                            static_cast<float>(lane);
                        nativeState.vf[4][lane] =
                            referenceState.vf[4][lane];
                    }
                    referenceState.q = -3.0f;
                    nativeState.q = -3.0f;
                    referenceState.pipeline.delayedQ = {
                        .active = true,
                        .result = 42.0f,
                        .cyclesRemaining = 5u,
                    };
                    nativeState.pipeline.delayedQ =
                        referenceState.pipeline.delayedQ;

                    VuTransactionalSideEffectSink
                        referenceEffects;
                    VuTransactionalSideEffectSink
                        nativeEffects;
                    VuExecutionContext referenceContext =
                        makeContext(
                            referenceState,
                            referenceFixture,
                            referenceEffects);
                    VuExecutionContext nativeContext =
                        makeContext(
                            nativeState, nativeFixture,
                            nativeEffects);
                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext, 1u);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext, 1u);
                    const std::string prefix =
                        "case " +
                        std::to_string(index) + ": ";
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        prefix + "run result differs");
                    std::string difference;
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState,
                            &difference),
                        prefix + "state differs at " +
                            difference);
                }
                const VuRecompilerDiagnostics after =
                    native.diagnostics();
                t.Equals(
                    after.inlinePairs -
                        before.inlinePairs,
                    static_cast<uint64_t>(
                        inputs.size()),
                    "every DIV case should execute inline");
                t.Equals(
                    after.helperPairs -
                        before.helperPairs,
                    uint64_t{0u},
                    "DIV should not route the pair helper");
            });

        tc.Run(
            "split native runs resume through a pending branch delay slot",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                configureSyntheticProgram(
                    referenceFixture.code);
                configureSyntheticProgram(
                    nativeFixture.code);
                referenceFixture.memory.markVU1CodeModified();
                nativeFixture.memory.markVU1CodeModified();

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                VuExecutionState referenceState =
                    initialSyntheticState();
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState, referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);

                constexpr std::array<uint32_t, 2u>
                    budgets{8u, 4u};
                for (size_t run = 0u;
                     run < budgets.size(); ++run)
                {
                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext,
                            budgets[run]);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext,
                            budgets[run]);
                    const std::string prefix =
                        "run " + std::to_string(run) +
                        ": ";
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        prefix + "run result differs");
                    std::string difference;
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState, &difference),
                        prefix + "state differs at " +
                            difference);
                }
            });

        tc.Run(
            "native VI backup specialization preserves known register identity and expiry",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable budgetGuards(
                    "PS2X_VU_BLOCK_BUDGET_GUARDS");
                budgetGuards.set("0");
                struct BackupCase
                {
                    const char *name;
                    std::array<uint32_t, 3u> setup;
                    uint8_t setupPairs;
                    uint8_t branchLeft;
                    uint8_t branchRight;
                    std::array<int32_t, 3u> initialVi;
                };
                const uint32_t lqiVi1 =
                    makeVuLqi(0xfu, 4u, 1u);
                const uint32_t lqiVi2 =
                    makeVuLqi(0xfu, 5u, 2u);
                const std::array<BackupCase, 3u> cases{{
                    {
                        "refresh same active register",
                        {lqiVi1, lqiVi1, 0u},
                        2u, 1u, 2u,
                        {5, 5, 0},
                    },
                    {
                        "replace active register",
                        {lqiVi1, lqiVi2, 0u},
                        2u, 2u, 3u,
                        {5, 9, 9},
                    },
                    {
                        "replace expired same register",
                        {lqiVi1, 0u, lqiVi1},
                        3u, 1u, 2u,
                        {5, 6, 0},
                    },
                }};

                for (const BackupCase &testCase : cases)
                {
                    VuNativeFixture referenceFixture;
                    VuNativeFixture nativeFixture;
                    t.IsTrue(
                        referenceFixture.initialize(),
                        std::string(testCase.name) +
                            ": reference memory should initialize");
                    t.IsTrue(
                        nativeFixture.initialize(),
                        std::string(testCase.name) +
                            ": native memory should initialize");
                    const uint32_t branchPair =
                        2u + testCase.setupPairs;
                    const auto configure =
                        [&](uint8_t *code)
                        {
                            for (uint32_t index = 0u;
                                 index <
                                     testCase.setupPairs;
                                 ++index)
                            {
                                writePair(
                                    code,
                                    (2u + index) * 8u,
                                    testCase.setup[index],
                                    kVuUpperNop);
                            }
                            writePair(
                                code, branchPair * 8u,
                                makeVuBranchOperation(
                                    0x28u,
                                    testCase.branchLeft,
                                    testCase.branchRight,
                                    3),
                                kVuUpperNop);
                            writePair(
                                code,
                                (branchPair + 2u) * 8u,
                                makeVuIaddiu(
                                    11u, 0u, 1),
                                kVuUpperNop |
                                    kVuUpperEnd);
                            writePair(
                                code,
                                (branchPair + 4u) * 8u,
                                makeVuIaddiu(
                                    11u, 0u, 2),
                                kVuUpperNop |
                                    kVuUpperEnd);
                        };
                    configure(referenceFixture.code);
                    configure(nativeFixture.code);
                    referenceFixture.markCodeModified();
                    nativeFixture.markCodeModified();

                    VuUnit referenceUnit(VuUnitId::Vu1);
                    VuUnit nativeUnit(VuUnitId::Vu1);
                    VuInterpreterBackend reference(
                        referenceUnit);
                    VuRecompilerBackend native(nativeUnit);
                    const uint32_t fullBudget =
                        branchPair + 6u;
                    for (uint32_t budget = 0u;
                         budget <= fullBudget;
                         ++budget)
                    {
                        VuExecutionState referenceState =
                            initialSyntheticState();
                        referenceState.vi[1] =
                            testCase.initialVi[0u];
                        referenceState.vi[2] =
                            testCase.initialVi[1u];
                        referenceState.vi[3] =
                            testCase.initialVi[2u];
                        VuExecutionState nativeState =
                            referenceState;
                        VuTransactionalSideEffectSink
                            referenceEffects;
                        VuTransactionalSideEffectSink
                            nativeEffects;
                        VuExecutionContext referenceContext =
                            makeContext(
                                referenceState,
                                referenceFixture,
                                referenceEffects);
                        VuExecutionContext nativeContext =
                            makeContext(
                                nativeState,
                                nativeFixture,
                                nativeEffects);
                        const VuRunResult referenceResult =
                            reference.run(
                                referenceContext, budget);
                        const VuRunResult nativeResult =
                            native.run(
                                nativeContext, budget);
                        const std::string prefix =
                            std::string(testCase.name) +
                            " budget " +
                            std::to_string(budget) +
                            ": ";
                        t.IsTrue(
                            sameRunResult(
                                referenceResult,
                                nativeResult),
                            prefix + "run result differs");
                        std::string difference;
                        t.IsTrue(
                            vuExecutionStatesEqual(
                                referenceState,
                                nativeState,
                                &difference),
                            prefix + "state differs at " +
                                difference);
                        if (budget == fullBudget)
                        {
                            t.Equals(
                                referenceState.vi[11],
                                int32_t{2},
                                prefix +
                                    "backup-visible branch marker");
                        }
                    }
                }
            });

        tc.Run(
            "every native branch form matches taken not-taken and split cuts",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                struct BranchCase
                {
                    const char *name;
                    uint32_t word;
                    int32_t first;
                    int32_t second;
                    bool taken;
                    bool backup = false;
                };
                const std::array<BranchCase, 17u> cases{{
                    {"B", makeVuBranchOperation(
                              0x20u, 0u, 0u, 3),
                     0, 0, true},
                    {"BAL", makeVuBranchOperation(
                                0x21u, 0u, 3u, 3),
                     0, 0, true},
                    {"JR", makeVuBranchOperation(
                               0x24u, 1u, 0u, 0),
                     4, 0, true},
                    {"JALR", makeVuBranchOperation(
                                 0x25u, 1u, 3u, 0),
                     4, 0, true},
                    {"IBEQ taken", makeVuBranchOperation(
                                        0x28u, 1u, 2u, 3),
                     5, 5, true},
                    {"IBEQ not taken", makeVuBranchOperation(
                                            0x28u, 1u, 2u, 3),
                     5, 6, false},
                    {"IBNE taken", makeVuBranchOperation(
                                        0x29u, 1u, 2u, 3),
                     5, 6, true},
                    {"IBNE not taken", makeVuBranchOperation(
                                            0x29u, 1u, 2u, 3),
                     5, 5, false},
                    {"IBLTZ taken", makeVuBranchOperation(
                                         0x2cu, 1u, 0u, 3),
                     -1, 0, true},
                    {"IBLTZ not taken", makeVuBranchOperation(
                                             0x2cu, 1u, 0u, 3),
                     0, 0, false},
                    {"IBGTZ taken", makeVuBranchOperation(
                                         0x2du, 1u, 0u, 3),
                     1, 0, true},
                    {"IBGTZ not taken", makeVuBranchOperation(
                                             0x2du, 1u, 0u, 3),
                     0, 0, false},
                    {"IBLEZ taken", makeVuBranchOperation(
                                         0x2eu, 1u, 0u, 3),
                     0, 0, true},
                    {"IBLEZ not taken", makeVuBranchOperation(
                                             0x2eu, 1u, 0u, 3),
                     1, 0, false},
                    {"IBGEZ taken", makeVuBranchOperation(
                                         0x2fu, 1u, 0u, 3),
                     0, 0, true},
                    {"IBGEZ not taken", makeVuBranchOperation(
                                             0x2fu, 1u, 0u, 3),
                     -1, 0, false},
                    {"IBEQ VI backup", makeVuBranchOperation(
                                            0x28u, 1u, 2u, 3),
                     6, 5, true, true},
                }};

                for (const BranchCase &branch : cases)
                {
                    VuNativeFixture referenceFixture;
                    VuNativeFixture nativeFixture;
                    t.IsTrue(
                        referenceFixture.initialize(),
                        std::string(branch.name) +
                            ": reference memory should initialize");
                    t.IsTrue(
                        nativeFixture.initialize(),
                        std::string(branch.name) +
                            ": native memory should initialize");
                    const auto configure =
                        [&](uint8_t *code)
                        {
                            writePair(
                                code, 0u, branch.word,
                                kVuUpperNop);
                            writePair(
                                code, 8u,
                                makeVuIaddiu(
                                    10u, 10u, 1),
                                kVuUpperNop);
                            writePair(
                                code, 16u,
                                makeVuIaddiu(
                                    11u, 0u, 1),
                                kVuUpperNop |
                                    kVuUpperEnd);
                            writePair(
                                code, 24u, 0u,
                                kVuUpperNop);
                            writePair(
                                code, 32u,
                                makeVuIaddiu(
                                    11u, 0u, 2),
                                kVuUpperNop |
                                    kVuUpperEnd);
                            writePair(
                                code, 40u, 0u,
                                kVuUpperNop);
                        };
                    configure(referenceFixture.code);
                    configure(nativeFixture.code);
                    referenceFixture.memory.
                        markVU1CodeModified();
                    nativeFixture.memory.
                        markVU1CodeModified();

                    const auto initialState =
                        [&]()
                        {
                            VuUnit seed(VuUnitId::Vu1);
                            VuExecutionState state =
                                seed.state();
                            state.active = true;
                            state.vi[1] = branch.first;
                            state.vi[2] = branch.second;
                            state.vi[3] = 0x1234;
                            if (branch.backup)
                            {
                                state.viBackupCycles = 2u;
                                state.viBackupRegister = 1u;
                                state.viBackupValue = 5;
                            }
                            return state;
                        };

                    VuUnit referenceUnit(VuUnitId::Vu1);
                    VuUnit nativeUnit(VuUnitId::Vu1);
                    VuInterpreterBackend reference(
                        referenceUnit);
                    VuRecompilerBackend native(nativeUnit);
                    for (uint32_t budget = 0u;
                         budget <= 4u; ++budget)
                    {
                        VuExecutionState referenceState =
                            initialState();
                        VuExecutionState nativeState =
                            referenceState;
                        VuTransactionalSideEffectSink
                            referenceEffects;
                        VuTransactionalSideEffectSink
                            nativeEffects;
                        VuExecutionContext referenceContext =
                            makeContext(
                                referenceState,
                                referenceFixture,
                                referenceEffects);
                        VuExecutionContext nativeContext =
                            makeContext(
                                nativeState,
                                nativeFixture,
                                nativeEffects);
                        const VuRecompilerDiagnostics before =
                            native.diagnostics();
                        const VuRunResult referenceResult =
                            reference.run(
                                referenceContext, budget);
                        const VuRunResult nativeResult =
                            native.run(
                                nativeContext, budget);
                        const std::string prefix =
                            std::string(branch.name) +
                            " budget " +
                            std::to_string(budget) +
                            ": ";
                        t.IsTrue(
                            sameRunResult(
                                referenceResult,
                                nativeResult),
                            prefix + "run result differs");
                        std::string difference;
                        t.IsTrue(
                            vuExecutionStatesEqual(
                                referenceState,
                                nativeState,
                                &difference),
                            prefix + "state differs at " +
                                difference);
                        if (budget == 1u)
                        {
                            const VuRecompilerDiagnostics
                                after =
                                    native.diagnostics();
                            t.Equals(
                                after.inlinePairs,
                                before.inlinePairs + 1u,
                                prefix +
                                    "branch should inline");
                            t.Equals(
                                after.helperPairs,
                                before.helperPairs,
                                prefix +
                                    "branch should not call a helper");
                        }
                        if (budget == 4u)
                        {
                            t.Equals(
                                referenceState.vi[11],
                                branch.taken
                                    ? int32_t{2}
                                    : int32_t{1},
                                prefix +
                                    "control-flow marker");
                            if ((branch.word >> 25u) ==
                                    0x21u ||
                                (branch.word >> 25u) ==
                                    0x25u)
                            {
                                t.Equals(
                                    referenceState.vi[3],
                                    int32_t{2},
                                    prefix +
                                        "link value");
                            }
                        }
                    }

                    for (const uint32_t firstBudget :
                         {1u, 2u})
                    {
                        VuExecutionState referenceState =
                            initialState();
                        VuExecutionState nativeState =
                            referenceState;
                        VuTransactionalSideEffectSink
                            referenceEffects;
                        VuTransactionalSideEffectSink
                            nativeEffects;
                        VuExecutionContext referenceContext =
                            makeContext(
                                referenceState,
                                referenceFixture,
                                referenceEffects);
                        VuExecutionContext nativeContext =
                            makeContext(
                                nativeState,
                                nativeFixture,
                                nativeEffects);
                        const std::array<uint32_t, 2u>
                            budgets{
                                firstBudget,
                                4u - firstBudget};
                        for (size_t run = 0u;
                             run < budgets.size();
                             ++run)
                        {
                            const VuRunResult
                                referenceResult =
                                    reference.run(
                                        referenceContext,
                                        budgets[run]);
                            const VuRunResult nativeResult =
                                native.run(
                                    nativeContext,
                                    budgets[run]);
                            const std::string prefix =
                                std::string(branch.name) +
                                " split " +
                                std::to_string(
                                    firstBudget) +
                                " run " +
                                std::to_string(run) +
                                ": ";
                            t.IsTrue(
                                sameRunResult(
                                    referenceResult,
                                    nativeResult),
                                prefix +
                                    "run result differs");
                            std::string difference;
                            t.IsTrue(
                                vuExecutionStatesEqual(
                                    referenceState,
                                    nativeState,
                                    &difference),
                                prefix +
                                    "state differs at " +
                                    difference);
                        }
                    }
                }
            });

        tc.Run(
            "canonical self links reduce host entries with exact profiles",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable guards(
                    "PS2X_VU_BLOCK_BUDGET_GUARDS");
                guards.set("1");
                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(
                        VuUnitId::Vu0),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(
                        VuUnitId::Vu0),
                    "native memory should initialize");
                const uint32_t loop =
                    makeVuBranchOperation(
                        0x20u, 0u, 0u, -1);
                writePair(
                    referenceFixture.code, 0u,
                    loop, kVuUpperNop);
                writePair(
                    nativeFixture.code, 0u,
                    loop, kVuUpperNop);
                writePair(
                    referenceFixture.code, 8u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop);
                writePair(
                    nativeFixture.code, 8u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop);
                referenceFixture.markCodeModified();
                nativeFixture.markCodeModified();

                VuUnit referenceUnit(VuUnitId::Vu0);
                VuUnit nativeUnit(VuUnitId::Vu0);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(
                    nativeUnit,
                    VuBlockProfilingConfiguration{
                        .enabled = true,
                        .maximumRecords = 16u,
                    });
                VuExecutionState referenceState =
                    initialSyntheticState();
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState,
                        referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState,
                        nativeFixture,
                        nativeEffects);
                VuProgramKey key;
                std::string diagnostic;
                t.IsTrue(
                    native.programKey(
                        nativeContext,
                        VuCompilationMode::Normal,
                        key, &diagnostic),
                    "the loop key should be valid: " +
                        diagnostic);

                constexpr uint32_t budget = 200u;
                const VuRunResult referenceResult =
                    reference.run(
                        referenceContext, budget);
                const VuRunResult cold =
                    native.run(nativeContext, budget);
                t.IsTrue(
                    sameRunResult(
                        referenceResult, cold),
                    "cold linked run result should match");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "cold linked state differs at " +
                        difference);

                const VuRecompilerDiagnostics coldStats =
                    native.diagnostics();
                t.Equals(
                    coldStats.nativePairs,
                    uint64_t{budget},
                    "cold linked pairs should be exact");
                t.Equals(
                    coldStats.nativeBlocks,
                    uint64_t{100u},
                    "two-pair loop blocks should be exact");
                t.Equals(
                    coldStats.nativeEntries,
                    uint64_t{2u},
                    "one cold miss should require two host entries");
                t.Equals(
                    coldStats.linkedEdges,
                    uint64_t{98u},
                    "the remainder should tail-link");
                t.Equals(
                    coldStats.slowLinkExits,
                    uint64_t{1u},
                    "the cold edge should resolve once");
                t.Equals(
                    coldStats.resolvedLinks,
                    uint64_t{1u},
                    "the self edge should populate once");
                t.Equals(
                    coldStats.blockCompletes +
                        coldStats.nativeBudgetExits,
                    coldStats.nativeBlocks,
                    "every loop block should complete or exhaust the budget");

                nativeState = initialSyntheticState();
                nativeEffects.clear();
                const VuRecompilerDiagnostics beforeWarm =
                    native.diagnostics();
                const VuRunResult warm =
                    native.run(nativeContext, budget);
                const VuRecompilerDiagnostics afterWarm =
                    native.diagnostics();
                t.IsTrue(
                    warm.reason ==
                        VuExitReason::CycleBudget,
                    "the warm loop should exhaust its budget");
                t.Equals(
                    afterWarm.nativeEntries -
                        beforeWarm.nativeEntries,
                    uint64_t{1u},
                    "a warm chain should cross the adapter once");
                t.Equals(
                    afterWarm.nativeBlocks -
                        beforeWarm.nativeBlocks,
                    uint64_t{100u},
                    "the warm chain should execute every block");
                t.Equals(
                    afterWarm.linkedEdges -
                        beforeWarm.linkedEdges,
                    uint64_t{99u},
                    "every warm inter-block edge should link");
                t.Equals(
                    afterWarm.slowLinkExits -
                        beforeWarm.slowLinkExits,
                    uint64_t{0u},
                    "the warm edge should not revisit the resolver");
                t.Equals(
                    afterWarm.fullBlockGuards -
                        beforeWarm.fullBlockGuards,
                    uint64_t{100u},
                    "every linked successor should run its own guard");
                t.Equals(
                    afterWarm.preciseTailEntries -
                        beforeWarm.preciseTailEntries,
                    uint64_t{0u},
                    "the exact full-block loop should avoid precise tails");
                t.Equals(
                    afterWarm.fullGuardPairs -
                        beforeWarm.fullGuardPairs,
                    uint64_t{budget},
                    "full guards should cover every linked pair");
                t.Equals(
                    afterWarm.budgetGuardComparisons -
                        beforeWarm.budgetGuardComparisons,
                    uint64_t{101u},
                    "the loop should guard each block plus its terminal target");
                t.Equals(
                    afterWarm.nativeBudgetExits -
                        beforeWarm.nativeBudgetExits,
                    uint64_t{1u},
                    "the exact linked boundary should report one budget exit");
                uint64_t nativeExits = 0u;
                for (const uint64_t count :
                     afterWarm.nativeExitReasons)
                {
                    nativeExits += count;
                }
                t.Equals(
                    nativeExits,
                    afterWarm.nativeBlocks,
                    "native exit reasons should cover every block");
                t.Equals(
                    afterWarm.nativeExitReasons[
                        static_cast<size_t>(
                            VuNativeBlockExit::
                                BlockComplete)] +
                        afterWarm.nativeExitReasons[
                            static_cast<size_t>(
                                VuNativeBlockExit::
                                    CycleBudget)],
                    afterWarm.nativeBlocks,
                    "every loop block should report completion or budget exit");

                const VuBlockProfilingSnapshot profile =
                    native
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                uint64_t executions = 0u;
                uint64_t pairs = 0u;
                uint64_t linkedEdges = 0u;
                uint64_t exits = 0u;
                for (const VuBlockProfileSnapshot &block :
                     profile.blocks)
                {
                    executions += block.executions;
                    pairs += block.guestPairs;
                    linkedEdges += block.linkedEdges;
                    for (const uint64_t count :
                         block.exitReasons)
                    {
                        exits += count;
                    }
                }
                t.Equals(
                    executions,
                    afterWarm.nativeBlocks,
                    "profile executions should include linked blocks");
                t.Equals(
                    pairs, afterWarm.nativePairs,
                    "profile pairs should include linked blocks");
                t.Equals(
                    linkedEdges,
                    afterWarm.linkedEdges,
                    "profile links should reconcile globally");
                t.Equals(
                    exits, executions,
                    "every profiled block should have one exit");

                const VuProgramHandle handle =
                    nativeUnit.programCache().lookup(key);
                const VuCompiledProgram *const program =
                    nativeUnit.programCache().resolve(
                        handle);
                t.IsNotNull(
                    program,
                    "the linked source should remain resident");
                bool foundStaticSelfLink = false;
                if (program && program->outgoingLinks)
                {
                    for (const VuNativeLinkSlot &slot :
                         program->outgoingLinks->
                             staticTargets)
                    {
                        if (slot.targetPc == 0u &&
                            slot.targetEntry != 0u)
                        {
                            foundStaticSelfLink = true;
                        }
                    }
                }
                t.IsTrue(
                    foundStaticSelfLink,
                    "the direct branch should use a static slot");
            });

        tc.Run(
            "block-linking switch retains canonical dispatch",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable linking(
                    "PS2X_VU_BLOCK_LINKING");
                linking.set("0");
                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuBranchOperation(
                        0x20u, 0u, 0u, -1),
                    kVuUpperNop);
                writePair(
                    fixture.code, 8u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                const VuRunResult result =
                    backend.run(context, 40u);
                const VuRecompilerDiagnostics diagnostics =
                    backend.diagnostics();
                t.IsFalse(
                    backend.blockLinkingEnabled(),
                    "the diagnostic switch should be visible");
                t.IsTrue(
                    result.reason ==
                        VuExitReason::CycleBudget,
                    "canonical dispatch should preserve the result");
                t.Equals(
                    state.vi[1], int32_t{20},
                    "canonical dispatch should preserve state");
                t.Equals(
                    diagnostics.nativeBlocks,
                    uint64_t{20u},
                    "canonical blocks should remain exact");
                t.Equals(
                    diagnostics.nativeEntries,
                    diagnostics.nativeBlocks,
                    "every unlinked block should enter from C++");
                t.Equals(
                    diagnostics.linkedEdges,
                    uint64_t{0u},
                    "disabled linking must not use a slot");
                t.Equals(
                    diagnostics.slowLinkExits,
                    uint64_t{0u},
                    "disabled linking must not invoke the resolver");
            });

        tc.Run(
            "dynamic jump targets populate bounded writable slots",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuBranchOperation(
                        0x24u, 1u, 0u, 0),
                    kVuUpperNop);
                writePair(
                    fixture.code, 8u,
                    makeVuIaddiu(2u, 2u, 1),
                    kVuUpperNop);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                state.vi[1] = 0;
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                VuProgramKey key;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        key, &diagnostic),
                    "the dynamic loop key should be valid: " +
                        diagnostic);

                const VuRunResult cold =
                    backend.run(context, 40u);
                t.IsTrue(
                    cold.reason ==
                        VuExitReason::CycleBudget,
                    "the dynamic loop should exhaust its budget");
                t.Equals(
                    state.vi[2], int32_t{20},
                    "every dynamic delay slot should retire");
                const VuRecompilerDiagnostics coldStats =
                    backend.diagnostics();
                t.Equals(
                    coldStats.nativeEntries,
                    uint64_t{2u},
                    "one cold dynamic edge should return twice");
                t.Equals(
                    coldStats.linkedEdges,
                    uint64_t{18u},
                    "the cold remainder should link");

                const VuProgramHandle handle =
                    unit.programCache().lookup(key);
                const VuCompiledProgram *const program =
                    unit.programCache().resolve(handle);
                t.IsNotNull(
                    program,
                    "the dynamic source should be resident");
                bool foundDynamicSelfLink = false;
                if (program && program->outgoingLinks)
                {
                    for (const VuNativeLinkSlot &slot :
                         program->outgoingLinks->
                             dynamicTargets)
                    {
                        if (slot.targetPc == 0u &&
                            slot.targetEntry != 0u)
                        {
                            foundDynamicSelfLink = true;
                        }
                    }
                }
                t.IsTrue(
                    foundDynamicSelfLink,
                    "JR should populate a dynamic slot");

                state = initialSyntheticState();
                state.vi[1] = 0;
                effects.clear();
                const VuRecompilerDiagnostics before =
                    backend.diagnostics();
                (void)backend.run(context, 40u);
                const VuRecompilerDiagnostics after =
                    backend.diagnostics();
                t.Equals(
                    after.nativeEntries -
                        before.nativeEntries,
                    uint64_t{1u},
                    "the warm dynamic target should stay in native code");
                t.Equals(
                    after.linkedEdges -
                        before.linkedEdges,
                    uint64_t{19u},
                    "all warm dynamic edges should link");
            });

        tc.Run(
            "normal and instrumented chains remain separate",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuBranchOperation(
                        0x20u, 0u, 0u, -1),
                    kVuUpperNop);
                writePair(
                    fixture.code, 8u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                std::vector<uint64_t> observedIndices;
                unit.setInstructionObserver(
                    [&](uint64_t index, uint32_t,
                        uint32_t, uint32_t,
                        const VuExecutionState &)
                    {
                        observedIndices.push_back(index);
                    });
                unit.setInstructionObserverEnabled(true);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                VuProgramKey normalKey;
                VuProgramKey instrumentedKey;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        normalKey, &diagnostic),
                    "the normal key should be valid: " +
                        diagnostic);
                (void)backend.run(context, 20u);
                const VuRecompilerDiagnostics
                    beforeInstrumented =
                        backend.diagnostics();

                state = initialSyntheticState();
                effects.clear();
                context.enableInstrumentation = true;
                context.enableNativeInstrumentation = true;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Instrumented,
                        instrumentedKey, &diagnostic),
                    "the instrumented key should be valid: " +
                        diagnostic);
                const VuRunResult cold =
                    backend.run(context, 20u);
                const VuRecompilerDiagnostics afterCold =
                    backend.diagnostics();
                t.IsTrue(
                    cold.reason ==
                        VuExitReason::CycleBudget,
                    "the instrumented loop should exhaust its budget");
                t.Equals(
                    observedIndices.size(), size_t{20u},
                    "every linked instrumented pair should be observed");
                t.Equals(
                    afterCold.instrumentedNativeEntries -
                        beforeInstrumented.
                            instrumentedNativeEntries,
                    uint64_t{2u},
                    "one cold instrumented edge should return twice");
                t.Equals(
                    afterCold.instrumentedNativeBlocks -
                        beforeInstrumented.
                            instrumentedNativeBlocks,
                    uint64_t{10u},
                    "instrumented linked blocks should be exact");
                t.Equals(
                    afterCold.linkedEdges -
                        beforeInstrumented.linkedEdges,
                    uint64_t{8u},
                    "the cold instrumented remainder should link");

                state = initialSyntheticState();
                effects.clear();
                observedIndices.clear();
                const VuRecompilerDiagnostics beforeWarm =
                    backend.diagnostics();
                (void)backend.run(context, 20u);
                const VuRecompilerDiagnostics afterWarm =
                    backend.diagnostics();
                t.Equals(
                    afterWarm.instrumentedNativeEntries -
                        beforeWarm.instrumentedNativeEntries,
                    uint64_t{1u},
                    "the warm instrumented chain should enter once");
                t.Equals(
                    afterWarm.linkedEdges -
                        beforeWarm.linkedEdges,
                    uint64_t{9u},
                    "all warm instrumented edges should link");
                t.Equals(
                    observedIndices.size(), size_t{20u},
                    "warm linked observation should stay exact");

                const VuProgramHandle normalHandle =
                    unit.programCache().lookup(normalKey);
                const VuProgramHandle instrumentedHandle =
                    unit.programCache().lookup(
                        instrumentedKey);
                const VuCompiledProgram *const normal =
                    unit.programCache().resolve(
                        normalHandle);
                const VuCompiledProgram *const instrumented =
                    unit.programCache().resolve(
                        instrumentedHandle);
                t.IsNotNull(
                    normal,
                    "the normal chain should remain resident");
                t.IsNotNull(
                    instrumented,
                    "the instrumented chain should remain resident");
                if (normal && instrumented &&
                    normal->outgoingLinks &&
                    instrumented->outgoingLinks)
                {
                    const auto selfTarget =
                        [](const VuNativeLinkState &links)
                        {
                            for (const VuNativeLinkSlot &slot :
                                 links.staticTargets)
                            {
                                if (slot.targetPc == 0u)
                                    return slot.targetEntry;
                            }
                            return uintptr_t{0u};
                        };
                    const uintptr_t normalTarget =
                        selfTarget(
                            *normal->outgoingLinks);
                    const uintptr_t instrumentedTarget =
                        selfTarget(
                            *instrumented->
                                outgoingLinks);
                    t.IsTrue(
                        normalTarget != 0u &&
                        instrumentedTarget != 0u,
                        "both compilation modes should link");
                    t.IsTrue(
                        normalTarget !=
                            instrumentedTarget,
                        "the modes must target distinct native code");
                }
            });

        tc.Run(
            "linked generations reject stale targets before reuse",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                const uint32_t loop =
                    makeVuBranchOperation(
                        0x20u, 0u, 0u, -1);
                writePair(
                    fixture.code, 0u,
                    loop, kVuUpperNop);
                writePair(
                    fixture.code, 8u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                VuProgramKey originalKey;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        originalKey, &diagnostic),
                    "the original key should be valid: " +
                        diagnostic);
                (void)backend.run(context, 40u);
                const VuProgramHandle originalHandle =
                    unit.programCache().lookup(
                        originalKey);
                const VuProgramCacheDiagnostics
                    beforeIdentical =
                        unit.programCache().
                            diagnostics();
                const VuRecompilerDiagnostics
                    nativeBeforeIdentical =
                        backend.diagnostics();

                fixture.markCodeModified();
                state = initialSyntheticState();
                effects.clear();
                const VuRunResult identical =
                    backend.run(context, 40u);
                t.IsTrue(
                    identical.reason ==
                        VuExitReason::CycleBudget,
                    "an identical generation should complete safely");
                t.Equals(
                    state.vi[1], int32_t{20},
                    "identical code should retain loop semantics");
                t.IsNull(
                    unit.programCache().resolve(
                        originalHandle),
                    "the prior generation handle must be stale");
                const VuProgramCacheDiagnostics
                    afterIdentical =
                        unit.programCache().
                            diagnostics();
                const VuRecompilerDiagnostics
                    nativeAfterIdentical =
                        backend.diagnostics();
                t.Equals(
                    afterIdentical.linkInvalidations -
                        beforeIdentical.linkInvalidations,
                    uint64_t{0u},
                    "generation guards should avoid a physical slot walk");
                t.Equals(
                    nativeAfterIdentical.slowLinkExits -
                        nativeBeforeIdentical.slowLinkExits,
                    uint64_t{1u},
                    "the cleared edge should resolve once");

                writePair(
                    fixture.code, 8u,
                    makeVuIaddiu(1u, 1u, 2),
                    kVuUpperNop);
                fixture.markCodeModified();
                state = initialSyntheticState();
                effects.clear();
                const VuProgramCacheDiagnostics
                    beforeChanged =
                        unit.programCache().
                            diagnostics();
                const VuRunResult changed =
                    backend.run(context, 40u);
                t.IsTrue(
                    changed.reason ==
                        VuExitReason::CycleBudget,
                    "changed code should execute safely");
                t.Equals(
                    state.vi[1], int32_t{40},
                    "the chain must execute replacement code");
                const VuProgramCacheDiagnostics
                    afterChanged =
                        unit.programCache().
                            diagnostics();
                t.Equals(
                    afterChanged.linkInvalidations -
                        beforeChanged.linkInvalidations,
                    uint64_t{0u},
                    "resident replacement images should invalidate logically");
                t.Equals(
                    afterChanged.compilations -
                        beforeChanged.compilations,
                    uint64_t{1u},
                    "changed content should compile once");
                t.Equals(
                    backend.diagnostics().faultExits,
                    uint64_t{0u},
                    "generation churn must not fault");
            });

        tc.Run(
            "maximum-size inline block fits the bounded emitter",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                const auto configure =
                    [](uint8_t *code)
                    {
                        for (uint32_t pair = 0u;
                             pair < 64u; ++pair)
                        {
                            writePair(
                                code, pair * 8u,
                                makeVuLq(
                                    0xfu, 5u, 1u,
                                    static_cast<int16_t>(
                                        pair & 0x0fu)),
                                makeVuUpper(
                                    0x29u, 0xfu,
                                    2u, 3u, 4u));
                        }
                    };
                configure(referenceFixture.code);
                configure(nativeFixture.code);
                referenceFixture.memory.markVU1CodeModified();
                nativeFixture.memory.markVU1CodeModified();

                VuExecutionState referenceState =
                    initialSyntheticState();
                referenceState.acc[0] = 1.0f;
                referenceState.acc[1] = -2.0f;
                referenceState.acc[2] = 3.0f;
                referenceState.acc[3] = -4.0f;
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState, referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);
                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);

                const VuRunResult referenceResult =
                    reference.run(referenceContext, 64u);
                const VuRunResult nativeResult =
                    native.run(nativeContext, 64u);
                t.IsTrue(
                    sameRunResult(
                        referenceResult, nativeResult),
                    "maximum block run result differs: " +
                        native.lastDiagnostic());
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "maximum block state differs at " +
                        difference);
                const VuProgramCacheDiagnostics cache =
                    nativeUnit.programCache().diagnostics();
                t.Equals(
                    cache.compilations, uint64_t{1u},
                    "one maximum block should compile once");
                t.IsTrue(
                    cache.generatedBytes > 0u,
                    "maximum block should emit native bytes");
                const VuRecompilerDiagnostics diagnostics =
                    native.diagnostics();
                t.Equals(
                    diagnostics.inlinePairs, uint64_t{64u},
                    "every maximum-block pair should inline");
                t.Equals(
                    diagnostics.faultExits, uint64_t{0u},
                    "maximum block should not fault");
            });

        tc.Run(
            "warm hits reuse exact images and replace changed native code",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuIaddiu(1u, 0u, 1),
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 8u, 0u, kVuUpperNop);
                fixture.memory.markVU1CodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                VuProgramKey oldKey;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context, VuCompilationMode::Normal,
                        oldKey, &diagnostic),
                    "the initial key should be valid");

                const VuRunResult cold =
                    backend.run(context, 2u);
                t.IsTrue(
                    cold.reason == VuExitReason::ProgramEnded,
                    "the cold native run should complete");
                t.Equals(
                    state.vi[1], int32_t{1},
                    "cold code should execute the original value");
                const VuProgramHandle oldHandle =
                    unit.programCache().lookup(oldKey);
                t.IsTrue(
                    oldHandle.valid(),
                    "the cold block should be resident");

                state = initialSyntheticState();
                effects.clear();
                const VuRunResult warm =
                    backend.run(context, 2u);
                t.IsTrue(
                    warm.reason == VuExitReason::ProgramEnded,
                    "the warm native run should complete");

                fixture.memory.markVU1CodeModified();
                state = initialSyntheticState();
                effects.clear();
                const VuRunResult identical =
                    backend.run(context, 2u);
                t.IsTrue(
                    identical.reason ==
                        VuExitReason::ProgramEnded,
                    "an identical reupload generation should complete");
                t.Equals(
                    state.vi[1], int32_t{1},
                    "an identical reupload should preserve semantics");
                t.IsNull(
                    unit.programCache().resolve(oldHandle),
                    "the write generation must stale its old direct handle");

                writePair(
                    fixture.code, 0u,
                    makeVuIaddiu(1u, 0u, 2),
                    kVuUpperNop | kVuUpperEnd);
                fixture.memory.markVU1CodeModified();
                state = initialSyntheticState();
                effects.clear();
                const VuRunResult replaced =
                    backend.run(context, 2u);
                t.IsTrue(
                    replaced.reason ==
                        VuExitReason::ProgramEnded,
                    "the replacement generation should complete");
                t.Equals(
                    state.vi[1], int32_t{2},
                    "the replacement generation must execute new code");
                t.IsNull(
                    unit.programCache().resolve(oldHandle),
                    "the pre-write handle must not resolve");

                const VuProgramCacheDiagnostics cache =
                    unit.programCache().diagnostics();
                t.Equals(
                    cache.compilations, uint64_t{2u},
                    "only cold and changed content should compile");
                t.Equals(
                    cache.invalidations, uint64_t{2u},
                    "both generation transitions should stale handles");
                t.Equals(
                    cache.generationRetentions, uint64_t{2u},
                    "content-tracked generations should retain residents");
                t.IsTrue(
                    cache.crossGenerationHits >= 1u,
                    "the identical image should hit across generations");
                t.Equals(
                    cache.residentPrograms, size_t{2u},
                    "changed and recurring code images should remain resident");
                t.IsTrue(
                    cache.hits >= 2u,
                    "explicit lookup and warm execution should hit");
            });

        tc.Run(
            "unsupported pair exits before architectural mutation",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u, 0x04000000u,
                    makeVuUpper(
                        0x34u, 0xfu, 2u, 1u, 3u));
                fixture.memory.markVU1CodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                const VuExecutionState initial = state;
                std::vector<uint8_t> initialData(
                    fixture.data,
                    fixture.data + PS2_VU1_DATA_SIZE);
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);

                const VuRunResult result =
                    backend.run(context, 1u);
                t.IsTrue(
                    result.reason ==
                        VuExitReason::UnsupportedInstruction,
                    "unsupported IR should have a structured exit");
                t.Equals(
                    result.executedCycles, uint32_t{0u},
                    "unsupported IR must not retire a pair");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        state, initial, &difference),
                    "unsupported IR mutated " + difference);
                t.IsTrue(
                    std::memcmp(
                        fixture.data, initialData.data(),
                        initialData.size()) == 0,
                    "unsupported IR mutated VU data");
                t.IsTrue(
                    effects.path1Packets().empty(),
                    "unsupported IR published side effects");
            });

        tc.Run(
            "unsupported boundary preserves cycle-budget precedence",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable guards(
                    "PS2X_VU_BLOCK_BUDGET_GUARDS");
                guards.set("1");
                VuNativeFixture checkedFixture;
                VuNativeFixture guardedFixture;
                t.IsTrue(
                    checkedFixture.initialize(),
                    "checked memory should initialize");
                t.IsTrue(
                    guardedFixture.initialize(),
                    "guarded memory should initialize");
                const auto configure =
                    [](uint8_t *code)
                    {
                        writePair(
                            code, 0u,
                            makeVuIaddiu(
                                1u, 1u, 1),
                            kVuUpperNop);
                        writePair(
                            code, 8u,
                            0x04000000u,
                            makeVuUpper(
                                0x34u, 0xfu,
                                2u, 1u, 3u));
                    };
                configure(checkedFixture.code);
                configure(guardedFixture.code);
                checkedFixture.markCodeModified();
                guardedFixture.markCodeModified();

                VuUnit guardedUnit(VuUnitId::Vu1);
                VuRecompilerBackend guarded(
                    guardedUnit);
                guards.set("0");
                VuUnit checkedUnit(VuUnitId::Vu1);
                VuRecompilerBackend checked(
                    checkedUnit);
                for (uint32_t budget = 0u;
                     budget <= 2u; ++budget)
                {
                    VuExecutionState checkedState =
                        initialSyntheticState();
                    VuExecutionState guardedState =
                        checkedState;
                    VuTransactionalSideEffectSink
                        checkedEffects;
                    VuTransactionalSideEffectSink
                        guardedEffects;
                    VuExecutionContext checkedContext =
                        makeContext(
                            checkedState,
                            checkedFixture,
                            checkedEffects);
                    VuExecutionContext guardedContext =
                        makeContext(
                            guardedState,
                            guardedFixture,
                            guardedEffects);

                    const VuRunResult checkedResult =
                        checked.run(
                            checkedContext, budget);
                    const VuRunResult guardedResult =
                        guarded.run(
                            guardedContext, budget);
                    const std::string prefix =
                        "budget " +
                        std::to_string(budget) +
                        ": ";
                    t.IsTrue(
                        sameRunResult(
                            checkedResult,
                            guardedResult),
                        prefix + "run result differs (" +
                            std::string(
                                vuExitReasonName(
                                    checkedResult.reason)) +
                            "/" +
                            std::to_string(
                                checkedResult.executedCycles) +
                            " versus " +
                            std::string(
                                vuExitReasonName(
                                    guardedResult.reason)) +
                            "/" +
                            std::to_string(
                                guardedResult.executedCycles) +
                            ")");
                    std::string difference;
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            checkedState,
                            guardedState,
                            &difference),
                        prefix + "guarded state differs at " +
                            difference);
                    t.IsTrue(
                        checkedEffects.path1Packets() ==
                            guardedEffects.path1Packets(),
                        prefix + "PATH1 effects differ");
                    if (budget == 1u)
                    {
                        t.IsTrue(
                            guardedResult.reason ==
                                VuExitReason::CycleBudget &&
                                guardedResult.executedCycles ==
                                    1u,
                            "the exact retiring budget should stop before unsupported");
                    }
                    if (budget == 2u)
                    {
                        t.IsTrue(
                            guardedResult.reason ==
                                VuExitReason::
                                    UnsupportedInstruction &&
                                guardedResult.executedCycles ==
                                    1u,
                            "spare budget should expose the unsupported boundary");
                    }
                }

                const VuRecompilerDiagnostics diagnostics =
                    guarded.diagnostics();
                t.Equals(
                    diagnostics.fullBlockGuards,
                    uint64_t{1u},
                    "the unsupported exit with spare budget should use the full guard");
                t.Equals(
                    diagnostics.preciseTailEntries,
                    uint64_t{1u},
                    "the exact retiring budget should use the precise tail");
                t.Equals(
                    diagnostics.fullGuardPairs,
                    uint64_t{1u},
                    "the full unsupported path should retire its prefix");
                t.Equals(
                    diagnostics.preciseTailPairs,
                    uint64_t{1u},
                    "the precise unsupported path should retire its prefix");
                t.Equals(
                    diagnostics.budgetGuardComparisons,
                    uint64_t{4u},
                    "selector and precise checks should be counted exactly");
                t.Equals(
                    diagnostics.nativeBudgetExits,
                    uint64_t{1u},
                    "the exact prefix budget should exit before unsupported");
            });

        tc.Run(
            "armed instrumentation uses the permanent interpreter",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuIaddiu(1u, 0u, 7),
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 8u, 0u, kVuUpperNop);
                fixture.memory.markVU1CodeModified();

                VuUnit unit(VuUnitId::Vu1);
                uint32_t observerCalls = 0u;
                unit.setInstructionObserver(
                    [&](uint64_t, uint32_t, uint32_t,
                        uint32_t, const VuExecutionState &)
                    {
                        ++observerCalls;
                    });
                unit.setInstructionObserverEnabled(true);
                unit.setProgressTrackingEnabled(true);

                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(
                        state, fixture, effects, true);
                const VuRunResult result =
                    backend.run(context, 2u);

                t.IsTrue(
                    result.reason == VuExitReason::ProgramEnded,
                    "instrumented fallback should complete");
                t.Equals(
                    observerCalls, uint32_t{2u},
                    "the interpreter observer should see both pairs");
                const VuProgressSnapshot progress =
                    unit.getProgressSnapshot();
                t.Equals(
                    progress.cycles, uint64_t{2u},
                    "interpreter progress should remain exact");
                const VuRecompilerDiagnostics diagnostics =
                    backend.diagnostics();
                t.Equals(
                    diagnostics.interpreterInstrumentationFallbacks,
                    uint64_t{1u},
                    "instrumented execution should count its fallback");
                t.Equals(
                    diagnostics.nativeEntries, uint64_t{0u},
                    "instrumented execution should not enter native code");
                t.IsNull(
                    unit.programCacheIfCreated(),
                    "instrumented fallback should not create a native cache");
            });

        tc.Run(
            "opt-in instrumentation uses distinct native blocks",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuIaddiu(1u, 0u, 7),
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 8u, 0u, kVuUpperNop);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                std::vector<uint32_t> observedPcs;
                std::vector<int32_t> observedVi1;
                unit.setInstructionObserver(
                    [&](uint64_t, uint32_t pc, uint32_t,
                        uint32_t, const VuExecutionState &state)
                    {
                        observedPcs.push_back(pc);
                        observedVi1.push_back(state.vi[1]);
                    });
                unit.setInstructionObserverEnabled(true);

                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(
                        state, fixture, effects, true);
                context.enableNativeInstrumentation = true;
                VuProgramKey key;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Instrumented,
                        key, &diagnostic),
                    "the instrumented key should be valid: " +
                        diagnostic);

                const VuRunResult result =
                    backend.run(context, 2u);
                t.IsTrue(
                    result.reason == VuExitReason::ProgramEnded,
                    "instrumented native execution should complete");
                t.IsTrue(
                    observedPcs ==
                        std::vector<uint32_t>{0u, 8u},
                    "native observation should retain pair order");
                t.IsTrue(
                    observedVi1 ==
                        std::vector<int32_t>{0, 7},
                    "native observation should precede pair mutation");

                const VuRecompilerDiagnostics diagnostics =
                    backend.diagnostics();
                t.Equals(
                    diagnostics.interpreterInstrumentationFallbacks,
                    uint64_t{0u},
                    "opt-in instrumentation should not fall back");
                t.IsTrue(
                    diagnostics.nativeEntries > 0u,
                    "opt-in instrumentation should enter native code");
                t.Equals(
                    diagnostics.nativePairs, uint64_t{2u},
                    "instrumented native accounting should be exact");
                t.IsTrue(
                    unit.programCache().lookup(key).valid(),
                    "the instrumented cache namespace should contain the block");
            });

        tc.Run(
            "retained XGKICK advances across a native side exit",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                const auto configure =
                    [](VuNativeFixture &fixture)
                    {
                        const uint64_t tag =
                            makeGifTag(1u, 2u, 0u);
                        std::memcpy(
                            fixture.data, &tag, sizeof(tag));
                        std::memset(
                            fixture.data + 16u, 0x3c, 16u);
                        writePair(
                            fixture.code, 0u,
                            makeVuLowerSpecial(
                                0x6cu, 1u),
                            kVuUpperNop);
                        writePair(
                            fixture.code, 8u,
                            makeVuSq(
                                0xfu, 3u, 2u, 0),
                            kVuUpperNop);
                        writePair(
                            fixture.code, 16u, 0u,
                            kVuUpperNop | kVuUpperEnd);
                        writePair(
                            fixture.code, 24u, 0u,
                            kVuUpperNop);
                        fixture.memory.markVU1CodeModified();
                    };
                configure(referenceFixture);
                configure(nativeFixture);

                VuExecutionState referenceState =
                    initialSyntheticState();
                VuExecutionState nativeState = referenceState;
                referenceState.vi[1] = 0;
                nativeState.vi[1] = 0;
                referenceState.vi[2] = 1;
                nativeState.vi[2] = 1;
                for (uint32_t lane = 0u; lane < 4u; ++lane)
                {
                    referenceState.vf[3][lane] =
                        100.0f + static_cast<float>(lane);
                    nativeState.vf[3][lane] =
                        referenceState.vf[3][lane];
                }

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                VuTransactionalSideEffectSink referenceEffects;
                VuTransactionalSideEffectSink nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState, referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);

                const VuRunResult referenceKick =
                    reference.run(referenceContext, 1u);
                const VuRunResult nativeKick =
                    native.run(nativeContext, 8u);
                t.IsTrue(
                    nativeKick.reason ==
                        VuExitReason::XgkickBoundary,
                    "XGKICK should leave native code explicitly");
                t.Equals(
                    nativeKick.executedCycles, uint32_t{1u},
                    "the kick pair should retire before side exit");
                t.Equals(
                    referenceKick.executedCycles,
                    nativeKick.executedCycles,
                    "both engines should retire the kick pair");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "kick-boundary state differs at " +
                        difference);

                const VuRecompilerDiagnostics beforeTail =
                    native.diagnostics();
                const VuRunResult referenceTail =
                    reference.run(referenceContext, 7u);
                const VuRunResult nativeTail =
                    native.run(nativeContext, 7u);
                const VuRecompilerDiagnostics afterTail =
                    native.diagnostics();
                t.IsTrue(
                    sameRunResult(
                        referenceTail, nativeTail),
                    "post-kick tail result should match");
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "post-kick state differs at " +
                        difference);
                t.IsTrue(
                    std::memcmp(
                        referenceFixture.data,
                        nativeFixture.data,
                        PS2_VU1_DATA_SIZE) == 0,
                    "post-kick VU data should match");
                t.IsTrue(
                    referenceEffects.path1Packets() ==
                        nativeEffects.path1Packets(),
                    "post-kick PATH1 packets should match");
                t.Equals(
                    afterTail.inlinePairs -
                        beforeTail.inlinePairs,
                    uint64_t{2u},
                    "active XGKICK should retain native surrounding pairs");
                t.Equals(
                    afterTail.helperPairs -
                        beforeTail.helperPairs,
                    uint64_t{1u},
                    "only E-bit completion should need the pair helper");
            });

        tc.Run(
            "native XGKICK credit-only pairs avoid the transfer helper",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                std::memset(
                    referenceFixture.data, 0x5au, 16u);
                std::memset(
                    nativeFixture.data, 0x5au, 16u);

                VuExecutionState referenceState =
                    initialSyntheticState();
                VuExecutionState nativeState =
                    referenceState;
                referenceState.pipeline.xgkick = {
                    .active = true,
                    .address = 0u,
                    .tagBytesRemaining = 16u,
                    .cycleCredit = 0u,
                    .tagEop = true,
                    .packet = {},
                };
                nativeState.pipeline.xgkick =
                    referenceState.pipeline.xgkick;

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState, referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);

                const VuRecompilerDiagnostics before =
                    native.diagnostics();
                const VuRunResult referenceCredit =
                    reference.run(referenceContext, 1u);
                const VuRunResult nativeCredit =
                    native.run(nativeContext, 1u);
                const VuRecompilerDiagnostics afterCredit =
                    native.diagnostics();
                std::string difference;
                t.IsTrue(
                    sameRunResult(
                        referenceCredit, nativeCredit),
                    "credit-only run result should match");
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "credit-only state differs at " +
                        difference);
                t.Equals(
                    afterCredit.xgkickAdvanceHelperCalls -
                        before.xgkickAdvanceHelperCalls,
                    uint64_t{0u},
                    "zero-to-one credit should remain native");

                const VuRunResult referenceTransfer =
                    reference.run(referenceContext, 1u);
                const VuRunResult nativeTransfer =
                    native.run(nativeContext, 1u);
                const VuRecompilerDiagnostics afterTransfer =
                    native.diagnostics();
                t.IsTrue(
                    sameRunResult(
                        referenceTransfer,
                        nativeTransfer),
                    "transfer run result should match");
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "transfer state differs at " +
                        difference);
                t.IsTrue(
                    referenceEffects.path1Packets() ==
                        nativeEffects.path1Packets(),
                    "completed PATH1 packet should match");
                t.Equals(
                    afterTransfer.xgkickAdvanceHelperCalls -
                        afterCredit.xgkickAdvanceHelperCalls,
                    uint64_t{1u},
                    "one-to-two credit should call the transfer helper");
            });

        tc.Run(
            "native XGKICK inlines prepared non-wrapping qwords",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable inlineXgkick(
                    "PS2X_VU_INLINE_XGKICK");
                inlineXgkick.set("1");
                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                for (uint32_t index = 0u;
                     index < 32u; ++index)
                {
                    const uint8_t value =
                        static_cast<uint8_t>(
                            0x40u + index);
                    referenceFixture.data[index] =
                        value;
                    nativeFixture.data[index] =
                        value;
                }

                VuExecutionState referenceState =
                    initialSyntheticState();
                referenceState.pipeline.xgkick = {
                    .active = true,
                    .address = 0u,
                    .tagBytesRemaining = 32u,
                    .cycleCredit = 0u,
                    .tagEop = true,
                    .packet = {},
                };
                referenceState.pipeline.xgkick.packet
                    .prepareAppend(32u);
                VuExecutionState nativeState =
                    referenceState;

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                t.IsTrue(
                    native.inlineXgkickEnabled(),
                    "the focused backend should enable inline XGKICK");
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState,
                        referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);
                const VuRecompilerDiagnostics before =
                    native.diagnostics();

                std::string difference;
                for (uint32_t step = 0u;
                     step < 2u; ++step)
                {
                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext, 1u);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext, 1u);
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        "inline prefix result should match");
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState,
                            &difference),
                        "inline prefix state differs at " +
                            difference);
                }
                const VuRecompilerDiagnostics afterInline =
                    native.diagnostics();
                t.Equals(
                    afterInline.xgkickAdvanceHelperCalls -
                        before.xgkickAdvanceHelperCalls,
                    uint64_t{0u},
                    "the prepared interior qword should stay in generated code");
                t.Equals(
                    nativeState.pipeline.xgkick.packet.size(),
                    size_t{16u},
                    "the inline path should commit exactly one qword");
                t.IsTrue(
                    std::memcmp(
                        nativeState.pipeline.xgkick.packet.data(),
                        nativeFixture.data, 16u) == 0,
                    "the inline path should copy current VU bytes");

                for (uint32_t step = 0u;
                     step < 2u; ++step)
                {
                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext, 1u);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext, 1u);
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        "terminal slow-path result should match");
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState,
                            &difference),
                        "terminal slow-path state differs at " +
                            difference);
                }
                const VuRecompilerDiagnostics afterFinal =
                    native.diagnostics();
                t.Equals(
                    afterFinal.xgkickAdvanceHelperCalls -
                        afterInline.xgkickAdvanceHelperCalls,
                    uint64_t{1u},
                    "the final qword should retain helper publication");
                t.IsTrue(
                    referenceEffects.path1Packets() ==
                        nativeEffects.path1Packets(),
                    "terminal PATH1 publication should remain exact");
            });

        tc.Run(
            "native inline XGKICK preserves budgets stores and wrap",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable inlineXgkick(
                    "PS2X_VU_INLINE_XGKICK");
                inlineXgkick.set("1");
                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize(),
                    "reference memory should initialize");
                t.IsTrue(
                    nativeFixture.initialize(),
                    "native memory should initialize");
                const auto configure =
                    [](VuNativeFixture &fixture)
                    {
                        writePair(
                            fixture.code, 8u,
                            makeVuSq(
                                0xfu, 3u, 1u, 0),
                            kVuUpperNop);
                        writePair(
                            fixture.code, 80u, 0u,
                            kVuUpperNop | kVuUpperEnd);
                        writePair(
                            fixture.code, 88u, 0u,
                            kVuUpperNop);
                        fixture.markCodeModified();
                    };
                configure(referenceFixture);
                configure(nativeFixture);

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(nativeUnit);
                t.IsTrue(
                    native.inlineXgkickEnabled(),
                    "the focused backend should enable inline XGKICK");

                const auto makeState =
                    [](uint32_t address,
                       uint32_t tagBytes)
                    {
                        VuExecutionState state =
                            initialSyntheticState();
                        state.vi[1] = 0u;
                        for (uint32_t lane = 0u;
                             lane < 4u; ++lane)
                        {
                            state.vf[3][lane] =
                                300.0f +
                                static_cast<float>(lane);
                        }
                        state.pipeline.xgkick = {
                            .active = true,
                            .address = address,
                            .tagBytesRemaining =
                                tagBytes,
                            .cycleCredit = 0u,
                            .tagEop = true,
                            .packet = {},
                        };
                        state.pipeline.xgkick.packet
                            .prepareAppend(tagBytes);
                        return state;
                    };
                const auto resetData =
                    [](VuNativeFixture &fixture)
                    {
                        for (uint32_t index = 0u;
                             index < fixture.dataSize;
                             ++index)
                        {
                            fixture.data[index] =
                                static_cast<uint8_t>(
                                    index * 17u + 3u);
                        }
                    };

                for (uint32_t budget = 0u;
                     budget <= 10u; ++budget)
                {
                    resetData(referenceFixture);
                    resetData(nativeFixture);
                    VuExecutionState referenceState =
                        makeState(0u, 64u);
                    VuExecutionState nativeState =
                        referenceState;
                    VuTransactionalSideEffectSink
                        referenceEffects;
                    VuTransactionalSideEffectSink
                        nativeEffects;
                    VuExecutionContext referenceContext =
                        makeContext(
                            referenceState,
                            referenceFixture,
                            referenceEffects);
                    VuExecutionContext nativeContext =
                        makeContext(
                            nativeState,
                            nativeFixture,
                            nativeEffects);

                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext, budget);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext, budget);
                    const std::string prefix =
                        "budget " +
                        std::to_string(budget) + ": ";
                    std::string difference;
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        prefix + "run result differs");
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState,
                            &difference),
                        prefix +
                            "state differs at " +
                            difference);
                    t.IsTrue(
                        std::memcmp(
                            referenceFixture.data,
                            nativeFixture.data,
                            referenceFixture.dataSize) == 0,
                        prefix + "VU data differs");
                    t.IsTrue(
                        referenceEffects.path1Packets() ==
                            nativeEffects.path1Packets(),
                        prefix + "PATH1 effects differ");
                    if (budget >= 2u &&
                        budget < 8u)
                    {
                        t.IsTrue(
                            std::memcmp(
                                nativeState.pipeline.xgkick
                                    .packet.data(),
                                nativeState.vf[3],
                                16u) == 0,
                            prefix +
                                "same-pair SQ must precede the copy");
                    }
                }

                resetData(referenceFixture);
                resetData(nativeFixture);
                VuExecutionState referenceState =
                    makeState(0u, 64u);
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState, referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);
                std::string difference;
                for (uint32_t step = 0u;
                     step < 8u; ++step)
                {
                    if (step == 2u)
                    {
                        std::memset(
                            referenceFixture.data + 16u,
                            0xa5, 16u);
                        std::memset(
                            nativeFixture.data + 16u,
                            0xa5, 16u);
                    }
                    const VuRunResult referenceResult =
                        reference.run(
                            referenceContext, 1u);
                    const VuRunResult nativeResult =
                        native.run(
                            nativeContext, 1u);
                    const std::string prefix =
                        "split step " +
                        std::to_string(step) + ": ";
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        prefix + "run result differs");
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState,
                            &difference),
                        prefix +
                            "state differs at " +
                            difference);
                    t.IsTrue(
                        referenceEffects.path1Packets() ==
                            nativeEffects.path1Packets(),
                        prefix + "PATH1 effects differ");
                    if (step == 3u)
                    {
                        t.IsTrue(
                            std::all_of(
                                nativeState.pipeline.xgkick
                                    .packet.begin() + 16u,
                                nativeState.pipeline.xgkick
                                    .packet.begin() + 32u,
                                [](uint8_t value)
                                {
                                    return value == 0xa5u;
                                }),
                            "future source mutation must remain visible");
                    }
                }

                resetData(referenceFixture);
                resetData(nativeFixture);
                referenceState = makeState(
                    referenceFixture.dataSize - 16u,
                    32u);
                nativeState = referenceState;
                VuTransactionalSideEffectSink
                    wrappedReferenceEffects;
                VuTransactionalSideEffectSink
                    wrappedNativeEffects;
                VuExecutionContext wrappedReferenceContext =
                    makeContext(
                    referenceState, referenceFixture,
                    wrappedReferenceEffects);
                VuExecutionContext wrappedNativeContext =
                    makeContext(
                    nativeState, nativeFixture,
                    wrappedNativeEffects);
                const VuRecompilerDiagnostics beforeWrap =
                    native.diagnostics();
                for (uint32_t step = 0u;
                     step < 4u; ++step)
                {
                    const VuRunResult referenceResult =
                        reference.run(
                            wrappedReferenceContext, 1u);
                    const VuRunResult nativeResult =
                        native.run(
                            wrappedNativeContext, 1u);
                    t.IsTrue(
                        sameRunResult(
                            referenceResult,
                            nativeResult),
                        "wrapped run result should match");
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState,
                            nativeState,
                            &difference),
                        "wrapped state differs at " +
                            difference);
                }
                const VuRecompilerDiagnostics afterWrap =
                    native.diagnostics();
                t.Equals(
                    afterWrap.xgkickAdvanceHelperCalls -
                        beforeWrap.xgkickAdvanceHelperCalls,
                    uint64_t{2u},
                    "wrap and final publication should use the helper");
                t.IsTrue(
                    wrappedReferenceEffects.path1Packets() ==
                        wrappedNativeEffects.path1Packets(),
                    "wrapped PATH1 packet should remain exact");
            });

        tc.Run(
            "helper-side code generation changes exit as invalidated",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                const uint64_t tag =
                    makeGifTag(1u, 2u, 0u);
                std::memcpy(
                    fixture.data, &tag, sizeof(tag));
                std::memset(
                    fixture.data + 16u, 0x6d, 16u);
                writePair(
                    fixture.code, 0u,
                    makeVuLowerSpecial(0x6cu, 1u),
                    kVuUpperNop);
                writePair(
                    fixture.code, 8u, 0u,
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 16u, 0u, kVuUpperNop);
                fixture.memory.markVU1CodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                state.vi[1] = 0;
                InvalidatingSideEffectSink effects(
                    fixture.memory);
                VuExecutionContext context =
                    makeContext(state, fixture, effects);

                const VuRunResult kick =
                    backend.run(context, 3u);
                t.IsTrue(
                    kick.reason == VuExitReason::XgkickBoundary,
                    "the initiating block should first side-exit");
                const VuRunResult completion =
                    backend.run(context, 2u);
                t.IsTrue(
                    completion.reason ==
                        VuExitReason::CodeInvalidated,
                    "a helper-visible generation change should win");
                t.Equals(
                    effects.submissions, uint32_t{1u},
                    "the completing helper should publish once");
                t.IsFalse(
                    state.active,
                    "E-bit completion should still write back state");
                t.Equals(
                    backend.diagnostics().codeInvalidationExits,
                    uint64_t{1u},
                    "helper-side invalidation should be diagnosed");
            });

        tc.Run(
            "native pipeline specialization rejects an impossible VI backup countdown",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u, 0u,
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 8u, 0u, kVuUpperNop);
                fixture.memory.markVU1CodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                const VuRunResult compileRun =
                    backend.run(context, 1u);
                t.Equals(
                    compileRun.executedCycles, uint32_t{1u},
                    "the valid cold run should compile one pair");

                state = initialSyntheticState();
                state.viBackupCycles = 3u;
                const VuRecompilerDiagnostics before =
                    backend.diagnostics();
                const VuRunResult invalid =
                    backend.run(context, 1u);
                t.Equals(
                    invalid.executedCycles, uint32_t{0u},
                    "an invalid entry countdown must not retire work");
                t.IsTrue(
                    invalid.reason == VuExitReason::Fault,
                    "an invalid entry countdown should fault");
                t.Equals(
                    state.viBackupCycles, uint8_t{3u},
                    "the rejected state must remain untouched");
                t.Equals(
                    backend.diagnostics().faultExits -
                        before.faultExits,
                    uint64_t{1u},
                    "the rejected entry should be diagnosed");
            });

        tc.Run(
            "perf symbols encode complete logical block identity",
            [](TestCase &t)
            {
                VuProgramKey key{
                    .unit = VuUnitId::Vu1,
                    .codeSize = PS2_VU1_CODE_SIZE,
                    .entryPc = 0x70u,
                    .codeGeneration =
                        0x1122334455667788ull,
                    .codeContentIdentity =
                        0x8877665544332211ull,
                    .compilationMode =
                        VuCompilationMode::Normal,
                };
                VuIrBlock block{
                    .entryPc = 0x70u,
                    .codeSize = PS2_VU1_CODE_SIZE,
                };
                block.pairs.resize(3u);
                block.pairs[0u].pc = 0x70u;
                block.pairs[1u].pc = 0x78u;
                block.pairs[2u].pc = 0x80u;
                const std::string normal =
                    vuPerfJitSymbolName(
                        key, block,
                        0x1234u);
                t.IsTrue(
                    normal.find("ps2x-vu1-") !=
                        std::string::npos,
                    "symbol should identify VU1");
                t.IsTrue(
                    normal.find(
                        "code-8877665544332211") !=
                        std::string::npos,
                    "symbol should identify the whole code image");
                t.IsTrue(
                    normal.find(
                        "gen-1122334455667788") !=
                        std::string::npos,
                    "symbol should identify the compilation generation");
                t.IsTrue(
                    normal.find(
                        "entry-0070-pcs-0070-0080") !=
                        std::string::npos,
                    "symbol should identify entry and block PC range");
                t.IsTrue(
                    normal.find("-normal-") !=
                        std::string::npos,
                    "normal mode should be explicit");
                t.IsTrue(
                    normal.find("-vfreg-canonical-") !=
                        std::string::npos,
                    "canonical VF state should be explicit");
                t.IsTrue(
                    normal.find("-xgkick-helper-") !=
                        std::string::npos,
                    "helper XGKICK progression should be explicit");
                t.IsTrue(
                    normal.find(
                        "compile-0000000000001234") !=
                        std::string::npos,
                    "symbol should identify the compilation instance");

                key.unit = VuUnitId::Vu0;
                key.compilationMode =
                    VuCompilationMode::Instrumented;
                key.blockLocalVfRegisters = true;
                key.inlineXgkick = true;
                const std::string instrumented =
                    vuPerfJitSymbolName(
                        key, block, 1u);
                t.IsTrue(
                    instrumented.find("ps2x-vu0-") !=
                        std::string::npos &&
                    instrumented.find(
                        "-instrumented-") !=
                        std::string::npos &&
                    instrumented.find(
                        "-vfreg-resident-") !=
                        std::string::npos &&
                    instrumented.find(
                        "-xgkick-inline-") !=
                        std::string::npos,
                    "unit and instrumented mode should not alias normal VU1 code");
            });

        tc.Run(
            "opt-in block counters survive executable cache eviction",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                configureSyntheticProgram(fixture.code);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(
                    unit,
                    VuBlockProfilingConfiguration{
                        .enabled = true,
                        .maximumRecords = 128u,
                    });
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                for (const uint32_t budget :
                     {1u, 12u})
                {
                    state = initialSyntheticState();
                    effects.clear();
                    (void)backend.run(context, budget);
                }

                const VuRecompilerDiagnostics diagnostics =
                    backend.diagnostics();
                const VuBlockProfilingSnapshot snapshot =
                    backend
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                t.IsTrue(
                    snapshot.enabled,
                    "explicit block profiling should be enabled");
                t.Equals(
                    snapshot.droppedRecords,
                    uint64_t{0u},
                    "the focused profile should not drop records");
                t.IsTrue(
                    !snapshot.blocks.empty(),
                    "native execution should create block records");

                uint64_t executions = 0u;
                uint64_t guestPairs = 0u;
                uint64_t helperBarriers = 0u;
                uint64_t fullEntries = 0u;
                uint64_t boundedEntries = 0u;
                uint64_t linkedEdges = 0u;
                uint64_t exits = 0u;
                for (size_t blockIndex = 0u;
                     blockIndex <
                         snapshot.blocks.size();
                     ++blockIndex)
                {
                    const VuBlockProfileSnapshot &block =
                        snapshot.blocks[blockIndex];
                    t.IsTrue(
                        block.compilationIdentity != 0u,
                        "profiled compilation identity should be nonzero");
                    t.IsTrue(
                        block.resident,
                        "newly compiled code should be resident");
                    t.IsTrue(
                        block.fixedCycles != 0u &&
                        block.blockPairs != 0u,
                        "block static cost should be present");
                    uint64_t opcodeOperations = 0u;
                    for (const uint64_t count :
                         block.opcodeOperations)
                    {
                        opcodeOperations += count;
                    }
                    t.Equals(
                        opcodeOperations,
                        static_cast<uint64_t>(
                            block.blockPairs) *
                            2u,
                        "each pair should retain both decoded operations");
                    for (size_t other = 0u;
                         other < blockIndex; ++other)
                    {
                        t.IsTrue(
                            block.compilationIdentity !=
                                snapshot.blocks[other]
                                    .compilationIdentity,
                            "compilation identities should be process-unique");
                    }
                    executions += block.executions;
                    guestPairs += block.guestPairs;
                    helperBarriers +=
                        block.helperBarriers;
                    fullEntries +=
                        block.fullBudgetEntries;
                    boundedEntries +=
                        block.boundedEntries;
                    linkedEdges +=
                        block.linkedEdges;
                    for (const uint64_t count :
                         block.exitReasons)
                    {
                        exits += count;
                    }
                }
                t.Equals(
                    executions, diagnostics.nativeBlocks,
                    "block executions should include linked blocks");
                t.Equals(
                    linkedEdges, diagnostics.linkedEdges,
                    "profile links should sum to global links");
                t.Equals(
                    diagnostics.nativeBlocks,
                    diagnostics.nativeEntries +
                        diagnostics.linkedEdges,
                    "block accounting should reconcile entries and links");
                t.Equals(
                    guestPairs, diagnostics.nativePairs,
                    "block guest pairs should sum to native pairs");
                t.Equals(
                    helperBarriers,
                    diagnostics.helperPairs,
                    "block helper barriers should sum to helper pairs");
                t.Equals(
                    exits, executions,
                    "every block execution should have one exit reason");
                t.Equals(
                    fullEntries + boundedEntries,
                    executions,
                    "every block execution should have one budget class");
                t.IsTrue(
                    fullEntries != 0u &&
                    boundedEntries != 0u,
                    "focused budgets should exercise full and bounded entries");

                backend
                    .resetBlockProfilingCountersWhileExecutionQuiescent();
                const VuBlockProfilingSnapshot reset =
                    backend
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                for (const VuBlockProfileSnapshot &block :
                     reset.blocks)
                {
                    t.Equals(
                        block.executions, uint64_t{0u},
                        "counter reset should clear dynamic execution counts");
                }

                state = initialSyntheticState();
                effects.clear();
                (void)backend.run(context, 12u);
                unit.programCache().flush();
                const VuBlockProfilingSnapshot evicted =
                    backend
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                uint64_t evictedExecutions = 0u;
                for (const VuBlockProfileSnapshot &block :
                     evicted.blocks)
                {
                    t.IsFalse(
                        block.resident,
                        "cache flush should expire every executable lifetime token");
                    evictedExecutions += block.executions;
                }
                t.IsTrue(
                    evictedExecutions != 0u,
                    "cache flush should not discard archived counters");
            });

        tc.Run(
            "native constants ignore VF0 and VI0 backing storage",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "constant fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuIaddiu(1u, 0u, 7),
                    makeVuUpper(
                        0x28u, 0xfu,
                        0u, 0u, 1u));
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                const std::array<float, 4u> poisonedVf0{
                    11.0f, 12.0f, 13.0f, 14.0f};
                std::memcpy(
                    state.vf[0], poisonedVf0.data(),
                    sizeof(state.vf[0]));
                state.vi[0] = 123;
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);

                const VuRunResult result =
                    backend.run(context, 1u);
                t.Equals(
                    result.executedCycles, uint32_t{1u},
                    "the constant pair should retire");
                const std::array<float, 4u> expectedVf1{
                    0.0f, 0.0f, 0.0f, 2.0f};
                t.IsTrue(
                    std::memcmp(
                        state.vf[1],
                        expectedVf1.data(),
                        sizeof(state.vf[1])) == 0,
                    "VF0 should be materialized as [0,0,0,1]");
                t.Equals(
                    state.vi[1], int32_t{7},
                    "VI0 should be materialized as zero");
                const std::array<float, 4u> expectedVf0{
                    0.0f, 0.0f, 0.0f, 1.0f};
                t.IsTrue(
                    std::memcmp(
                        state.vf[0],
                        expectedVf0.data(),
                        sizeof(state.vf[0])) == 0,
                    "the pair should restore architectural VF0");
                t.Equals(
                    state.vi[0], int32_t{0},
                    "the pair should restore architectural VI0");
                t.Equals(
                    backend.diagnostics().helperPairs,
                    uint64_t{0u},
                    "the constant forms should remain inline");
            });

        tc.Run(
            "helper barriers preserve selective resident VF state",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize() &&
                        nativeFixture.initialize(),
                    "helper fixtures should initialize");
                configureHelperRegisterProgram(
                    referenceFixture.code);
                configureHelperRegisterProgram(
                    nativeFixture.code);
                referenceFixture.markCodeModified();
                nativeFixture.markCodeModified();

                ScopedEnvironmentVariable registers(
                    "PS2X_VU_BLOCK_LOCAL_VF_REGISTERS");
                registers.set("1");
                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(
                    nativeUnit,
                    VuBlockProfilingConfiguration{
                        .enabled = true,
                        .maximumRecords = 16u,
                    });
                VuExecutionState referenceState =
                    initialSyntheticState();
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState,
                        referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState, nativeFixture,
                        nativeEffects);

                const VuRunResult expected =
                    reference.run(referenceContext, 5u);
                const VuRunResult actual =
                    native.run(nativeContext, 5u);
                t.IsTrue(
                    sameRunResult(expected, actual),
                    "selective helper result should match");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "selective helper state differs at " +
                        difference);
                t.IsTrue(
                    std::memcmp(
                        referenceFixture.data,
                        nativeFixture.data,
                        PS2_VU1_DATA_SIZE) == 0,
                    "selective helper VU data should match");
                t.IsTrue(
                    referenceEffects.path1Packets() ==
                        nativeEffects.path1Packets(),
                    "selective helper PATH1 should match");

                const VuBlockProfilingSnapshot profile =
                    native
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                uint64_t loads = 0u;
                uint64_t stores = 0u;
                uint64_t spills = 0u;
                uint64_t reloads = 0u;
                uint64_t helperMaterializations = 0u;
                uint64_t residentBlocks = 0u;
                for (const VuBlockProfileSnapshot &block :
                     profile.blocks)
                {
                    if (block.residentVfRegisters != 0u)
                        ++residentBlocks;
                    loads += block.vfRegisterLoads;
                    stores += block.vfRegisterStores;
                    spills += block.vfRegisterSpills;
                    reloads += block.vfRegisterReloads;
                    helperMaterializations +=
                        block.registerMaterializations[
                            static_cast<size_t>(
                                VuRegisterMaterializationCause::
                                    PairHelper)];
                }
                t.Equals(
                    residentBlocks, uint64_t{1u},
                    "the focused program should allocate one resident block");
                t.Equals(
                    loads, uint64_t{5u},
                    "entry and declared clobber loads should be exact");
                t.Equals(
                    stores, uint64_t{4u},
                    "only one helper-read VF plus three exit VFs should materialize");
                t.Equals(
                    spills, uint64_t{2u},
                    "only dirty non-read host VFs should be preserved");
                t.Equals(
                    reloads, uint64_t{2u},
                    "every preserved host VF should be restored");
                t.Equals(
                    helperMaterializations, uint64_t{1u},
                    "the LOI pair should cross one helper barrier");
            });

        tc.Run(
            "block-local VF profiles reconcile canonical traffic",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture referenceFixture;
                VuNativeFixture nativeFixture;
                t.IsTrue(
                    referenceFixture.initialize() &&
                        nativeFixture.initialize(),
                    "profile fixtures should initialize");
                configureSyntheticProgram(
                    referenceFixture.code);
                configureSyntheticProgram(
                    nativeFixture.code);
                referenceFixture.markCodeModified();
                nativeFixture.markCodeModified();

                ScopedEnvironmentVariable registers(
                    "PS2X_VU_BLOCK_LOCAL_VF_REGISTERS");
                registers.set("1");
                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(
                    referenceUnit);
                VuRecompilerBackend native(
                    nativeUnit,
                    VuBlockProfilingConfiguration{
                        .enabled = true,
                        .maximumRecords = 128u,
                    });
                VuExecutionState referenceState =
                    initialSyntheticState();
                VuExecutionState nativeState =
                    referenceState;
                VuTransactionalSideEffectSink
                    referenceEffects;
                VuTransactionalSideEffectSink
                    nativeEffects;
                VuExecutionContext referenceContext =
                    makeContext(
                        referenceState,
                        referenceFixture,
                        referenceEffects);
                VuExecutionContext nativeContext =
                    makeContext(
                        nativeState,
                        nativeFixture,
                        nativeEffects);

                const VuRunResult expected =
                    reference.run(referenceContext, 12u);
                const VuRunResult actual =
                    native.run(nativeContext, 12u);
                t.IsTrue(
                    sameRunResult(expected, actual),
                    "register-resident result should match");
                std::string difference;
                t.IsTrue(
                    vuExecutionStatesEqual(
                        referenceState, nativeState,
                        &difference),
                    "register-resident state differs at " +
                        difference);
                t.IsTrue(
                    std::memcmp(
                        referenceFixture.data,
                        nativeFixture.data,
                        PS2_VU1_DATA_SIZE) == 0,
                    "register-resident VU data should match");
                t.IsTrue(
                    referenceEffects.path1Packets() ==
                        nativeEffects.path1Packets(),
                    "register-resident PATH1 should match");
                t.IsTrue(
                    native.blockLocalVfRegistersEnabled(),
                    "the opt-in backend should expose its mode");

                const VuBlockProfilingSnapshot profile =
                    native
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                uint64_t loads = 0u;
                uint64_t stores = 0u;
                uint64_t spills = 0u;
                uint64_t reloads = 0u;
                uint64_t materializations = 0u;
                uint64_t residentBlocks = 0u;
                for (const VuBlockProfileSnapshot &block :
                     profile.blocks)
                {
                    if (block.residentVfRegisters != 0u)
                    {
                        ++residentBlocks;
                        t.IsTrue(
                            block.residentVfRegisters <= 3u,
                            "allocation should honor the current host limit");
                        t.IsTrue(
                            block.allocatedVfAccesses <=
                                block.vfAccesses,
                            "allocated accesses should be a subset");
                        t.IsTrue(
                            block.residentVfDirtyRegisters <=
                                block.residentVfRegisters,
                            "dirty residents should be assigned");
                        t.IsTrue(
                            block.residentVfDirtyLanes <=
                                block.residentVfDirtyRegisters *
                                    4u,
                            "dirty lane accounting should be bounded");
                    }
                    loads += block.vfRegisterLoads;
                    stores += block.vfRegisterStores;
                    spills += block.vfRegisterSpills;
                    reloads += block.vfRegisterReloads;
                    for (const uint64_t count :
                         block.registerMaterializations)
                    {
                        materializations += count;
                    }
                }
                t.IsTrue(
                    residentBlocks != 0u,
                    "the synthetic block should allocate VFs");
                t.IsTrue(
                    loads != 0u && stores != 0u,
                    "canonical traffic should be counted");
                t.IsTrue(
                    spills != 0u && reloads != 0u,
                    "helper traffic should be counted");
                t.IsTrue(
                    spills == reloads,
                    "every helper-preserved VF should be restored");
                t.IsTrue(
                    materializations != 0u,
                    "materialization causes should be counted");

                native
                    .resetBlockProfilingCountersWhileExecutionQuiescent();
                const VuBlockProfilingSnapshot reset =
                    native
                        .blockProfilingSnapshotWhileExecutionQuiescent();
                for (const VuBlockProfileSnapshot &block :
                     reset.blocks)
                {
                    t.Equals(
                        block.vfRegisterLoads +
                            block.vfRegisterStores +
                            block.vfRegisterSpills +
                            block.vfRegisterReloads,
                        uint64_t{0u},
                        "traffic reset should clear dynamic counts");
                    for (const uint64_t count :
                         block.registerMaterializations)
                    {
                        t.Equals(
                            count, uint64_t{0u},
                            "cause reset should clear dynamic counts");
                    }
                }
            });

        tc.Run(
            "raw entry preserves SysV registers and MXCSR",
            [](TestCase &t)
            {
#if PS2X_TEST_HAS_VU_RECOMPILER_ABI_PROBE && \
    (defined(__x86_64__) || defined(_M_X64))
                if (!VuRecompilerBackend::supported())
                    return;

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u, 0u,
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 8u, 0u, kVuUpperNop);
                fixture.memory.markVU1CodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);
                VuProgramKey key;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context, VuCompilationMode::Normal,
                        key, &diagnostic),
                    "the ABI block key should be valid");
                const VuRunResult compileRun =
                    backend.run(context, 1u);
                t.Equals(
                    compileRun.executedCycles, uint32_t{1u},
                    "the cold run should compile one pair");

                const VuProgramHandle handle =
                    unit.programCache().lookup(key);
                const VuCompiledProgram *const program =
                    unit.programCache().resolve(handle);
                t.IsNotNull(
                    program,
                    "the compiled ABI block should resolve");
                if (!program)
                    return;
                t.IsTrue(
                    program->nativeFastEntry() !=
                        program->nativeEntry(),
                    "the internal fast entry should be distinct from the raw ABI entry");

                VuNativeBlockEntry entry = nullptr;
                const void *const address =
                    program->nativeEntry();
                static_assert(
                    sizeof(entry) == sizeof(address));
                std::memcpy(
                    &entry, &address, sizeof(entry));

                state = initialSyntheticState();
                const uint32_t originalMxcsr = _mm_getcsr();
                const uint32_t probeMxcsr =
                    (originalMxcsr &
                     ~((3u << 13u) |
                       (1u << 15u) |
                       (1u << 6u))) |
                    (2u << 13u);
                _mm_setcsr(probeMxcsr);
                std::array<uint64_t, 6u> preserved{};
                const uint64_t packed =
                    ps2xTestVuNativeProbeCalleeSaved(
                        entry, &context, 1u,
                        preserved.data());
                const uint32_t restoredMxcsr = _mm_getcsr();
                _mm_setcsr(originalMxcsr);

                const std::array<uint64_t, 6u> expected{
                    0x1122334455667788ull,
                    0x2233445566778899ull,
                    0x33445566778899aaull,
                    0x445566778899aabbull,
                    0x5566778899aabbccull,
                    0x66778899aabbccddull,
                };
                t.IsTrue(
                    preserved == expected,
                    "native code corrupted a callee-saved register");
                t.Equals(
                    restoredMxcsr, probeMxcsr,
                    "native code leaked its VU MXCSR mode");
                const VuNativeBlockResult native =
                    VuNativeBlockResult::decode(packed);
                t.Equals(
                    native.executedCycles, uint32_t{1u},
                    "raw ABI entry should report exact work");
                t.IsTrue(
                    native.exit ==
                        VuNativeBlockExit::CycleBudget,
                    "raw ABI entry should report its budget exit");
#else
                (void)t;
#endif
            });

        tc.Run(
            "linked resident VI blocks preserve the shared native frame",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                ScopedEnvironmentVariable registers(
                    "PS2X_VU_BLOCK_LOCAL_VF_REGISTERS");
                registers.set("1");
                ScopedEnvironmentVariable guards(
                    "PS2X_VU_BLOCK_BUDGET_GUARDS");
                guards.set("1");

                VuNativeFixture fixture;
                t.IsTrue(
                    fixture.initialize(),
                    "VU memory fixture should initialize");
                writePair(
                    fixture.code, 0u,
                    makeVuBranch(1), kVuUpperNop);
                writePair(
                    fixture.code, 8u, 0u,
                    kVuUpperNop);
                writePair(
                    fixture.code, 16u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop | kVuUpperEnd);
                writePair(
                    fixture.code, 24u, 0u,
                    kVuUpperNop);
                fixture.markCodeModified();

                VuUnit unit(VuUnitId::Vu1);
                VuRecompilerBackend backend(unit);
                VuExecutionState state =
                    initialSyntheticState();
                state.vi[1] = 7;
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context =
                    makeContext(state, fixture, effects);

                VuProgramKey sourceKey;
                std::string diagnostic;
                t.IsTrue(
                    backend.programKey(
                        context,
                        VuCompilationMode::Normal,
                        sourceKey, &diagnostic),
                    "the linked ABI source key should be valid");
                const VuRunResult cold =
                    backend.run(context, 4u);
                t.IsTrue(
                    cold.reason ==
                        VuExitReason::ProgramEnded,
                    "the cold linked program should complete");
                t.IsTrue(
                    backend.diagnostics().
                            resolvedLinks != 0u,
                    "the cold run should populate the VI target link");

                state = initialSyntheticState();
                state.vi[1] = 7;
                const VuRecompilerDiagnostics
                    beforeWarm = backend.diagnostics();
                const VuRunResult warm =
                    backend.run(context, 4u);
                const VuRecompilerDiagnostics
                    afterWarm = backend.diagnostics();
                t.IsTrue(
                    warm.reason ==
                        VuExitReason::ProgramEnded,
                    "the warm linked VI program should complete");
                t.Equals(
                    warm.executedCycles,
                    uint32_t{4u},
                    "the warm linked VI chain should report four pairs, got " +
                        std::to_string(
                            warm.executedCycles));
                t.Equals(
                    state.vi[1], int32_t{8},
                    "the resident VI target should materialize eight, got " +
                        std::to_string(state.vi[1]));
                t.Equals(
                    afterWarm.linkedEdges -
                        beforeWarm.linkedEdges,
                    uint64_t{1u},
                    "the warm source should enter the resident VI target through its native link");
            });
    });
}
