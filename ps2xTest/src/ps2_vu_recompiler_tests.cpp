#include "MiniTest.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_recompiler.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace
{
    constexpr uint32_t kVuUpperNop = 0x000002ffu;
    constexpr uint32_t kVuUpperEnd = 1u << 30u;

    struct VuNativeFixture
    {
        PS2Memory memory;
        uint8_t *code = nullptr;
        uint8_t *data = nullptr;

        bool initialize()
        {
            if (!memory.initialize())
                return false;
            code = memory.getVU1Code();
            data = memory.getVU1Data();
            if (!code || !data)
                return false;
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            std::memset(data, 0, PS2_VU1_DATA_SIZE);
            for (uint32_t pc = 0u;
                 pc < PS2_VU1_CODE_SIZE; pc += 8u)
            {
                std::memcpy(
                    code + pc + 4u,
                    &kVuUpperNop,
                    sizeof(kVuUpperNop));
            }
            memory.markVU1CodeModified();
            return true;
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
            .codeSize = PS2_VU1_CODE_SIZE,
            .data = fixture.data,
            .dataSize = PS2_VU1_DATA_SIZE,
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
            "synthetic block matches every cycle-budget cut",
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
                configureSyntheticProgram(nativeFixture.code);
                referenceFixture.memory.markVU1CodeModified();
                nativeFixture.memory.markVU1CodeModified();

                VuUnit referenceUnit(VuUnitId::Vu1);
                VuUnit nativeUnit(VuUnitId::Vu1);
                VuInterpreterBackend reference(referenceUnit);
                VuRecompilerBackend native(nativeUnit);

                for (uint32_t budget = 0u;
                     budget <= 12u; ++budget)
                {
                    std::memset(
                        referenceFixture.data, 0x5a,
                        PS2_VU1_DATA_SIZE);
                    std::memset(
                        nativeFixture.data, 0x5a,
                        PS2_VU1_DATA_SIZE);
                    VuExecutionState referenceState =
                        initialSyntheticState();
                    VuExecutionState nativeState =
                        referenceState;
                    VuTransactionalSideEffectSink
                        referenceEffects;
                    VuTransactionalSideEffectSink nativeEffects;
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
                        native.run(nativeContext, budget);
                    const std::string prefix =
                        "budget " + std::to_string(budget) +
                        ": ";

                    t.IsTrue(
                        sameRunResult(
                            referenceResult, nativeResult),
                        prefix + "run result differs (" +
                            std::string(vuExitReasonName(
                                referenceResult.reason)) +
                            "/" +
                            std::string(vuExitReasonName(
                                nativeResult.reason)) +
                            ")");
                    std::string difference;
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            referenceState, nativeState,
                            &difference),
                        prefix + "state differs at " +
                            difference);
                    t.IsTrue(
                        std::memcmp(
                            referenceFixture.data,
                            nativeFixture.data,
                            PS2_VU1_DATA_SIZE) == 0,
                        prefix + "VU data differs");
                    t.IsTrue(
                        referenceEffects.path1Packets() ==
                            nativeEffects.path1Packets(),
                        prefix + "PATH1 effects differ");
                }

                const VuProgramCacheDiagnostics cache =
                    nativeUnit.programCache().diagnostics();
                t.IsTrue(
                    cache.compilations > 0u,
                    "the differential should compile native blocks");
                t.IsTrue(
                    cache.hits > 0u,
                    "later budget cuts should reuse native blocks");
                const VuRecompilerDiagnostics diagnostics =
                    native.diagnostics();
                t.IsTrue(
                    diagnostics.inlinePairs > 0u,
                    "eligible memory/MULq pairs should run inline");
                t.IsTrue(
                    diagnostics.helperPairs > 0u,
                    "delicate pairs should remain on shared helpers");
                t.Equals(
                    diagnostics.inlinePairs +
                        diagnostics.helperPairs,
                    diagnostics.nativePairs,
                    "native pair accounting should be complete");
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
    });
}
