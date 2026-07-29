#include "runtime/ps2_vu_recompiler.h"

#include "runtime/ps2_memory.h"
#include "runtime/ps2_perf_jitdump.h"
#include "ps2_vu_float_mode.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef PS2X_HAS_VU_X64_RECOMPILER
#define PS2X_HAS_VU_X64_RECOMPILER 0
#endif

#if PS2X_HAS_VU_X64_RECOMPILER
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#endif

struct VuBlockProfileRecord
{
    VuProgramKey key{};
    uint64_t compilationIdentity = 0u;
    uint64_t nativeAddress = 0u;
    uint64_t nativeBytes = 0u;
    uint32_t firstPc = 0u;
    uint32_t lastPc = 0u;
    uint32_t blockPairs = 0u;
    uint32_t fixedCycles = 0u;
    VuIrBlockExit blockExit = VuIrBlockExit::PairLimit;
    std::weak_ptr<void> codeLifetime;
    bool archived = false;
    uint64_t executions = 0u;
    uint64_t guestPairs = 0u;
    uint64_t fullBudgetEntries = 0u;
    uint64_t boundedEntries = 0u;
    uint64_t linkedEdges = 0u;
    uint64_t helperBarriers = 0u;
    std::array<
        uint64_t,
        kVuNativeBlockExitCount> exitReasons{};
    std::array<
        uint64_t,
        kVuIrOpcodeCount> opcodeOperations{};
    std::vector<VuBlockJitRegistrationSnapshot>
        jitRegistrations;
};

namespace
{
    using Clock = std::chrono::steady_clock;
    constexpr uint32_t kNativePairExecuted = 1u << 31u;
    constexpr uint32_t kNativePairExitMask = 0xffu;
    constexpr uint64_t kFnv1a64Offset =
        14695981039346656037ull;
    constexpr uint64_t kFnv1a64Prime =
        1099511628211ull;
    constexpr size_t kDefaultMaximumBlockProfiles =
        16384u;
    std::atomic<uint64_t> g_nextCompilationIdentity{1u};

    bool environmentFlagEnabled(const char *value)
    {
        if (!value || value[0] == '\0')
            return false;
        return
            std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 &&
            std::strcmp(value, "FALSE") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "OFF") != 0 &&
            std::strcmp(value, "no") != 0 &&
            std::strcmp(value, "NO") != 0;
    }

    VuBlockProfilingConfiguration
    blockProfilingConfigurationFromEnvironment()
    {
        VuBlockProfilingConfiguration configuration{
            .enabled = environmentFlagEnabled(
                std::getenv(
                    "PS2X_VU_BLOCK_PROFILE")),
            .maximumRecords =
                kDefaultMaximumBlockProfiles,
        };
        if (const char *const text =
                std::getenv(
                    "PS2X_VU_BLOCK_PROFILE_LIMIT");
            text && text[0] != '\0')
        {
            uint64_t value = 0u;
            const char *const end =
                text + std::strlen(text);
            const auto result =
                std::from_chars(text, end, value);
            if (result.ec == std::errc{} &&
                result.ptr == end &&
                value != 0u &&
                value <=
                    std::numeric_limits<size_t>::max())
            {
                configuration.maximumRecords =
                    static_cast<size_t>(value);
            }
        }
        return configuration;
    }

    uint64_t nextCompilationIdentity()
    {
        uint64_t value =
            g_nextCompilationIdentity.fetch_add(
                1u, std::memory_order_relaxed);
        if (value == 0u)
        {
            value =
                g_nextCompilationIdentity.fetch_add(
                    1u, std::memory_order_relaxed);
        }
        return value;
    }

    bool fail(
        std::string message, std::string *diagnostic)
    {
        if (diagnostic)
            *diagnostic = std::move(message);
        return false;
    }

    uint64_t hashCodeImage(
        const uint8_t *code, uint32_t codeSize)
    {
        uint64_t hash = kFnv1a64Offset;
        for (uint32_t index = 0u;
             index < codeSize; ++index)
        {
            hash ^= code[index];
            hash *= kFnv1a64Prime;
        }
        return hash;
    }

    bool isStaticVuBranch(VuIrOpcode opcode)
    {
        switch (opcode)
        {
        case VuIrOpcode::LowerB:
        case VuIrOpcode::LowerBal:
        case VuIrOpcode::LowerIbeq:
        case VuIrOpcode::LowerIbne:
        case VuIrOpcode::LowerIbltz:
        case VuIrOpcode::LowerIbgtz:
        case VuIrOpcode::LowerIblez:
        case VuIrOpcode::LowerIbgez:
            return true;
        default:
            return false;
        }
    }

    uint32_t staticVuBranchTarget(
        const VuIrInstructionPair &pair,
        uint32_t addressMask)
    {
        const uint32_t encoded =
            pair.lowerWord & 0x7ffu;
        const int32_t immediate =
            encoded < 0x400u
                ? static_cast<int32_t>(encoded)
                : static_cast<int32_t>(encoded) -
                      0x800;
        return
            static_cast<uint32_t>(
                static_cast<int32_t>(pair.pc + 8u) +
                immediate * 8) &
            addressMask;
    }

    std::shared_ptr<VuNativeLinkState>
    makeNativeLinkState(
        const VuIrBlock &block,
        const VuProgramKey &key,
        const VuRecompilerBackend *backend)
    {
        auto links =
            std::make_shared<VuNativeLinkState>();
        links->backendIdentity =
            reinterpret_cast<uintptr_t>(backend);
        links->sourceEntryPc = key.entryPc;

        std::vector<uint32_t> targets;
        targets.reserve(block.pairs.size() + 1u);
        const auto append =
            [&](uint32_t target)
            {
                if (std::find(
                        targets.begin(), targets.end(),
                        target) == targets.end())
                {
                    targets.push_back(target);
                }
            };
        if (!block.pairs.empty())
        {
            const uint32_t nextPc =
                (block.pairs.back().pc + 8u) &
                key.addressMask;
            append(nextPc);
        }
        for (const VuIrInstructionPair &pair :
             block.pairs)
        {
            if (isStaticVuBranch(
                    pair.lower.opcode))
            {
                append(
                    staticVuBranchTarget(
                        pair, key.addressMask));
            }
        }

        links->staticTargets.reserve(targets.size());
        for (const uint32_t target : targets)
        {
            links->staticTargets.push_back({
                .targetPc = target,
            });
        }
        return links;
    }

    struct NativeContextView
    {
        VuExecutionState *state = nullptr;
        uint8_t *data = nullptr;
        uint32_t dataSize = 0u;
    };

    using VuNativeFastBlockEntry = uint64_t (*)(
        VuExecutionContext *context,
        uint32_t maximumCycles,
        const NativeContextView *view);

#if PS2X_HAS_VU_X64_RECOMPILER
    constexpr uint32_t kMxcsrRoundingMask = 3u << 13u;
    constexpr uint32_t kMxcsrRoundTowardZero = 3u << 13u;
    constexpr uint32_t kMxcsrDenormalsAreZero = 1u << 6u;
    constexpr uint32_t kMxcsrFlushToZero = 1u << 15u;
    constexpr uint32_t kVuMxcsrBits =
        kMxcsrRoundTowardZero |
        kMxcsrDenormalsAreZero |
        kMxcsrFlushToZero;
    constexpr uint64_t kHostFeatureAvx = 1u << 2u;
    constexpr uint64_t kHostFeatureFma = 1u << 4u;
    constexpr size_t kEmitterCapacity = 128u * 1024u;

    static_assert(
        std::is_standard_layout_v<VuExecutionState>);
    static_assert(
        std::is_standard_layout_v<VuPipelineState>);
    static_assert(
        std::is_standard_layout_v<
            VuPipelineState::DelayedQ>);
    static_assert(
        std::is_standard_layout_v<
            VuPipelineState::DelayedP>);
    static_assert(
        std::is_standard_layout_v<
            VuPipelineState::FmacFlags>);
    static_assert(
        std::is_standard_layout_v<VuNativeLinkSlot>);
    static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
    static_assert(
        std::is_standard_layout_v<VuNativeChainRuntime>);
    static_assert(
        std::is_standard_layout_v<VuBlockProfileRecord>);

    void loadNativeContextView(
        VuExecutionContext *context,
        NativeContextView *view) noexcept
    {
        if (!view)
            return;
        if (!context)
        {
            *view = {};
            return;
        }
        *view = {
            .state = &context->state,
            .data = context->data,
            .dataSize = context->dataSize,
        };
    }

    class X64VuBlockEmitter
    {
    public:
        X64VuBlockEmitter(
            const VuIrBlock &block,
            VuRecompilerBackend *backend,
            const void *pairHelper,
            const void *xgkickHelper,
            uint64_t hostFeatures,
            VuCompilationMode compilationMode,
            VuNativeLinkState *outgoingLinks,
            VuNativeChainRuntime *chainRuntime,
            VuBlockProfileRecord *blockProfile,
            bool blockLinkingEnabled,
            bool blockBudgetGuardsEnabled)
            : m_buffer(kEmitterCapacity),
              m_code(m_buffer.size(), m_buffer.data()),
              m_hostFeatures(hostFeatures),
              m_codeAddressMask(
                  block.codeSize != 0u
                      ? block.codeSize - 1u
                      : 0u),
              m_instrumented(
                  compilationMode ==
                  VuCompilationMode::Instrumented),
              m_backend(backend),
              m_outgoingLinks(outgoingLinks),
              m_chainRuntime(chainRuntime),
              m_blockProfile(blockProfile),
              m_blockLinkingEnabled(
                  blockLinkingEnabled),
              m_blockBudgetGuardsEnabled(
                  blockBudgetGuardsEnabled)
        {
            emit(
                block, pairHelper, xgkickHelper);
            m_code.ready();
        }

        [[nodiscard]] const uint8_t *data() const
        {
            return m_code.getCode<const uint8_t *>();
        }

        [[nodiscard]] size_t size() const
        {
            return m_code.getSize();
        }

        [[nodiscard]] size_t fastEntryOffset() const
        {
            return m_fastEntryOffset;
        }

        [[nodiscard]] size_t linkedEntryOffset() const
        {
            return m_linkedEntryOffset;
        }

    private:
        enum class KnownFmacSlot : uint8_t
        {
            Dynamic,
            Inactive,
            Active,
        };

        enum class KnownViBackup : uint8_t
        {
            Dynamic,
            Inactive,
            Active,
        };

        enum class KnownDelayedScalar : uint8_t
        {
            Dynamic,
            Inactive,
            Active,
        };

        std::vector<uint8_t> m_buffer;
        Xbyak::CodeGenerator m_code;
        uint64_t m_hostFeatures = 0u;
        uint32_t m_codeAddressMask = 0x3fffu;
        bool m_instrumented = false;
        VuRecompilerBackend *m_backend = nullptr;
        VuNativeLinkState *m_outgoingLinks = nullptr;
        VuNativeChainRuntime *m_chainRuntime = nullptr;
        VuBlockProfileRecord *m_blockProfile = nullptr;
        bool m_blockLinkingEnabled = false;
        bool m_blockBudgetGuardsEnabled = false;
        size_t m_fastEntryOffset = 0u;
        size_t m_linkedEntryOffset = 0u;
        bool m_usesAvx = false;

#if defined(_WIN32)
        static constexpr int kStackBytes = 96;
        static constexpr int kSavedMxcsrOffset = 32;
        static constexpr int kVuMxcsrOffset = 36;
        static constexpr int kResultOffset = 48;
        static constexpr int kContextViewOffset = 64;
        static constexpr int kPrecisePairCountOffset = 40;
#else
        static constexpr int kStackBytes = 64;
        static constexpr int kSavedMxcsrOffset = 0;
        static constexpr int kVuMxcsrOffset = 4;
        static constexpr int kResultOffset = 16;
        static constexpr int kContextViewOffset = 32;
        static constexpr int kPrecisePairCountOffset = 8;
#endif
        static constexpr int kRawEntryFlagOffset =
            kStackBytes - 1;
        static_assert(
            kContextViewOffset +
                    static_cast<int>(
                        sizeof(NativeContextView)) <=
                kRawEntryFlagOffset);
        static_assert(
            kPrecisePairCountOffset +
                    static_cast<int>(sizeof(uint64_t)) <=
                kRawEntryFlagOffset);
        static_assert(
            kPrecisePairCountOffset +
                        static_cast<int>(
                            sizeof(uint64_t)) <=
                    kResultOffset ||
                kResultOffset + 16 <=
                    kPrecisePairCountOffset);
        static_assert(
            kPrecisePairCountOffset +
                        static_cast<int>(
                            sizeof(uint64_t)) <=
                    kContextViewOffset ||
                kContextViewOffset +
                        static_cast<int>(
                            sizeof(NativeContextView)) <=
                    kPrecisePairCountOffset);

        static constexpr int stateOffset(size_t value)
        {
            return static_cast<int>(value);
        }

        static constexpr int pipelineOffset(size_t value)
        {
            return stateOffset(
                offsetof(VuExecutionState, pipeline) +
                value);
        }

        bool canInlineUpper(
            const VuIrOperation &operation) const
        {
            return VuRecompilerBackend::
                canInlineUpperOpcode(
                    operation.opcode,
                    m_hostFeatures);
        }

        static bool canInlineLower(
            const VuIrOperation &operation)
        {
            return VuRecompilerBackend::
                canInlineLowerOpcode(
                    operation.opcode);
        }

        bool canInlinePair(
            const VuIrInstructionPair &pair) const
        {
            return
                pair.order != VuIrPairOrder::UpperThenLoi &&
                canInlineUpper(pair.upper) &&
                canInlineLower(pair.lower) &&
                !vuIrHasPairFlag(
                    pair, VuIrPairUnsupported) &&
                !vuIrHasPairFlag(
                    pair, VuIrPairXgkick);
        }

        void emitAdvanceDelayedScalar(
            size_t delayedOffset,
            size_t architecturalOffset,
            KnownDelayedScalar knownState,
            uint32_t knownCyclesRemaining)
        {
            using DelayedQ = VuPipelineState::DelayedQ;
            using namespace Xbyak;

            if (knownState ==
                KnownDelayedScalar::Inactive)
            {
                return;
            }

            Label done;
            Label commit;
            const int base =
                pipelineOffset(delayedOffset);
            if (knownState ==
                KnownDelayedScalar::Dynamic)
            {
                m_code.cmp(
                    m_code.byte[
                        m_code.rbx + base +
                        offsetof(DelayedQ, active)],
                    0u);
                m_code.je(done);
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + base +
                        offsetof(
                            DelayedQ,
                            cyclesRemaining)]);
                m_code.test(m_code.eax, m_code.eax);
                m_code.jz(commit);
                m_code.dec(m_code.eax);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + base +
                        offsetof(
                            DelayedQ,
                            cyclesRemaining)],
                    m_code.eax);
                m_code.jnz(done);
            }
            else if (knownCyclesRemaining > 1u)
            {
                m_code.dec(
                    m_code.dword[
                        m_code.rbx + base +
                        offsetof(
                            DelayedQ,
                            cyclesRemaining)]);
                return;
            }

            m_code.L(commit);
            m_code.movss(
                m_code.xmm0,
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(DelayedQ, result)]);
            m_code.movss(
                m_code.dword[
                    m_code.rbx +
                    stateOffset(architecturalOffset)],
                m_code.xmm0);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + base +
                    offsetof(DelayedQ, active)],
                0u);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(DelayedQ, result)],
                0u);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(
                        DelayedQ, cyclesRemaining)],
                0u);
            m_code.L(done);
        }

        void emitAdvanceFmacFlags(
            KnownFmacSlot knownSlot)
        {
            using Flags = VuPipelineState::FmacFlags;
            using namespace Xbyak;

            constexpr int indexOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState,
                        fmacFlagIndex));
            constexpr int flagsOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState, fmacFlags));
            Label done;

            m_code.movzx(
                m_code.eax,
                m_code.byte[
                    m_code.rbx + indexOffset]);
            m_code.inc(m_code.eax);
            m_code.and_(m_code.eax, 3u);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + indexOffset],
                m_code.al);
            if (knownSlot == KnownFmacSlot::Inactive)
                return;
            m_code.lea(
                m_code.eax,
                m_code.ptr[
                    m_code.rax + m_code.rax * 2u]);
            m_code.shl(m_code.eax, 2u);
            m_code.lea(
                m_code.rdx,
                m_code.ptr[
                    m_code.rbx + flagsOffset]);
            m_code.add(m_code.rdx, m_code.rax);
            if (knownSlot == KnownFmacSlot::Dynamic)
            {
                m_code.cmp(
                    m_code.byte[
                        m_code.rdx +
                        offsetof(Flags, active)],
                    0u);
                m_code.je(done);
            }

            m_code.mov(
                m_code.eax,
                m_code.dword[
                    m_code.rdx +
                    offsetof(Flags, mac)]);
            m_code.mov(
                m_code.dword[
                    m_code.rbx +
                    stateOffset(
                        offsetof(
                            VuExecutionState, mac))],
                m_code.eax);
            m_code.mov(
                m_code.eax,
                m_code.dword[
                    m_code.rbx +
                    stateOffset(
                        offsetof(
                            VuExecutionState, status))]);
            m_code.and_(m_code.eax, 0xff0u);
            m_code.mov(
                m_code.ecx,
                m_code.dword[
                    m_code.rdx +
                    offsetof(Flags, status)]);
            m_code.and_(m_code.ecx, 0x0fu);
            m_code.or_(m_code.eax, m_code.ecx);
            m_code.shl(m_code.ecx, 6u);
            m_code.or_(m_code.eax, m_code.ecx);
            m_code.mov(
                m_code.dword[
                    m_code.rbx +
                    stateOffset(
                        offsetof(
                            VuExecutionState, status))],
                m_code.eax);
            m_code.mov(
                m_code.qword[m_code.rdx], 0u);
            m_code.mov(
                m_code.dword[
                    m_code.rdx +
                    offsetof(Flags, status)],
                0u);
            if (knownSlot == KnownFmacSlot::Dynamic)
                m_code.L(done);
        }

        void emitFlushDelayedScalar(
            size_t delayedOffset,
            size_t architecturalOffset)
        {
            using DelayedQ = VuPipelineState::DelayedQ;
            using namespace Xbyak;

            const int base =
                pipelineOffset(delayedOffset);
            Label done;
            m_code.cmp(
                m_code.byte[
                    m_code.rbx + base +
                    offsetof(DelayedQ, active)],
                0u);
            m_code.je(done);
            m_code.movss(
                m_code.xmm0,
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(DelayedQ, result)]);
            m_code.movss(
                m_code.dword[
                    m_code.rbx +
                    stateOffset(
                        architecturalOffset)],
                m_code.xmm0);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + base +
                    offsetof(DelayedQ, active)],
                0u);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(DelayedQ, result)],
                0u);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(
                        DelayedQ, cyclesRemaining)],
                0u);
            m_code.L(done);
        }

        void emitAdvanceViBackup(
            KnownViBackup knownViBackup)
        {
            if (knownViBackup ==
                KnownViBackup::Inactive)
            {
                return;
            }

            constexpr int cyclesOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupCycles));
            if (knownViBackup ==
                KnownViBackup::Active)
            {
                m_code.dec(
                    m_code.byte[
                        m_code.rbx + cyclesOffset]);
                return;
            }

            Xbyak::Label noBackup;
            m_code.cmp(
                m_code.byte[
                    m_code.rbx + cyclesOffset],
                0u);
            m_code.je(noBackup);
            m_code.dec(
                m_code.byte[
                    m_code.rbx + cyclesOffset]);
            m_code.L(noBackup);
        }

        void emitAdvancePipelines(
            KnownFmacSlot knownFmacSlot,
            KnownViBackup knownViBackup,
            KnownDelayedScalar knownQ,
            uint32_t knownQCyclesRemaining,
            KnownDelayedScalar knownP,
            uint32_t knownPCyclesRemaining)
        {
            emitAdvanceDelayedScalar(
                offsetof(
                    VuPipelineState, delayedQ),
                offsetof(VuExecutionState, q),
                knownQ, knownQCyclesRemaining);
            emitAdvanceDelayedScalar(
                offsetof(
                    VuPipelineState, delayedP),
                offsetof(VuExecutionState, p),
                knownP, knownPCyclesRemaining);
            emitAdvanceFmacFlags(knownFmacSlot);
            emitAdvanceViBackup(knownViBackup);
        }

        void emitBackupVi(uint8_t reg)
        {
            if (reg == 0u)
                return;

            using namespace Xbyak;
            constexpr int cyclesOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupCycles));
            constexpr int registerOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupRegister));
            constexpr int valueOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupValue));
            const int viOffset =
                stateOffset(
                    offsetof(VuExecutionState, vi) +
                    static_cast<size_t>(reg) *
                        sizeof(int32_t));
            Label replace;
            Label done;

            m_code.cmp(
                m_code.byte[
                    m_code.rbx + cyclesOffset],
                0u);
            m_code.je(replace);
            m_code.cmp(
                m_code.byte[
                    m_code.rbx + registerOffset],
                reg);
            m_code.jne(replace);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + cyclesOffset],
                2u);
            m_code.jmp(done);

            m_code.L(replace);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + cyclesOffset],
                2u);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + registerOffset],
                reg);
            m_code.movsx(
                m_code.eax,
                m_code.word[
                    m_code.rbx + viOffset]);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + valueOffset],
                m_code.eax);
            m_code.L(done);
        }

        void emitReadViForBranch(
            uint8_t reg, const Xbyak::Reg32 &result,
            const Xbyak::Reg32 &scratch)
        {
            if (reg == 0u)
            {
                m_code.xor_(result, result);
                return;
            }

            using namespace Xbyak;
            constexpr int cyclesOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupCycles));
            constexpr int registerOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupRegister));
            constexpr int valueOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        viBackupValue));
            const int viOffset =
                stateOffset(
                    offsetof(VuExecutionState, vi) +
                    static_cast<size_t>(reg) *
                        sizeof(int32_t));
            Label liveValue;
            Label done;
            m_code.cmp(
                m_code.byte[
                    m_code.rbx + cyclesOffset],
                0u);
            m_code.je(liveValue);
            m_code.movzx(
                scratch,
                m_code.byte[
                    m_code.rbx + registerOffset]);
            m_code.cmp(scratch, reg);
            m_code.jne(liveValue);
            m_code.movsx(
                result,
                m_code.word[
                    m_code.rbx + valueOffset]);
            m_code.jmp(done);

            m_code.L(liveValue);
            m_code.movsx(
                result,
                m_code.word[
                    m_code.rbx + viOffset]);
            m_code.L(done);
        }

        void emitArmBranch(uint32_t target)
        {
            constexpr int pendingOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        branchPending));
            constexpr int targetOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        branchTarget));
            constexpr int delayOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        branchDelay));
            m_code.mov(
                m_code.byte[
                    m_code.rbx + pendingOffset],
                1u);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + targetOffset],
                target & m_codeAddressMask);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + delayOffset],
                1u);
        }

        void emitArmBranchFromEax()
        {
            constexpr int pendingOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        branchPending));
            constexpr int targetOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        branchTarget));
            constexpr int delayOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState,
                        branchDelay));
            m_code.and_(
                m_code.eax, m_codeAddressMask);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + pendingOffset],
                1u);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + targetOffset],
                m_code.eax);
            m_code.mov(
                m_code.byte[
                    m_code.rbx + delayOffset],
                1u);
        }

        void emitApplyVectorFromMemory(
            uint8_t target, uint8_t destination,
            const Xbyak::Reg64 &address)
        {
            if (target == 0u || destination == 0u)
                return;

            const int targetOffset =
                stateOffset(
                    offsetof(VuExecutionState, vf) +
                    static_cast<size_t>(target) *
                        sizeof(float) * 4u);
            if (destination == 0x0fu)
            {
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[address]);
                m_code.movups(
                    m_code.ptr[
                        m_code.rbx + targetOffset],
                    m_code.xmm0);
                return;
            }

            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                if ((destination & (0x8u >> lane)) == 0u)
                    continue;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        address +
                        static_cast<int>(lane * 4u)]);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + targetOffset +
                        static_cast<int>(lane * 4u)],
                    m_code.eax);
            }
        }

        void emitApplyVectorToMemory(
            uint8_t source, uint8_t destination,
            const Xbyak::Reg64 &address)
        {
            if (destination == 0u)
                return;

            const int sourceOffset =
                stateOffset(
                    offsetof(VuExecutionState, vf) +
                    static_cast<size_t>(source) *
                        sizeof(float) * 4u);
            if (destination == 0x0fu)
            {
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[
                        m_code.rbx + sourceOffset]);
                m_code.movups(
                    m_code.ptr[address],
                    m_code.xmm0);
                return;
            }

            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                if ((destination & (0x8u >> lane)) == 0u)
                    continue;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + sourceOffset +
                        static_cast<int>(lane * 4u)]);
                m_code.mov(
                    m_code.dword[
                        address +
                        static_cast<int>(lane * 4u)],
                    m_code.eax);
            }
        }

        void emitApplyResultToOffset(
            int targetOffset, uint8_t destination)
        {
            if (destination == 0u)
                return;

            if (destination == 0x0fu)
            {
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[
                        m_code.rsp + kResultOffset]);
                m_code.movups(
                    m_code.ptr[
                        m_code.rbx + targetOffset],
                    m_code.xmm0);
                return;
            }

            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                if ((destination & (0x8u >> lane)) == 0u)
                    continue;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rsp + kResultOffset +
                        static_cast<int>(lane * 4u)]);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + targetOffset +
                        static_cast<int>(lane * 4u)],
                    m_code.eax);
            }
        }

        void emitApplyResultToVf(
            uint8_t target, uint8_t destination)
        {
            if (target == 0u)
                return;
            emitApplyResultToOffset(
                stateOffset(
                    offsetof(VuExecutionState, vf) +
                    static_cast<size_t>(target) *
                        sizeof(float) * 4u),
                destination);
        }

        void emitStoreSelectedWords(
            uint8_t sourceVi, uint8_t destination,
            const Xbyak::Reg64 &address)
        {
            const int sourceOffset =
                stateOffset(
                    offsetof(VuExecutionState, vi) +
                    static_cast<size_t>(sourceVi) *
                        sizeof(int32_t));
            m_code.movzx(
                m_code.eax,
                m_code.word[
                    m_code.rbx + sourceOffset]);
            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                if ((destination & (0x8u >> lane)) == 0u)
                    continue;
                m_code.mov(
                    m_code.dword[
                        address +
                        static_cast<int>(lane * 4u)],
                    m_code.eax);
            }
        }

        void emitLoadSelectedWord(
            uint8_t targetVi, uint8_t destination,
            const Xbyak::Reg64 &address)
        {
            if (targetVi == 0u)
                return;

            uint32_t lane = 3u;
            if ((destination & 0x8u) != 0u)
                lane = 0u;
            else if ((destination & 0x4u) != 0u)
                lane = 1u;
            else if ((destination & 0x2u) != 0u)
                lane = 2u;
            const int targetOffset =
                stateOffset(
                    offsetof(VuExecutionState, vi) +
                    static_cast<size_t>(targetVi) *
                        sizeof(int32_t));
            m_code.movsx(
                m_code.eax,
                m_code.word[
                    address +
                    static_cast<int>(lane * 4u)]);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + targetOffset],
                m_code.eax);
        }

        void emitIncrementVi(uint8_t reg)
        {
            if (reg == 0u)
                return;
            const int viOffset =
                stateOffset(
                    offsetof(VuExecutionState, vi) +
                    static_cast<size_t>(reg) *
                        sizeof(int32_t));
            m_code.movsx(
                m_code.eax,
                m_code.word[
                    m_code.rbx + viOffset]);
            m_code.inc(m_code.eax);
            m_code.movsx(
                m_code.eax, m_code.ax);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + viOffset],
                m_code.eax);
        }

        void emitDataAddress(
            uint8_t base, int32_t immediate,
            bool unsignedBase, Xbyak::Label &outOfBounds)
        {
            const int viOffset =
                stateOffset(
                    offsetof(VuExecutionState, vi) +
                    static_cast<size_t>(base) *
                        sizeof(int32_t));
            if (unsignedBase)
            {
                m_code.movzx(
                    m_code.eax,
                    m_code.word[
                        m_code.rbx + viOffset]);
            }
            else
            {
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + viOffset]);
                if (immediate != 0)
                    m_code.add(m_code.eax, immediate);
            }
            m_code.shl(m_code.eax, 4u);
            m_code.mov(
                m_code.ecx,
                m_code.dword[
                    m_code.rsp +
                    kContextViewOffset +
                    offsetof(
                        NativeContextView, dataSize)]);
            m_code.test(m_code.ecx, m_code.ecx);
            m_code.jz(outOfBounds);
            m_code.dec(m_code.ecx);
            m_code.and_(m_code.eax, m_code.ecx);
            m_code.inc(m_code.ecx);
            m_code.lea(
                m_code.r8d,
                m_code.ptr[m_code.rax + 16]);
            m_code.cmp(m_code.r8d, m_code.ecx);
            m_code.ja(outOfBounds);
            m_code.mov(
                m_code.rdx,
                m_code.qword[
                    m_code.rsp +
                    kContextViewOffset +
                    offsetof(
                        NativeContextView, data)]);
            m_code.add(m_code.rdx, m_code.rax);
        }

        void emitLower(
            const VuIrInstructionPair &pair)
        {
            using namespace Xbyak;
            const uint32_t word = pair.lowerWord;
            const uint8_t destination =
                static_cast<uint8_t>(
                    (word >> 21u) & 0x0fu);
            const uint8_t vfT =
                static_cast<uint8_t>(
                    (word >> 16u) & 0x1fu);
            const uint8_t vfS =
                static_cast<uint8_t>(
                    (word >> 11u) & 0x1fu);
            const uint8_t viT =
                static_cast<uint8_t>(
                    (word >> 16u) & 0x0fu);
            const uint8_t viS =
                static_cast<uint8_t>(
                    (word >> 11u) & 0x0fu);
            const uint8_t viD =
                static_cast<uint8_t>(
                    (word >> 6u) & 0x0fu);
            const int32_t immediate =
                static_cast<int32_t>(
                    word << 21u) >>
                21u;
            const int32_t immediate15 =
                static_cast<int16_t>(
                    (word & 0x7ffu) |
                    ((word >> 10u) & 0x7800u));
            const int32_t immediate5 =
                (word & (1u << 10u)) != 0u
                    ? static_cast<int32_t>(
                          (word >> 6u) & 0x1fu) -
                          32
                    : static_cast<int32_t>(
                          (word >> 6u) & 0x1fu);
            const auto viOffset =
                [](uint8_t reg)
                {
                    return stateOffset(
                        offsetof(VuExecutionState, vi) +
                        static_cast<size_t>(reg) *
                            sizeof(int32_t));
                };

            switch (pair.lower.opcode)
            {
            case VuIrOpcode::Nop:
                return;
            case VuIrOpcode::LowerLq:
            {
                Label outOfBounds;
                emitDataAddress(
                    viS, immediate, false,
                    outOfBounds);
                emitApplyVectorFromMemory(
                    vfT, destination, m_code.rdx);
                m_code.L(outOfBounds);
                return;
            }
            case VuIrOpcode::LowerSq:
            {
                Label outOfBounds;
                emitDataAddress(
                    viT, immediate, false,
                    outOfBounds);
                emitApplyVectorToMemory(
                    vfS, destination, m_code.rdx);
                m_code.L(outOfBounds);
                return;
            }
            case VuIrOpcode::LowerIlw:
            {
                Label outOfBounds;
                emitDataAddress(
                    viS, immediate, false,
                    outOfBounds);
                emitLoadSelectedWord(
                    viT, destination, m_code.rdx);
                m_code.L(outOfBounds);
                return;
            }
            case VuIrOpcode::LowerIsw:
            {
                Label outOfBounds;
                emitDataAddress(
                    viS, immediate, false,
                    outOfBounds);
                emitStoreSelectedWords(
                    viT, destination, m_code.rdx);
                m_code.L(outOfBounds);
                return;
            }
            case VuIrOpcode::LowerIaddiu:
            case VuIrOpcode::LowerIsubiu:
            {
                if (viT == 0u)
                    return;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + viOffset(viS)]);
                if (pair.lower.opcode ==
                    VuIrOpcode::LowerIaddiu)
                {
                    m_code.add(
                        m_code.eax, immediate15);
                }
                else
                {
                    m_code.sub(
                        m_code.eax, immediate15);
                }
                m_code.movsx(
                    m_code.eax, m_code.ax);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + viOffset(viT)],
                    m_code.eax);
                return;
            }
            case VuIrOpcode::LowerIadd:
            case VuIrOpcode::LowerIsub:
            {
                if (viD == 0u)
                    return;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + viOffset(viS)]);
                if (pair.lower.opcode ==
                    VuIrOpcode::LowerIadd)
                {
                    m_code.add(
                        m_code.eax,
                        m_code.dword[
                            m_code.rbx +
                            viOffset(viT)]);
                }
                else
                {
                    m_code.sub(
                        m_code.eax,
                        m_code.dword[
                            m_code.rbx +
                            viOffset(viT)]);
                }
                m_code.movsx(
                    m_code.eax, m_code.ax);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + viOffset(viD)],
                    m_code.eax);
                return;
            }
            case VuIrOpcode::LowerIaddi:
            {
                if (viT == 0u)
                    return;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + viOffset(viS)]);
                m_code.add(
                    m_code.eax, immediate5);
                m_code.movsx(
                    m_code.eax, m_code.ax);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + viOffset(viT)],
                    m_code.eax);
                return;
            }
            case VuIrOpcode::LowerIand:
            case VuIrOpcode::LowerIor:
            {
                if (viD == 0u)
                    return;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx + viOffset(viS)]);
                if (pair.lower.opcode ==
                    VuIrOpcode::LowerIand)
                {
                    m_code.and_(
                        m_code.eax,
                        m_code.dword[
                            m_code.rbx +
                            viOffset(viT)]);
                }
                else
                {
                    m_code.or_(
                        m_code.eax,
                        m_code.dword[
                            m_code.rbx +
                            viOffset(viT)]);
                }
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + viOffset(viD)],
                    m_code.eax);
                return;
            }
            case VuIrOpcode::LowerMove:
            {
                const int sourceOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, vf) +
                        static_cast<size_t>(vfS) *
                            sizeof(float) * 4u);
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[
                        m_code.rbx + sourceOffset]);
                m_code.movups(
                    m_code.ptr[
                        m_code.rsp + kResultOffset],
                    m_code.xmm0);
                emitApplyResultToVf(
                    vfT, destination);
                return;
            }
            case VuIrOpcode::LowerMr32:
            {
                const int sourceOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, vf) +
                        static_cast<size_t>(vfS) *
                            sizeof(float) * 4u);
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[
                        m_code.rbx + sourceOffset]);
                m_code.shufps(
                    m_code.xmm0, m_code.xmm0,
                    0x39u);
                m_code.movups(
                    m_code.ptr[
                        m_code.rsp + kResultOffset],
                    m_code.xmm0);
                emitApplyResultToVf(
                    vfT, destination);
                return;
            }
            case VuIrOpcode::LowerLqi:
            {
                Label outOfBounds;
                emitBackupVi(viS);
                emitDataAddress(
                    viS, 0, true, outOfBounds);
                emitApplyVectorFromMemory(
                    vfT, destination, m_code.rdx);
                m_code.L(outOfBounds);
                emitIncrementVi(viS);
                return;
            }
            case VuIrOpcode::LowerSqi:
            {
                Label outOfBounds;
                emitBackupVi(viT);
                emitDataAddress(
                    viT, 0, true, outOfBounds);
                emitApplyVectorToMemory(
                    vfS, destination, m_code.rdx);
                m_code.L(outOfBounds);
                emitIncrementVi(viT);
                return;
            }
            case VuIrOpcode::LowerMtir:
            {
                if (viT == 0u)
                    return;
                emitBackupVi(viT);
                const uint8_t component =
                    static_cast<uint8_t>(
                        (word >> 21u) & 3u);
                const int sourceOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, vf) +
                        static_cast<size_t>(vfS) *
                            sizeof(float) * 4u +
                        static_cast<size_t>(component) *
                            sizeof(float));
                const int targetOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, vi) +
                        static_cast<size_t>(viT) *
                            sizeof(int32_t));
                m_code.movsx(
                    m_code.eax,
                    m_code.word[
                        m_code.rbx + sourceOffset]);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + targetOffset],
                    m_code.eax);
                return;
            }
            case VuIrOpcode::LowerMfir:
            {
                const int sourceOffset =
                    viOffset(viS);
                m_code.movsx(
                    m_code.eax,
                    m_code.word[
                        m_code.rbx + sourceOffset]);
                for (uint32_t lane = 0u;
                     lane < 4u; ++lane)
                {
                    m_code.mov(
                        m_code.dword[
                            m_code.rsp +
                            kResultOffset +
                            static_cast<int>(
                                lane * 4u)],
                        m_code.eax);
                }
                emitApplyResultToVf(
                    vfT, destination);
                return;
            }
            case VuIrOpcode::LowerIlwr:
            {
                Label outOfBounds;
                emitDataAddress(
                    viS, 0, true, outOfBounds);
                emitLoadSelectedWord(
                    viT, destination, m_code.rdx);
                m_code.L(outOfBounds);
                return;
            }
            case VuIrOpcode::LowerIswr:
            {
                Label outOfBounds;
                emitDataAddress(
                    viS, 0, true, outOfBounds);
                emitStoreSelectedWords(
                    viT, destination, m_code.rdx);
                m_code.L(outOfBounds);
                return;
            }
            case VuIrOpcode::LowerXtop:
            case VuIrOpcode::LowerXitop:
            {
                if (viT == 0u)
                    return;
                const size_t source =
                    pair.lower.opcode ==
                            VuIrOpcode::LowerXtop
                        ? offsetof(
                              VuExecutionState, top)
                        : offsetof(
                              VuExecutionState, itop);
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx +
                        stateOffset(source)]);
                m_code.and_(m_code.eax, 0x3ffu);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + viOffset(viT)],
                    m_code.eax);
                return;
            }
            case VuIrOpcode::LowerDiv:
            {
                const uint8_t numeratorComponent =
                    static_cast<uint8_t>(
                        (word >> 21u) & 3u);
                const uint8_t denominatorComponent =
                    static_cast<uint8_t>(
                        (word >> 23u) & 3u);
                const int numeratorOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, vf) +
                        static_cast<size_t>(vfS) *
                            sizeof(float) * 4u +
                        static_cast<size_t>(
                            numeratorComponent) *
                            sizeof(float));
                const int denominatorOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, vf) +
                        static_cast<size_t>(vfT) *
                            sizeof(float) * 4u +
                        static_cast<size_t>(
                            denominatorComponent) *
                            sizeof(float));
                Label divide;
                Label negativeSaturation;
                Label schedule;
                m_code.movss(
                    m_code.xmm0,
                    m_code.dword[
                        m_code.rbx + numeratorOffset]);
                m_code.movss(
                    m_code.xmm1,
                    m_code.dword[
                        m_code.rbx +
                        denominatorOffset]);
                m_code.xorps(
                    m_code.xmm2, m_code.xmm2);
                m_code.ucomiss(
                    m_code.xmm1, m_code.xmm2);
                m_code.jp(divide);
                m_code.jne(divide);

                m_code.ucomiss(
                    m_code.xmm0, m_code.xmm2);
                m_code.jp(negativeSaturation);
                m_code.jb(negativeSaturation);
                m_code.mov(
                    m_code.eax, 0x7f7fffffu);
                m_code.movd(
                    m_code.xmm0, m_code.eax);
                m_code.jmp(schedule);

                m_code.L(negativeSaturation);
                m_code.mov(
                    m_code.eax, 0xff7fffffu);
                m_code.movd(
                    m_code.xmm0, m_code.eax);
                m_code.jmp(schedule);

                m_code.L(divide);
                m_code.divss(
                    m_code.xmm0, m_code.xmm1);

                m_code.L(schedule);
                constexpr int delayedOffset =
                    pipelineOffset(
                        offsetof(
                            VuPipelineState,
                            delayedQ));
                m_code.movss(
                    m_code.dword[
                        m_code.rbx + delayedOffset +
                        offsetof(
                            VuPipelineState::DelayedQ,
                            result)],
                    m_code.xmm0);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + delayedOffset +
                        offsetof(
                            VuPipelineState::DelayedQ,
                            cyclesRemaining)],
                    7u);
                m_code.mov(
                    m_code.byte[
                        m_code.rbx + delayedOffset +
                        offsetof(
                            VuPipelineState::DelayedQ,
                            active)],
                    1u);
                return;
            }
            case VuIrOpcode::LowerB:
            case VuIrOpcode::LowerBal:
            {
                const uint32_t target =
                    static_cast<uint32_t>(
                        static_cast<int32_t>(
                            pair.pc + 8u) +
                        immediate * 8) &
                    m_codeAddressMask;
                if (pair.lower.opcode ==
                        VuIrOpcode::LowerBal &&
                    viT != 0u)
                {
                    m_code.mov(
                        m_code.dword[
                            m_code.rbx +
                            viOffset(viT)],
                        (pair.pc + 16u) / 8u);
                }
                emitArmBranch(target);
                return;
            }
            case VuIrOpcode::LowerJr:
            case VuIrOpcode::LowerJalr:
            {
                m_code.movzx(
                    m_code.eax,
                    m_code.word[
                        m_code.rbx +
                        viOffset(viS)]);
                m_code.shl(m_code.eax, 3u);
                if (pair.lower.opcode ==
                        VuIrOpcode::LowerJalr &&
                    viT != 0u)
                {
                    m_code.mov(
                        m_code.dword[
                            m_code.rbx +
                            viOffset(viT)],
                        (pair.pc + 16u) / 8u);
                }
                emitArmBranchFromEax();
                return;
            }
            case VuIrOpcode::LowerIbeq:
            case VuIrOpcode::LowerIbne:
            {
                Label notTaken;
                emitReadViForBranch(
                    viS, m_code.eax, m_code.edx);
                emitReadViForBranch(
                    viT, m_code.ecx, m_code.edx);
                m_code.cmp(
                    m_code.eax, m_code.ecx);
                if (pair.lower.opcode ==
                    VuIrOpcode::LowerIbeq)
                {
                    m_code.jne(notTaken);
                }
                else
                {
                    m_code.je(notTaken);
                }
                const uint32_t target =
                    static_cast<uint32_t>(
                        static_cast<int32_t>(
                            pair.pc + 8u) +
                        immediate * 8) &
                    m_codeAddressMask;
                emitArmBranch(target);
                m_code.L(notTaken);
                return;
            }
            case VuIrOpcode::LowerIbltz:
            case VuIrOpcode::LowerIbgtz:
            case VuIrOpcode::LowerIblez:
            case VuIrOpcode::LowerIbgez:
            {
                Label notTaken;
                emitReadViForBranch(
                    viS, m_code.eax, m_code.ecx);
                m_code.test(
                    m_code.eax, m_code.eax);
                switch (pair.lower.opcode)
                {
                case VuIrOpcode::LowerIbltz:
                    m_code.jns(notTaken);
                    break;
                case VuIrOpcode::LowerIbgtz:
                    m_code.jle(notTaken);
                    break;
                case VuIrOpcode::LowerIblez:
                    m_code.jg(notTaken);
                    break;
                case VuIrOpcode::LowerIbgez:
                    m_code.js(notTaken);
                    break;
                default:
                    break;
                }
                const uint32_t target =
                    static_cast<uint32_t>(
                        static_cast<int32_t>(
                            pair.pc + 8u) +
                        immediate * 8) &
                    m_codeAddressMask;
                emitArmBranch(target);
                m_code.L(notTaken);
                return;
            }
            default:
                return;
            }
        }

        void emitScheduleFmacFlags(uint8_t destination)
        {
            using Flags = VuPipelineState::FmacFlags;
            using namespace Xbyak;

            m_code.xor_(m_code.r8d, m_code.r8d);
            for (uint32_t lane = 0u; lane < 4u; ++lane)
            {
                if ((destination & (0x8u >> lane)) == 0u)
                    continue;

                const uint32_t shift = 3u - lane;
                Label noSign;
                Label zero;
                Label denormal;
                Label overflow;
                Label laneDone;
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rsp + kResultOffset +
                        static_cast<int>(lane * 4u)]);
                m_code.test(
                    m_code.eax, 0x80000000u);
                m_code.jz(noSign);
                m_code.or_(
                    m_code.r8d,
                    0x0010u << shift);
                m_code.L(noSign);

                m_code.mov(m_code.ecx, m_code.eax);
                m_code.and_(
                    m_code.ecx, 0x7fffffffu);
                m_code.jz(zero);
                m_code.mov(m_code.ecx, m_code.eax);
                m_code.shr(m_code.ecx, 23u);
                m_code.and_(m_code.ecx, 0xffu);
                m_code.jz(denormal);
                m_code.cmp(m_code.ecx, 0xffu);
                m_code.je(overflow);
                m_code.jmp(laneDone);

                m_code.L(zero);
                m_code.or_(
                    m_code.r8d,
                    0x0001u << shift);
                m_code.jmp(laneDone);
                m_code.L(denormal);
                m_code.or_(
                    m_code.r8d,
                    0x0101u << shift);
                m_code.jmp(laneDone);
                m_code.L(overflow);
                m_code.or_(
                    m_code.r8d,
                    0x1000u << shift);
                m_code.L(laneDone);
            }

            m_code.xor_(m_code.eax, m_code.eax);
            constexpr uint32_t masks[] = {
                0x000fu, 0x00f0u, 0x0f00u, 0xf000u};
            constexpr uint32_t bits[] = {
                0x1u, 0x2u, 0x4u, 0x8u};
            for (size_t index = 0u;
                 index < std::size(masks); ++index)
            {
                Label absent;
                m_code.test(m_code.r8d, masks[index]);
                m_code.jz(absent);
                m_code.or_(m_code.eax, bits[index]);
                m_code.L(absent);
            }

            constexpr int workingMacOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState, workingMac));
            constexpr int indexOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState,
                        fmacFlagIndex));
            constexpr int flagsOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState, fmacFlags));
            m_code.mov(
                m_code.dword[
                    m_code.rbx + workingMacOffset],
                m_code.r8d);
            m_code.movzx(
                m_code.ecx,
                m_code.byte[
                    m_code.rbx + indexOffset]);
            m_code.lea(
                m_code.ecx,
                m_code.ptr[
                    m_code.rcx + m_code.rcx * 2u]);
            m_code.shl(m_code.ecx, 2u);
            m_code.lea(
                m_code.rdx,
                m_code.ptr[
                    m_code.rbx + flagsOffset]);
            m_code.add(m_code.rdx, m_code.rcx);
            m_code.mov(
                m_code.byte[
                    m_code.rdx +
                    offsetof(Flags, active)],
                1u);
            m_code.mov(
                m_code.dword[
                    m_code.rdx +
                    offsetof(Flags, mac)],
                m_code.r8d);
            m_code.mov(
                m_code.dword[
                    m_code.rdx +
                    offsetof(Flags, status)],
                m_code.eax);
        }

        void emitUpperConversion(
            const VuIrInstructionPair &pair)
        {
            const uint32_t word = pair.upperWord;
            const uint8_t destination =
                static_cast<uint8_t>(
                    (word >> 21u) & 0x0fu);
            const uint8_t target =
                static_cast<uint8_t>(
                    (word >> 16u) & 0x1fu);
            const uint8_t source =
                static_cast<uint8_t>(
                    (word >> 11u) & 0x1fu);
            const int sourceOffset =
                stateOffset(
                    offsetof(VuExecutionState, vf) +
                    static_cast<size_t>(source) *
                        sizeof(float) * 4u);

            bool integerToFloat = false;
            uint32_t scaleBits = 0x3f800000u;
            switch (pair.upper.opcode)
            {
            case VuIrOpcode::UpperItof0:
                integerToFloat = true;
                break;
            case VuIrOpcode::UpperItof4:
                integerToFloat = true;
                scaleBits = 0x3d800000u;
                break;
            case VuIrOpcode::UpperItof12:
                integerToFloat = true;
                scaleBits = 0x39800000u;
                break;
            case VuIrOpcode::UpperItof15:
                integerToFloat = true;
                scaleBits = 0x38000000u;
                break;
            case VuIrOpcode::UpperFtoi0:
                break;
            case VuIrOpcode::UpperFtoi4:
                scaleBits = 0x41800000u;
                break;
            case VuIrOpcode::UpperFtoi12:
                scaleBits = 0x45800000u;
                break;
            case VuIrOpcode::UpperFtoi15:
                scaleBits = 0x47000000u;
                break;
            default:
                return;
            }

            m_code.movups(
                m_code.xmm0,
                m_code.ptr[
                    m_code.rbx + sourceOffset]);
            if (integerToFloat)
            {
                m_code.cvtdq2ps(
                    m_code.xmm0, m_code.xmm0);
                if (scaleBits != 0x3f800000u)
                {
                    m_code.mov(m_code.eax, scaleBits);
                    m_code.movd(
                        m_code.xmm1, m_code.eax);
                    m_code.shufps(
                        m_code.xmm1, m_code.xmm1, 0u);
                    m_code.mulps(
                        m_code.xmm0, m_code.xmm1);
                }
            }
            else
            {
                if (scaleBits != 0x3f800000u)
                {
                    m_code.mov(m_code.eax, scaleBits);
                    m_code.movd(
                        m_code.xmm1, m_code.eax);
                    m_code.shufps(
                        m_code.xmm1, m_code.xmm1, 0u);
                    m_code.mulps(
                        m_code.xmm0, m_code.xmm1);
                }

                // CVTTPS2DQ returns INT_MIN for every unrepresentable
                // input. Flip that value only for ordered positive
                // overflow, preserving INT_MIN for negative overflow
                // and NaN.
                m_code.mov(
                    m_code.eax, 0x4effffffu);
                m_code.movd(
                    m_code.xmm1, m_code.eax);
                m_code.shufps(
                    m_code.xmm1, m_code.xmm1, 0u);
                m_code.cmpltps(
                    m_code.xmm1, m_code.xmm0);
                m_code.cvttps2dq(
                    m_code.xmm0, m_code.xmm0);
                m_code.pxor(
                    m_code.xmm0, m_code.xmm1);
            }

            m_code.movups(
                m_code.ptr[
                    m_code.rsp + kResultOffset],
                m_code.xmm0);
            emitApplyResultToVf(
                target, destination);
        }

        void emitUpper(
            const VuIrInstructionPair &pair)
        {
            if (pair.upper.opcode == VuIrOpcode::Nop)
                return;

            switch (pair.upper.opcode)
            {
            case VuIrOpcode::UpperItof0:
            case VuIrOpcode::UpperItof4:
            case VuIrOpcode::UpperItof12:
            case VuIrOpcode::UpperItof15:
            case VuIrOpcode::UpperFtoi0:
            case VuIrOpcode::UpperFtoi4:
            case VuIrOpcode::UpperFtoi12:
            case VuIrOpcode::UpperFtoi15:
                emitUpperConversion(pair);
                return;
            default:
                break;
            }

            enum class Arithmetic
            {
                Add,
                Subtract,
                Multiply,
                Maximum,
                Minimum,
                FusedAdd,
                FusedSubtract,
            };
            enum class RightSource
            {
                Vector,
                Broadcast,
                Q,
                I,
            };

            Arithmetic arithmetic = Arithmetic::Add;
            RightSource rightSource = RightSource::Vector;
            bool accumulatorDestination = false;
            switch (pair.upper.opcode)
            {
            case VuIrOpcode::UpperAddBc:
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperSubBc:
                arithmetic = Arithmetic::Subtract;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMaxBc:
                arithmetic = Arithmetic::Maximum;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMiniBc:
                arithmetic = Arithmetic::Minimum;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMulBc:
                arithmetic = Arithmetic::Multiply;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMaddBc:
                arithmetic = Arithmetic::FusedAdd;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMsubBc:
                arithmetic = Arithmetic::FusedSubtract;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMulQ:
                arithmetic = Arithmetic::Multiply;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperMaxI:
                arithmetic = Arithmetic::Maximum;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperMulI:
                arithmetic = Arithmetic::Multiply;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperMiniI:
                arithmetic = Arithmetic::Minimum;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperAddQ:
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperMaddQ:
                arithmetic = Arithmetic::FusedAdd;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperAddI:
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperMaddI:
                arithmetic = Arithmetic::FusedAdd;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperSubQ:
                arithmetic = Arithmetic::Subtract;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperMsubQ:
                arithmetic = Arithmetic::FusedSubtract;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperSubI:
                arithmetic = Arithmetic::Subtract;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperMsubI:
                arithmetic = Arithmetic::FusedSubtract;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperAdd:
                break;
            case VuIrOpcode::UpperMul:
                arithmetic = Arithmetic::Multiply;
                break;
            case VuIrOpcode::UpperMadd:
                arithmetic = Arithmetic::FusedAdd;
                break;
            case VuIrOpcode::UpperMax:
                arithmetic = Arithmetic::Maximum;
                break;
            case VuIrOpcode::UpperSub:
                arithmetic = Arithmetic::Subtract;
                break;
            case VuIrOpcode::UpperMsub:
                arithmetic = Arithmetic::FusedSubtract;
                break;
            case VuIrOpcode::UpperMini:
                arithmetic = Arithmetic::Minimum;
                break;
            case VuIrOpcode::UpperAddaBc:
                accumulatorDestination = true;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperSubaBc:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Subtract;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMulaBc:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Multiply;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMaddaBc:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedAdd;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMsubaBc:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedSubtract;
                rightSource = RightSource::Broadcast;
                break;
            case VuIrOpcode::UpperMulaQ:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Multiply;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperMulaI:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Multiply;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperAddaQ:
                accumulatorDestination = true;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperMaddaQ:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedAdd;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperAddaI:
                accumulatorDestination = true;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperMaddaI:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedAdd;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperSubaQ:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Subtract;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperMsubaQ:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedSubtract;
                rightSource = RightSource::Q;
                break;
            case VuIrOpcode::UpperSubaI:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Subtract;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperMsubaI:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedSubtract;
                rightSource = RightSource::I;
                break;
            case VuIrOpcode::UpperAdda:
                accumulatorDestination = true;
                break;
            case VuIrOpcode::UpperMula:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Multiply;
                break;
            case VuIrOpcode::UpperMadda:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedAdd;
                break;
            case VuIrOpcode::UpperSuba:
                accumulatorDestination = true;
                arithmetic = Arithmetic::Subtract;
                break;
            case VuIrOpcode::UpperMsuba:
                accumulatorDestination = true;
                arithmetic = Arithmetic::FusedSubtract;
                break;
            default:
                return;
            }

            const uint32_t word = pair.upperWord;
            const uint8_t destination =
                static_cast<uint8_t>(
                    (word >> 21u) & 0x0fu);
            const uint8_t right =
                static_cast<uint8_t>(
                    (word >> 16u) & 0x1fu);
            const uint8_t source =
                static_cast<uint8_t>(
                    (word >> 11u) & 0x1fu);
            const uint8_t target =
                static_cast<uint8_t>(
                    (word >> 6u) & 0x1fu);
            const int sourceOffset =
                stateOffset(
                    offsetof(VuExecutionState, vf) +
                    static_cast<size_t>(source) *
                        sizeof(float) * 4u);
            const int rightOffset =
                stateOffset(
                    offsetof(VuExecutionState, vf) +
                    static_cast<size_t>(right) *
                        sizeof(float) * 4u);
            const int qOffset =
                stateOffset(
                    offsetof(VuExecutionState, q));
            const int iOffset =
                stateOffset(
                    offsetof(VuExecutionState, i));

            m_code.movups(
                m_code.xmm0,
                m_code.ptr[
                    m_code.rbx + sourceOffset]);
            switch (rightSource)
            {
            case RightSource::Vector:
                m_code.movups(
                    m_code.xmm1,
                    m_code.ptr[
                        m_code.rbx + rightOffset]);
                break;
            case RightSource::Broadcast:
                m_code.movups(
                    m_code.xmm1,
                    m_code.ptr[
                        m_code.rbx + rightOffset]);
                m_code.shufps(
                    m_code.xmm1, m_code.xmm1,
                    static_cast<uint8_t>(
                        pair.upper.selector * 0x55u));
                break;
            case RightSource::Q:
                m_code.movss(
                    m_code.xmm1,
                    m_code.dword[
                        m_code.rbx + qOffset]);
                m_code.shufps(
                    m_code.xmm1, m_code.xmm1, 0u);
                break;
            case RightSource::I:
                m_code.movss(
                    m_code.xmm1,
                    m_code.dword[
                        m_code.rbx + iOffset]);
                m_code.shufps(
                    m_code.xmm1, m_code.xmm1, 0u);
                break;
            }
            switch (arithmetic)
            {
            case Arithmetic::Add:
                m_code.addps(
                    m_code.xmm0, m_code.xmm1);
                break;
            case Arithmetic::Subtract:
                m_code.subps(
                    m_code.xmm0, m_code.xmm1);
                break;
            case Arithmetic::Multiply:
                m_code.mulps(
                    m_code.xmm0, m_code.xmm1);
                break;
            case Arithmetic::Maximum:
                m_code.maxps(
                    m_code.xmm0, m_code.xmm1);
                break;
            case Arithmetic::Minimum:
                m_code.minps(
                    m_code.xmm0, m_code.xmm1);
                break;
            case Arithmetic::FusedAdd:
                m_code.movaps(
                    m_code.xmm2, m_code.xmm0);
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[
                        m_code.rbx +
                        stateOffset(
                            offsetof(
                                VuExecutionState,
                                acc))]);
                m_code.vfmadd231ps(
                    m_code.xmm0,
                    m_code.xmm2,
                    m_code.xmm1);
                m_usesAvx = true;
                break;
            case Arithmetic::FusedSubtract:
                m_code.movaps(
                    m_code.xmm2, m_code.xmm0);
                m_code.movups(
                    m_code.xmm0,
                    m_code.ptr[
                        m_code.rbx +
                        stateOffset(
                            offsetof(
                                VuExecutionState,
                                acc))]);
                m_code.vfnmadd231ps(
                    m_code.xmm0,
                    m_code.xmm2,
                    m_code.xmm1);
                m_usesAvx = true;
                break;
            }
            m_code.movups(
                m_code.ptr[
                    m_code.rsp + kResultOffset],
                m_code.xmm0);
            if (accumulatorDestination)
            {
                emitApplyResultToOffset(
                    stateOffset(
                        offsetof(
                            VuExecutionState, acc)),
                    destination);
            }
            else
            {
                emitApplyResultToVf(
                    target, destination);
            }
            if (vuIrHasOpFlag(
                    pair.upper, VuIrOpFmac))
            {
                emitScheduleFmacFlags(destination);
            }
        }

        void emitInlinePair(
            const VuIrInstructionPair &pair,
            uint32_t codeSize,
            Xbyak::Label *fallback,
            const void *xgkickHelper,
            Xbyak::Label &dynamicExit,
            bool enforceZeroRegisters,
            bool handleBranchState,
            KnownFmacSlot knownFmacSlot,
            KnownViBackup knownViBackup,
            KnownDelayedScalar knownQ,
            uint32_t knownQCyclesRemaining,
            KnownDelayedScalar knownP,
            uint32_t knownPCyclesRemaining,
            bool preciseBudget)
        {
            using namespace Xbyak;
            constexpr int activeOffset =
                stateOffset(
                    offsetof(VuExecutionState, active));
            constexpr int pcOffset =
                stateOffset(
                    offsetof(VuExecutionState, pc));
            constexpr int ebitOffset =
                stateOffset(
                    offsetof(VuExecutionState, ebit));
            constexpr int branchPendingOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState, branchPending));
            constexpr int branchTargetOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState, branchTarget));
            constexpr int branchDelayOffset =
                stateOffset(
                    offsetof(
                        VuExecutionState, branchDelay));
            constexpr int xgkickActiveOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState, xgkick) +
                    offsetof(
                        VuPipelineState::Xgkick,
                        active));
            constexpr int xgkickTagBytesOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState, xgkick) +
                    offsetof(
                        VuPipelineState::Xgkick,
                        tagBytesRemaining));
            constexpr int xgkickCreditOffset =
                pipelineOffset(
                    offsetof(
                        VuPipelineState, xgkick) +
                    offsetof(
                        VuPipelineState::Xgkick,
                        cycleCredit));

            if (fallback)
            {
                m_code.test(m_code.rbx, m_code.rbx);
                m_code.jz(*fallback, m_code.T_NEAR);
                m_code.cmp(
                    m_code.byte[
                        m_code.rbx + activeOffset],
                    0u);
                m_code.je(*fallback, m_code.T_NEAR);
                m_code.cmp(
                    m_code.dword[
                        m_code.rbx + pcOffset],
                    pair.pc);
                m_code.jne(*fallback, m_code.T_NEAR);
                m_code.cmp(
                    m_code.byte[
                        m_code.rbx + ebitOffset],
                    0u);
                m_code.jne(*fallback, m_code.T_NEAR);
            }
            emitAdvancePipelines(
                knownFmacSlot, knownViBackup,
                knownQ, knownQCyclesRemaining,
                knownP, knownPCyclesRemaining);
            if (vuIrHasOpFlag(
                    pair.lower, VuIrOpQBarrier))
            {
                emitFlushDelayedScalar(
                    offsetof(
                        VuPipelineState, delayedQ),
                    offsetof(VuExecutionState, q));
            }
            if (pair.order ==
                VuIrPairOrder::LowerThenUpper)
            {
                emitLower(pair);
                emitUpper(pair);
            }
            else
            {
                emitUpper(pair);
                emitLower(pair);
            }

            if (enforceZeroRegisters)
            {
                const int vf0Offset =
                    stateOffset(
                        offsetof(VuExecutionState, vf));
                const int vi0Offset =
                    stateOffset(
                        offsetof(VuExecutionState, vi));
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + vf0Offset],
                    0u);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + vf0Offset + 4],
                    0u);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + vf0Offset + 8],
                    0u);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + vf0Offset + 12],
                    0x3f800000u);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + vi0Offset],
                    0u);
            }

            Xbyak::Label noActiveXgkick;
            m_code.xor_(m_code.r11d, m_code.r11d);
            m_code.cmp(
                m_code.byte[
                    m_code.rbx + xgkickActiveOffset],
                0u);
            m_code.je(noActiveXgkick);
            Xbyak::Label callXgkickHelper;
            m_code.cmp(
                m_code.dword[
                    m_code.rbx +
                    xgkickTagBytesOffset],
                0u);
            m_code.je(callXgkickHelper);
            m_code.cmp(
                m_code.dword[
                    m_code.rbx + xgkickCreditOffset],
                0u);
            m_code.jne(callXgkickHelper);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + xgkickCreditOffset],
                1u);
            m_code.jmp(noActiveXgkick);

            m_code.L(callXgkickHelper);
            if (m_usesAvx)
                m_code.vzeroupper();
#if defined(_WIN32)
            m_code.mov(
                m_code.rcx,
                reinterpret_cast<uint64_t>(
                    m_backend));
            m_code.mov(m_code.rdx, m_code.r12);
#else
            m_code.mov(
                m_code.rdi,
                reinterpret_cast<uint64_t>(
                    m_backend));
            m_code.mov(m_code.rsi, m_code.r12);
#endif
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    xgkickHelper));
            m_code.call(m_code.rax);
            m_code.mov(m_code.r11d, m_code.eax);
            m_code.L(noActiveXgkick);

            const uint32_t nextPc =
                pair.pc + 8u >= codeSize
                    ? 0u
                    : pair.pc + 8u;
            m_code.mov(
                m_code.dword[
                    m_code.rbx + pcOffset],
                nextPc);
            if (handleBranchState)
            {
                Xbyak::Label commitBranch;
                Xbyak::Label branchDone;
                m_code.cmp(
                    m_code.byte[
                        m_code.rbx +
                        branchPendingOffset],
                    0u);
                m_code.je(branchDone);
                m_code.cmp(
                    m_code.byte[
                        m_code.rbx +
                        branchDelayOffset],
                    0u);
                m_code.je(commitBranch);
                m_code.dec(
                    m_code.byte[
                        m_code.rbx +
                        branchDelayOffset]);
                m_code.jmp(branchDone);
                m_code.L(commitBranch);
                m_code.mov(
                    m_code.eax,
                    m_code.dword[
                        m_code.rbx +
                        branchTargetOffset]);
                m_code.and_(
                    m_code.eax, m_codeAddressMask);
                m_code.mov(
                    m_code.dword[
                        m_code.rbx + pcOffset],
                    m_code.eax);
                m_code.mov(
                    m_code.byte[
                        m_code.rbx +
                        branchPendingOffset],
                    0u);
                m_code.L(branchDone);
            }
            if (vuIrHasPairFlag(
                    pair, VuIrPairEnd))
            {
                m_code.mov(
                    m_code.byte[
                        m_code.rbx + ebitOffset],
                    1u);
            }
            m_code.inc(m_code.r14d);
            if (preciseBudget)
            {
                m_code.inc(
                    m_code.qword[
                        m_code.rsp +
                        kPrecisePairCountOffset]);
            }
            m_code.test(m_code.r11d, m_code.r11d);
            m_code.jnz(
                dynamicExit, m_code.T_NEAR);
        }

        void emitHelperPair(
            const VuIrInstructionPair &pair,
            const void *pairHelper,
            Xbyak::Label &dynamicExit,
            bool preciseBudget,
            bool checkedBudget)
        {
            if (m_usesAvx)
                m_code.vzeroupper();
            const uint64_t words =
                static_cast<uint64_t>(pair.lowerWord) |
                (static_cast<uint64_t>(
                     pair.upperWord)
                 << 32u);
#if defined(_WIN32)
            m_code.mov(
                m_code.rcx,
                reinterpret_cast<uint64_t>(
                    m_backend));
            m_code.mov(m_code.rdx, m_code.r12);
            m_code.mov(m_code.r8d, pair.pc);
            m_code.mov(m_code.r9, words);
#else
            m_code.mov(
                m_code.rdi,
                reinterpret_cast<uint64_t>(
                    m_backend));
            m_code.mov(m_code.rsi, m_code.r12);
            m_code.mov(m_code.edx, pair.pc);
            m_code.mov(m_code.rcx, words);
#endif
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    pairHelper));
            m_code.call(m_code.rax);

            m_code.mov(m_code.r11d, m_code.eax);
            m_code.test(
                m_code.r11d,
                kNativePairExecuted);
            Xbyak::Label pairNotExecuted;
            Xbyak::Label pairAccounted;
            m_code.jz(
                preciseBudget
                    ? pairNotExecuted
                    : pairAccounted);
            m_code.inc(m_code.r14d);
            if (preciseBudget)
            {
                m_code.inc(
                    m_code.qword[
                        m_code.rsp +
                        kPrecisePairCountOffset]);
            }
            if (m_blockProfile)
            {
                m_code.mov(
                    m_code.rax,
                    reinterpret_cast<uint64_t>(
                        m_blockProfile));
                m_code.inc(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuBlockProfileRecord,
                            helperBarriers)]);
            }
            if (checkedBudget)
            {
                m_code.jmp(pairAccounted);
                m_code.L(pairNotExecuted);
                m_code.mov(
                    m_code.rax,
                    reinterpret_cast<uint64_t>(
                        m_chainRuntime));
                m_code.inc(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            preciseNonRetiredChecks)]);
            }
            m_code.L(pairAccounted);
            m_code.and_(
                m_code.r11d,
                kNativePairExitMask);
            m_code.test(m_code.r11d, m_code.r11d);
            m_code.jnz(
                dynamicExit, m_code.T_NEAR);
        }

        void emitResetChainRuntime(bool rawEntry)
        {
            if (!m_chainRuntime)
                return;
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    m_chainRuntime));
            if (rawEntry)
            {
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            codeGeneration)],
                    0u);
            }
            m_code.mov(
                m_code.qword[
                    m_code.rax +
                    offsetof(
                        VuNativeChainRuntime,
                        slowLinkSource)],
                0u);
            m_code.mov(
                m_code.dword[
                    m_code.rax +
                    offsetof(
                        VuNativeChainRuntime,
                        slowLinkTargetPc)],
                0u);
            if (m_blockBudgetGuardsEnabled)
            {
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            preciseTailEntries)],
                    0u);
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            preciseTailPairs)],
                    0u);
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            preciseNonRetiredChecks)],
                    0u);
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            returnBoundaryChecks)],
                    0u);
                m_code.mov(
                    m_code.qword[
                        m_code.rsp +
                        kPrecisePairCountOffset],
                    0u);
            }
        }

        void emitProfileEntry()
        {
            if (!m_blockProfile || !m_chainRuntime)
                return;

            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    m_blockProfile));
            m_code.inc(
                m_code.qword[
                    m_code.rax +
                    offsetof(
                        VuBlockProfileRecord,
                        executions)]);

            m_code.mov(
                m_code.r8,
                reinterpret_cast<uint64_t>(
                    m_chainRuntime));
            m_code.mov(
                m_code.dword[
                    m_code.r8 +
                    offsetof(
                        VuNativeChainRuntime,
                        blockStartCycles)],
                m_code.r14d);
        }

        void emitProfileBudgetClass(bool precise)
        {
            if (!m_blockProfile)
                return;
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    m_blockProfile));
            m_code.inc(
                m_code.qword[
                    m_code.rax +
                    (precise
                         ? offsetof(
                               VuBlockProfileRecord,
                               boundedEntries)
                         : offsetof(
                               VuBlockProfileRecord,
                               fullBudgetEntries))]);
        }

        void emitProfileLegacyBudgetClass()
        {
            if (!m_blockProfile)
                return;

            Xbyak::Label precise;
            Xbyak::Label classified;
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    m_blockProfile));
            m_code.mov(m_code.r11d, m_code.r13d);
            m_code.sub(m_code.r11d, m_code.r14d);
            m_code.cmp(
                m_code.r11d,
                m_blockProfile->fixedCycles);
            m_code.jb(precise);
            m_code.inc(
                m_code.qword[
                    m_code.rax +
                    offsetof(
                        VuBlockProfileRecord,
                        fullBudgetEntries)]);
            m_code.jmp(classified);
            m_code.L(precise);
            m_code.inc(
                m_code.qword[
                    m_code.rax +
                    offsetof(
                        VuBlockProfileRecord,
                        boundedEntries)]);
            m_code.L(classified);
        }

        // r10d contains the native exit and remains valid on return.
        void emitProfileExit()
        {
            if (!m_blockProfile || !m_chainRuntime)
                return;

            m_code.mov(
                m_code.r8,
                reinterpret_cast<uint64_t>(
                    m_chainRuntime));
            m_code.mov(m_code.r11d, m_code.r14d);
            m_code.sub(
                m_code.r11d,
                m_code.dword[
                    m_code.r8 +
                    offsetof(
                        VuNativeChainRuntime,
                        blockStartCycles)]);
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    m_blockProfile));
            m_code.add(
                m_code.qword[
                    m_code.rax +
                    offsetof(
                        VuBlockProfileRecord,
                        guestPairs)],
                m_code.r11);

            Xbyak::Label validExit;
            m_code.cmp(
                m_code.r10d,
                static_cast<uint32_t>(
                    kVuNativeBlockExitCount));
            m_code.jb(validExit);
            m_code.mov(
                m_code.r10d,
                static_cast<uint32_t>(
                    VuNativeBlockExit::Fault));
            m_code.L(validExit);
            m_code.mov(m_code.r11d, m_code.r10d);
            m_code.inc(
                m_code.qword[
                    m_code.rax +
                    m_code.r11 * 8 +
                    offsetof(
                        VuBlockProfileRecord,
                        exitReasons)]);
        }

        void emitProfileLinkedEdge()
        {
            if (!m_blockProfile)
                return;
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    m_blockProfile));
            m_code.inc(
                m_code.qword[
                    m_code.rax +
                    offsetof(
                        VuBlockProfileRecord,
                        linkedEdges)]);
        }

        void emitLinkOrComplete(
            Xbyak::Label &packResult)
        {
            using namespace Xbyak;
            Label complete;
            Label miss;

            if (!m_blockBudgetGuardsEnabled)
            {
                m_code.cmp(
                    m_code.r14d, m_code.r13d);
                m_code.jae(
                    complete, m_code.T_NEAR);
            }
            if (!m_blockLinkingEnabled ||
                !m_outgoingLinks || !m_chainRuntime)
            {
                m_code.jmp(complete, m_code.T_NEAR);
            }
            else
            {
                m_code.bt(m_code.r15, 63u);
                m_code.jc(complete, m_code.T_NEAR);
                constexpr int pcOffset =
                    stateOffset(
                        offsetof(
                            VuExecutionState, pc));
                m_code.mov(
                    m_code.r9d,
                    m_code.dword[
                        m_code.rbx + pcOffset]);

                const auto emitSlot =
                    [&](VuNativeLinkSlot &slot)
                    {
                        Label next;
                        m_code.mov(
                            m_code.r10,
                            reinterpret_cast<uint64_t>(
                                &slot));
                        m_code.cmp(
                            m_code.r9d,
                            m_code.dword[
                                m_code.r10 +
                                offsetof(
                                    VuNativeLinkSlot,
                                    targetPc)]);
                        m_code.jne(
                            next, m_code.T_NEAR);
                        m_code.mov(
                            m_code.r11,
                            reinterpret_cast<uint64_t>(
                                m_chainRuntime));
                        m_code.mov(
                            m_code.rax,
                            m_code.qword[
                                m_code.r11 +
                                offsetof(
                                    VuNativeChainRuntime,
                                    codeGeneration)]);
                        m_code.cmp(
                            m_code.qword[
                                m_code.r10 +
                                offsetof(
                                    VuNativeLinkSlot,
                                    codeGeneration)],
                            m_code.rax);
                        m_code.jne(
                            miss, m_code.T_NEAR);
                        m_code.mov(
                            m_code.r11,
                            m_code.qword[
                                m_code.r10 +
                                offsetof(
                                    VuNativeLinkSlot,
                                    targetEntry)]);
                        m_code.test(
                            m_code.r11, m_code.r11);
                        m_code.jz(
                            miss, m_code.T_NEAR);

                        m_code.mov(
                            m_code.r10d,
                            static_cast<uint32_t>(
                                VuNativeBlockExit::
                                    BlockComplete));
                        emitProfileExit();
                        if (m_usesAvx)
                            m_code.vzeroupper();
                        if (m_blockProfile)
                        {
                            m_code.mov(
                                m_code.r11,
                                reinterpret_cast<uint64_t>(
                                    &slot));
                            m_code.mov(
                                m_code.r11,
                                m_code.qword[
                                    m_code.r11 +
                                    offsetof(
                                        VuNativeLinkSlot,
                                        targetEntry)]);
                        }
                        if (m_blockBudgetGuardsEnabled)
                        {
                            emitProfileLinkedEdge();
                            m_code.inc(m_code.r15);
                            m_code.mov(
                                m_code.r10,
                                reinterpret_cast<uint64_t>(
                                    m_blockProfile));
                        }
                        else
                        {
                            emitProfileLinkedEdge();
                            m_code.inc(m_code.r15);
                        }
                        m_code.jmp(m_code.r11);
                        m_code.L(next);
                    };

                for (VuNativeLinkSlot &slot :
                     m_outgoingLinks->staticTargets)
                {
                    emitSlot(slot);
                }
                for (VuNativeLinkSlot &slot :
                     m_outgoingLinks->dynamicTargets)
                {
                    emitSlot(slot);
                }

                m_code.L(miss);
                m_code.mov(
                    m_code.rax,
                    reinterpret_cast<uint64_t>(
                        m_chainRuntime));
                m_code.mov(
                    m_code.r11,
                    reinterpret_cast<uint64_t>(
                        m_outgoingLinks));
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            slowLinkSource)],
                    m_code.r11);
                m_code.mov(
                    m_code.dword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            slowLinkTargetPc)],
                    m_code.r9d);
            }

            m_code.L(complete);
            if (m_blockBudgetGuardsEnabled)
            {
                Label blockComplete;
                m_code.mov(
                    m_code.rax,
                    reinterpret_cast<uint64_t>(
                        m_chainRuntime));
                m_code.inc(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            returnBoundaryChecks)]);
                m_code.cmp(
                    m_code.r14d, m_code.r13d);
                m_code.jb(blockComplete);
                if (m_chainRuntime)
                {
                    m_code.mov(
                        m_code.qword[
                            m_code.rax +
                            offsetof(
                                VuNativeChainRuntime,
                                slowLinkSource)],
                        0u);
                }
                m_code.mov(
                    m_code.r10d,
                    static_cast<uint32_t>(
                        VuNativeBlockExit::
                            CycleBudget));
                m_code.jmp(
                    packResult, m_code.T_NEAR);
                m_code.L(blockComplete);
            }
            m_code.mov(
                m_code.r10d,
                static_cast<uint32_t>(
                    VuNativeBlockExit::BlockComplete));
            m_code.jmp(packResult, m_code.T_NEAR);
        }

        void emitEntryFrame()
        {
            m_code.push(m_code.rbx);
            m_code.push(m_code.r12);
            m_code.push(m_code.r13);
            m_code.push(m_code.r14);
            m_code.push(m_code.r15);
            m_code.sub(m_code.rsp, kStackBytes);

#if defined(_WIN32)
            m_code.mov(m_code.r12, m_code.rcx);
            m_code.mov(m_code.r13d, m_code.edx);
#else
            m_code.mov(m_code.r12, m_code.rdi);
            m_code.mov(m_code.r13d, m_code.esi);
#endif
            m_code.xor_(m_code.r14d, m_code.r14d);
            m_code.xor_(m_code.r15d, m_code.r15d);
        }

        void emitRawEntry(
            Xbyak::Label &body)
        {
            emitEntryFrame();
            m_code.mov(
                m_code.r15,
                uint64_t{1u} << 63u);
            emitResetChainRuntime(true);
            m_code.mov(
                m_code.byte[
                    m_code.rsp + kRawEntryFlagOffset],
                1u);
            m_code.stmxcsr(
                m_code.dword[
                    m_code.rsp + kSavedMxcsrOffset]);
            m_code.mov(
                m_code.eax,
                m_code.dword[
                    m_code.rsp + kSavedMxcsrOffset]);
            m_code.and_(
                m_code.eax,
                static_cast<uint32_t>(
                    ~kMxcsrRoundingMask));
            m_code.or_(m_code.eax, kVuMxcsrBits);
            m_code.mov(
                m_code.dword[
                    m_code.rsp + kVuMxcsrOffset],
                m_code.eax);
            m_code.ldmxcsr(
                m_code.dword[
                    m_code.rsp + kVuMxcsrOffset]);

#if defined(_WIN32)
            m_code.mov(m_code.rcx, m_code.r12);
            m_code.lea(
                m_code.rdx,
                m_code.ptr[
                    m_code.rsp + kContextViewOffset]);
#else
            m_code.mov(m_code.rdi, m_code.r12);
            m_code.lea(
                m_code.rsi,
                m_code.ptr[
                    m_code.rsp + kContextViewOffset]);
#endif
            m_code.mov(
                m_code.rax,
                reinterpret_cast<uint64_t>(
                    &loadNativeContextView));
            m_code.call(m_code.rax);
            m_code.mov(
                m_code.rbx,
                m_code.qword[
                    m_code.rsp +
                    kContextViewOffset +
                    offsetof(
                        NativeContextView, state)]);
            m_code.jmp(body, m_code.T_NEAR);
        }

        void emitFastEntry(
            Xbyak::Label &body)
        {
            emitEntryFrame();
            emitResetChainRuntime(false);
            m_code.mov(
                m_code.byte[
                    m_code.rsp + kRawEntryFlagOffset],
                0u);
#if defined(_WIN32)
            const Xbyak::Reg64 &view = m_code.r8;
#else
            const Xbyak::Reg64 &view = m_code.rdx;
#endif
            m_code.mov(
                m_code.rbx,
                m_code.qword[
                    view +
                    offsetof(
                        NativeContextView, state)]);
            m_code.mov(
                m_code.rax,
                m_code.qword[
                    view +
                    offsetof(
                        NativeContextView, data)]);
            m_code.mov(
                m_code.qword[
                    m_code.rsp +
                    kContextViewOffset +
                    offsetof(
                        NativeContextView, data)],
                m_code.rax);
            m_code.mov(
                m_code.eax,
                m_code.dword[
                    view +
                    offsetof(
                        NativeContextView, dataSize)]);
            m_code.mov(
                m_code.dword[
                    m_code.rsp +
                    kContextViewOffset +
                    offsetof(
                        NativeContextView, dataSize)],
                m_code.eax);
            m_code.jmp(body, m_code.T_NEAR);
        }

        void emit(
            const VuIrBlock &block,
            const void *pairHelper,
            const void *xgkickHelper)
        {
            using namespace Xbyak;

            Label packResult;
            Label packUnprofiledResult;
            Label epilogue;
            Label body;
            Label preciseBody;
            Label linkedZeroBudget;
            Label linkedZeroProfileDone;
            Label skipFloatModeRestore;

            emitRawEntry(body);
            m_fastEntryOffset = m_code.getSize();
            emitFastEntry(body);
            uint32_t retiringCycles = 0u;
            for (const VuIrInstructionPair &pair :
                 block.pairs)
            {
                if (vuIrHasPairFlag(
                        pair, VuIrPairUnsupported))
                {
                    break;
                }
                retiringCycles += pair.cycles;
            }
            uint32_t fullGuardCycles =
                retiringCycles;
            if (block.exit ==
                    VuIrBlockExit::
                        UnsupportedInstruction &&
                fullGuardCycles !=
                    std::numeric_limits<uint32_t>::max())
            {
                ++fullGuardCycles;
            }
            if (m_blockBudgetGuardsEnabled)
            {
                Label targetSelector;

                // Public and fast host entries start a fresh native chain.
                // Keep their setup out of the fall-through linked path.
                m_code.L(body);
                m_code.mov(m_code.r11d, m_code.r13d);
                m_code.sub(m_code.r11d, m_code.r14d);
                m_code.jmp(
                    targetSelector, m_code.T_NEAR);

                // Linked entries already have an active native frame. The
                // accepted nonzero case falls directly into the selector;
                // terminal correction stays in a cold out-of-line path.
                m_linkedEntryOffset = m_code.getSize();
                m_code.mov(m_code.r11d, m_code.r13d);
                m_code.sub(m_code.r11d, m_code.r14d);
                m_code.test(
                    m_code.r11d, m_code.r11d);
                m_code.jz(
                    linkedZeroBudget,
                    m_code.T_NEAR);
                m_code.L(targetSelector);
                emitProfileEntry();
                m_code.cmp(
                    m_code.r11d,
                    fullGuardCycles);
                m_code.jb(
                    preciseBody, m_code.T_NEAR);
                emitProfileBudgetClass(false);
            }
            else
            {
                m_linkedEntryOffset = m_code.getSize();
                m_code.L(body);
                emitProfileEntry();
                emitProfileLegacyBudgetClass();
            }

            const auto emitBody =
                [&](bool checkedBudget,
                    bool preciseBudget)
                {
                    Label cycleBudgetExit;
                    Label controlFlowExit;
                    Label dynamicExit;
                    Label linkOrComplete;
                    Label invalidPipelineState;
                    Label pipelineStateValid;

                    m_code.test(
                        m_code.rbx, m_code.rbx);
                    m_code.jz(pipelineStateValid);
                    m_code.cmp(
                        m_code.byte[
                            m_code.rbx +
                            stateOffset(
                                offsetof(
                                    VuExecutionState,
                                    viBackupCycles))],
                        2u);
                    m_code.ja(
                        invalidPipelineState,
                        m_code.T_NEAR);
                    m_code.L(pipelineStateValid);

                    if (block.pairs.empty())
                    {
                        const VuNativeBlockExit exit =
                            block.exit == VuIrBlockExit::CodeBounds
                                ? VuNativeBlockExit::CodeBounds
                                : VuNativeBlockExit::Fault;
                        m_code.mov(
                            m_code.r10d,
                            static_cast<uint32_t>(exit));
                        m_code.jmp(packResult, m_code.T_NEAR);
                    }

                    bool viBackupKnown = false;
                    uint8_t viBackupCycles = 2u;
                    bool qKnown = false;
                    uint32_t qCyclesRemaining = 0u;
                    bool pKnown = false;
                    uint32_t pCyclesRemaining = 0u;
                    for (size_t pairIndex = 0u;
                         pairIndex < block.pairs.size();
                         ++pairIndex)
                    {
                        const VuIrInstructionPair &pair =
                            block.pairs[pairIndex];
                        if (checkedBudget)
                        {
                            m_code.cmp(
                                m_code.r14d, m_code.r13d);
                            m_code.jae(
                                cycleBudgetExit,
                                m_code.T_NEAR);
                        }

                        if (vuIrHasPairFlag(
                                pair, VuIrPairUnsupported))
                        {
                            m_code.mov(
                                m_code.r10d,
                                static_cast<uint32_t>(
                                    VuNativeBlockExit::
                                        UnsupportedInstruction));
                            m_code.jmp(packResult, m_code.T_NEAR);
                            break;
                        }

                        const bool ebitDelayPair =
                            pairIndex != 0u &&
                            vuIrHasPairFlag(
                                block.pairs[pairIndex - 1u],
                                VuIrPairEnd);
                        const bool inlinePair =
                            !m_instrumented &&
                            canInlinePair(pair) &&
                            !ebitDelayPair;
                        const bool handleBranchState =
                            pairIndex == 0u ||
                            vuIrHasPairFlag(
                                pair, VuIrPairBranch) ||
                            (pairIndex != 0u &&
                             vuIrHasPairFlag(
                                 block.pairs[pairIndex - 1u],
                                 VuIrPairBranch));
                        KnownFmacSlot knownFmacSlot =
                            KnownFmacSlot::Dynamic;
                        if (pairIndex >=
                            VuPipelineState::kFmacFlagStages)
                        {
                            const VuIrInstructionPair &producer =
                                block.pairs[
                                    pairIndex -
                                    VuPipelineState::
                                        kFmacFlagStages];
                            knownFmacSlot =
                                vuIrHasOpFlag(
                                    producer.upper,
                                    VuIrOpFmac)
                                    ? KnownFmacSlot::Active
                                    : KnownFmacSlot::Inactive;
                        }
                        const KnownViBackup knownViBackup =
                            !viBackupKnown
                                ? KnownViBackup::Dynamic
                                : viBackupCycles != 0u
                                      ? KnownViBackup::Active
                                      : KnownViBackup::Inactive;
                        const KnownDelayedScalar knownQ =
                            !qKnown
                                ? KnownDelayedScalar::Dynamic
                                : qCyclesRemaining != 0u
                                      ? KnownDelayedScalar::Active
                                      : KnownDelayedScalar::Inactive;
                        const KnownDelayedScalar knownP =
                            !pKnown
                                ? KnownDelayedScalar::Dynamic
                                : pCyclesRemaining != 0u
                                      ? KnownDelayedScalar::Active
                                      : KnownDelayedScalar::Inactive;

                        if (inlinePair)
                        {
                            if (pairIndex == 0u)
                            {
                                Label fallback;
                                Label done;
                                emitInlinePair(
                                    pair, block.codeSize,
                                    &fallback, xgkickHelper,
                                    dynamicExit, true,
                                    handleBranchState,
                                    knownFmacSlot,
                                    knownViBackup,
                                    knownQ, qCyclesRemaining,
                                    knownP, pCyclesRemaining,
                                    preciseBudget);
                                m_code.jmp(
                                    done, m_code.T_NEAR);
                                m_code.L(fallback);
                                emitHelperPair(
                                    pair, pairHelper,
                                    dynamicExit,
                                    preciseBudget,
                                    checkedBudget);
                                m_code.L(done);
                            }
                            else
                            {
                                emitInlinePair(
                                    pair, block.codeSize,
                                    nullptr, xgkickHelper,
                                    dynamicExit, false,
                                    handleBranchState,
                                    knownFmacSlot,
                                    knownViBackup,
                                    knownQ, qCyclesRemaining,
                                    knownP, pCyclesRemaining,
                                    preciseBudget);
                            }
                        }
                        else
                        {
                            emitHelperPair(
                                pair, pairHelper, dynamicExit,
                                preciseBudget,
                                checkedBudget);
                        }

                        if (viBackupCycles != 0u)
                            --viBackupCycles;
                        if (!viBackupKnown &&
                            viBackupCycles == 0u)
                        {
                            viBackupKnown = true;
                        }
                        uint8_t backedUpRegister = 0u;
                        switch (pair.lower.opcode)
                        {
                        case VuIrOpcode::LowerLqi:
                        case VuIrOpcode::LowerLqd:
                            backedUpRegister =
                                static_cast<uint8_t>(
                                    (pair.lowerWord >> 11u) &
                                    0x0fu);
                            break;
                        case VuIrOpcode::LowerSqi:
                        case VuIrOpcode::LowerSqd:
                        case VuIrOpcode::LowerMtir:
                            backedUpRegister =
                                static_cast<uint8_t>(
                                    (pair.lowerWord >> 16u) &
                                    0x0fu);
                            break;
                        default:
                            break;
                        }
                        if (backedUpRegister != 0u)
                        {
                            viBackupKnown = true;
                            viBackupCycles = 2u;
                        }
                        if (qKnown &&
                            qCyclesRemaining != 0u)
                        {
                            --qCyclesRemaining;
                        }
                        if (vuIrHasOpFlag(
                                pair.lower, VuIrOpQBarrier))
                        {
                            qKnown = true;
                            qCyclesRemaining =
                                vuIrHasOpFlag(
                                    pair.lower,
                                    VuIrOpWritesQ)
                                    ? pair.lower.latency
                                    : 0u;
                        }
                        if (pKnown &&
                            pCyclesRemaining != 0u)
                        {
                            --pCyclesRemaining;
                        }
                        if (vuIrHasOpFlag(
                                pair.lower, VuIrOpPBarrier))
                        {
                            pKnown = true;
                            pCyclesRemaining =
                                vuIrHasOpFlag(
                                    pair.lower,
                                    VuIrOpWritesP)
                                    ? pair.lower.latency
                                    : 0u;
                        }

                        if (pairIndex + 1u <
                                block.pairs.size() &&
                            (!inlinePair ||
                             handleBranchState))
                        {
                            constexpr int pcOffset =
                                stateOffset(
                                    offsetof(
                                        VuExecutionState, pc));
                            m_code.cmp(
                                m_code.dword[
                                    m_code.rbx + pcOffset],
                                block.pairs[
                                    pairIndex + 1u].pc);
                            m_code.jne(
                                controlFlowExit,
                                m_code.T_NEAR);
                        }
                    }

                    switch (block.exit)
                    {
                    case VuIrBlockExit::PairLimit:
                    case VuIrBlockExit::BranchBoundary:
                        m_code.jmp(
                            linkOrComplete, m_code.T_NEAR);
                        break;
                    case VuIrBlockExit::XgkickBoundary:
                        m_code.mov(
                            m_code.r10d,
                            static_cast<uint32_t>(
                                VuNativeBlockExit::XgkickBoundary));
                        break;
                    case VuIrBlockExit::UnsupportedInstruction:
                        m_code.mov(
                            m_code.r10d,
                            static_cast<uint32_t>(
                                VuNativeBlockExit::
                                    UnsupportedInstruction));
                        break;
                    case VuIrBlockExit::CodeBounds:
                        m_code.mov(
                            m_code.r10d,
                            static_cast<uint32_t>(
                                VuNativeBlockExit::CodeBounds));
                        break;
                    case VuIrBlockExit::ProgramEndBoundary:
                        // The E-bit delay pair must have returned ProgramEnded.
                        // Falling through means the helper and verified IR disagree.
                        m_code.mov(
                            m_code.r10d,
                            static_cast<uint32_t>(
                                VuNativeBlockExit::Fault));
                        break;
                    }
                    m_code.jmp(packResult, m_code.T_NEAR);

                    m_code.L(cycleBudgetExit);
                    m_code.mov(
                        m_code.r10d,
                        static_cast<uint32_t>(
                            VuNativeBlockExit::CycleBudget));
                    m_code.jmp(packResult, m_code.T_NEAR);

                    m_code.L(controlFlowExit);
                    m_code.jmp(
                        linkOrComplete, m_code.T_NEAR);

                    m_code.L(dynamicExit);
                    m_code.mov(m_code.r10d, m_code.r11d);

                    m_code.jmp(packResult, m_code.T_NEAR);
                    m_code.L(invalidPipelineState);
                    m_code.mov(
                        m_code.r10d,
                        static_cast<uint32_t>(
                            VuNativeBlockExit::Fault));
                    m_code.jmp(packResult, m_code.T_NEAR);

                    m_code.L(linkOrComplete);
                    emitLinkOrComplete(packResult);
                };

            emitBody(
                !m_blockBudgetGuardsEnabled,
                false);
            m_code.L(packResult);
            emitProfileExit();
            m_code.L(packUnprofiledResult);
            if (m_chainRuntime)
            {
                m_code.mov(
                    m_code.rax,
                    reinterpret_cast<uint64_t>(
                        m_chainRuntime));
                m_code.mov(
                    m_code.r11,
                    m_code.r15);
                m_code.btr(m_code.r11, 63u);
                m_code.mov(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            linkedEdges)],
                    m_code.r11);
                if (m_blockBudgetGuardsEnabled)
                {
                    m_code.mov(
                        m_code.r11,
                        m_code.qword[
                            m_code.rsp +
                            kPrecisePairCountOffset]);
                    m_code.mov(
                        m_code.qword[
                            m_code.rax +
                            offsetof(
                                VuNativeChainRuntime,
                                preciseTailPairs)],
                        m_code.r11);
                }
            }
            m_code.mov(m_code.eax, m_code.r14d);
            m_code.mov(m_code.r11d, m_code.r10d);
            m_code.shl(m_code.r11, 32u);
            m_code.or_(m_code.rax, m_code.r11);
            m_code.jmp(epilogue);

            m_code.L(epilogue);
            if (m_usesAvx)
                m_code.vzeroupper();
            m_code.cmp(
                m_code.byte[
                    m_code.rsp + kRawEntryFlagOffset],
                0u);
            m_code.je(skipFloatModeRestore);
            m_code.ldmxcsr(
                m_code.dword[
                    m_code.rsp + kSavedMxcsrOffset]);
            m_code.L(skipFloatModeRestore);
            m_code.add(m_code.rsp, kStackBytes);
            m_code.pop(m_code.r15);
            m_code.pop(m_code.r14);
            m_code.pop(m_code.r13);
            m_code.pop(m_code.r12);
            m_code.pop(m_code.rbx);
            m_code.ret();

            if (m_blockBudgetGuardsEnabled)
            {
                m_code.L(preciseBody);
                m_code.mov(
                    m_code.rax,
                    reinterpret_cast<uint64_t>(
                        m_chainRuntime));
                m_code.inc(
                    m_code.qword[
                        m_code.rax +
                        offsetof(
                            VuNativeChainRuntime,
                            preciseTailEntries)]);
                emitProfileBudgetClass(true);
                emitBody(true, true);

                constexpr int blockCompleteProfileOffset =
                    static_cast<int>(
                        offsetof(
                            VuBlockProfileRecord,
                            exitReasons) +
                        sizeof(uint64_t) *
                            static_cast<size_t>(
                                VuNativeBlockExit::
                                    BlockComplete));
                constexpr int cycleBudgetProfileOffset =
                    static_cast<int>(
                        offsetof(
                            VuBlockProfileRecord,
                            exitReasons) +
                        sizeof(uint64_t) *
                            static_cast<size_t>(
                                VuNativeBlockExit::
                                    CycleBudget));
                m_code.L(linkedZeroBudget);
                m_code.dec(m_code.r15);
                m_code.test(
                    m_code.r10, m_code.r10);
                m_code.jz(linkedZeroProfileDone);
                m_code.dec(
                    m_code.qword[
                        m_code.r10 +
                        offsetof(
                            VuBlockProfileRecord,
                            linkedEdges)]);
                m_code.dec(
                    m_code.qword[
                        m_code.r10 +
                        blockCompleteProfileOffset]);
                m_code.inc(
                    m_code.qword[
                        m_code.r10 +
                        cycleBudgetProfileOffset]);
                m_code.L(linkedZeroProfileDone);
                m_code.mov(
                    m_code.r10d,
                    static_cast<uint32_t>(
                        VuNativeBlockExit::
                            CycleBudget));
                m_code.jmp(
                    packUnprofiledResult,
                    m_code.T_NEAR);
            }
        }
    };
#endif
}

std::string_view vuNativeBlockExitName(
    VuNativeBlockExit exit)
{
    switch (exit)
    {
    case VuNativeBlockExit::BlockComplete:
        return "block-complete";
    case VuNativeBlockExit::CycleBudget:
        return "cycle-budget";
    case VuNativeBlockExit::Inactive:
        return "inactive";
    case VuNativeBlockExit::ProgramEnded:
        return "program-ended";
    case VuNativeBlockExit::CodeBounds:
        return "code-bounds";
    case VuNativeBlockExit::XgkickBoundary:
        return "xgkick-boundary";
    case VuNativeBlockExit::DebugObserver:
        return "debug-observer";
    case VuNativeBlockExit::CodeInvalidated:
        return "code-invalidated";
    case VuNativeBlockExit::UnsupportedInstruction:
        return "unsupported-instruction";
    case VuNativeBlockExit::Fault:
        return "fault";
    }
    return "unknown";
}

std::string vuPerfJitSymbolName(
    const VuProgramKey &key,
    const VuIrBlock &block,
    uint64_t compilationIdentity)
{
    const uint32_t firstPc =
        block.pairs.empty()
            ? key.entryPc
            : block.pairs.front().pc;
    const uint32_t lastPc =
        block.pairs.empty()
            ? key.entryPc
            : block.pairs.back().pc;
    std::ostringstream output;
    output
        << "ps2x-"
        << (key.unit == VuUnitId::Vu0
                ? "vu0" : "vu1")
        << "-code-"
        << std::hex << std::setfill('0')
        << std::setw(16)
        << key.codeContentIdentity
        << "-size-"
        << std::setw(4) << key.codeSize
        << "-gen-"
        << std::setw(16)
        << key.codeGeneration
        << "-entry-"
        << std::setw(4) << key.entryPc
        << "-pcs-"
        << std::setw(4) << firstPc
        << "-" << std::setw(4) << lastPc
        << "-"
        << (key.compilationMode ==
                    VuCompilationMode::Instrumented
                ? "instrumented" : "normal")
        << "-"
        << (key.blockForm ==
                    VuIrBlockForm::Basic
                ? "basic" : "linear")
        << "-compile-"
        << std::setw(16)
        << compilationIdentity;
    return output.str();
}

VuRecompilerBackend::VuRecompilerBackend(VuUnit &unit)
    : VuRecompilerBackend(
          unit,
          blockProfilingConfigurationFromEnvironment())
{
}

VuRecompilerBackend::VuRecompilerBackend(
    VuUnit &unit,
    VuBlockProfilingConfiguration configuration)
    : m_unit(unit), m_semantics(unit)
{
    const char *const linking =
        std::getenv("PS2X_VU_BLOCK_LINKING");
    m_blockLinkingEnabled =
        !linking ||
        environmentFlagEnabled(linking);
    const char *const budgetGuards =
        std::getenv(
            "PS2X_VU_BLOCK_BUDGET_GUARDS");
    m_blockBudgetGuardsEnabled =
        !budgetGuards ||
        environmentFlagEnabled(budgetGuards);
    m_blockProfilingEnabled = configuration.enabled;
    m_maximumBlockProfiles =
        configuration.maximumRecords;
    m_perfJitDumpEnabled =
        ps2PerfJitDumpEnvironmentRequested();
    if (m_perfJitDumpEnabled)
    {
        (void)ps2PerfJitDumpInitialize(
            &m_lastJitDiagnostic);
    }
    if (m_blockProfilingEnabled)
    {
        m_blockProfiles.reserve(
            std::min<size_t>(
                m_maximumBlockProfiles,
                kDefaultMaximumBlockProfiles));
    }
}

VuBlockProfilingSnapshot
VuRecompilerBackend::
    blockProfilingSnapshotWhileExecutionQuiescent() const
{
    VuBlockProfilingSnapshot snapshot{
        .enabled = m_blockProfilingEnabled,
        .maximumRecords =
            static_cast<uint64_t>(
                m_maximumBlockProfiles),
        .droppedRecords =
            m_droppedBlockProfiles,
    };
    snapshot.blocks.reserve(m_blockProfiles.size());
    for (const auto &sourcePointer : m_blockProfiles)
    {
        if (!sourcePointer)
            continue;
        const VuBlockProfileRecord &source =
            *sourcePointer;
        VuBlockProfileSnapshot block{
            .key = source.key,
            .compilationIdentity =
                source.compilationIdentity,
            .nativeAddress = source.nativeAddress,
            .nativeBytes = source.nativeBytes,
            .firstPc = source.firstPc,
            .lastPc = source.lastPc,
            .blockPairs = source.blockPairs,
            .fixedCycles = source.fixedCycles,
            .blockExit = source.blockExit,
            .resident =
                !source.codeLifetime.expired(),
            .executions = source.executions,
            .guestPairs = source.guestPairs,
            .fullBudgetEntries =
                source.fullBudgetEntries,
            .boundedEntries =
                source.boundedEntries,
            .linkedEdges = source.linkedEdges,
            .helperBarriers =
                source.helperBarriers,
            .exitReasons = source.exitReasons,
            .opcodeOperations =
                source.opcodeOperations,
            .jitRegistrations =
                source.jitRegistrations,
        };
        snapshot.blocks.push_back(std::move(block));
    }
    return snapshot;
}

void VuRecompilerBackend::
    resetBlockProfilingCountersWhileExecutionQuiescent()
{
    for (const auto &profile : m_blockProfiles)
    {
        if (!profile)
            continue;
        profile->executions = 0u;
        profile->guestPairs = 0u;
        profile->fullBudgetEntries = 0u;
        profile->boundedEntries = 0u;
        profile->linkedEdges = 0u;
        profile->helperBarriers = 0u;
        profile->exitReasons.fill(0u);
    }
}

std::string_view VuRecompilerBackend::name() const
{
    return built()
               ? "x86-64-recompiler"
               : "recompiler-unavailable";
}

bool VuRecompilerBackend::built()
{
#if PS2X_HAS_VU_X64_RECOMPILER
    return true;
#else
    return false;
#endif
}

bool VuRecompilerBackend::supported()
{
#if PS2X_HAS_VU_X64_RECOMPILER
    return VuExecutableMemory::supported() &&
           (hostFeatures() & 1u) != 0u;
#else
    return false;
#endif
}

uint64_t VuRecompilerBackend::hostFeatures()
{
#if PS2X_HAS_VU_X64_RECOMPILER
    static const uint64_t features = []()
    {
        const Xbyak::util::Cpu cpu;
        uint64_t value = 0u;
        if (cpu.has(Xbyak::util::Cpu::tSSE2))
            value |= 1u << 0u;
        if (cpu.has(Xbyak::util::Cpu::tSSE41))
            value |= 1u << 1u;
        if (cpu.has(Xbyak::util::Cpu::tAVX))
            value |= 1u << 2u;
        if (cpu.has(Xbyak::util::Cpu::tAVX2))
            value |= 1u << 3u;
        if (cpu.has(Xbyak::util::Cpu::tFMA))
            value |= kHostFeatureFma;
        return value;
    }();
    return features;
#else
    return 0u;
#endif
}

bool VuRecompilerBackend::canInlineUpperOpcode(
    VuIrOpcode opcode, uint64_t hostFeatures)
{
    switch (opcode)
    {
    case VuIrOpcode::Nop:
    case VuIrOpcode::UpperAddBc:
    case VuIrOpcode::UpperSubBc:
    case VuIrOpcode::UpperMaxBc:
    case VuIrOpcode::UpperMiniBc:
    case VuIrOpcode::UpperMulBc:
    case VuIrOpcode::UpperMulQ:
    case VuIrOpcode::UpperMaxI:
    case VuIrOpcode::UpperMulI:
    case VuIrOpcode::UpperMiniI:
    case VuIrOpcode::UpperAddQ:
    case VuIrOpcode::UpperAddI:
    case VuIrOpcode::UpperSubQ:
    case VuIrOpcode::UpperSubI:
    case VuIrOpcode::UpperAdd:
    case VuIrOpcode::UpperMul:
    case VuIrOpcode::UpperMax:
    case VuIrOpcode::UpperSub:
    case VuIrOpcode::UpperMini:
    case VuIrOpcode::UpperAddaBc:
    case VuIrOpcode::UpperSubaBc:
    case VuIrOpcode::UpperMulaBc:
    case VuIrOpcode::UpperMulaQ:
    case VuIrOpcode::UpperMulaI:
    case VuIrOpcode::UpperAddaQ:
    case VuIrOpcode::UpperAddaI:
    case VuIrOpcode::UpperSubaQ:
    case VuIrOpcode::UpperSubaI:
    case VuIrOpcode::UpperAdda:
    case VuIrOpcode::UpperMula:
    case VuIrOpcode::UpperSuba:
    case VuIrOpcode::UpperItof0:
    case VuIrOpcode::UpperItof4:
    case VuIrOpcode::UpperItof12:
    case VuIrOpcode::UpperItof15:
    case VuIrOpcode::UpperFtoi0:
    case VuIrOpcode::UpperFtoi4:
    case VuIrOpcode::UpperFtoi12:
    case VuIrOpcode::UpperFtoi15:
        return true;
    case VuIrOpcode::UpperMaddBc:
    case VuIrOpcode::UpperMsubBc:
    case VuIrOpcode::UpperMaddQ:
    case VuIrOpcode::UpperMaddI:
    case VuIrOpcode::UpperMsubQ:
    case VuIrOpcode::UpperMsubI:
    case VuIrOpcode::UpperMadd:
    case VuIrOpcode::UpperMsub:
    case VuIrOpcode::UpperMaddaBc:
    case VuIrOpcode::UpperMsubaBc:
    case VuIrOpcode::UpperMaddaQ:
    case VuIrOpcode::UpperMaddaI:
    case VuIrOpcode::UpperMsubaQ:
    case VuIrOpcode::UpperMsubaI:
    case VuIrOpcode::UpperMadda:
    case VuIrOpcode::UpperMsuba:
        return
            (hostFeatures &
             (kHostFeatureAvx | kHostFeatureFma)) ==
            (kHostFeatureAvx | kHostFeatureFma);
    default:
        return false;
    }
}

bool VuRecompilerBackend::canInlineLowerOpcode(
    VuIrOpcode opcode)
{
    switch (opcode)
    {
    case VuIrOpcode::Nop:
    case VuIrOpcode::LowerLq:
    case VuIrOpcode::LowerSq:
    case VuIrOpcode::LowerIlw:
    case VuIrOpcode::LowerIsw:
    case VuIrOpcode::LowerIaddiu:
    case VuIrOpcode::LowerIsubiu:
    case VuIrOpcode::LowerIadd:
    case VuIrOpcode::LowerIsub:
    case VuIrOpcode::LowerIaddi:
    case VuIrOpcode::LowerIand:
    case VuIrOpcode::LowerIor:
    case VuIrOpcode::LowerMove:
    case VuIrOpcode::LowerMr32:
    case VuIrOpcode::LowerLqi:
    case VuIrOpcode::LowerSqi:
    case VuIrOpcode::LowerMtir:
    case VuIrOpcode::LowerMfir:
    case VuIrOpcode::LowerIlwr:
    case VuIrOpcode::LowerIswr:
    case VuIrOpcode::LowerXtop:
    case VuIrOpcode::LowerXitop:
    case VuIrOpcode::LowerDiv:
    case VuIrOpcode::LowerB:
    case VuIrOpcode::LowerBal:
    case VuIrOpcode::LowerJr:
    case VuIrOpcode::LowerJalr:
    case VuIrOpcode::LowerIbeq:
    case VuIrOpcode::LowerIbne:
    case VuIrOpcode::LowerIbltz:
    case VuIrOpcode::LowerIbgtz:
    case VuIrOpcode::LowerIblez:
    case VuIrOpcode::LowerIbgez:
        return true;
    default:
        return false;
    }
}

VuRecompilerOpcodeDisposition
VuRecompilerBackend::opcodeDisposition(
    VuIrOpcode opcode, uint64_t hostFeatures)
{
    if (opcode == VuIrOpcode::Unsupported)
        return VuRecompilerOpcodeDisposition::
            InterpreterSideExit;
    if (canInlineUpperOpcode(opcode, hostFeatures) ||
        canInlineLowerOpcode(opcode))
    {
        return VuRecompilerOpcodeDisposition::NativeInline;
    }
    return VuRecompilerOpcodeDisposition::NativeHelper;
}

bool VuRecompilerBackend::programKey(
    const VuExecutionContext &context,
    VuCompilationMode mode,
    VuProgramKey &key,
    std::string *diagnostic)
{
    if (!supported())
    {
        return fail(
            built()
                ? "the x86-64 VU recompiler is unsupported on this host"
                : "the x86-64 VU recompiler was not built",
            diagnostic);
    }
    if (!context.memory || !context.code)
    {
        return fail(
            "native VU execution requires tracked code memory",
            diagnostic);
    }

    VuUnitId codeUnit;
    uint32_t expectedCodeSize = 0u;
    uint64_t generation = 0u;
    if (context.code == context.memory->getVU0Code())
    {
        codeUnit = VuUnitId::Vu0;
        expectedCodeSize = PS2_VU0_CODE_SIZE;
        generation =
            context.memory->getVU0CodeGeneration();
    }
    else if (context.code == context.memory->getVU1Code())
    {
        codeUnit = VuUnitId::Vu1;
        expectedCodeSize = PS2_VU1_CODE_SIZE;
        generation =
            context.memory->getVU1CodeGeneration();
    }
    else
    {
        return fail(
            "native VU code is not owned by the supplied memory",
            diagnostic);
    }
    if (codeUnit != m_unit.unitId())
    {
        return fail(
            "native VU code belongs to a different unit",
            diagnostic);
    }
    if (context.codeSize != expectedCodeSize)
    {
        return fail(
            "native VU code size does not match its unit",
            diagnostic);
    }
    const uint64_t contentIdentity =
        codeContentIdentity(context, generation);
    if (contentIdentity == 0u)
    {
        return fail(
            "native VU code content identity is unavailable",
            diagnostic);
    }

    key = {
        .unit = codeUnit,
        .memoryIdentity =
            reinterpret_cast<uintptr_t>(context.memory),
        .codeIdentity =
            reinterpret_cast<uintptr_t>(context.code),
        .codeSize = context.codeSize,
        .addressMask = context.codeSize - 1u,
        .entryPc = context.state.pc,
        .codeGeneration = generation,
        .codeContentIdentity = contentIdentity,
        .backend = VuNativeBackendMode::X86_64,
        .hostFeatures = hostFeatures(),
        .compilationMode = mode,
        .blockForm =
            m_blockBudgetGuardsEnabled
                ? VuIrBlockForm::Basic
                : VuIrBlockForm::LinearTrace,
    };
    return VuProgramCache::validKey(key, diagnostic);
}

uint64_t VuRecompilerBackend::codeContentIdentity(
    const VuExecutionContext &context,
    uint64_t generation)
{
    const uintptr_t memoryIdentity =
        reinterpret_cast<uintptr_t>(context.memory);
    const uintptr_t codeIdentity =
        reinterpret_cast<uintptr_t>(context.code);
    const bool sameScope =
        m_currentCodeMemoryIdentity == memoryIdentity &&
        m_currentCodeIdentity == codeIdentity &&
        m_currentCodeSize == context.codeSize;
    if (sameScope &&
        m_currentCodeContentIdentity != 0u &&
        m_currentCodeGeneration == generation)
    {
        return m_currentCodeContentIdentity;
    }
    if (!sameScope)
    {
        m_codeImages.clear();
        m_currentCodeContentIdentity = 0u;
    }

    const uint64_t hash =
        hashCodeImage(context.code, context.codeSize);
    uint64_t identity = 0u;
    for (const CodeImageIdentity &image : m_codeImages)
    {
        if (image.memoryIdentity == memoryIdentity &&
            image.codeIdentity == codeIdentity &&
            image.codeSize == context.codeSize &&
            image.hash == hash &&
            image.bytes.size() == context.codeSize &&
            std::memcmp(
                image.bytes.data(), context.code,
                context.codeSize) == 0)
        {
            identity = image.identity;
            ++m_diagnostics.codeImageReuses;
            break;
        }
    }

    if (identity == 0u)
    {
        if (m_codeImages.size() >=
            kMaximumRememberedCodeImages)
        {
            m_codeImages.erase(m_codeImages.begin());
            ++m_diagnostics.codeImageCatalogEvictions;
        }
        if (m_nextCodeContentIdentity == 0u)
        {
            m_codeImages.clear();
            m_unit.programCache().flush();
            m_nextCodeContentIdentity = 1u;
        }
        identity = m_nextCodeContentIdentity++;
        ++m_diagnostics.codeImageIdentities;
        m_codeImages.push_back({
            .memoryIdentity = memoryIdentity,
            .codeIdentity = codeIdentity,
            .codeSize = context.codeSize,
            .hash = hash,
            .identity = identity,
            .bytes = std::vector<uint8_t>(
                context.code,
                context.code + context.codeSize),
        });
    }

    m_currentCodeMemoryIdentity = memoryIdentity;
    m_currentCodeIdentity = codeIdentity;
    m_currentCodeSize = context.codeSize;
    m_currentCodeGeneration = generation;
    m_currentCodeContentIdentity = identity;
    return identity;
}

VuRunResult VuRecompilerBackend::run(
    VuExecutionContext &context,
    uint32_t maximumCycles)
{
    VuExecutionState &state = context.state;
    const bool activeBefore = state.active;
    const auto finish =
        [&](uint32_t executed, VuExitReason reason)
        {
            return VuRunResult{
                .requestedCycles = maximumCycles,
                .executedCycles = executed,
                .reason = reason,
                .activeBefore = activeBefore,
                .activeAfter = state.active,
                .completed = !state.active,
            };
        };

    if (!state.active)
        return finish(0u, VuExitReason::Inactive);
    if (!supported())
    {
        m_lastDiagnostic = built()
                               ? "x86-64 VU recompiler unsupported on this host"
                               : "x86-64 VU recompiler not built";
        ++m_diagnostics.faultExits;
        return finish(0u, VuExitReason::Fault);
    }
    uint32_t totalExecuted = 0u;
    VuUnit::ProgressTracker progress(
        m_unit, state,
        context.enableInstrumentation &&
            context.enableProgressAccounting,
        totalExecuted);
    const bool nativeInstrumentation =
        usesNativeInstrumentation(context);
    if (needsInterpreterInstrumentation(
            context, nativeInstrumentation))
    {
        ++m_diagnostics.interpreterInstrumentationFallbacks;
        VuRunResult result = runInterpreterFallback(
            context, maximumCycles);
        totalExecuted = result.executedCycles;
        progress.publish();
        return result;
    }
    if (maximumCycles == 0u)
    {
        ++m_diagnostics.cycleBudgetExits;
        return finish(0u, VuExitReason::CycleBudget);
    }

    const ScopedVuFloatMode vuFloatMode;
    VuProgramKey key;
    const VuCompilationMode compilationMode =
        nativeInstrumentation
            ? VuCompilationMode::Instrumented
            : VuCompilationMode::Normal;
    if (!programKey(
            context, compilationMode,
            key, &m_lastDiagnostic))
    {
        ++m_diagnostics.faultExits;
        return finish(totalExecuted, VuExitReason::Fault);
    }
    const NativeContextView nativeContextView{
        .state = &state,
        .data = context.data,
        .dataSize = context.dataSize,
    };
    m_instrumentedPairIndex = 0u;
    const auto runNativeBlocks =
        [&]() -> VuRunResult
    {
        while (state.active &&
               totalExecuted < maximumCycles)
        {
            const uint32_t remaining =
                maximumCycles - totalExecuted;
            key.entryPc = state.pc;
            // Long scheduler slices benefit from fewer native transitions.
            // Short slices use branch-bounded blocks so most entries can
            // retire under one exact full-block guard instead of entering
            // the per-pair precise tail.
            constexpr uint32_t
                kLinearTraceBudgetThreshold =
                    4u * kMaximumBlockPairs;
            key.blockForm =
                m_blockBudgetGuardsEnabled &&
                        remaining <
                            kLinearTraceBudgetThreshold
                    ? VuIrBlockForm::Basic
                    : VuIrBlockForm::LinearTrace;
            const bool useBlockBudgetGuards =
                m_blockBudgetGuardsEnabled &&
                key.blockForm == VuIrBlockForm::Basic;
            if ((key.entryPc & 7u) != 0u ||
                key.entryPc >= key.codeSize)
            {
                m_lastDiagnostic =
                    "native VU block entry PC is outside its code extent";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }

            VuProgramCache &cache =
                m_unit.programCache();
            const size_t dispatchIndex =
                key.entryPc / 8u;
            VuProgramHandle handle =
                m_dispatchHandles[dispatchIndex];
            const VuCompiledProgram *program =
                cache.resolveCurrent(handle, key);
            if (!program)
            {
                handle = lookupOrCompile(context, key);
                program = cache.resolve(handle);
                if (program)
                {
                    m_dispatchHandles[dispatchIndex] =
                        handle;
                }
            }
            if (!program)
            {
                if (m_lastDiagnostic.empty())
                {
                    m_lastDiagnostic =
                        "compiled VU program handle did not resolve";
                }
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }

            VuNativeFastBlockEntry entry = nullptr;
            const void *const address =
                program->nativeFastEntry();
            static_assert(
                sizeof(entry) == sizeof(address));
            std::memcpy(
                &entry, &address, sizeof(entry));

            const uint64_t helperPairsBefore =
                m_helperPairsIssued;
            m_chainRuntime.codeGeneration =
                key.codeGeneration;
            m_chainRuntime.slowLinkSource = nullptr;
            m_chainRuntime.slowLinkTargetPc = 0u;
            m_chainRuntime.linkedEdges = 0u;
            m_chainRuntime.preciseTailEntries = 0u;
            m_chainRuntime.preciseTailPairs = 0u;
            m_chainRuntime.preciseNonRetiredChecks = 0u;
            m_chainRuntime.returnBoundaryChecks = 0u;
            const VuNativeBlockResult native =
                VuNativeBlockResult::decode(
                    entry(
                        &context, remaining,
                        &nativeContextView));
            const uint64_t helperPairs =
                m_helperPairsIssued -
                helperPairsBefore;
            const VuNativeLinkState *const
                slowLinkSource =
                    m_chainRuntime.slowLinkSource;
            const uint32_t slowLinkTargetPc =
                m_chainRuntime.slowLinkTargetPc;
            const uint64_t linkedEdges =
                m_chainRuntime.linkedEdges;
            const uint64_t nativeBlocks =
                linkedEdges + 1u;
            const uint64_t preciseTailEntries =
                useBlockBudgetGuards
                    ? m_chainRuntime.
                          preciseTailEntries
                    : nativeBlocks;
            const uint64_t preciseTailPairs =
                useBlockBudgetGuards
                    ? m_chainRuntime.
                          preciseTailPairs
                    : native.executedCycles;
            const uint64_t
                preciseNonRetiredChecks =
                    m_chainRuntime.
                        preciseNonRetiredChecks;
            const uint64_t returnBoundaryChecks =
                m_chainRuntime.
                    returnBoundaryChecks;
            const uint64_t blockBoundaryGuards =
                linkedEdges +
                (native.exit ==
                         VuNativeBlockExit::BlockComplete
                     ? 1u
                     : 0u);
            const bool nativeBudgetExit =
                native.exit ==
                    VuNativeBlockExit::CycleBudget ||
                (native.exit ==
                     VuNativeBlockExit::BlockComplete &&
                 native.executedCycles == remaining);
            ++m_diagnostics.nativeEntries;
            if (linkedEdges >
                native.executedCycles)
            {
                m_lastDiagnostic =
                    "native VU chain reported more links than retired pairs";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }
            if (preciseTailEntries > nativeBlocks)
            {
                m_lastDiagnostic =
                    "native VU chain reported too many precise entries";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }
            if (preciseTailPairs >
                native.executedCycles)
            {
                m_lastDiagnostic =
                    "native VU chain reported too many precise pairs";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }
            if (slowLinkSource &&
                native.exit !=
                    VuNativeBlockExit::BlockComplete)
            {
                m_lastDiagnostic =
                    "native VU chain retained a slow link on a terminal exit";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }
            m_diagnostics.nativeBlocks +=
                nativeBlocks;
            m_diagnostics.linkedEdges +=
                linkedEdges;
            m_diagnostics.fullBlockGuards +=
                nativeBlocks -
                preciseTailEntries;
            m_diagnostics.preciseTailEntries +=
                preciseTailEntries;
            m_diagnostics.fullGuardPairs +=
                native.executedCycles -
                preciseTailPairs;
            m_diagnostics.preciseTailPairs +=
                preciseTailPairs;
            if (useBlockBudgetGuards)
            {
                m_diagnostics.
                    budgetGuardComparisons +=
                        nativeBlocks +
                        preciseTailPairs +
                        preciseNonRetiredChecks +
                        returnBoundaryChecks +
                        (native.exit ==
                                 VuNativeBlockExit::
                                     CycleBudget &&
                                 returnBoundaryChecks ==
                                     0u
                             ? 1u
                             : 0u);
            }
            else
            {
                m_diagnostics.
                    budgetGuardComparisons +=
                        native.executedCycles +
                        preciseNonRetiredChecks +
                        blockBoundaryGuards +
                        (native.exit ==
                                 VuNativeBlockExit::
                                     CycleBudget
                             ? 1u
                             : 0u) +
                        (native.exit ==
                                 VuNativeBlockExit::
                                     UnsupportedInstruction
                             ? 1u
                             : 0u);
            }
            if (nativeBudgetExit)
            {
                ++m_diagnostics.nativeBudgetExits;
            }
            m_diagnostics.nativeExitReasons[
                static_cast<size_t>(
                    VuNativeBlockExit::
                        BlockComplete)] +=
                            linkedEdges;
            size_t nativeExitIndex =
                static_cast<size_t>(native.exit);
            if (nativeExitIndex >=
                kVuNativeBlockExitCount)
            {
                nativeExitIndex =
                    static_cast<size_t>(
                        VuNativeBlockExit::Fault);
            }
            ++m_diagnostics.
                nativeExitReasons[nativeExitIndex];
            m_diagnostics.blockCompletes +=
                linkedEdges;
            if (nativeInstrumentation)
            {
                ++m_diagnostics.instrumentedNativeEntries;
                m_diagnostics.instrumentedNativeBlocks +=
                    nativeBlocks;
            }
            if (native.executedCycles > remaining)
            {
                m_lastDiagnostic =
                    "native VU block exceeded its cycle budget";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }
            if (helperPairs > native.executedCycles)
            {
                m_lastDiagnostic =
                    "native VU block reported fewer cycles than its helpers";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            }

            totalExecuted += native.executedCycles;
            state.issuedCycles += native.executedCycles;
            m_diagnostics.nativePairs +=
                native.executedCycles;
            if (nativeInstrumentation)
            {
                m_diagnostics.instrumentedNativePairs +=
                    native.executedCycles;
            }
            m_diagnostics.helperPairs += helperPairs;
            m_diagnostics.inlinePairs +=
                native.executedCycles - helperPairs;
            progress.publish();

            const uint64_t currentGeneration =
                key.unit == VuUnitId::Vu0
                    ? context.memory->
                          getVU0CodeGeneration()
                    : context.memory->
                          getVU1CodeGeneration();
            if (currentGeneration !=
                key.codeGeneration)
            {
                VuProgramKey currentKey = key;
                currentKey.codeGeneration =
                    currentGeneration;
                (void)m_unit.programCache().lookup(
                    currentKey);
                ++m_diagnostics.codeInvalidationExits;
                return finish(
                    totalExecuted,
                    VuExitReason::CodeInvalidated);
            }

            switch (native.exit)
            {
            case VuNativeBlockExit::BlockComplete:
                ++m_diagnostics.blockCompletes;
                if (native.executedCycles == 0u)
                {
                    m_lastDiagnostic =
                        "native VU block completed without progress";
                    ++m_diagnostics.faultExits;
                    return finish(
                        totalExecuted,
                        VuExitReason::Fault);
                }
                if (slowLinkSource)
                {
                    ++m_diagnostics.slowLinkExits;
                    if (slowLinkTargetPc != state.pc ||
                        (slowLinkTargetPc & 7u) != 0u ||
                        slowLinkTargetPc >= key.codeSize)
                    {
                        m_lastDiagnostic =
                            "native VU slow link target is invalid";
                        ++m_diagnostics.faultExits;
                        return finish(
                            totalExecuted,
                            VuExitReason::Fault);
                    }

                    VuProgramKey targetKey = key;
                    targetKey.entryPc =
                        slowLinkTargetPc;
                    VuProgramHandle targetHandle =
                        lookupOrCompile(
                            context, targetKey);
                    if (!targetHandle.valid())
                    {
                        if (m_lastDiagnostic.empty())
                        {
                            m_lastDiagnostic =
                                "native VU slow link target did not compile";
                        }
                        ++m_diagnostics.faultExits;
                        return finish(
                            totalExecuted,
                            VuExitReason::Fault);
                    }
                    m_dispatchHandles[
                        slowLinkTargetPc / 8u] =
                            targetHandle;
                    const VuProgramLinkResult linked =
                        cache.populateLink(
                            slowLinkSource,
                            targetHandle, targetKey);
                    switch (linked)
                    {
                    case VuProgramLinkResult::Linked:
                        ++m_diagnostics.resolvedLinks;
                        break;
                    case VuProgramLinkResult::
                        Incompatible:
                        ++m_diagnostics.
                            incompatibleLinkExits;
                        break;
                    case VuProgramLinkResult::
                        SourceUnavailable:
                    case VuProgramLinkResult::
                        TargetUnavailable:
                        ++m_diagnostics.
                            abandonedLinkResolutions;
                        break;
                    }
                }
                continue;
            case VuNativeBlockExit::CycleBudget:
                ++m_diagnostics.cycleBudgetExits;
                return finish(
                    totalExecuted,
                    VuExitReason::CycleBudget);
            case VuNativeBlockExit::XgkickBoundary:
                ++m_diagnostics.xgkickExits;
                return finish(
                    totalExecuted,
                    VuExitReason::XgkickBoundary);
            case VuNativeBlockExit::
                UnsupportedInstruction:
                ++m_diagnostics.unsupportedExits;
                return finish(
                    totalExecuted,
                    VuExitReason::
                        UnsupportedInstruction);
            case VuNativeBlockExit::Fault:
                m_lastDiagnostic =
                    "native VU pair helper reported a fault";
                ++m_diagnostics.faultExits;
                return finish(
                    totalExecuted, VuExitReason::Fault);
            default:
                return finish(
                    totalExecuted,
                    publicExit(native.exit));
            }
        }

        ++m_diagnostics.cycleBudgetExits;
        return finish(
            totalExecuted, VuExitReason::CycleBudget);
    };
    return runNativeBlocks();
}

VuRunResult VuRecompilerBackend::runInterpreterFallback(
    VuExecutionContext &context,
    uint32_t maximumCycles)
{
    VuExecutionContext fallbackContext = context;
    fallbackContext.enableProgressAccounting = false;
    VuRunResult result =
        m_semantics.run(fallbackContext, maximumCycles);
    m_diagnostics.interpreterFallbackPairs +=
        result.executedCycles;
    return result;
}

uint32_t VuRecompilerBackend::executePair(
    VuExecutionContext &context, uint32_t expectedPc,
    uint32_t lowerWord, uint32_t upperWord) noexcept
{
    VuExecutionState &state = context.state;
    if (!state.active)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::Inactive);
    }
    if (!context.code || context.codeSize < 8u ||
        expectedPc != state.pc ||
        expectedPc > context.codeSize - 8u)
    {
        if (!context.code ||
            state.pc > context.codeSize - 8u)
        {
            state.active = false;
            return static_cast<uint32_t>(
                VuNativeBlockExit::CodeBounds);
        }
        return static_cast<uint32_t>(
            VuNativeBlockExit::Fault);
    }

    const VuIrInstructionPair pair =
        decodeVuIrInstructionPair(
            expectedPc, lowerWord, upperWord);
    if (vuIrHasPairFlag(
            pair, VuIrPairUnsupported))
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::UnsupportedInstruction);
    }

    VuExecutionState *const previousState =
        m_semantics.m_state;
    const uint32_t previousCodeAddressMask =
        m_semantics.m_codeAddressMask;
    m_semantics.m_state = &state;
    m_semantics.m_codeAddressMask =
        context.codeSize != 0u
            ? context.codeSize - 1u
            : 0u;
    struct StateBindingGuard
    {
        VuExecutionState *&bound;
        VuExecutionState *previous;
        uint32_t &boundCodeAddressMask;
        uint32_t previousCodeAddressMask;

        ~StateBindingGuard()
        {
            bound = previous;
            boundCodeAddressMask =
                previousCodeAddressMask;
        }
    };
    const StateBindingGuard binding{
        m_semantics.m_state, previousState,
        m_semantics.m_codeAddressMask,
        previousCodeAddressMask};

    try
    {
        m_semantics.advanceQPipeline();
        m_semantics.advancePPipeline();
        m_semantics.advanceFmacFlagPipeline();
        if (vuIrHasOpFlag(pair.lower, VuIrOpQBarrier))
            m_semantics.flushQPipeline();
        if (vuIrHasOpFlag(pair.lower, VuIrOpPBarrier))
            m_semantics.flushPPipeline();

        if (state.viBackupCycles != 0u)
            --state.viBackupCycles;

        switch (pair.order)
        {
        case VuIrPairOrder::UpperThenLoi:
            m_semantics.execUpper(pair.upperWord);
            std::memcpy(
                &state.i, &pair.lowerWord,
                sizeof(pair.lowerWord));
            break;
        case VuIrPairOrder::LowerThenUpper:
            m_semantics.execLower(
                pair.lowerWord, context.data,
                context.dataSize, context.sideEffects,
                context.memory, pair.upperWord);
            m_semantics.execUpper(pair.upperWord);
            break;
        case VuIrPairOrder::UpperThenLower:
            m_semantics.execUpper(pair.upperWord);
            if (pair.lower.opcode != VuIrOpcode::Nop)
            {
                m_semantics.execLower(
                    pair.lowerWord, context.data,
                    context.dataSize, context.sideEffects,
                    context.memory, pair.upperWord);
            }
            break;
        }

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;

        if (state.pipeline.xgkick.active)
        {
            m_semantics.advanceXgkick(
                context.data, context.dataSize,
                context.sideEffects, context.memory,
                1u, false);
        }

        uint32_t nextPc = state.pc + 8u;
        if (nextPc >= context.codeSize)
            nextPc = 0u;
        state.pc = nextPc;

        if (state.branchPending)
        {
            if (state.branchDelay == 0u)
            {
                state.pc =
                    state.branchTarget &
                    m_semantics.m_codeAddressMask;
                state.branchPending = false;
            }
            else
            {
                --state.branchDelay;
            }
        }

        VuNativeBlockExit exit =
            VuNativeBlockExit::BlockComplete;
        if (state.ebit)
        {
            if (state.pipeline.xgkick.active)
            {
                m_semantics.advanceXgkick(
                    context.data, context.dataSize,
                    context.sideEffects, context.memory,
                    0u, true);
            }
            m_semantics.flushQPipeline();
            m_semantics.flushPPipeline();
            m_semantics.flushFmacFlagPipeline();
            state.viBackupCycles = 0u;
            state.active = false;
            exit = VuNativeBlockExit::ProgramEnded;
        }
        else
        {
            if (vuIrHasPairFlag(pair, VuIrPairEnd))
                state.ebit = true;
            if (vuIrHasPairFlag(
                    pair, VuIrPairXgkick))
            {
                exit =
                    VuNativeBlockExit::XgkickBoundary;
            }
        }

        ++m_helperPairsIssued;
        return kNativePairExecuted |
               static_cast<uint32_t>(exit);
    }
    catch (...)
    {
        ++m_helperPairsIssued;
        return kNativePairExecuted |
               static_cast<uint32_t>(
                   VuNativeBlockExit::Fault);
    }
}

uint32_t VuRecompilerBackend::executePairThunk(
    VuRecompilerBackend *backend,
    VuExecutionContext *context,
    uint32_t expectedPc,
    uint64_t instructionWords) noexcept
{
    if (!backend || !context)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::Fault);
    }
    return backend->executePair(
        *context, expectedPc,
        static_cast<uint32_t>(instructionWords),
        static_cast<uint32_t>(
            instructionWords >> 32u));
}

uint32_t VuRecompilerBackend::executeInstrumentedPair(
    VuExecutionContext &context, uint32_t expectedPc,
    uint32_t lowerWord, uint32_t upperWord) noexcept
{
    VuExecutionState &state = context.state;
    if (!state.active ||
        !context.code ||
        context.codeSize < 8u ||
        expectedPc != state.pc ||
        expectedPc > context.codeSize - 8u)
    {
        return executePair(
            context, expectedPc,
            lowerWord, upperWord);
    }

    try
    {
        if (context.enableInstrumentation &&
            m_unit.m_instructionObserverEnabled.load(
                std::memory_order_relaxed) &&
            m_unit.m_instructionObserver)
        {
            m_unit.m_instructionObserver(
                m_instrumentedPairIndex,
                expectedPc, lowerWord, upperWord,
                state);
        }
        ++m_instrumentedPairIndex;
    }
    catch (...)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::Fault);
    }

    return executePair(
        context, expectedPc,
        lowerWord, upperWord);
}

uint32_t VuRecompilerBackend::executeInstrumentedPairThunk(
    VuRecompilerBackend *backend,
    VuExecutionContext *context,
    uint32_t expectedPc,
    uint64_t instructionWords) noexcept
{
    if (!backend || !context)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::Fault);
    }
    return backend->executeInstrumentedPair(
        *context, expectedPc,
        static_cast<uint32_t>(instructionWords),
        static_cast<uint32_t>(
            instructionWords >> 32u));
}

uint32_t VuRecompilerBackend::advanceXgkick(
    VuExecutionContext &context) noexcept
{
    ++m_diagnostics.xgkickAdvanceHelperCalls;
    VuExecutionState &state = context.state;
    if (!state.pipeline.xgkick.active)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::BlockComplete);
    }

    const uint64_t generationBefore =
        m_unit.unitId() == VuUnitId::Vu0
            ? context.memory->getVU0CodeGeneration()
            : context.memory->getVU1CodeGeneration();
    VuExecutionState *const previousState =
        m_semantics.m_state;
    m_semantics.m_state = &state;
    struct StateBindingGuard
    {
        VuExecutionState *&bound;
        VuExecutionState *previous;

        ~StateBindingGuard()
        {
            bound = previous;
        }
    };
    const StateBindingGuard binding{
        m_semantics.m_state, previousState};

    try
    {
        m_semantics.advanceXgkick(
            context.data, context.dataSize,
            context.sideEffects, context.memory,
            1u, false);
    }
    catch (...)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::Fault);
    }

    const uint64_t generationAfter =
        m_unit.unitId() == VuUnitId::Vu0
            ? context.memory->getVU0CodeGeneration()
            : context.memory->getVU1CodeGeneration();
    return static_cast<uint32_t>(
        generationAfter == generationBefore
            ? VuNativeBlockExit::BlockComplete
            : VuNativeBlockExit::CodeInvalidated);
}

uint32_t VuRecompilerBackend::advanceXgkickThunk(
    VuRecompilerBackend *backend,
    VuExecutionContext *context) noexcept
{
    if (!backend || !context || !context->memory)
    {
        return static_cast<uint32_t>(
            VuNativeBlockExit::Fault);
    }
    return backend->advanceXgkick(*context);
}

VuProgramHandle VuRecompilerBackend::lookupOrCompile(
    VuExecutionContext &context,
    const VuProgramKey &key)
{
    VuProgramCache &cache = m_unit.programCache();
    VuProgramHandle handle = cache.lookup(key);
    if (handle.valid())
    {
        if (const VuCompiledProgram *const program =
                cache.resolve(handle))
        {
            ensureJitRegistration(*program);
        }
        return handle;
    }

    VuCompiledProgram program;
    if (!compile(
            context, key, program,
            m_lastDiagnostic))
    {
        return {};
    }
    handle = cache.insert(
        std::move(program), &m_lastDiagnostic);
    if (const VuCompiledProgram *const inserted =
            cache.resolve(handle))
    {
        if (inserted->blockProfile &&
            !inserted->blockProfile->archived)
        {
            inserted->blockProfile->archived = true;
            m_blockProfiles.push_back(
                inserted->blockProfile);
        }
        ensureJitRegistration(*inserted);
    }
    return handle;
}

void VuRecompilerBackend::ensureJitRegistration(
    const VuCompiledProgram &program)
{
    if (!m_perfJitDumpEnabled ||
        program.jitCodeIndex != 0u)
    {
        return;
    }
    const uint8_t *const code =
        program.nativeCode.executableData();
    const size_t sizeBytes =
        program.nativeCode.usedSize();
    if (!code || sizeBytes == 0u)
    {
        ++m_diagnostics.jitDumpFailures;
        m_lastJitDiagnostic =
            "compiled VU program has no code to register";
        return;
    }

    const std::string symbol =
        vuPerfJitSymbolName(
            program.key, program.block,
            program.compilationIdentity);
    uint64_t codeIndex = 0u;
    std::string diagnostic;
    if (!ps2PerfJitDumpRegisterCode(
            code, sizeBytes, symbol,
            &codeIndex, &diagnostic))
    {
        ++m_diagnostics.jitDumpFailures;
        m_lastJitDiagnostic = std::move(diagnostic);
        return;
    }

    program.jitCodeIndex = codeIndex;
    ++m_diagnostics.jitDumpRegistrations;
    m_lastJitDiagnostic.clear();
    if (program.blockProfile)
    {
        program.blockProfile->
            jitRegistrations.push_back({
                .generation =
                    program.key.codeGeneration,
                .codeIndex = codeIndex,
            });
    }
}

bool VuRecompilerBackend::compile(
    VuExecutionContext &context,
    const VuProgramKey &key,
    VuCompiledProgram &program,
    std::string &diagnostic)
{
#if PS2X_HAS_VU_X64_RECOMPILER
    const Clock::time_point start = Clock::now();
    VuIrBlock block = decodeVuIrBlock(
        context.code, context.codeSize,
        key.entryPc, kMaximumBlockPairs,
        key.blockForm);
    VuIrVerificationError verification;
    if (!verifyVuIrBlock(block, &verification))
    {
        diagnostic =
            "cannot compile malformed VU IR at PC " +
            std::to_string(verification.pc) +
            ": " + verification.message;
        return false;
    }

    try
    {
        const uint64_t compilationIdentity =
            m_blockProfilingEnabled ||
                    m_perfJitDumpEnabled
                ? nextCompilationIdentity()
                : 0u;
        std::shared_ptr<VuBlockProfileRecord>
            blockProfile;
        std::shared_ptr<void> blockProfileLifetime;
        if (m_blockProfilingEnabled &&
            m_blockProfiles.size() <
                m_maximumBlockProfiles)
        {
            blockProfile =
                std::make_shared<
                    VuBlockProfileRecord>();
            blockProfileLifetime =
                std::make_shared<uint8_t>(0u);
            blockProfile->key = key;
            blockProfile->compilationIdentity =
                compilationIdentity;
            blockProfile->blockPairs =
                static_cast<uint32_t>(
                    block.pairs.size());
            blockProfile->blockExit = block.exit;
            blockProfile->firstPc =
                block.pairs.empty()
                    ? key.entryPc
                    : block.pairs.front().pc;
            blockProfile->lastPc =
                block.pairs.empty()
                    ? key.entryPc
                    : block.pairs.back().pc;
            for (const VuIrInstructionPair &pair :
                 block.pairs)
            {
                blockProfile->fixedCycles +=
                    pair.cycles;
                ++blockProfile->opcodeOperations[
                    static_cast<size_t>(
                        pair.upper.opcode)];
                ++blockProfile->opcodeOperations[
                    static_cast<size_t>(
                        pair.lower.opcode)];
            }
            blockProfile->codeLifetime =
                blockProfileLifetime;
        }

        std::shared_ptr<VuNativeLinkState>
            outgoingLinks =
                makeNativeLinkState(
                    block, key, this);
        const void *const pairHelper =
            key.compilationMode ==
                    VuCompilationMode::Instrumented
                ? reinterpret_cast<const void *>(
                      &VuRecompilerBackend::
                          executeInstrumentedPairThunk)
                : reinterpret_cast<const void *>(
                      &VuRecompilerBackend::
                          executePairThunk);
        X64VuBlockEmitter emitter(
            block, this, pairHelper,
            reinterpret_cast<const void *>(
                &VuRecompilerBackend::
                    advanceXgkickThunk),
            key.hostFeatures,
            key.compilationMode,
            outgoingLinks.get(),
            &m_chainRuntime,
            blockProfile.get(),
            m_blockLinkingEnabled,
            m_blockBudgetGuardsEnabled &&
                key.blockForm ==
                    VuIrBlockForm::Basic);
        if (emitter.size() == 0u)
        {
            diagnostic =
                "x86-64 VU emitter produced no code";
            return false;
        }

        VuExecutableMemory nativeCode =
            VuExecutableMemory::allocate(
                emitter.size(), &diagnostic);
        if (nativeCode.empty() ||
            !nativeCode.write(
                0u, emitter.data(), emitter.size(),
                &diagnostic) ||
            !nativeCode.finalize(
                emitter.size(), &diagnostic))
        {
            return false;
        }

        if (blockProfile)
        {
            blockProfile->nativeAddress =
                static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(
                        nativeCode.executableData()));
            blockProfile->nativeBytes =
                static_cast<uint64_t>(
                    nativeCode.usedSize());
        }
        const uint64_t nanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    Clock::now() - start)
                    .count());
        if (m_blockProfilingEnabled &&
            !blockProfile)
        {
            ++m_droppedBlockProfiles;
        }
        program = {
            .key = key,
            .block = std::move(block),
            .nativeCode = std::move(nativeCode),
            .entryOffset = 0u,
            .fastEntryOffset =
                emitter.fastEntryOffset(),
            .linkedEntryOffset =
                emitter.linkedEntryOffset(),
            .compilationNanoseconds = nanoseconds,
            .compilationIdentity =
                compilationIdentity,
            .outgoingLinks =
                std::move(outgoingLinks),
            .blockProfile =
                std::move(blockProfile),
            .blockProfileLifetime =
                std::move(blockProfileLifetime),
        };
        diagnostic.clear();
        return true;
    }
    catch (const std::exception &error)
    {
        diagnostic =
            "x86-64 VU emission failed: " +
            std::string(error.what());
        return false;
    }
    catch (...)
    {
        diagnostic =
            "x86-64 VU emission failed with an unknown error";
        return false;
    }
#else
    (void)context;
    (void)key;
    (void)program;
    diagnostic =
        "the x86-64 VU recompiler was not built";
    return false;
#endif
}

bool VuRecompilerBackend::usesNativeInstrumentation(
    const VuExecutionContext &context) const
{
    return
        context.enableInstrumentation &&
        context.enableNativeInstrumentation &&
        m_unit.m_instructionObserverEnabled.load(
            std::memory_order_relaxed) &&
        static_cast<bool>(m_unit.m_instructionObserver);
}

bool VuRecompilerBackend::needsInterpreterInstrumentation(
    const VuExecutionContext &context,
    bool nativeInstrumentation) const
{
    if (!context.enableInstrumentation)
        return false;
    if (!nativeInstrumentation &&
        m_unit.m_instructionObserverEnabled.load(
            std::memory_order_relaxed) &&
        m_unit.m_instructionObserver)
    {
        return true;
    }
    if (!context.memory)
        return false;
    return context.memory->isVif1DmaTraceActive() ||
           context.memory->isVu1WorkloadProfileEnabled() ||
           m_unit.m_workloadProfileInvocationActive;
}

VuExitReason VuRecompilerBackend::publicExit(
    VuNativeBlockExit exit)
{
    switch (exit)
    {
    case VuNativeBlockExit::CycleBudget:
        return VuExitReason::CycleBudget;
    case VuNativeBlockExit::Inactive:
        return VuExitReason::Inactive;
    case VuNativeBlockExit::ProgramEnded:
        return VuExitReason::ProgramEnded;
    case VuNativeBlockExit::CodeBounds:
        return VuExitReason::CodeBounds;
    case VuNativeBlockExit::XgkickBoundary:
        return VuExitReason::XgkickBoundary;
    case VuNativeBlockExit::DebugObserver:
        return VuExitReason::DebugObserver;
    case VuNativeBlockExit::CodeInvalidated:
        return VuExitReason::CodeInvalidated;
    case VuNativeBlockExit::UnsupportedInstruction:
        return VuExitReason::UnsupportedInstruction;
    case VuNativeBlockExit::BlockComplete:
    case VuNativeBlockExit::Fault:
        return VuExitReason::Fault;
    }
    return VuExitReason::Fault;
}
