#include "runtime/ps2_vu_recompiler.h"

#include "runtime/ps2_memory.h"
#include "ps2_vu_float_mode.h"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
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

namespace
{
    using Clock = std::chrono::steady_clock;
    constexpr uint32_t kNativePairExecuted = 1u << 31u;
    constexpr uint32_t kNativePairExitMask = 0xffu;

    bool fail(
        std::string message, std::string *diagnostic)
    {
        if (diagnostic)
            *diagnostic = std::move(message);
        return false;
    }

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

    struct NativeContextView
    {
        VuExecutionState *state = nullptr;
        uint8_t *data = nullptr;
        uint32_t dataSize = 0u;
    };

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
            uint64_t hostFeatures)
            : m_buffer(kEmitterCapacity),
              m_code(m_buffer.size(), m_buffer.data()),
              m_hostFeatures(hostFeatures)
        {
            emit(
                block, backend, pairHelper,
                xgkickHelper);
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

    private:
        enum class KnownFmacSlot : uint8_t
        {
            Dynamic,
            Inactive,
            Active,
        };

        std::vector<uint8_t> m_buffer;
        Xbyak::CodeGenerator m_code;
        uint64_t m_hostFeatures = 0u;
        bool m_usesAvx = false;

#if defined(_WIN32)
        static constexpr int kStackBytes = 96;
        static constexpr int kSavedMxcsrOffset = 32;
        static constexpr int kVuMxcsrOffset = 36;
        static constexpr int kResultOffset = 48;
        static constexpr int kContextViewOffset = 64;
#else
        static constexpr int kStackBytes = 64;
        static constexpr int kSavedMxcsrOffset = 0;
        static constexpr int kVuMxcsrOffset = 4;
        static constexpr int kResultOffset = 16;
        static constexpr int kContextViewOffset = 32;
#endif

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
            switch (operation.opcode)
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
                    (m_hostFeatures &
                     (kHostFeatureAvx |
                      kHostFeatureFma)) ==
                    (kHostFeatureAvx |
                     kHostFeatureFma);
            default:
                return false;
            }
        }

        static bool canInlineLower(
            const VuIrOperation &operation)
        {
            switch (operation.opcode)
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
            size_t architecturalOffset)
        {
            using DelayedQ = VuPipelineState::DelayedQ;
            using namespace Xbyak;

            Label done;
            Label commit;
            const int base =
                pipelineOffset(delayedOffset);
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
                        DelayedQ, cyclesRemaining)]);
            m_code.test(m_code.eax, m_code.eax);
            m_code.jz(commit);
            m_code.dec(m_code.eax);
            m_code.mov(
                m_code.dword[
                    m_code.rbx + base +
                    offsetof(
                        DelayedQ, cyclesRemaining)],
                m_code.eax);
            m_code.jnz(done);

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

        void emitAdvancePipelines(
            KnownFmacSlot knownFmacSlot)
        {
            emitAdvanceDelayedScalar(
                offsetof(
                    VuPipelineState, delayedQ),
                offsetof(VuExecutionState, q));
            emitAdvanceDelayedScalar(
                offsetof(
                    VuPipelineState, delayedP),
                offsetof(VuExecutionState, p));
            emitAdvanceFmacFlags(knownFmacSlot);

            Xbyak::Label noBackup;
            m_code.cmp(
                m_code.byte[
                    m_code.rbx +
                    stateOffset(
                        offsetof(
                            VuExecutionState,
                            viBackupCycles))],
                0u);
            m_code.je(noBackup);
            m_code.dec(
                m_code.byte[
                    m_code.rbx +
                    stateOffset(
                        offsetof(
                            VuExecutionState,
                            viBackupCycles))]);
            m_code.L(noBackup);
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
                target & 0x3fffu);
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
            m_code.and_(m_code.eax, 0x3fffu);
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
                    0x3fffu;
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
                    0x3fffu;
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
                    0x3fffu;
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
            KnownFmacSlot knownFmacSlot)
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
            emitAdvancePipelines(knownFmacSlot);
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
            m_code.mov(m_code.rcx, m_code.r15);
            m_code.mov(m_code.rdx, m_code.r12);
#else
            m_code.mov(m_code.rdi, m_code.r15);
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
                m_code.and_(m_code.eax, 0x3fffu);
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
            m_code.test(m_code.r11d, m_code.r11d);
            m_code.jnz(
                dynamicExit, m_code.T_NEAR);
        }

        void emitHelperPair(
            const VuIrInstructionPair &pair,
            const void *pairHelper,
            Xbyak::Label &dynamicExit)
        {
            if (m_usesAvx)
                m_code.vzeroupper();
            const uint64_t words =
                static_cast<uint64_t>(pair.lowerWord) |
                (static_cast<uint64_t>(
                     pair.upperWord)
                 << 32u);
#if defined(_WIN32)
            m_code.mov(m_code.rcx, m_code.r15);
            m_code.mov(m_code.rdx, m_code.r12);
            m_code.mov(m_code.r8d, pair.pc);
            m_code.mov(m_code.r9, words);
#else
            m_code.mov(m_code.rdi, m_code.r15);
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
            m_code.jz(pairNotExecuted);
            m_code.inc(m_code.r14d);
            m_code.L(pairNotExecuted);
            m_code.and_(
                m_code.r11d,
                kNativePairExitMask);
            m_code.test(m_code.r11d, m_code.r11d);
            m_code.jnz(
                dynamicExit, m_code.T_NEAR);
        }

        void emit(
            const VuIrBlock &block,
            VuRecompilerBackend *backend,
            const void *pairHelper,
            const void *xgkickHelper)
        {
            using namespace Xbyak;

            Label cycleBudgetExit;
            Label controlFlowExit;
            Label dynamicExit;
            Label packResult;
            Label epilogue;

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
            m_code.mov(
                m_code.r15,
                reinterpret_cast<uint64_t>(backend));

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

            for (size_t pairIndex = 0u;
                 pairIndex < block.pairs.size();
                 ++pairIndex)
            {
                const VuIrInstructionPair &pair =
                    block.pairs[pairIndex];
                m_code.cmp(m_code.r14d, m_code.r13d);
                m_code.jae(cycleBudgetExit, m_code.T_NEAR);

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
                            knownFmacSlot);
                        m_code.jmp(
                            done, m_code.T_NEAR);
                        m_code.L(fallback);
                        emitHelperPair(
                            pair, pairHelper,
                            dynamicExit);
                        m_code.L(done);
                    }
                    else
                    {
                        emitInlinePair(
                            pair, block.codeSize,
                            nullptr, xgkickHelper,
                            dynamicExit, false,
                            handleBranchState,
                            knownFmacSlot);
                    }
                }
                else
                {
                    emitHelperPair(
                        pair, pairHelper, dynamicExit);
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
                m_code.mov(
                    m_code.r10d,
                    static_cast<uint32_t>(
                        VuNativeBlockExit::BlockComplete));
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
            m_code.mov(
                m_code.r10d,
                static_cast<uint32_t>(
                    VuNativeBlockExit::BlockComplete));
            m_code.jmp(packResult, m_code.T_NEAR);

            m_code.L(dynamicExit);
            m_code.mov(m_code.r10d, m_code.r11d);

            m_code.L(packResult);
            m_code.mov(m_code.eax, m_code.r14d);
            m_code.mov(m_code.r11d, m_code.r10d);
            m_code.shl(m_code.r11, 32u);
            m_code.or_(m_code.rax, m_code.r11);
            m_code.jmp(epilogue);

            m_code.L(epilogue);
            if (m_usesAvx)
                m_code.vzeroupper();
            m_code.ldmxcsr(
                m_code.dword[
                    m_code.rsp + kSavedMxcsrOffset]);
            m_code.add(m_code.rsp, kStackBytes);
            m_code.pop(m_code.r15);
            m_code.pop(m_code.r14);
            m_code.pop(m_code.r13);
            m_code.pop(m_code.r12);
            m_code.pop(m_code.rbx);
            m_code.ret();
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

VuRecompilerBackend::VuRecompilerBackend(VuUnit &unit)
    : m_unit(unit), m_semantics(unit)
{
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

bool VuRecompilerBackend::programKey(
    const VuExecutionContext &context,
    VuCompilationMode mode,
    VuProgramKey &key,
    std::string *diagnostic) const
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
        .backend = VuNativeBackendMode::X86_64,
        .hostFeatures = hostFeatures(),
        .compilationMode = mode,
    };
    return VuProgramCache::validKey(key, diagnostic);
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
    if (needsInterpreterInstrumentation(context))
    {
        ++m_diagnostics.interpreterInstrumentationFallbacks;
        return m_semantics.run(context, maximumCycles);
    }
    if (maximumCycles == 0u)
    {
        ++m_diagnostics.cycleBudgetExits;
        return finish(0u, VuExitReason::CycleBudget);
    }

    const ScopedVuFloatMode vuFloatMode;
    uint32_t totalExecuted = 0u;
    while (state.active && totalExecuted < maximumCycles)
    {
        VuProgramKey key;
        if (!programKey(
                context, VuCompilationMode::Normal,
                key, &m_lastDiagnostic))
        {
            ++m_diagnostics.faultExits;
            return finish(totalExecuted, VuExitReason::Fault);
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
            return finish(totalExecuted, VuExitReason::Fault);
        }

        VuNativeBlockEntry entry = nullptr;
        const void *const address = program->nativeEntry();
        static_assert(sizeof(entry) == sizeof(address));
        std::memcpy(&entry, &address, sizeof(entry));

        const uint32_t remaining =
            maximumCycles - totalExecuted;
        const uint64_t helperPairsBefore =
            m_helperPairsIssued;
        const VuNativeBlockResult native =
            VuNativeBlockResult::decode(
                entry(&context, remaining));
        const uint64_t helperPairs =
            m_helperPairsIssued - helperPairsBefore;
        ++m_diagnostics.nativeEntries;
        if (native.executedCycles > remaining)
        {
            m_lastDiagnostic =
                "native VU block exceeded its cycle budget";
            ++m_diagnostics.faultExits;
            return finish(totalExecuted, VuExitReason::Fault);
        }
        if (helperPairs > native.executedCycles)
        {
            m_lastDiagnostic =
                "native VU block reported fewer cycles than its helpers";
            ++m_diagnostics.faultExits;
            return finish(totalExecuted, VuExitReason::Fault);
        }

        totalExecuted += native.executedCycles;
        state.issuedCycles += native.executedCycles;
        m_diagnostics.nativePairs +=
            native.executedCycles;
        m_diagnostics.helperPairs += helperPairs;
        m_diagnostics.inlinePairs +=
            native.executedCycles - helperPairs;

        const uint64_t currentGeneration =
            key.unit == VuUnitId::Vu0
                ? context.memory->getVU0CodeGeneration()
                : context.memory->getVU1CodeGeneration();
        if (currentGeneration != key.codeGeneration)
        {
            VuProgramKey currentKey = key;
            currentKey.codeGeneration = currentGeneration;
            (void)m_unit.programCache().lookup(currentKey);
            ++m_diagnostics.codeInvalidationExits;
            return finish(
                totalExecuted, VuExitReason::CodeInvalidated);
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
                    totalExecuted, VuExitReason::Fault);
            }
            continue;
        case VuNativeBlockExit::CycleBudget:
            ++m_diagnostics.cycleBudgetExits;
            return finish(
                totalExecuted, VuExitReason::CycleBudget);
        case VuNativeBlockExit::XgkickBoundary:
            ++m_diagnostics.xgkickExits;
            return finish(
                totalExecuted, VuExitReason::XgkickBoundary);
        case VuNativeBlockExit::UnsupportedInstruction:
            ++m_diagnostics.unsupportedExits;
            return finish(
                totalExecuted,
                VuExitReason::UnsupportedInstruction);
        case VuNativeBlockExit::Fault:
            m_lastDiagnostic =
                "native VU pair helper reported a fault";
            ++m_diagnostics.faultExits;
            return finish(totalExecuted, VuExitReason::Fault);
        default:
            return finish(
                totalExecuted, publicExit(native.exit));
        }
    }

    ++m_diagnostics.cycleBudgetExits;
    return finish(totalExecuted, VuExitReason::CycleBudget);
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
                    state.branchTarget & 0x3fffu;
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
        return handle;

    VuCompiledProgram program;
    if (!compile(
            context, key, program,
            m_lastDiagnostic))
    {
        return {};
    }
    handle = cache.insert(
        std::move(program), &m_lastDiagnostic);
    return handle;
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
        VuIrBlockForm::LinearTrace);
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
        X64VuBlockEmitter emitter(
            block, this,
            reinterpret_cast<const void *>(
                &VuRecompilerBackend::
                    executePairThunk),
            reinterpret_cast<const void *>(
                &VuRecompilerBackend::
                    advanceXgkickThunk),
            key.hostFeatures);
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

        const uint64_t nanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    Clock::now() - start)
                    .count());
        program = {
            .key = key,
            .block = std::move(block),
            .nativeCode = std::move(nativeCode),
            .entryOffset = 0u,
            .compilationNanoseconds = nanoseconds,
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

bool VuRecompilerBackend::needsInterpreterInstrumentation(
    const VuExecutionContext &context) const
{
    if (!context.enableInstrumentation)
        return false;
    if (m_unit.m_instructionObserverEnabled.load(
            std::memory_order_relaxed) &&
        m_unit.m_instructionObserver)
    {
        return true;
    }
    if (m_unit.m_progressTrackingEnabled.load(
            std::memory_order_relaxed))
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
