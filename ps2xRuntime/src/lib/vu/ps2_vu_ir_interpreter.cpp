#include "runtime/ps2_vu_ir.h"
#include "runtime/ps2_vu1.h"

#include "ps2_vu_float_mode.h"

#include <cstring>

VuIrInterpreterBackend::VuIrInterpreterBackend(VuUnit &unit)
    : m_semantics(unit)
{
}

std::string_view VuIrInterpreterBackend::name() const
{
    return "ir-interpreter";
}

VuRunResult VuIrInterpreterBackend::run(
    VuExecutionContext &context, uint32_t maxCycles)
{
    VuExecutionState &state = context.state;
    VuExecutionState *const previousState = m_semantics.m_state;
    const uint32_t previousCodeAddressMask =
        m_semantics.m_codeAddressMask;
    m_semantics.m_state = &state;
    m_semantics.m_codeAddressMask =
        context.codeSize != 0u
            ? context.codeSize - 1u
            : 0u;
    struct StateBindingGuard
    {
        VuExecutionState *&boundState;
        VuExecutionState *previousState;
        uint32_t &boundCodeAddressMask;
        uint32_t previousCodeAddressMask;

        ~StateBindingGuard()
        {
            boundState = previousState;
            boundCodeAddressMask =
                previousCodeAddressMask;
        }
    };
    const StateBindingGuard stateBinding{
        m_semantics.m_state, previousState,
        m_semantics.m_codeAddressMask,
        previousCodeAddressMask};

    if (!state.active)
    {
        return VuRunResult{
            .requestedCycles = maxCycles,
            .reason = VuExitReason::Inactive,
            .activeBefore = false,
            .activeAfter = false,
        };
    }

    const ScopedVuFloatMode vuFloatMode;
    uint32_t executedCycles = 0u;
    VuExitReason exitReason = VuExitReason::CycleBudget;

    for (uint32_t cycle = 0u; cycle < maxCycles; ++cycle)
    {
        if (!context.code || state.pc + 8u > context.codeSize)
        {
            state.active = false;
            exitReason = VuExitReason::CodeBounds;
            break;
        }

        uint32_t lowerWord = 0u;
        uint32_t upperWord = 0u;
        std::memcpy(
            &lowerWord, context.code + state.pc,
            sizeof(lowerWord));
        std::memcpy(
            &upperWord, context.code + state.pc + 4u,
            sizeof(upperWord));
        const VuIrInstructionPair pair =
            decodeVuIrInstructionPair(
                state.pc, lowerWord, upperWord);
        if (!verifyVuIrInstructionPair(pair))
        {
            exitReason = VuExitReason::Fault;
            break;
        }
        if (vuIrHasPairFlag(pair, VuIrPairUnsupported))
        {
            exitReason = VuExitReason::UnsupportedInstruction;
            break;
        }

        ++executedCycles;
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
                pair.lowerWord, context.data, context.dataSize,
                context.sideEffects, context.observer,
                pair.upperWord);
            m_semantics.execUpper(pair.upperWord);
            break;
        case VuIrPairOrder::UpperThenLower:
            m_semantics.execUpper(pair.upperWord);
            if (pair.lower.opcode != VuIrOpcode::Nop)
            {
                m_semantics.execLower(
                    pair.lowerWord, context.data,
                    context.dataSize, context.sideEffects,
                    context.observer, pair.upperWord);
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
                context.sideEffects, context.observer, 1u, false);
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

        if (state.ebit)
        {
            if (state.pipeline.xgkick.active)
            {
                m_semantics.advanceXgkick(
                    context.data, context.dataSize,
                    context.sideEffects, context.observer,
                    0u, true);
            }
            m_semantics.flushQPipeline();
            m_semantics.flushPPipeline();
            m_semantics.flushFmacFlagPipeline();
            state.viBackupCycles = 0u;
            state.active = false;
            exitReason = VuExitReason::ProgramEnded;
            break;
        }

        if (vuIrHasPairFlag(pair, VuIrPairEnd))
            state.ebit = true;
    }

    state.issuedCycles += executedCycles;
    return VuRunResult{
        .requestedCycles = maxCycles,
        .executedCycles = executedCycles,
        .reason = exitReason,
        .activeBefore = true,
        .activeAfter = state.active,
        .completed = !state.active,
    };
}
