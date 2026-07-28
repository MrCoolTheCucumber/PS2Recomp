#include "runtime/ps2_vu1.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu_recompiler.h"
#include "ps2_vu_float_mode.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

// Keep selection cheap for the permanent interpreter and keep its cold decode
// cache maintenance outside the pair loop under whole-program optimization.
#if defined(_MSC_VER)
#define PS2X_VU_ALWAYS_INLINE __forceinline
#define PS2X_VU_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PS2X_VU_ALWAYS_INLINE inline __attribute__((always_inline))
#define PS2X_VU_NOINLINE __attribute__((noinline))
#else
#define PS2X_VU_ALWAYS_INLINE inline
#define PS2X_VU_NOINLINE
#endif

namespace
{
    uint8_t vuLowerSpecialOp(uint32_t lower)
    {
        if ((lower & 0x80000000u) == 0u ||
            (lower & 0x3fu) < 0x3cu)
        {
            return 0xffu;
        }
        return static_cast<uint8_t>(
            (lower & 0x3u) | ((lower >> 4u) & 0x7cu));
    }

    bool vuLowerWaitsForQ(uint32_t lower)
    {
        switch (vuLowerSpecialOp(lower))
        {
        case 0x38u: // DIV
        case 0x39u: // SQRT
        case 0x3au: // RSQRT
        case 0x3bu: // WAITQ
            return true;
        default:
            return false;
        }
    }

    bool vuLowerWaitsForP(uint32_t lower)
    {
        switch (vuLowerSpecialOp(lower))
        {
        case 0x70u: // ESADD
        case 0x71u: // ERSADD
        case 0x72u: // ELENG
        case 0x73u: // ERLENG
        case 0x7au: // ERCPR
        case 0x7bu: // WAITP
        case 0x7du: // EATAN
            return true;
        default:
            return false;
        }
    }

    class RuntimeVuSideEffectSink final : public IVuSideEffectSink
    {
    public:
        RuntimeVuSideEffectSink(GS &gs, PS2Memory *memory)
            : m_gs(gs), m_memory(memory)
        {
        }

        void submitPath1Packet(
            const uint8_t *data, uint32_t sizeBytes) override
        {
            if (m_memory)
            {
                m_memory->submitGifPacket(
                    GifPathId::Path1, data, sizeBytes);
                return;
            }
            m_gs.processGIFPacket(data, sizeBytes);
        }

    private:
        GS &m_gs;
        PS2Memory *m_memory;
    };
}

std::string_view vuBackendKindName(VuBackendKind kind)
{
    switch (kind)
    {
    case VuBackendKind::Auto:
        return "auto";
    case VuBackendKind::Interpreter:
        return "interpreter";
    case VuBackendKind::Recompiler:
        return "recompiler";
    case VuBackendKind::Verify:
        return "verify";
    }
    return "unknown";
}

bool parseVuBackendKind(std::string_view text, VuBackendKind &kind)
{
    for (const VuBackendKind candidate : {
             VuBackendKind::Auto,
             VuBackendKind::Interpreter,
             VuBackendKind::Recompiler,
             VuBackendKind::Verify,
         })
    {
        if (text == vuBackendKindName(candidate))
        {
            kind = candidate;
            return true;
        }
    }
    return false;
}

std::string_view vuExitReasonName(VuExitReason reason)
{
    switch (reason)
    {
    case VuExitReason::Inactive:
        return "inactive";
    case VuExitReason::CycleBudget:
        return "cycle-budget";
    case VuExitReason::ProgramEnded:
        return "program-ended";
    case VuExitReason::CodeBounds:
        return "code-bounds";
    case VuExitReason::BranchBoundary:
        return "branch-boundary";
    case VuExitReason::XgkickBoundary:
        return "xgkick-boundary";
    case VuExitReason::DebugObserver:
        return "debug-observer";
    case VuExitReason::CodeInvalidated:
        return "code-invalidated";
    case VuExitReason::UnsupportedInstruction:
        return "unsupported-instruction";
    case VuExitReason::Fault:
        return "fault";
    }
    return "unknown";
}

bool vuExecutionStatesEqual(
    const VuExecutionState &left, const VuExecutionState &right,
    std::string *firstDifference)
{
    const auto fail = [&](const char *field)
    {
        if (firstDifference)
            *firstDifference = field;
        return false;
    };
    const auto differentBytes =
        [](const void *leftBytes, const void *rightBytes, size_t size)
    {
        return std::memcmp(leftBytes, rightBytes, size) != 0;
    };

    if (differentBytes(left.vf, right.vf, sizeof(left.vf)))
        return fail("vf");
    if (differentBytes(left.vi, right.vi, sizeof(left.vi)))
        return fail("vi");
    if (differentBytes(left.acc, right.acc, sizeof(left.acc)))
        return fail("acc");
    if (differentBytes(&left.q, &right.q, sizeof(left.q)))
        return fail("q");
    if (differentBytes(&left.p, &right.p, sizeof(left.p)))
        return fail("p");
    if (differentBytes(&left.i, &right.i, sizeof(left.i)))
        return fail("i");
#define VU_COMPARE_STATE_FIELD(field) \
    if (left.field != right.field)    \
        return fail(#field)
    VU_COMPARE_STATE_FIELD(pc);
    VU_COMPARE_STATE_FIELD(mac);
    VU_COMPARE_STATE_FIELD(clip);
    VU_COMPARE_STATE_FIELD(status);
    VU_COMPARE_STATE_FIELD(ebit);
    VU_COMPARE_STATE_FIELD(top);
    VU_COMPARE_STATE_FIELD(itop);
    VU_COMPARE_STATE_FIELD(branchPending);
    VU_COMPARE_STATE_FIELD(branchTarget);
    VU_COMPARE_STATE_FIELD(branchDelay);
    VU_COMPARE_STATE_FIELD(viBackupCycles);
    VU_COMPARE_STATE_FIELD(viBackupRegister);
    VU_COMPARE_STATE_FIELD(viBackupValue);
    VU_COMPARE_STATE_FIELD(active);
    VU_COMPARE_STATE_FIELD(issuedCycles);
#undef VU_COMPARE_STATE_FIELD

    const VuPipelineState &leftPipeline = left.pipeline;
    const VuPipelineState &rightPipeline = right.pipeline;
#define VU_COMPARE_PIPELINE_FIELD(field)                  \
    if (leftPipeline.field != rightPipeline.field)        \
        return fail("pipeline." #field)
    VU_COMPARE_PIPELINE_FIELD(xgkick.active);
    VU_COMPARE_PIPELINE_FIELD(xgkick.address);
    VU_COMPARE_PIPELINE_FIELD(xgkick.tagBytesRemaining);
    VU_COMPARE_PIPELINE_FIELD(xgkick.cycleCredit);
    VU_COMPARE_PIPELINE_FIELD(xgkick.tagEop);
    VU_COMPARE_PIPELINE_FIELD(xgkick.packet);
    VU_COMPARE_PIPELINE_FIELD(delayedQ.active);
    if (differentBytes(
            &leftPipeline.delayedQ.result,
            &rightPipeline.delayedQ.result,
            sizeof(leftPipeline.delayedQ.result)))
    {
        return fail("pipeline.delayedQ.result");
    }
    VU_COMPARE_PIPELINE_FIELD(delayedQ.cyclesRemaining);
    VU_COMPARE_PIPELINE_FIELD(delayedP.active);
    if (differentBytes(
            &leftPipeline.delayedP.result,
            &rightPipeline.delayedP.result,
            sizeof(leftPipeline.delayedP.result)))
    {
        return fail("pipeline.delayedP.result");
    }
    VU_COMPARE_PIPELINE_FIELD(delayedP.cyclesRemaining);
    for (size_t index = 0u;
         index < leftPipeline.fmacFlags.size(); ++index)
    {
        const VuPipelineState::FmacFlags &leftFlags =
            leftPipeline.fmacFlags[index];
        const VuPipelineState::FmacFlags &rightFlags =
            rightPipeline.fmacFlags[index];
        if (leftFlags.active != rightFlags.active)
            return fail("pipeline.fmacFlags.active");
        if (leftFlags.mac != rightFlags.mac)
            return fail("pipeline.fmacFlags.mac");
        if (leftFlags.status != rightFlags.status)
            return fail("pipeline.fmacFlags.status");
    }
    VU_COMPARE_PIPELINE_FIELD(fmacFlagIndex);
    VU_COMPARE_PIPELINE_FIELD(workingMac);
#undef VU_COMPARE_PIPELINE_FIELD

    if (firstDifference)
        firstDifference->clear();
    return true;
}

void VuTransactionalSideEffectSink::submitPath1Packet(
    const uint8_t *data, uint32_t sizeBytes)
{
    std::vector<uint8_t> &packet = m_path1Packets.emplace_back();
    if (sizeBytes != 0u)
        packet.assign(data, data + sizeBytes);
}

void VuTransactionalSideEffectSink::clear()
{
    m_path1Packets.clear();
}

void VuTransactionalSideEffectSink::commitTo(
    IVuSideEffectSink &destination) const
{
    for (const std::vector<uint8_t> &packet : m_path1Packets)
    {
        destination.submitPath1Packet(
            packet.data(), static_cast<uint32_t>(packet.size()));
    }
}

VuUnit::VuUnit()
    : VuUnit(VuUnitId::Vu1)
{
}

VuUnit::VuUnit(VuUnitId unit)
    : m_unitId(unit),
      m_interpreter(std::make_unique<VuInterpreterBackend>(*this)),
      m_backend(m_interpreter.get())
{
    reset();
}

VuUnit::~VuUnit() = default;

VuProgramCache &VuUnit::programCache()
{
    if (!m_programCache)
    {
        m_programCache =
            std::make_unique<VuProgramCache>(m_unitId);
    }
    return *m_programCache;
}

const VuRecompilerDiagnostics *
VuUnit::recompilerDiagnosticsIfCreated() const
{
    return m_recompiler
               ? &m_recompiler->m_diagnostics
               : nullptr;
}

VuInterpreterBackend::VuInterpreterBackend(VuUnit &unit)
    : m_unit(unit)
{
}

std::string_view VuInterpreterBackend::name() const
{
    return "interpreter";
}

void VuUnit::reset()
{
    if (m_workloadProfileInvocationActive &&
        m_workloadProfileMemory)
    {
        m_workloadProfileMemory->
            endVu1WorkloadProfileInvocation(false);
    }
    if (m_workloadProfileMemory)
    {
        m_workloadProfileMemory->
            resetVu1WorkloadProfileEpoch();
    }
    m_workloadProfileMemory = nullptr;
    m_workloadProfileInvocationPending = false;
    m_workloadProfileInvocationActive = false;
    m_state = {};
    m_state.vf[0][3] = 1.0f; // VF0.w = 1.0
    m_state.q = 1.0f;
    m_progressActive.store(0u, std::memory_order_relaxed);
    m_progressInvocations.store(0u, std::memory_order_relaxed);
    m_progressCycles.store(0u, std::memory_order_relaxed);
    m_progressPc.store(0u, std::memory_order_relaxed);
}

VuProgressSnapshot VuUnit::getProgressSnapshot() const
{
    VuProgressSnapshot snapshot{};
    snapshot.enabled =
        m_progressTrackingEnabled.load(std::memory_order_relaxed);
    snapshot.active =
        m_progressActive.load(std::memory_order_relaxed) != 0u;
    snapshot.invocations =
        m_progressInvocations.load(std::memory_order_relaxed);
    snapshot.cycles =
        m_progressCycles.load(std::memory_order_relaxed);
    snapshot.pc = m_progressPc.load(std::memory_order_relaxed);
    return snapshot;
}

void VuUnit::setProgressTrackingEnabled(bool enabled)
{
    m_progressTrackingEnabled.store(enabled, std::memory_order_release);
}

VuUnit::ProgressTracker::ProgressTracker(
    VuUnit &unit, VuExecutionState &state,
    bool enabled, uint32_t &executedCycles)
    : m_unit(unit),
      m_state(state),
      m_executedCycles(executedCycles),
      m_enabled(
          enabled &&
          unit.m_progressTrackingEnabled.load(
              std::memory_order_relaxed))
{
    if (!m_enabled)
        return;

    m_unit.m_progressActive.fetch_add(
        1u, std::memory_order_relaxed);
    m_unit.m_progressInvocations.fetch_add(
        1u, std::memory_order_relaxed);
    m_unit.m_progressPc.store(
        m_state.pc, std::memory_order_relaxed);
}

VuUnit::ProgressTracker::~ProgressTracker()
{
    if (!m_enabled)
        return;

    publish();
    m_unit.m_progressPc.store(
        m_state.pc, std::memory_order_relaxed);
    m_unit.m_progressActive.fetch_sub(
        1u, std::memory_order_relaxed);
}

void VuUnit::ProgressTracker::publish()
{
    if (!m_enabled ||
        m_executedCycles <= m_committedCycles)
    {
        return;
    }

    m_unit.m_progressCycles.fetch_add(
        m_executedCycles - m_committedCycles,
        std::memory_order_relaxed);
    m_committedCycles = m_executedCycles;
    m_unit.m_progressPc.store(
        m_state.pc, std::memory_order_relaxed);
}

void VuUnit::setInstructionObserver(VuInstructionObserver observer)
{
    m_instructionObserver = std::move(observer);
    m_instructionObserverEnabled.store(
        static_cast<bool>(m_instructionObserver), std::memory_order_release);
}

void VuUnit::setInstructionObserverEnabled(bool enabled)
{
    m_instructionObserverEnabled.store(enabled, std::memory_order_release);
}

bool VuUnit::setBackend(
    VuBackendKind requested, std::string *diagnostic)
{
    if (m_state.active && requested != m_requestedBackend)
    {
        if (diagnostic)
        {
            *diagnostic =
                "cannot change a VU backend while the unit is active";
        }
        return false;
    }

    VuBackendKind resolved = VuBackendKind::Interpreter;
    IVuExecutionBackend *backend = m_interpreter.get();
    const bool explicitNative =
        requested == VuBackendKind::Recompiler;
    if (explicitNative &&
        !VuRecompilerBackend::supported())
    {
        if (diagnostic)
        {
            *diagnostic =
                VuRecompilerBackend::built()
                    ? "the x86-64 VU recompiler is unsupported on this host"
                    : "the x86-64 VU recompiler was not built";
        }
        return false;
    }
    const bool automaticNative =
        m_unitId == VuUnitId::Vu1 &&
        requested == VuBackendKind::Auto &&
        VuRecompilerBackend::supported();
    if (explicitNative || automaticNative)
    {
        if (!m_recompiler)
        {
            m_recompiler =
                std::make_unique<VuRecompilerBackend>(*this);
        }
        resolved = VuBackendKind::Recompiler;
        backend = m_recompiler.get();
    }

    m_requestedBackend = requested;
    m_resolvedBackend = resolved;
    m_backend = backend;
    if (diagnostic)
        diagnostic->clear();
    return true;
}

std::string_view VuUnit::backendName() const
{
    return m_backend ? m_backend->name() : "none";
}

float VuInterpreterBackend::broadcast(const float *vf, uint8_t bc)
{
    return vf[bc & 3];
}

void VuInterpreterBackend::applyDest(
    float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8)
        dst[0] = result[0]; // x
    if (dest & 0x4)
        dst[1] = result[1]; // y
    if (dest & 0x2)
        dst[2] = result[2]; // z
    if (dest & 0x1)
        dst[3] = result[3]; // w
}

void VuInterpreterBackend::applyDestAcc(
    const float *result, uint8_t dest)
{
    VuExecutionState &state = *m_state;
    applyDest(state.acc, result, dest);
}

PS2X_VU_NOINLINE
VuInterpreterBackend::DecodedInstructionPair
VuInterpreterBackend::decodeInstructionPair(
    const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));

    decoded.iBit = ((decoded.upper >> 31) & 1u) != 0u;
    decoded.eBit = ((decoded.upper >> 30) & 1u) != 0u;
    decoded.lowerBeforeUpper = !decoded.iBit && vuLowerShouldRunBeforeUpper(decoded.upper, decoded.lower);
    return decoded;
}

PS2X_VU_NOINLINE
void VuInterpreterBackend::rebuildDecodedCodeCache(
    const uint8_t *vuCode, uint32_t codeSize,
    const PS2Memory *memory, uint64_t generation)
{
    VuUnit::InterpreterCache &cache = m_unit.m_interpreterCache;
    const uint32_t pairCount = codeSize / 8u;
    cache.decodedCode.resize(pairCount);
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        cache.decodedCode[i] =
            decodeInstructionPair(vuCode, i * 8u);
    }

    cache.vuCode = vuCode;
    cache.memory = memory;
    cache.codeSize = codeSize;
    cache.codeGeneration = generation;
    cache.valid = true;
}

#undef PS2X_VU_NOINLINE

void VuUnit::start(
    uint32_t startPC, uint32_t top, uint32_t itop,
    PS2Memory *memory)
{
    if (m_workloadProfileInvocationActive &&
        m_workloadProfileMemory)
    {
        m_workloadProfileMemory->
            endVu1WorkloadProfileInvocation(false);
    }
    m_workloadProfileInvocationActive = false;
    m_workloadProfileInvocationPending = true;
    const uint32_t codeAddressMask =
        m_unitId == VuUnitId::Vu0
            ? PS2_VU0_CODE_SIZE - 1u
            : PS2_VU1_CODE_SIZE - 1u;
    m_state.pc = startPC & codeAddressMask;
    m_state.ebit = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.viBackupCycles = 0;
    m_state.pipeline.fmacFlags = {};
    m_state.pipeline.fmacFlagIndex = 0u;
    m_state.pipeline.workingMac = m_state.mac;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    m_state.active = true;
    m_state.issuedCycles = 0u;
    if (memory && memory->isVif1DmaTraceActive())
        memory->traceVu1Invocation(m_state.pc, top, itop, false, m_state);
}

void VuUnit::resumeState(
    uint32_t top, uint32_t itop, PS2Memory *memory)
{
    if (m_workloadProfileInvocationActive &&
        m_workloadProfileMemory)
    {
        m_workloadProfileMemory->
            endVu1WorkloadProfileInvocation(false);
    }
    m_workloadProfileInvocationActive = false;
    m_workloadProfileInvocationPending = true;
    m_state.ebit = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.active = true;
    m_state.issuedCycles = 0u;
    if (memory && memory->isVif1DmaTraceActive())
        memory->traceVu1Invocation(m_state.pc, top, itop, true, m_state);
}

VuRunResult VuUnit::advance(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    if (!m_state.active)
    {
        return VuRunResult{
            .requestedCycles = maxCycles,
            .reason = VuExitReason::Inactive,
            .activeBefore = false,
            .activeAfter = false,
        };
    }

    return run(
        vuCode, codeSize, vuData, dataSize,
        gs, memory, maxCycles, false);
}

void VuUnit::execute(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory,
    uint32_t startPC, uint32_t top, uint32_t itop,
    uint32_t maxCycles)
{
    start(startPC, top, itop, memory);
    (void)run(
        vuCode, codeSize, vuData, dataSize,
        gs, memory, maxCycles, true);
}

void VuUnit::resume(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory,
    uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    resumeState(top, itop, memory);
    (void)run(
        vuCode, codeSize, vuData, dataSize,
        gs, memory, maxCycles, true);
}

VuRunResult VuUnit::continueExecution(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    if (!m_state.active)
    {
        return VuRunResult{
            .requestedCycles = maxCycles,
            .reason = VuExitReason::Inactive,
            .activeBefore = false,
            .activeAfter = false,
        };
    }
    return run(
        vuCode, codeSize, vuData, dataSize,
        gs, memory, maxCycles, true);
}

PS2X_VU_ALWAYS_INLINE VuRunResult VuUnit::run(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles,
    bool traceBudgetBoundary)
{
    if (m_resolvedBackend == VuBackendKind::Recompiler)
        [[unlikely]]
    {
        return runRecompiler(
            vuCode, codeSize, vuData, dataSize,
            gs, memory, maxCycles,
            traceBudgetBoundary);
    }

    RuntimeVuSideEffectSink sideEffects(gs, memory);
    VuExecutionContext context{
        .state = m_state,
        .code = vuCode,
        .codeSize = codeSize,
        .data = vuData,
        .dataSize = dataSize,
        .sideEffects = sideEffects,
        .memory = memory,
        .traceBudgetBoundary = traceBudgetBoundary,
        .enableInstrumentation = true,
    };

    // Keep the interpreter call concrete so auto/debug mode retains its
    // optimizer-visible hot loop.
    return m_interpreter->run(context, maxCycles);
}

#undef PS2X_VU_ALWAYS_INLINE

VuRunResult VuUnit::runRecompiler(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles,
    bool traceBudgetBoundary)
{
    if (!m_recompiler)
    {
        return {
            .requestedCycles = maxCycles,
            .reason = VuExitReason::Fault,
            .activeBefore = m_state.active,
            .activeAfter = m_state.active,
            .completed = !m_state.active,
        };
    }

    RuntimeVuSideEffectSink sideEffects(gs, memory);
    VuExecutionContext context{
        .state = m_state,
        .code = vuCode,
        .codeSize = codeSize,
        .data = vuData,
        .dataSize = dataSize,
        .sideEffects = sideEffects,
        .memory = memory,
        .traceBudgetBoundary = traceBudgetBoundary,
        .enableInstrumentation = true,
        .enableProgressAccounting = false,
    };
    const bool activeBefore = m_state.active;
    uint32_t executedCycles = 0u;
    ProgressTracker progress(
        *this, m_state, true, executedCycles);
    const auto finish =
        [&](VuRunResult result)
        {
            result.requestedCycles = maxCycles;
            result.executedCycles = executedCycles;
            result.activeBefore = activeBefore;
            result.activeAfter = m_state.active;
            result.completed = !m_state.active;
            return result;
        };
    while (m_state.active &&
           executedCycles < maxCycles)
    {
        VuRunResult result =
            m_recompiler->run(
                context,
                maxCycles - executedCycles);
        executedCycles += result.executedCycles;
        progress.publish();
        if (result.reason ==
            VuExitReason::XgkickBoundary)
        {
            if (result.executedCycles == 0u)
            {
                result.reason = VuExitReason::Fault;
                return finish(result);
            }
            continue;
        }
        if (result.reason !=
            VuExitReason::UnsupportedInstruction)
        {
            return finish(result);
        }

        result = m_recompiler->
            runInterpreterFallback(context, 1u);
        executedCycles += result.executedCycles;
        progress.publish();
        if (result.executedCycles != 1u)
        {
            result.reason = VuExitReason::Fault;
            return finish(result);
        }
        if (result.reason != VuExitReason::CycleBudget ||
            !m_state.active)
        {
            return finish(result);
        }
    }
    return {
        .requestedCycles = maxCycles,
        .executedCycles = executedCycles,
        .reason = m_state.active
                      ? VuExitReason::CycleBudget
                      : VuExitReason::ProgramEnded,
        .activeBefore = activeBefore,
        .activeAfter = m_state.active,
        .completed = !m_state.active,
    };
}

VuRunResult VuInterpreterBackend::run(
    VuExecutionContext &context, uint32_t maxCycles)
{
    VuExecutionState &state = context.state;
    VuExecutionState *const previousState = m_state;
    const uint32_t previousCodeAddressMask =
        m_codeAddressMask;
    m_state = &state;
    m_codeAddressMask =
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
        m_state, previousState,
        m_codeAddressMask, previousCodeAddressMask};

    if (!state.active)
    {
        return VuRunResult{
            .requestedCycles = maxCycles,
            .reason = VuExitReason::Inactive,
            .activeBefore = false,
            .activeAfter = false,
        };
    }

    const uint8_t *const vuCode = context.code;
    const uint32_t codeSize = context.codeSize;
    uint8_t *const vuData = context.data;
    const uint32_t dataSize = context.dataSize;
    IVuSideEffectSink &sideEffects = context.sideEffects;
    PS2Memory *const memory = context.memory;
    const bool traceBudgetBoundary = context.traceBudgetBoundary;
    const bool instrumentation = context.enableInstrumentation;

    // The VU performs 24-bit floating-point calculations by truncating toward
    // zero. Keep that host mode local to VU execution so callers retain their
    // own floating-point environment.
    const ScopedVuFloatMode vuFloatMode;
    const bool traceVu1 =
        instrumentation && memory && memory->isVif1DmaTraceActive();
    const bool profileVu1 =
        instrumentation && memory &&
        vuCode == memory->getVU1Code() &&
        memory->isVu1WorkloadProfileEnabled();
    if (profileVu1 && m_unit.m_workloadProfileInvocationPending)
    {
        memory->beginVu1WorkloadProfileInvocation(state.pc);
        m_unit.m_workloadProfileMemory = memory;
        m_unit.m_workloadProfileInvocationPending = false;
        m_unit.m_workloadProfileInvocationActive = true;
    }
    const bool trackProgress =
        instrumentation &&
        m_unit.m_progressTrackingEnabled.load(std::memory_order_relaxed) &&
        context.enableProgressAccounting;
    uint32_t committedProgressCycles = 0u;
    VuUnit::InterpreterCache &cache = m_unit.m_interpreterCache;
    if (trackProgress)
    {
        m_unit.m_progressActive.fetch_add(1u, std::memory_order_relaxed);
        m_unit.m_progressInvocations.fetch_add(
            1u, std::memory_order_relaxed);
        m_unit.m_progressPc.store(
            state.pc, std::memory_order_relaxed);
    }
    struct ProgressGuard
    {
        VuUnit &unit;
        VuExecutionState &state;
        bool enabled;
        uint32_t &executed;
        uint32_t &committed;

        ~ProgressGuard()
        {
            if (!enabled)
            {
                return;
            }
            if (executed > committed)
            {
                unit.m_progressCycles.fetch_add(
                    executed - committed, std::memory_order_relaxed);
            }
            unit.m_progressPc.store(
                state.pc, std::memory_order_relaxed);
            unit.m_progressActive.fetch_sub(
                1u, std::memory_order_relaxed);
        }
    };
    uint32_t executedCycles = 0u;
    ProgressGuard progress{
        m_unit, state, trackProgress,
        executedCycles, committedProgressCycles};
    bool ended = false;
    VuExitReason exitReason = VuExitReason::CycleBudget;

    for (uint32_t cycle = 0; cycle < maxCycles; ++cycle)
    {
        if (state.pc + 8 > codeSize)
        {
            state.active = false;
            exitReason = VuExitReason::CodeBounds;
            break;
        }

        const uint32_t issuedPc = state.pc;
        DecodedInstructionPair decoded;
        bool decodedFromCache = false;
        if ((issuedPc & 7u) == 0u && memory)
        {
            uint64_t generation = 0u;
            bool cacheable = false;
            if (vuCode == memory->getVU0Code())
            {
                generation = memory->getVU0CodeGeneration();
                cacheable = true;
            }
            else if (vuCode == memory->getVU1Code())
            {
                generation = memory->getVU1CodeGeneration();
                cacheable = true;
            }

            if (cacheable)
            {
                const bool rebuild =
                    !cache.valid ||
                    cache.vuCode != vuCode ||
                    cache.memory != memory ||
                    cache.codeSize != codeSize ||
                    cache.codeGeneration != generation;
                if (rebuild)
                {
                    rebuildDecodedCodeCache(
                        vuCode, codeSize, memory, generation);
                }
                decoded = cache.decodedCode[issuedPc / 8u];
                decodedFromCache = true;
            }
        }
        if (!decodedFromCache)
            decoded = decodeInstructionPair(vuCode, issuedPc);
        if (traceVu1)
        {
            memory->traceVu1Instruction(
                issuedPc, decoded.lower, decoded.upper, state);
        }
        if (instrumentation &&
            m_unit.m_workloadProfileInvocationActive &&
            m_unit.m_workloadProfileMemory)
        {
            m_unit.m_workloadProfileMemory->
                recordVu1WorkloadProfileInstruction(
                    issuedPc, decoded.lower, decoded.upper);
        }
        if (instrumentation &&
            m_unit.m_instructionObserverEnabled.load(
                std::memory_order_relaxed) &&
            m_unit.m_instructionObserver)
        {
            m_unit.m_instructionObserver(
                executedCycles, state.pc, decoded.lower, decoded.upper,
                state);
        }
        ++executedCycles;
        if (trackProgress &&
            executedCycles - committedProgressCycles >= 256u)
        {
            m_unit.m_progressCycles.fetch_add(
                executedCycles - committedProgressCycles,
                std::memory_order_relaxed);
            committedProgressCycles = executedCycles;
            m_unit.m_progressPc.store(
                state.pc, std::memory_order_relaxed);
        }

        // DIV/SQRT results are written to Q seven cycles after issue
        // (RSQRT takes thirteen). EFU results similarly remain held outside
        // architectural P for their documented latency. The instruction
        // observer deliberately runs first because pipeline writes occur
        // after the current pair has been fetched. A new operation on the
        // same scalar unit, WAITQ, or WAITP drains the prior result before
        // the current pair executes. There is no implicit P dependency for
        // MFP; VU microcode must synchronize it explicitly with WAITP.
        advanceQPipeline();
        advancePPipeline();
        advanceFmacFlagPipeline();
        if (!decoded.iBit && vuLowerWaitsForQ(decoded.lower))
            flushQPipeline();
        if (!decoded.iBit && vuLowerWaitsForP(decoded.lower))
            flushPPipeline();

        // Integer writes such as MTIR and auto-incrementing loads/stores expose
        // the pre-update VI value to a branch in the immediately following
        // instruction pair. PCSX2 models this as a two-cycle backup which is
        // aged before each subsequent pair.
        if (state.viBackupCycles != 0u)
            --state.viBackupCycles;

        // LOI is controlled by the upper I-bit.  The lower word is the float immediate.
        // DobieStation executes the upper instruction first, then commits lower into I.
        if (decoded.iBit)
        {
            // LOI is special: the upper instruction sees the old I value, then LOI loads I.
            execUpper(decoded.upper);
            std::memcpy(&state.i, &decoded.lower, sizeof(decoded.lower));
        }
        else if (decoded.lowerBeforeUpper)
        {
            // VU upper/lower execute as a pair.  If the upper op writes a VF register
            // that the lower op reads or also writes, Dobie runs the lower side first
            // so it observes the old VF value and the upper write has priority.
            execLower(
                decoded.lower, vuData, dataSize,
                sideEffects, memory, decoded.upper);
            execUpper(decoded.upper);
        }
        else
        {
            execUpper(decoded.upper);
            if (decoded.lower != 0x00000000u &&
                decoded.lower != 0x8000033cu)
            {
                execLower(
                    decoded.lower, vuData, dataSize,
                    sideEffects, memory, decoded.upper);
            }
        }

        // Enforce VF0 invariant
        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        // Enforce VI0 invariant
        state.vi[0] = 0;

        // PATH1 consumes one quadword per two VU cycles while the microprogram
        // continues to run and may keep producing data in the same ring buffer.
        if (state.pipeline.xgkick.active)
        {
            advanceXgkick(
                vuData, dataSize, sideEffects, memory, 1u, false);
        }

        uint32_t nextPC = state.pc + 8;
        if (nextPC >= codeSize)
            nextPC = 0;
        state.pc = nextPC;

        // VU branch/jump has a delay slot. Branch handlers set a pending target;
        // we execute one sequential instruction before committing the branch.
        if (state.branchPending)
        {
            if (state.branchDelay == 0)
            {
                state.pc =
                    state.branchTarget &
                    m_codeAddressMask;
                state.branchPending = false;
            }
            else
            {
                --state.branchDelay;
            }
        }
        if (instrumentation &&
            m_unit.m_workloadProfileInvocationActive &&
            m_unit.m_workloadProfileMemory)
        {
            m_unit.m_workloadProfileMemory->
                recordVu1WorkloadProfileTransition(
                    issuedPc, state.pc);
        }

        if (state.ebit)
        {
            // VU1 completion drains the pending XGKICK before becoming idle.
            if (state.pipeline.xgkick.active)
            {
                advanceXgkick(
                    vuData, dataSize, sideEffects, memory, 0u, true);
            }
            flushQPipeline();
            flushPPipeline();
            flushFmacFlagPipeline();
            state.viBackupCycles = 0u;
            ended = true;
            state.active = false;
            exitReason = VuExitReason::ProgramEnded;
            break;
        }

        if (decoded.eBit)
            state.ebit = true;
    }

    if (traceVu1 &&
        (!state.active || traceBudgetBoundary))
    {
        memory->traceVu1InvocationEnd(
            state.pc, ended, !ended && executedCycles == maxCycles,
            state.vi, std::size(state.vi));
    }
    if (instrumentation &&
        !state.active &&
        m_unit.m_workloadProfileInvocationActive &&
        m_unit.m_workloadProfileMemory)
    {
        m_unit.m_workloadProfileMemory->
            endVu1WorkloadProfileInvocation(ended);
        m_unit.m_workloadProfileInvocationActive = false;
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

void VuInterpreterBackend::advanceQPipeline()
{
    VuExecutionState &state = *m_state;
    VuPipelineState::DelayedQ &delayedQ = state.pipeline.delayedQ;
    if (!delayedQ.active)
        return;

    if (delayedQ.cyclesRemaining != 0u)
        --delayedQ.cyclesRemaining;
    if (delayedQ.cyclesRemaining == 0u)
        flushQPipeline();
}

void VuInterpreterBackend::flushQPipeline()
{
    VuExecutionState &state = *m_state;
    VuPipelineState::DelayedQ &delayedQ = state.pipeline.delayedQ;
    if (!delayedQ.active)
        return;

    state.q = delayedQ.result;
    delayedQ = {};
}

void VuInterpreterBackend::scheduleQ(float result, uint32_t latency)
{
    VuExecutionState &state = *m_state;
    state.pipeline.delayedQ = {
        .active = true,
        .result = result,
        .cyclesRemaining = latency,
    };
}

void VuInterpreterBackend::advancePPipeline()
{
    VuExecutionState &state = *m_state;
    VuPipelineState::DelayedP &delayedP =
        state.pipeline.delayedP;
    if (!delayedP.active)
        return;

    if (delayedP.cyclesRemaining != 0u)
        --delayedP.cyclesRemaining;
    if (delayedP.cyclesRemaining != 0u)
        return;

    state.p = delayedP.result;
    delayedP = {};
}

void VuInterpreterBackend::flushPPipeline()
{
    VuExecutionState &state = *m_state;
    VuPipelineState::DelayedP &delayedP =
        state.pipeline.delayedP;
    if (!delayedP.active)
        return;
    state.p = delayedP.result;
    delayedP = {};
}

void VuInterpreterBackend::scheduleP(
    float result, uint32_t latency)
{
    VuExecutionState &state = *m_state;
    state.pipeline.delayedP = {
        .active = true,
        .result = result,
        .cyclesRemaining = latency,
    };
}

void VuInterpreterBackend::advanceFmacFlagPipeline()
{
    VuExecutionState &state = *m_state;
    VuPipelineState &pipeline = state.pipeline;
    pipeline.fmacFlagIndex = static_cast<uint8_t>(
        (pipeline.fmacFlagIndex + 1u) %
        VuPipelineState::kFmacFlagStages);
    VuPipelineState::FmacFlags &pending =
        pipeline.fmacFlags[pipeline.fmacFlagIndex];
    if (!pending.active)
        return;

    commitFmacFlags(pending);
    pending = {};
}

void VuInterpreterBackend::flushFmacFlagPipeline()
{
    VuExecutionState &state = *m_state;
    VuPipelineState &pipeline = state.pipeline;
    // Walk from the next stage due through the stage used by the current
    // instruction pair. This commits retained results from oldest to newest.
    for (uint8_t offset = 1u;
         offset <= VuPipelineState::kFmacFlagStages; ++offset)
    {
        const uint8_t index = static_cast<uint8_t>(
            (pipeline.fmacFlagIndex + offset) %
            VuPipelineState::kFmacFlagStages);
        VuPipelineState::FmacFlags &pending =
            pipeline.fmacFlags[index];
        if (!pending.active)
            continue;
        commitFmacFlags(pending);
        pending = {};
    }
}

void VuInterpreterBackend::commitFmacFlags(
    const VuPipelineState::FmacFlags &pending)
{
    VuExecutionState &state = *m_state;
    state.mac = pending.mac;
    state.status =
        (state.status & 0xff0u) |
        (pending.status & 0x0fu) |
        ((pending.status & 0x0fu) << 6u);
}

void VuInterpreterBackend::scheduleFmacFlags(
    const float result[4], uint8_t dest, bool preserveUnselected)
{
    VuExecutionState &state = *m_state;
    VuPipelineState &pipeline = state.pipeline;
    uint32_t mac = preserveUnselected ? pipeline.workingMac : 0u;
    for (uint32_t lane = 0u; lane < 4u; ++lane)
    {
        const uint32_t laneMask = 0x8u >> lane;
        if ((dest & laneMask) == 0u)
            continue;

        const uint32_t shift = 3u - lane;
        const uint32_t flagsMask = 0x1111u << shift;
        mac &= ~flagsMask;

        uint32_t bits = 0u;
        std::memcpy(&bits, &result[lane], sizeof(bits));
        if ((bits & 0x80000000u) != 0u)
            mac |= 0x0010u << shift;

        const uint32_t exponent = (bits >> 23u) & 0xffu;
        if ((bits & 0x7fffffffu) == 0u)
        {
            mac |= 0x0001u << shift;
        }
        else if (exponent == 0u)
        {
            mac |= 0x0101u << shift;
        }
        else if (exponent == 0xffu)
        {
            mac |= 0x1000u << shift;
        }
    }

    uint32_t status = 0u;
    if ((mac & 0x000fu) != 0u)
        status |= 0x1u;
    if ((mac & 0x00f0u) != 0u)
        status |= 0x2u;
    if ((mac & 0x0f00u) != 0u)
        status |= 0x4u;
    if ((mac & 0xf000u) != 0u)
        status |= 0x8u;

    pipeline.workingMac = mac;
    pipeline.fmacFlags[pipeline.fmacFlagIndex] = {
        .active = true,
        .mac = mac,
        .status = status,
    };
}

void VuInterpreterBackend::backupVi(uint8_t reg)
{
    VuExecutionState &state = *m_state;
    if (reg == 0u)
        return;

    if (state.viBackupCycles != 0u && state.viBackupRegister == reg)
    {
        // Consecutive auto-increments retain the value from before the chain.
        state.viBackupCycles = 2u;
        return;
    }

    state.viBackupCycles = 2u;
    state.viBackupRegister = reg;
    state.viBackupValue = static_cast<int16_t>(state.vi[reg]);
}

int32_t VuInterpreterBackend::readViForBranch(uint8_t reg) const
{
    const VuExecutionState &state = *m_state;
    if (state.viBackupCycles != 0u && state.viBackupRegister == reg)
        return static_cast<int16_t>(state.viBackupValue);
    return static_cast<int16_t>(state.vi[reg]);
}

void VuInterpreterBackend::cancelXgkick()
{
    VuExecutionState &state = *m_state;
    state.pipeline.xgkick = {};
}

void VuInterpreterBackend::beginXgkick(uint32_t sourceQword,
                                               uint8_t *vuData,
                                               uint32_t dataSize,
                                               IVuSideEffectSink &sideEffects,
                                               PS2Memory *memory)
{
    VuExecutionState &state = *m_state;
    if (!vuData || dataSize < 16u)
        return;

    if (memory)
        memory->traceVu1Xgkick(sourceQword);

    VuPipelineState::Xgkick &xgkick = state.pipeline.xgkick;
    // A second XGKICK stalls behind and drains the prior PATH1 transfer.
    if (xgkick.active)
    {
        advanceXgkick(
            vuData, dataSize, sideEffects, memory, 0u, true);
    }

    xgkick = {};
    xgkick.active = true;
    xgkick.address = ((sourceQword & 0x3FFu) * 16u) % dataSize;
}

void VuInterpreterBackend::advanceXgkick(uint8_t *vuData,
                                         uint32_t dataSize,
                                         IVuSideEffectSink &sideEffects,
                                         PS2Memory *memory,
                                         uint32_t cycles,
                                         bool flush)
{
    VuExecutionState &state = *m_state;
    VuPipelineState::Xgkick &xgkick = state.pipeline.xgkick;
    if (!xgkick.active || !vuData || dataSize < 16u)
        return;

    constexpr size_t kMaxPath1PacketBytes = 64u * 1024u * 1024u;

    if (cycles > std::numeric_limits<uint32_t>::max() - xgkick.cycleCredit)
        xgkick.cycleCredit = std::numeric_limits<uint32_t>::max();
    else
        xgkick.cycleCredit += cycles;

    uint32_t transferQwords = flush
        ? std::numeric_limits<uint32_t>::max()
        : xgkick.cycleCredit / 2u;
    uint64_t flushScanBytes = 0u;
    bool validateFlushChain =
        flush && xgkick.tagBytesRemaining == 0u;

    auto read64Wrap = [&](uint32_t address)
    {
        uint8_t bytes[8];
        for (uint32_t i = 0; i < 8u; ++i)
            bytes[i] = vuData[(address + i) % dataSize];
        uint64_t value = 0u;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    };

    auto loadTag = [&]() -> bool
    {
        const uint64_t tagLo = read64Wrap(xgkick.address);
        const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60u) & 0xFu);
        if (nreg == 0u)
            nreg = 16u;

        uint64_t payloadBytes = 0u;
        if (flg == 0u)
        {
            payloadBytes = static_cast<uint64_t>(nloop) * nreg * 16u;
        }
        else if (flg == 1u)
        {
            payloadBytes =
                (static_cast<uint64_t>(nloop) * nreg * 8u + 15u) & ~15ull;
        }
        else
        {
            payloadBytes = static_cast<uint64_t>(nloop) * 16u;
        }

        const uint64_t tagBytes = 16u + payloadBytes;
        // A single PATH1 tag cannot occupy an entire VU1 data-memory ring.
        // PCSX2 likewise rejects a packet-size scan once it reaches 16 KiB.
        if (tagBytes >= dataSize || tagBytes > std::numeric_limits<uint32_t>::max())
            return false;
        if (validateFlushChain)
        {
            flushScanBytes += tagBytes;
            if (flushScanBytes >= dataSize)
                return false;
        }

        xgkick.tagBytesRemaining = static_cast<uint32_t>(tagBytes);
        xgkick.tagEop = ((tagLo >> 15u) & 1u) != 0u;
        return true;
    };

    while (xgkick.active)
    {
        if (xgkick.tagBytesRemaining == 0u)
        {
            if (!loadTag())
            {
                cancelXgkick();
                return;
            }
        }

        if (!flush && transferQwords == 0u)
            break;

        const uint32_t remainingQwords =
            xgkick.tagBytesRemaining / 16u;
        const uint32_t qwords = flush
            ? remainingQwords
            : std::min(remainingQwords, transferQwords);
        const uint32_t bytes = qwords * 16u;
        if (bytes == 0u ||
            xgkick.packet.size() > kMaxPath1PacketBytes - bytes)
        {
            cancelXgkick();
            return;
        }

        const size_t oldSize = xgkick.packet.size();
        xgkick.packet.resize(oldSize + bytes);
        for (uint32_t i = 0u; i < bytes; ++i)
        {
            xgkick.packet[oldSize + i] =
                vuData[(xgkick.address + i) % dataSize];
        }

        xgkick.address = (xgkick.address + bytes) % dataSize;
        xgkick.tagBytesRemaining -= bytes;
        if (!flush)
        {
            transferQwords -= qwords;
            xgkick.cycleCredit -= qwords * 2u;
        }

        if (xgkick.tagBytesRemaining != 0u)
            continue;

        if (xgkick.tagEop)
        {
            sideEffects.submitPath1Packet(
                xgkick.packet.data(),
                static_cast<uint32_t>(xgkick.packet.size()));
            cancelXgkick();
            return;
        }

        if (flush)
            validateFlushChain = true;
    }
}
