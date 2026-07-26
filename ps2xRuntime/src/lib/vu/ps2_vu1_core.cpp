#include "runtime/ps2_vu1.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <limits>
#include <utility>

#if defined(__SSE__)
#include <xmmintrin.h>
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

    class ScopedVuFloatMode
    {
    public:
        ScopedVuFloatMode()
            : m_previous(std::fegetround())
        {
#if defined(__SSE__)
            m_previousMxcsr = _mm_getcsr();
#endif
            if (m_previous != FE_TOWARDZERO)
                m_changed = std::fesetround(FE_TOWARDZERO) == 0;

#if defined(__SSE__)
            // The VU treats denormal inputs as zero and clamps exponent
            // underflow to signed zero. DAZ and FTZ provide those semantics
            // for the host scalar floating-point operations used below.
            constexpr uint32_t kDenormalsAreZero = 1u << 6u;
            constexpr uint32_t kFlushToZero = 1u << 15u;
            _mm_setcsr(_mm_getcsr() | kDenormalsAreZero | kFlushToZero);
#endif
        }

        ~ScopedVuFloatMode()
        {
            if (m_changed && m_previous != -1)
                std::fesetround(m_previous);
#if defined(__SSE__)
            _mm_setcsr(m_previousMxcsr);
#endif
        }

        ScopedVuFloatMode(const ScopedVuFloatMode &) = delete;
        ScopedVuFloatMode &operator=(const ScopedVuFloatMode &) = delete;

    private:
        int m_previous = -1;
        bool m_changed = false;
#if defined(__SSE__)
        uint32_t m_previousMxcsr = 0u;
#endif
    };
}

VU1Interpreter::VU1Interpreter()
{
    reset();
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f; // VF0.w = 1.0
    m_state.q = 1.0f;
    m_active = false;
    m_xgkick = {};
    m_qPipeline = {};
    m_fmacFlagPipeline.clear();
    m_workingMac = 0u;
    m_progressActive.store(0u, std::memory_order_relaxed);
    m_progressInvocations.store(0u, std::memory_order_relaxed);
    m_progressCycles.store(0u, std::memory_order_relaxed);
    m_progressPc.store(0u, std::memory_order_relaxed);
}

VU1ProgressSnapshot VU1Interpreter::getProgressSnapshot() const
{
    VU1ProgressSnapshot snapshot{};
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

void VU1Interpreter::setProgressTrackingEnabled(bool enabled)
{
    m_progressTrackingEnabled.store(enabled, std::memory_order_release);
}

void VU1Interpreter::setInstructionObserver(InstructionObserver observer)
{
    m_instructionObserver = std::move(observer);
    m_instructionObserverEnabled.store(
        static_cast<bool>(m_instructionObserver), std::memory_order_release);
}

void VU1Interpreter::setInstructionObserverEnabled(bool enabled)
{
    m_instructionObserverEnabled.store(enabled, std::memory_order_release);
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return vf[bc & 3];
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
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

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));

    decoded.iBit = ((decoded.upper >> 31) & 1u) != 0u;
    decoded.eBit = ((decoded.upper >> 30) & 1u) != 0u;
    decoded.lowerBeforeUpper = !decoded.iBit && vuLowerShouldRunBeforeUpper(decoded.upper, decoded.lower);
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = codeSize / 8u;
    m_decodedCodeCache.resize(pairCount);
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);
    }

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::getDecodedInstructionPairForPc(const uint8_t *vuCode,
                                                                                      uint32_t codeSize,
                                                                                      PS2Memory *memory,
                                                                                      uint32_t pc)
{
    // Only 8-byte aligned VU instruction pairs can use the decode cache.
    if ((pc & 7u) != 0u)
    {
        return decodeInstructionPair(vuCode, pc);
    }

    if (!memory)
    {
        return decodeInstructionPair(vuCode, pc);
    }

    uint64_t generation = 0u;
    if (vuCode == memory->getVU0Code())
    {
        generation = memory->getVU0CodeGeneration();
    }
    else if (vuCode == memory->getVU1Code())
    {
        generation = memory->getVU1CodeGeneration();
    }
    else
    {
        return decodeInstructionPair(vuCode, pc);
    }

    const bool rebuild =
        !m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation;

    if (rebuild)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }

    return m_decodedCodeCache[pc / 8u];
}

void VU1Interpreter::start(
    uint32_t startPC, uint32_t top, uint32_t itop,
    PS2Memory *memory)
{
    m_state.pc = startPC & 0x3FFFu;
    m_state.ebit = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.viBackupCycles = 0;
    m_fmacFlagPipeline.clear();
    m_workingMac = m_state.mac;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    m_active = true;
    if (memory && memory->isVif1DmaTraceActive())
        memory->traceVu1Invocation(m_state.pc, top, itop, false, m_state);
}

void VU1Interpreter::resumeState(
    uint32_t top, uint32_t itop, PS2Memory *memory)
{
    m_state.ebit = false;
    m_state.top = top;
    m_state.itop = itop;
    m_active = true;
    if (memory && memory->isVif1DmaTraceActive())
        memory->traceVu1Invocation(m_state.pc, top, itop, true, m_state);
}

VU1AdvanceResult VU1Interpreter::advance(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    if (!m_active)
    {
        return VU1AdvanceResult{
            .requestedCycles = maxCycles,
            .activeBefore = false,
            .activeAfter = false,
        };
    }

    return run(
        vuCode, codeSize, vuData, dataSize,
        gs, memory, maxCycles, false);
}

void VU1Interpreter::execute(
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

void VU1Interpreter::resume(
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

VU1AdvanceResult VU1Interpreter::continueExecution(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    if (!m_active)
    {
        return VU1AdvanceResult{
            .requestedCycles = maxCycles,
            .activeBefore = false,
            .activeAfter = false,
        };
    }
    return run(
        vuCode, codeSize, vuData, dataSize,
        gs, memory, maxCycles, true);
}

VU1AdvanceResult VU1Interpreter::run(
    uint8_t *vuCode, uint32_t codeSize,
    uint8_t *vuData, uint32_t dataSize,
    GS &gs, PS2Memory *memory, uint32_t maxCycles,
    bool traceBudgetBoundary)
{
    // The VU performs 24-bit floating-point calculations by truncating toward
    // zero. Keep that host mode local to VU execution so callers retain their
    // own floating-point environment.
    const ScopedVuFloatMode vuFloatMode;
    const bool traceVu1 = memory && memory->isVif1DmaTraceActive();
    const bool trackProgress =
        m_progressTrackingEnabled.load(std::memory_order_relaxed);
    uint32_t committedProgressCycles = 0u;
    if (trackProgress)
    {
        m_progressActive.fetch_add(1u, std::memory_order_relaxed);
        m_progressInvocations.fetch_add(1u, std::memory_order_relaxed);
        m_progressPc.store(m_state.pc, std::memory_order_relaxed);
    }
    struct ProgressGuard
    {
        VU1Interpreter &interpreter;
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
                interpreter.m_progressCycles.fetch_add(
                    executed - committed, std::memory_order_relaxed);
            }
            interpreter.m_progressPc.store(
                interpreter.m_state.pc, std::memory_order_relaxed);
            interpreter.m_progressActive.fetch_sub(
                1u, std::memory_order_relaxed);
        }
    };
    uint32_t executedCycles = 0u;
    ProgressGuard progress{
        *this, trackProgress, executedCycles, committedProgressCycles};
    bool ended = false;

    for (uint32_t cycle = 0; cycle < maxCycles; ++cycle)
    {
        if (m_state.pc + 8 > codeSize)
        {
            m_active = false;
            break;
        }

        const DecodedInstructionPair decoded = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        if (traceVu1)
            memory->traceVu1Instruction(m_state.pc, decoded.lower, decoded.upper, m_state);
        if (m_instructionObserverEnabled.load(std::memory_order_relaxed) &&
            m_instructionObserver)
        {
            m_instructionObserver(
                executedCycles, m_state.pc, decoded.lower, decoded.upper,
                m_state);
        }
        ++executedCycles;
        if (trackProgress &&
            executedCycles - committedProgressCycles >= 256u)
        {
            m_progressCycles.fetch_add(
                executedCycles - committedProgressCycles,
                std::memory_order_relaxed);
            committedProgressCycles = executedCycles;
            m_progressPc.store(m_state.pc, std::memory_order_relaxed);
        }

        // DIV/SQRT results are written to Q seven cycles after issue
        // (RSQRT takes thirteen). The instruction observer deliberately runs
        // first because the architectural Q write occurs after the current
        // pair has been fetched. A new FDIV operation or WAITQ stalls until
        // the previous result is available, and its paired upper instruction
        // can consume that result.
        advanceQPipeline();
        advanceFmacFlagPipeline();
        if (!decoded.iBit && vuLowerWaitsForQ(decoded.lower))
            flushQPipeline();

        // Integer writes such as MTIR and auto-incrementing loads/stores expose
        // the pre-update VI value to a branch in the immediately following
        // instruction pair. PCSX2 models this as a two-cycle backup which is
        // aged before each subsequent pair.
        if (m_state.viBackupCycles != 0u)
            --m_state.viBackupCycles;

        // LOI is controlled by the upper I-bit.  The lower word is the float immediate.
        // DobieStation executes the upper instruction first, then commits lower into I.
        if (decoded.iBit)
        {
            // LOI is special: the upper instruction sees the old I value, then LOI loads I.
            execUpper(decoded.upper);
            std::memcpy(&m_state.i, &decoded.lower, sizeof(decoded.lower));
        }
        else if (decoded.lowerBeforeUpper)
        {
            // VU upper/lower execute as a pair.  If the upper op writes a VF register
            // that the lower op reads or also writes, Dobie runs the lower side first
            // so it observes the old VF value and the upper write has priority.
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            execUpper(decoded.upper);
        }
        else
        {
            execUpper(decoded.upper);
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        // Enforce VF0 invariant
        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        // Enforce VI0 invariant
        m_state.vi[0] = 0;

        // PATH1 consumes one quadword per two VU cycles while the microprogram
        // continues to run and may keep producing data in the same ring buffer.
        advanceXgkick(vuData, dataSize, gs, memory, 1u, false);

        uint32_t nextPC = m_state.pc + 8;
        if (nextPC >= codeSize)
            nextPC = 0;
        m_state.pc = nextPC;

        // VU branch/jump has a delay slot. Branch handlers set a pending target;
        // we execute one sequential instruction before committing the branch.
        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0)
            {
                m_state.pc = m_state.branchTarget & 0x3FFFu;
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        if (m_state.ebit)
        {
            // VU1 completion drains the pending XGKICK before becoming idle.
            advanceXgkick(vuData, dataSize, gs, memory, 0u, true);
            flushQPipeline();
            flushFmacFlagPipeline();
            m_state.viBackupCycles = 0u;
            ended = true;
            m_active = false;
            break;
        }

        if (decoded.eBit)
            m_state.ebit = true;
    }

    if (traceVu1 &&
        (!m_active || traceBudgetBoundary))
    {
        memory->traceVu1InvocationEnd(
            m_state.pc, ended, !ended && executedCycles == maxCycles,
            m_state.vi, std::size(m_state.vi));
    }
    return VU1AdvanceResult{
        .requestedCycles = maxCycles,
        .executedCycles = executedCycles,
        .activeBefore = true,
        .activeAfter = m_active,
        .completed = !m_active,
    };
}

void VU1Interpreter::advanceQPipeline()
{
    if (!m_qPipeline.active)
        return;

    if (m_qPipeline.cyclesRemaining != 0u)
        --m_qPipeline.cyclesRemaining;
    if (m_qPipeline.cyclesRemaining == 0u)
        flushQPipeline();
}

void VU1Interpreter::flushQPipeline()
{
    if (!m_qPipeline.active)
        return;

    m_state.q = m_qPipeline.result;
    m_qPipeline = {};
}

void VU1Interpreter::scheduleQ(float result, uint32_t latency)
{
    m_qPipeline.active = true;
    m_qPipeline.result = result;
    m_qPipeline.cyclesRemaining = latency;
}

void VU1Interpreter::advanceFmacFlagPipeline()
{
    for (FmacFlagPipelineState &pending : m_fmacFlagPipeline)
    {
        if (pending.cyclesRemaining != 0u)
            --pending.cyclesRemaining;
    }

    while (!m_fmacFlagPipeline.empty() &&
           m_fmacFlagPipeline.front().cyclesRemaining == 0u)
    {
        const FmacFlagPipelineState pending =
            m_fmacFlagPipeline.front();
        m_fmacFlagPipeline.erase(m_fmacFlagPipeline.begin());
        m_state.mac = pending.mac;
        m_state.status =
            (m_state.status & 0xff0u) |
            (pending.status & 0x0fu) |
            ((pending.status & 0x0fu) << 6u);
    }
}

void VU1Interpreter::flushFmacFlagPipeline()
{
    for (FmacFlagPipelineState &pending : m_fmacFlagPipeline)
        pending.cyclesRemaining = 0u;
    advanceFmacFlagPipeline();
}

void VU1Interpreter::scheduleFmacFlags(
    const float result[4], uint8_t dest, bool preserveUnselected)
{
    uint32_t mac = preserveUnselected ? m_workingMac : 0u;
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

    m_workingMac = mac;
    m_fmacFlagPipeline.push_back({mac, status, 4u});
}

void VU1Interpreter::backupVi(uint8_t reg)
{
    if (reg == 0u)
        return;

    if (m_state.viBackupCycles != 0u && m_state.viBackupRegister == reg)
    {
        // Consecutive auto-increments retain the value from before the chain.
        m_state.viBackupCycles = 2u;
        return;
    }

    m_state.viBackupCycles = 2u;
    m_state.viBackupRegister = reg;
    m_state.viBackupValue = static_cast<int16_t>(m_state.vi[reg]);
}

int32_t VU1Interpreter::readViForBranch(uint8_t reg) const
{
    if (m_state.viBackupCycles != 0u && m_state.viBackupRegister == reg)
        return static_cast<int16_t>(m_state.viBackupValue);
    return static_cast<int16_t>(m_state.vi[reg]);
}

void VU1Interpreter::cancelXgkick()
{
    m_xgkick = {};
}

void VU1Interpreter::beginXgkick(uint32_t sourceQword,
                                 uint8_t *vuData,
                                 uint32_t dataSize,
                                 GS &gs,
                                 PS2Memory *memory)
{
    if (!vuData || dataSize < 16u)
        return;

    if (memory)
        memory->traceVu1Xgkick(sourceQword);

    // A second XGKICK stalls behind and drains the prior PATH1 transfer.
    if (m_xgkick.active)
        advanceXgkick(vuData, dataSize, gs, memory, 0u, true);

    m_xgkick = {};
    m_xgkick.active = true;
    m_xgkick.address = ((sourceQword & 0x3FFu) * 16u) % dataSize;
}

void VU1Interpreter::advanceXgkick(uint8_t *vuData,
                                   uint32_t dataSize,
                                   GS &gs,
                                   PS2Memory *memory,
                                   uint32_t cycles,
                                   bool flush)
{
    if (!m_xgkick.active || !vuData || dataSize < 16u)
        return;

    constexpr size_t kMaxPath1PacketBytes = 64u * 1024u * 1024u;

    if (cycles > std::numeric_limits<uint32_t>::max() - m_xgkick.cycleCredit)
        m_xgkick.cycleCredit = std::numeric_limits<uint32_t>::max();
    else
        m_xgkick.cycleCredit += cycles;

    uint32_t transferQwords = flush
        ? std::numeric_limits<uint32_t>::max()
        : m_xgkick.cycleCredit / 2u;
    uint64_t flushScanBytes = 0u;
    bool validateFlushChain = flush && m_xgkick.tagBytesRemaining == 0u;

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
        const uint64_t tagLo = read64Wrap(m_xgkick.address);
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

        m_xgkick.tagBytesRemaining = static_cast<uint32_t>(tagBytes);
        m_xgkick.tagEop = ((tagLo >> 15u) & 1u) != 0u;
        return true;
    };

    while (m_xgkick.active)
    {
        if (m_xgkick.tagBytesRemaining == 0u)
        {
            if (!loadTag())
            {
                cancelXgkick();
                return;
            }
        }

        if (!flush && transferQwords == 0u)
            break;

        const uint32_t remainingQwords = m_xgkick.tagBytesRemaining / 16u;
        const uint32_t qwords = flush
            ? remainingQwords
            : std::min(remainingQwords, transferQwords);
        const uint32_t bytes = qwords * 16u;
        if (bytes == 0u ||
            m_xgkick.packet.size() > kMaxPath1PacketBytes - bytes)
        {
            cancelXgkick();
            return;
        }

        const size_t oldSize = m_xgkick.packet.size();
        m_xgkick.packet.resize(oldSize + bytes);
        for (uint32_t i = 0u; i < bytes; ++i)
        {
            m_xgkick.packet[oldSize + i] =
                vuData[(m_xgkick.address + i) % dataSize];
        }

        m_xgkick.address = (m_xgkick.address + bytes) % dataSize;
        m_xgkick.tagBytesRemaining -= bytes;
        if (!flush)
        {
            transferQwords -= qwords;
            m_xgkick.cycleCredit -= qwords * 2u;
        }

        if (m_xgkick.tagBytesRemaining != 0u)
            continue;

        if (m_xgkick.tagEop)
        {
            if (memory)
            {
                memory->submitGifPacket(
                    GifPathId::Path1,
                    m_xgkick.packet.data(),
                    static_cast<uint32_t>(m_xgkick.packet.size()));
            }
            else
            {
                gs.processGIFPacket(
                    m_xgkick.packet.data(),
                    static_cast<uint32_t>(m_xgkick.packet.size()));
            }
            cancelXgkick();
            return;
        }

        if (flush)
            validateFlushChain = true;
    }
}
