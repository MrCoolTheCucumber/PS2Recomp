#include "runtime/ps2_vu_recompiler.h"

#include "runtime/ps2_memory.h"
#include "ps2_vu_float_mode.h"

#include <chrono>
#include <cstring>
#include <exception>
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
    constexpr size_t kEmitterCapacity = 64u * 1024u;

    class X64VuBlockEmitter
    {
    public:
        X64VuBlockEmitter(
            const VuIrBlock &block,
            VuRecompilerBackend *backend,
            const void *pairHelper)
            : m_buffer(kEmitterCapacity),
              m_code(m_buffer.size(), m_buffer.data())
        {
            emit(block, backend, pairHelper);
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
        std::vector<uint8_t> m_buffer;
        Xbyak::CodeGenerator m_code;

        void emit(
            const VuIrBlock &block,
            VuRecompilerBackend *backend,
            const void *pairHelper)
        {
            using namespace Xbyak;

            Label cycleBudgetExit;
            Label dynamicExit;
            Label packResult;
            Label epilogue;

#if defined(_WIN32)
            constexpr int stackBytes = 56;
            constexpr int savedMxcsrOffset = 32;
            constexpr int vuMxcsrOffset = 36;
#else
            constexpr int stackBytes = 24;
            constexpr int savedMxcsrOffset = 0;
            constexpr int vuMxcsrOffset = 4;
#endif

            m_code.push(m_code.r12);
            m_code.push(m_code.r13);
            m_code.push(m_code.r14);
            m_code.push(m_code.r15);
            m_code.sub(m_code.rsp, stackBytes);

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
                    m_code.rsp + savedMxcsrOffset]);
            m_code.mov(
                m_code.eax,
                m_code.dword[
                    m_code.rsp + savedMxcsrOffset]);
            m_code.and_(
                m_code.eax,
                static_cast<uint32_t>(
                    ~kMxcsrRoundingMask));
            m_code.or_(m_code.eax, kVuMxcsrBits);
            m_code.mov(
                m_code.dword[
                    m_code.rsp + vuMxcsrOffset],
                m_code.eax);
            m_code.ldmxcsr(
                m_code.dword[
                    m_code.rsp + vuMxcsrOffset]);

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

            for (const VuIrInstructionPair &pair :
                 block.pairs)
            {
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
                Label pairNotExecuted;
                m_code.jz(pairNotExecuted);
                m_code.inc(m_code.r14d);
                m_code.L(pairNotExecuted);
                m_code.and_(
                    m_code.r11d,
                    kNativePairExitMask);
                m_code.test(m_code.r11d, m_code.r11d);
                m_code.jnz(dynamicExit, m_code.T_NEAR);
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

            m_code.L(dynamicExit);
            m_code.mov(m_code.r10d, m_code.r11d);

            m_code.L(packResult);
            m_code.mov(m_code.eax, m_code.r14d);
            m_code.mov(m_code.r11d, m_code.r10d);
            m_code.shl(m_code.r11, 32u);
            m_code.or_(m_code.rax, m_code.r11);
            m_code.jmp(epilogue);

            m_code.L(epilogue);
            m_code.ldmxcsr(
                m_code.dword[
                    m_code.rsp + savedMxcsrOffset]);
            m_code.add(m_code.rsp, stackBytes);
            m_code.pop(m_code.r15);
            m_code.pop(m_code.r14);
            m_code.pop(m_code.r13);
            m_code.pop(m_code.r12);
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

        const VuProgramHandle handle =
            lookupOrCompile(context, key);
        const VuCompiledProgram *const program =
            m_unit.programCache().resolve(handle);
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
        const VuNativeBlockResult native =
            VuNativeBlockResult::decode(
                entry(&context, remaining));
        ++m_diagnostics.nativeEntries;
        if (native.executedCycles > remaining)
        {
            m_lastDiagnostic =
                "native VU block exceeded its cycle budget";
            ++m_diagnostics.faultExits;
            return finish(totalExecuted, VuExitReason::Fault);
        }

        totalExecuted += native.executedCycles;
        state.issuedCycles += native.executedCycles;
        m_diagnostics.nativePairs +=
            native.executedCycles;

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

        return kNativePairExecuted |
               static_cast<uint32_t>(exit);
    }
    catch (...)
    {
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
        key.entryPc, kMaximumBlockPairs);
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
                    executePairThunk));
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
