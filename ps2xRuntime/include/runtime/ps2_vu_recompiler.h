#ifndef PS2_VU_RECOMPILER_H
#define PS2_VU_RECOMPILER_H

#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu1.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

enum class VuNativeBlockExit : uint32_t
{
    BlockComplete,
    CycleBudget,
    Inactive,
    ProgramEnded,
    CodeBounds,
    XgkickBoundary,
    DebugObserver,
    CodeInvalidated,
    UnsupportedInstruction,
    Fault,
};

inline constexpr size_t kVuNativeBlockExitCount =
    static_cast<size_t>(VuNativeBlockExit::Fault) + 1u;
inline constexpr size_t kVuIrOpcodeCount =
    static_cast<size_t>(VuIrOpcode::Unsupported) + 1u;

enum class VuRegisterMaterializationCause : uint8_t
{
    CanonicalExit,
    LinkBoundary,
    PairHelper,
    XgkickHelper,
};

inline constexpr size_t
    kVuRegisterMaterializationCauseCount =
        static_cast<size_t>(
            VuRegisterMaterializationCause::
                XgkickHelper) +
        1u;

struct VuNativeBlockResult
{
    uint32_t executedCycles = 0u;
    VuNativeBlockExit exit = VuNativeBlockExit::Fault;

    [[nodiscard]] static VuNativeBlockResult decode(
        uint64_t packed)
    {
        return {
            .executedCycles =
                static_cast<uint32_t>(packed),
            .exit = static_cast<VuNativeBlockExit>(
                packed >> 32u),
        };
    }

    [[nodiscard]] uint64_t encode() const
    {
        return
            static_cast<uint64_t>(executedCycles) |
            (static_cast<uint64_t>(exit) << 32u);
    }
};

using VuNativeBlockEntry = uint64_t (*)(
    VuExecutionContext *context, uint32_t maximumCycles);

struct VuRecompilerDiagnostics
{
    uint64_t nativeEntries = 0u;
    uint64_t nativeBlocks = 0u;
    uint64_t nativePairs = 0u;
    uint64_t instrumentedNativeEntries = 0u;
    uint64_t instrumentedNativeBlocks = 0u;
    uint64_t instrumentedNativePairs = 0u;
    uint64_t inlinePairs = 0u;
    uint64_t helperPairs = 0u;
    uint64_t linkedEdges = 0u;
    uint64_t slowLinkExits = 0u;
    uint64_t resolvedLinks = 0u;
    uint64_t abandonedLinkResolutions = 0u;
    uint64_t incompatibleLinkExits = 0u;
    uint64_t fullBlockGuards = 0u;
    uint64_t preciseTailEntries = 0u;
    uint64_t fullGuardPairs = 0u;
    uint64_t preciseTailPairs = 0u;
    uint64_t budgetGuardComparisons = 0u;
    uint64_t nativeBudgetExits = 0u;
    uint64_t blockCompletes = 0u;
    uint64_t cycleBudgetExits = 0u;
    uint64_t xgkickExits = 0u;
    uint64_t xgkickAdvanceHelperCalls = 0u;
    uint64_t unsupportedExits = 0u;
    uint64_t codeInvalidationExits = 0u;
    uint64_t faultExits = 0u;
    std::array<
        uint64_t,
        kVuNativeBlockExitCount> nativeExitReasons{};
    uint64_t interpreterInstrumentationFallbacks = 0u;
    uint64_t interpreterFallbackPairs = 0u;
    uint64_t codeImageIdentities = 0u;
    uint64_t codeImageReuses = 0u;
    uint64_t codeImageCatalogEvictions = 0u;
    uint64_t jitDumpRegistrations = 0u;
    uint64_t jitDumpFailures = 0u;
};

struct VuNativeChainRuntime
{
    uint64_t codeGeneration = 0u;
    const VuNativeLinkState *slowLinkSource = nullptr;
    uint32_t slowLinkTargetPc = 0u;
    uint32_t blockStartCycles = 0u;
    uint64_t linkedEdges = 0u;
    uint64_t preciseTailEntries = 0u;
    uint64_t preciseTailPairs = 0u;
    uint64_t preciseNonRetiredChecks = 0u;
    uint64_t returnBoundaryChecks = 0u;
};

enum class VuRecompilerOpcodeDisposition : uint8_t
{
    NativeInline,
    NativeHelper,
    InterpreterSideExit,
};

struct VuBlockJitRegistrationSnapshot
{
    uint64_t generation = 0u;
    uint64_t codeIndex = 0u;
};

struct VuBlockProfileSnapshot
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
    bool resident = false;
    uint64_t executions = 0u;
    uint64_t guestPairs = 0u;
    uint64_t fullBudgetEntries = 0u;
    uint64_t boundedEntries = 0u;
    uint64_t linkedEdges = 0u;
    uint64_t helperBarriers = 0u;
    uint32_t residentVfRegisters = 0u;
    uint32_t residentVfDirtyRegisters = 0u;
    uint32_t residentVfDirtyLanes = 0u;
    uint32_t maximumLiveVfRegisters = 0u;
    uint32_t vfAccesses = 0u;
    uint32_t allocatedVfAccesses = 0u;
    uint64_t vfRegisterLoads = 0u;
    uint64_t vfRegisterStores = 0u;
    uint64_t vfRegisterSpills = 0u;
    uint64_t vfRegisterReloads = 0u;
    std::array<
        uint64_t,
        kVuRegisterMaterializationCauseCount>
        registerMaterializations{};
    std::array<
        uint64_t,
        kVuNativeBlockExitCount> exitReasons{};
    std::array<
        uint64_t,
        kVuIrOpcodeCount> opcodeOperations{};
    std::vector<VuBlockJitRegistrationSnapshot>
        jitRegistrations;
};

[[nodiscard]] std::string_view
vuRegisterMaterializationCauseName(
    VuRegisterMaterializationCause cause);

struct VuBlockProfilingSnapshot
{
    bool enabled = false;
    uint64_t maximumRecords = 0u;
    uint64_t droppedRecords = 0u;
    std::vector<VuBlockProfileSnapshot> blocks;
};

struct VuBlockProfilingConfiguration
{
    bool enabled = false;
    size_t maximumRecords = 16384u;
};

class VuRecompilerBackend final : public IVuExecutionBackend
{
public:
    explicit VuRecompilerBackend(VuUnit &unit);
    VuRecompilerBackend(
        VuUnit &unit,
        VuBlockProfilingConfiguration configuration);

    [[nodiscard]] VuRunResult run(
        VuExecutionContext &context,
        uint32_t maximumCycles) override;
    [[nodiscard]] std::string_view name() const override;

    [[nodiscard]] static bool built();
    [[nodiscard]] static bool supported();
    [[nodiscard]] static uint64_t hostFeatures();
    [[nodiscard]] static bool canInlineUpperOpcode(
        VuIrOpcode opcode, uint64_t hostFeatures);
    [[nodiscard]] static bool canInlineLowerOpcode(
        VuIrOpcode opcode);
    [[nodiscard]] static VuRecompilerOpcodeDisposition
    opcodeDisposition(
        VuIrOpcode opcode, uint64_t hostFeatures);

    [[nodiscard]] bool programKey(
        const VuExecutionContext &context,
        VuCompilationMode mode,
        VuProgramKey &key,
        std::string *diagnostic = nullptr);

    [[nodiscard]] VuRecompilerDiagnostics diagnostics() const
    {
        return m_diagnostics;
    }
    [[nodiscard]] const std::string &lastDiagnostic() const
    {
        return m_lastDiagnostic;
    }
    [[nodiscard]] const std::string &
    lastJitDiagnostic() const
    {
        return m_lastJitDiagnostic;
    }
    [[nodiscard]] bool blockLinkingEnabled() const
    {
        return m_blockLinkingEnabled;
    }
    [[nodiscard]] bool blockBudgetGuardsEnabled() const
    {
        return m_blockBudgetGuardsEnabled;
    }
    [[nodiscard]] bool blockLocalVfRegistersEnabled() const
    {
        return m_blockLocalVfRegistersEnabled;
    }
    [[nodiscard]] bool
    blockLocalVfRegistersAutomatic() const
    {
        return m_blockLocalVfRegistersAutomatic;
    }
    [[nodiscard]] bool inlineXgkickEnabled() const
    {
        return m_inlineXgkickEnabled;
    }
    // The caller must prove that VU execution is quiescent while copying or
    // resetting these single-owner-thread records.
    [[nodiscard]] VuBlockProfilingSnapshot
    blockProfilingSnapshotWhileExecutionQuiescent() const;
    void resetBlockProfilingCountersWhileExecutionQuiescent();

private:
    friend class VuUnit;

    static constexpr uint32_t kMaximumBlockPairs = 64u;
    static constexpr size_t kMaximumDispatchEntries =
        0x4000u / 8u;
    static constexpr size_t kMaximumRememberedCodeImages = 64u;

    struct CodeImageIdentity
    {
        uintptr_t memoryIdentity = 0u;
        uintptr_t codeIdentity = 0u;
        uint32_t codeSize = 0u;
        uint64_t hash = 0u;
        uint64_t identity = 0u;
        std::vector<uint8_t> bytes;
    };

    [[nodiscard]] uint32_t executePair(
        VuExecutionContext &context, uint32_t expectedPc,
        uint32_t lowerWord, uint32_t upperWord) noexcept;
    [[nodiscard]] static uint32_t executePairThunk(
        VuRecompilerBackend *backend,
        VuExecutionContext *context,
        uint32_t expectedPc,
        uint64_t instructionWords) noexcept;
    [[nodiscard]] uint32_t executeInstrumentedPair(
        VuExecutionContext &context, uint32_t expectedPc,
        uint32_t lowerWord, uint32_t upperWord) noexcept;
    [[nodiscard]] static uint32_t executeInstrumentedPairThunk(
        VuRecompilerBackend *backend,
        VuExecutionContext *context,
        uint32_t expectedPc,
        uint64_t instructionWords) noexcept;
    [[nodiscard]] uint32_t advanceXgkick(
        VuExecutionContext &context) noexcept;
    [[nodiscard]] static uint32_t advanceXgkickThunk(
        VuRecompilerBackend *backend,
        VuExecutionContext *context) noexcept;
    [[nodiscard]] VuProgramHandle lookupOrCompile(
        VuExecutionContext &context,
        const VuProgramKey &key);
    void ensureJitRegistration(
        const VuCompiledProgram &program);
    [[nodiscard]] uint64_t codeContentIdentity(
        const VuExecutionContext &context,
        uint64_t generation);
    [[nodiscard]] bool compile(
        VuExecutionContext &context,
        const VuProgramKey &key,
        VuCompiledProgram &program,
        std::string &diagnostic);
    [[nodiscard]] bool usesNativeInstrumentation(
        const VuExecutionContext &context) const;
    [[nodiscard]] bool needsInterpreterInstrumentation(
        const VuExecutionContext &context,
        bool nativeInstrumentation) const;
    [[nodiscard]] VuRunResult runInterpreterFallback(
        VuExecutionContext &context,
        uint32_t maximumCycles);
    [[nodiscard]] static VuExitReason publicExit(
        VuNativeBlockExit exit);

    VuUnit &m_unit;
    VuInterpreterBackend m_semantics;
    VuRecompilerDiagnostics m_diagnostics{};
    std::array<
        VuProgramHandle,
        kMaximumDispatchEntries> m_dispatchHandles{};
    std::vector<CodeImageIdentity> m_codeImages;
    uintptr_t m_currentCodeMemoryIdentity = 0u;
    uintptr_t m_currentCodeIdentity = 0u;
    uint32_t m_currentCodeSize = 0u;
    uint64_t m_currentCodeGeneration = 0u;
    uint64_t m_currentCodeContentIdentity = 0u;
    uint64_t m_nextCodeContentIdentity = 1u;
    uint64_t m_helperPairsIssued = 0u;
    uint64_t m_instrumentedPairIndex = 0u;
    VuNativeChainRuntime m_chainRuntime{};
    bool m_blockLinkingEnabled = true;
    bool m_blockBudgetGuardsEnabled = true;
    bool m_blockLocalVfRegistersEnabled = true;
    bool m_blockLocalVfRegistersAutomatic = true;
    bool m_inlineXgkickEnabled = false;
    bool m_blockProfilingEnabled = false;
    size_t m_maximumBlockProfiles = 0u;
    uint64_t m_droppedBlockProfiles = 0u;
    bool m_perfJitDumpEnabled = false;
    std::vector<
        std::shared_ptr<VuBlockProfileRecord>>
        m_blockProfiles;
    std::string m_lastDiagnostic;
    std::string m_lastJitDiagnostic;
};

[[nodiscard]] std::string_view vuNativeBlockExitName(
    VuNativeBlockExit exit);
[[nodiscard]] std::string vuPerfJitSymbolName(
    const VuProgramKey &key,
    const VuIrBlock &block,
    uint64_t compilationIdentity);

#endif
