#ifndef PS2_VU_ANALYSIS_H
#define PS2_VU_ANALYSIS_H

#include "runtime/ps2_vu_ir.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

enum VuAnalysisScalarResource : uint32_t
{
    VuAnalysisScalarNone = 0u,
    VuAnalysisScalarQ = 1u << 0u,
    VuAnalysisScalarP = 1u << 1u,
    VuAnalysisScalarI = 1u << 2u,
    VuAnalysisScalarR = 1u << 3u,
    VuAnalysisScalarMac = 1u << 4u,
    VuAnalysisScalarClip = 1u << 5u,
    VuAnalysisScalarStatus = 1u << 6u,
    VuAnalysisScalarTop = 1u << 7u,
    VuAnalysisScalarItop = 1u << 8u,
};

enum VuAnalysisControlResource : uint16_t
{
    VuAnalysisControlNone = 0u,
    VuAnalysisControlPc = 1u << 0u,
    VuAnalysisControlEbit = 1u << 1u,
    VuAnalysisControlActive = 1u << 2u,
    VuAnalysisControlBranchPending = 1u << 3u,
    VuAnalysisControlBranchTarget = 1u << 4u,
    VuAnalysisControlBranchDelay = 1u << 5u,
    VuAnalysisControlIssuedCycles = 1u << 6u,
};

enum VuAnalysisPipelineResource : uint16_t
{
    VuAnalysisPipelineNone = 0u,
    VuAnalysisPipelineDelayedQ = 1u << 0u,
    VuAnalysisPipelineDelayedP = 1u << 1u,
    VuAnalysisPipelineFmacFlags = 1u << 2u,
    VuAnalysisPipelineFmacIndex = 1u << 3u,
    VuAnalysisPipelineWorkingMac = 1u << 4u,
    VuAnalysisPipelineViBackup = 1u << 5u,
    VuAnalysisPipelineXgkick = 1u << 6u,
};

enum VuAnalysisMemoryResource : uint8_t
{
    VuAnalysisMemoryNone = 0u,
    VuAnalysisMemoryVuData = 1u << 0u,
    VuAnalysisMemoryVuCode = 1u << 1u,
    VuAnalysisMemoryExternal = 1u << 2u,
};

enum VuAnalysisSideEffectResource : uint8_t
{
    VuAnalysisSideEffectNone = 0u,
    VuAnalysisSideEffectPath1 = 1u << 0u,
    VuAnalysisSideEffectDebug = 1u << 1u,
    VuAnalysisSideEffectVerification = 1u << 2u,
    VuAnalysisSideEffectCodeInvalidation = 1u << 3u,
    VuAnalysisSideEffectUnknown = 1u << 4u,
};

// VF component masks use the VU destination-bit convention retained in the
// pair IR. A set can therefore distinguish a partial vector definition from
// a whole-register definition.
struct VuAnalysisResourceSet
{
    std::array<uint8_t, 32u> vf{};
    uint16_t vi = 0u;
    uint8_t acc = 0u;
    uint32_t scalars = VuAnalysisScalarNone;
    uint16_t control = VuAnalysisControlNone;
    uint16_t pipelines = VuAnalysisPipelineNone;
    uint8_t memory = VuAnalysisMemoryNone;
    uint8_t sideEffects = VuAnalysisSideEffectNone;

    bool operator==(const VuAnalysisResourceSet &) const = default;
};

void vuAnalysisUnionInto(
    VuAnalysisResourceSet &destination,
    const VuAnalysisResourceSet &source);
void vuAnalysisSubtractFrom(
    VuAnalysisResourceSet &destination,
    const VuAnalysisResourceSet &source);
[[nodiscard]] bool vuAnalysisResourcesEmpty(
    const VuAnalysisResourceSet &resources);
[[nodiscard]] VuAnalysisResourceSet
vuAnalysisAllArchitecturalResources();
[[nodiscard]] VuAnalysisResourceSet
vuAnalysisAllGuestResources();
[[nodiscard]] std::string vuAnalysisFormatResources(
    const VuAnalysisResourceSet &resources);

enum class VuAnalysisBarrierKind : uint8_t
{
    NativeHelper,
    XgkickStart,
    XgkickAdvance,
    DebugObserver,
    Verification,
    UnknownMemory,
    UnsupportedOperation,
};

enum class VuAnalysisBarrierCondition : uint8_t
{
    Always,
    IfXgkickActive,
};

enum class VuAnalysisBarrierPhase : uint8_t
{
    BeforePair,
    PairExecution,
    AfterPair,
    WholePair,
};

struct VuAnalysisBarrier
{
    VuAnalysisBarrierKind kind =
        VuAnalysisBarrierKind::NativeHelper;
    VuAnalysisBarrierPhase phase =
        VuAnalysisBarrierPhase::PairExecution;
    VuAnalysisBarrierCondition condition =
        VuAnalysisBarrierCondition::Always;
    VuAnalysisResourceSet reads;
    VuAnalysisResourceSet clobbers;
};

enum class VuAnalysisKnownState : uint8_t
{
    Dynamic,
    Inactive,
    Active,
};

struct VuAnalysisPipelineFacts
{
    bool vf0Constant = true;
    bool vi0Constant = true;
    VuAnalysisKnownState delayedQ =
        VuAnalysisKnownState::Dynamic;
    VuAnalysisKnownState delayedP =
        VuAnalysisKnownState::Dynamic;
    VuAnalysisKnownState viBackup =
        VuAnalysisKnownState::Dynamic;
    VuAnalysisKnownState advancingFmacSlot =
        VuAnalysisKnownState::Dynamic;
    uint32_t qCyclesRemaining = 0u;
    uint32_t pCyclesRemaining = 0u;
    uint8_t viBackupCycles = 0u;

    bool operator==(const VuAnalysisPipelineFacts &) const = default;
};

struct VuAnalysisPair
{
    VuIrInstructionPair ir;
    bool retires = true;
    bool usesNativeHelper = false;
    VuAnalysisResourceSet reads;
    VuAnalysisResourceSet writes;
    VuAnalysisResourceSet definiteWrites;
    VuAnalysisResourceSet liveIn;
    VuAnalysisResourceSet liveOut;
    VuAnalysisPipelineFacts entryFacts;
    VuAnalysisPipelineFacts exitFacts;
    std::vector<VuAnalysisBarrier> barriers;
};

enum class VuAnalysisPendingBranchKind : uint8_t
{
    None,
    Static,
    Dynamic,
};

struct VuAnalysisEntryState
{
    uint32_t pc = 0u;
    bool endPending = false;
    VuAnalysisPendingBranchKind pendingBranch =
        VuAnalysisPendingBranchKind::None;
    uint32_t pendingBranchTarget = 0u;

    bool operator==(const VuAnalysisEntryState &) const = default;
};

enum class VuAnalysisSuccessorKind : uint8_t
{
    Fallthrough,
    PairLimit,
    BranchTaken,
    BranchNotTaken,
    IndirectBranch,
    XgkickResume,
};

struct VuAnalysisSuccessor
{
    VuAnalysisSuccessorKind kind =
        VuAnalysisSuccessorKind::Fallthrough;
    VuAnalysisEntryState entry;
    uint32_t targetBlock = UINT32_MAX;
    bool dynamicTarget = false;
    uint32_t fixedPairs = 0u;
    uint32_t fixedCycles = 0u;
};

// Keep this in lockstep with VuNativeBlockExit. The analysis layer does not
// depend on the recompiler header, so it can remain reusable by interpreters
// and test tools.
enum class VuAnalysisNativeExit : uint8_t
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

inline constexpr size_t kVuAnalysisNativeExitCount =
    static_cast<size_t>(VuAnalysisNativeExit::Fault) + 1u;

struct VuAnalysisExit
{
    VuAnalysisNativeExit kind =
        VuAnalysisNativeExit::Fault;
    bool possible = false;
    uint32_t minimumPairs = 0u;
    uint32_t maximumPairs = 0u;
    uint32_t minimumCycles = 0u;
    uint32_t maximumCycles = 0u;
    VuAnalysisResourceSet materialize;
};

enum class VuAnalysisBlockTerminal : uint8_t
{
    PairLimit,
    CodeWrap,
    BranchBoundary,
    ProgramEnd,
    XgkickBoundary,
    UnsupportedInstruction,
    CodeBounds,
};

struct VuAnalysisBlock
{
    uint32_t index = 0u;
    VuAnalysisEntryState entry;
    VuAnalysisBlockTerminal terminal =
        VuAnalysisBlockTerminal::PairLimit;
    uint32_t fixedPairs = 0u;
    uint32_t fixedCycles = 0u;
    VuAnalysisResourceSet reads;
    VuAnalysisResourceSet writes;
    VuAnalysisResourceSet definiteWrites;
    VuAnalysisResourceSet uses;
    VuAnalysisResourceSet liveIn;
    VuAnalysisResourceSet liveOut;
    VuAnalysisResourceSet exitMaterialization;
    std::vector<VuAnalysisPair> pairs;
    std::vector<VuAnalysisSuccessor> successors;
    std::vector<VuAnalysisExit> exits;
};

struct VuAnalysisConfiguration
{
    uint32_t maximumPairsPerBlock = 64u;
    uint32_t maximumBlocks = 16384u;
    uint64_t hostFeatures = 0u;
    bool instrumented = false;
    bool verification = false;
};

struct VuControlFlowGraph
{
    bool valid = false;
    std::string diagnostic;
    uint32_t codeSize = 0u;
    VuAnalysisEntryState entry;
    VuAnalysisConfiguration configuration;
    std::array<bool, kVuAnalysisNativeExitCount>
        modeledNativeExits{};
    std::vector<VuAnalysisBlock> blocks;
};

// A deterministic whole-block assignment to abstract host vector slots.
// Slot zero in vfHostSlots means memory-resident; assigned slots are encoded
// as one plus their zero-based index so the structure remains aggregate-safe.
struct VuAnalysisRegisterAllocation
{
    std::array<uint8_t, 32u> vfHostSlots{};
    std::array<uint8_t, 32u> vfDirtyLanes{};
    std::array<uint8_t, 31u> vfRegisters{};
    std::array<uint8_t, 16u> viHostSlots{};
    std::array<uint8_t, 15u> viRegisters{};
    uint8_t vfRegisterCount = 0u;
    uint8_t maximumVfRegisters = 0u;
    uint8_t viRegisterCount = 0u;
    uint8_t maximumViRegisters = 0u;
    uint16_t viDirtyRegisters = 0u;
    uint32_t vfAccesses = 0u;
    uint32_t allocatedVfAccesses = 0u;
    uint32_t candidateVfRegisters = 0u;
    uint32_t spilledVfRegisters = 0u;
    uint32_t maximumLiveVfRegisters = 0u;
    uint32_t viAccesses = 0u;
    uint32_t allocatedViAccesses = 0u;
    uint32_t candidateViRegisters = 0u;
    uint32_t spilledViRegisters = 0u;
    uint32_t maximumLiveViRegisters = 0u;
};

[[nodiscard]] VuControlFlowGraph analyzeVuControlFlow(
    const uint8_t *code, uint32_t codeSize,
    const VuAnalysisEntryState &entry,
    VuAnalysisConfiguration configuration = {});
[[nodiscard]] VuControlFlowGraph analyzeVuControlFlow(
    const uint8_t *code, uint32_t codeSize,
    uint32_t entryPc,
    VuAnalysisConfiguration configuration = {});
[[nodiscard]] VuAnalysisBlock analyzeVuIrBlock(
    const VuIrBlock &block,
    VuAnalysisConfiguration configuration = {});
[[nodiscard]] VuAnalysisRegisterAllocation
allocateVuAnalysisRegisters(
    const VuAnalysisBlock &block,
    uint8_t maximumVfRegisters,
    uint8_t maximumViRegisters = 0u);

[[nodiscard]] bool vuAnalysisCanInlineUpperOpcode(
    VuIrOpcode opcode, uint64_t hostFeatures);
[[nodiscard]] bool vuAnalysisCanInlineLowerOpcode(
    VuIrOpcode opcode);
[[nodiscard]] bool vuAnalysisPairUsesNativeHelper(
    const VuIrInstructionPair &pair,
    uint64_t hostFeatures,
    bool instrumented,
    bool endPending);

[[nodiscard]] std::string serializeVuAnalysis(
    const VuControlFlowGraph &graph);
[[nodiscard]] std::string_view vuAnalysisBarrierName(
    VuAnalysisBarrierKind kind);
[[nodiscard]] std::string_view vuAnalysisBarrierPhaseName(
    VuAnalysisBarrierPhase phase);
[[nodiscard]] std::string_view vuAnalysisKnownStateName(
    VuAnalysisKnownState state);
[[nodiscard]] std::string_view vuAnalysisSuccessorName(
    VuAnalysisSuccessorKind kind);
[[nodiscard]] std::string_view vuAnalysisNativeExitName(
    VuAnalysisNativeExit exit);
[[nodiscard]] std::string_view vuAnalysisBlockTerminalName(
    VuAnalysisBlockTerminal terminal);

#endif
