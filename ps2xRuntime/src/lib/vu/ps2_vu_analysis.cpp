#include "runtime/ps2_vu_analysis.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <iomanip>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>

namespace
{
    constexpr uint8_t kAllVectorLanes = 0x0fu;
    constexpr uint64_t kHostFeatureAvx = 1u << 2u;
    constexpr uint64_t kHostFeatureFma = 1u << 4u;

    struct EntryStateLess
    {
        bool operator()(
            const VuAnalysisEntryState &left,
            const VuAnalysisEntryState &right) const
        {
            return
                std::tie(
                    left.pc, left.endPending,
                    left.pendingBranch,
                    left.pendingBranchTarget) <
                std::tie(
                    right.pc, right.endPending,
                    right.pendingBranch,
                    right.pendingBranchTarget);
        }
    };

    struct PathState
    {
        VuAnalysisEntryState entry;
        VuAnalysisSuccessorKind edgeKind =
            VuAnalysisSuccessorKind::Fallthrough;
        bool dynamicTarget = false;
    };

    bool samePath(
        const PathState &left, const PathState &right)
    {
        return
            left.entry == right.entry &&
            left.edgeKind == right.edgeKind &&
            left.dynamicTarget == right.dynamicTarget;
    }

    void deduplicatePaths(std::vector<PathState> &paths)
    {
        std::vector<PathState> unique;
        unique.reserve(paths.size());
        for (const PathState &path : paths)
        {
            const bool present =
                std::any_of(
                    unique.begin(), unique.end(),
                    [&](const PathState &candidate)
                    {
                        return samePath(path, candidate);
                    });
            if (!present)
                unique.push_back(path);
        }
        paths = std::move(unique);
    }

    void addVfRegisters(
        VuAnalysisResourceSet &resources,
        uint32_t registers, uint8_t lanes)
    {
        for (uint32_t reg = 0u; reg < 32u; ++reg)
        {
            if ((registers & (1u << reg)) != 0u)
                resources.vf[reg] |= lanes;
        }
    }

    void stripArchitecturalConstants(
        VuAnalysisResourceSet &resources)
    {
        resources.vf[0] = 0u;
        resources.vi &= static_cast<uint16_t>(~1u);
    }

    VuAnalysisResourceSet resourcesWithOnlyMemoryAndOperands(
        const VuAnalysisResourceSet &resources)
    {
        VuAnalysisResourceSet result;
        result.vf = resources.vf;
        result.vi = resources.vi;
        result.acc = resources.acc;
        result.scalars = resources.scalars;
        result.memory = resources.memory;
        result.sideEffects = resources.sideEffects;
        return result;
    }

    bool isConditionalBranch(VuIrOpcode opcode)
    {
        switch (opcode)
        {
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

    bool isDirectUnconditionalBranch(VuIrOpcode opcode)
    {
        return
            opcode == VuIrOpcode::LowerB ||
            opcode == VuIrOpcode::LowerBal;
    }

    bool isIndirectBranch(VuIrOpcode opcode)
    {
        return
            opcode == VuIrOpcode::LowerJr ||
            opcode == VuIrOpcode::LowerJalr;
    }

    uint32_t directBranchTarget(
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

    uint32_t sequentialPc(
        uint32_t pc, uint32_t codeSize)
    {
        pc += 8u;
        return pc >= codeSize ? 0u : pc;
    }

    struct BranchOutcome
    {
        bool armed = false;
        VuAnalysisPendingBranchKind kind =
            VuAnalysisPendingBranchKind::None;
        uint32_t target = 0u;
        VuAnalysisSuccessorKind edgeKind =
            VuAnalysisSuccessorKind::Fallthrough;
    };

    std::vector<BranchOutcome> branchOutcomes(
        const VuIrInstructionPair &pair,
        uint32_t addressMask)
    {
        const VuIrOpcode opcode = pair.lower.opcode;
        if (isDirectUnconditionalBranch(opcode))
        {
            return {{
                .armed = true,
                .kind =
                    VuAnalysisPendingBranchKind::Static,
                .target =
                    directBranchTarget(
                        pair, addressMask),
                .edgeKind =
                    VuAnalysisSuccessorKind::BranchTaken,
            }};
        }
        if (isIndirectBranch(opcode))
        {
            return {{
                .armed = true,
                .kind =
                    VuAnalysisPendingBranchKind::Dynamic,
                .edgeKind =
                    VuAnalysisSuccessorKind::IndirectBranch,
            }};
        }
        if (isConditionalBranch(opcode))
        {
            return {
                {
                    .armed = true,
                    .kind =
                        VuAnalysisPendingBranchKind::Static,
                    .target =
                        directBranchTarget(
                            pair, addressMask),
                    .edgeKind =
                        VuAnalysisSuccessorKind::BranchTaken,
                },
                {
                    .edgeKind =
                        VuAnalysisSuccessorKind::BranchNotTaken,
                },
            };
        }
        return {{
            .edgeKind =
                VuAnalysisSuccessorKind::Fallthrough,
        }};
    }

    std::vector<PathState> advanceControl(
        const std::vector<PathState> &input,
        const VuIrInstructionPair &pair,
        uint32_t codeSize)
    {
        const uint32_t addressMask = codeSize - 1u;
        const uint32_t nextPc =
            sequentialPc(pair.pc, codeSize);
        const std::vector<BranchOutcome> outcomes =
            branchOutcomes(pair, addressMask);
        std::vector<PathState> output;
        output.reserve(input.size() * outcomes.size());

        for (const PathState &path : input)
        {
            for (const BranchOutcome &outcome : outcomes)
            {
                PathState next;
                next.entry.endPending =
                    vuIrHasPairFlag(
                        pair, VuIrPairEnd);
                next.edgeKind =
                    outcome.edgeKind ==
                            VuAnalysisSuccessorKind::
                                Fallthrough
                        ? path.edgeKind
                        : outcome.edgeKind;

                if (outcome.armed)
                {
                    // A branch in a delay pair replaces the older pending
                    // branch before the post-pair control update.
                    next.entry.pc = nextPc;
                    next.entry.pendingBranch =
                        outcome.kind;
                    next.entry.pendingBranchTarget =
                        outcome.target;
                }
                else if (
                    path.entry.pendingBranch ==
                    VuAnalysisPendingBranchKind::Static)
                {
                    next.entry.pc =
                        path.entry.pendingBranchTarget &
                        addressMask;
                }
                else if (
                    path.entry.pendingBranch ==
                    VuAnalysisPendingBranchKind::Dynamic)
                {
                    next.dynamicTarget = true;
                    next.entry.pc = 0u;
                    next.edgeKind =
                        VuAnalysisSuccessorKind::
                            IndirectBranch;
                }
                else
                {
                    next.entry.pc = nextPc;
                    if (outcome.edgeKind ==
                        VuAnalysisSuccessorKind::
                            BranchNotTaken)
                    {
                        next.edgeKind =
                            VuAnalysisSuccessorKind::
                                BranchNotTaken;
                    }
                }
                output.push_back(next);
            }
        }
        deduplicatePaths(output);
        return output;
    }

    void addOperationResources(
        const VuIrOperation &operation,
        VuAnalysisResourceSet &reads,
        VuAnalysisResourceSet &writes,
        VuAnalysisResourceSet &definiteWrites)
    {
        addVfRegisters(
            reads, operation.vfReadMask,
            kAllVectorLanes);
        const uint8_t destination =
            operation.destinationMask != 0u
                ? operation.destinationMask
                : kAllVectorLanes;
        addVfRegisters(
            writes, operation.vfWriteMask,
            destination);
        addVfRegisters(
            definiteWrites, operation.vfWriteMask,
            destination);
        reads.vi |= operation.viReadMask;
        writes.vi |= operation.viWriteMask;
        definiteWrites.vi |= operation.viWriteMask;

        const auto addScalar =
            [&](VuIrOpFlag readFlag,
                VuIrOpFlag writeFlag,
                uint32_t resource,
                bool definite)
        {
            if (vuIrHasOpFlag(operation, readFlag))
                reads.scalars |= resource;
            if (vuIrHasOpFlag(operation, writeFlag))
            {
                writes.scalars |= resource;
                if (definite)
                    definiteWrites.scalars |= resource;
            }
        };

        if (vuIrHasOpFlag(
                operation, VuIrOpReadsAcc))
        {
            reads.acc |= destination;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpWritesAcc))
        {
            writes.acc |= destination;
            definiteWrites.acc |= destination;
        }
        addScalar(
            VuIrOpReadsQ, VuIrOpWritesQ,
            VuAnalysisScalarQ, false);
        addScalar(
            VuIrOpReadsP, VuIrOpWritesP,
            VuAnalysisScalarP, false);
        addScalar(
            VuIrOpReadsI, VuIrOpWritesI,
            VuAnalysisScalarI, true);
        addScalar(
            VuIrOpReadsMac, VuIrOpWritesMac,
            VuAnalysisScalarMac,
            !vuIrHasOpFlag(
                operation, VuIrOpFmac));
        addScalar(
            VuIrOpReadsClip, VuIrOpWritesClip,
            VuAnalysisScalarClip, true);
        addScalar(
            VuIrOpReadsStatus,
            VuIrOpWritesStatus,
            VuAnalysisScalarStatus,
            !vuIrHasOpFlag(
                operation, VuIrOpFmac));

        if (vuIrHasOpFlag(
                operation, VuIrOpReadsTop))
        {
            reads.scalars |= VuAnalysisScalarTop;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpReadsItop))
        {
            reads.scalars |= VuAnalysisScalarItop;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpReadsVuData))
        {
            reads.memory |= VuAnalysisMemoryVuData;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpWritesVuData))
        {
            writes.memory |= VuAnalysisMemoryVuData;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpFmac))
        {
            writes.pipelines |=
                VuAnalysisPipelineFmacFlags |
                VuAnalysisPipelineWorkingMac;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpWritesQ))
        {
            writes.pipelines |=
                VuAnalysisPipelineDelayedQ;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpWritesP))
        {
            writes.pipelines |=
                VuAnalysisPipelineDelayedP;
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpBranch))
        {
            reads.control |= VuAnalysisControlPc;
            writes.control |=
                VuAnalysisControlBranchPending |
                VuAnalysisControlBranchTarget |
                VuAnalysisControlBranchDelay;
            if (!isConditionalBranch(operation.opcode))
            {
                definiteWrites.control |=
                    VuAnalysisControlBranchPending |
                    VuAnalysisControlBranchTarget |
                    VuAnalysisControlBranchDelay;
            }
        }
        if (vuIrHasOpFlag(
                operation, VuIrOpExternalEffect))
        {
            writes.memory |=
                VuAnalysisMemoryExternal;
            writes.sideEffects |=
                VuAnalysisSideEffectPath1;
        }
    }

    VuAnalysisResourceSet xgkickReads()
    {
        VuAnalysisResourceSet resources;
        resources.pipelines |=
            VuAnalysisPipelineXgkick;
        resources.memory |=
            VuAnalysisMemoryVuData |
            VuAnalysisMemoryExternal;
        return resources;
    }

    VuAnalysisResourceSet xgkickClobbers()
    {
        VuAnalysisResourceSet resources;
        resources.pipelines |=
            VuAnalysisPipelineXgkick;
        resources.memory |=
            VuAnalysisMemoryExternal;
        resources.sideEffects |=
            VuAnalysisSideEffectPath1 |
            VuAnalysisSideEffectCodeInvalidation;
        return resources;
    }

    VuAnalysisPair analyzePair(
        const VuIrInstructionPair &pair,
        const VuAnalysisConfiguration &configuration,
        bool endPending)
    {
        VuAnalysisPair analysis;
        analysis.ir = pair;
        analysis.retires =
            !vuIrHasPairFlag(
                pair, VuIrPairUnsupported);
        if (!analysis.retires)
        {
            analysis.barriers.push_back({
                .kind =
                    VuAnalysisBarrierKind::
                        UnsupportedOperation,
                .phase =
                    VuAnalysisBarrierPhase::BeforePair,
                .condition =
                    VuAnalysisBarrierCondition::Always,
                .reads = {},
                .clobbers = {},
            });
            return analysis;
        }
        analysis.usesNativeHelper =
            vuAnalysisPairUsesNativeHelper(
                pair, configuration.hostFeatures,
                configuration.instrumented,
                endPending);

        VuAnalysisResourceSet upperReads;
        VuAnalysisResourceSet upperWrites;
        VuAnalysisResourceSet upperDefinite;
        VuAnalysisResourceSet lowerReads;
        VuAnalysisResourceSet lowerWrites;
        VuAnalysisResourceSet lowerDefinite;
        addOperationResources(
            pair.upper, upperReads,
            upperWrites, upperDefinite);
        addOperationResources(
            pair.lower, lowerReads,
            lowerWrites, lowerDefinite);

        analysis.reads = upperReads;
        vuAnalysisUnionInto(
            analysis.reads, lowerReads);
        analysis.writes = upperWrites;
        vuAnalysisUnionInto(
            analysis.writes, lowerWrites);
        analysis.definiteWrites = upperDefinite;
        vuAnalysisUnionInto(
            analysis.definiteWrites,
            lowerDefinite);

        // Every retired pair advances the retained pipelines and control
        // state, even when both decoded operations are NOPs.
        analysis.reads.control |=
            VuAnalysisControlPc |
            VuAnalysisControlEbit |
            VuAnalysisControlActive |
            VuAnalysisControlBranchPending |
            VuAnalysisControlBranchTarget |
            VuAnalysisControlBranchDelay;
        analysis.writes.control |=
            VuAnalysisControlPc |
            VuAnalysisControlEbit |
            VuAnalysisControlActive |
            VuAnalysisControlBranchPending |
            VuAnalysisControlBranchTarget |
            VuAnalysisControlBranchDelay |
            VuAnalysisControlIssuedCycles;
        analysis.definiteWrites.control |=
            VuAnalysisControlPc |
            VuAnalysisControlIssuedCycles;
        if (endPending)
        {
            analysis.definiteWrites.control |=
                VuAnalysisControlActive;
        }
        if (vuIrHasPairFlag(pair, VuIrPairEnd))
        {
            analysis.definiteWrites.control |=
                VuAnalysisControlEbit;
        }

        constexpr uint16_t pipelineState =
            VuAnalysisPipelineDelayedQ |
            VuAnalysisPipelineDelayedP |
            VuAnalysisPipelineFmacFlags |
            VuAnalysisPipelineFmacIndex |
            VuAnalysisPipelineWorkingMac |
            VuAnalysisPipelineViBackup |
            VuAnalysisPipelineXgkick;
        analysis.reads.pipelines |= pipelineState;
        analysis.writes.pipelines |= pipelineState;
        analysis.writes.scalars |=
            VuAnalysisScalarQ |
            VuAnalysisScalarP |
            VuAnalysisScalarMac |
            VuAnalysisScalarStatus;
        analysis.reads.scalars |=
            VuAnalysisScalarStatus;

        // Architectural constants are re-established after every pair.
        analysis.writes.vf[0] |= kAllVectorLanes;
        analysis.definiteWrites.vf[0] |=
            kAllVectorLanes;
        analysis.writes.vi |= 1u;
        analysis.definiteWrites.vi |= 1u;

        const VuAnalysisBarrier advance{
            .kind =
                VuAnalysisBarrierKind::XgkickAdvance,
            .phase =
                VuAnalysisBarrierPhase::AfterPair,
            .condition =
                VuAnalysisBarrierCondition::
                    IfXgkickActive,
            .reads = xgkickReads(),
            .clobbers = xgkickClobbers(),
        };
        analysis.barriers.push_back(advance);
        vuAnalysisUnionInto(
            analysis.reads, advance.reads);
        vuAnalysisUnionInto(
            analysis.writes, advance.clobbers);

        if (vuIrHasPairFlag(pair, VuIrPairXgkick))
        {
            VuAnalysisResourceSet startReads =
                lowerReads;
            startReads.memory |=
                VuAnalysisMemoryVuData |
                VuAnalysisMemoryExternal;
            VuAnalysisResourceSet startClobbers =
                lowerWrites;
            startClobbers.pipelines |=
                VuAnalysisPipelineXgkick;
            startClobbers.memory |=
                VuAnalysisMemoryExternal;
            startClobbers.sideEffects |=
                VuAnalysisSideEffectPath1 |
                VuAnalysisSideEffectCodeInvalidation;
            analysis.barriers.push_back({
                .kind =
                    VuAnalysisBarrierKind::
                        XgkickStart,
                .phase =
                    VuAnalysisBarrierPhase::
                        PairExecution,
                .condition =
                    VuAnalysisBarrierCondition::Always,
                .reads = startReads,
                .clobbers = startClobbers,
            });
        }

        const bool unknownMemory =
            vuIrHasOpFlag(
                pair.upper, VuIrOpReadsVuData) ||
            vuIrHasOpFlag(
                pair.upper, VuIrOpWritesVuData) ||
            vuIrHasOpFlag(
                pair.lower, VuIrOpReadsVuData) ||
            vuIrHasOpFlag(
                pair.lower, VuIrOpWritesVuData);
        if (unknownMemory &&
            !vuIrHasPairFlag(
                pair, VuIrPairXgkick))
        {
            analysis.barriers.push_back({
                .kind =
                    VuAnalysisBarrierKind::
                        UnknownMemory,
                .phase =
                    VuAnalysisBarrierPhase::
                        PairExecution,
                .condition =
                    VuAnalysisBarrierCondition::Always,
                .reads =
                    resourcesWithOnlyMemoryAndOperands(
                        lowerReads),
                .clobbers =
                    resourcesWithOnlyMemoryAndOperands(
                        lowerWrites),
            });
        }

        if (analysis.usesNativeHelper)
        {
            analysis.barriers.push_back({
                .kind =
                    VuAnalysisBarrierKind::
                        NativeHelper,
                .phase =
                    VuAnalysisBarrierPhase::WholePair,
                .condition =
                    VuAnalysisBarrierCondition::Always,
                .reads = analysis.reads,
                .clobbers = analysis.writes,
            });
        }
        if (configuration.instrumented)
        {
            VuAnalysisResourceSet debugReads =
                vuAnalysisAllArchitecturalResources();
            debugReads.memory |=
                VuAnalysisMemoryVuCode;
            VuAnalysisResourceSet debugClobbers;
            debugClobbers.sideEffects |=
                VuAnalysisSideEffectDebug;
            analysis.barriers.push_back({
                .kind =
                    VuAnalysisBarrierKind::
                        DebugObserver,
                .phase =
                    VuAnalysisBarrierPhase::BeforePair,
                .condition =
                    VuAnalysisBarrierCondition::Always,
                .reads = debugReads,
                .clobbers = debugClobbers,
            });
            vuAnalysisUnionInto(
                analysis.reads, debugReads);
            vuAnalysisUnionInto(
                analysis.writes, debugClobbers);
        }
        if (configuration.verification)
        {
            VuAnalysisResourceSet all =
                vuAnalysisAllGuestResources();
            all.sideEffects |=
                VuAnalysisSideEffectVerification;
            analysis.barriers.push_back({
                .kind =
                    VuAnalysisBarrierKind::
                        Verification,
                .phase =
                    VuAnalysisBarrierPhase::WholePair,
                .condition =
                    VuAnalysisBarrierCondition::Always,
                .reads = all,
                .clobbers = all,
            });
            vuAnalysisUnionInto(
                analysis.reads, all);
            vuAnalysisUnionInto(
                analysis.writes, all);
        }
        return analysis;
    }

    uint8_t backedUpRegister(
        const VuIrInstructionPair &pair)
    {
        switch (pair.lower.opcode)
        {
        case VuIrOpcode::LowerLqi:
        case VuIrOpcode::LowerLqd:
            return static_cast<uint8_t>(
                (pair.lowerWord >> 11u) & 0x0fu);
        case VuIrOpcode::LowerSqi:
        case VuIrOpcode::LowerSqd:
        case VuIrOpcode::LowerMtir:
            return static_cast<uint8_t>(
                (pair.lowerWord >> 16u) & 0x0fu);
        default:
            return 0u;
        }
    }

    struct LocalPipelineTracker
    {
        bool viKnown = false;
        uint8_t viCycles = 2u;
        bool qKnown = false;
        uint32_t qCycles = 0u;
        bool pKnown = false;
        uint32_t pCycles = 0u;
    };

    VuAnalysisPipelineFacts pipelineFacts(
        const LocalPipelineTracker &tracker,
        const std::vector<VuAnalysisPair> &pairs,
        size_t pairIndex)
    {
        VuAnalysisPipelineFacts facts;
        facts.viBackup =
            !tracker.viKnown
                ? VuAnalysisKnownState::Dynamic
                : tracker.viCycles != 0u
                      ? VuAnalysisKnownState::Active
                      : VuAnalysisKnownState::Inactive;
        facts.viBackupCycles = tracker.viCycles;
        facts.delayedQ =
            !tracker.qKnown
                ? VuAnalysisKnownState::Dynamic
                : tracker.qCycles != 0u
                      ? VuAnalysisKnownState::Active
                      : VuAnalysisKnownState::Inactive;
        facts.qCyclesRemaining = tracker.qCycles;
        facts.delayedP =
            !tracker.pKnown
                ? VuAnalysisKnownState::Dynamic
                : tracker.pCycles != 0u
                      ? VuAnalysisKnownState::Active
                      : VuAnalysisKnownState::Inactive;
        facts.pCyclesRemaining = tracker.pCycles;
        if (pairIndex >= 4u)
        {
            facts.advancingFmacSlot =
                vuIrHasOpFlag(
                    pairs[pairIndex - 4u].ir.upper,
                    VuIrOpFmac)
                    ? VuAnalysisKnownState::Active
                    : VuAnalysisKnownState::Inactive;
        }
        return facts;
    }

    void advancePipelineFacts(
        LocalPipelineTracker &tracker,
        const VuIrInstructionPair &pair)
    {
        if (tracker.viCycles != 0u)
            --tracker.viCycles;
        if (!tracker.viKnown &&
            tracker.viCycles == 0u)
        {
            tracker.viKnown = true;
        }
        if (backedUpRegister(pair) != 0u)
        {
            tracker.viKnown = true;
            tracker.viCycles = 2u;
        }

        if (tracker.qKnown &&
            tracker.qCycles != 0u)
        {
            --tracker.qCycles;
        }
        if (vuIrHasOpFlag(
                pair.lower, VuIrOpQBarrier))
        {
            tracker.qKnown = true;
            tracker.qCycles =
                vuIrHasOpFlag(
                    pair.lower, VuIrOpWritesQ)
                    ? pair.lower.latency
                    : 0u;
        }

        if (tracker.pKnown &&
            tracker.pCycles != 0u)
        {
            --tracker.pCycles;
        }
        if (vuIrHasOpFlag(
                pair.lower, VuIrOpPBarrier))
        {
            tracker.pKnown = true;
            tracker.pCycles =
                vuIrHasOpFlag(
                    pair.lower, VuIrOpWritesP)
                    ? pair.lower.latency
                    : 0u;
        }
    }

    void attachPipelineFacts(VuAnalysisBlock &block)
    {
        LocalPipelineTracker tracker;
        for (size_t index = 0u;
             index < block.pairs.size(); ++index)
        {
            VuAnalysisPair &pair = block.pairs[index];
            pair.entryFacts =
                pipelineFacts(
                    tracker, block.pairs, index);
            if (pair.retires)
                advancePipelineFacts(tracker, pair.ir);
            pair.exitFacts =
                pipelineFacts(
                    tracker, block.pairs,
                    index + 1u);
        }
    }

    void aggregateBlockResources(
        VuAnalysisBlock &block)
    {
        VuAnalysisResourceSet defined;
        for (const VuAnalysisPair &pair : block.pairs)
        {
            vuAnalysisUnionInto(
                block.reads, pair.reads);
            vuAnalysisUnionInto(
                block.writes, pair.writes);

            VuAnalysisResourceSet uses = pair.reads;
            vuAnalysisSubtractFrom(uses, defined);
            vuAnalysisUnionInto(block.uses, uses);
            vuAnalysisUnionInto(
                defined, pair.definiteWrites);
        }
        block.definiteWrites = defined;
        stripArchitecturalConstants(block.uses);
        stripArchitecturalConstants(
            block.definiteWrites);
    }

    void appendExit(
        VuAnalysisBlock &block,
        VuAnalysisNativeExit kind,
        bool possible,
        uint32_t minimumPairs,
        uint32_t maximumPairs,
        uint32_t minimumCycles,
        uint32_t maximumCycles)
    {
        block.exits.push_back({
            .kind = kind,
            .possible = possible,
            .minimumPairs = minimumPairs,
            .maximumPairs = maximumPairs,
            .minimumCycles = minimumCycles,
            .maximumCycles = maximumCycles,
            .materialize =
                vuAnalysisAllGuestResources(),
        });
    }

    void attachNativeExits(
        VuAnalysisBlock &block)
    {
        const bool blockComplete =
            block.terminal ==
                VuAnalysisBlockTerminal::PairLimit ||
            block.terminal ==
                VuAnalysisBlockTerminal::CodeWrap ||
            block.terminal ==
                VuAnalysisBlockTerminal::BranchBoundary;
        const bool programEnded =
            block.terminal ==
            VuAnalysisBlockTerminal::ProgramEnd;
        const bool codeBounds =
            block.terminal ==
            VuAnalysisBlockTerminal::CodeBounds;
        const bool xgkick =
            block.terminal ==
            VuAnalysisBlockTerminal::XgkickBoundary;
        const bool unsupported =
            block.terminal ==
            VuAnalysisBlockTerminal::
                UnsupportedInstruction;
        const bool codeBoundsBudget =
            !codeBounds;
        const uint32_t budgetMaximumPairs =
            unsupported
                ? block.fixedPairs
                : block.fixedPairs == 0u
                      ? 0u
                      : block.fixedPairs - 1u;
        const uint32_t finalRetiredPairCycles =
            block.fixedPairs == 0u
                ? 0u
                : block.pairs[
                      static_cast<size_t>(
                          block.fixedPairs - 1u)]
                      .ir.cycles;
        const uint32_t budgetMaximumCycles =
            unsupported
                ? block.fixedCycles
                : block.fixedCycles == 0u
                      ? 0u
                      : block.fixedCycles -
                            finalRetiredPairCycles;
        const uint32_t firstRetiredPairCycles =
            block.fixedPairs == 0u
                ? 0u
                : block.pairs.front().ir.cycles;

        appendExit(
            block,
            VuAnalysisNativeExit::BlockComplete,
            blockComplete,
            block.fixedPairs,
            block.fixedPairs,
            block.fixedCycles,
            block.fixedCycles);
        appendExit(
            block, VuAnalysisNativeExit::CycleBudget,
            codeBoundsBudget,
            0u, budgetMaximumPairs,
            0u, budgetMaximumCycles);
        appendExit(
            block, VuAnalysisNativeExit::Inactive,
            true, 0u, 0u, 0u, 0u);
        appendExit(
            block, VuAnalysisNativeExit::ProgramEnded,
            programEnded,
            block.fixedPairs,
            block.fixedPairs,
            block.fixedCycles,
            block.fixedCycles);
        appendExit(
            block, VuAnalysisNativeExit::CodeBounds,
            codeBounds, 0u, 0u, 0u, 0u);
        appendExit(
            block,
            VuAnalysisNativeExit::XgkickBoundary,
            xgkick,
            block.fixedPairs,
            block.fixedPairs,
            block.fixedCycles,
            block.fixedCycles);
        // The native enum reserves an observer exit, but today's observer is
        // a void callback: it either returns or is reported as Fault. Keeping
        // the declaration explicit prevents a future stop path from being
        // added without updating the CFG contract.
        appendExit(
            block,
            VuAnalysisNativeExit::DebugObserver,
            false, 0u, 0u, 0u, 0u);
        appendExit(
            block,
            VuAnalysisNativeExit::CodeInvalidated,
            block.fixedCycles != 0u,
            block.fixedPairs != 0u ? 1u : 0u,
            block.fixedPairs,
            firstRetiredPairCycles,
            block.fixedCycles);
        appendExit(
            block,
            VuAnalysisNativeExit::
                UnsupportedInstruction,
            unsupported,
            block.fixedPairs,
            block.fixedPairs,
            block.fixedCycles,
            block.fixedCycles);
        appendExit(
            block, VuAnalysisNativeExit::Fault,
            true,
            0u, block.fixedPairs,
            0u, block.fixedCycles);

        block.exitMaterialization =
            vuAnalysisAllGuestResources();
    }

    VuAnalysisSuccessorKind successorKind(
        const PathState &path,
        VuAnalysisBlockTerminal terminal)
    {
        if (terminal ==
            VuAnalysisBlockTerminal::XgkickBoundary)
        {
            return
                VuAnalysisSuccessorKind::XgkickResume;
        }
        if (terminal ==
            VuAnalysisBlockTerminal::PairLimit)
        {
            return
                VuAnalysisSuccessorKind::PairLimit;
        }
        if (terminal ==
                VuAnalysisBlockTerminal::CodeWrap &&
            path.edgeKind ==
                VuAnalysisSuccessorKind::Fallthrough)
        {
            return
                VuAnalysisSuccessorKind::Fallthrough;
        }
        return path.edgeKind;
    }

    void appendSuccessors(
        VuAnalysisBlock &block,
        const std::vector<PathState> &paths)
    {
        for (const PathState &path : paths)
        {
            VuAnalysisSuccessor successor{
                .kind =
                    successorKind(
                        path, block.terminal),
                .entry = path.entry,
                .dynamicTarget = path.dynamicTarget,
                .fixedPairs = block.fixedPairs,
                .fixedCycles = block.fixedCycles,
            };
            const bool duplicate =
                std::any_of(
                    block.successors.begin(),
                    block.successors.end(),
                    [&](const VuAnalysisSuccessor &other)
                    {
                        return
                            other.kind == successor.kind &&
                            other.entry ==
                                successor.entry &&
                            other.dynamicTarget ==
                                successor.dynamicTarget;
                    });
            if (!duplicate)
            {
                block.successors.push_back(successor);
            }
        }
    }

    VuAnalysisBlock buildBlock(
        uint32_t index,
        const VuAnalysisEntryState &entry,
        const std::vector<VuIrInstructionPair> &decoded,
        uint32_t codeSize,
        const VuAnalysisConfiguration &configuration)
    {
        VuAnalysisBlock block;
        block.index = index;
        block.entry = entry;

        std::vector<PathState> paths{{
            .entry = entry,
        }};
        bool branchDelayBoundary =
            entry.pendingBranch !=
            VuAnalysisPendingBranchKind::None;
        bool endDelayBoundary = entry.endPending;
        const uint32_t maximumPairs =
            configuration.maximumPairsPerBlock;

        for (uint32_t pairIndex = 0u;
             pairIndex < maximumPairs; ++pairIndex)
        {
            if (paths.empty() ||
                paths.front().dynamicTarget ||
                paths.front().entry.pc >= codeSize)
            {
                block.terminal =
                    VuAnalysisBlockTerminal::CodeBounds;
                break;
            }
            const uint32_t pc = paths.front().entry.pc;
            const bool samePc =
                std::all_of(
                    paths.begin(), paths.end(),
                    [&](const PathState &path)
                    {
                        return
                            !path.dynamicTarget &&
                            path.entry.pc == pc;
                    });
            if (!samePc || (pc & 7u) != 0u)
            {
                block.terminal =
                    VuAnalysisBlockTerminal::CodeBounds;
                break;
            }

            const VuIrInstructionPair &ir =
                decoded[pc / 8u];
            const bool endPending =
                endDelayBoundary;
            VuAnalysisPair pair =
                analyzePair(
                    ir, configuration,
                    endPending);
            block.pairs.push_back(std::move(pair));
            if (!block.pairs.back().retires)
            {
                block.terminal =
                    VuAnalysisBlockTerminal::
                        UnsupportedInstruction;
                break;
            }

            ++block.fixedPairs;
            block.fixedCycles += ir.cycles;
            std::vector<PathState> next =
                advanceControl(paths, ir, codeSize);

            if (endPending)
            {
                block.terminal =
                    VuAnalysisBlockTerminal::ProgramEnd;
                paths.clear();
                break;
            }
            if (vuIrHasPairFlag(
                    ir, VuIrPairXgkick))
            {
                block.terminal =
                    VuAnalysisBlockTerminal::
                        XgkickBoundary;
                paths = std::move(next);
                appendSuccessors(block, paths);
                break;
            }
            if (branchDelayBoundary)
            {
                block.terminal =
                    VuAnalysisBlockTerminal::
                        BranchBoundary;
                paths = std::move(next);
                appendSuccessors(block, paths);
                break;
            }

            const bool beginsBranchDelay =
                vuIrHasPairFlag(
                    ir, VuIrPairBranch);
            const bool beginsEndDelay =
                vuIrHasPairFlag(
                    ir, VuIrPairEnd);
            paths = std::move(next);
            branchDelayBoundary = beginsBranchDelay;
            endDelayBoundary = beginsEndDelay;

            if (!branchDelayBoundary &&
                !endDelayBoundary)
            {
                const bool wraps =
                    std::any_of(
                        paths.begin(), paths.end(),
                        [](const PathState &path)
                        {
                            return
                                !path.dynamicTarget &&
                                path.entry.pc == 0u;
                        });
                if (wraps)
                {
                    block.terminal =
                        VuAnalysisBlockTerminal::CodeWrap;
                    appendSuccessors(block, paths);
                    break;
                }
            }

            if (pairIndex + 1u == maximumPairs)
            {
                block.terminal =
                    VuAnalysisBlockTerminal::PairLimit;
                appendSuccessors(block, paths);
            }
        }

        attachPipelineFacts(block);
        aggregateBlockResources(block);
        attachNativeExits(block);
        return block;
    }

    VuAnalysisResourceSet terminalObservation(
        const VuAnalysisBlock &block)
    {
        switch (block.terminal)
        {
        case VuAnalysisBlockTerminal::ProgramEnd:
        case VuAnalysisBlockTerminal::
            UnsupportedInstruction:
        case VuAnalysisBlockTerminal::CodeBounds:
            return vuAnalysisAllGuestResources();
        default:
            break;
        }
        for (const VuAnalysisSuccessor &successor :
             block.successors)
        {
            if (successor.dynamicTarget)
                return vuAnalysisAllGuestResources();
        }
        return {};
    }

    void computeLiveness(VuControlFlowGraph &graph)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (size_t reverse = graph.blocks.size();
                 reverse != 0u; --reverse)
            {
                VuAnalysisBlock &block =
                    graph.blocks[reverse - 1u];
                VuAnalysisResourceSet liveOut =
                    terminalObservation(block);
                for (const VuAnalysisSuccessor &successor :
                     block.successors)
                {
                    if (successor.dynamicTarget ||
                        successor.targetBlock ==
                            UINT32_MAX)
                    {
                        vuAnalysisUnionInto(
                            liveOut,
                            vuAnalysisAllGuestResources());
                    }
                    else
                    {
                        vuAnalysisUnionInto(
                            liveOut,
                            graph.blocks[
                                successor.targetBlock]
                                .liveIn);
                    }
                }
                stripArchitecturalConstants(liveOut);

                VuAnalysisResourceSet liveIn = liveOut;
                vuAnalysisSubtractFrom(
                    liveIn, block.definiteWrites);
                vuAnalysisUnionInto(
                    liveIn, block.uses);
                stripArchitecturalConstants(liveIn);
                if (liveOut != block.liveOut ||
                    liveIn != block.liveIn)
                {
                    block.liveOut = liveOut;
                    block.liveIn = liveIn;
                    changed = true;
                }
            }
        }

        for (VuAnalysisBlock &block : graph.blocks)
        {
            VuAnalysisResourceSet live =
                block.liveOut;
            vuAnalysisUnionInto(
                live, terminalObservation(block));
            for (size_t reverse = block.pairs.size();
                 reverse != 0u; --reverse)
            {
                VuAnalysisPair &pair =
                    block.pairs[reverse - 1u];
                pair.liveOut = live;
                vuAnalysisSubtractFrom(
                    live, pair.definiteWrites);
                vuAnalysisUnionInto(
                    live, pair.reads);
                stripArchitecturalConstants(live);
                pair.liveIn = live;
            }
        }
    }

    std::string laneNames(uint8_t lanes)
    {
        std::string result;
        if ((lanes & 0x8u) != 0u)
            result += 'x';
        if ((lanes & 0x4u) != 0u)
            result += 'y';
        if ((lanes & 0x2u) != 0u)
            result += 'z';
        if ((lanes & 0x1u) != 0u)
            result += 'w';
        return result;
    }

    template <typename Mask>
    void appendNamedBits(
        std::vector<std::string> &parts,
        Mask mask,
        std::initializer_list<
            std::pair<Mask, const char *>> names)
    {
        for (const auto &[bit, name] : names)
        {
            if ((mask & bit) != 0)
                parts.emplace_back(name);
        }
    }

    std::string formatEntryState(
        const VuAnalysisEntryState &entry,
        bool dynamicTarget = false)
    {
        std::ostringstream text;
        if (dynamicTarget)
            text << "pc=dynamic";
        else
        {
            text
                << "pc=" << std::hex << std::setw(4)
                << std::setfill('0') << entry.pc
                << std::dec << std::setfill(' ');
        }
        text << ",end=" << (entry.endPending ? 1 : 0);
        switch (entry.pendingBranch)
        {
        case VuAnalysisPendingBranchKind::None:
            text << ",pending=none";
            break;
        case VuAnalysisPendingBranchKind::Static:
            text
                << ",pending=static:"
                << std::hex << std::setw(4)
                << std::setfill('0')
                << entry.pendingBranchTarget
                << std::dec << std::setfill(' ');
            break;
        case VuAnalysisPendingBranchKind::Dynamic:
            text << ",pending=dynamic";
            break;
        }
        return text.str();
    }

    std::string formatFacts(
        const VuAnalysisPipelineFacts &facts)
    {
        std::ostringstream text;
        text
            << "vf0=constant,vi0=constant"
            << ",q="
            << vuAnalysisKnownStateName(
                   facts.delayedQ)
            << ":" << facts.qCyclesRemaining
            << ",p="
            << vuAnalysisKnownStateName(
                   facts.delayedP)
            << ":" << facts.pCyclesRemaining
            << ",vi="
            << vuAnalysisKnownStateName(
                   facts.viBackup)
            << ":"
            << static_cast<uint32_t>(
                   facts.viBackupCycles)
            << ",fmac="
            << vuAnalysisKnownStateName(
                   facts.advancingFmacSlot);
        return text.str();
    }
}

void vuAnalysisUnionInto(
    VuAnalysisResourceSet &destination,
    const VuAnalysisResourceSet &source)
{
    for (size_t reg = 0u;
         reg < destination.vf.size(); ++reg)
    {
        destination.vf[reg] |= source.vf[reg];
    }
    destination.vi |= source.vi;
    destination.acc |= source.acc;
    destination.scalars |= source.scalars;
    destination.control |= source.control;
    destination.pipelines |= source.pipelines;
    destination.memory |= source.memory;
    destination.sideEffects |= source.sideEffects;
}

void vuAnalysisSubtractFrom(
    VuAnalysisResourceSet &destination,
    const VuAnalysisResourceSet &source)
{
    for (size_t reg = 0u;
         reg < destination.vf.size(); ++reg)
    {
        destination.vf[reg] &=
            static_cast<uint8_t>(~source.vf[reg]);
    }
    destination.vi &=
        static_cast<uint16_t>(~source.vi);
    destination.acc &=
        static_cast<uint8_t>(~source.acc);
    destination.scalars &= ~source.scalars;
    destination.control &=
        static_cast<uint16_t>(~source.control);
    destination.pipelines &=
        static_cast<uint16_t>(~source.pipelines);
    destination.memory &=
        static_cast<uint8_t>(~source.memory);
    destination.sideEffects &=
        static_cast<uint8_t>(~source.sideEffects);
}

bool vuAnalysisResourcesEmpty(
    const VuAnalysisResourceSet &resources)
{
    return
        std::all_of(
            resources.vf.begin(),
            resources.vf.end(),
            [](uint8_t lanes)
            {
                return lanes == 0u;
            }) &&
        resources.vi == 0u &&
        resources.acc == 0u &&
        resources.scalars == 0u &&
        resources.control == 0u &&
        resources.pipelines == 0u &&
        resources.memory == 0u &&
        resources.sideEffects == 0u;
}

VuAnalysisResourceSet
vuAnalysisAllArchitecturalResources()
{
    VuAnalysisResourceSet resources;
    resources.vf.fill(kAllVectorLanes);
    resources.vi = UINT16_MAX;
    resources.acc = kAllVectorLanes;
    resources.scalars =
        VuAnalysisScalarQ |
        VuAnalysisScalarP |
        VuAnalysisScalarI |
        VuAnalysisScalarR |
        VuAnalysisScalarMac |
        VuAnalysisScalarClip |
        VuAnalysisScalarStatus |
        VuAnalysisScalarTop |
        VuAnalysisScalarItop;
    resources.control =
        VuAnalysisControlPc |
        VuAnalysisControlEbit |
        VuAnalysisControlActive |
        VuAnalysisControlBranchPending |
        VuAnalysisControlBranchTarget |
        VuAnalysisControlBranchDelay |
        VuAnalysisControlIssuedCycles;
    resources.pipelines =
        VuAnalysisPipelineDelayedQ |
        VuAnalysisPipelineDelayedP |
        VuAnalysisPipelineFmacFlags |
        VuAnalysisPipelineFmacIndex |
        VuAnalysisPipelineWorkingMac |
        VuAnalysisPipelineViBackup |
        VuAnalysisPipelineXgkick;
    return resources;
}

VuAnalysisResourceSet vuAnalysisAllGuestResources()
{
    VuAnalysisResourceSet resources =
        vuAnalysisAllArchitecturalResources();
    resources.memory =
        VuAnalysisMemoryVuData |
        VuAnalysisMemoryVuCode |
        VuAnalysisMemoryExternal;
    resources.sideEffects =
        VuAnalysisSideEffectPath1 |
        VuAnalysisSideEffectDebug |
        VuAnalysisSideEffectVerification |
        VuAnalysisSideEffectCodeInvalidation |
        VuAnalysisSideEffectUnknown;
    return resources;
}

std::string vuAnalysisFormatResources(
    const VuAnalysisResourceSet &resources)
{
    std::vector<std::string> parts;
    for (size_t reg = 0u;
         reg < resources.vf.size(); ++reg)
    {
        if (resources.vf[reg] == 0u)
            continue;
        parts.push_back(
            "vf" + std::to_string(reg) + "." +
            laneNames(resources.vf[reg]));
    }
    for (uint32_t reg = 0u; reg < 16u; ++reg)
    {
        if ((resources.vi & (1u << reg)) != 0u)
            parts.push_back("vi" + std::to_string(reg));
    }
    if (resources.acc != 0u)
        parts.push_back("acc." + laneNames(resources.acc));
    appendNamedBits<uint32_t>(
        parts, resources.scalars,
        {
            {VuAnalysisScalarQ, "q"},
            {VuAnalysisScalarP, "p"},
            {VuAnalysisScalarI, "i"},
            {VuAnalysisScalarR, "r"},
            {VuAnalysisScalarMac, "mac"},
            {VuAnalysisScalarClip, "clip"},
            {VuAnalysisScalarStatus, "status"},
            {VuAnalysisScalarTop, "top"},
            {VuAnalysisScalarItop, "itop"},
        });
    appendNamedBits<uint16_t>(
        parts, resources.control,
        {
            {VuAnalysisControlPc, "pc"},
            {VuAnalysisControlEbit, "ebit"},
            {VuAnalysisControlActive, "active"},
            {VuAnalysisControlBranchPending, "branch-pending"},
            {VuAnalysisControlBranchTarget, "branch-target"},
            {VuAnalysisControlBranchDelay, "branch-delay"},
            {VuAnalysisControlIssuedCycles, "issued-cycles"},
        });
    appendNamedBits<uint16_t>(
        parts, resources.pipelines,
        {
            {VuAnalysisPipelineDelayedQ, "pipe-q"},
            {VuAnalysisPipelineDelayedP, "pipe-p"},
            {VuAnalysisPipelineFmacFlags, "pipe-fmac"},
            {VuAnalysisPipelineFmacIndex, "pipe-fmac-index"},
            {VuAnalysisPipelineWorkingMac, "pipe-working-mac"},
            {VuAnalysisPipelineViBackup, "pipe-vi"},
            {VuAnalysisPipelineXgkick, "pipe-xgkick"},
        });
    appendNamedBits<uint8_t>(
        parts, resources.memory,
        {
            {VuAnalysisMemoryVuData, "mem-vu-data"},
            {VuAnalysisMemoryVuCode, "mem-vu-code"},
            {VuAnalysisMemoryExternal, "mem-external"},
        });
    appendNamedBits<uint8_t>(
        parts, resources.sideEffects,
        {
            {VuAnalysisSideEffectPath1, "effect-path1"},
            {VuAnalysisSideEffectDebug, "effect-debug"},
            {VuAnalysisSideEffectVerification, "effect-verify"},
            {VuAnalysisSideEffectCodeInvalidation, "effect-code"},
            {VuAnalysisSideEffectUnknown, "effect-unknown"},
        });

    if (parts.empty())
        return "-";
    std::ostringstream text;
    for (size_t index = 0u;
         index < parts.size(); ++index)
    {
        if (index != 0u)
            text << ",";
        text << parts[index];
    }
    return text.str();
}

VuControlFlowGraph analyzeVuControlFlow(
    const uint8_t *code, uint32_t codeSize,
    const VuAnalysisEntryState &requestedEntry,
    VuAnalysisConfiguration configuration)
{
    VuControlFlowGraph graph;
    graph.codeSize = codeSize;
    graph.entry = requestedEntry;
    graph.configuration = configuration;
    graph.modeledNativeExits.fill(true);

    if (!code || codeSize < 8u ||
        (codeSize & 7u) != 0u ||
        (codeSize & (codeSize - 1u)) != 0u)
    {
        graph.diagnostic =
            "VU analysis requires non-null, pair-aligned "
            "power-of-two code";
        return graph;
    }
    if (configuration.maximumPairsPerBlock == 0u ||
        configuration.maximumBlocks == 0u)
    {
        graph.diagnostic =
            "VU analysis limits must be nonzero";
        return graph;
    }

    VuAnalysisEntryState entry = requestedEntry;
    if ((entry.pc & 7u) != 0u ||
        entry.pc > codeSize - 8u)
    {
        graph.diagnostic =
            "VU analysis entry PC is outside the code extent";
        return graph;
    }
    if (entry.pendingBranch ==
        VuAnalysisPendingBranchKind::Static)
    {
        entry.pendingBranchTarget &= codeSize - 1u;
    }
    else
    {
        entry.pendingBranchTarget = 0u;
    }
    graph.entry = entry;

    std::vector<VuIrInstructionPair> decoded;
    decoded.reserve(codeSize / 8u);
    for (uint32_t pc = 0u;
         pc < codeSize; pc += 8u)
    {
        uint32_t lower = 0u;
        uint32_t upper = 0u;
        std::memcpy(
            &lower, code + pc, sizeof(lower));
        std::memcpy(
            &upper, code + pc + 4u, sizeof(upper));
        VuIrInstructionPair pair =
            decodeVuIrInstructionPair(
                pc, lower, upper);
        VuIrVerificationError verification;
        if (!verifyVuIrInstructionPair(
                pair, &verification))
        {
            graph.diagnostic =
                "malformed VU pair IR at PC " +
                std::to_string(verification.pc) +
                ": " + verification.message;
            return graph;
        }
        decoded.push_back(pair);
    }

    std::map<
        VuAnalysisEntryState, uint32_t,
        EntryStateLess> indices;
    indices.emplace(entry, 0u);
    VuAnalysisBlock initialBlock;
    initialBlock.index = 0u;
    initialBlock.entry = entry;
    graph.blocks.push_back(std::move(initialBlock));

    for (size_t cursor = 0u;
         cursor < graph.blocks.size(); ++cursor)
    {
        const VuAnalysisEntryState blockEntry =
            graph.blocks[cursor].entry;
        VuAnalysisBlock block =
            buildBlock(
                static_cast<uint32_t>(cursor),
                blockEntry, decoded, codeSize,
                configuration);

        for (VuAnalysisSuccessor &successor :
             block.successors)
        {
            if (successor.dynamicTarget)
                continue;
            auto found = indices.find(successor.entry);
            if (found == indices.end())
            {
                if (graph.blocks.size() >=
                    configuration.maximumBlocks)
                {
                    graph.diagnostic =
                        "VU analysis exceeded its block limit";
                    return graph;
                }
                const uint32_t target =
                    static_cast<uint32_t>(
                        graph.blocks.size());
                found = indices.emplace(
                    successor.entry, target).first;
                VuAnalysisBlock pendingBlock;
                pendingBlock.index = target;
                pendingBlock.entry =
                    successor.entry;
                graph.blocks.push_back(
                    std::move(pendingBlock));
            }
            successor.targetBlock = found->second;
        }
        graph.blocks[cursor] = std::move(block);
    }

    computeLiveness(graph);
    graph.valid = true;
    graph.diagnostic.clear();
    return graph;
}

VuControlFlowGraph analyzeVuControlFlow(
    const uint8_t *code, uint32_t codeSize,
    uint32_t entryPc,
    VuAnalysisConfiguration configuration)
{
    return analyzeVuControlFlow(
        code, codeSize,
        VuAnalysisEntryState{.pc = entryPc},
        configuration);
}

bool vuAnalysisCanInlineUpperOpcode(
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

bool vuAnalysisCanInlineLowerOpcode(
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

bool vuAnalysisPairUsesNativeHelper(
    const VuIrInstructionPair &pair,
    uint64_t hostFeatures,
    bool instrumented,
    bool endPending)
{
    if (vuIrHasPairFlag(
            pair, VuIrPairUnsupported))
    {
        return false;
    }
    return
        instrumented ||
        endPending ||
        pair.order == VuIrPairOrder::UpperThenLoi ||
        vuIrHasPairFlag(pair, VuIrPairXgkick) ||
        !vuAnalysisCanInlineUpperOpcode(
            pair.upper.opcode, hostFeatures) ||
        !vuAnalysisCanInlineLowerOpcode(
            pair.lower.opcode);
}

std::string serializeVuAnalysis(
    const VuControlFlowGraph &graph)
{
    std::ostringstream text;
    text
        << "vu-analysis-v1 valid="
        << (graph.valid ? 1 : 0)
        << " code=" << graph.codeSize
        << " entry={"
        << formatEntryState(graph.entry)
        << "} blocks=" << graph.blocks.size()
        << "\n";
    text << "native-exits=";
    for (size_t exit = 0u;
         exit < graph.modeledNativeExits.size();
         ++exit)
    {
        if (exit != 0u)
            text << ",";
        text
            << vuAnalysisNativeExitName(
                   static_cast<VuAnalysisNativeExit>(
                       exit))
            << ":"
            << (graph.modeledNativeExits[exit] ? 1 : 0);
    }
    text << "\n";
    if (!graph.diagnostic.empty())
        text << "diagnostic=" << graph.diagnostic << "\n";

    for (const VuAnalysisBlock &block : graph.blocks)
    {
        text
            << "block " << block.index
            << " entry={"
            << formatEntryState(block.entry)
            << "} terminal="
            << vuAnalysisBlockTerminalName(
                   block.terminal)
            << " pairs=" << block.fixedPairs
            << " cycles=" << block.fixedCycles
            << "\n";
        text
            << "  reads={"
            << vuAnalysisFormatResources(block.reads)
            << "} writes={"
            << vuAnalysisFormatResources(block.writes)
            << "} defs={"
            << vuAnalysisFormatResources(
                   block.definiteWrites)
            << "}\n";
        text
            << "  live-in={"
            << vuAnalysisFormatResources(block.liveIn)
            << "} live-out={"
            << vuAnalysisFormatResources(block.liveOut)
            << "}\n";
        for (size_t pairIndex = 0u;
             pairIndex < block.pairs.size();
             ++pairIndex)
        {
            const VuAnalysisPair &pair =
                block.pairs[pairIndex];
            text
                << "  pair " << pairIndex
                << " pc=" << std::hex << std::setw(4)
                << std::setfill('0') << pair.ir.pc
                << std::dec << std::setfill(' ')
                << " op="
                << vuIrOpcodeName(pair.ir.upper.opcode)
                << "/"
                << vuIrOpcodeName(pair.ir.lower.opcode)
                << " retires="
                << (pair.retires ? 1 : 0)
                << " helper="
                << (pair.usesNativeHelper ? 1 : 0)
                << "\n";
            text
                << "    reads={"
                << vuAnalysisFormatResources(pair.reads)
                << "} writes={"
                << vuAnalysisFormatResources(pair.writes)
                << "} defs={"
                << vuAnalysisFormatResources(
                       pair.definiteWrites)
                << "}\n";
            text
                << "    live-in={"
                << vuAnalysisFormatResources(pair.liveIn)
                << "} live-out={"
                << vuAnalysisFormatResources(pair.liveOut)
                << "}\n";
            text
                << "    facts-in={"
                << formatFacts(pair.entryFacts)
                << "} facts-out={"
                << formatFacts(pair.exitFacts)
                << "}\n";
            for (const VuAnalysisBarrier &barrier :
                 pair.barriers)
            {
                text
                    << "    barrier="
                    << vuAnalysisBarrierName(
                           barrier.kind)
                    << " phase="
                    << vuAnalysisBarrierPhaseName(
                           barrier.phase)
                    << " condition="
                    << (barrier.condition ==
                                VuAnalysisBarrierCondition::
                                    Always
                            ? "always"
                            : "if-xgkick-active")
                    << " reads={"
                    << vuAnalysisFormatResources(
                           barrier.reads)
                    << "} clobbers={"
                    << vuAnalysisFormatResources(
                           barrier.clobbers)
                    << "}\n";
            }
        }
        for (const VuAnalysisSuccessor &successor :
             block.successors)
        {
            text
                << "  edge="
                << vuAnalysisSuccessorName(
                       successor.kind)
                << " target={"
                << formatEntryState(
                       successor.entry,
                       successor.dynamicTarget)
                << "} block=";
            if (successor.targetBlock == UINT32_MAX)
                text << "dynamic";
            else
                text << successor.targetBlock;
            text
                << " cost=" << successor.fixedPairs
                << "/" << successor.fixedCycles
                << "\n";
        }
        for (const VuAnalysisExit &exit : block.exits)
        {
            text
                << "  exit="
                << vuAnalysisNativeExitName(exit.kind)
                << " possible="
                << (exit.possible ? 1 : 0)
                << " pairs=" << exit.minimumPairs
                << ".." << exit.maximumPairs
                << " cycles=" << exit.minimumCycles
                << ".." << exit.maximumCycles
                << "\n";
        }
    }
    return text.str();
}

std::string_view vuAnalysisBarrierName(
    VuAnalysisBarrierKind kind)
{
    switch (kind)
    {
    case VuAnalysisBarrierKind::NativeHelper:
        return "native-helper";
    case VuAnalysisBarrierKind::XgkickStart:
        return "xgkick-start";
    case VuAnalysisBarrierKind::XgkickAdvance:
        return "xgkick-advance";
    case VuAnalysisBarrierKind::DebugObserver:
        return "debug-observer";
    case VuAnalysisBarrierKind::Verification:
        return "verification";
    case VuAnalysisBarrierKind::UnknownMemory:
        return "unknown-memory";
    case VuAnalysisBarrierKind::UnsupportedOperation:
        return "unsupported-operation";
    }
    return "unknown";
}

std::string_view vuAnalysisBarrierPhaseName(
    VuAnalysisBarrierPhase phase)
{
    switch (phase)
    {
    case VuAnalysisBarrierPhase::BeforePair:
        return "before-pair";
    case VuAnalysisBarrierPhase::PairExecution:
        return "pair-execution";
    case VuAnalysisBarrierPhase::AfterPair:
        return "after-pair";
    case VuAnalysisBarrierPhase::WholePair:
        return "whole-pair";
    }
    return "unknown";
}

std::string_view vuAnalysisKnownStateName(
    VuAnalysisKnownState state)
{
    switch (state)
    {
    case VuAnalysisKnownState::Dynamic:
        return "dynamic";
    case VuAnalysisKnownState::Inactive:
        return "inactive";
    case VuAnalysisKnownState::Active:
        return "active";
    }
    return "unknown";
}

std::string_view vuAnalysisSuccessorName(
    VuAnalysisSuccessorKind kind)
{
    switch (kind)
    {
    case VuAnalysisSuccessorKind::Fallthrough:
        return "fallthrough";
    case VuAnalysisSuccessorKind::PairLimit:
        return "pair-limit";
    case VuAnalysisSuccessorKind::BranchTaken:
        return "branch-taken";
    case VuAnalysisSuccessorKind::BranchNotTaken:
        return "branch-not-taken";
    case VuAnalysisSuccessorKind::IndirectBranch:
        return "indirect-branch";
    case VuAnalysisSuccessorKind::XgkickResume:
        return "xgkick-resume";
    }
    return "unknown";
}

std::string_view vuAnalysisNativeExitName(
    VuAnalysisNativeExit exit)
{
    switch (exit)
    {
    case VuAnalysisNativeExit::BlockComplete:
        return "block-complete";
    case VuAnalysisNativeExit::CycleBudget:
        return "cycle-budget";
    case VuAnalysisNativeExit::Inactive:
        return "inactive";
    case VuAnalysisNativeExit::ProgramEnded:
        return "program-ended";
    case VuAnalysisNativeExit::CodeBounds:
        return "code-bounds";
    case VuAnalysisNativeExit::XgkickBoundary:
        return "xgkick-boundary";
    case VuAnalysisNativeExit::DebugObserver:
        return "debug-observer";
    case VuAnalysisNativeExit::CodeInvalidated:
        return "code-invalidated";
    case VuAnalysisNativeExit::UnsupportedInstruction:
        return "unsupported-instruction";
    case VuAnalysisNativeExit::Fault:
        return "fault";
    }
    return "unknown";
}

std::string_view vuAnalysisBlockTerminalName(
    VuAnalysisBlockTerminal terminal)
{
    switch (terminal)
    {
    case VuAnalysisBlockTerminal::PairLimit:
        return "pair-limit";
    case VuAnalysisBlockTerminal::CodeWrap:
        return "code-wrap";
    case VuAnalysisBlockTerminal::BranchBoundary:
        return "branch-boundary";
    case VuAnalysisBlockTerminal::ProgramEnd:
        return "program-end";
    case VuAnalysisBlockTerminal::XgkickBoundary:
        return "xgkick-boundary";
    case VuAnalysisBlockTerminal::UnsupportedInstruction:
        return "unsupported-instruction";
    case VuAnalysisBlockTerminal::CodeBounds:
        return "code-bounds";
    }
    return "unknown";
}
