#ifndef PS2_VU_IR_H
#define PS2_VU_IR_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

enum class VuIrOpcode : uint8_t
{
    Nop,
    Loi,

    UpperAddBc,
    UpperSubBc,
    UpperMaddBc,
    UpperMsubBc,
    UpperMaxBc,
    UpperMiniBc,
    UpperMulBc,
    UpperMulQ,
    UpperMaxI,
    UpperMulI,
    UpperMiniI,
    UpperAddQ,
    UpperMaddQ,
    UpperAddI,
    UpperMaddI,
    UpperSubQ,
    UpperMsubQ,
    UpperSubI,
    UpperMsubI,
    UpperAdd,
    UpperMadd,
    UpperMul,
    UpperMax,
    UpperSub,
    UpperMsub,
    UpperOpMsub,
    UpperMini,
    UpperAddaBc,
    UpperSubaBc,
    UpperMaddaBc,
    UpperMsubaBc,
    UpperItof0,
    UpperItof4,
    UpperItof12,
    UpperItof15,
    UpperFtoi0,
    UpperFtoi4,
    UpperFtoi12,
    UpperFtoi15,
    UpperMulaBc,
    UpperMulaQ,
    UpperAbs,
    UpperMulaI,
    UpperClip,
    UpperAddaQ,
    UpperMaddaQ,
    UpperAddaI,
    UpperMaddaI,
    UpperSubaQ,
    UpperMsubaQ,
    UpperSubaI,
    UpperMsubaI,
    UpperAdda,
    UpperMadda,
    UpperMula,
    UpperSuba,
    UpperMsuba,
    UpperOpmula,

    LowerLq,
    LowerSq,
    LowerIlw,
    LowerIsw,
    LowerIaddiu,
    LowerIsubiu,
    LowerFceq,
    LowerFcset,
    LowerFcand,
    LowerFcor,
    LowerFseq,
    LowerFsset,
    LowerFsand,
    LowerFsor,
    LowerFmand,
    LowerFmeq,
    LowerFmor,
    LowerFcget,
    LowerB,
    LowerBal,
    LowerJr,
    LowerJalr,
    LowerIbeq,
    LowerIbne,
    LowerIbltz,
    LowerIbgtz,
    LowerIblez,
    LowerIbgez,
    LowerIadd,
    LowerIsub,
    LowerIaddi,
    LowerIand,
    LowerIor,
    LowerMove,
    LowerMr32,
    LowerLqi,
    LowerSqi,
    LowerLqd,
    LowerSqd,
    LowerDiv,
    LowerSqrt,
    LowerRsqrt,
    LowerWaitQ,
    LowerMtir,
    LowerMfir,
    LowerIlwr,
    LowerIswr,
    LowerRnext,
    LowerRget,
    LowerRinit,
    LowerRxor,
    LowerMfp,
    LowerXtop,
    LowerXitop,
    LowerXgkick,
    LowerEsadd,
    LowerErsadd,
    LowerEleng,
    LowerErleng,
    LowerErcpr,
    LowerWaitP,
    LowerEatan,

    Unsupported,
};

enum VuIrOpFlag : uint32_t
{
    VuIrOpNone = 0u,
    VuIrOpReadsAcc = 1u << 0u,
    VuIrOpWritesAcc = 1u << 1u,
    VuIrOpReadsQ = 1u << 2u,
    VuIrOpWritesQ = 1u << 3u,
    VuIrOpReadsP = 1u << 4u,
    VuIrOpWritesP = 1u << 5u,
    VuIrOpReadsI = 1u << 6u,
    VuIrOpWritesI = 1u << 7u,
    VuIrOpReadsStatus = 1u << 8u,
    VuIrOpWritesStatus = 1u << 9u,
    VuIrOpReadsMac = 1u << 10u,
    VuIrOpWritesMac = 1u << 11u,
    VuIrOpReadsClip = 1u << 12u,
    VuIrOpWritesClip = 1u << 13u,
    VuIrOpReadsVuData = 1u << 14u,
    VuIrOpWritesVuData = 1u << 15u,
    VuIrOpBranch = 1u << 16u,
    VuIrOpIndirectBranch = 1u << 17u,
    VuIrOpLink = 1u << 18u,
    VuIrOpFmac = 1u << 19u,
    VuIrOpXgkick = 1u << 20u,
    VuIrOpExternalEffect = 1u << 21u,
    VuIrOpUnsupported = 1u << 22u,
    VuIrOpQBarrier = 1u << 23u,
    VuIrOpPBarrier = 1u << 24u,
    VuIrOpReadsTop = 1u << 25u,
    VuIrOpReadsItop = 1u << 26u,
};

enum class VuIrPairOrder : uint8_t
{
    UpperThenLower,
    LowerThenUpper,
    UpperThenLoi,
};

enum VuIrPairFlag : uint8_t
{
    VuIrPairNone = 0u,
    VuIrPairImmediate = 1u << 0u,
    VuIrPairEnd = 1u << 1u,
    VuIrPairBranch = 1u << 2u,
    VuIrPairXgkick = 1u << 3u,
    VuIrPairUnsupported = 1u << 4u,
    VuIrPairAdvancesPipelines = 1u << 5u,
};

struct VuIrOperation
{
    uint32_t vfReadMask = 0u;
    uint32_t vfWriteMask = 0u;
    uint32_t flags = VuIrOpNone;
    uint16_t viReadMask = 0u;
    uint16_t viWriteMask = 0u;
    VuIrOpcode opcode = VuIrOpcode::Unsupported;
    uint8_t destinationMask = 0u;
    uint8_t latency = 0u;
    uint8_t selector = 0u;
};

struct VuIrInstructionPair
{
    uint32_t pc = 0u;
    uint32_t lowerWord = 0u;
    uint32_t upperWord = 0u;
    VuIrOperation upper;
    VuIrOperation lower;
    VuIrPairOrder order = VuIrPairOrder::UpperThenLower;
    uint8_t flags = VuIrPairNone;
    uint8_t cycles = 1u;
};

static_assert(
    sizeof(VuIrInstructionPair) <= 64u,
    "VU instruction-pair IR must remain compact");

enum class VuIrBlockExit : uint8_t
{
    PairLimit,
    BranchBoundary,
    ProgramEndBoundary,
    XgkickBoundary,
    UnsupportedInstruction,
    CodeBounds,
};

enum class VuIrBlockForm : uint8_t
{
    Basic,
    LinearTrace,
};

struct VuIrBlock
{
    uint32_t entryPc = 0u;
    uint32_t codeSize = 0u;
    VuIrBlockForm form = VuIrBlockForm::Basic;
    VuIrBlockExit exit = VuIrBlockExit::PairLimit;
    std::vector<VuIrInstructionPair> pairs;
};

struct VuIrVerificationError
{
    uint32_t pc = 0u;
    uint32_t lowerWord = 0u;
    uint32_t upperWord = 0u;
    std::string message;
};

[[nodiscard]] VuIrInstructionPair decodeVuIrInstructionPair(
    uint32_t pc, uint32_t lowerWord, uint32_t upperWord);
[[nodiscard]] VuIrBlock decodeVuIrBlock(
    const uint8_t *code, uint32_t codeSize,
    uint32_t entryPc, uint32_t maximumPairs,
    VuIrBlockForm form = VuIrBlockForm::Basic);
[[nodiscard]] bool verifyVuIrInstructionPair(
    const VuIrInstructionPair &pair,
    VuIrVerificationError *error = nullptr);
[[nodiscard]] bool verifyVuIrBlock(
    const VuIrBlock &block,
    VuIrVerificationError *error = nullptr);
[[nodiscard]] std::string_view vuIrOpcodeName(VuIrOpcode opcode);
[[nodiscard]] std::string_view vuIrBlockExitName(VuIrBlockExit exit);

[[nodiscard]] constexpr bool vuIrHasOpFlag(
    const VuIrOperation &operation, VuIrOpFlag flag)
{
    return (operation.flags & static_cast<uint32_t>(flag)) != 0u;
}

[[nodiscard]] constexpr bool vuIrHasPairFlag(
    const VuIrInstructionPair &pair, VuIrPairFlag flag)
{
    return (pair.flags & static_cast<uint8_t>(flag)) != 0u;
}

#endif
