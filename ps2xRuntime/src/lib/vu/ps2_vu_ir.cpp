#include "runtime/ps2_vu_ir.h"

#include "ps2_vu1_detail.h"

#include <cstring>
#include <utility>

namespace
{
    constexpr uint32_t vfMask(uint8_t reg)
    {
        return 1u << (reg & 0x1fu);
    }

    constexpr uint16_t viMask(uint8_t reg)
    {
        return static_cast<uint16_t>(1u << (reg & 0x0fu));
    }

    void addVfRead(VuIrOperation &operation, uint8_t reg)
    {
        operation.vfReadMask |= vfMask(reg);
    }

    void addVfWrite(
        VuIrOperation &operation, uint8_t reg, uint8_t destination)
    {
        if (reg != 0u && destination != 0u)
            operation.vfWriteMask |= vfMask(reg);
    }

    void addViRead(VuIrOperation &operation, uint8_t reg)
    {
        operation.viReadMask |= viMask(reg);
    }

    void addViWrite(VuIrOperation &operation, uint8_t reg)
    {
        if (reg != 0u)
            operation.viWriteMask |= viMask(reg);
    }

    void markFmac(VuIrOperation &operation)
    {
        operation.flags |=
            VuIrOpFmac | VuIrOpWritesMac | VuIrOpWritesStatus;
        operation.latency = 4u;
    }

    void markPProducer(VuIrOperation &operation, uint8_t latency)
    {
        operation.flags |= VuIrOpWritesP | VuIrOpPBarrier;
        operation.latency = latency;
    }

    void markUnsupported(VuIrOperation &operation)
    {
        operation.opcode = VuIrOpcode::Unsupported;
        operation.flags |= VuIrOpUnsupported;
    }

    void markRecognizedUnsupported(
        VuIrOperation &operation, VuIrOpcode opcode)
    {
        operation.opcode = opcode;
        operation.flags |= VuIrOpUnsupported;
    }

    VuIrOperation decodeUpper(uint32_t word)
    {
        VuIrOperation operation;
        const uint8_t destination = DEST(word);
        const uint8_t ft = FT(word);
        const uint8_t fs = FS(word);
        const uint8_t fd = FD(word);
        const uint8_t primary = static_cast<uint8_t>(word & 0x3fu);
        const uint8_t special = static_cast<uint8_t>(
            (word & 0x3u) | ((word >> 4u) & 0x7cu));
        operation.destinationMask = destination;
        if (primary <= 0x1bu)
            operation.selector = static_cast<uint8_t>(primary & 3u);

        const auto vfBinaryDestination =
            [&](VuIrOpcode opcode, bool readsAcc, bool fmac)
        {
            operation.opcode = opcode;
            addVfRead(operation, fs);
            addVfRead(operation, ft);
            addVfWrite(operation, fd, destination);
            if (readsAcc)
                operation.flags |= VuIrOpReadsAcc;
            if (fmac)
                markFmac(operation);
        };
        const auto vfScalarDestination =
            [&](VuIrOpcode opcode, uint32_t scalarFlag, bool readsAcc,
                bool fmac)
        {
            operation.opcode = opcode;
            addVfRead(operation, fs);
            addVfWrite(operation, fd, destination);
            operation.flags |= scalarFlag;
            if (readsAcc)
                operation.flags |= VuIrOpReadsAcc;
            if (fmac)
                markFmac(operation);
        };
        const auto accBinary =
            [&](VuIrOpcode opcode, bool readsAcc)
        {
            operation.opcode = opcode;
            addVfRead(operation, fs);
            addVfRead(operation, ft);
            operation.flags |= VuIrOpWritesAcc;
            if (readsAcc)
                operation.flags |= VuIrOpReadsAcc;
            markFmac(operation);
        };
        const auto accScalar =
            [&](VuIrOpcode opcode, uint32_t scalarFlag, bool readsAcc)
        {
            operation.opcode = opcode;
            addVfRead(operation, fs);
            operation.flags |= VuIrOpWritesAcc | scalarFlag;
            if (readsAcc)
                operation.flags |= VuIrOpReadsAcc;
            markFmac(operation);
        };

        switch (primary)
        {
        case 0x00u:
        case 0x01u:
        case 0x02u:
        case 0x03u:
            vfBinaryDestination(VuIrOpcode::UpperAddBc, false, true);
            return operation;
        case 0x04u:
        case 0x05u:
        case 0x06u:
        case 0x07u:
            vfBinaryDestination(VuIrOpcode::UpperSubBc, false, true);
            return operation;
        case 0x08u:
        case 0x09u:
        case 0x0au:
        case 0x0bu:
            vfBinaryDestination(VuIrOpcode::UpperMaddBc, true, true);
            return operation;
        case 0x0cu:
        case 0x0du:
        case 0x0eu:
        case 0x0fu:
            vfBinaryDestination(VuIrOpcode::UpperMsubBc, true, true);
            return operation;
        case 0x10u:
        case 0x11u:
        case 0x12u:
        case 0x13u:
            vfBinaryDestination(VuIrOpcode::UpperMaxBc, false, false);
            return operation;
        case 0x14u:
        case 0x15u:
        case 0x16u:
        case 0x17u:
            vfBinaryDestination(VuIrOpcode::UpperMiniBc, false, false);
            return operation;
        case 0x18u:
        case 0x19u:
        case 0x1au:
        case 0x1bu:
            vfBinaryDestination(VuIrOpcode::UpperMulBc, false, true);
            return operation;
        case 0x1cu:
            vfScalarDestination(
                VuIrOpcode::UpperMulQ, VuIrOpReadsQ, false, true);
            return operation;
        case 0x1du:
            vfScalarDestination(
                VuIrOpcode::UpperMaxI, VuIrOpReadsI, false, false);
            return operation;
        case 0x1eu:
            vfScalarDestination(
                VuIrOpcode::UpperMulI, VuIrOpReadsI, false, true);
            return operation;
        case 0x1fu:
            vfScalarDestination(
                VuIrOpcode::UpperMiniI, VuIrOpReadsI, false, false);
            return operation;
        case 0x20u:
            vfScalarDestination(
                VuIrOpcode::UpperAddQ, VuIrOpReadsQ, false, true);
            return operation;
        case 0x21u:
            vfScalarDestination(
                VuIrOpcode::UpperMaddQ, VuIrOpReadsQ, true, true);
            return operation;
        case 0x22u:
            vfScalarDestination(
                VuIrOpcode::UpperAddI, VuIrOpReadsI, false, true);
            return operation;
        case 0x23u:
            vfScalarDestination(
                VuIrOpcode::UpperMaddI, VuIrOpReadsI, true, true);
            return operation;
        case 0x24u:
            vfScalarDestination(
                VuIrOpcode::UpperSubQ, VuIrOpReadsQ, false, true);
            return operation;
        case 0x25u:
            vfScalarDestination(
                VuIrOpcode::UpperMsubQ, VuIrOpReadsQ, true, true);
            return operation;
        case 0x26u:
            vfScalarDestination(
                VuIrOpcode::UpperSubI, VuIrOpReadsI, false, true);
            return operation;
        case 0x27u:
            vfScalarDestination(
                VuIrOpcode::UpperMsubI, VuIrOpReadsI, true, true);
            return operation;
        case 0x28u:
            vfBinaryDestination(VuIrOpcode::UpperAdd, false, true);
            return operation;
        case 0x29u:
            vfBinaryDestination(VuIrOpcode::UpperMadd, true, true);
            return operation;
        case 0x2au:
            vfBinaryDestination(VuIrOpcode::UpperMul, false, true);
            return operation;
        case 0x2bu:
            vfBinaryDestination(VuIrOpcode::UpperMax, false, false);
            return operation;
        case 0x2cu:
            vfBinaryDestination(VuIrOpcode::UpperSub, false, true);
            return operation;
        case 0x2du:
            vfBinaryDestination(VuIrOpcode::UpperMsub, true, true);
            return operation;
        case 0x2eu:
            vfBinaryDestination(VuIrOpcode::UpperOpMsub, true, true);
            return operation;
        case 0x2fu:
            vfBinaryDestination(VuIrOpcode::UpperMini, false, false);
            return operation;
        case 0x3cu:
        case 0x3du:
        case 0x3eu:
        case 0x3fu:
            break;
        default:
            markUnsupported(operation);
            return operation;
        }

        operation.selector = 0u;
        if (special <= 0x0fu ||
            (special >= 0x18u && special <= 0x1bu))
        {
            operation.selector = static_cast<uint8_t>(special & 3u);
        }
        switch (special)
        {
        case 0x00u:
        case 0x01u:
        case 0x02u:
        case 0x03u:
            accBinary(VuIrOpcode::UpperAddaBc, false);
            return operation;
        case 0x04u:
        case 0x05u:
        case 0x06u:
        case 0x07u:
            accBinary(VuIrOpcode::UpperSubaBc, false);
            return operation;
        case 0x08u:
        case 0x09u:
        case 0x0au:
        case 0x0bu:
            accBinary(VuIrOpcode::UpperMaddaBc, true);
            return operation;
        case 0x0cu:
        case 0x0du:
        case 0x0eu:
        case 0x0fu:
            accBinary(VuIrOpcode::UpperMsubaBc, true);
            return operation;
        case 0x10u:
        case 0x11u:
        case 0x12u:
        case 0x13u:
        case 0x14u:
        case 0x15u:
        case 0x16u:
        case 0x17u:
        {
            static constexpr VuIrOpcode conversionOpcodes[] = {
                VuIrOpcode::UpperItof0,
                VuIrOpcode::UpperItof4,
                VuIrOpcode::UpperItof12,
                VuIrOpcode::UpperItof15,
                VuIrOpcode::UpperFtoi0,
                VuIrOpcode::UpperFtoi4,
                VuIrOpcode::UpperFtoi12,
                VuIrOpcode::UpperFtoi15,
            };
            operation.opcode = conversionOpcodes[special - 0x10u];
            addVfRead(operation, fs);
            addVfWrite(operation, ft, destination);
            return operation;
        }
        case 0x18u:
        case 0x19u:
        case 0x1au:
        case 0x1bu:
            accBinary(VuIrOpcode::UpperMulaBc, false);
            return operation;
        case 0x1cu:
            accScalar(VuIrOpcode::UpperMulaQ, VuIrOpReadsQ, false);
            return operation;
        case 0x1du:
            operation.opcode = VuIrOpcode::UpperAbs;
            addVfRead(operation, fs);
            addVfWrite(operation, ft, destination);
            return operation;
        case 0x1eu:
            accScalar(VuIrOpcode::UpperMulaI, VuIrOpReadsI, false);
            return operation;
        case 0x1fu:
            operation.opcode = VuIrOpcode::UpperClip;
            addVfRead(operation, fs);
            addVfRead(operation, ft);
            operation.flags |= VuIrOpReadsClip | VuIrOpWritesClip;
            return operation;
        case 0x20u:
            accScalar(VuIrOpcode::UpperAddaQ, VuIrOpReadsQ, false);
            return operation;
        case 0x21u:
            accScalar(VuIrOpcode::UpperMaddaQ, VuIrOpReadsQ, true);
            return operation;
        case 0x22u:
            accScalar(VuIrOpcode::UpperAddaI, VuIrOpReadsI, false);
            return operation;
        case 0x23u:
            accScalar(VuIrOpcode::UpperMaddaI, VuIrOpReadsI, true);
            return operation;
        case 0x24u:
            accScalar(VuIrOpcode::UpperSubaQ, VuIrOpReadsQ, false);
            return operation;
        case 0x25u:
            accScalar(VuIrOpcode::UpperMsubaQ, VuIrOpReadsQ, true);
            return operation;
        case 0x26u:
            accScalar(VuIrOpcode::UpperSubaI, VuIrOpReadsI, false);
            return operation;
        case 0x27u:
            accScalar(VuIrOpcode::UpperMsubaI, VuIrOpReadsI, true);
            return operation;
        case 0x28u:
            accBinary(VuIrOpcode::UpperAdda, false);
            return operation;
        case 0x29u:
            accBinary(VuIrOpcode::UpperMadda, true);
            return operation;
        case 0x2au:
            accBinary(VuIrOpcode::UpperMula, false);
            return operation;
        case 0x2cu:
            accBinary(VuIrOpcode::UpperSuba, false);
            return operation;
        case 0x2du:
            accBinary(VuIrOpcode::UpperMsuba, true);
            return operation;
        case 0x2eu:
            accBinary(VuIrOpcode::UpperOpmula, false);
            return operation;
        case 0x2fu:
        case 0x30u:
            operation.opcode = VuIrOpcode::Nop;
            return operation;
        default:
            markUnsupported(operation);
            return operation;
        }
    }

    VuIrOperation decodeLower(uint32_t word)
    {
        VuIrOperation operation;
        if (word == 0u || word == 0x8000033cu)
        {
            operation.opcode = VuIrOpcode::Nop;
            return operation;
        }

        const uint8_t primary =
            static_cast<uint8_t>((word >> 25u) & 0x7fu);
        const uint8_t vfT = FT(word);
        const uint8_t vfS = FS(word);
        const uint8_t viT = VIT(word);
        const uint8_t viS = VIS(word);
        const uint8_t viD = VID(word);
        const uint8_t destination =
            static_cast<uint8_t>((word >> 21u) & 0x0fu);

        const auto branch =
            [&](VuIrOpcode opcode, bool indirect, bool link)
        {
            operation.opcode = opcode;
            operation.flags |= VuIrOpBranch;
            if (indirect)
                operation.flags |= VuIrOpIndirectBranch;
            if (link)
                operation.flags |= VuIrOpLink;
        };

        switch (primary)
        {
        case 0x00u:
            operation.opcode = VuIrOpcode::LowerLq;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addVfWrite(operation, vfT, destination);
            operation.flags |= VuIrOpReadsVuData;
            return operation;
        case 0x01u:
            operation.opcode = VuIrOpcode::LowerSq;
            operation.destinationMask = destination;
            addVfRead(operation, vfS);
            addViRead(operation, viT);
            operation.flags |= VuIrOpWritesVuData;
            return operation;
        case 0x04u:
            operation.opcode = VuIrOpcode::LowerIlw;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            operation.flags |= VuIrOpReadsVuData;
            return operation;
        case 0x05u:
            operation.opcode = VuIrOpcode::LowerIsw;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addViRead(operation, viT);
            operation.flags |= VuIrOpWritesVuData;
            return operation;
        case 0x08u:
            operation.opcode = VuIrOpcode::LowerIaddiu;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x09u:
            operation.opcode = VuIrOpcode::LowerIsubiu;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x10u:
            operation.opcode = VuIrOpcode::LowerFceq;
            operation.flags |= VuIrOpReadsClip;
            addViWrite(operation, 1u);
            return operation;
        case 0x11u:
            operation.opcode = VuIrOpcode::LowerFcset;
            operation.flags |= VuIrOpWritesClip;
            return operation;
        case 0x12u:
            operation.opcode = VuIrOpcode::LowerFcand;
            operation.flags |= VuIrOpReadsClip;
            addViWrite(operation, 1u);
            return operation;
        case 0x13u:
            operation.opcode = VuIrOpcode::LowerFcor;
            operation.flags |= VuIrOpReadsClip;
            addViWrite(operation, 1u);
            return operation;
        case 0x14u:
            operation.opcode = VuIrOpcode::LowerFseq;
            operation.flags |= VuIrOpReadsStatus;
            addViWrite(operation, 1u);
            return operation;
        case 0x15u:
            operation.opcode = VuIrOpcode::LowerFsset;
            operation.flags |= VuIrOpWritesStatus;
            return operation;
        case 0x16u:
            operation.opcode = VuIrOpcode::LowerFsand;
            operation.flags |= VuIrOpReadsStatus;
            addViWrite(operation, 1u);
            return operation;
        case 0x17u:
            operation.opcode = VuIrOpcode::LowerFsor;
            operation.flags |= VuIrOpReadsStatus;
            addViWrite(operation, 1u);
            return operation;
        case 0x18u:
            operation.opcode = VuIrOpcode::LowerFmand;
            operation.flags |= VuIrOpReadsMac;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x1au:
            operation.opcode = VuIrOpcode::LowerFmeq;
            operation.flags |= VuIrOpReadsMac;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x1cu:
            operation.opcode = VuIrOpcode::LowerFmor;
            operation.flags |= VuIrOpReadsMac;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x20u:
            branch(VuIrOpcode::LowerB, false, false);
            return operation;
        case 0x21u:
            branch(VuIrOpcode::LowerBal, false, true);
            addViWrite(operation, viT);
            return operation;
        case 0x24u:
            branch(VuIrOpcode::LowerJr, true, false);
            addViRead(operation, viS);
            return operation;
        case 0x25u:
            branch(VuIrOpcode::LowerJalr, true, true);
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x28u:
            branch(VuIrOpcode::LowerIbeq, false, false);
            addViRead(operation, viS);
            addViRead(operation, viT);
            return operation;
        case 0x29u:
            branch(VuIrOpcode::LowerIbne, false, false);
            addViRead(operation, viS);
            addViRead(operation, viT);
            return operation;
        case 0x2cu:
            branch(VuIrOpcode::LowerIbltz, false, false);
            addViRead(operation, viS);
            return operation;
        case 0x2du:
            branch(VuIrOpcode::LowerIbgtz, false, false);
            addViRead(operation, viS);
            return operation;
        case 0x2eu:
            branch(VuIrOpcode::LowerIblez, false, false);
            addViRead(operation, viS);
            return operation;
        case 0x2fu:
            branch(VuIrOpcode::LowerIbgez, false, false);
            addViRead(operation, viS);
            return operation;
        case 0x40u:
            break;
        default:
            markUnsupported(operation);
            return operation;
        }

        const uint8_t function = static_cast<uint8_t>(word & 0x3fu);
        switch (function)
        {
        case 0x30u:
            operation.opcode = VuIrOpcode::LowerIadd;
            addViRead(operation, viS);
            addViRead(operation, viT);
            addViWrite(operation, viD);
            return operation;
        case 0x31u:
            operation.opcode = VuIrOpcode::LowerIsub;
            addViRead(operation, viS);
            addViRead(operation, viT);
            addViWrite(operation, viD);
            return operation;
        case 0x32u:
            operation.opcode = VuIrOpcode::LowerIaddi;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            return operation;
        case 0x34u:
            operation.opcode = VuIrOpcode::LowerIand;
            addViRead(operation, viS);
            addViRead(operation, viT);
            addViWrite(operation, viD);
            return operation;
        case 0x35u:
            operation.opcode = VuIrOpcode::LowerIor;
            addViRead(operation, viS);
            addViRead(operation, viT);
            addViWrite(operation, viD);
            return operation;
        case 0x3cu:
        case 0x3du:
        case 0x3eu:
        case 0x3fu:
            break;
        default:
            markUnsupported(operation);
            return operation;
        }

        const uint8_t special = static_cast<uint8_t>(
            (word & 0x3u) | ((word >> 4u) & 0x7cu));
        switch (special)
        {
        case 0x30u:
            operation.opcode = VuIrOpcode::LowerMove;
            operation.destinationMask = destination;
            addVfRead(operation, vfS);
            addVfWrite(operation, vfT, destination);
            return operation;
        case 0x31u:
            operation.opcode = VuIrOpcode::LowerMr32;
            operation.destinationMask = destination;
            addVfRead(operation, vfS);
            addVfWrite(operation, vfT, destination);
            return operation;
        case 0x34u:
            operation.opcode = VuIrOpcode::LowerLqi;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addViWrite(operation, viS);
            addVfWrite(operation, vfT, destination);
            operation.flags |= VuIrOpReadsVuData;
            return operation;
        case 0x35u:
            operation.opcode = VuIrOpcode::LowerSqi;
            operation.destinationMask = destination;
            addVfRead(operation, vfS);
            addViRead(operation, viT);
            addViWrite(operation, viT);
            operation.flags |= VuIrOpWritesVuData;
            return operation;
        case 0x36u:
            operation.opcode = VuIrOpcode::LowerLqd;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addViWrite(operation, viS);
            addVfWrite(operation, vfT, destination);
            operation.flags |= VuIrOpReadsVuData;
            return operation;
        case 0x37u:
            operation.opcode = VuIrOpcode::LowerSqd;
            operation.destinationMask = destination;
            addVfRead(operation, vfS);
            addViRead(operation, viT);
            addViWrite(operation, viT);
            operation.flags |= VuIrOpWritesVuData;
            return operation;
        case 0x38u:
            operation.opcode = VuIrOpcode::LowerDiv;
            operation.selector = destination;
            addVfRead(operation, vfS);
            addVfRead(operation, vfT);
            operation.flags |= VuIrOpWritesQ | VuIrOpQBarrier;
            operation.latency = 7u;
            return operation;
        case 0x39u:
            operation.opcode = VuIrOpcode::LowerSqrt;
            operation.selector = destination;
            addVfRead(operation, vfT);
            operation.flags |= VuIrOpWritesQ | VuIrOpQBarrier;
            operation.latency = 7u;
            return operation;
        case 0x3au:
            operation.opcode = VuIrOpcode::LowerRsqrt;
            operation.selector = destination;
            addVfRead(operation, vfS);
            addVfRead(operation, vfT);
            operation.flags |= VuIrOpWritesQ | VuIrOpQBarrier;
            operation.latency = 13u;
            return operation;
        case 0x3bu:
            operation.opcode = VuIrOpcode::LowerWaitQ;
            operation.flags |= VuIrOpReadsQ | VuIrOpQBarrier;
            return operation;
        case 0x3cu:
            operation.opcode = VuIrOpcode::LowerMtir;
            operation.selector =
                static_cast<uint8_t>(destination & 3u);
            addVfRead(operation, vfS);
            addViWrite(operation, viT);
            return operation;
        case 0x3du:
            operation.opcode = VuIrOpcode::LowerMfir;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addVfWrite(operation, vfT, destination);
            return operation;
        case 0x3eu:
            operation.opcode = VuIrOpcode::LowerIlwr;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addViWrite(operation, viT);
            operation.flags |= VuIrOpReadsVuData;
            return operation;
        case 0x3fu:
            operation.opcode = VuIrOpcode::LowerIswr;
            operation.destinationMask = destination;
            addViRead(operation, viS);
            addViRead(operation, viT);
            operation.flags |= VuIrOpWritesVuData;
            return operation;
        case 0x40u:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerRnext);
            operation.destinationMask = destination;
            addVfWrite(operation, vfT, destination);
            return operation;
        case 0x41u:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerRget);
            operation.destinationMask = destination;
            addVfWrite(operation, vfT, destination);
            return operation;
        case 0x42u:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerRinit);
            operation.selector =
                static_cast<uint8_t>(destination & 3u);
            addVfRead(operation, vfS);
            return operation;
        case 0x43u:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerRxor);
            operation.selector =
                static_cast<uint8_t>(destination & 3u);
            addVfRead(operation, vfS);
            return operation;
        case 0x64u:
            operation.opcode = VuIrOpcode::LowerMfp;
            operation.destinationMask = destination;
            operation.flags |= VuIrOpReadsP;
            addVfWrite(operation, vfT, destination);
            return operation;
        case 0x68u:
            operation.opcode = VuIrOpcode::LowerXtop;
            addViWrite(operation, viT);
            operation.flags |= VuIrOpReadsTop;
            return operation;
        case 0x69u:
            operation.opcode = VuIrOpcode::LowerXitop;
            addViWrite(operation, viT);
            operation.flags |= VuIrOpReadsItop;
            return operation;
        case 0x6cu:
            operation.opcode = VuIrOpcode::LowerXgkick;
            addViRead(operation, viS);
            operation.flags |=
                VuIrOpReadsVuData | VuIrOpXgkick | VuIrOpExternalEffect;
            return operation;
        case 0x70u:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerEsadd);
            addVfRead(operation, vfS);
            markPProducer(operation, 11u);
            return operation;
        case 0x71u:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerErsadd);
            addVfRead(operation, vfS);
            markPProducer(operation, 18u);
            return operation;
        case 0x72u:
            operation.opcode = VuIrOpcode::LowerEleng;
            addVfRead(operation, vfS);
            markPProducer(operation, 18u);
            return operation;
        case 0x73u:
            operation.opcode = VuIrOpcode::LowerErleng;
            addVfRead(operation, vfS);
            markPProducer(operation, 24u);
            return operation;
        case 0x7au:
            operation.opcode = VuIrOpcode::LowerErcpr;
            operation.selector =
                static_cast<uint8_t>(destination & 3u);
            addVfRead(operation, vfS);
            markPProducer(operation, 12u);
            return operation;
        case 0x7bu:
            operation.opcode = VuIrOpcode::LowerWaitP;
            operation.flags |= VuIrOpReadsP | VuIrOpPBarrier;
            return operation;
        case 0x7du:
            markRecognizedUnsupported(
                operation, VuIrOpcode::LowerEatan);
            operation.selector =
                static_cast<uint8_t>(destination & 3u);
            addVfRead(operation, vfS);
            markPProducer(operation, 54u);
            return operation;
        default:
            markUnsupported(operation);
            return operation;
        }
    }

    bool sameOperation(
        const VuIrOperation &left, const VuIrOperation &right)
    {
        return
            left.vfReadMask == right.vfReadMask &&
            left.vfWriteMask == right.vfWriteMask &&
            left.flags == right.flags &&
            left.viReadMask == right.viReadMask &&
            left.viWriteMask == right.viWriteMask &&
            left.opcode == right.opcode &&
            left.destinationMask == right.destinationMask &&
            left.latency == right.latency &&
            left.selector == right.selector;
    }

    bool failVerification(
        const VuIrInstructionPair &pair, std::string message,
        VuIrVerificationError *error)
    {
        if (error)
        {
            *error = {
                .pc = pair.pc,
                .lowerWord = pair.lowerWord,
                .upperWord = pair.upperWord,
                .message = std::move(message),
            };
        }
        return false;
    }
}

VuIrInstructionPair decodeVuIrInstructionPair(
    uint32_t pc, uint32_t lowerWord, uint32_t upperWord)
{
    VuIrInstructionPair pair;
    pair.pc = pc;
    pair.lowerWord = lowerWord;
    pair.upperWord = upperWord;
    pair.flags = VuIrPairAdvancesPipelines;
    pair.upper = decodeUpper(upperWord);

    const bool immediate = (upperWord & 0x80000000u) != 0u;
    if (immediate)
    {
        pair.lower.opcode = VuIrOpcode::Loi;
        pair.lower.flags = VuIrOpWritesI;
        pair.order = VuIrPairOrder::UpperThenLoi;
        pair.flags |= VuIrPairImmediate;
    }
    else
    {
        pair.lower = decodeLower(lowerWord);
        if (vuLowerShouldRunBeforeUpper(upperWord, lowerWord))
            pair.order = VuIrPairOrder::LowerThenUpper;
    }

    if ((upperWord & 0x40000000u) != 0u)
        pair.flags |= VuIrPairEnd;
    if (vuIrHasOpFlag(pair.lower, VuIrOpBranch))
        pair.flags |= VuIrPairBranch;
    if (vuIrHasOpFlag(pair.lower, VuIrOpXgkick))
        pair.flags |= VuIrPairXgkick;
    if (vuIrHasOpFlag(pair.upper, VuIrOpUnsupported) ||
        vuIrHasOpFlag(pair.lower, VuIrOpUnsupported))
    {
        pair.flags |= VuIrPairUnsupported;
    }
    return pair;
}

VuIrBlock decodeVuIrBlock(
    const uint8_t *code, uint32_t codeSize,
    uint32_t entryPc, uint32_t maximumPairs)
{
    VuIrBlock block;
    block.entryPc = entryPc;
    block.codeSize = codeSize;
    if (!code || codeSize < 8u || (codeSize & 7u) != 0u ||
        (entryPc & 7u) != 0u || entryPc > codeSize - 8u)
    {
        block.exit = VuIrBlockExit::CodeBounds;
        return block;
    }

    VuIrBlockExit delayedExit = VuIrBlockExit::PairLimit;
    bool exitAfterCurrentPair = false;
    uint32_t pc = entryPc;
    for (uint32_t index = 0u; index < maximumPairs; ++index)
    {
        uint32_t lower = 0u;
        uint32_t upper = 0u;
        std::memcpy(&lower, code + pc, sizeof(lower));
        std::memcpy(&upper, code + pc + 4u, sizeof(upper));
        block.pairs.push_back(
            decodeVuIrInstructionPair(pc, lower, upper));
        const VuIrInstructionPair &pair = block.pairs.back();

        if (vuIrHasPairFlag(pair, VuIrPairUnsupported))
        {
            block.exit = VuIrBlockExit::UnsupportedInstruction;
            return block;
        }
        if (vuIrHasPairFlag(pair, VuIrPairXgkick))
        {
            block.exit = VuIrBlockExit::XgkickBoundary;
            return block;
        }
        if (exitAfterCurrentPair)
        {
            block.exit = delayedExit;
            return block;
        }
        if (vuIrHasPairFlag(pair, VuIrPairBranch))
        {
            delayedExit = VuIrBlockExit::BranchBoundary;
            exitAfterCurrentPair = true;
        }
        if (vuIrHasPairFlag(pair, VuIrPairEnd))
        {
            delayedExit = VuIrBlockExit::ProgramEndBoundary;
            exitAfterCurrentPair = true;
        }

        pc += 8u;
        if (pc >= codeSize)
            pc = 0u;
    }
    block.exit = VuIrBlockExit::PairLimit;
    return block;
}

bool verifyVuIrInstructionPair(
    const VuIrInstructionPair &pair,
    VuIrVerificationError *error)
{
    if ((pair.pc & 7u) != 0u)
        return failVerification(pair, "VU IR PC is not pair-aligned", error);
    if (pair.cycles != 1u)
        return failVerification(pair, "VU IR pair must cost one cycle", error);

    const VuIrInstructionPair expected = decodeVuIrInstructionPair(
        pair.pc, pair.lowerWord, pair.upperWord);
    if (!sameOperation(pair.upper, expected.upper))
        return failVerification(pair, "malformed upper operation", error);
    if (!sameOperation(pair.lower, expected.lower))
        return failVerification(pair, "malformed lower operation", error);
    if (pair.order != expected.order)
        return failVerification(pair, "invalid upper/lower ordering", error);
    if (pair.flags != expected.flags)
        return failVerification(pair, "invalid instruction-pair flags", error);
    if (error)
        *error = {};
    return true;
}

bool verifyVuIrBlock(
    const VuIrBlock &block,
    VuIrVerificationError *error)
{
    const bool validEntry =
        block.codeSize >= 8u &&
        (block.codeSize & 7u) == 0u &&
        (block.entryPc & 7u) == 0u &&
        block.entryPc <= block.codeSize - 8u;
    if (block.pairs.empty())
    {
        const VuIrBlockExit expectedExit =
            validEntry
                ? VuIrBlockExit::PairLimit
                : VuIrBlockExit::CodeBounds;
        if (block.exit == expectedExit)
        {
            if (error)
                *error = {};
            return true;
        }
        if (error)
        {
            *error = {
                .pc = block.entryPc,
                .message =
                    "empty VU IR block has an invalid entry or exit",
            };
        }
        return false;
    }
    if (!validEntry ||
        block.pairs.front().pc != block.entryPc)
    {
        return failVerification(
            block.pairs.front(), "invalid VU IR block entry", error);
    }

    uint32_t expectedPc = block.entryPc;
    bool delayedExitPending = false;
    VuIrBlockExit delayedExit = VuIrBlockExit::PairLimit;
    VuIrBlockExit expectedExit = VuIrBlockExit::PairLimit;
    size_t terminalIndex = block.pairs.size();
    for (size_t index = 0u; index < block.pairs.size(); ++index)
    {
        const VuIrInstructionPair &pair = block.pairs[index];
        if (pair.pc != expectedPc)
            return failVerification(pair, "nonsequential VU IR block", error);
        if (!verifyVuIrInstructionPair(pair, error))
            return false;

        if (vuIrHasPairFlag(pair, VuIrPairUnsupported))
        {
            expectedExit = VuIrBlockExit::UnsupportedInstruction;
            terminalIndex = index;
        }
        else if (vuIrHasPairFlag(pair, VuIrPairXgkick))
        {
            expectedExit = VuIrBlockExit::XgkickBoundary;
            terminalIndex = index;
        }
        else if (delayedExitPending)
        {
            expectedExit = delayedExit;
            terminalIndex = index;
        }

        if (terminalIndex != block.pairs.size())
            break;
        if (vuIrHasPairFlag(pair, VuIrPairBranch))
        {
            delayedExit = VuIrBlockExit::BranchBoundary;
            delayedExitPending = true;
        }
        if (vuIrHasPairFlag(pair, VuIrPairEnd))
        {
            delayedExit = VuIrBlockExit::ProgramEndBoundary;
            delayedExitPending = true;
        }

        expectedPc += 8u;
        if (expectedPc >= block.codeSize)
            expectedPc = 0u;
    }

    if (terminalIndex != block.pairs.size() &&
        terminalIndex + 1u != block.pairs.size())
    {
        return failVerification(
            block.pairs[terminalIndex],
            "VU IR block continues after its required exit",
            error);
    }
    if (block.exit != expectedExit)
    {
        return failVerification(
            block.pairs.back(),
            "VU IR block exit does not match its terminal pair",
            error);
    }
    if (error)
        *error = {};
    return true;
}

std::string_view vuIrOpcodeName(VuIrOpcode opcode)
{
#define VU_IR_OPCODE_NAME(name) \
    case VuIrOpcode::name:      \
        return #name
    switch (opcode)
    {
        VU_IR_OPCODE_NAME(Nop);
        VU_IR_OPCODE_NAME(Loi);
        VU_IR_OPCODE_NAME(UpperAddBc);
        VU_IR_OPCODE_NAME(UpperSubBc);
        VU_IR_OPCODE_NAME(UpperMaddBc);
        VU_IR_OPCODE_NAME(UpperMsubBc);
        VU_IR_OPCODE_NAME(UpperMaxBc);
        VU_IR_OPCODE_NAME(UpperMiniBc);
        VU_IR_OPCODE_NAME(UpperMulBc);
        VU_IR_OPCODE_NAME(UpperMulQ);
        VU_IR_OPCODE_NAME(UpperMaxI);
        VU_IR_OPCODE_NAME(UpperMulI);
        VU_IR_OPCODE_NAME(UpperMiniI);
        VU_IR_OPCODE_NAME(UpperAddQ);
        VU_IR_OPCODE_NAME(UpperMaddQ);
        VU_IR_OPCODE_NAME(UpperAddI);
        VU_IR_OPCODE_NAME(UpperMaddI);
        VU_IR_OPCODE_NAME(UpperSubQ);
        VU_IR_OPCODE_NAME(UpperMsubQ);
        VU_IR_OPCODE_NAME(UpperSubI);
        VU_IR_OPCODE_NAME(UpperMsubI);
        VU_IR_OPCODE_NAME(UpperAdd);
        VU_IR_OPCODE_NAME(UpperMadd);
        VU_IR_OPCODE_NAME(UpperMul);
        VU_IR_OPCODE_NAME(UpperMax);
        VU_IR_OPCODE_NAME(UpperSub);
        VU_IR_OPCODE_NAME(UpperMsub);
        VU_IR_OPCODE_NAME(UpperOpMsub);
        VU_IR_OPCODE_NAME(UpperMini);
        VU_IR_OPCODE_NAME(UpperAddaBc);
        VU_IR_OPCODE_NAME(UpperSubaBc);
        VU_IR_OPCODE_NAME(UpperMaddaBc);
        VU_IR_OPCODE_NAME(UpperMsubaBc);
        VU_IR_OPCODE_NAME(UpperItof0);
        VU_IR_OPCODE_NAME(UpperItof4);
        VU_IR_OPCODE_NAME(UpperItof12);
        VU_IR_OPCODE_NAME(UpperItof15);
        VU_IR_OPCODE_NAME(UpperFtoi0);
        VU_IR_OPCODE_NAME(UpperFtoi4);
        VU_IR_OPCODE_NAME(UpperFtoi12);
        VU_IR_OPCODE_NAME(UpperFtoi15);
        VU_IR_OPCODE_NAME(UpperMulaBc);
        VU_IR_OPCODE_NAME(UpperMulaQ);
        VU_IR_OPCODE_NAME(UpperAbs);
        VU_IR_OPCODE_NAME(UpperMulaI);
        VU_IR_OPCODE_NAME(UpperClip);
        VU_IR_OPCODE_NAME(UpperAddaQ);
        VU_IR_OPCODE_NAME(UpperMaddaQ);
        VU_IR_OPCODE_NAME(UpperAddaI);
        VU_IR_OPCODE_NAME(UpperMaddaI);
        VU_IR_OPCODE_NAME(UpperSubaQ);
        VU_IR_OPCODE_NAME(UpperMsubaQ);
        VU_IR_OPCODE_NAME(UpperSubaI);
        VU_IR_OPCODE_NAME(UpperMsubaI);
        VU_IR_OPCODE_NAME(UpperAdda);
        VU_IR_OPCODE_NAME(UpperMadda);
        VU_IR_OPCODE_NAME(UpperMula);
        VU_IR_OPCODE_NAME(UpperSuba);
        VU_IR_OPCODE_NAME(UpperMsuba);
        VU_IR_OPCODE_NAME(UpperOpmula);
        VU_IR_OPCODE_NAME(LowerLq);
        VU_IR_OPCODE_NAME(LowerSq);
        VU_IR_OPCODE_NAME(LowerIlw);
        VU_IR_OPCODE_NAME(LowerIsw);
        VU_IR_OPCODE_NAME(LowerIaddiu);
        VU_IR_OPCODE_NAME(LowerIsubiu);
        VU_IR_OPCODE_NAME(LowerFceq);
        VU_IR_OPCODE_NAME(LowerFcset);
        VU_IR_OPCODE_NAME(LowerFcand);
        VU_IR_OPCODE_NAME(LowerFcor);
        VU_IR_OPCODE_NAME(LowerFseq);
        VU_IR_OPCODE_NAME(LowerFsset);
        VU_IR_OPCODE_NAME(LowerFsand);
        VU_IR_OPCODE_NAME(LowerFsor);
        VU_IR_OPCODE_NAME(LowerFmand);
        VU_IR_OPCODE_NAME(LowerFmeq);
        VU_IR_OPCODE_NAME(LowerFmor);
        VU_IR_OPCODE_NAME(LowerB);
        VU_IR_OPCODE_NAME(LowerBal);
        VU_IR_OPCODE_NAME(LowerJr);
        VU_IR_OPCODE_NAME(LowerJalr);
        VU_IR_OPCODE_NAME(LowerIbeq);
        VU_IR_OPCODE_NAME(LowerIbne);
        VU_IR_OPCODE_NAME(LowerIbltz);
        VU_IR_OPCODE_NAME(LowerIbgtz);
        VU_IR_OPCODE_NAME(LowerIblez);
        VU_IR_OPCODE_NAME(LowerIbgez);
        VU_IR_OPCODE_NAME(LowerIadd);
        VU_IR_OPCODE_NAME(LowerIsub);
        VU_IR_OPCODE_NAME(LowerIaddi);
        VU_IR_OPCODE_NAME(LowerIand);
        VU_IR_OPCODE_NAME(LowerIor);
        VU_IR_OPCODE_NAME(LowerMove);
        VU_IR_OPCODE_NAME(LowerMr32);
        VU_IR_OPCODE_NAME(LowerLqi);
        VU_IR_OPCODE_NAME(LowerSqi);
        VU_IR_OPCODE_NAME(LowerLqd);
        VU_IR_OPCODE_NAME(LowerSqd);
        VU_IR_OPCODE_NAME(LowerDiv);
        VU_IR_OPCODE_NAME(LowerSqrt);
        VU_IR_OPCODE_NAME(LowerRsqrt);
        VU_IR_OPCODE_NAME(LowerWaitQ);
        VU_IR_OPCODE_NAME(LowerMtir);
        VU_IR_OPCODE_NAME(LowerMfir);
        VU_IR_OPCODE_NAME(LowerIlwr);
        VU_IR_OPCODE_NAME(LowerIswr);
        VU_IR_OPCODE_NAME(LowerRnext);
        VU_IR_OPCODE_NAME(LowerRget);
        VU_IR_OPCODE_NAME(LowerRinit);
        VU_IR_OPCODE_NAME(LowerRxor);
        VU_IR_OPCODE_NAME(LowerMfp);
        VU_IR_OPCODE_NAME(LowerXtop);
        VU_IR_OPCODE_NAME(LowerXitop);
        VU_IR_OPCODE_NAME(LowerXgkick);
        VU_IR_OPCODE_NAME(LowerEsadd);
        VU_IR_OPCODE_NAME(LowerErsadd);
        VU_IR_OPCODE_NAME(LowerEleng);
        VU_IR_OPCODE_NAME(LowerErleng);
        VU_IR_OPCODE_NAME(LowerErcpr);
        VU_IR_OPCODE_NAME(LowerWaitP);
        VU_IR_OPCODE_NAME(LowerEatan);
        VU_IR_OPCODE_NAME(Unsupported);
    }
#undef VU_IR_OPCODE_NAME
    return "Unknown";
}

std::string_view vuIrBlockExitName(VuIrBlockExit exit)
{
    switch (exit)
    {
    case VuIrBlockExit::PairLimit:
        return "pair-limit";
    case VuIrBlockExit::BranchBoundary:
        return "branch-boundary";
    case VuIrBlockExit::ProgramEndBoundary:
        return "program-end-boundary";
    case VuIrBlockExit::XgkickBoundary:
        return "xgkick-boundary";
    case VuIrBlockExit::UnsupportedInstruction:
        return "unsupported-instruction";
    case VuIrBlockExit::CodeBounds:
        return "code-bounds";
    }
    return "unknown";
}
