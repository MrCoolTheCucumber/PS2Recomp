#include "runtime/ps2_vu1_command_stream.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace
{
    constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
    constexpr uint64_t kFnvPrime = 1099511628211ull;

    void hashBytes(
        uint64_t &hash, const void *data, size_t size) noexcept
    {
        const auto *const bytes =
            static_cast<const uint8_t *>(data);
        for (size_t index = 0u; index < size; ++index)
        {
            hash ^= bytes[index];
            hash *= kFnvPrime;
        }
    }

    template <typename T>
    void hashScalar(uint64_t &hash, T value) noexcept
    {
        using Value = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Value, bool>)
        {
            const uint8_t encoded = value ? 1u : 0u;
            hashBytes(hash, &encoded, sizeof(encoded));
        }
        else if constexpr (std::is_enum_v<Value>)
        {
            using Underlying = std::underlying_type_t<Value>;
            using Unsigned = std::make_unsigned_t<Underlying>;
            hashScalar(hash, static_cast<Unsigned>(value));
        }
        else
        {
            static_assert(std::is_integral_v<Value>);
            using Unsigned = std::make_unsigned_t<Value>;
            const Unsigned encoded = static_cast<Unsigned>(value);
            for (size_t index = 0u; index < sizeof(encoded); ++index)
            {
                const uint8_t byte = static_cast<uint8_t>(
                    encoded >> (index * 8u));
                hashBytes(hash, &byte, sizeof(byte));
            }
        }
    }

    void hashFloat(uint64_t &hash, float value) noexcept
    {
        hashScalar(hash, std::bit_cast<uint32_t>(value));
    }

    void hashVector(
        uint64_t &hash,
        const std::vector<uint8_t> &bytes) noexcept
    {
        hashScalar(hash, static_cast<uint64_t>(bytes.size()));
        hashBytes(hash, bytes.data(), bytes.size());
    }

    void hashVifState(
        uint64_t &hash, const Vu1VifState &state) noexcept
    {
        hashScalar(hash, state.cycle);
        hashScalar(hash, state.mode);
        hashScalar(hash, state.mask);
        for (uint32_t value : state.row)
            hashScalar(hash, value);
        for (uint32_t value : state.column)
            hashScalar(hash, value);
        hashScalar(hash, state.tops);
    }

    void hashExecutionState(
        uint64_t &hash,
        const VuExecutionState &state) noexcept
    {
        for (const auto &vector : state.vf)
        {
            for (float value : vector)
                hashFloat(hash, value);
        }
        for (int32_t value : state.vi)
            hashScalar(hash, value);
        for (float value : state.acc)
            hashFloat(hash, value);
        hashFloat(hash, state.q);
        hashFloat(hash, state.p);
        hashFloat(hash, state.i);
        hashScalar(hash, state.pc);
        hashScalar(hash, state.mac);
        hashScalar(hash, state.clip);
        hashScalar(hash, state.status);
        hashScalar(hash, state.ebit);
        hashScalar(hash, state.top);
        hashScalar(hash, state.itop);
        hashScalar(hash, state.branchPending);
        hashScalar(hash, state.branchTarget);
        hashScalar(hash, state.branchDelay);
        hashScalar(hash, state.viBackupCycles);
        hashScalar(hash, state.viBackupRegister);
        hashScalar(hash, state.viBackupValue);
        hashScalar(hash, state.active);
        hashScalar(hash, state.issuedCycles);

        const VuPipelineState &pipeline = state.pipeline;
        hashScalar(hash, pipeline.xgkick.active);
        hashScalar(hash, pipeline.xgkick.address);
        hashScalar(hash, pipeline.xgkick.tagBytesRemaining);
        hashScalar(hash, pipeline.xgkick.cycleCredit);
        hashScalar(hash, pipeline.xgkick.tagEop);
        hashScalar(
            hash,
            static_cast<uint64_t>(pipeline.xgkick.packet.size()));
        hashBytes(
            hash, pipeline.xgkick.packet.data(),
            pipeline.xgkick.packet.size());
        hashScalar(hash, pipeline.delayedQ.active);
        hashFloat(hash, pipeline.delayedQ.result);
        hashScalar(hash, pipeline.delayedQ.cyclesRemaining);
        hashScalar(hash, pipeline.delayedP.active);
        hashFloat(hash, pipeline.delayedP.result);
        hashScalar(hash, pipeline.delayedP.cyclesRemaining);
        for (const VuPipelineState::FmacFlags &flags :
             pipeline.fmacFlags)
        {
            hashScalar(hash, flags.active);
            hashScalar(hash, flags.fmacActive);
            hashScalar(hash, flags.clipActive);
            hashScalar(hash, flags.mac);
            hashScalar(hash, flags.status);
            hashScalar(hash, flags.clip);
        }
        hashScalar(hash, pipeline.fmacFlagIndex);
        hashScalar(hash, pipeline.workingMac);
        hashScalar(hash, pipeline.workingClip);
    }

    uint64_t snapshotHash(const Vu1Snapshot &snapshot) noexcept
    {
        return vu1ArchitecturalStateHash(
            snapshot.state,
            snapshot.microMemory,
            snapshot.dataMemory,
            snapshot.vif,
            snapshot.codeGeneration);
    }

    uint32_t unpackBytesPerVector(
        const Vu1DecodedUnpackCommand &command) noexcept
    {
        const uint32_t components = command.componentCount;
        uint32_t bitsPerComponent = 32u;
        switch (command.vectorLength)
        {
        case 0u:
            bitsPerComponent = 32u;
            break;
        case 1u:
            bitsPerComponent = 16u;
            break;
        case 2u:
            bitsPerComponent = 8u;
            break;
        case 3u:
            bitsPerComponent = components == 4u ? 4u : 16u;
            break;
        default:
            return 0u;
        }
        const uint32_t bits =
            command.vectorLength == 3u && components == 4u
                ? 16u
                : components * bitsPerComponent;
        return (bits + 7u) / 8u;
    }

    bool applyDecodedUnpack(
        const Vu1DecodedUnpackCommand &command,
        Vu1VifState &vif,
        uint8_t *dataMemory,
        uint32_t dataMemorySize)
    {
        if (!dataMemory || dataMemorySize < 16u ||
            command.componentCount < 1u ||
            command.componentCount > 4u ||
            command.vectorLength > 3u ||
            command.writeVectorCount == 0u ||
            command.sourceWordAlignment < 1u ||
            command.sourceWordAlignment > 4u)
        {
            return false;
        }

        const uint32_t bytesPerVector =
            unpackBytesPerVector(command);
        if (bytesPerVector == 0u)
            return false;
        const uint64_t sourceBytes =
            static_cast<uint64_t>(command.sourceVectorCount) *
            bytesPerVector;
        const uint64_t paddedBytes = (sourceBytes + 3u) & ~3ull;
        if (paddedBytes != command.bytes.size())
            return false;

        uint32_t cl = vif.cycle & 0xffu;
        uint32_t wl = (vif.cycle >> 8u) & 0xffu;
        if (cl == 0u)
            cl = 1u;
        if (wl == 0u)
            wl = 1u;

        uint32_t expectedSourceCount = command.writeVectorCount;
        if (cl < wl)
        {
            const uint32_t fullBlocks =
                command.writeVectorCount / wl;
            uint32_t remainder = command.writeVectorCount % wl;
            if (remainder > cl)
                remainder = cl;
            expectedSourceCount = fullBlocks * cl + remainder;
        }
        if (expectedSourceCount != command.sourceVectorCount)
            return false;

        uint32_t vuAddress = command.immediate & 0x3ffu;
        if ((command.immediate & 0x8000u) != 0u)
        {
            vuAddress =
                (vuAddress + (vif.tops & 0x3ffu)) & 0x3ffu;
        }

        const uint8_t *const sourceBase = command.bytes.data();
        const uint32_t mode = vif.mode & 3u;
        uint32_t sourceIndex = 0u;
        for (uint32_t writeIndex = 0u;
             writeIndex < command.writeVectorCount;
             ++writeIndex)
        {
            const uint32_t cyclePosition = writeIndex % wl;
            const bool sourceAvailable =
                cl >= wl || cyclePosition < cl;
            const uint32_t destinationVector =
                cl >= wl
                    ? (vuAddress + (writeIndex / wl) * cl +
                       cyclePosition) &
                          0x3ffu
                    : (vuAddress + writeIndex) & 0x3ffu;
            const uint32_t destinationOffset =
                destinationVector * 16u;
            if (destinationOffset > dataMemorySize ||
                dataMemorySize - destinationOffset < 16u)
            {
                if (sourceAvailable &&
                    sourceIndex < command.sourceVectorCount)
                {
                    ++sourceIndex;
                }
                continue;
            }

            uint32_t lanes[4]{};
            std::memcpy(
                lanes, dataMemory + destinationOffset,
                sizeof(lanes));
            uint32_t decompressed[4]{
                lanes[0], lanes[1], lanes[2], lanes[3]};
            bool decoded = false;
            const uint8_t *sourceVector = nullptr;
            if (sourceAvailable &&
                sourceIndex < command.sourceVectorCount)
            {
                sourceVector =
                    sourceBase + sourceIndex * bytesPerVector;
                ++sourceIndex;
                decoded = true;
            }

            const auto sourceHasBytes =
                [&](uint32_t offset, uint32_t count)
                {
                    if (!sourceVector)
                        return false;
                    const size_t baseOffset =
                        static_cast<size_t>(
                            sourceVector - sourceBase);
                    return baseOffset + offset + count <=
                           command.bytes.size();
                };
            const auto extend16 =
                [&](uint16_t raw)
                {
                    return command.zeroExtend
                        ? static_cast<uint32_t>(raw)
                        : static_cast<uint32_t>(
                              static_cast<int32_t>(
                                  static_cast<int16_t>(raw)));
                };
            const auto extend8 =
                [&](uint8_t raw)
                {
                    return command.zeroExtend
                        ? static_cast<uint32_t>(raw)
                        : static_cast<uint32_t>(
                              static_cast<int32_t>(
                                  static_cast<int8_t>(raw)));
                };

            bool handledFormat = true;
            const uint32_t components = command.componentCount;
            const uint8_t vl = command.vectorLength;
            if (!decoded)
            {
                handledFormat = false;
            }
            else if (vl == 0u)
            {
                if (components == 1u)
                {
                    uint32_t scalar = 0u;
                    std::memcpy(&scalar, sourceVector, sizeof(scalar));
                    std::fill_n(decompressed, 4u, scalar);
                }
                else
                {
                    const uint32_t limit =
                        components == 3u ? 4u : components;
                    for (uint32_t component = 0u;
                         component < limit; ++component)
                    {
                        if (!sourceHasBytes(component * 4u, 4u))
                            break;
                        std::memcpy(
                            &decompressed[component],
                            sourceVector + component * 4u,
                            sizeof(uint32_t));
                    }
                    if (components == 2u)
                    {
                        decompressed[2] = decompressed[0];
                        decompressed[3] = decompressed[1];
                    }
                }
            }
            else if (vl == 1u)
            {
                if (components == 1u)
                {
                    uint16_t raw = 0u;
                    std::memcpy(&raw, sourceVector, sizeof(raw));
                    std::fill_n(decompressed, 4u, extend16(raw));
                }
                else
                {
                    const uint32_t limit =
                        components == 3u ? 4u : components;
                    for (uint32_t component = 0u;
                         component < limit; ++component)
                    {
                        if (!sourceHasBytes(component * 2u, 2u))
                            break;
                        uint16_t raw = 0u;
                        std::memcpy(
                            &raw, sourceVector + component * 2u,
                            sizeof(raw));
                        decompressed[component] = extend16(raw);
                    }
                    if (components == 2u)
                    {
                        decompressed[2] = decompressed[0];
                        decompressed[3] = decompressed[1];
                    }
                }
            }
            else if (vl == 2u)
            {
                if (components == 1u)
                {
                    std::fill_n(
                        decompressed, 4u,
                        extend8(sourceVector[0]));
                }
                else
                {
                    const uint32_t limit =
                        components == 3u ? 4u : components;
                    for (uint32_t component = 0u;
                         component < limit; ++component)
                    {
                        if (!sourceHasBytes(component, 1u))
                            break;
                        decompressed[component] =
                            extend8(sourceVector[component]);
                    }
                    if (components == 2u)
                    {
                        decompressed[2] = decompressed[0];
                        decompressed[3] = decompressed[1];
                    }
                }
            }
            else if (vl == 3u && components == 4u)
            {
                uint16_t packed = 0u;
                std::memcpy(&packed, sourceVector, sizeof(packed));
                decompressed[0] = packed & 0x1fu;
                decompressed[1] = (packed >> 5u) & 0x1fu;
                decompressed[2] = (packed >> 10u) & 0x1fu;
                decompressed[3] = (packed >> 15u) & 0x01u;
            }
            else
            {
                handledFormat = false;
            }

            if (decoded && components == 3u)
            {
                bool zeroW = false;
                if (vl == 0u)
                {
                    const uint32_t iteration =
                        (sourceIndex - 1u) & 1u;
                    zeroW =
                        iteration != command.sourceWordAlignment;
                }
                else if (vl == 1u)
                {
                    const uint32_t iteration = sourceIndex;
                    const uint32_t boundary =
                        ((iteration / 4u) + 1u +
                         (4u - command.sourceWordAlignment)) &
                        3u;
                    zeroW =
                        (iteration & 1u) == 0u && boundary == 0u;
                }
                else if (vl == 2u)
                {
                    zeroW =
                        sourceIndex != command.sourceWordAlignment;
                }
                if (zeroW)
                    decompressed[3] = 0u;
            }
            if (decoded && components == 2u && vl == 0u)
                decompressed[3] = 0u;

            if (!handledFormat && decoded &&
                !command.maskEnabled &&
                (mode == 0u || mode == 3u))
            {
                const uint32_t copyBytes =
                    std::min(bytesPerVector, 16u);
                std::memcpy(
                    dataMemory + destinationOffset,
                    sourceVector, copyBytes);
                continue;
            }

            const bool canAdd =
                vl != 3u || components != 4u;
            const uint32_t columnIndex =
                std::min(cyclePosition, 3u);
            const uint32_t maskCycle =
                std::min(cyclePosition, 3u);
            for (uint32_t field = 0u; field < 4u; ++field)
            {
                uint32_t maskSpec = 0u;
                if (command.maskEnabled)
                {
                    const uint32_t shift =
                        ((maskCycle * 4u) + field) * 2u;
                    maskSpec = (vif.mask >> shift) & 0x3u;
                }
                if (!decoded && maskSpec == 0u)
                    maskSpec = 1u;

                uint32_t writeValue = lanes[field];
                if (maskSpec == 0u)
                {
                    if (handledFormat)
                    {
                        writeValue = decompressed[field];
                        if (canAdd && (mode == 1u || mode == 2u))
                        {
                            writeValue += vif.row[field];
                            if (mode == 2u)
                                vif.row[field] = writeValue;
                        }
                    }
                }
                else if (maskSpec == 1u)
                {
                    writeValue = vif.row[field];
                }
                else if (maskSpec == 2u)
                {
                    writeValue = vif.column[columnIndex];
                }
                else
                {
                    continue;
                }
                lanes[field] = writeValue;
            }
            std::memcpy(
                dataMemory + destinationOffset,
                lanes, sizeof(lanes));
        }
        return sourceIndex == command.sourceVectorCount;
    }

    bool validPayload(const Vu1CommandPayload &payload) noexcept
    {
        return std::visit(
            [](const auto &value) noexcept
            {
                using T = std::decay_t<decltype(value)>;
                if constexpr (
                    std::is_same_v<T, Vu1MicroMemoryWriteCommand> ||
                    std::is_same_v<T, Vu1DataMemoryWriteCommand>)
                {
                    return !value.bytes.empty();
                }
                else if constexpr (
                    std::is_same_v<T, Vu1DecodedUnpackCommand>)
                {
                    return !value.bytes.empty() &&
                           value.writeVectorCount != 0u &&
                           value.sourceVectorCount != 0u;
                }
                else if constexpr (
                    std::is_same_v<T, Vu1AdvanceSliceCommand>)
                {
                    return value.maximumCycles != 0u;
                }
                else
                {
                    return true;
                }
            },
            payload);
    }
}

const char *vu1CommandTypeName(Vu1CommandType type) noexcept
{
    switch (type)
    {
    case Vu1CommandType::MicroMemoryWrite:
        return "micro-memory-write";
    case Vu1CommandType::DataMemoryWrite:
        return "data-memory-write";
    case Vu1CommandType::DecodedUnpack:
        return "decoded-unpack";
    case Vu1CommandType::VifStateUpdate:
        return "vif-state-update";
    case Vu1CommandType::RegisterWrite:
        return "register-write";
    case Vu1CommandType::Mscal:
        return "mscal";
    case Vu1CommandType::Mscalf:
        return "mscalf";
    case Vu1CommandType::Mscnt:
        return "mscnt";
    case Vu1CommandType::AdvanceSlice:
        return "advance-slice";
    case Vu1CommandType::Reset:
        return "reset";
    case Vu1CommandType::Snapshot:
        return "snapshot";
    case Vu1CommandType::Restore:
        return "restore";
    case Vu1CommandType::Barrier:
        return "barrier";
    case Vu1CommandType::Shutdown:
        return "shutdown";
    }
    return "unknown";
}

Vu1CommandType vu1CommandType(
    const Vu1CommandPayload &payload) noexcept
{
    return std::visit(
        [](const auto &value) noexcept
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Vu1MicroMemoryWriteCommand>)
                return Vu1CommandType::MicroMemoryWrite;
            else if constexpr (std::is_same_v<T, Vu1DataMemoryWriteCommand>)
                return Vu1CommandType::DataMemoryWrite;
            else if constexpr (std::is_same_v<T, Vu1DecodedUnpackCommand>)
                return Vu1CommandType::DecodedUnpack;
            else if constexpr (std::is_same_v<T, Vu1VifStateUpdateCommand>)
                return Vu1CommandType::VifStateUpdate;
            else if constexpr (std::is_same_v<T, Vu1RegisterWriteCommand>)
                return Vu1CommandType::RegisterWrite;
            else if constexpr (std::is_same_v<T, Vu1MscalCommand>)
                return Vu1CommandType::Mscal;
            else if constexpr (std::is_same_v<T, Vu1MscalfCommand>)
                return Vu1CommandType::Mscalf;
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
                return Vu1CommandType::Mscnt;
            else if constexpr (std::is_same_v<T, Vu1AdvanceSliceCommand>)
                return Vu1CommandType::AdvanceSlice;
            else if constexpr (std::is_same_v<T, Vu1ResetCommand>)
                return Vu1CommandType::Reset;
            else if constexpr (std::is_same_v<T, Vu1SnapshotCommand>)
                return Vu1CommandType::Snapshot;
            else if constexpr (std::is_same_v<T, Vu1RestoreCommand>)
                return Vu1CommandType::Restore;
            else if constexpr (std::is_same_v<T, Vu1BarrierCommand>)
                return Vu1CommandType::Barrier;
            else
                return Vu1CommandType::Shutdown;
        },
        payload);
}

uint64_t vu1CommandPayloadSize(
    const Vu1CommandPayload &payload) noexcept
{
    return std::visit(
        [](const auto &value) -> uint64_t
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (
                std::is_same_v<T, Vu1MicroMemoryWriteCommand> ||
                std::is_same_v<T, Vu1DataMemoryWriteCommand>)
            {
                return sizeof(value.offset) + sizeof(value.wrap) +
                       sizeof(uint64_t) + value.bytes.size();
            }
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackCommand>)
            {
                return 11u + sizeof(uint64_t) + value.bytes.size();
            }
            else if constexpr (
                std::is_same_v<T, Vu1VifStateUpdateCommand>)
            {
                return 12u * sizeof(uint32_t);
            }
            else if constexpr (
                std::is_same_v<T, Vu1RegisterWriteCommand>)
            {
                return 3u + 4u * sizeof(uint32_t);
            }
            else if constexpr (
                std::is_same_v<T, Vu1MscalCommand> ||
                std::is_same_v<T, Vu1MscalfCommand>)
            {
                return 3u * sizeof(uint32_t);
            }
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
            {
                return 2u * sizeof(uint32_t);
            }
            else if constexpr (
                std::is_same_v<T, Vu1AdvanceSliceCommand>)
            {
                return sizeof(uint32_t) + sizeof(bool);
            }
            else if constexpr (std::is_same_v<T, Vu1ResetCommand>)
            {
                return 2u * sizeof(bool);
            }
            else if constexpr (std::is_same_v<T, Vu1RestoreCommand>)
            {
                return sizeof(uint64_t) +
                       value.snapshot.microMemory.size() +
                       value.snapshot.dataMemory.size();
            }
            else
            {
                return 0u;
            }
        },
        payload);
}

uint64_t vu1CommandPayloadHash(
    const Vu1CommandPayload &payload)
{
    uint64_t hash = kFnvOffsetBasis;
    hashScalar(hash, vu1CommandType(payload));
    std::visit(
        [&](const auto &value)
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (
                std::is_same_v<T, Vu1MicroMemoryWriteCommand> ||
                std::is_same_v<T, Vu1DataMemoryWriteCommand>)
            {
                hashScalar(hash, value.offset);
                hashScalar(hash, value.wrap);
                hashVector(hash, value.bytes);
            }
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackCommand>)
            {
                hashScalar(hash, value.immediate);
                hashScalar(hash, value.vectorLength);
                hashScalar(hash, value.componentCount);
                hashScalar(hash, value.writeVectorCount);
                hashScalar(hash, value.sourceVectorCount);
                hashScalar(hash, value.sourceWordAlignment);
                hashScalar(hash, value.maskEnabled);
                hashScalar(hash, value.zeroExtend);
                hashVector(hash, value.bytes);
            }
            else if constexpr (
                std::is_same_v<T, Vu1VifStateUpdateCommand>)
            {
                hashVifState(hash, value.state);
            }
            else if constexpr (
                std::is_same_v<T, Vu1RegisterWriteCommand>)
            {
                hashScalar(hash, value.kind);
                hashScalar(hash, value.index);
                hashScalar(hash, value.laneMask);
                for (uint32_t word : value.words)
                    hashScalar(hash, word);
            }
            else if constexpr (
                std::is_same_v<T, Vu1MscalCommand> ||
                std::is_same_v<T, Vu1MscalfCommand>)
            {
                hashScalar(hash, value.startPc);
                hashScalar(hash, value.top);
                hashScalar(hash, value.itop);
            }
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
            {
                hashScalar(hash, value.top);
                hashScalar(hash, value.itop);
            }
            else if constexpr (
                std::is_same_v<T, Vu1AdvanceSliceCommand>)
            {
                hashScalar(hash, value.maximumCycles);
                hashScalar(hash, value.captureState);
            }
            else if constexpr (std::is_same_v<T, Vu1ResetCommand>)
            {
                hashScalar(hash, value.clearMicroMemory);
                hashScalar(hash, value.clearDataMemory);
            }
            else if constexpr (std::is_same_v<T, Vu1RestoreCommand>)
            {
                hashScalar(hash, snapshotHash(value.snapshot));
            }
        },
        payload);
    return hash;
}

Vu1CommandDigest vu1CommandDigest(const Vu1Command &command)
{
    return {
        .sequence = command.identity.sequence,
        .generation = command.identity.generation,
        .guestTick = command.identity.guestTick,
        .type = command.type,
        .payloadSize = command.payloadSize,
        .payloadHash = vu1CommandPayloadHash(command.payload),
    };
}

uint64_t vu1CommandDigestHash(
    const Vu1CommandDigest &digest) noexcept
{
    uint64_t hash = kFnvOffsetBasis;
    hashScalar(hash, digest.sequence);
    hashScalar(hash, digest.generation);
    hashScalar(hash, digest.guestTick);
    hashScalar(hash, digest.type);
    hashScalar(hash, digest.payloadSize);
    hashScalar(hash, digest.payloadHash);
    return hash;
}

uint64_t vu1ArchitecturalStateHash(
    const VuExecutionState &state,
    std::span<const uint8_t> microMemory,
    std::span<const uint8_t> dataMemory,
    const Vu1VifState &vif,
    uint64_t codeGeneration) noexcept
{
    uint64_t hash = kFnvOffsetBasis;
    hashExecutionState(hash, state);
    hashScalar(hash, codeGeneration);
    hashScalar(hash, static_cast<uint64_t>(microMemory.size()));
    hashBytes(hash, microMemory.data(), microMemory.size());
    hashScalar(hash, static_cast<uint64_t>(dataMemory.size()));
    hashBytes(hash, dataMemory.data(), dataMemory.size());
    hashVifState(hash, vif);
    return hash;
}

Vu1CommandProcessor::Vu1CommandProcessor(
    VuUnit &unit,
    Vu1CommandProcessorConfiguration configuration)
    : m_unit(unit), m_configuration(configuration)
{
    if (unit.unitId() != VuUnitId::Vu1)
    {
        throw std::invalid_argument(
            "VU1 command processor requires a VU1 unit");
    }
}

void Vu1CommandProcessor::bindMemory(
    uint8_t *microMemory, uint32_t microMemorySize,
    uint8_t *dataMemory, uint32_t dataMemorySize,
    uint64_t codeGeneration,
    IVuExecutionObserver *diagnosticsObserver)
{
    if (!microMemory || microMemorySize == 0u ||
        !dataMemory || dataMemorySize == 0u)
    {
        throw std::invalid_argument(
            "VU1 command processor requires complete memory");
    }
    m_microMemory = microMemory;
    m_microMemorySize = microMemorySize;
    m_dataMemory = dataMemory;
    m_dataMemorySize = dataMemorySize;
    m_codeGeneration.store(
        codeGeneration, std::memory_order_relaxed);
    m_diagnosticsObserver = diagnosticsObserver;
}

bool Vu1CommandProcessor::validMemoryRange(
    uint32_t offset, size_t size,
    uint32_t memorySize, bool wrap) const noexcept
{
    if (size == 0u || offset >= memorySize)
        return false;
    if (wrap)
        return size <= memorySize;
    return size <= memorySize - offset;
}

void Vu1CommandProcessor::writeMemory(
    uint8_t *destination, uint32_t destinationSize,
    uint32_t offset, std::span<const uint8_t> bytes,
    bool wrap)
{
    if (!wrap)
    {
        std::memcpy(destination + offset, bytes.data(), bytes.size());
        return;
    }
    const size_t first =
        std::min<size_t>(bytes.size(), destinationSize - offset);
    std::memcpy(destination + offset, bytes.data(), first);
    if (bytes.size() != first)
    {
        std::memcpy(
            destination, bytes.data() + first,
            bytes.size() - first);
    }
}

Vu1Snapshot Vu1CommandProcessor::captureSnapshot() const
{
    Vu1Snapshot snapshot{
        .state = m_unit.state(),
        .vif = m_vifState,
        .codeGeneration = codeGeneration(),
    };
    if (m_microMemory && m_microMemorySize != 0u)
    {
        snapshot.microMemory.assign(
            m_microMemory,
            m_microMemory + m_microMemorySize);
    }
    if (m_dataMemory && m_dataMemorySize != 0u)
    {
        snapshot.dataMemory.assign(
            m_dataMemory,
            m_dataMemory + m_dataMemorySize);
    }
    return snapshot;
}

uint64_t Vu1CommandProcessor::stateHash() const noexcept
{
    if (!m_configuration.captureArchitecturalStateHashes ||
        !m_microMemory || !m_dataMemory)
    {
        return 0u;
    }
    return vu1ArchitecturalStateHash(
        m_unit.state(),
        std::span<const uint8_t>(
            m_microMemory, m_microMemorySize),
        std::span<const uint8_t>(
            m_dataMemory, m_dataMemorySize),
        m_vifState, codeGeneration());
}

Vu1CommandResultPayload Vu1CommandProcessor::apply(
    const Vu1CommandPayload &payload,
    Vu1CommandDisposition &disposition)
{
    return std::visit(
        [&](const auto &value) -> Vu1CommandResultPayload
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Vu1MicroMemoryWriteCommand>)
            {
                if (!validMemoryRange(
                        value.offset, value.bytes.size(),
                        m_microMemorySize, value.wrap))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                writeMemory(
                    m_microMemory, m_microMemorySize,
                    value.offset, value.bytes, value.wrap);
                m_codeGeneration.fetch_add(
                    1u, std::memory_order_relaxed);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1DataMemoryWriteCommand>)
            {
                if (!validMemoryRange(
                        value.offset, value.bytes.size(),
                        m_dataMemorySize, value.wrap))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                writeMemory(
                    m_dataMemory, m_dataMemorySize,
                    value.offset, value.bytes, value.wrap);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1DecodedUnpackCommand>)
            {
                if (!applyDecodedUnpack(
                        value, m_vifState,
                        m_dataMemory, m_dataMemorySize))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                }
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1VifStateUpdateCommand>)
            {
                m_vifState = value.state;
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1RegisterWriteCommand>)
            {
                VuExecutionState &state = m_unit.state();
                const auto copyFloat = [](float &target, uint32_t word)
                {
                    std::memcpy(&target, &word, sizeof(target));
                };
                switch (value.kind)
                {
                case Vu1RegisterKind::VectorFloat:
                    if (value.index >= 32u)
                        break;
                    for (uint32_t lane = 0u; lane < 4u; ++lane)
                    {
                        if ((value.laneMask & (1u << lane)) != 0u)
                            copyFloat(state.vf[value.index][lane], value.words[lane]);
                    }
                    if (value.index == 0u)
                    {
                        state.vf[0][0] = 0.0f;
                        state.vf[0][1] = 0.0f;
                        state.vf[0][2] = 0.0f;
                        state.vf[0][3] = 1.0f;
                    }
                    return Vu1NoResult{};
                case Vu1RegisterKind::VectorInteger:
                    if (value.index >= 16u)
                        break;
                    if (value.index != 0u)
                        state.vi[value.index] =
                            static_cast<int32_t>(value.words[0]);
                    return Vu1NoResult{};
                case Vu1RegisterKind::Accumulator:
                    for (uint32_t lane = 0u; lane < 4u; ++lane)
                    {
                        if ((value.laneMask & (1u << lane)) != 0u)
                            copyFloat(state.acc[lane], value.words[lane]);
                    }
                    return Vu1NoResult{};
                case Vu1RegisterKind::Q:
                    copyFloat(state.q, value.words[0]);
                    return Vu1NoResult{};
                case Vu1RegisterKind::P:
                    copyFloat(state.p, value.words[0]);
                    return Vu1NoResult{};
                case Vu1RegisterKind::I:
                    copyFloat(state.i, value.words[0]);
                    return Vu1NoResult{};
                case Vu1RegisterKind::ProgramCounter:
                    state.pc = value.words[0] &
                               (m_microMemorySize - 1u);
                    return Vu1NoResult{};
                case Vu1RegisterKind::Mac:
                    state.mac = value.words[0];
                    return Vu1NoResult{};
                case Vu1RegisterKind::Clip:
                    state.clip = value.words[0];
                    return Vu1NoResult{};
                case Vu1RegisterKind::Status:
                    state.status = value.words[0];
                    return Vu1NoResult{};
                case Vu1RegisterKind::Top:
                    state.top = value.words[0] & 0x3ffu;
                    return Vu1NoResult{};
                case Vu1RegisterKind::Itop:
                    state.itop = value.words[0] & 0x3ffu;
                    return Vu1NoResult{};
                }
                disposition = Vu1CommandDisposition::Malformed;
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1MscalCommand>)
            {
                m_unit.start(
                    value.startPc, value.top, value.itop, this);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1MscalfCommand>)
            {
                m_unit.start(
                    value.startPc, value.top, value.itop, this);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
            {
                m_unit.resumeState(value.top, value.itop, this);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1AdvanceSliceCommand>)
            {
                VuTransactionalSideEffectSink effects;
                VuExecutionContext context{
                    .state = m_unit.state(),
                    .code = m_microMemory,
                    .codeSize = m_microMemorySize,
                    .data = m_dataMemory,
                    .dataSize = m_dataMemorySize,
                    .sideEffects = effects,
                    .codeUnit = VuUnitId::Vu1,
                    .memoryIdentity = reinterpret_cast<uintptr_t>(this),
                    .codeIdentity = reinterpret_cast<uintptr_t>(m_microMemory),
                    .codeGeneration = codeGeneration(),
                    .trackedCode = true,
                    .observer = this,
                    .enableInstrumentation = true,
                    .enableNativeInstrumentation =
                        m_unit.nativeInstrumentationEnabled(),
                    .enableProgressAccounting = true,
                };
                Vu1SliceResult slice{
                    .run = m_unit.advance(
                        context, value.maximumCycles),
                };
                for (const std::vector<uint8_t> &packet :
                     effects.path1Packets())
                {
                    slice.path1Packets.push_back({
                        .bytes = packet,
                    });
                }
                if (value.captureState)
                    slice.state = m_unit.state();
                slice.architecturalStateHash = stateHash();
                slice.vifCanResume = !m_unit.isActive();
                if (slice.run.reason == VuExitReason::Fault)
                    slice.fault = "VU1 backend reported a fault";
                return slice;
            }
            else if constexpr (std::is_same_v<T, Vu1ResetCommand>)
            {
                m_unit.reset();
                if (value.clearMicroMemory && m_microMemory)
                {
                    std::memset(m_microMemory, 0, m_microMemorySize);
                    m_codeGeneration.fetch_add(
                        1u, std::memory_order_relaxed);
                }
                if (value.clearDataMemory && m_dataMemory)
                    std::memset(m_dataMemory, 0, m_dataMemorySize);
                m_vifState = {};
                if (m_generation ==
                    std::numeric_limits<uint64_t>::max())
                {
                    throw std::overflow_error(
                        "VU1 command generation exhausted");
                }
                ++m_generation;
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1SnapshotCommand>)
            {
                Vu1SnapshotResult result{
                    .snapshot = captureSnapshot(),
                };
                result.architecturalStateHash =
                    m_configuration.captureArchitecturalStateHashes
                        ? snapshotHash(result.snapshot)
                        : 0u;
                return result;
            }
            else if constexpr (std::is_same_v<T, Vu1RestoreCommand>)
            {
                if (value.snapshot.microMemory.size() !=
                        m_microMemorySize ||
                    value.snapshot.dataMemory.size() !=
                        m_dataMemorySize)
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                m_unit.state() = value.snapshot.state;
                std::memcpy(
                    m_microMemory,
                    value.snapshot.microMemory.data(),
                    m_microMemorySize);
                std::memcpy(
                    m_dataMemory,
                    value.snapshot.dataMemory.data(),
                    m_dataMemorySize);
                m_vifState = value.snapshot.vif;
                m_codeGeneration.store(
                    value.snapshot.codeGeneration,
                    std::memory_order_relaxed);
                if (m_generation ==
                    std::numeric_limits<uint64_t>::max())
                {
                    throw std::overflow_error(
                        "VU1 command generation exhausted");
                }
                ++m_generation;
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1BarrierCommand>)
            {
                return Vu1BarrierResult{
                    .architecturalStateHash = stateHash(),
                };
            }
            else
            {
                m_shutdown = true;
                return Vu1NoResult{};
            }
        },
        payload);
}

Vu1CommandResult Vu1CommandProcessor::process(Vu1Command command)
{
    Vu1CommandResult result{
        .identity = command.identity,
    };
    if (m_configuration.captureCommandDigests)
        result.digest = vu1CommandDigest(command);
    else
    {
        result.digest.sequence = command.identity.sequence;
        result.digest.generation = command.identity.generation;
        result.digest.guestTick = command.identity.guestTick;
        result.digest.type = command.type;
        result.digest.payloadSize = command.payloadSize;
    }

    if (m_shutdown)
    {
        result.disposition = Vu1CommandDisposition::Shutdown;
        return result;
    }
    if (command.identity.generation < m_generation)
    {
        result.disposition = Vu1CommandDisposition::StaleGeneration;
        return result;
    }
    if (command.identity.generation > m_generation)
    {
        result.disposition = Vu1CommandDisposition::FutureGeneration;
        return result;
    }
    if (command.identity.sequence != m_nextSequence)
    {
        result.disposition = Vu1CommandDisposition::OutOfOrder;
        return result;
    }
    if (command.type != vu1CommandType(command.payload) ||
        command.payloadSize != vu1CommandPayloadSize(command.payload) ||
        !validPayload(command.payload) ||
        !m_microMemory || !m_dataMemory)
    {
        result.disposition = Vu1CommandDisposition::Malformed;
        return result;
    }

    result.payload = apply(command.payload, result.disposition);
    if (result.disposition == Vu1CommandDisposition::Completed)
    {
        if (m_nextSequence ==
            std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error(
                "VU1 command sequence exhausted");
        }
        ++m_nextSequence;
    }
    return result;
}

bool Vu1CommandProcessor::vuTraceEnabled() const
{
    return m_diagnosticsObserver &&
           m_diagnosticsObserver->vuTraceEnabled();
}

bool Vu1CommandProcessor::vuWorkloadProfileEnabled() const
{
    return m_diagnosticsObserver &&
           m_diagnosticsObserver->vuWorkloadProfileEnabled();
}

uint64_t Vu1CommandProcessor::currentVuCodeGeneration(
    VuUnitId unit) const
{
    if (unit == VuUnitId::Vu1)
        return codeGeneration();
    return m_diagnosticsObserver
        ? m_diagnosticsObserver->currentVuCodeGeneration(unit)
        : 0u;
}

void Vu1CommandProcessor::traceVuInvocation(
    uint32_t startPc, uint32_t top, uint32_t itop,
    bool resume, const VuExecutionState &state)
{
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->traceVuInvocation(
            startPc, top, itop, resume, state);
    }
}

void Vu1CommandProcessor::traceVuInstruction(
    uint32_t pc, uint32_t lower, uint32_t upper,
    const VuExecutionState &state)
{
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->traceVuInstruction(
            pc, lower, upper, state);
    }
}

void Vu1CommandProcessor::traceVuXgkick(uint32_t sourceQword)
{
    if (m_diagnosticsObserver)
        m_diagnosticsObserver->traceVuXgkick(sourceQword);
}

void Vu1CommandProcessor::traceVuInvocationEnd(
    uint32_t finalPc, bool ended, bool hitCycleLimit,
    const int32_t *viRegisters, size_t viRegisterCount)
{
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->traceVuInvocationEnd(
            finalPc, ended, hitCycleLimit,
            viRegisters, viRegisterCount);
    }
}

void Vu1CommandProcessor::beginVuWorkloadProfileInvocation(
    uint32_t startPc, const uint8_t *code,
    uint32_t codeSize, uint64_t generation)
{
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->beginVuWorkloadProfileInvocation(
            startPc, code, codeSize, generation);
    }
}

void Vu1CommandProcessor::recordVuWorkloadProfileInstruction(
    uint32_t pc, uint32_t lower, uint32_t upper)
{
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->recordVuWorkloadProfileInstruction(
            pc, lower, upper);
    }
}

void Vu1CommandProcessor::recordVuWorkloadProfileTransition(
    uint32_t pc, uint32_t nextPc)
{
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->recordVuWorkloadProfileTransition(
            pc, nextPc);
    }
}

void Vu1CommandProcessor::endVuWorkloadProfileInvocation(
    bool completed)
{
    if (m_diagnosticsObserver)
        m_diagnosticsObserver->endVuWorkloadProfileInvocation(completed);
}

void Vu1CommandProcessor::resetVuWorkloadProfileEpoch()
{
    if (m_diagnosticsObserver)
        m_diagnosticsObserver->resetVuWorkloadProfileEpoch();
}

InlineVu1Executor::InlineVu1Executor(
    Vu1CommandProcessor &processor)
    : m_processor(processor),
      m_nextSequence(processor.nextSequence())
{
}

Vu1CommandResult InlineVu1Executor::submit(
    Vu1CommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (m_nextSequence == 0u)
    {
        throw std::overflow_error(
            "VU1 command sequence exhausted");
    }
    Vu1Command command{
        .identity = {
            .sequence = m_nextSequence,
            .generation = m_processor.generation(),
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        },
        .type = vu1CommandType(payload),
        .payloadSize = vu1CommandPayloadSize(payload),
        .payload = std::move(payload),
    };
    Vu1CommandResult result =
        m_processor.process(std::move(command));
    if (result.disposition == Vu1CommandDisposition::Completed)
    {
        if (m_nextSequence ==
            std::numeric_limits<uint64_t>::max())
        {
            m_nextSequence = 0u;
        }
        else
        {
            ++m_nextSequence;
        }
    }
    return result;
}
