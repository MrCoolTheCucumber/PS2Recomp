#include "runtime/ps2_vu1_command_stream.h"
#include "runtime/ps2_build_config.h"

#include "ThreadNaming.h"

#include <algorithm>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <immintrin.h>
#include <limits>
#include <smmintrin.h>
#include <stdexcept>
#include <type_traits>

struct Vu1ExecutionEpochState
{
    struct Checkpoint
    {
        uint64_t revision = 0u;
        uint32_t cycles = 0u;
        bool active = false;
        bool consumed = false;
        Vu1CommandResult result{};
    };

    explicit Vu1ExecutionEpochState(
        Vu1ExecutionEpochOptions epochOptions)
        : options(epochOptions),
          demandedCycles(epochOptions.initialCycles)
    {
    }

    mutable std::mutex mutex;
    std::condition_variable changed;
    Vu1ExecutionEpochOptions options{};
    std::deque<std::shared_ptr<Checkpoint>> journal;
    std::exception_ptr failure;
    uint64_t demandRevision = 0u;
    uint32_t demandedCycles = 0u;
    bool stopRequested = false;
    bool workerStarted = false;
    bool stopped = false;
};

namespace
{
    constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
    constexpr uint64_t kFnvPrime = 1099511628211ull;

    uint64_t checkedVu1Increment(
        uint64_t value, const char *identityName)
    {
        if (value == std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error(
                std::string("VU1 command ") +
                identityName + " exhausted");
        }
        return value + 1u;
    }

    bool startsVu1Generation(
        const Vu1CommandPayload &payload) noexcept
    {
        return std::holds_alternative<Vu1ResetCommand>(payload) ||
               std::holds_alternative<Vu1RestoreCommand>(payload);
    }

    class Vu1SliceSideEffectSink final : public IVuSideEffectSink
    {
    public:
        explicit Vu1SliceSideEffectSink(
            std::vector<Vu1Path1Packet> &packets)
            : m_packets(packets)
        {
        }

        void submitPath1Packet(
            const uint8_t *data,
            uint32_t sizeBytes) override
        {
            Vu1Path1Packet &packet =
                m_packets.emplace_back();
            if (sizeBytes != 0u)
            {
                packet.bytes.assign(
                    data, data + sizeBytes);
            }
        }

    private:
        std::vector<Vu1Path1Packet> &m_packets;
    };

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

    bool validDecodedUnpackPayload(
        const Vu1DecodedUnpackCommand &command) noexcept
    {
        return !command.bytes.empty() &&
               command.writeVectorCount != 0u &&
               command.sourceVectorCount != 0u;
    }

    uint64_t decodedUnpackPayloadSize(
        const Vu1DecodedUnpackCommand &command) noexcept
    {
        return 11u + sizeof(uint64_t) + command.bytes.size() +
               (command.vifStateBefore
                    ? 12u * sizeof(uint32_t)
                    : 0u);
    }

    uint64_t decodedUnpackBatchPayloadSize(
        const Vu1DecodedUnpackBatch &batch) noexcept
    {
        uint64_t size = sizeof(uint64_t);
        for (const Vu1DecodedUnpackCommand &command :
             batch.commands)
        {
            const uint64_t commandSize =
                decodedUnpackPayloadSize(command);
            if (commandSize >
                std::numeric_limits<uint64_t>::max() - size)
            {
                return std::numeric_limits<uint64_t>::max();
            }
            size += commandSize;
        }
        return size;
    }

    uint64_t decodedUnpackOperationCount(
        const Vu1CommandPayload &payload) noexcept
    {
        return std::visit(
            [](const auto &value) noexcept -> uint64_t
            {
                using T = std::decay_t<decltype(value)>;
                if constexpr (
                    std::is_same_v<T, Vu1DecodedUnpackCommand>)
                {
                    return 1u;
                }
                else if constexpr (
                    std::is_same_v<T, Vu1DecodedUnpackBatchCommand>)
                {
                    return value.batch
                        ? value.batch->commands.size()
                        : 0u;
                }
                else if constexpr (
                    std::is_same_v<T, Vu1MscalCommand> ||
                    std::is_same_v<T, Vu1MscalfCommand> ||
                    std::is_same_v<T, Vu1MscntCommand>)
                {
                    return value.unpacksBefore
                        ? value.unpacksBefore->commands.size()
                        : 0u;
                }
                else
                {
                    return 0u;
                }
            },
            payload);
    }

    void hashDecodedUnpack(
        uint64_t &hash,
        const Vu1DecodedUnpackCommand &command)
    {
        hashScalar(hash, command.immediate);
        hashScalar(hash, command.vectorLength);
        hashScalar(hash, command.componentCount);
        hashScalar(hash, command.writeVectorCount);
        hashScalar(hash, command.sourceVectorCount);
        hashScalar(hash, command.sourceWordAlignment);
        hashScalar(hash, command.maskEnabled);
        hashScalar(hash, command.zeroExtend);
        hashVector(hash, command.bytes);
        if (command.vifStateBefore)
            hashVifState(hash, *command.vifStateBefore);
    }

    void copyWrappedBytes(
        uint8_t *destination, uint32_t destinationSize,
        uint32_t destinationOffset, const uint8_t *source,
        uint32_t sizeBytes)
    {
        while (sizeBytes != 0u)
        {
            const uint32_t chunk = std::min(
                sizeBytes, destinationSize - destinationOffset);
            std::memcpy(
                destination + destinationOffset, source, chunk);
            source += chunk;
            sizeBytes -= chunk;
            destinationOffset = 0u;
        }
    }

    bool tryUnpackContiguousV4(
        uint8_t *destination, uint32_t destinationSize,
        uint32_t destinationVector, const uint8_t *source,
        uint32_t vectorCount, uint8_t componentWidth,
        bool zeroExtend, uint32_t mode, const uint32_t *row)
    {
        if (!destination || !source || !row ||
            destinationSize < 16u ||
            (destinationSize & 15u) != 0u ||
            componentWidth > 2u || mode > 1u)
        {
            return false;
        }

        const uint32_t destinationVectors = destinationSize / 16u;
        destinationVector %= destinationVectors;
        if (componentWidth == 0u && mode != 1u)
        {
            copyWrappedBytes(
                destination, destinationSize,
                destinationVector * 16u, source,
                vectorCount * 16u);
            return true;
        }

        const uint32_t sourceStride =
            componentWidth == 0u ? 16u :
            componentWidth == 1u ? 8u : 4u;
        const __m128i rowValues = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(row));
        uint32_t remaining = vectorCount;
        while (remaining != 0u)
        {
            const uint32_t chunk = std::min(
                remaining,
                destinationVectors - destinationVector);
            uint8_t *output =
                destination + destinationVector * 16u;
            for (uint32_t index = 0u; index < chunk; ++index)
            {
                __m128i value{};
                if (componentWidth == 0u)
                {
                    value = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(source));
                }
                else if (componentWidth == 1u)
                {
                    const __m128i packed = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(source));
                    value = zeroExtend
                        ? _mm_cvtepu16_epi32(packed)
                        : _mm_cvtepi16_epi32(packed);
                }
                else
                {
                    uint32_t packed = 0u;
                    std::memcpy(&packed, source, sizeof(packed));
                    const __m128i bytes = _mm_cvtsi32_si128(
                        static_cast<int32_t>(packed));
                    value = zeroExtend
                        ? _mm_cvtepu8_epi32(bytes)
                        : _mm_cvtepi8_epi32(bytes);
                }

                if (mode == 1u)
                    value = _mm_add_epi32(value, rowValues);
                _mm_storeu_si128(
                    reinterpret_cast<__m128i *>(output), value);
                source += sourceStride;
                output += 16u;
            }
            remaining -= chunk;
            destinationVector = 0u;
        }
        return true;
    }

    bool applyVu1DecodedUnpackImpl(
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
        if (!command.maskEnabled &&
            command.componentCount == 4u &&
            command.vectorLength <= 2u &&
            cl == wl && mode <= 1u &&
            tryUnpackContiguousV4(
                dataMemory, dataMemorySize, vuAddress,
                sourceBase, command.writeVectorCount,
                command.vectorLength, command.zeroExtend,
                mode, vif.row.data()))
        {
            return true;
        }

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

    bool applyVu1DecodedUnpackBatchImpl(
        const Vu1DecodedUnpackBatch &batch,
        Vu1VifState &vif,
        uint8_t *dataMemory,
        uint32_t dataMemorySize)
    {
        if (batch.commands.empty())
            return false;
        for (const Vu1DecodedUnpackCommand &command :
             batch.commands)
        {
            if (command.vifStateBefore)
                vif = *command.vifStateBefore;
            if (!applyVu1DecodedUnpackImpl(
                    command, vif, dataMemory,
                    dataMemorySize))
            {
                return false;
            }
        }
        return true;
    }

    bool validPayload(const Vu1CommandPayload &payload) noexcept
    {
        return std::visit(
            [](const auto &value) noexcept
            {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, Vu1BindMemoryCommand>)
                {
                    return !value.microMemory.empty() &&
                           !value.dataMemory.empty() &&
                           value.microMemory.size() <=
                               std::numeric_limits<uint32_t>::max() &&
                           value.dataMemory.size() <=
                               std::numeric_limits<uint32_t>::max();
                }
                else if constexpr (
                    std::is_same_v<T, Vu1MicroMemoryWriteCommand> ||
                    std::is_same_v<T, Vu1DataMemoryWriteCommand>)
                {
                    return !value.bytes.empty();
                }
                else if constexpr (
                    std::is_same_v<T, Vu1DecodedUnpackCommand>)
                {
                    return validDecodedUnpackPayload(value);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1DecodedUnpackBatchCommand>)
                {
                    return value.batch &&
                           !value.batch->commands.empty() &&
                           std::all_of(
                               value.batch->commands.begin(),
                               value.batch->commands.end(),
                               validDecodedUnpackPayload);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1MscalCommand> ||
                    std::is_same_v<T, Vu1MscalfCommand> ||
                    std::is_same_v<T, Vu1MscntCommand>)
                {
                    return !value.unpacksBefore ||
                           (!value.unpacksBefore->commands.empty() &&
                            std::all_of(
                                value.unpacksBefore->commands.begin(),
                                value.unpacksBefore->commands.end(),
                                validDecodedUnpackPayload));
                }
                else if constexpr (
                    std::is_same_v<T, Vu1AdvanceSliceCommand>)
                {
                    return value.maximumCycles != 0u;
                }
                else if constexpr (
                    std::is_same_v<T, Vu1RestoreCommand>)
                {
                    return static_cast<bool>(value.snapshot);
                }
                else
                {
                    return true;
                }
            },
            payload);
    }
}

bool applyVu1DecodedUnpack(
    const Vu1DecodedUnpackCommand &command,
    Vu1VifState &vif,
    uint8_t *dataMemory,
    uint32_t dataMemorySize)
{
    return applyVu1DecodedUnpackImpl(
        command, vif, dataMemory, dataMemorySize);
}

const char *vu1CommandTypeName(Vu1CommandType type) noexcept
{
    switch (type)
    {
    case Vu1CommandType::BindMemory:
        return "bind-memory";
    case Vu1CommandType::SetDiagnostics:
        return "set-diagnostics";
    case Vu1CommandType::SetBackend:
        return "set-backend";
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
            if constexpr (std::is_same_v<T, Vu1BindMemoryCommand>)
                return Vu1CommandType::BindMemory;
            else if constexpr (std::is_same_v<T, Vu1SetDiagnosticsCommand>)
                return Vu1CommandType::SetDiagnostics;
            else if constexpr (std::is_same_v<T, Vu1SetBackendCommand>)
                return Vu1CommandType::SetBackend;
            else if constexpr (std::is_same_v<T, Vu1MicroMemoryWriteCommand>)
                return Vu1CommandType::MicroMemoryWrite;
            else if constexpr (std::is_same_v<T, Vu1DataMemoryWriteCommand>)
                return Vu1CommandType::DataMemoryWrite;
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackCommand> ||
                std::is_same_v<T, Vu1DecodedUnpackBatchCommand>)
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
            if constexpr (std::is_same_v<T, Vu1BindMemoryCommand>)
            {
                return 2u * sizeof(uint64_t) + sizeof(uint64_t) +
                       sizeof(bool) + value.microMemory.size() +
                       value.dataMemory.size();
            }
            else if constexpr (
                std::is_same_v<T, Vu1SetDiagnosticsCommand>)
            {
                return 2u * sizeof(bool);
            }
            else if constexpr (std::is_same_v<T, Vu1SetBackendCommand>)
            {
                return sizeof(uint8_t);
            }
            else if constexpr (
                std::is_same_v<T, Vu1MicroMemoryWriteCommand> ||
                std::is_same_v<T, Vu1DataMemoryWriteCommand>)
            {
                return sizeof(value.offset) + sizeof(value.wrap) +
                       sizeof(uint64_t) + value.bytes.size();
            }
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackCommand>)
            {
                return decodedUnpackPayloadSize(value);
            }
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackBatchCommand>)
            {
                return value.batch
                    ? decodedUnpackBatchPayloadSize(*value.batch)
                    : sizeof(uint64_t);
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
                return 3u * sizeof(uint32_t) +
                       (value.unpacksBefore
                            ? decodedUnpackBatchPayloadSize(
                                  *value.unpacksBefore)
                            : 0u) +
                       (value.vifStateBefore
                            ? 12u * sizeof(uint32_t)
                            : 0u);
            }
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
            {
                return 2u * sizeof(uint32_t) +
                       (value.unpacksBefore
                            ? decodedUnpackBatchPayloadSize(
                                  *value.unpacksBefore)
                            : 0u) +
                       (value.vifStateBefore
                            ? 12u * sizeof(uint32_t)
                            : 0u);
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
            else if constexpr (std::is_same_v<T, Vu1SnapshotCommand>)
            {
                return sizeof(bool);
            }
            else if constexpr (std::is_same_v<T, Vu1RestoreCommand>)
            {
                return sizeof(uint64_t) +
                       (value.snapshot
                            ? value.snapshot->microMemory.size() +
                                  value.snapshot->dataMemory.size()
                            : 0u);
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
            if constexpr (std::is_same_v<T, Vu1BindMemoryCommand>)
            {
                hashVector(hash, value.microMemory);
                hashVector(hash, value.dataMemory);
                hashScalar(hash, value.codeGeneration);
                hashScalar(hash, value.deferredDiagnostics);
            }
            else if constexpr (
                std::is_same_v<T, Vu1SetDiagnosticsCommand>)
            {
                hashScalar(hash, value.traceEnabled);
                hashScalar(hash, value.workloadProfileEnabled);
            }
            else if constexpr (std::is_same_v<T, Vu1SetBackendCommand>)
            {
                hashScalar(hash, value.backend);
            }
            else if constexpr (std::is_same_v<T, Vu1SnapshotCommand>)
            {
                hashScalar(hash, value.includeBackendDiagnostics);
            }
            else if constexpr (
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
                hashDecodedUnpack(hash, value);
            }
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackBatchCommand>)
            {
                const uint64_t count = value.batch
                    ? value.batch->commands.size()
                    : 0u;
                hashScalar(hash, count);
                if (value.batch)
                {
                    for (const Vu1DecodedUnpackCommand &command :
                         value.batch->commands)
                    {
                        hashDecodedUnpack(hash, command);
                    }
                }
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
                if (value.unpacksBefore)
                {
                    hashScalar(
                        hash,
                        static_cast<uint64_t>(
                            value.unpacksBefore->commands.size()));
                    for (const Vu1DecodedUnpackCommand &command :
                         value.unpacksBefore->commands)
                    {
                        hashDecodedUnpack(hash, command);
                    }
                }
                if (value.vifStateBefore)
                    hashVifState(hash, *value.vifStateBefore);
            }
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
            {
                hashScalar(hash, value.top);
                hashScalar(hash, value.itop);
                if (value.unpacksBefore)
                {
                    hashScalar(
                        hash,
                        static_cast<uint64_t>(
                            value.unpacksBefore->commands.size()));
                    for (const Vu1DecodedUnpackCommand &command :
                         value.unpacksBefore->commands)
                    {
                        hashDecodedUnpack(hash, command);
                    }
                }
                if (value.vifStateBefore)
                    hashVifState(hash, *value.vifStateBefore);
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
                if (value.snapshot)
                    hashScalar(hash, snapshotHash(*value.snapshot));
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

void replayVu1Diagnostics(
    std::span<const Vu1DiagnosticRecord> diagnostics,
    IVuExecutionObserver &observer)
{
    for (const Vu1DiagnosticRecord &record : diagnostics)
    {
        std::visit(
            [&](const auto &value)
            {
                using T = std::decay_t<decltype(value)>;
                if constexpr (
                    std::is_same_v<T, Vu1TraceInvocationDiagnostic>)
                {
                    observer.traceVuInvocation(
                        value.startPc, value.top, value.itop,
                        value.resume, value.state);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1TraceInstructionDiagnostic>)
                {
                    observer.traceVuInstruction(
                        value.pc, value.lower, value.upper,
                        value.state);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1TraceXgkickDiagnostic>)
                {
                    observer.traceVuXgkick(value.sourceQword);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1TraceInvocationEndDiagnostic>)
                {
                    observer.traceVuInvocationEnd(
                        value.finalPc, value.ended,
                        value.hitCycleLimit,
                        value.viRegisters.data(),
                        value.viRegisters.size());
                }
                else if constexpr (
                    std::is_same_v<T, Vu1WorkloadBeginDiagnostic>)
                {
                    observer.beginVuWorkloadProfileInvocation(
                        value.startPc, value.code.data(),
                        static_cast<uint32_t>(value.code.size()),
                        value.generation);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1WorkloadInstructionDiagnostic>)
                {
                    observer.recordVuWorkloadProfileInstruction(
                        value.pc, value.lower, value.upper);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1WorkloadTransitionDiagnostic>)
                {
                    observer.recordVuWorkloadProfileTransition(
                        value.pc, value.nextPc);
                }
                else if constexpr (
                    std::is_same_v<T, Vu1WorkloadEndDiagnostic>)
                {
                    observer.endVuWorkloadProfileInvocation(
                        value.completed);
                }
                else
                {
                    observer.resetVuWorkloadProfileEpoch();
                }
            },
            record);
    }
}

Vu1CommandProcessor::Vu1CommandProcessor(
    VuUnit &unit,
    Vu1CommandProcessorConfiguration configuration)
    : m_unit(unit),
      m_configuration(configuration),
      m_ownerThread(std::this_thread::get_id())
{
    if (unit.unitId() != VuUnitId::Vu1)
    {
        throw std::invalid_argument(
            "VU1 command processor requires a VU1 unit");
    }
}

void Vu1CommandProcessor::claimOwnerThread(
    std::thread::id owner)
{
    if (owner == std::thread::id{})
    {
        throw std::invalid_argument(
            "VU1 owner thread id must be valid");
    }
    m_ownerThread = owner;
}

void Vu1CommandProcessor::assertOwnerThread() const
{
    if (m_ownerThread != std::this_thread::get_id())
    {
        throw std::logic_error(
            "VU1 command processor accessed outside its owner thread");
    }
}

void Vu1CommandProcessor::bindMemory(
    uint8_t *microMemory, uint32_t microMemorySize,
    uint8_t *dataMemory, uint32_t dataMemorySize,
    uint64_t codeGeneration,
    IVuExecutionObserver *diagnosticsObserver)
{
    assertOwnerThread();
    if (!microMemory || microMemorySize == 0u ||
        !dataMemory || dataMemorySize == 0u)
    {
        throw std::invalid_argument(
            "VU1 command processor requires complete memory");
    }
    m_ownedMicroMemory.clear();
    m_ownedDataMemory.clear();
    m_microMemory = microMemory;
    m_microMemorySize = microMemorySize;
    m_dataMemory = dataMemory;
    m_dataMemorySize = dataMemorySize;
    m_codeGeneration.store(
        codeGeneration, std::memory_order_relaxed);
    m_diagnosticsObserver = diagnosticsObserver;
    m_deferredDiagnostics = false;
    m_traceEnabled = false;
    m_workloadProfileEnabled = false;
}

void Vu1CommandProcessor::bindInlineDiagnosticsObserver(
    IVuExecutionObserver *diagnosticsObserver)
{
    assertOwnerThread();
    if (m_deferredDiagnostics)
    {
        throw std::logic_error(
            "deferred VU1 diagnostics cannot bind an inline observer");
    }
    m_diagnosticsObserver = diagnosticsObserver;
}

Vu1CommandProcessor::SpeculativeSliceCheckpoint
Vu1CommandProcessor::captureProcessorCheckpoint()
{
    assertOwnerThread();
    if (!m_dataMemory || m_dataMemorySize == 0u)
    {
        throw std::logic_error(
            "VU1 speculation requires bound data memory");
    }

    const auto captureAt = std::chrono::steady_clock::now();
    SpeculativeSliceCheckpoint checkpoint{
        .state = m_unit.m_state,
        .vif = m_vifState,
        .progress = m_unit.getProgressSnapshot(),
        .lastExitReason = m_unit.m_lastExitReason,
        .verify = m_unit.m_verifyDiagnostics,
        .workloadProfileObserver =
            m_unit.m_workloadProfileObserver,
        .generation = m_generation,
        .nextSequence = m_nextSequence,
        .codeGeneration = codeGeneration(),
        .dataMemory = std::vector<uint8_t>(m_dataMemorySize),
        .workloadProfileInvocationPending =
            m_unit.m_workloadProfileInvocationPending,
        .workloadProfileInvocationActive =
            m_unit.m_workloadProfileInvocationActive,
    };
    std::memcpy(
        checkpoint.dataMemory.data(),
        m_dataMemory, m_dataMemorySize);

    const uint64_t captureNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - captureAt)
                .count());
    m_speculativeSlicesCaptured.fetch_add(
        1u, std::memory_order_relaxed);
    m_speculativeCheckpointBytesCopied.fetch_add(
        m_dataMemorySize, std::memory_order_relaxed);
    m_speculativeCheckpointCaptureNanoseconds.fetch_add(
        captureNanoseconds, std::memory_order_relaxed);
    return checkpoint;
}

void Vu1CommandProcessor::restoreProcessorCheckpoint(
    const SpeculativeSliceCheckpoint &checkpoint)
{
    assertOwnerThread();
    if (!m_dataMemory ||
        checkpoint.dataMemory.size() != m_dataMemorySize)
    {
        throw std::logic_error(
            "invalid VU1 processor checkpoint");
    }

    const auto rollbackAt = std::chrono::steady_clock::now();
    m_unit.m_state = checkpoint.state;
    m_vifState = checkpoint.vif;
    m_unit.m_progressActive.store(
        checkpoint.progress.active ? 1u : 0u,
        std::memory_order_relaxed);
    m_unit.m_progressInvocations.store(
        checkpoint.progress.invocations,
        std::memory_order_relaxed);
    m_unit.m_progressCycles.store(
        checkpoint.progress.cycles,
        std::memory_order_relaxed);
    m_unit.m_progressPc.store(
        checkpoint.progress.pc,
        std::memory_order_relaxed);
    m_unit.m_lastExitReason = checkpoint.lastExitReason;
    m_unit.m_verifyDiagnostics = checkpoint.verify;
    m_unit.m_workloadProfileObserver =
        checkpoint.workloadProfileObserver;
    m_unit.m_workloadProfileInvocationPending =
        checkpoint.workloadProfileInvocationPending;
    m_unit.m_workloadProfileInvocationActive =
        checkpoint.workloadProfileInvocationActive;
    m_generation = checkpoint.generation;
    m_nextSequence = checkpoint.nextSequence;
    m_codeGeneration.store(
        checkpoint.codeGeneration,
        std::memory_order_relaxed);
    std::memcpy(
        m_dataMemory,
        checkpoint.dataMemory.data(),
        m_dataMemorySize);
    m_pendingDiagnostics.clear();

    const uint64_t rollbackNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - rollbackAt)
                .count());
    m_speculativeSlicesRolledBack.fetch_add(
        1u, std::memory_order_relaxed);
    m_speculativeRollbackNanoseconds.fetch_add(
        rollbackNanoseconds, std::memory_order_relaxed);
}

void Vu1CommandProcessor::captureSpeculativeSliceCheckpoint()
{
    assertOwnerThread();
    if (m_speculativeSliceCheckpoint)
    {
        throw std::logic_error(
            "VU1 already has an unresolved speculative slice");
    }
    m_speculativeSliceCheckpoint.emplace(
        captureProcessorCheckpoint());
}

void Vu1CommandProcessor::rollbackSpeculativeSlice()
{
    assertOwnerThread();
    if (!m_speculativeSliceCheckpoint)
    {
        throw std::logic_error(
            "VU1 has no speculative slice to roll back");
    }

    SpeculativeSliceCheckpoint checkpoint =
        std::move(*m_speculativeSliceCheckpoint);
    m_speculativeSliceCheckpoint.reset();
    restoreProcessorCheckpoint(checkpoint);
}

void Vu1CommandProcessor::commitSpeculativeSlice()
{
    assertOwnerThread();
    if (!m_speculativeSliceCheckpoint)
    {
        throw std::logic_error(
            "VU1 has no speculative slice to commit");
    }
    m_speculativeSliceCheckpoint.reset();
    m_speculativeSlicesCommitted.fetch_add(
        1u, std::memory_order_relaxed);
}

void Vu1CommandProcessor::resolveSpeculativeSlice(
    Vu1SpeculationResolution resolution)
{
    assertOwnerThread();
    if (resolution == Vu1SpeculationResolution::None)
    {
        if (m_speculativeSliceCheckpoint)
        {
            throw std::logic_error(
                "VU1 speculative slice requires an explicit resolution");
        }
        return;
    }
    if (resolution == Vu1SpeculationResolution::Commit)
        commitSpeculativeSlice();
    else
        rollbackSpeculativeSlice();
}

Vu1SpeculationStatistics
Vu1CommandProcessor::speculationStatistics() const noexcept
{
    return Vu1SpeculationStatistics{
        .capturedSlices = m_speculativeSlicesCaptured.load(
            std::memory_order_acquire),
        .checkpointBytesCopied =
            m_speculativeCheckpointBytesCopied.load(
                std::memory_order_acquire),
        .checkpointCaptureNanoseconds =
            m_speculativeCheckpointCaptureNanoseconds.load(
                std::memory_order_acquire),
        .committedSlices = m_speculativeSlicesCommitted.load(
            std::memory_order_acquire),
        .rolledBackSlices =
            m_speculativeSlicesRolledBack.load(
                std::memory_order_acquire),
        .rollbackNanoseconds =
            m_speculativeRollbackNanoseconds.load(
                std::memory_order_acquire),
    };
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

Vu1Snapshot Vu1CommandProcessor::captureSnapshot(
    bool includeBackendDiagnostics) const
{
    Vu1Snapshot snapshot{
        .state = m_unit.state(),
        .vif = m_vifState,
        .codeGeneration = codeGeneration(),
        .lastExitReason = m_unit.lastExitReason(),
        .requestedBackend = m_unit.requestedBackend(),
        .resolvedBackend = m_unit.resolvedBackend(),
        .backendName = std::string(m_unit.backendName()),
        .nativeInstrumentationEnabled =
            m_unit.nativeInstrumentationEnabled(),
        .progress = m_unit.getProgressSnapshot(),
        .verify = m_unit.verifyDiagnostics(),
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
    if (includeBackendDiagnostics)
    {
        Vu1BackendDiagnosticsSnapshot diagnostics{};
        if (const VuProgramCache *const cache =
                m_unit.programCacheIfCreated())
        {
            diagnostics.programCache =
                cache->diagnosticsWhileExecutionQuiescent();
        }
        if (const VuRecompilerBackend *const backend =
                m_unit.recompilerIfCreated())
        {
            diagnostics.recompiler = Vu1RecompilerSnapshot{
                .diagnostics = backend->diagnostics(),
                .blockProfile =
                    backend->
                        blockProfilingSnapshotWhileExecutionQuiescent(),
                .lastJitDiagnostic = backend->lastJitDiagnostic(),
                .blockLinkingEnabled =
                    backend->blockLinkingEnabled(),
                .blockBudgetGuardsEnabled =
                    backend->blockBudgetGuardsEnabled(),
                .blockLocalVfRegistersEnabled =
                    backend->blockLocalVfRegistersEnabled(),
                .blockLocalVfRegistersAutomatic =
                    backend->blockLocalVfRegistersAutomatic(),
                .inlineXgkickEnabled =
                    backend->inlineXgkickEnabled(),
            };
        }
        snapshot.backendDiagnostics = std::move(diagnostics);
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

Vu1SliceResult Vu1CommandProcessor::advanceSlice(
    const Vu1AdvanceSliceCommand &command)
{
    Vu1SliceResult slice{};
    Vu1SliceSideEffectSink effects(slice.path1Packets);
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
        // Inline mode can use the already-narrow diagnostics observer
        // directly. A threaded owner records diagnostics locally through the
        // processor and returns the journal for publication by EE.
        .observer = m_diagnosticsObserver
                        ? m_diagnosticsObserver
                        : this,
        .enableInstrumentation =
            PS2X_ENABLE_RUNTIME_DIAGNOSTICS != 0,
        .enableNativeInstrumentation =
            m_unit.nativeInstrumentationEnabled(),
        .enableProgressAccounting =
            PS2X_ENABLE_RUNTIME_DIAGNOSTICS != 0,
    };
    slice.run = m_unit.advance(
        context, command.maximumCycles);
    if (command.captureState)
    {
        slice.state =
            std::make_unique<VuExecutionState>(
                m_unit.state());
    }
    slice.architecturalStateHash = stateHash();
    slice.vifCanResume = !m_unit.isActive();
    if (slice.run.reason == VuExitReason::Fault)
    {
        slice.fault = std::make_unique<std::string>(
            "VU1 backend reported a fault");
    }
    return slice;
}

Vu1CommandResultPayload Vu1CommandProcessor::apply(
    Vu1CommandPayload &payload,
    Vu1CommandDisposition &disposition)
{
    return std::visit(
        [&](auto &value) -> Vu1CommandResultPayload
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Vu1BindMemoryCommand>)
            {
                m_ownedMicroMemory = std::move(value.microMemory);
                m_ownedDataMemory = std::move(value.dataMemory);
                m_microMemory = m_ownedMicroMemory.data();
                m_microMemorySize = static_cast<uint32_t>(
                    m_ownedMicroMemory.size());
                m_dataMemory = m_ownedDataMemory.data();
                m_dataMemorySize = static_cast<uint32_t>(
                    m_ownedDataMemory.size());
                m_codeGeneration.store(
                    value.codeGeneration,
                    std::memory_order_relaxed);
                m_deferredDiagnostics =
                    value.deferredDiagnostics;
                if (m_deferredDiagnostics)
                    m_diagnosticsObserver = nullptr;
                return Vu1NoResult{};
            }
            else if constexpr (
                std::is_same_v<T, Vu1SetDiagnosticsCommand>)
            {
                m_traceEnabled = value.traceEnabled;
                m_workloadProfileEnabled =
                    value.workloadProfileEnabled;
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1SetBackendCommand>)
            {
                std::string diagnostic;
                const bool accepted = m_unit.setBackend(
                    value.backend, &diagnostic);
                return Vu1BackendStatusResult{
                    .accepted = accepted,
                    .requested = m_unit.requestedBackend(),
                    .resolved = m_unit.resolvedBackend(),
                    .name = std::string(m_unit.backendName()),
                    .active = m_unit.isActive(),
                    .diagnostic = std::move(diagnostic),
                };
            }
            else if constexpr (std::is_same_v<T, Vu1MicroMemoryWriteCommand>)
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
                if (value.vifStateBefore)
                    m_vifState = *value.vifStateBefore;
                if (!applyVu1DecodedUnpack(
                        value, m_vifState,
                        m_dataMemory, m_dataMemorySize))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                }
                return Vu1VifStateResult{m_vifState};
            }
            else if constexpr (
                std::is_same_v<T, Vu1DecodedUnpackBatchCommand>)
            {
                if (!value.batch ||
                    !applyVu1DecodedUnpackBatchImpl(
                        *value.batch, m_vifState,
                        m_dataMemory, m_dataMemorySize))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                }
                return Vu1VifStateResult{m_vifState};
            }
            else if constexpr (std::is_same_v<T, Vu1VifStateUpdateCommand>)
            {
                m_vifState = value.state;
                return Vu1VifStateResult{m_vifState};
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
                if (value.unpacksBefore &&
                    !applyVu1DecodedUnpackBatchImpl(
                        *value.unpacksBefore, m_vifState,
                        m_dataMemory, m_dataMemorySize))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                if (value.vifStateBefore)
                    m_vifState = *value.vifStateBefore;
                m_unit.start(
                    value.startPc, value.top, value.itop,
                    m_diagnosticsObserver
                        ? m_diagnosticsObserver
                        : this);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1MscalfCommand>)
            {
                if (value.unpacksBefore &&
                    !applyVu1DecodedUnpackBatchImpl(
                        *value.unpacksBefore, m_vifState,
                        m_dataMemory, m_dataMemorySize))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                if (value.vifStateBefore)
                    m_vifState = *value.vifStateBefore;
                m_unit.start(
                    value.startPc, value.top, value.itop,
                    m_diagnosticsObserver
                        ? m_diagnosticsObserver
                        : this);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1MscntCommand>)
            {
                if (value.unpacksBefore &&
                    !applyVu1DecodedUnpackBatchImpl(
                        *value.unpacksBefore, m_vifState,
                        m_dataMemory, m_dataMemorySize))
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                if (value.vifStateBefore)
                    m_vifState = *value.vifStateBefore;
                m_unit.resumeState(
                    value.top, value.itop,
                    m_diagnosticsObserver
                        ? m_diagnosticsObserver
                        : this);
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1AdvanceSliceCommand>)
            {
                return advanceSlice(value);
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
                return Vu1NoResult{};
            }
            else if constexpr (std::is_same_v<T, Vu1SnapshotCommand>)
            {
                Vu1SnapshotResult result{
                    .snapshot =
                        std::make_shared<const Vu1Snapshot>(
                            captureSnapshot(
                                value.includeBackendDiagnostics)),
                };
                result.architecturalStateHash =
                    m_configuration.captureArchitecturalStateHashes
                        ? snapshotHash(*result.snapshot)
                        : 0u;
                return result;
            }
            else if constexpr (std::is_same_v<T, Vu1RestoreCommand>)
            {
                if (!value.snapshot ||
                    value.snapshot->microMemory.size() !=
                        m_microMemorySize ||
                    value.snapshot->dataMemory.size() !=
                        m_dataMemorySize)
                {
                    disposition = Vu1CommandDisposition::Malformed;
                    return Vu1NoResult{};
                }
                const Vu1Snapshot &snapshot = *value.snapshot;
                m_unit.state() = snapshot.state;
                std::memcpy(
                    m_microMemory,
                    snapshot.microMemory.data(),
                    m_microMemorySize);
                std::memcpy(
                    m_dataMemory,
                    snapshot.dataMemory.data(),
                    m_dataMemorySize);
                m_vifState = snapshot.vif;
                m_codeGeneration.store(
                    snapshot.codeGeneration,
                    std::memory_order_relaxed);
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
    assertOwnerThread();
    if (!m_pendingDiagnostics.empty())
    {
        throw std::logic_error(
            "VU1 diagnostic journal was not published");
    }
    Vu1CommandResult result{
        .identity = command.identity,
        .ownerGeneration = m_generation,
        .codeGeneration = codeGeneration(),
        .programCounter = m_unit.state().pc,
        .active = m_unit.isActive(),
        .progress = m_unit.getProgressSnapshot(),
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
    const bool opensGeneration =
        startsVu1Generation(command.payload);
    if (opensGeneration)
    {
        if (command.identity.generation <= m_generation)
        {
            result.disposition =
                Vu1CommandDisposition::StaleGeneration;
            return result;
        }
        if (m_generation == std::numeric_limits<uint64_t>::max() ||
            command.identity.generation != m_generation + 1u)
        {
            result.disposition =
                Vu1CommandDisposition::FutureGeneration;
            return result;
        }
        if (command.identity.sequence != 1u)
        {
            result.disposition = Vu1CommandDisposition::OutOfOrder;
            return result;
        }
    }
    else
    {
        if (command.identity.generation < m_generation)
        {
            result.disposition =
                Vu1CommandDisposition::StaleGeneration;
            return result;
        }
        if (command.identity.generation > m_generation)
        {
            result.disposition =
                Vu1CommandDisposition::FutureGeneration;
            return result;
        }
        if (command.identity.sequence != m_nextSequence)
        {
            result.disposition = Vu1CommandDisposition::OutOfOrder;
            return result;
        }
    }
    const bool requiresBoundMemory =
        command.type != Vu1CommandType::BindMemory &&
        command.type != Vu1CommandType::SetDiagnostics &&
        command.type != Vu1CommandType::SetBackend &&
        command.type != Vu1CommandType::Reset &&
        command.type != Vu1CommandType::Snapshot &&
        command.type != Vu1CommandType::Barrier &&
        command.type != Vu1CommandType::Shutdown;
    if (command.type != vu1CommandType(command.payload) ||
        command.payloadSize != vu1CommandPayloadSize(command.payload) ||
        !validPayload(command.payload) ||
        (requiresBoundMemory &&
         (!m_microMemory || !m_dataMemory)))
    {
        result.disposition = Vu1CommandDisposition::Malformed;
        return result;
    }

    result.payload = apply(command.payload, result.disposition);
    result.diagnostics = std::move(m_pendingDiagnostics);
    m_pendingDiagnostics.clear();
    result.codeGeneration = codeGeneration();
    if (result.disposition == Vu1CommandDisposition::Completed)
    {
        if (opensGeneration)
        {
            m_generation = command.identity.generation;
            m_nextSequence = 2u;
        }
        else
        {
            if (m_nextSequence ==
                std::numeric_limits<uint64_t>::max())
            {
                throw std::overflow_error(
                    "VU1 command sequence exhausted");
            }
            ++m_nextSequence;
        }
    }
    result.ownerGeneration = m_generation;
    result.programCounter = m_unit.state().pc;
    result.active = m_unit.isActive();
    result.progress = m_unit.getProgressSnapshot();
    return result;
}

Vu1CommandResult Vu1CommandProcessor::processDecodedUnpack(
    Vu1WorkIdentity identity,
    const Vu1DecodedUnpackCommand &command)
{
    assertOwnerThread();
    if (!m_pendingDiagnostics.empty())
    {
        throw std::logic_error(
            "VU1 diagnostic journal was not published");
    }
    const uint64_t payloadSize =
        11u + sizeof(uint64_t) + command.bytes.size() +
        (command.vifStateBefore
             ? 12u * sizeof(uint32_t)
             : 0u);
    Vu1CommandResult result{
        .identity = identity,
        .digest = {
            .sequence = identity.sequence,
            .generation = identity.generation,
            .guestTick = identity.guestTick,
            .type = Vu1CommandType::DecodedUnpack,
            .payloadSize = payloadSize,
        },
        .ownerGeneration = m_generation,
        .codeGeneration = codeGeneration(),
        .programCounter = m_unit.state().pc,
        .active = m_unit.isActive(),
        .progress = m_unit.getProgressSnapshot(),
    };
    if (m_configuration.captureCommandDigests)
    {
        const Vu1CommandPayload payload{command};
        result.digest.payloadHash =
            vu1CommandPayloadHash(payload);
    }

    if (m_shutdown)
    {
        result.disposition = Vu1CommandDisposition::Shutdown;
        return result;
    }
    if (identity.generation < m_generation)
    {
        result.disposition = Vu1CommandDisposition::StaleGeneration;
        return result;
    }
    if (identity.generation > m_generation)
    {
        result.disposition = Vu1CommandDisposition::FutureGeneration;
        return result;
    }
    if (identity.sequence != m_nextSequence)
    {
        result.disposition = Vu1CommandDisposition::OutOfOrder;
        return result;
    }
    if (command.bytes.empty() ||
        command.writeVectorCount == 0u ||
        command.sourceVectorCount == 0u ||
        !m_microMemory || !m_dataMemory)
    {
        result.disposition = Vu1CommandDisposition::Malformed;
        return result;
    }

    if (command.vifStateBefore)
        m_vifState = *command.vifStateBefore;
    if (!applyVu1DecodedUnpack(
            command, m_vifState,
            m_dataMemory, m_dataMemorySize))
    {
        result.disposition = Vu1CommandDisposition::Malformed;
    }
    result.payload = Vu1VifStateResult{m_vifState};
    result.codeGeneration = codeGeneration();
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
    result.ownerGeneration = m_generation;
    result.programCounter = m_unit.state().pc;
    result.active = m_unit.isActive();
    result.progress = m_unit.getProgressSnapshot();
    return result;
}

Vu1CommandResult Vu1CommandProcessor::processVifStateUpdate(
    Vu1WorkIdentity identity,
    const Vu1VifStateUpdateCommand &command)
{
    assertOwnerThread();
    if (!m_pendingDiagnostics.empty())
    {
        throw std::logic_error(
            "VU1 diagnostic journal was not published");
    }
    constexpr uint64_t payloadSize =
        12u * sizeof(uint32_t);
    Vu1CommandResult result{
        .identity = identity,
        .digest = {
            .sequence = identity.sequence,
            .generation = identity.generation,
            .guestTick = identity.guestTick,
            .type = Vu1CommandType::VifStateUpdate,
            .payloadSize = payloadSize,
        },
        .ownerGeneration = m_generation,
        .codeGeneration = codeGeneration(),
        .programCounter = m_unit.state().pc,
        .active = m_unit.isActive(),
        .progress = m_unit.getProgressSnapshot(),
    };
    if (m_configuration.captureCommandDigests)
    {
        const Vu1CommandPayload payload{command};
        result.digest.payloadHash =
            vu1CommandPayloadHash(payload);
    }

    if (m_shutdown)
    {
        result.disposition = Vu1CommandDisposition::Shutdown;
        return result;
    }
    if (identity.generation < m_generation)
    {
        result.disposition = Vu1CommandDisposition::StaleGeneration;
        return result;
    }
    if (identity.generation > m_generation)
    {
        result.disposition = Vu1CommandDisposition::FutureGeneration;
        return result;
    }
    if (identity.sequence != m_nextSequence)
    {
        result.disposition = Vu1CommandDisposition::OutOfOrder;
        return result;
    }
    if (!m_microMemory || !m_dataMemory)
    {
        result.disposition = Vu1CommandDisposition::Malformed;
        return result;
    }

    m_vifState = command.state;
    result.payload = Vu1VifStateResult{m_vifState};
    result.codeGeneration = codeGeneration();
    if (m_nextSequence ==
        std::numeric_limits<uint64_t>::max())
    {
        throw std::overflow_error(
            "VU1 command sequence exhausted");
    }
    ++m_nextSequence;
    result.ownerGeneration = m_generation;
    result.programCounter = m_unit.state().pc;
    result.active = m_unit.isActive();
    result.progress = m_unit.getProgressSnapshot();
    return result;
}

Vu1CommandResult Vu1CommandProcessor::processAdvanceSlice(
    Vu1WorkIdentity identity,
    const Vu1AdvanceSliceCommand &command,
    bool speculative)
{
    return processAdvanceSliceImpl(
        identity, command, speculative, false);
}

Vu1CommandResult Vu1CommandProcessor::processEpochContinuation(
    Vu1WorkIdentity identity,
    const Vu1AdvanceSliceCommand &command)
{
    return processAdvanceSliceImpl(
        identity, command, false, true);
}

Vu1CommandResult Vu1CommandProcessor::processAdvanceSliceImpl(
    Vu1WorkIdentity identity,
    const Vu1AdvanceSliceCommand &command,
    bool speculative,
    bool epochContinuation)
{
    assertOwnerThread();
    if (!m_pendingDiagnostics.empty())
    {
        throw std::logic_error(
            "VU1 diagnostic journal was not published");
    }
    constexpr uint64_t payloadSize =
        sizeof(command.maximumCycles) +
        sizeof(command.captureState);
    Vu1CommandResult result{
        .identity = identity,
        .digest = {
            .sequence = identity.sequence,
            .generation = identity.generation,
            .guestTick = identity.guestTick,
            .type = Vu1CommandType::AdvanceSlice,
            .payloadSize = payloadSize,
        },
        .ownerGeneration = m_generation,
        .codeGeneration = codeGeneration(),
        .programCounter = m_unit.state().pc,
        .active = m_unit.isActive(),
        .progress = m_unit.getProgressSnapshot(),
    };
    if (m_configuration.captureCommandDigests)
    {
        const Vu1CommandPayload payload{command};
        result.digest.payloadHash =
            vu1CommandPayloadHash(payload);
    }

    if (m_shutdown)
    {
        result.disposition = Vu1CommandDisposition::Shutdown;
        return result;
    }
    if (identity.generation < m_generation)
    {
        result.disposition = Vu1CommandDisposition::StaleGeneration;
        return result;
    }
    if (identity.generation > m_generation)
    {
        result.disposition = Vu1CommandDisposition::FutureGeneration;
        return result;
    }
    const bool validSequence = epochContinuation
        ? identity.sequence !=
                  std::numeric_limits<uint64_t>::max() &&
              m_nextSequence == identity.sequence + 1u
        : identity.sequence == m_nextSequence;
    if (!validSequence)
    {
        result.disposition = Vu1CommandDisposition::OutOfOrder;
        return result;
    }
    if (command.maximumCycles == 0u ||
        !m_microMemory || !m_dataMemory)
    {
        result.disposition = Vu1CommandDisposition::Malformed;
        return result;
    }
    if (!epochContinuation &&
        m_nextSequence ==
        std::numeric_limits<uint64_t>::max())
    {
        throw std::overflow_error(
            "VU1 command sequence exhausted");
    }

    if (speculative)
        captureSpeculativeSliceCheckpoint();
    try
    {
        result.payload = advanceSlice(command);
        result.diagnostics = std::move(m_pendingDiagnostics);
        m_pendingDiagnostics.clear();
        result.codeGeneration = codeGeneration();
        if (!epochContinuation)
            ++m_nextSequence;
    }
    catch (...)
    {
        if (speculative && m_speculativeSliceCheckpoint)
            rollbackSpeculativeSlice();
        throw;
    }
    result.ownerGeneration = m_generation;
    result.programCounter = m_unit.state().pc;
    result.active = m_unit.isActive();
    result.progress = m_unit.getProgressSnapshot();
    return result;
}

Vu1CommandResult
Vu1CommandProcessor::commitExtendedSpeculativeAdvance(
    Vu1WorkIdentity identity,
    const Vu1AdvanceSliceCommand &publishedCommand,
    uint32_t extensionCycles)
{
    assertOwnerThread();
    if (!m_speculativeSliceCheckpoint)
    {
        throw std::logic_error(
            "VU1 has no speculative slice to extend");
    }
    if (!m_pendingDiagnostics.empty())
    {
        throw std::logic_error(
            "VU1 diagnostic journal was not published");
    }
    if (publishedCommand.maximumCycles == 0u ||
        extensionCycles > publishedCommand.maximumCycles ||
        identity.generation != m_generation ||
        identity.sequence ==
            std::numeric_limits<uint64_t>::max() ||
        m_nextSequence != identity.sequence + 1u ||
        !m_microMemory || !m_dataMemory)
    {
        throw std::logic_error(
            "invalid VU1 speculative extension");
    }

    constexpr uint64_t payloadSize =
        sizeof(publishedCommand.maximumCycles) +
        sizeof(publishedCommand.captureState);
    Vu1CommandResult result{
        .identity = identity,
        .digest = {
            .sequence = identity.sequence,
            .generation = identity.generation,
            .guestTick = identity.guestTick,
            .type = Vu1CommandType::AdvanceSlice,
            .payloadSize = payloadSize,
        },
        .ownerGeneration = m_generation,
        .codeGeneration = codeGeneration(),
        .programCounter = m_unit.state().pc,
        .active = m_unit.isActive(),
        .progress = m_unit.getProgressSnapshot(),
    };
    if (m_configuration.captureCommandDigests)
    {
        const Vu1CommandPayload payload{publishedCommand};
        result.digest.payloadHash =
            vu1CommandPayloadHash(payload);
    }

    try
    {
        Vu1SliceResult extension{};
        if (extensionCycles != 0u)
        {
            Vu1AdvanceSliceCommand extensionCommand =
                publishedCommand;
            extensionCommand.maximumCycles = extensionCycles;
            extension = advanceSlice(extensionCommand);
        }
        else
        {
            const bool active = m_unit.isActive();
            extension.run = {
                .requestedCycles = 0u,
                .executedCycles = 0u,
                .reason = active
                              ? VuExitReason::CycleBudget
                              : VuExitReason::Inactive,
                .activeBefore = active,
                .activeAfter = active,
                .completed = !active,
            };
            if (publishedCommand.captureState)
            {
                extension.state =
                    std::make_unique<VuExecutionState>(
                        m_unit.state());
            }
            extension.architecturalStateHash = stateHash();
            extension.vifCanResume = !active;
        }
        result.payload = std::move(extension);
        result.diagnostics = std::move(m_pendingDiagnostics);
        m_pendingDiagnostics.clear();
        result.codeGeneration = codeGeneration();
        commitSpeculativeSlice();
    }
    catch (...)
    {
        if (m_speculativeSliceCheckpoint)
            rollbackSpeculativeSlice();
        throw;
    }

    result.ownerGeneration = m_generation;
    result.programCounter = m_unit.state().pc;
    result.active = m_unit.isActive();
    result.progress = m_unit.getProgressSnapshot();
    return result;
}

bool Vu1CommandProcessor::vuTraceEnabled() const
{
    if (m_deferredDiagnostics)
        return m_traceEnabled;
    return m_diagnosticsObserver &&
           m_diagnosticsObserver->vuTraceEnabled();
}

bool Vu1CommandProcessor::vuWorkloadProfileEnabled() const
{
    if (m_deferredDiagnostics)
        return m_workloadProfileEnabled;
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
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1TraceInvocationDiagnostic{
                .startPc = startPc,
                .top = top,
                .itop = itop,
                .resume = resume,
                .state = state,
            });
        return;
    }
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
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1TraceInstructionDiagnostic{
                .pc = pc,
                .lower = lower,
                .upper = upper,
                .state = state,
            });
        return;
    }
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->traceVuInstruction(
            pc, lower, upper, state);
    }
}

void Vu1CommandProcessor::traceVuXgkick(uint32_t sourceQword)
{
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1TraceXgkickDiagnostic{sourceQword});
        return;
    }
    if (m_diagnosticsObserver)
        m_diagnosticsObserver->traceVuXgkick(sourceQword);
}

void Vu1CommandProcessor::traceVuInvocationEnd(
    uint32_t finalPc, bool ended, bool hitCycleLimit,
    const int32_t *viRegisters, size_t viRegisterCount)
{
    if (m_deferredDiagnostics)
    {
        Vu1TraceInvocationEndDiagnostic record{
            .finalPc = finalPc,
            .ended = ended,
            .hitCycleLimit = hitCycleLimit,
        };
        if (viRegisters && viRegisterCount != 0u)
        {
            record.viRegisters.assign(
                viRegisters,
                viRegisters + viRegisterCount);
        }
        m_pendingDiagnostics.emplace_back(std::move(record));
        return;
    }
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
    if (m_deferredDiagnostics)
    {
        Vu1WorkloadBeginDiagnostic record{
            .startPc = startPc,
            .generation = generation,
        };
        if (code && codeSize != 0u)
            record.code.assign(code, code + codeSize);
        m_pendingDiagnostics.emplace_back(std::move(record));
        return;
    }
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->beginVuWorkloadProfileInvocation(
            startPc, code, codeSize, generation);
    }
}

void Vu1CommandProcessor::recordVuWorkloadProfileInstruction(
    uint32_t pc, uint32_t lower, uint32_t upper)
{
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1WorkloadInstructionDiagnostic{
                .pc = pc,
                .lower = lower,
                .upper = upper,
            });
        return;
    }
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->recordVuWorkloadProfileInstruction(
            pc, lower, upper);
    }
}

void Vu1CommandProcessor::recordVuWorkloadProfileTransition(
    uint32_t pc, uint32_t nextPc)
{
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1WorkloadTransitionDiagnostic{
                .pc = pc,
                .nextPc = nextPc,
            });
        return;
    }
    if (m_diagnosticsObserver)
    {
        m_diagnosticsObserver->recordVuWorkloadProfileTransition(
            pc, nextPc);
    }
}

void Vu1CommandProcessor::endVuWorkloadProfileInvocation(
    bool completed)
{
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1WorkloadEndDiagnostic{completed});
        return;
    }
    if (m_diagnosticsObserver)
        m_diagnosticsObserver->endVuWorkloadProfileInvocation(completed);
}

void Vu1CommandProcessor::resetVuWorkloadProfileEpoch()
{
    if (m_deferredDiagnostics)
    {
        m_pendingDiagnostics.emplace_back(
            Vu1WorkloadResetDiagnostic{});
        return;
    }
    if (m_diagnosticsObserver)
        m_diagnosticsObserver->resetVuWorkloadProfileEpoch();
}

Vu1CommandResult Vu1CommandExecutor::submitAdvanceSlice(
    Vu1AdvanceSliceCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return submit(
        Vu1CommandPayload{command},
        guestTick, publicationToken);
}

Vu1CommandResult Vu1CommandExecutor::submitDecodedUnpack(
    Vu1DecodedUnpackCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return submit(
        Vu1CommandPayload{std::move(command)},
        guestTick, publicationToken);
}

Vu1CommandResult Vu1CommandExecutor::submitVifStateUpdate(
    Vu1VifStateUpdateCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return submit(
        Vu1CommandPayload{command},
        guestTick, publicationToken);
}

InlineVu1Executor::InlineVu1Executor(
    Vu1CommandProcessor &processor)
    : m_processor(processor),
      m_generation(processor.generation()),
      m_nextSequence(processor.nextSequence())
{
    m_processor.claimOwnerThread(std::this_thread::get_id());
}

Vu1WorkIdentity InlineVu1Executor::nextIdentity(
    bool startsGeneration,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    guestTick = std::max(guestTick, m_lastGuestTick);
    if (startsGeneration)
    {
        return {
            .sequence = 1u,
            .generation = checkedVu1Increment(
                m_generation, "generation"),
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        };
    }
    if (m_nextSequence == 0u)
    {
        throw std::overflow_error(
            "VU1 command sequence exhausted");
    }
    return {
        .sequence = m_nextSequence,
        .generation = m_generation,
        .guestTick = guestTick,
        .publicationToken = publicationToken,
    };
}

void InlineVu1Executor::completeIdentity(
    bool startsGeneration,
    const Vu1CommandResult &result)
{
    if (result.disposition != Vu1CommandDisposition::Completed)
        return;
    if (startsGeneration)
    {
        m_generation = result.identity.generation;
        m_nextSequence = 2u;
    }
    else if (result.identity.sequence ==
             std::numeric_limits<uint64_t>::max())
    {
        m_nextSequence = 0u;
    }
    else
    {
        m_nextSequence = result.identity.sequence + 1u;
    }
    m_lastGuestTick = result.identity.guestTick;
}

Vu1CommandResult InlineVu1Executor::submit(
    Vu1CommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    // Inline execution has no persistent worker. The runtime serializes its
    // EE and control callers at a guest boundary, so the caller becomes the
    // physical owner for this command without adding a hot-path mutex.
    m_processor.claimOwnerThread(std::this_thread::get_id());
    const bool opensGeneration = startsVu1Generation(payload);
    const Vu1WorkIdentity identity = nextIdentity(
        opensGeneration, guestTick, publicationToken);
    Vu1Command command{
        .identity = identity,
        .type = vu1CommandType(payload),
        .payloadSize = vu1CommandPayloadSize(payload),
        .payload = std::move(payload),
    };
    Vu1CommandResult result =
        m_processor.process(std::move(command));
    completeIdentity(opensGeneration, result);
    return result;
}

Vu1CommandResult InlineVu1Executor::submitAdvanceSlice(
    Vu1AdvanceSliceCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    m_processor.claimOwnerThread(std::this_thread::get_id());
    const Vu1WorkIdentity identity = nextIdentity(
        false, guestTick, publicationToken);
    Vu1CommandResult result =
        m_processor.processAdvanceSlice(identity, command);
    completeIdentity(false, result);
    return result;
}

Vu1CommandResult InlineVu1Executor::submitDecodedUnpack(
    Vu1DecodedUnpackCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    m_processor.claimOwnerThread(std::this_thread::get_id());
    const Vu1WorkIdentity identity = nextIdentity(
        false, guestTick, publicationToken);
    Vu1CommandResult result =
        m_processor.processDecodedUnpack(identity, command);
    completeIdentity(false, result);
    return result;
}

Vu1CommandResult InlineVu1Executor::submitVifStateUpdate(
    Vu1VifStateUpdateCommand command,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    m_processor.claimOwnerThread(std::this_thread::get_id());
    const Vu1WorkIdentity identity = nextIdentity(
        false, guestTick, publicationToken);
    Vu1CommandResult result =
        m_processor.processVifStateUpdate(identity, command);
    completeIdentity(false, result);
    return result;
}

Vu1CommandSubmission::Vu1CommandSubmission(
    uint64_t ticket,
    Vu1WorkIdentity identity,
    Vu1CommandType type,
    std::future<Vu1CommandResult> completion)
    : m_ticket(ticket),
      m_identity(identity),
      m_type(type),
      m_completion(std::move(completion))
{
}

bool Vu1CommandSubmission::valid() const noexcept
{
    return m_completion.valid();
}

bool Vu1CommandSubmission::ready() const
{
    if (!m_completion.valid())
        return false;
    return m_completion.wait_for(
               std::chrono::seconds(0)) ==
           std::future_status::ready;
}

Vu1CommandResult Vu1CommandSubmission::wait()
{
    if (!m_completion.valid())
    {
        throw std::logic_error(
            "VU1 command submission has no completion");
    }
    return m_completion.get();
}

namespace
{
    std::exception_ptr makeVu1ExecutorCancellation()
    {
        return std::make_exception_ptr(
            std::runtime_error(
                "threaded VU1 executor cancelled queued work"));
    }

    Vu1CommandResult makeStaleQueuedVu1Result(
        Vu1Command &command,
        uint64_t ownerGeneration,
        uint64_t codeGeneration)
    {
        Vu1CommandResult result{};
        result.identity = command.identity;
        result.disposition =
            Vu1CommandDisposition::StaleGeneration;
        result.digest = {
            .sequence = command.identity.sequence,
            .generation = command.identity.generation,
            .guestTick = command.identity.guestTick,
            .type = command.type,
            .payloadSize = command.payloadSize,
            .payloadHash = 0u,
        };
        result.ownerGeneration = ownerGeneration;
        result.codeGeneration = codeGeneration;
        return result;
    }

    template <typename T>
    void updateVu1AtomicMaximum(
        std::atomic<T> &destination,
        T candidate) noexcept
    {
        T current = destination.load(
            std::memory_order_relaxed);
        while (current < candidate &&
               !destination.compare_exchange_weak(
                   current, candidate,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed))
        {
        }
    }

    bool supportsVu1ResultlessAdmission(
        const Vu1CommandPayload &payload) noexcept
    {
        switch (vu1CommandType(payload))
        {
        case Vu1CommandType::MicroMemoryWrite:
        case Vu1CommandType::DataMemoryWrite:
        case Vu1CommandType::DecodedUnpack:
        case Vu1CommandType::VifStateUpdate:
            return true;
        default:
            return false;
        }
    }
}

ThreadedVu1Executor::ThreadedVu1Executor(
    Vu1CommandProcessor &processor,
    ThreadedVu1ExecutorOptions options)
    : m_processor(processor),
      m_options(std::move(options)),
      m_queue(m_options.queueCapacity),
      m_generation(processor.generation()),
      m_nextSequence(processor.nextSequence()),
      m_minimumAcceptedGeneration(processor.generation())
{
    if (m_options.payloadCapacityBytes == 0u)
    {
        throw std::invalid_argument(
            "threaded VU1 payload capacity must be non-zero");
    }
}

ThreadedVu1Executor::~ThreadedVu1Executor()
{
    shutdown(Vu1ExecutorShutdownMode::Drain);
}

bool ThreadedVu1Executor::startsGeneration(
    const Vu1CommandPayload &payload) noexcept
{
    return startsVu1Generation(payload);
}

void ThreadedVu1Executor::ensureStarted()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    if (m_started)
        return;
    if (!m_accepting.load(std::memory_order_acquire))
    {
        throw std::runtime_error(
            "threaded VU1 executor is not accepting work");
    }

    try
    {
        m_worker = std::thread(
            [this]()
            {
                workerMain();
            });
        m_started = true;
    }
    catch (...)
    {
        m_fatalFailure = std::current_exception();
        m_accepting.store(false, std::memory_order_release);
        throw;
    }
}

void ThreadedVu1Executor::updateQueueHighWater(
    size_t depth) noexcept
{
    updateVu1AtomicMaximum(m_queueHighWater, depth);
}

void ThreadedVu1Executor::updatePayloadHighWater(
    uint64_t bytes) noexcept
{
    updateVu1AtomicMaximum(m_payloadHighWaterBytes, bytes);
}

ThreadedVu1Executor::AdmissionResult
ThreadedVu1Executor::tryEnqueue(WorkItem &item)
{
    if (m_queue.full())
        return AdmissionResult::SlotCapacity;

    const uint64_t currentPayload =
        m_queuedPayloadBytes.load(std::memory_order_acquire);
    if (item.payloadBytes >
        m_options.payloadCapacityBytes -
            std::min(
                currentPayload,
                m_options.payloadCapacityBytes))
    {
        return AdmissionResult::PayloadCapacity;
    }

    const size_t reservedDepth = m_queue.size() + 1u;
    const uint64_t reserved =
        m_queuedPayloadBytes.fetch_add(
            item.payloadBytes,
            std::memory_order_acq_rel) +
        item.payloadBytes;
    if (!m_queue.tryEmplace(std::move(item)))
    {
        m_queuedPayloadBytes.fetch_sub(
            item.payloadBytes,
            std::memory_order_acq_rel);
        return AdmissionResult::SlotCapacity;
    }

    updateQueueHighWater(reservedDepth);
    updatePayloadHighWater(reserved);
    return AdmissionResult::Enqueued;
}

void ThreadedVu1Executor::signalWorkAvailable() noexcept
{
    m_workNotificationCount.fetch_add(
        1u, std::memory_order_relaxed);
    if (m_workSignals.exchange(
            1u, std::memory_order_release) == 0u)
    {
        m_workSignals.notify_one();
    }
}

bool ThreadedVu1Executor::tryAcquireWorkSignal() noexcept
{
    return m_workSignals.exchange(
               0u, std::memory_order_acq_rel) != 0u;
}

void ThreadedVu1Executor::acquireWorkSignal() noexcept
{
    while (!tryAcquireWorkSignal())
    {
        m_workSignals.wait(
            0u, std::memory_order_acquire);
    }
}

void ThreadedVu1Executor::signalSpaceAvailable() noexcept
{
    const uint64_t prior = m_spaceEpoch.fetch_add(
        1u, std::memory_order_release);
    if (prior == std::numeric_limits<uint64_t>::max())
        std::terminate();
    m_spaceEpoch.notify_all();
}

void ThreadedVu1Executor::closeAdmissionAndQuiesceProducer() noexcept
{
    m_accepting.store(false, std::memory_order_release);
    signalSpaceAvailable();

    std::lock_guard<std::mutex> submitLock(m_submitMutex);
    requestActiveExecutionEpochStopLocked();
}

void ThreadedVu1Executor::requestActiveExecutionEpochStopLocked() noexcept
{
    if (!m_activeExecutionEpoch)
        return;
    {
        std::lock_guard<std::mutex> lock(
            m_activeExecutionEpoch->mutex);
        m_activeExecutionEpoch->stopRequested = true;
    }
    m_activeExecutionEpoch->changed.notify_all();
    m_activeExecutionEpoch.reset();
}

[[noreturn]] void
ThreadedVu1Executor::throwSubmissionUnavailable() const
{
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        failure = m_fatalFailure;
    }
    if (failure)
        std::rethrow_exception(failure);
    throw std::runtime_error(
        "threaded VU1 executor is not accepting work");
}

Vu1CommandSubmission ThreadedVu1Executor::submitAsyncImpl(
    Vu1CommandPayload payload,
    Vu1SpeculationResolution resolution,
    bool beginsSpeculativeSlice,
    bool commitsExtendedSpeculativeAdvance,
    uint32_t speculativeExtensionCycles,
    bool retainCompletion,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    std::unique_lock<std::mutex> submitLock(m_submitMutex);
    if (!m_accepting.load(std::memory_order_acquire))
        throwSubmissionUnavailable();

    // Any later ordered command observes the last EE-published epoch
    // checkpoint. Ask the owner to discard its unpublished lookahead before
    // the command reaches the head of the same FIFO.
    requestActiveExecutionEpochStopLocked();

    std::optional<Vu1WorkIdentity> extendedIdentity;
    if (commitsExtendedSpeculativeAdvance)
    {
        const auto *const advance =
            std::get_if<Vu1AdvanceSliceCommand>(&payload);
        if (resolution != Vu1SpeculationResolution::None ||
            beginsSpeculativeSlice ||
            !m_producerSpeculationCheckpoint ||
            !advance ||
            advance->maximumCycles == 0u ||
            speculativeExtensionCycles >
                advance->maximumCycles ||
            publicationToken !=
                m_producerSpeculationCheckpoint->
                    identity.publicationToken)
        {
            throw std::logic_error(
                "invalid threaded VU1 speculative extension");
        }
        extendedIdentity =
            m_producerSpeculationCheckpoint->identity;
    }
    else if (resolution == Vu1SpeculationResolution::None)
    {
        if (m_producerSpeculationCheckpoint)
        {
            throw std::logic_error(
                "threaded VU1 speculation requires an explicit resolution");
        }
    }
    else
    {
        if (!m_producerSpeculationCheckpoint)
        {
            throw std::logic_error(
                "threaded VU1 has no speculation to resolve");
        }
        if (resolution == Vu1SpeculationResolution::Rollback)
        {
            m_generation =
                m_producerSpeculationCheckpoint->generation;
            m_nextSequence =
                m_producerSpeculationCheckpoint->nextSequence;
            m_lastGuestTick =
                m_producerSpeculationCheckpoint->lastGuestTick;
        }
        m_producerSpeculationCheckpoint.reset();
    }
    if (beginsSpeculativeSlice &&
        !std::holds_alternative<Vu1AdvanceSliceCommand>(payload))
    {
        throw std::invalid_argument(
            "only a VU1 advance may begin speculation");
    }

    ensureStarted();
    const uint64_t previousGeneration = m_generation;
    const uint64_t previousNextSequence = m_nextSequence;
    const uint64_t previousLastGuestTick = m_lastGuestTick;
    guestTick = std::max(guestTick, m_lastGuestTick);

    uint64_t generation =
        extendedIdentity
            ? extendedIdentity->generation
            : m_generation;
    uint64_t sequence =
        extendedIdentity
            ? extendedIdentity->sequence
            : m_nextSequence;
    if (!extendedIdentity && startsGeneration(payload))
    {
        generation = checkedVu1Increment(
            generation, "generation");
        sequence = 1u;
    }
    else if (!extendedIdentity && sequence == 0u)
    {
        throw std::overflow_error(
            "VU1 command sequence exhausted");
    }
    if (extendedIdentity &&
        (generation != m_generation ||
         sequence == std::numeric_limits<uint64_t>::max() ||
         m_nextSequence != sequence + 1u))
    {
        throw std::logic_error(
            "threaded VU1 speculative extension identity is stale");
    }
    const uint64_t ticket = checkedVu1Increment(
        m_nextTicket, "transport ticket");
    const uint64_t payloadBytes =
        vu1CommandPayloadSize(payload);
    const uint64_t decodedUnpackOperations =
        decodedUnpackOperationCount(payload);
    if (payloadBytes > m_options.payloadCapacityBytes)
    {
        throw std::length_error(
            "VU1 command exceeds the bounded payload capacity");
    }

    Vu1Command command{
        .identity = {
            .sequence = sequence,
            .generation = generation,
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        },
        .type = vu1CommandType(payload),
        .payloadSize = payloadBytes,
        .payload = std::move(payload),
    };
    const Vu1WorkIdentity identity = command.identity;
    const Vu1CommandType commandType = command.type;
    WorkItem item{
        .ticket = ticket,
        .payloadBytes = payloadBytes,
        .decodedUnpackOperations =
            decodedUnpackOperations,
        .command = std::move(command),
        .speculationResolution = resolution,
        .beginsSpeculativeSlice = beginsSpeculativeSlice,
        .commitsExtendedSpeculativeAdvance =
            commitsExtendedSpeculativeAdvance,
        .speculativeExtensionCycles =
            speculativeExtensionCycles,
    };
    std::future<Vu1CommandResult> completion;
    if (retainCompletion)
    {
        item.completion.emplace();
        completion = item.completion->get_future();
    }

    bool countedBlock = false;
    std::chrono::steady_clock::time_point blockedAt{};
    const auto recordBlockedDuration = [&]()
    {
        if (!countedBlock)
            return;
        const uint64_t nanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - blockedAt)
                    .count());
        m_producerBlockedNanoseconds.fetch_add(
            nanoseconds, std::memory_order_relaxed);
        countedBlock = false;
    };
    for (;;)
    {
        if (!m_accepting.load(std::memory_order_acquire))
        {
            recordBlockedDuration();
            throwSubmissionUnavailable();
        }

        const uint64_t spaceEpoch = m_spaceEpoch.load(
            std::memory_order_acquire);
        const AdmissionResult admission = tryEnqueue(item);
        if (admission == AdmissionResult::Enqueued)
            break;
        if (!countedBlock)
        {
            countedBlock = true;
            blockedAt = std::chrono::steady_clock::now();
            m_producerBlockCount.fetch_add(
                1u, std::memory_order_relaxed);
        }

        if (!m_accepting.load(std::memory_order_acquire))
        {
            recordBlockedDuration();
            throwSubmissionUnavailable();
        }

        const auto waitAt = std::chrono::steady_clock::now();
        m_spaceEpoch.wait(
            spaceEpoch, std::memory_order_acquire);
        const uint64_t waitNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - waitAt)
                    .count());
        if (admission == AdmissionResult::SlotCapacity)
        {
            m_producerSlotWaitCount.fetch_add(
                1u, std::memory_order_relaxed);
            m_producerSlotWaitNanoseconds.fetch_add(
                waitNanoseconds, std::memory_order_relaxed);
        }
        else
        {
            m_producerPayloadWaitCount.fetch_add(
                1u, std::memory_order_relaxed);
            m_producerPayloadWaitNanoseconds.fetch_add(
                waitNanoseconds, std::memory_order_relaxed);
        }
    }
    recordBlockedDuration();

    if (commitsExtendedSpeculativeAdvance)
    {
        m_producerSpeculationCheckpoint.reset();
    }
    else
    {
        m_generation = generation;
        m_nextSequence =
            sequence == std::numeric_limits<uint64_t>::max()
                ? 0u
                : sequence + 1u;
    }
    m_nextTicket = ticket;
    m_lastGuestTick = guestTick;
    m_submittedTickets.store(
        ticket, std::memory_order_release);
    m_submittedGeneration.store(
        generation, std::memory_order_release);
    m_submittedSequence.store(
        sequence, std::memory_order_release);
    m_submittedCommandsByType[
        vu1CommandTypeIndex(commandType)]
        .fetch_add(1u, std::memory_order_relaxed);
    m_submittedDecodedUnpackOperations.fetch_add(
        decodedUnpackOperations,
        std::memory_order_relaxed);
    if (beginsSpeculativeSlice)
    {
        m_producerSpeculationCheckpoint.emplace(
            ProducerSpeculationCheckpoint{
                .identity = identity,
                .generation = previousGeneration,
                .nextSequence = previousNextSequence,
                .lastGuestTick = previousLastGuestTick,
            });
    }
    signalWorkAvailable();
    submitLock.unlock();

    return Vu1CommandSubmission(
        ticket, identity, commandType,
        std::move(completion));
}

Vu1CommandSubmission ThreadedVu1Executor::submitAsync(
    Vu1CommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return submitAsyncImpl(
        std::move(payload),
        Vu1SpeculationResolution::None,
        false, false, 0u, true,
        guestTick, publicationToken);
}

Vu1CommandAdmission ThreadedVu1Executor::submitResultless(
    Vu1CommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!supportsVu1ResultlessAdmission(payload))
    {
        throw std::invalid_argument(
            "response-producing VU1 command requires a completion");
    }
    Vu1CommandSubmission submission = submitAsyncImpl(
        std::move(payload),
        Vu1SpeculationResolution::None,
        false, false, 0u, false,
        guestTick, publicationToken);
    return {
        .ticket = submission.ticket(),
        .identity = submission.identity(),
        .type = submission.type(),
    };
}

Vu1ExecutionEpoch ThreadedVu1Executor::submitExecutionEpoch(
    Vu1ExecutionEpochOptions options,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    constexpr size_t kMaximumCheckpointCapacity = 64u;
    if (options.initialCycles == 0u ||
        options.followupCycles == 0u ||
        options.checkpointCapacity == 0u ||
        options.checkpointCapacity >
            kMaximumCheckpointCapacity)
    {
        throw std::invalid_argument(
            "invalid threaded VU1 execution epoch options");
    }

    std::unique_lock<std::mutex> submitLock(m_submitMutex);
    if (!m_accepting.load(std::memory_order_acquire))
        throwSubmissionUnavailable();
    if (m_producerSpeculationCheckpoint)
    {
        throw std::logic_error(
            "threaded VU1 execution epoch conflicts with slice speculation");
    }
    requestActiveExecutionEpochStopLocked();
    ensureStarted();

    guestTick = std::max(guestTick, m_lastGuestTick);
    const uint64_t generation = m_generation;
    const uint64_t sequence = m_nextSequence;
    if (sequence == 0u ||
        sequence == std::numeric_limits<uint64_t>::max())
    {
        throw std::overflow_error(
            "VU1 command sequence exhausted");
    }
    const uint64_t ticket = checkedVu1Increment(
        m_nextTicket, "transport ticket");
    Vu1CommandPayload payload{
        Vu1AdvanceSliceCommand{
            .maximumCycles = options.initialCycles,
            .captureState = options.captureState,
        }};
    const uint64_t payloadBytes =
        vu1CommandPayloadSize(payload);
    auto state = std::make_shared<Vu1ExecutionEpochState>(
        options);
    Vu1Command command{
        .identity = {
            .sequence = sequence,
            .generation = generation,
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        },
        .type = Vu1CommandType::AdvanceSlice,
        .payloadSize = payloadBytes,
        .payload = std::move(payload),
    };
    const Vu1WorkIdentity identity = command.identity;
    WorkItem item{
        .ticket = ticket,
        .payloadBytes = payloadBytes,
        .command = std::move(command),
        .executionEpoch = state,
    };

    bool countedBlock = false;
    std::chrono::steady_clock::time_point blockedAt{};
    const auto recordBlockedDuration = [&]()
    {
        if (!countedBlock)
            return;
        const uint64_t nanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - blockedAt)
                    .count());
        m_producerBlockedNanoseconds.fetch_add(
            nanoseconds, std::memory_order_relaxed);
        countedBlock = false;
    };
    for (;;)
    {
        if (!m_accepting.load(std::memory_order_acquire))
        {
            recordBlockedDuration();
            throwSubmissionUnavailable();
        }
        const uint64_t spaceEpoch = m_spaceEpoch.load(
            std::memory_order_acquire);
        const AdmissionResult admission = tryEnqueue(item);
        if (admission == AdmissionResult::Enqueued)
            break;
        if (!countedBlock)
        {
            countedBlock = true;
            blockedAt = std::chrono::steady_clock::now();
            m_producerBlockCount.fetch_add(
                1u, std::memory_order_relaxed);
        }
        if (!m_accepting.load(std::memory_order_acquire))
        {
            recordBlockedDuration();
            throwSubmissionUnavailable();
        }
        const auto waitAt = std::chrono::steady_clock::now();
        m_spaceEpoch.wait(
            spaceEpoch, std::memory_order_acquire);
        const uint64_t waitNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - waitAt)
                    .count());
        if (admission == AdmissionResult::SlotCapacity)
        {
            m_producerSlotWaitCount.fetch_add(
                1u, std::memory_order_relaxed);
            m_producerSlotWaitNanoseconds.fetch_add(
                waitNanoseconds, std::memory_order_relaxed);
        }
        else
        {
            m_producerPayloadWaitCount.fetch_add(
                1u, std::memory_order_relaxed);
            m_producerPayloadWaitNanoseconds.fetch_add(
                waitNanoseconds, std::memory_order_relaxed);
        }
    }
    recordBlockedDuration();

    m_nextSequence =
        sequence == std::numeric_limits<uint64_t>::max()
            ? 0u
            : sequence + 1u;
    m_nextTicket = ticket;
    m_lastGuestTick = guestTick;
    m_activeExecutionEpoch = state;
    m_submittedTickets.store(
        ticket, std::memory_order_release);
    m_submittedGeneration.store(
        generation, std::memory_order_release);
    m_submittedSequence.store(
        sequence, std::memory_order_release);
    m_submittedCommandsByType[
        vu1CommandTypeIndex(Vu1CommandType::AdvanceSlice)]
        .fetch_add(1u, std::memory_order_relaxed);
    m_executionEpochsSubmitted.fetch_add(
        1u, std::memory_order_relaxed);
    signalWorkAvailable();
    submitLock.unlock();

    return Vu1ExecutionEpoch(
        ticket, identity, std::move(state));
}

size_t ThreadedVu1Executor::executionEpochReadyCheckpointCount(
    const Vu1ExecutionEpoch &epoch) const
{
    if (!epoch.m_state)
        return 0u;
    std::lock_guard<std::mutex> lock(epoch.m_state->mutex);
    size_t ready = 0u;
    for (const auto &checkpoint : epoch.m_state->journal)
    {
        if (!checkpoint->consumed &&
            checkpoint->revision ==
                epoch.m_state->demandRevision)
        {
            ++ready;
        }
    }
    return ready;
}

bool ThreadedVu1Executor::executionEpochCheckpointReady(
    const Vu1ExecutionEpoch &epoch,
    uint32_t exactCycles) const
{
    if (!epoch.m_state || exactCycles == 0u)
        return false;
    std::lock_guard<std::mutex> lock(epoch.m_state->mutex);
    for (const auto &checkpoint : epoch.m_state->journal)
    {
        if (checkpoint->consumed)
            continue;
        return checkpoint->revision ==
                   epoch.m_state->demandRevision &&
               checkpoint->cycles == exactCycles;
    }
    return false;
}

Vu1CommandResult
ThreadedVu1Executor::waitExecutionEpochCheckpoint(
    Vu1ExecutionEpoch &epoch,
    uint32_t exactCycles,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!epoch.m_state || exactCycles == 0u)
    {
        throw std::invalid_argument(
            "invalid threaded VU1 execution epoch checkpoint");
    }
    const auto waitAt = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(epoch.m_state->mutex);
    const auto firstReady = [&]()
        -> std::shared_ptr<Vu1ExecutionEpochState::Checkpoint>
    {
        for (const auto &checkpoint : epoch.m_state->journal)
        {
            if (!checkpoint->consumed)
                return checkpoint;
        }
        return {};
    };

    std::shared_ptr<Vu1ExecutionEpochState::Checkpoint> checkpoint =
        firstReady();
    const bool needsRebase = checkpoint
        ? checkpoint->cycles != exactCycles ||
              checkpoint->revision !=
                  epoch.m_state->demandRevision
        : epoch.m_state->demandedCycles != exactCycles;
    if (needsRebase)
    {
        if (epoch.m_state->demandRevision ==
            std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error(
                "VU1 execution epoch demand revision exhausted");
        }
        epoch.m_state->demandedCycles = exactCycles;
        ++epoch.m_state->demandRevision;
        m_epochDemandRebases.fetch_add(
            1u, std::memory_order_relaxed);
        epoch.m_state->changed.notify_all();
    }

    epoch.m_state->changed.wait(
        lock,
        [&]()
        {
            if (epoch.m_state->failure ||
                epoch.m_state->stopped)
            {
                return true;
            }
            const auto ready = firstReady();
            return ready &&
                   ready->revision ==
                       epoch.m_state->demandRevision &&
                   ready->cycles == exactCycles;
        });
    if (epoch.m_state->failure)
        std::rethrow_exception(epoch.m_state->failure);
    checkpoint = firstReady();
    if (!checkpoint || checkpoint->cycles != exactCycles ||
        checkpoint->revision != epoch.m_state->demandRevision)
    {
        throw std::runtime_error(
            "threaded VU1 execution epoch stopped before checkpoint");
    }

    Vu1CommandResult result = std::move(checkpoint->result);
    checkpoint->consumed = true;
    epoch.m_state->demandedCycles =
        epoch.m_state->options.followupCycles;
    m_epochCheckpointsPublished.fetch_add(
        1u, std::memory_order_relaxed);
    lock.unlock();
    epoch.m_state->changed.notify_all();

    result.identity = epoch.m_identity;
    result.identity.guestTick = guestTick;
    result.identity.publicationToken = publicationToken;
    result.digest.sequence = epoch.m_identity.sequence;
    result.digest.generation = epoch.m_identity.generation;
    result.digest.guestTick = guestTick;
    const uint64_t waitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitAt)
                .count());
    m_epochCheckpointWaitCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_epochCheckpointWaitNanoseconds.fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    if (result.disposition != Vu1CommandDisposition::Completed)
    {
        throw std::logic_error(
            "threaded VU1 epoch checkpoint was rejected");
    }
    return result;
}

void ThreadedVu1Executor::stopExecutionEpoch(
    Vu1ExecutionEpoch &epoch) noexcept
{
    if (!epoch.m_state)
        return;
    std::lock_guard<std::mutex> submitLock(m_submitMutex);
    {
        std::lock_guard<std::mutex> lock(epoch.m_state->mutex);
        epoch.m_state->stopRequested = true;
    }
    epoch.m_state->changed.notify_all();
    if (m_activeExecutionEpoch == epoch.m_state)
        m_activeExecutionEpoch.reset();
}

Vu1CommandSubmission
ThreadedVu1Executor::submitSpeculativeAdvance(
    Vu1AdvanceSliceCommand command,
    Vu1SpeculationResolution previousResolution,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return submitAsyncImpl(
        Vu1CommandPayload{command},
        previousResolution, true, false, 0u, true,
        guestTick, publicationToken);
}

Vu1CommandResult
ThreadedVu1Executor::submitResolvingSpeculation(
    Vu1CommandPayload payload,
    Vu1SpeculationResolution resolution,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return wait(submitAsyncImpl(
        std::move(payload), resolution, false,
        false, 0u, true,
        guestTick, publicationToken));
}

Vu1CommandAdmission
ThreadedVu1Executor::submitResultlessResolvingSpeculation(
    Vu1CommandPayload payload,
    Vu1SpeculationResolution resolution,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    if (!supportsVu1ResultlessAdmission(payload))
    {
        throw std::invalid_argument(
            "response-producing VU1 command requires a completion");
    }
    Vu1CommandSubmission submission = submitAsyncImpl(
        std::move(payload), resolution, false,
        false, 0u, false,
        guestTick, publicationToken);
    return {
        .ticket = submission.ticket(),
        .identity = submission.identity(),
        .type = submission.type(),
    };
}

Vu1CommandResult
ThreadedVu1Executor::commitExtendedSpeculativeAdvance(
    Vu1AdvanceSliceCommand publishedCommand,
    uint32_t extensionCycles,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return wait(submitAsyncImpl(
        Vu1CommandPayload{publishedCommand},
        Vu1SpeculationResolution::None,
        false, true, extensionCycles, true,
        guestTick, publicationToken));
}

Vu1CommandResult ThreadedVu1Executor::wait(
    Vu1CommandSubmission submission)
{
    const size_t commandType =
        vu1CommandTypeIndex(submission.type());
    const auto waitAt = std::chrono::steady_clock::now();
    Vu1CommandResult result = submission.wait();
    const uint64_t waitNanoseconds =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitAt)
                .count());
    m_resultWaitCount.fetch_add(
        1u, std::memory_order_relaxed);
    m_resultWaitNanoseconds.fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    m_resultWaitCountByType[commandType].fetch_add(
        1u, std::memory_order_relaxed);
    m_resultWaitNanosecondsByType[commandType].fetch_add(
        waitNanoseconds, std::memory_order_relaxed);
    if (result.disposition != Vu1CommandDisposition::Completed)
    {
        throw std::logic_error(
            std::string("threaded VU1 command rejected: ") +
            vu1CommandTypeName(result.digest.type));
    }
    return result;
}

Vu1CommandResult ThreadedVu1Executor::submit(
    Vu1CommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken)
{
    return wait(submitAsync(
        std::move(payload), guestTick, publicationToken));
}

uint64_t ThreadedVu1Executor::generation() const
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    return m_generation;
}

uint64_t ThreadedVu1Executor::lastSubmittedSequence() const
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    return m_nextSequence == 0u
        ? std::numeric_limits<uint64_t>::max()
        : m_nextSequence - 1u;
}

uint64_t ThreadedVu1Executor::lastSubmittedTicket() const noexcept
{
    return m_submittedTickets.load(
        std::memory_order_acquire);
}

uint64_t ThreadedVu1Executor::lastCompletedTicket() const noexcept
{
    return m_completedTickets.load(
        std::memory_order_acquire);
}

void ThreadedVu1Executor::cancelPendingBeforeGeneration(
    uint64_t generation)
{
    if (generation == 0u)
    {
        throw std::invalid_argument(
            "VU1 cancellation generation must be non-zero");
    }
    {
        std::lock_guard<std::mutex> lock(m_submitMutex);
        if (generation > m_generation)
        {
            throw std::invalid_argument(
                "VU1 cancellation generation has not been submitted");
        }
    }

    updateVu1AtomicMaximum(
        m_minimumAcceptedGeneration,
        generation);
}

void ThreadedVu1Executor::releasePayload(
    uint64_t bytes) noexcept
{
    const uint64_t prior =
        m_queuedPayloadBytes.fetch_sub(
            bytes, std::memory_order_acq_rel);
    if (prior < bytes)
        std::terminate();
    signalSpaceAvailable();
}

void ThreadedVu1Executor::recordFatalFailure(
    std::exception_ptr failure) noexcept
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (!m_fatalFailure)
            m_fatalFailure = std::move(failure);
    }
    closeAdmissionAndQuiesceProducer();
    m_cancelRequested.store(true, std::memory_order_release);
    signalWorkAvailable();
}

void ThreadedVu1Executor::cancelQueued(
    const std::exception_ptr &reason) noexcept
{
    while (std::optional<WorkItem> item = m_queue.tryPop())
    {
        try
        {
            if (item->executionEpoch)
                failExecutionEpoch(item->executionEpoch, reason);
            if (item->completion)
                item->completion->set_exception(reason);
        }
        catch (...)
        {
        }
        releasePayload(item->payloadBytes);
        m_completedTickets.store(
            item->ticket, std::memory_order_release);
    }
}

void ThreadedVu1Executor::failExecutionEpoch(
    const std::shared_ptr<Vu1ExecutionEpochState> &state,
    const std::exception_ptr &failure) noexcept
{
    if (!state)
        return;
    try
    {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->failure)
                state->failure = failure;
            state->stopped = true;
        }
        state->changed.notify_all();
    }
    catch (...)
    {
    }
}

void ThreadedVu1Executor::runExecutionEpoch(WorkItem &item)
{
    if (!item.executionEpoch)
    {
        throw std::logic_error(
            "threaded VU1 execution epoch has no state");
    }
    const auto *const initialAdvance =
        std::get_if<Vu1AdvanceSliceCommand>(
            &item.command.payload);
    if (!initialAdvance)
    {
        throw std::logic_error(
            "threaded VU1 execution epoch is not an advance");
    }
    if (item.command.identity.sequence ==
        std::numeric_limits<uint64_t>::max())
    {
        throw std::overflow_error(
            "VU1 execution epoch sequence exhausted");
    }

    struct LocalCheckpoint
    {
        std::shared_ptr<
            Vu1ExecutionEpochState::Checkpoint>
            shared;
        Vu1CommandProcessor::SpeculativeSliceCheckpoint
            processor;
    };

    auto baseline = m_processor.captureProcessorCheckpoint();
    // The transport identity is consumed even if no speculative checkpoint
    // becomes guest-visible. Ordered work behind the epoch therefore keeps a
    // simple monotonic sequence after rollback to the pre-epoch architecture.
    baseline.nextSequence = item.command.identity.sequence + 1u;
    std::deque<LocalCheckpoint> localJournal;
    uint64_t handledRevision = 0u;
    bool initialExecution = true;
    bool reachedInactive = false;
    {
        std::lock_guard<std::mutex> lock(
            item.executionEpoch->mutex);
        item.executionEpoch->workerStarted = true;
        handledRevision =
            item.executionEpoch->demandRevision;
    }
    item.executionEpoch->changed.notify_all();

    try
    {
        for (;;)
        {
            enum class Action : uint8_t
            {
                Produce,
                Rebase,
                Stop,
                Finish,
            };
            Action action = Action::Produce;
            uint32_t cycles = 0u;
            uint64_t revision = handledRevision;
            uint64_t discarded = 0u;
            {
                std::unique_lock<std::mutex> lock(
                    item.executionEpoch->mutex);
                for (;;)
                {
                    while (!localJournal.empty() &&
                           localJournal.front().shared->consumed)
                    {
                        const bool publishedActive =
                            localJournal.front().shared->active;
                        baseline = std::move(
                            localJournal.front().processor);
                        if (item.executionEpoch->journal.empty() ||
                            item.executionEpoch->journal.front() !=
                                localJournal.front().shared)
                        {
                            throw std::logic_error(
                                "VU1 epoch journal order diverged");
                        }
                        item.executionEpoch->journal.pop_front();
                        localJournal.pop_front();
                        if (!publishedActive)
                        {
                            discarded = localJournal.size();
                            item.executionEpoch->journal.clear();
                            localJournal.clear();
                            action = Action::Finish;
                            break;
                        }
                    }
                    if (action == Action::Finish)
                        break;
                    if (item.executionEpoch->stopRequested)
                    {
                        discarded = localJournal.size();
                        item.executionEpoch->journal.clear();
                        localJournal.clear();
                        action = Action::Stop;
                        break;
                    }
                    if (item.executionEpoch->demandRevision !=
                        handledRevision)
                    {
                        discarded = localJournal.size();
                        item.executionEpoch->journal.clear();
                        localJournal.clear();
                        handledRevision =
                            item.executionEpoch->demandRevision;
                        revision = handledRevision;
                        reachedInactive = false;
                        action = Action::Rebase;
                        break;
                    }
                    if (!reachedInactive &&
                        localJournal.size() <
                            item.executionEpoch->options
                                .checkpointCapacity)
                    {
                        cycles = localJournal.empty()
                            ? item.executionEpoch->demandedCycles
                            : item.executionEpoch->options
                                  .followupCycles;
                        revision = handledRevision;
                        action = Action::Produce;
                        break;
                    }

                    item.executionEpoch->changed.wait(lock);
                }
            }
            item.executionEpoch->changed.notify_all();

            if (discarded != 0u)
            {
                m_epochSuffixCheckpointsDiscarded.fetch_add(
                    discarded, std::memory_order_relaxed);
            }
            if (action == Action::Stop ||
                action == Action::Finish)
            {
                m_processor.restoreProcessorCheckpoint(baseline);
                {
                    std::lock_guard<std::mutex> lock(
                        item.executionEpoch->mutex);
                    item.executionEpoch->stopped = true;
                }
                item.executionEpoch->changed.notify_all();
                return;
            }
            if (action == Action::Rebase)
            {
                m_processor.restoreProcessorCheckpoint(baseline);
                initialExecution = false;
                continue;
            }
            if (cycles == 0u)
            {
                throw std::logic_error(
                    "VU1 epoch requested a zero-cycle checkpoint");
            }

            Vu1AdvanceSliceCommand command{
                .maximumCycles = cycles,
                .captureState =
                    item.executionEpoch->options.captureState,
            };
            Vu1CommandResult result = initialExecution
                ? m_processor.processAdvanceSlice(
                      item.command.identity, command, false)
                : m_processor.processEpochContinuation(
                      item.command.identity, command);
            initialExecution = false;
            if (result.disposition !=
                Vu1CommandDisposition::Completed)
            {
                throw std::logic_error(
                    "VU1 owner rejected execution epoch checkpoint");
            }
            auto processorCheckpoint =
                m_processor.captureProcessorCheckpoint();
            auto sharedCheckpoint =
                std::make_shared<
                    Vu1ExecutionEpochState::Checkpoint>();
            sharedCheckpoint->revision = revision;
            sharedCheckpoint->cycles = cycles;
            sharedCheckpoint->active = result.active;
            sharedCheckpoint->result = std::move(result);
            if (m_options.beforePublish)
            {
                m_options.beforePublish(
                    sharedCheckpoint->result,
                    item.ticket);
            }

            {
                std::lock_guard<std::mutex> lock(
                    item.executionEpoch->mutex);
                item.executionEpoch->journal.push_back(
                    sharedCheckpoint);
                localJournal.push_back(
                    LocalCheckpoint{
                        .shared = sharedCheckpoint,
                        .processor =
                            std::move(processorCheckpoint),
                    });
                reachedInactive = !sharedCheckpoint->active;
                updateVu1AtomicMaximum(
                    m_epochMaximumJournalDepth,
                    localJournal.size());
            }
            m_epochCheckpointsProduced.fetch_add(
                1u, std::memory_order_relaxed);
            item.executionEpoch->changed.notify_all();
        }
    }
    catch (...)
    {
        const std::exception_ptr failure =
            std::current_exception();
        try
        {
            m_processor.restoreProcessorCheckpoint(baseline);
        }
        catch (...)
        {
        }
        failExecutionEpoch(item.executionEpoch, failure);
        std::rethrow_exception(failure);
    }
}

void ThreadedVu1Executor::workerMain() noexcept
{
    ThreadNaming::SetCurrentThreadName("PS2Vu1Owner");
    try
    {
        m_processor.claimOwnerThread(
            std::this_thread::get_id());
        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            m_ownerThreadId = std::this_thread::get_id();
        }
    }
    catch (...)
    {
        recordFatalFailure(std::current_exception());
    }

    bool drainingReadyRun = false;
    for (;;)
    {
        if (!drainingReadyRun)
        {
            const bool lifecycleRequested =
                m_cancelRequested.load(
                    std::memory_order_acquire) ||
                m_drainRequested.load(
                    std::memory_order_acquire);
            if (m_queue.empty() && !lifecycleRequested &&
                !tryAcquireWorkSignal())
            {
                const auto idleAt =
                    std::chrono::steady_clock::now();
                acquireWorkSignal();
                const uint64_t idleNanoseconds =
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - idleAt)
                            .count());
                m_workerIdleNanoseconds.fetch_add(
                    idleNanoseconds, std::memory_order_relaxed);
                m_workerWakeCount.fetch_add(
                    1u, std::memory_order_relaxed);
            }
            else
            {
                // Queue state and lifecycle requests are authoritative even
                // when an earlier ready run consumed their coalesced signal.
                (void)tryAcquireWorkSignal();
            }
            m_workerDrainCount.fetch_add(
                1u, std::memory_order_relaxed);
        }
        else
        {
            // Commands admitted while the owner was active leave one latched
            // notification. Clear it while continuing the same ready run;
            // the queue itself is the authoritative work source.
            (void)tryAcquireWorkSignal();
        }
        drainingReadyRun = false;

        if (m_cancelRequested.load(std::memory_order_acquire))
        {
            std::exception_ptr reason;
            {
                std::lock_guard<std::mutex> lock(m_stateMutex);
                reason = m_fatalFailure;
            }
            if (!reason)
                reason = makeVu1ExecutorCancellation();
            cancelQueued(reason);
            break;
        }

        std::optional<WorkItem> item = m_queue.tryPop();
        if (!item)
        {
            if (m_drainRequested.load(std::memory_order_acquire))
                break;
            continue;
        }

        signalSpaceAvailable();
        const Vu1CommandType commandType =
            item->command.type;
        const auto activeAt =
            std::chrono::steady_clock::now();
        try
        {
            if (m_options.beforeProcess)
            {
                m_options.beforeProcess(
                    item->command, item->ticket);
            }

            if (m_cancelRequested.load(std::memory_order_acquire))
            {
                const std::exception_ptr reason =
                    makeVu1ExecutorCancellation();
                if (item->executionEpoch)
                    failExecutionEpoch(
                        item->executionEpoch, reason);
                if (item->completion)
                    item->completion->set_exception(reason);
                releasePayload(item->payloadBytes);
                m_completedTickets.store(
                    item->ticket, std::memory_order_release);
                cancelQueued(reason);
                break;
            }

            Vu1CommandResult result{};
            const bool stale =
                item->command.identity.generation <
                m_minimumAcceptedGeneration.load(
                    std::memory_order_acquire);
            const bool executionEpoch =
                static_cast<bool>(item->executionEpoch);
            if (executionEpoch)
            {
                m_processor.resolveSpeculativeSlice(
                    item->speculationResolution);
                if (stale)
                {
                    throw std::logic_error(
                        "threaded VU1 execution epoch became stale");
                }
                runExecutionEpoch(*item);
                result = Vu1CommandResult{
                    .identity = item->command.identity,
                    .digest = vu1CommandDigest(item->command),
                    .ownerGeneration = m_processor.generation(),
                    .codeGeneration = m_processor.codeGeneration(),
                };
                m_executionEpochsCompleted.fetch_add(
                    1u, std::memory_order_relaxed);
            }
            else if (item->commitsExtendedSpeculativeAdvance)
            {
                if (stale)
                {
                    m_processor.resolveSpeculativeSlice(
                        Vu1SpeculationResolution::Rollback);
                    result = makeStaleQueuedVu1Result(
                        item->command,
                        m_processor.generation(),
                        m_processor.codeGeneration());
                }
                else
                {
                    const auto *const advance =
                        std::get_if<Vu1AdvanceSliceCommand>(
                            &item->command.payload);
                    if (!advance)
                    {
                        throw std::logic_error(
                            "threaded VU1 speculative extension is not an advance");
                    }
                    result =
                        m_processor.commitExtendedSpeculativeAdvance(
                            item->command.identity,
                            *advance,
                            item->speculativeExtensionCycles);
                }
            }
            else
            {
                m_processor.resolveSpeculativeSlice(
                    item->speculationResolution);
                if (stale)
                {
                    result = makeStaleQueuedVu1Result(
                        item->command,
                        m_processor.generation(),
                        m_processor.codeGeneration());
                }
                else if (item->beginsSpeculativeSlice)
                {
                    const auto *const advance =
                        std::get_if<Vu1AdvanceSliceCommand>(
                            &item->command.payload);
                    if (!advance)
                    {
                        throw std::logic_error(
                            "threaded VU1 speculative work is not an advance");
                    }
                    result = m_processor.processAdvanceSlice(
                        item->command.identity,
                        *advance, true);
                }
                else
                {
                    result = m_processor.process(
                        std::move(item->command));
                }
            }
            if (!stale &&
                result.disposition !=
                    Vu1CommandDisposition::Completed)
            {
                throw std::logic_error(
                    std::string(
                        "VU1 owner rejected queued command: ") +
                    vu1CommandTypeName(result.digest.type));
            }
            if (!stale)
            {
                m_executedDecodedUnpackOperations.fetch_add(
                    item->decodedUnpackOperations,
                    std::memory_order_relaxed);
            }

            const uint64_t completedGeneration =
                result.identity.generation;
            const uint64_t completedSequence =
                result.identity.sequence;
            if (!executionEpoch && m_options.beforePublish)
            {
                m_options.beforePublish(result, item->ticket);
            }
            m_completedTickets.store(
                item->ticket, std::memory_order_release);
            m_completedGeneration.store(
                completedGeneration, std::memory_order_release);
            m_completedSequence.store(
                completedSequence, std::memory_order_release);
            m_completedCommandsByType[
                vu1CommandTypeIndex(commandType)]
                .fetch_add(1u, std::memory_order_relaxed);
            if (item->completion)
                item->completion->set_value(std::move(result));
            releasePayload(item->payloadBytes);
        }
        catch (...)
        {
            const std::exception_ptr failure =
                std::current_exception();
            try
            {
                if (item->executionEpoch)
                {
                    failExecutionEpoch(
                        item->executionEpoch, failure);
                }
                if (item->completion)
                    item->completion->set_exception(failure);
            }
            catch (...)
            {
            }
            m_completedTickets.store(
                item->ticket, std::memory_order_release);
            recordFatalFailure(failure);
            releasePayload(item->payloadBytes);
            cancelQueued(failure);
            const uint64_t activeNanoseconds =
                static_cast<uint64_t>(
                    std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - activeAt)
                        .count());
            m_workerActiveNanoseconds.fetch_add(
                activeNanoseconds, std::memory_order_relaxed);
            break;
        }

        const uint64_t activeNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - activeAt)
                    .count());
        m_workerActiveNanoseconds.fetch_add(
            activeNanoseconds, std::memory_order_relaxed);
        drainingReadyRun = !m_queue.empty();
    }

    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_workerExited = true;
    }
    signalSpaceAvailable();
}

void ThreadedVu1Executor::shutdown(
    Vu1ExecutorShutdownMode mode) noexcept
{
    std::lock_guard<std::mutex> shutdownLock(m_shutdownMutex);
    closeAdmissionAndQuiesceProducer();
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (mode == Vu1ExecutorShutdownMode::Cancel)
        {
            m_cancelRequested.store(
                true, std::memory_order_release);
        }
        else
        {
            m_drainRequested.store(
                true, std::memory_order_release);
        }
        if (!m_started)
            m_workerExited = true;
    }
    signalWorkAvailable();

    if (m_worker.joinable())
    {
        if (m_worker.get_id() == std::this_thread::get_id())
            std::terminate();
        m_worker.join();
    }
}

void ThreadedVu1Executor::rethrowFailure() const
{
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        failure = m_fatalFailure;
    }
    if (failure)
        std::rethrow_exception(failure);
}

ThreadedVu1ExecutorStatistics
ThreadedVu1Executor::statistics() const
{
    ThreadedVu1ExecutorStatistics result{
        .queueCapacity = m_queue.capacity(),
        .payloadCapacityBytes = m_options.payloadCapacityBytes,
        .queueDepth = m_queue.size(),
        .queueHighWater = m_queueHighWater.load(
            std::memory_order_acquire),
        .queuedPayloadBytes = m_queuedPayloadBytes.load(
            std::memory_order_acquire),
        .payloadHighWaterBytes = m_payloadHighWaterBytes.load(
            std::memory_order_acquire),
        .submittedTickets = m_submittedTickets.load(
            std::memory_order_acquire),
        .completedTickets = m_completedTickets.load(
            std::memory_order_acquire),
        .submittedGeneration = m_submittedGeneration.load(
            std::memory_order_acquire),
        .submittedSequence = m_submittedSequence.load(
            std::memory_order_acquire),
        .completedGeneration = m_completedGeneration.load(
            std::memory_order_acquire),
        .completedSequence = m_completedSequence.load(
            std::memory_order_acquire),
        .producerBlockCount = m_producerBlockCount.load(
            std::memory_order_acquire),
        .producerBlockedNanoseconds =
            m_producerBlockedNanoseconds.load(
                std::memory_order_acquire),
        .producerSlotWaitCount = m_producerSlotWaitCount.load(
            std::memory_order_acquire),
        .producerSlotWaitNanoseconds =
            m_producerSlotWaitNanoseconds.load(
                std::memory_order_acquire),
        .producerPayloadWaitCount =
            m_producerPayloadWaitCount.load(
                std::memory_order_acquire),
        .producerPayloadWaitNanoseconds =
            m_producerPayloadWaitNanoseconds.load(
                std::memory_order_acquire),
        .workerActiveNanoseconds =
            m_workerActiveNanoseconds.load(
                std::memory_order_acquire),
        .workerIdleNanoseconds =
            m_workerIdleNanoseconds.load(
                std::memory_order_acquire),
        .workNotificationCount =
            m_workNotificationCount.load(
                std::memory_order_acquire),
        .workerWakeCount = m_workerWakeCount.load(
            std::memory_order_acquire),
        .workerDrainCount = m_workerDrainCount.load(
            std::memory_order_acquire),
        .resultWaitCount = m_resultWaitCount.load(
            std::memory_order_acquire),
        .resultWaitNanoseconds =
            m_resultWaitNanoseconds.load(
                std::memory_order_acquire),
        .submittedDecodedUnpackOperations =
            m_submittedDecodedUnpackOperations.load(
                std::memory_order_acquire),
        .executedDecodedUnpackOperations =
            m_executedDecodedUnpackOperations.load(
                std::memory_order_acquire),
        .speculation = m_processor.speculationStatistics(),
        .executionEpoch = {
            .epochsSubmitted =
                m_executionEpochsSubmitted.load(
                    std::memory_order_acquire),
            .epochsCompleted =
                m_executionEpochsCompleted.load(
                    std::memory_order_acquire),
            .checkpointsProduced =
                m_epochCheckpointsProduced.load(
                    std::memory_order_acquire),
            .checkpointsPublished =
                m_epochCheckpointsPublished.load(
                    std::memory_order_acquire),
            .suffixCheckpointsDiscarded =
                m_epochSuffixCheckpointsDiscarded.load(
                    std::memory_order_acquire),
            .demandRebases =
                m_epochDemandRebases.load(
                    std::memory_order_acquire),
            .checkpointWaitCount =
                m_epochCheckpointWaitCount.load(
                    std::memory_order_acquire),
            .checkpointWaitNanoseconds =
                m_epochCheckpointWaitNanoseconds.load(
                    std::memory_order_acquire),
            .maximumJournalDepth =
                m_epochMaximumJournalDepth.load(
                    std::memory_order_acquire),
        },
        .accepting = m_accepting.load(
            std::memory_order_acquire),
        .drainRequested = m_drainRequested.load(
            std::memory_order_acquire),
        .cancelRequested = m_cancelRequested.load(
            std::memory_order_acquire),
    };
    for (size_t index = 0u;
         index < kVu1CommandTypeCount; ++index)
    {
        result.commandTypes[index] =
            ThreadedVu1CommandTypeStatistics{
                .submitted =
                    m_submittedCommandsByType[index].load(
                        std::memory_order_acquire),
                .completed =
                    m_completedCommandsByType[index].load(
                        std::memory_order_acquire),
                .resultWaitCount =
                    m_resultWaitCountByType[index].load(
                        std::memory_order_acquire),
                .resultWaitNanoseconds =
                    m_resultWaitNanosecondsByType[index].load(
                        std::memory_order_acquire),
            };
    }
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        result.started = m_started;
        result.running = m_started && !m_workerExited;
        result.failed = static_cast<bool>(m_fatalFailure);
    }
    return result;
}

std::thread::id ThreadedVu1Executor::ownerThreadId() const
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_ownerThreadId;
}
