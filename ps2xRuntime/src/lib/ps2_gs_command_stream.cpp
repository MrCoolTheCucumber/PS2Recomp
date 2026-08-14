#include "runtime/ps2_gs_command_stream.h"

#include <algorithm>
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
    void hashScalar(uint64_t &hash, const T &value) noexcept
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
            const Unsigned encoded =
                static_cast<Unsigned>(value);
            for (size_t index = 0u;
                 index < sizeof(encoded); ++index)
            {
                const uint8_t byte = static_cast<uint8_t>(
                    encoded >> (index * 8u));
                hashBytes(hash, &byte, sizeof(byte));
            }
        }
        else
        {
            static_assert(std::is_integral_v<Value>);
            using Unsigned = std::make_unsigned_t<Value>;
            const Unsigned encoded =
                static_cast<Unsigned>(value);
            for (size_t index = 0u;
                 index < sizeof(encoded); ++index)
            {
                const uint8_t byte = static_cast<uint8_t>(
                    encoded >> (index * 8u));
                hashBytes(hash, &byte, sizeof(byte));
            }
        }
    }

    void hashString(uint64_t &hash, const std::string &value) noexcept
    {
        const uint64_t size = value.size();
        hashScalar(hash, size);
        hashBytes(hash, value.data(), value.size());
    }

    uint64_t hashReplayState(const GsReplayState &state)
    {
        std::vector<uint8_t> encoded;
        std::string error;
        if (!encodeGsReplayState(state, encoded, &error))
        {
            throw std::invalid_argument(
                "invalid GS replay state in command: " + error);
        }
        uint64_t hash = kFnvOffsetBasis;
        hashBytes(hash, encoded.data(), encoded.size());
        return hash;
    }

    uint64_t checkedIncrement(
        uint64_t value, const char *identityName)
    {
        if (value == std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error(
                std::string("GS command ") +
                identityName + " exhausted");
        }
        return value + 1u;
    }

    bool isHotCommand(const GsCommandPayload &payload) noexcept
    {
        return
            std::holds_alternative<GsDrainBatchCommand>(payload) ||
            std::holds_alternative<GsWriteRegisterCommand>(payload) ||
            std::holds_alternative<GsNativePackedGifCommand>(payload) ||
            std::holds_alternative<GsNativeImageUploadCommand>(payload) ||
            std::holds_alternative<GsClearFramebufferCommand>(payload);
    }

    bool isValidDrainBatch(
        const GifArbiterDrainBatch &batch) noexcept
    {
        for (const GifArbiterDrainBatch::Packet &packet :
             batch.packets)
        {
            const size_t pathIndex =
                static_cast<size_t>(packet.pathId);
            if (pathIndex <
                    static_cast<size_t>(GifPathId::Path1) ||
                pathIndex >
                    static_cast<size_t>(GifPathId::Path3) ||
                packet.storageIndex != pathIndex ||
                packet.size == 0u ||
                packet.size >
                    std::numeric_limits<uint32_t>::max() ||
                packet.offset >
                    batch.storage[pathIndex].size() ||
                packet.size >
                    batch.storage[pathIndex].size() -
                        packet.offset ||
                (packet.path2DirectHl &&
                 packet.pathId != GifPathId::Path2) ||
                (packet.path3Image &&
                 packet.pathId != GifPathId::Path3))
            {
                return false;
            }
        }
        return true;
    }

    bool isValidCommandPayload(
        const GsCommandPayload &payload)
    {
        return std::visit(
            [](const auto &value)
            {
                using T = std::decay_t<decltype(value)>;
                if constexpr (
                    std::is_same_v<T, GsDrainBatchCommand>)
                {
                    return isValidDrainBatch(value.batch);
                }
                else if constexpr (
                    std::is_same_v<T, GsNativePackedGifCommand> ||
                    std::is_same_v<T, GsNativeImageUploadCommand>)
                {
                    return value.bytes.size() <=
                           std::numeric_limits<uint32_t>::max();
                }
                else if constexpr (
                    std::is_same_v<T, GsRestoreReplayStateCommand>)
                {
                    std::vector<uint8_t> encoded;
                    return encodeGsReplayState(
                        value.state, encoded);
                }
                else
                {
                    return true;
                }
            },
            payload);
    }
}

const char *gsCommandTypeName(GsCommandType type) noexcept
{
    switch (type)
    {
    case GsCommandType::DrainBatch:
        return "drain-batch";
    case GsCommandType::BeginRenderBatch:
        return "begin-render-batch";
    case GsCommandType::FlushRenderBatch:
        return "flush-render-batch";
    case GsCommandType::EndRenderBatch:
        return "end-render-batch";
    case GsCommandType::WriteRegister:
        return "write-register";
    case GsCommandType::NativePackedGif:
        return "native-packed-gif";
    case GsCommandType::NativeImageUpload:
        return "native-image-upload";
    case GsCommandType::Reset:
        return "reset";
    case GsCommandType::ClearFramebuffer:
        return "clear-framebuffer";
    case GsCommandType::ReadFifo:
        return "read-fifo";
    case GsCommandType::CaptureReplayState:
        return "capture-replay-state";
    case GsCommandType::RestoreReplayState:
        return "restore-replay-state";
    case GsCommandType::DebugSnapshot:
        return "debug-snapshot";
    case GsCommandType::ProgressSnapshot:
        return "progress-snapshot";
    case GsCommandType::DebugHistory:
        return "debug-history";
    case GsCommandType::RecentGifPackets:
        return "recent-gif-packets";
    case GsCommandType::ClearDebugHistory:
        return "clear-debug-history";
    case GsCommandType::DebugHistoryPaused:
        return "debug-history-paused";
    case GsCommandType::SetDebugHistoryPaused:
        return "set-debug-history-paused";
    case GsCommandType::SetProgressTracking:
        return "set-progress-tracking";
    case GsCommandType::LatchPresentation:
        return "latch-presentation";
    case GsCommandType::CopyPresentation:
        return "copy-presentation";
    case GsCommandType::ConfigureVulkan:
        return "configure-vulkan";
    case GsCommandType::SetRendererMode:
        return "set-renderer-mode";
    case GsCommandType::RendererStatus:
        return "renderer-status";
    case GsCommandType::SetBackendCountersEnabled:
        return "set-backend-counters-enabled";
    case GsCommandType::ResetBackendCounters:
        return "reset-backend-counters";
    case GsCommandType::BackendCounters:
        return "backend-counters";
    case GsCommandType::SetDrawCommandLimit:
        return "set-draw-command-limit";
    case GsCommandType::ClearDrawCommandLimit:
        return "clear-draw-command-limit";
    case GsCommandType::DrawStatus:
        return "draw-status";
    case GsCommandType::Barrier:
        return "barrier";
    case GsCommandType::CopyVram:
        return "copy-vram";
    }
    return "unknown";
}

GsCommandType gsCommandType(
    const GsCommandPayload &payload) noexcept
{
    return std::visit(
        [](const auto &value) noexcept
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, GsDrainBatchCommand>)
                return GsCommandType::DrainBatch;
            else if constexpr (std::is_same_v<T, GsBeginRenderBatchCommand>)
                return GsCommandType::BeginRenderBatch;
            else if constexpr (std::is_same_v<T, GsFlushRenderBatchCommand>)
                return GsCommandType::FlushRenderBatch;
            else if constexpr (std::is_same_v<T, GsEndRenderBatchCommand>)
                return GsCommandType::EndRenderBatch;
            else if constexpr (std::is_same_v<T, GsWriteRegisterCommand>)
                return GsCommandType::WriteRegister;
            else if constexpr (std::is_same_v<T, GsNativePackedGifCommand>)
                return GsCommandType::NativePackedGif;
            else if constexpr (std::is_same_v<T, GsNativeImageUploadCommand>)
                return GsCommandType::NativeImageUpload;
            else if constexpr (std::is_same_v<T, GsResetCommand>)
                return GsCommandType::Reset;
            else if constexpr (std::is_same_v<T, GsClearFramebufferCommand>)
                return GsCommandType::ClearFramebuffer;
            else if constexpr (std::is_same_v<T, GsReadFifoCommand>)
                return GsCommandType::ReadFifo;
            else if constexpr (std::is_same_v<T, GsCaptureReplayStateCommand>)
                return GsCommandType::CaptureReplayState;
            else if constexpr (std::is_same_v<T, GsCopyVramCommand>)
                return GsCommandType::CopyVram;
            else if constexpr (std::is_same_v<T, GsRestoreReplayStateCommand>)
                return GsCommandType::RestoreReplayState;
            else if constexpr (std::is_same_v<T, GsDebugSnapshotCommand>)
                return GsCommandType::DebugSnapshot;
            else if constexpr (std::is_same_v<T, GsProgressSnapshotCommand>)
                return GsCommandType::ProgressSnapshot;
            else if constexpr (std::is_same_v<T, GsDebugHistoryCommand>)
                return GsCommandType::DebugHistory;
            else if constexpr (std::is_same_v<T, GsRecentGifPacketsCommand>)
                return GsCommandType::RecentGifPackets;
            else if constexpr (std::is_same_v<T, GsClearDebugHistoryCommand>)
                return GsCommandType::ClearDebugHistory;
            else if constexpr (std::is_same_v<T, GsDebugHistoryPausedCommand>)
                return GsCommandType::DebugHistoryPaused;
            else if constexpr (std::is_same_v<T, GsSetDebugHistoryPausedCommand>)
                return GsCommandType::SetDebugHistoryPaused;
            else if constexpr (std::is_same_v<T, GsSetProgressTrackingCommand>)
                return GsCommandType::SetProgressTracking;
            else if constexpr (std::is_same_v<T, GsLatchPresentationCommand>)
                return GsCommandType::LatchPresentation;
            else if constexpr (std::is_same_v<T, GsCopyPresentationCommand>)
                return GsCommandType::CopyPresentation;
            else if constexpr (std::is_same_v<T, GsConfigureVulkanCommand>)
                return GsCommandType::ConfigureVulkan;
            else if constexpr (std::is_same_v<T, GsSetRendererModeCommand>)
                return GsCommandType::SetRendererMode;
            else if constexpr (std::is_same_v<T, GsRendererStatusCommand>)
                return GsCommandType::RendererStatus;
            else if constexpr (std::is_same_v<T, GsSetBackendCountersEnabledCommand>)
                return GsCommandType::SetBackendCountersEnabled;
            else if constexpr (std::is_same_v<T, GsResetBackendCountersCommand>)
                return GsCommandType::ResetBackendCounters;
            else if constexpr (std::is_same_v<T, GsBackendCountersCommand>)
                return GsCommandType::BackendCounters;
            else if constexpr (std::is_same_v<T, GsSetDrawCommandLimitCommand>)
                return GsCommandType::SetDrawCommandLimit;
            else if constexpr (std::is_same_v<T, GsClearDrawCommandLimitCommand>)
                return GsCommandType::ClearDrawCommandLimit;
            else if constexpr (std::is_same_v<T, GsDrawStatusCommand>)
                return GsCommandType::DrawStatus;
            else
                return GsCommandType::Barrier;
        },
        payload);
}

uint64_t gsCommandPayloadSize(
    const GsCommandPayload &payload) noexcept
{
    return std::visit(
        [](const auto &value) noexcept -> uint64_t
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, GsDrainBatchCommand>)
            {
                uint64_t size = 8u + 1u + 1u;
                for (const GifArbiterDrainBatch::Packet &packet :
                     value.batch.packets)
                {
                    constexpr uint64_t metadataSize =
                        1u + 1u + 1u + 8u;
                    if (metadataSize >
                        std::numeric_limits<uint64_t>::max() - size)
                    {
                        return std::numeric_limits<uint64_t>::max();
                    }
                    size += metadataSize;
                    if (packet.size >
                        std::numeric_limits<uint64_t>::max() - size)
                    {
                        return std::numeric_limits<uint64_t>::max();
                    }
                    size += static_cast<uint64_t>(packet.size);
                }
                return size;
            }
            else if constexpr (std::is_same_v<T, GsWriteRegisterCommand>)
                return sizeof(value.address) + sizeof(value.value);
            else if constexpr (std::is_same_v<T, GsNativePackedGifCommand>)
                return 8u + value.bytes.size();
            else if constexpr (std::is_same_v<T, GsNativeImageUploadCommand>)
                return sizeof(value.bitbltbuf) + sizeof(value.trxpos) +
                       sizeof(value.trxreg) + sizeof(value.trxdir) +
                       8u + value.bytes.size();
            else if constexpr (std::is_same_v<T, GsClearFramebufferCommand>)
                return sizeof(value.contextIndex) + sizeof(value.rgba) +
                       sizeof(value.activeContext);
            else if constexpr (std::is_same_v<T, GsReadFifoCommand>)
                return sizeof(value.maximumBytes);
            else if constexpr (std::is_same_v<T, GsRestoreReplayStateCommand>)
            {
                std::vector<uint8_t> encoded;
                return encodeGsReplayState(value.state, encoded)
                    ? encoded.size()
                    : 0u;
            }
            else if constexpr (std::is_same_v<T, GsRecentGifPacketsCommand>)
                return sizeof(uint64_t);
            else if constexpr (
                std::is_same_v<T, GsSetDebugHistoryPausedCommand> ||
                std::is_same_v<T, GsSetProgressTrackingCommand> ||
                std::is_same_v<T, GsSetBackendCountersEnabledCommand>)
                return sizeof(bool);
            else if constexpr (std::is_same_v<T, GsConfigureVulkanCommand>)
                return 1u +
                       8u + value.service.probe.loaderPath.size() +
                       sizeof(uint32_t) + sizeof(uint32_t) +
                       sizeof(uint64_t) + sizeof(uint8_t) +
                       8u + value.backend.verificationArtifactDirectory.size() +
                       sizeof(uint64_t) + 10u * sizeof(uint64_t);
            else if constexpr (std::is_same_v<T, GsSetRendererModeCommand>)
                return sizeof(uint8_t);
            else if constexpr (std::is_same_v<T, GsSetDrawCommandLimitCommand>)
                return sizeof(value.maximumCommands);
            else
                return 0u;
        },
        payload);
}

uint64_t gsCommandPayloadHash(
    const GsCommandPayload &payload)
{
    uint64_t hash = kFnvOffsetBasis;
    const GsCommandType type = gsCommandType(payload);
    hashScalar(hash, type);
    std::visit(
        [&](const auto &value)
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, GsDrainBatchCommand>)
            {
                const uint64_t count = value.batch.packets.size();
                hashScalar(hash, count);
                hashScalar(hash, value.beginSubmissionBatch);
                hashScalar(hash, value.endSubmissionBatch);
                for (const GifArbiterDrainBatch::Packet &packet :
                     value.batch.packets)
                {
                    hashScalar(hash, packet.pathId);
                    hashScalar(hash, packet.path2DirectHl);
                    hashScalar(hash, packet.path3Image);
                    const std::span<const uint8_t> bytes =
                        value.batch.packetBytes(packet);
                    const uint64_t size = bytes.size();
                    hashScalar(hash, size);
                    hashBytes(hash, bytes.data(), bytes.size());
                }
            }
            else if constexpr (std::is_same_v<T, GsWriteRegisterCommand>)
            {
                hashScalar(hash, value.address);
                hashScalar(hash, value.value);
            }
            else if constexpr (std::is_same_v<T, GsNativePackedGifCommand>)
            {
                const uint64_t size = value.bytes.size();
                hashScalar(hash, size);
                hashBytes(hash, value.bytes.data(), value.bytes.size());
            }
            else if constexpr (std::is_same_v<T, GsNativeImageUploadCommand>)
            {
                hashScalar(hash, value.bitbltbuf);
                hashScalar(hash, value.trxpos);
                hashScalar(hash, value.trxreg);
                hashScalar(hash, value.trxdir);
                const uint64_t size = value.bytes.size();
                hashScalar(hash, size);
                hashBytes(hash, value.bytes.data(), value.bytes.size());
            }
            else if constexpr (std::is_same_v<T, GsClearFramebufferCommand>)
            {
                hashScalar(hash, value.contextIndex);
                hashScalar(hash, value.rgba);
                hashScalar(hash, value.activeContext);
            }
            else if constexpr (std::is_same_v<T, GsReadFifoCommand>)
                hashScalar(hash, value.maximumBytes);
            else if constexpr (std::is_same_v<T, GsRestoreReplayStateCommand>)
            {
                const uint64_t stateHash = hashReplayState(value.state);
                hashScalar(hash, stateHash);
            }
            else if constexpr (std::is_same_v<T, GsRecentGifPacketsCommand>)
                hashScalar(
                    hash,
                    static_cast<uint64_t>(value.limit));
            else if constexpr (std::is_same_v<T, GsSetDebugHistoryPausedCommand>)
                hashScalar(hash, value.paused);
            else if constexpr (std::is_same_v<T, GsSetProgressTrackingCommand>)
                hashScalar(hash, value.enabled);
            else if constexpr (std::is_same_v<T, GsConfigureVulkanCommand>)
            {
                hashScalar(hash, value.service.probe.enableValidation);
                hashString(hash, value.service.probe.loaderPath);
                hashScalar(hash, value.service.probe.preferredVendorId);
                hashScalar(hash, value.service.probe.preferredDeviceId);
                hashScalar(hash, value.service.fenceTimeoutNanoseconds);
                hashScalar(hash, value.backend.mode);
                hashString(hash, value.backend.verificationArtifactDirectory);
                hashScalar(
                    hash,
                    static_cast<uint64_t>(
                        value.backend.maximumResidentBatchCommands));
                hashScalar(hash, value.backend.minimumHybridSpritePixels);
                hashScalar(hash, value.backend.minimumHybridSourceCopyAlphaSpritePixels);
                hashScalar(hash, value.backend.minimumHybridDepthCt32SpritePixels);
                hashScalar(hash, value.backend.minimumHybridFramebufferOnlyAlphaFailDepthCt32RunPixels);
                hashScalar(hash, value.backend.minimumHybridNearestCt32SpritePixels);
                hashScalar(hash, value.backend.minimumHybridLinearCt32SpritePixels);
                hashScalar(hash, value.backend.minimumHybridLinearCt32ClampSpritePixels);
                hashScalar(hash, value.backend.minimumHybridFeedbackLinearDepthCt32RunPixels);
                hashScalar(hash, value.backend.minimumHybridTriangleCandidatePixels);
                hashScalar(hash, value.backend.minimumHybridGouraudDepthCt32TriangleRunPixels);
            }
            else if constexpr (std::is_same_v<T, GsSetRendererModeCommand>)
                hashScalar(hash, value.mode);
            else if constexpr (std::is_same_v<T, GsSetBackendCountersEnabledCommand>)
                hashScalar(hash, value.enabled);
            else if constexpr (std::is_same_v<T, GsSetDrawCommandLimitCommand>)
                hashScalar(hash, value.maximumCommands);
        },
        payload);
    return hash;
}

GsCommandDigest makeGsCommandDigest(
    const GsCommand &command)
{
    return {
        .sequence = command.identity.sequence,
        .generation = command.identity.generation,
        .guestTick = command.identity.guestTick,
        .type = command.type,
        .payloadSize = command.payloadSize,
        .payloadHash = gsCommandPayloadHash(command.payload),
    };
}

uint64_t gsCommandDigestHash(
    const GsCommandDigest &digest) noexcept
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

GsCommandProcessor::GsCommandProcessor(
    GS &gs, bool commandDigestsEnabled)
    : m_gs(gs),
      m_commandDigestsEnabled(commandDigestsEnabled)
{
    m_gs.setVSyncTickProvider(
        [this]()
        {
            return m_currentDebugVsyncTick;
        });
}

GsCommandProcessor::~GsCommandProcessor()
{
    m_gs.setVSyncTickProvider({});
}

GsCommandResult GsCommandProcessor::process(
    GsCommand command)
{
    GsCommandResult result{};
    result.identity = command.identity;
    result.completedSequence = m_completedSequence;
    result.digest = {
        .sequence = command.identity.sequence,
        .generation = command.identity.generation,
        .guestTick = command.identity.guestTick,
        .type = command.type,
        .payloadSize = command.payloadSize,
        .payloadHash = 0u,
    };

    const GsCommandType actualType =
        gsCommandType(command.payload);
    const uint64_t actualSize =
        gsCommandPayloadSize(command.payload);
    if (command.type != actualType ||
        command.payloadSize != actualSize ||
        command.identity.sequence == 0u ||
        command.identity.generation == 0u ||
        !isValidCommandPayload(command.payload))
    {
        result.disposition =
            GsCommandDisposition::Malformed;
        return result;
    }
    if (m_commandDigestsEnabled)
    {
        result.digest.payloadHash =
            gsCommandPayloadHash(command.payload);
    }

    if (command.identity.generation < m_generation)
    {
        result.disposition =
            GsCommandDisposition::StaleGeneration;
        return result;
    }
    const bool startsGeneration =
        std::holds_alternative<GsResetCommand>(command.payload) ||
        std::holds_alternative<GsRestoreReplayStateCommand>(command.payload);
    if (startsGeneration &&
        command.identity.generation == m_generation)
    {
        result.disposition =
            GsCommandDisposition::FutureGeneration;
        return result;
    }
    const bool beginsNewGeneration =
        command.identity.generation > m_generation;
    if (beginsNewGeneration)
    {
        if (!startsGeneration ||
            m_generation ==
                std::numeric_limits<uint64_t>::max() ||
            command.identity.generation !=
                m_generation + 1u)
        {
            result.disposition =
                GsCommandDisposition::FutureGeneration;
            return result;
        }
        if (command.identity.sequence != 1u)
        {
            result.disposition =
                GsCommandDisposition::OutOfOrder;
            return result;
        }
        m_generation = command.identity.generation;
        m_completedSequence = 0u;
    }

    if (!beginsNewGeneration &&
        (m_completedSequence ==
            std::numeric_limits<uint64_t>::max() ||
        command.identity.sequence !=
            m_completedSequence + 1u))
    {
        result.disposition =
            GsCommandDisposition::OutOfOrder;
        return result;
    }

    m_currentDebugVsyncTick = command.debugVsyncTick;
    result.payload = std::visit(
        [&](auto &value) -> GsCommandResultPayload
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, GsDrainBatchCommand>)
            {
                if (value.beginSubmissionBatch)
                    m_gs.beginRenderBatch();
                struct BatchScope
                {
                    GS &gs;
                    bool active = false;
                    ~BatchScope()
                    {
                        if (active)
                            gs.endRenderSubmissionBatch();
                    }
                } batch{
                    .gs = m_gs,
                    .active = value.endSubmissionBatch,
                };
                for (const GifArbiterDrainBatch::Packet &packet :
                     value.batch.packets)
                {
                    const std::span<const uint8_t> bytes =
                        value.batch.packetBytes(packet);
                    if (bytes.empty())
                        continue;
                    m_gs.processGIFPacket(
                        bytes.data(),
                        static_cast<uint32_t>(bytes.size()));
                }
                if (value.endSubmissionBatch)
                {
                    m_gs.endRenderSubmissionBatch();
                    batch.active = false;
                }
                for (std::vector<uint8_t> &storage :
                     value.batch.storage)
                {
                    storage.clear();
                }
                value.batch.packets.clear();
                return GsDrainBatchResult{
                    .batch = std::move(value.batch),
                };
            }
            else if constexpr (std::is_same_v<T, GsBeginRenderBatchCommand>)
            {
                m_gs.beginRenderBatch();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsFlushRenderBatchCommand>)
            {
                m_gs.flushRenderBatch();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsEndRenderBatchCommand>)
            {
                m_gs.endRenderBatch();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsWriteRegisterCommand>)
            {
                m_gs.writeRegister(value.address, value.value);
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsNativePackedGifCommand>)
            {
                return GsBooleanResult{
                    m_gs.processNativePackedGIFPacket(
                        value.bytes.data(),
                        static_cast<uint32_t>(value.bytes.size()))};
            }
            else if constexpr (std::is_same_v<T, GsNativeImageUploadCommand>)
            {
                m_gs.uploadImageNative(
                    value.bitbltbuf, value.trxpos,
                    value.trxreg, value.trxdir,
                    value.bytes.data(),
                    static_cast<uint32_t>(value.bytes.size()));
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsResetCommand>)
            {
                m_gs.reset();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsClearFramebufferCommand>)
            {
                const bool cleared = value.activeContext
                    ? m_gs.clearActiveFramebuffer(value.rgba)
                    : m_gs.clearFramebufferContext(
                          value.contextIndex, value.rgba);
                return GsBooleanResult{cleared};
            }
            else if constexpr (std::is_same_v<T, GsReadFifoCommand>)
            {
                GsFifoReadResult fifo{};
                fifo.bytes.resize(value.maximumBytes);
                const uint32_t read = m_gs.consumeLocalToHostBytes(
                    fifo.bytes.data(), value.maximumBytes);
                fifo.bytes.resize(read);
                return fifo;
            }
            else if constexpr (std::is_same_v<T, GsCaptureReplayStateCommand>)
                return GsReplayStateResult{m_gs.captureReplayState()};
            else if constexpr (std::is_same_v<T, GsCopyVramCommand>)
            {
                GsVramSnapshotResult snapshot{};
                snapshot.available = m_gs.copyVram(snapshot.bytes);
                return snapshot;
            }
            else if constexpr (std::is_same_v<T, GsRestoreReplayStateCommand>)
                return GsBooleanResult{m_gs.restoreReplayState(value.state)};
            else if constexpr (std::is_same_v<T, GsDebugSnapshotCommand>)
                return GsDebugSnapshotResult{m_gs.getDebugSnapshot()};
            else if constexpr (std::is_same_v<T, GsProgressSnapshotCommand>)
                return GsProgressSnapshotResult{m_gs.getProgressSnapshot()};
            else if constexpr (std::is_same_v<T, GsDebugHistoryCommand>)
                return GsDebugHistoryResult{m_gs.getDebugHistory()};
            else if constexpr (std::is_same_v<T, GsRecentGifPacketsCommand>)
            {
                GsRecentGifPacketsResult packets{};
                m_gs.copyRecentGifPackets(
                    value.limit,
                    packets.stream,
                    packets.sizes,
                    packets.initialVram,
                    &packets.initialState);
                return packets;
            }
            else if constexpr (std::is_same_v<T, GsClearDebugHistoryCommand>)
            {
                m_gs.clearDebugHistory();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsDebugHistoryPausedCommand>)
                return GsBooleanResult{m_gs.isDebugHistoryPaused()};
            else if constexpr (std::is_same_v<T, GsSetDebugHistoryPausedCommand>)
            {
                m_gs.setDebugHistoryPaused(value.paused);
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsSetProgressTrackingCommand>)
            {
                m_gs.setProgressTrackingEnabled(value.enabled);
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsLatchPresentationCommand>)
            {
                m_gs.latchHostPresentationFrame();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsCopyPresentationCommand>)
            {
                GsPresentationResult frame{};
                frame.available =
                    m_gs.copyLatchedHostPresentationFrame(
                        frame.pixels, frame.width, frame.height,
                        &frame.displayFbp, &frame.sourceFbp,
                        &frame.usedPreferred);
                return frame;
            }
            else if constexpr (std::is_same_v<T, GsConfigureVulkanCommand>)
            {
                return GsBooleanResult{
                    m_gs.configureVulkanRenderer(
                        value.service, std::move(value.backend))};
            }
            else if constexpr (std::is_same_v<T, GsSetRendererModeCommand>)
                return GsBooleanResult{m_gs.setRendererMode(value.mode)};
            else if constexpr (std::is_same_v<T, GsRendererStatusCommand>)
            {
                return GsRendererStatusResult{
                    .mode = m_gs.rendererMode(),
                    .diagnostic = m_gs.rendererDiagnostic(),
                    .capabilities = m_gs.vulkanRendererCapabilities(),
                    .serviceStatistics =
                        m_gs.vulkanRendererServiceStatistics(),
                    .backendStatistics =
                        m_gs.vulkanRendererBackendStatistics(),
                };
            }
            else if constexpr (std::is_same_v<T, GsSetBackendCountersEnabledCommand>)
            {
                m_gs.setBackendCountersEnabled(value.enabled);
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsResetBackendCountersCommand>)
            {
                m_gs.resetBackendCounters();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsBackendCountersCommand>)
                return GsBackendCountersResult{m_gs.backendCounters()};
            else if constexpr (std::is_same_v<T, GsSetDrawCommandLimitCommand>)
            {
                m_gs.setDrawCommandLimit(value.maximumCommands);
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsClearDrawCommandLimitCommand>)
            {
                m_gs.clearDrawCommandLimit();
                return GsNoResult{};
            }
            else if constexpr (std::is_same_v<T, GsDrawStatusCommand>)
            {
                return GsDrawStatusResult{
                    .limitReached = m_gs.drawCommandLimitReached(),
                    .submittedCommands =
                        m_gs.submittedDrawCommandCount(),
                };
            }
            else
            {
                m_gs.flushRenderBatch();
                return GsNoResult{};
            }
        },
        command.payload);

    m_completedSequence = command.identity.sequence;
    result.completedSequence = m_completedSequence;
    result.disposition = GsCommandDisposition::Completed;
    return result;
}

InlineGsExecutor::InlineGsExecutor(
    GsCommandProcessor &processor)
    : m_processor(processor),
      m_generation(processor.generation()),
      m_nextSequence(processor.completedSequence())
{
}

uint64_t InlineGsExecutor::generation() const
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    return m_generation;
}

uint64_t InlineGsExecutor::lastSubmittedSequence() const
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    return m_nextSequence;
}

bool InlineGsExecutor::startsGeneration(
    const GsCommandPayload &payload) noexcept
{
    return std::holds_alternative<GsResetCommand>(payload) ||
           std::holds_alternative<GsRestoreReplayStateCommand>(payload);
}

void InlineGsExecutor::assertHotProducerThread(
    const GsCommandPayload &payload)
{
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (!isHotCommand(payload))
        return;
    const std::thread::id current =
        std::this_thread::get_id();
    if (!m_hotProducerThread.has_value())
    {
        m_hotProducerThread = current;
        return;
    }
    if (*m_hotProducerThread != current)
    {
        throw std::logic_error(
            "GS hot command submitted from multiple producer threads");
    }
#else
    (void)payload;
#endif
}

GsCommandResult InlineGsExecutor::submit(
    GsCommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken,
    uint64_t debugVsyncTick)
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    assertHotProducerThread(payload);
    guestTick = std::max(guestTick, m_lastGuestTick);

    if (startsGeneration(payload))
    {
        m_generation = checkedIncrement(
            m_generation, "generation");
        m_nextSequence = 0u;
    }
    const uint64_t sequence = checkedIncrement(
        m_nextSequence, "sequence");
    GsCommand command{
        .identity = {
            .sequence = sequence,
            .generation = m_generation,
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        },
        .type = gsCommandType(payload),
        .payloadSize = gsCommandPayloadSize(payload),
        .debugVsyncTick = debugVsyncTick,
        .payload = std::move(payload),
    };
    GsCommandResult result =
        m_processor.process(std::move(command));
    if (!result.completed())
    {
        throw std::logic_error(
            std::string("inline GS command rejected: ") +
            gsCommandTypeName(result.digest.type));
    }
    m_nextSequence = sequence;
    m_lastGuestTick = guestTick;
    return result;
}

GsCommandSubmission::GsCommandSubmission(
    uint64_t ticket,
    GsWorkIdentity identity,
    std::future<GsCommandResult> completion)
    : m_ticket(ticket),
      m_identity(identity),
      m_completion(std::move(completion))
{
}

bool GsCommandSubmission::valid() const noexcept
{
    return m_completion.valid();
}

bool GsCommandSubmission::ready() const
{
    if (!m_completion.valid())
        return false;
    return m_completion.wait_for(
               std::chrono::seconds(0)) ==
           std::future_status::ready;
}

GsCommandResult GsCommandSubmission::wait()
{
    if (!m_completion.valid())
    {
        throw std::logic_error(
            "GS command submission has no completion");
    }
    return m_completion.get();
}

namespace
{
    std::exception_ptr makeGsExecutorCancellation()
    {
        return std::make_exception_ptr(
            std::runtime_error(
                "threaded GS executor cancelled queued work"));
    }

    GsCommandResult makeStaleQueuedGsResult(
        GsCommand &command,
        uint64_t completedSequence)
    {
        GsCommandResult result{};
        result.identity = command.identity;
        result.disposition =
            GsCommandDisposition::StaleGeneration;
        result.completedSequence = completedSequence;
        result.digest = {
            .sequence = command.identity.sequence,
            .generation = command.identity.generation,
            .guestTick = command.identity.guestTick,
            .type = command.type,
            .payloadSize = command.payloadSize,
            .payloadHash = 0u,
        };
        if (auto *batch =
                std::get_if<GsDrainBatchCommand>(
                    &command.payload))
        {
            result.payload = GsDrainBatchResult{
                .batch = std::move(batch->batch),
            };
        }
        return result;
    }

    template <typename T>
    void updateAtomicMaximum(
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
}

ThreadedGsExecutor::ThreadedGsExecutor(
    GsCommandProcessor &processor,
    ThreadedGsExecutorOptions options)
    : m_processor(processor),
      m_options(std::move(options)),
      m_queue(m_options.queueCapacity),
      m_generation(processor.generation()),
      m_nextSequence(processor.completedSequence()),
      m_minimumAcceptedGeneration(processor.generation())
{
    if (m_options.payloadCapacityBytes == 0u)
    {
        throw std::invalid_argument(
            "threaded GS payload capacity must be non-zero");
    }
}

ThreadedGsExecutor::~ThreadedGsExecutor()
{
    shutdown(GsExecutorShutdownMode::Drain);
}

bool ThreadedGsExecutor::startsGeneration(
    const GsCommandPayload &payload) noexcept
{
    return std::holds_alternative<GsResetCommand>(payload) ||
           std::holds_alternative<GsRestoreReplayStateCommand>(payload);
}

void ThreadedGsExecutor::assertHotProducerThread(
    const GsCommandPayload &payload)
{
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    if (!isHotCommand(payload))
        return;
    const std::thread::id current =
        std::this_thread::get_id();
    if (!m_hotProducerThread.has_value())
    {
        m_hotProducerThread = current;
        return;
    }
    if (*m_hotProducerThread != current)
    {
        throw std::logic_error(
            "GS hot command submitted from multiple producer threads");
    }
#else
    (void)payload;
#endif
}

void ThreadedGsExecutor::ensureStarted()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    if (m_started)
        return;
    if (!m_accepting.load(std::memory_order_acquire))
    {
        throw std::runtime_error(
            "threaded GS executor is not accepting work");
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

void ThreadedGsExecutor::updateQueueHighWater(
    size_t depth) noexcept
{
    updateAtomicMaximum(m_queueHighWater, depth);
}

void ThreadedGsExecutor::updatePayloadHighWater(
    uint64_t bytes) noexcept
{
    updateAtomicMaximum(m_payloadHighWaterBytes, bytes);
}

bool ThreadedGsExecutor::tryEnqueue(WorkItem &item)
{
    if (m_queue.full())
        return false;

    const uint64_t currentPayload =
        m_queuedPayloadBytes.load(std::memory_order_acquire);
    if (item.payloadBytes >
        m_options.payloadCapacityBytes -
            std::min(
                currentPayload,
                m_options.payloadCapacityBytes))
    {
        return false;
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
        return false;
    }

    updateQueueHighWater(reservedDepth);
    updatePayloadHighWater(reserved);
    return true;
}

[[noreturn]] void
ThreadedGsExecutor::throwSubmissionUnavailable() const
{
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        failure = m_fatalFailure;
    }
    if (failure)
        std::rethrow_exception(failure);
    throw std::runtime_error(
        "threaded GS executor is not accepting work");
}

GsCommandSubmission ThreadedGsExecutor::submitAsync(
    GsCommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken,
    uint64_t debugVsyncTick)
{
    std::unique_lock<std::mutex> submitLock(
        m_submitMutex);
    if (!m_accepting.load(std::memory_order_acquire))
        throwSubmissionUnavailable();

    assertHotProducerThread(payload);
    ensureStarted();
    guestTick = std::max(guestTick, m_lastGuestTick);

    uint64_t generation = m_generation;
    uint64_t sequenceBase = m_nextSequence;
    if (startsGeneration(payload))
    {
        generation = checkedIncrement(
            generation, "generation");
        sequenceBase = 0u;
    }
    const uint64_t sequence = checkedIncrement(
        sequenceBase, "sequence");
    const uint64_t ticket = checkedIncrement(
        m_nextTicket, "transport ticket");
    const uint64_t payloadBytes =
        gsCommandPayloadSize(payload);
    if (payloadBytes > m_options.payloadCapacityBytes)
    {
        throw std::length_error(
            "GS command exceeds the bounded payload capacity");
    }

    GsCommand command{
        .identity = {
            .sequence = sequence,
            .generation = generation,
            .guestTick = guestTick,
            .publicationToken = publicationToken,
        },
        .type = gsCommandType(payload),
        .payloadSize = payloadBytes,
        .debugVsyncTick = debugVsyncTick,
        .payload = std::move(payload),
    };
    const GsWorkIdentity identity = command.identity;
    WorkItem item{
        .ticket = ticket,
        .payloadBytes = payloadBytes,
        .command = std::move(command),
    };
    std::future<GsCommandResult> completion =
        item.completion.get_future();

    bool countedBlock = false;
    std::chrono::steady_clock::time_point blockedAt{};
    for (;;)
    {
        bool accepting = false;
        bool enqueued = false;
        {
            // M3 deliberately keeps a conservative admission lock. Closing
            // the executor takes the same lock, so no producer can publish a
            // command after fatal/cancel/drain has begun.
            std::lock_guard<std::mutex> stateLock(
                m_stateMutex);
            accepting = m_accepting.load(
                std::memory_order_acquire);
            if (accepting && !m_fatalFailure)
                enqueued = tryEnqueue(item);
        }
        if (enqueued)
            break;
        if (!accepting)
            throwSubmissionUnavailable();
        if (!countedBlock)
        {
            countedBlock = true;
            blockedAt = std::chrono::steady_clock::now();
            m_producerBlockCount.fetch_add(
                1u, std::memory_order_relaxed);
        }

        std::unique_lock<std::mutex> stateLock(
            m_stateMutex);
        m_spaceCv.wait(
            stateLock,
            [this, payloadBytes]()
            {
                if (!m_accepting.load(
                        std::memory_order_acquire) ||
                    m_fatalFailure)
                {
                    return true;
                }
                const uint64_t queued =
                    m_queuedPayloadBytes.load(
                        std::memory_order_acquire);
                return !m_queue.full() &&
                       queued <=
                           m_options.payloadCapacityBytes &&
                       payloadBytes <=
                           m_options.payloadCapacityBytes - queued;
            });
    }
    if (countedBlock)
    {
        const uint64_t nanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - blockedAt)
                    .count());
        m_producerBlockedNanoseconds.fetch_add(
            nanoseconds, std::memory_order_relaxed);
    }

    m_generation = generation;
    m_nextSequence = sequence;
    m_nextTicket = ticket;
    m_lastGuestTick = guestTick;
    m_submittedTickets.store(
        ticket, std::memory_order_release);
    submitLock.unlock();
    m_workCv.notify_one();

    return GsCommandSubmission(
        ticket, identity, std::move(completion));
}

GsCommandResult ThreadedGsExecutor::submit(
    GsCommandPayload payload,
    uint64_t guestTick,
    uint64_t publicationToken,
    uint64_t debugVsyncTick)
{
    GsCommandSubmission submission = submitAsync(
        std::move(payload), guestTick,
        publicationToken, debugVsyncTick);
    GsCommandResult result = submission.wait();
    if (!result.completed())
    {
        throw std::logic_error(
            std::string("threaded GS command rejected: ") +
            gsCommandTypeName(result.digest.type));
    }
    return result;
}

uint64_t ThreadedGsExecutor::generation() const
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    return m_generation;
}

uint64_t ThreadedGsExecutor::lastSubmittedSequence() const
{
    std::lock_guard<std::mutex> lock(m_submitMutex);
    return m_nextSequence;
}

uint64_t ThreadedGsExecutor::lastSubmittedTicket() const noexcept
{
    return m_submittedTickets.load(
        std::memory_order_acquire);
}

uint64_t ThreadedGsExecutor::lastCompletedTicket() const noexcept
{
    return m_completedTickets.load(
        std::memory_order_acquire);
}

void ThreadedGsExecutor::cancelPendingBeforeGeneration(
    uint64_t generation)
{
    if (generation == 0u)
    {
        throw std::invalid_argument(
            "GS cancellation generation must be non-zero");
    }
    {
        std::lock_guard<std::mutex> lock(m_submitMutex);
        if (generation > m_generation)
        {
            throw std::invalid_argument(
                "GS cancellation generation has not been submitted");
        }
    }

    updateAtomicMaximum(
        m_minimumAcceptedGeneration,
        generation);
    m_workCv.notify_one();
}

void ThreadedGsExecutor::releasePayload(
    uint64_t bytes) noexcept
{
    const uint64_t prior =
        m_queuedPayloadBytes.fetch_sub(
            bytes, std::memory_order_acq_rel);
    if (prior < bytes)
        std::terminate();
    m_spaceCv.notify_all();
}

void ThreadedGsExecutor::recordFatalFailure(
    std::exception_ptr failure) noexcept
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (!m_fatalFailure)
            m_fatalFailure = std::move(failure);
        m_accepting.store(false, std::memory_order_release);
    }
    m_cancelRequested.store(true, std::memory_order_release);
    m_workCv.notify_all();
    m_spaceCv.notify_all();
}

void ThreadedGsExecutor::cancelQueued(
    const std::exception_ptr &reason) noexcept
{
    while (std::optional<WorkItem> item =
               m_queue.tryPop())
    {
        try
        {
            item->completion.set_exception(reason);
        }
        catch (...)
        {
        }
        releasePayload(item->payloadBytes);
        m_completedTickets.store(
            item->ticket, std::memory_order_release);
    }
    m_spaceCv.notify_all();
}

void ThreadedGsExecutor::workerMain() noexcept
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_ownerThreadId = std::this_thread::get_id();
    }

    for (;;)
    {
        std::optional<WorkItem> item =
            m_queue.tryPop();
        if (!item)
        {
            const auto idleAt =
                std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(
                m_stateMutex);
            m_workCv.wait(
                lock,
                [this]()
                {
                    return !m_queue.empty() ||
                           m_cancelRequested.load(
                               std::memory_order_acquire) ||
                           m_drainRequested.load(
                               std::memory_order_acquire) ||
                           m_fatalFailure;
                });
            const uint64_t idleNanoseconds =
                static_cast<uint64_t>(
                    std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - idleAt)
                        .count());
            m_workerIdleNanoseconds.fetch_add(
                idleNanoseconds, std::memory_order_relaxed);

            if (m_cancelRequested.load(
                    std::memory_order_acquire) ||
                m_fatalFailure)
            {
                const std::exception_ptr reason =
                    m_fatalFailure
                        ? m_fatalFailure
                        : makeGsExecutorCancellation();
                lock.unlock();
                cancelQueued(reason);
                break;
            }
            if (m_drainRequested.load(
                    std::memory_order_acquire) &&
                m_queue.empty())
            {
                break;
            }
            continue;
        }

        m_spaceCv.notify_all();
        const auto activeAt =
            std::chrono::steady_clock::now();
        try
        {
            if (m_options.beforeProcess)
            {
                m_options.beforeProcess(
                    item->command, item->ticket);
            }

            if (m_cancelRequested.load(
                    std::memory_order_acquire))
            {
                const std::exception_ptr reason =
                    makeGsExecutorCancellation();
                item->completion.set_exception(reason);
                releasePayload(item->payloadBytes);
                m_completedTickets.store(
                    item->ticket, std::memory_order_release);
                cancelQueued(reason);
                break;
            }

            GsCommandResult result{};
            if (item->command.identity.generation <
                m_minimumAcceptedGeneration.load(
                    std::memory_order_acquire))
            {
                result = makeStaleQueuedGsResult(
                    item->command,
                    m_processor.completedSequence());
            }
            else
            {
                result = m_processor.process(
                    std::move(item->command));
                if (!result.completed())
                {
                    throw std::logic_error(
                        std::string(
                            "GS owner rejected queued command: ") +
                        gsCommandTypeName(result.digest.type));
                }
            }

            const bool barrier =
                result.digest.type ==
                GsCommandType::Barrier;
            item->completion.set_value(std::move(result));
            releasePayload(item->payloadBytes);
            m_completedTickets.store(
                item->ticket, std::memory_order_release);
            if (barrier)
            {
                m_barriersCompleted.fetch_add(
                    1u, std::memory_order_relaxed);
            }
        }
        catch (...)
        {
            const std::exception_ptr failure =
                std::current_exception();
            try
            {
                item->completion.set_exception(failure);
            }
            catch (...)
            {
            }
            releasePayload(item->payloadBytes);
            m_completedTickets.store(
                item->ticket, std::memory_order_release);
            recordFatalFailure(failure);
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
    }

    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_workerExited = true;
    }
    m_spaceCv.notify_all();
    m_workCv.notify_all();
}

void ThreadedGsExecutor::shutdown(
    GsExecutorShutdownMode mode) noexcept
{
    std::lock_guard<std::mutex> shutdownLock(
        m_shutdownMutex);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_accepting.store(false, std::memory_order_release);
        if (mode == GsExecutorShutdownMode::Cancel)
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
    m_workCv.notify_all();
    m_spaceCv.notify_all();

    if (m_worker.joinable())
    {
        if (m_worker.get_id() ==
            std::this_thread::get_id())
        {
            std::terminate();
        }
        m_worker.join();
    }
}

void ThreadedGsExecutor::rethrowFailure() const
{
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        failure = m_fatalFailure;
    }
    if (failure)
        std::rethrow_exception(failure);
}

ThreadedGsExecutorStatistics
ThreadedGsExecutor::statistics() const
{
    ThreadedGsExecutorStatistics result{
        .queueCapacity = m_queue.capacity(),
        .payloadCapacityBytes =
            m_options.payloadCapacityBytes,
        .queueDepth = m_queue.size(),
        .queueHighWater = m_queueHighWater.load(
            std::memory_order_acquire),
        .queuedPayloadBytes =
            m_queuedPayloadBytes.load(
                std::memory_order_acquire),
        .payloadHighWaterBytes =
            m_payloadHighWaterBytes.load(
                std::memory_order_acquire),
        .submittedTickets =
            m_submittedTickets.load(
                std::memory_order_acquire),
        .completedTickets =
            m_completedTickets.load(
                std::memory_order_acquire),
        .producerBlockCount =
            m_producerBlockCount.load(
                std::memory_order_acquire),
        .producerBlockedNanoseconds =
            m_producerBlockedNanoseconds.load(
                std::memory_order_acquire),
        .workerActiveNanoseconds =
            m_workerActiveNanoseconds.load(
                std::memory_order_acquire),
        .workerIdleNanoseconds =
            m_workerIdleNanoseconds.load(
                std::memory_order_acquire),
        .barriersCompleted =
            m_barriersCompleted.load(
                std::memory_order_acquire),
        .accepting = m_accepting.load(
            std::memory_order_acquire),
    };
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        result.started = m_started;
        result.running =
            m_started && !m_workerExited;
        result.failed =
            static_cast<bool>(m_fatalFailure);
    }
    return result;
}

std::thread::id ThreadedGsExecutor::ownerThreadId() const
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_ownerThreadId;
}
