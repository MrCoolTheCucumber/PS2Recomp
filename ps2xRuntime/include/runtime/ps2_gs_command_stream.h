#ifndef PS2_GS_COMMAND_STREAM_H
#define PS2_GS_COMMAND_STREAM_H

#include "runtime/ps2_build_config.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_gs_gpu.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

struct GsWorkIdentity
{
    uint64_t sequence = 0u;
    uint64_t generation = 0u;
    uint64_t guestTick = 0u;
    uint64_t publicationToken = 0u;

    friend constexpr bool operator==(
        GsWorkIdentity, GsWorkIdentity) noexcept = default;
};

enum class GsCommandType : uint8_t
{
    DrainBatch,
    BeginRenderBatch,
    FlushRenderBatch,
    EndRenderBatch,
    WriteRegister,
    NativePackedGif,
    NativeImageUpload,
    Reset,
    ClearFramebuffer,
    ReadFifo,
    CaptureReplayState,
    RestoreReplayState,
    DebugSnapshot,
    ProgressSnapshot,
    DebugHistory,
    RecentGifPackets,
    ClearDebugHistory,
    DebugHistoryPaused,
    SetDebugHistoryPaused,
    SetProgressTracking,
    LatchPresentation,
    CopyPresentation,
    ConfigureVulkan,
    SetRendererMode,
    RendererStatus,
    SetBackendCountersEnabled,
    ResetBackendCounters,
    BackendCounters,
    SetDrawCommandLimit,
    ClearDrawCommandLimit,
    DrawStatus,
    Barrier,
};

[[nodiscard]] const char *gsCommandTypeName(
    GsCommandType type) noexcept;

struct GsDrainBatchCommand
{
    GifArbiterDrainBatch batch;
    bool beginSubmissionBatch = true;
    bool endSubmissionBatch = true;
};

struct GsBeginRenderBatchCommand
{
};

struct GsFlushRenderBatchCommand
{
};

struct GsEndRenderBatchCommand
{
};

struct GsWriteRegisterCommand
{
    uint8_t address = 0u;
    uint64_t value = 0u;
};

struct GsNativePackedGifCommand
{
    std::vector<uint8_t> bytes;
};

struct GsNativeImageUploadCommand
{
    uint64_t bitbltbuf = 0u;
    uint64_t trxpos = 0u;
    uint64_t trxreg = 0u;
    uint64_t trxdir = 0u;
    std::vector<uint8_t> bytes;
};

struct GsResetCommand
{
};

struct GsClearFramebufferCommand
{
    uint32_t contextIndex = 0u;
    uint32_t rgba = 0u;
    bool activeContext = false;
};

struct GsReadFifoCommand
{
    uint32_t maximumBytes = 0u;
};

struct GsCaptureReplayStateCommand
{
};

struct GsRestoreReplayStateCommand
{
    GsReplayState state{};
};

struct GsDebugSnapshotCommand
{
};

struct GsProgressSnapshotCommand
{
};

struct GsDebugHistoryCommand
{
};

struct GsRecentGifPacketsCommand
{
    size_t limit = 0u;
};

struct GsClearDebugHistoryCommand
{
};

struct GsDebugHistoryPausedCommand
{
};

struct GsSetDebugHistoryPausedCommand
{
    bool paused = true;
};

struct GsSetProgressTrackingCommand
{
    bool enabled = false;
};

struct GsLatchPresentationCommand
{
};

struct GsCopyPresentationCommand
{
};

struct GsConfigureVulkanCommand
{
    GsVulkanServiceConfig service{};
    GsVulkanRasterBackendConfig backend{};
};

struct GsSetRendererModeCommand
{
    GsRendererMode mode = GsRendererMode::Software;
};

struct GsRendererStatusCommand
{
};

struct GsSetBackendCountersEnabledCommand
{
    bool enabled = false;
};

struct GsResetBackendCountersCommand
{
};

struct GsBackendCountersCommand
{
};

struct GsSetDrawCommandLimitCommand
{
    uint64_t maximumCommands = 0u;
};

struct GsClearDrawCommandLimitCommand
{
};

struct GsDrawStatusCommand
{
};

struct GsBarrierCommand
{
};

using GsCommandPayload = std::variant<
    GsDrainBatchCommand,
    GsBeginRenderBatchCommand,
    GsFlushRenderBatchCommand,
    GsEndRenderBatchCommand,
    GsWriteRegisterCommand,
    GsNativePackedGifCommand,
    GsNativeImageUploadCommand,
    GsResetCommand,
    GsClearFramebufferCommand,
    GsReadFifoCommand,
    GsCaptureReplayStateCommand,
    GsRestoreReplayStateCommand,
    GsDebugSnapshotCommand,
    GsProgressSnapshotCommand,
    GsDebugHistoryCommand,
    GsRecentGifPacketsCommand,
    GsClearDebugHistoryCommand,
    GsDebugHistoryPausedCommand,
    GsSetDebugHistoryPausedCommand,
    GsSetProgressTrackingCommand,
    GsLatchPresentationCommand,
    GsCopyPresentationCommand,
    GsConfigureVulkanCommand,
    GsSetRendererModeCommand,
    GsRendererStatusCommand,
    GsSetBackendCountersEnabledCommand,
    GsResetBackendCountersCommand,
    GsBackendCountersCommand,
    GsSetDrawCommandLimitCommand,
    GsClearDrawCommandLimitCommand,
    GsDrawStatusCommand,
    GsBarrierCommand>;

struct GsCommand
{
    GsWorkIdentity identity{};
    GsCommandType type = GsCommandType::Barrier;
    uint64_t payloadSize = 0u;
    // GS debug history records VSync ordinals rather than raw EE ticks. Keep
    // that owner-local value beside the architectural work identity.
    uint64_t debugVsyncTick = 0u;
    GsCommandPayload payload = GsBarrierCommand{};
};

struct GsCommandDigest
{
    uint64_t sequence = 0u;
    uint64_t generation = 0u;
    uint64_t guestTick = 0u;
    GsCommandType type = GsCommandType::Barrier;
    uint64_t payloadSize = 0u;
    uint64_t payloadHash = 0u;

    friend constexpr bool operator==(
        GsCommandDigest, GsCommandDigest) noexcept = default;
};

struct GsNoResult
{
};

struct GsDrainBatchResult
{
    GifArbiterDrainBatch batch;
};

struct GsBooleanResult
{
    bool value = false;
};

struct GsFifoReadResult
{
    std::vector<uint8_t> bytes;
};

struct GsReplayStateResult
{
    GsReplayState state{};
};

struct GsDebugSnapshotResult
{
    GSDebugSnapshot snapshot{};
};

struct GsProgressSnapshotResult
{
    GSProgressSnapshot snapshot{};
};

struct GsDebugHistoryResult
{
    std::vector<GSDebugHistoryEntry> entries;
};

struct GsRecentGifPacketsResult
{
    std::vector<uint8_t> stream;
    std::vector<uint32_t> sizes;
    std::vector<uint8_t> initialVram;
    GsReplayState initialState{};
};

struct GsPresentationResult
{
    std::vector<uint8_t> pixels;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferred = false;
    bool available = false;
};

struct GsRendererStatusResult
{
    GsRendererMode mode = GsRendererMode::Software;
    std::string diagnostic;
    GsVulkanCapabilityReport capabilities{};
    GsVulkanServiceStatistics serviceStatistics{};
    GsVulkanRasterBackendStatistics backendStatistics{};
};

struct GsBackendCountersResult
{
    GsBackendCounters counters{};
};

struct GsDrawStatusResult
{
    bool limitReached = false;
    uint64_t submittedCommands = 0u;
};

struct GsSignalEffect
{
    uint32_t id = 0u;
    uint32_t mask = 0u;
};

struct GsFinishEffect
{
};

struct GsLabelEffect
{
    uint32_t id = 0u;
    uint32_t mask = 0u;
};

using GsPrivilegedSideEffect = std::variant<
    GsSignalEffect, GsFinishEffect, GsLabelEffect>;

using GsCommandResultPayload = std::variant<
    GsNoResult,
    GsDrainBatchResult,
    GsBooleanResult,
    GsFifoReadResult,
    GsReplayStateResult,
    GsDebugSnapshotResult,
    GsProgressSnapshotResult,
    GsDebugHistoryResult,
    GsRecentGifPacketsResult,
    GsPresentationResult,
    GsRendererStatusResult,
    GsBackendCountersResult,
    GsDrawStatusResult>;

enum class GsCommandDisposition : uint8_t
{
    Completed,
    StaleGeneration,
    FutureGeneration,
    OutOfOrder,
    Malformed,
};

struct GsCommandResult
{
    GsWorkIdentity identity{};
    GsCommandDisposition disposition =
        GsCommandDisposition::Malformed;
    uint64_t completedSequence = 0u;
    GsCommandDigest digest{};
    std::vector<GsPrivilegedSideEffect> privilegedEffects;
    GsCommandResultPayload payload = GsNoResult{};

    [[nodiscard]] bool completed() const noexcept
    {
        return disposition ==
               GsCommandDisposition::Completed;
    }
};

template <typename T>
[[nodiscard]] T takeGsCommandResult(
    GsCommandResult result)
{
    if (!result.completed())
    {
        throw std::logic_error(
            "GS command did not complete");
    }
    T *const payload =
        std::get_if<T>(&result.payload);
    if (!payload)
    {
        throw std::logic_error(
            "GS command returned an unexpected result type");
    }
    return std::move(*payload);
}

[[nodiscard]] GsCommandType gsCommandType(
    const GsCommandPayload &payload) noexcept;
[[nodiscard]] uint64_t gsCommandPayloadSize(
    const GsCommandPayload &payload) noexcept;
[[nodiscard]] uint64_t gsCommandPayloadHash(
    const GsCommandPayload &payload);
[[nodiscard]] GsCommandDigest makeGsCommandDigest(
    const GsCommand &command);
[[nodiscard]] uint64_t gsCommandDigestHash(
    const GsCommandDigest &digest) noexcept;

class GsCommandProcessor
{
public:
    explicit GsCommandProcessor(
        GS &gs, bool commandDigestsEnabled = false);
    ~GsCommandProcessor();

    GsCommandProcessor(const GsCommandProcessor &) = delete;
    GsCommandProcessor &operator=(const GsCommandProcessor &) = delete;

    [[nodiscard]] GsCommandResult process(GsCommand command);
    [[nodiscard]] uint64_t generation() const noexcept
    {
        return m_generation;
    }
    [[nodiscard]] uint64_t completedSequence() const noexcept
    {
        return m_completedSequence;
    }

private:
    GS &m_gs;
    bool m_commandDigestsEnabled = false;
    uint64_t m_generation = 1u;
    uint64_t m_completedSequence = 0u;
    uint64_t m_currentDebugVsyncTick = 0u;
};

enum class GsExecutionMode : uint8_t
{
    Inline,
};

[[nodiscard]] constexpr const char *gsExecutionModeName(
    GsExecutionMode mode) noexcept
{
    switch (mode)
    {
    case GsExecutionMode::Inline:
        return "inline";
    }
    return "unknown";
}

class GsCommandExecutor
{
public:
    virtual ~GsCommandExecutor() = default;

    GsCommandExecutor(const GsCommandExecutor &) = delete;
    GsCommandExecutor &operator=(const GsCommandExecutor &) = delete;

    [[nodiscard]] virtual GsCommandResult submit(
        GsCommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u,
        uint64_t debugVsyncTick = 0u) = 0;

    [[nodiscard]] virtual uint64_t generation() const = 0;
    [[nodiscard]] virtual uint64_t lastSubmittedSequence() const = 0;

protected:
    GsCommandExecutor() = default;
};

class InlineGsExecutor final : public GsCommandExecutor
{
public:
    explicit InlineGsExecutor(GsCommandProcessor &processor);

    InlineGsExecutor(const InlineGsExecutor &) = delete;
    InlineGsExecutor &operator=(const InlineGsExecutor &) = delete;

    [[nodiscard]] GsCommandResult submit(
        GsCommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u,
        uint64_t debugVsyncTick = 0u) override;

    [[nodiscard]] uint64_t generation() const override;
    [[nodiscard]] uint64_t lastSubmittedSequence() const override;

private:
    [[nodiscard]] static bool startsGeneration(
        const GsCommandPayload &payload) noexcept;
    void assertHotProducerThread(
        const GsCommandPayload &payload);

    GsCommandProcessor &m_processor;
    mutable std::mutex m_submitMutex;
    uint64_t m_generation = 1u;
    uint64_t m_nextSequence = 0u;
    uint64_t m_lastGuestTick = 0u;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    std::optional<std::thread::id> m_hotProducerThread;
#endif
};

#endif
