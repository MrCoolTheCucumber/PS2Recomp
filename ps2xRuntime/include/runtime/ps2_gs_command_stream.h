#ifndef PS2_GS_COMMAND_STREAM_H
#define PS2_GS_COMMAND_STREAM_H

#include "runtime/ps2_build_config.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_spsc_queue.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
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
    CopyVram,
    FieldMarker,
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

struct GsCopyVramCommand
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
    // Ordered benchmark/status boundaries require all already-submitted
    // renderer work in the snapshot. Periodic watchdog samples leave this
    // false so observation cannot collapse normal overlap.
    bool quiesceRenderer = false;
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
    GsPrivilegedRegisterSnapshot registers{};
};

struct GsCopyPresentationCommand
{
};

struct GsFieldMarkerCommand
{
    uint64_t fieldSequence = 0u;
    GsPrivilegedRegisterSnapshot registers{};
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
    GsCopyVramCommand,
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
    GsBarrierCommand,
    GsFieldMarkerCommand>;

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

struct GsVramSnapshotResult
{
    std::vector<uint8_t> bytes;
    bool available = false;
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
    uint64_t completedFieldSequence = 0u;
};

struct GsFieldMarkerResult
{
    uint64_t fieldSequence = 0u;
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

using GsCommandResultPayload = std::variant<
    GsNoResult,
    GsDrainBatchResult,
    GsBooleanResult,
    GsFifoReadResult,
    GsReplayStateResult,
    GsVramSnapshotResult,
    GsDebugSnapshotResult,
    GsProgressSnapshotResult,
    GsDebugHistoryResult,
    GsRecentGifPacketsResult,
    GsPresentationResult,
    GsRendererStatusResult,
    GsBackendCountersResult,
    GsDrawStatusResult,
    GsFieldMarkerResult>;

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
    uint64_t m_completedFieldSequence = 0u;
};

enum class GsExecutionMode : uint8_t
{
    Inline,
    ThreadedSynchronous,
    ThreadedAsync,
};

[[nodiscard]] constexpr const char *gsExecutionModeName(
    GsExecutionMode mode) noexcept
{
    switch (mode)
    {
    case GsExecutionMode::Inline:
        return "inline";
    case GsExecutionMode::ThreadedSynchronous:
        return "threaded-sync";
    case GsExecutionMode::ThreadedAsync:
        return "threaded-async";
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

enum class GsExecutorShutdownMode : uint8_t
{
    Drain,
    Cancel,
};

struct ThreadedGsExecutorOptions
{
    size_t queueCapacity = 64u;
    uint64_t payloadCapacityBytes = 64u * 1024u * 1024u;
    // A deterministic owner-thread hook for focused delay/failure tests. The
    // production runtime leaves it empty.
    std::function<void(const GsCommand &, uint64_t)> beforeProcess;
    // Runs after semantic processing but before the future becomes ready.
    // It makes result-publication races deterministic without changing the
    // command processor or production scheduling.
    std::function<void(const GsCommandResult &, uint64_t)> beforePublish;
};

struct ThreadedGsExecutorStatistics
{
    size_t queueCapacity = 0u;
    uint64_t payloadCapacityBytes = 0u;
    size_t queueDepth = 0u;
    size_t queueHighWater = 0u;
    uint64_t queuedPayloadBytes = 0u;
    uint64_t payloadHighWaterBytes = 0u;
    uint64_t submittedTickets = 0u;
    uint64_t completedTickets = 0u;
    uint64_t submittedGeneration = 0u;
    uint64_t submittedSequence = 0u;
    uint64_t completedGeneration = 0u;
    uint64_t completedSequence = 0u;
    uint64_t producerBlockCount = 0u;
    uint64_t producerBlockedNanoseconds = 0u;
    uint64_t producerSlotWaitCount = 0u;
    uint64_t producerSlotWaitNanoseconds = 0u;
    uint64_t producerPayloadWaitCount = 0u;
    uint64_t producerPayloadWaitNanoseconds = 0u;
    uint64_t workerActiveNanoseconds = 0u;
    uint64_t workerIdleNanoseconds = 0u;
    uint64_t barriersCompleted = 0u;
    uint64_t barrierWaitCount = 0u;
    uint64_t barrierWaitNanoseconds = 0u;
    uint64_t fieldMarkersSubmitted = 0u;
    uint64_t fieldMarkersCompleted = 0u;
    uint64_t lastSubmittedFieldSequence = 0u;
    uint64_t lastCompletedFieldSequence = 0u;
    bool started = false;
    bool running = false;
    bool accepting = false;
    bool drainRequested = false;
    bool cancelRequested = false;
    bool failed = false;
};

class GsCommandSubmission
{
public:
    GsCommandSubmission() = default;
    GsCommandSubmission(GsCommandSubmission &&) noexcept = default;
    GsCommandSubmission &operator=(GsCommandSubmission &&) noexcept = default;

    GsCommandSubmission(const GsCommandSubmission &) = delete;
    GsCommandSubmission &operator=(const GsCommandSubmission &) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool ready() const;
    [[nodiscard]] uint64_t ticket() const noexcept
    {
        return m_ticket;
    }
    [[nodiscard]] GsWorkIdentity identity() const noexcept
    {
        return m_identity;
    }
    [[nodiscard]] GsCommandResult wait();

private:
    friend class ThreadedGsExecutor;

    GsCommandSubmission(
        uint64_t ticket,
        GsWorkIdentity identity,
        std::future<GsCommandResult> completion);

    uint64_t m_ticket = 0u;
    GsWorkIdentity m_identity{};
    std::future<GsCommandResult> m_completion;
};

class ThreadedGsExecutor final : public GsCommandExecutor
{
public:
    explicit ThreadedGsExecutor(
        GsCommandProcessor &processor,
        ThreadedGsExecutorOptions options = {});
    ~ThreadedGsExecutor() override;

    ThreadedGsExecutor(const ThreadedGsExecutor &) = delete;
    ThreadedGsExecutor &operator=(const ThreadedGsExecutor &) = delete;

    [[nodiscard]] GsCommandResult submit(
        GsCommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u,
        uint64_t debugVsyncTick = 0u) override;

    // Synchronous policy calls submit() and waits after every command. The
    // asynchronous runtime retains this move-only completion in its ordered
    // EE publication journal while using the exact same transport.
    [[nodiscard]] GsCommandSubmission submitAsync(
        GsCommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u,
        uint64_t debugVsyncTick = 0u);

    [[nodiscard]] uint64_t generation() const override;
    [[nodiscard]] uint64_t lastSubmittedSequence() const override;
    [[nodiscard]] uint64_t lastSubmittedTicket() const noexcept;
    [[nodiscard]] uint64_t lastCompletedTicket() const noexcept;
    [[nodiscard]] uint64_t lastCompletedFieldSequence() const noexcept;

    // Commands below this generation complete as typed stale results without
    // reaching GS state. Lifecycle code must enqueue the corresponding reset
    // or restore command at the new generation before relying on this floor.
    void cancelPendingBeforeGeneration(uint64_t generation);

    void shutdown(
        GsExecutorShutdownMode mode =
            GsExecutorShutdownMode::Drain) noexcept;
    void rethrowFailure() const;
    [[nodiscard]] ThreadedGsExecutorStatistics statistics() const;
    [[nodiscard]] std::thread::id ownerThreadId() const;

private:
    enum class AdmissionResult : uint8_t
    {
        Enqueued,
        SlotCapacity,
        PayloadCapacity,
    };

    struct WorkItem
    {
        uint64_t ticket = 0u;
        uint64_t payloadBytes = 0u;
        GsCommand command{};
        std::promise<GsCommandResult> completion;
    };

    [[nodiscard]] static bool startsGeneration(
        const GsCommandPayload &payload) noexcept;
    void assertHotProducerThread(
        const GsCommandPayload &payload);
    void ensureStarted();
    [[nodiscard]] AdmissionResult tryEnqueue(WorkItem &item);
    [[noreturn]] void throwSubmissionUnavailable() const;
    void signalWorkAvailable() noexcept;
    [[nodiscard]] bool tryAcquireWorkSignal() noexcept;
    void acquireWorkSignal() noexcept;
    void signalSpaceAvailable() noexcept;
    void closeAdmissionAndQuiesceProducer() noexcept;
    void workerMain() noexcept;
    void cancelQueued(const std::exception_ptr &reason) noexcept;
    void recordFatalFailure(std::exception_ptr failure) noexcept;
    void releasePayload(uint64_t bytes) noexcept;
    void updateQueueHighWater(size_t depth) noexcept;
    void updatePayloadHighWater(uint64_t bytes) noexcept;

    GsCommandProcessor &m_processor;
    ThreadedGsExecutorOptions m_options;
    BoundedSpscQueue<WorkItem> m_queue;

    mutable std::mutex m_submitMutex;
    uint64_t m_generation = 1u;
    uint64_t m_nextSequence = 0u;
    uint64_t m_nextTicket = 0u;
    uint64_t m_lastGuestTick = 0u;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    std::optional<std::thread::id> m_hotProducerThread;
#endif

    mutable std::mutex m_stateMutex;
    std::thread m_worker;
    std::thread::id m_ownerThreadId{};
    std::exception_ptr m_fatalFailure;
    bool m_started = false;
    bool m_workerExited = false;

    mutable std::mutex m_shutdownMutex;
    std::atomic<bool> m_accepting{true};
    std::atomic<bool> m_drainRequested{false};
    std::atomic<bool> m_cancelRequested{false};
    // Successful producer/consumer traffic uses only the SPSC ring and these
    // counted atomic signals. State locking is reserved for lifecycle and
    // low-frequency observation.
    std::atomic<uint64_t> m_workSignals{0u};
    std::atomic<uint64_t> m_spaceEpoch{0u};
    std::atomic<uint64_t> m_minimumAcceptedGeneration{1u};
    std::atomic<uint64_t> m_queuedPayloadBytes{0u};
    std::atomic<size_t> m_queueHighWater{0u};
    std::atomic<uint64_t> m_payloadHighWaterBytes{0u};
    std::atomic<uint64_t> m_submittedTickets{0u};
    std::atomic<uint64_t> m_completedTickets{0u};
    std::atomic<uint64_t> m_submittedGeneration{0u};
    std::atomic<uint64_t> m_submittedSequence{0u};
    std::atomic<uint64_t> m_completedGeneration{0u};
    std::atomic<uint64_t> m_completedSequence{0u};
    std::atomic<uint64_t> m_producerBlockCount{0u};
    std::atomic<uint64_t> m_producerBlockedNanoseconds{0u};
    std::atomic<uint64_t> m_producerSlotWaitCount{0u};
    std::atomic<uint64_t> m_producerSlotWaitNanoseconds{0u};
    std::atomic<uint64_t> m_producerPayloadWaitCount{0u};
    std::atomic<uint64_t> m_producerPayloadWaitNanoseconds{0u};
    std::atomic<uint64_t> m_workerActiveNanoseconds{0u};
    std::atomic<uint64_t> m_workerIdleNanoseconds{0u};
    std::atomic<uint64_t> m_barriersCompleted{0u};
    std::atomic<uint64_t> m_barrierWaitCount{0u};
    std::atomic<uint64_t> m_barrierWaitNanoseconds{0u};
    std::atomic<uint64_t> m_fieldMarkersSubmitted{0u};
    std::atomic<uint64_t> m_fieldMarkersCompleted{0u};
    std::atomic<uint64_t> m_lastSubmittedFieldSequence{0u};
    std::atomic<uint64_t> m_lastCompletedFieldSequence{0u};
};

#endif
