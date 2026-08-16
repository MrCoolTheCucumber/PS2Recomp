#ifndef PS2_VU1_COMMAND_STREAM_H
#define PS2_VU1_COMMAND_STREAM_H

#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu_recompiler.h"
#include "runtime/ps2_spsc_queue.h"

#include <array>
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
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

struct Vu1WorkIdentity
{
    uint64_t sequence = 0u;
    uint64_t generation = 0u;
    uint64_t guestTick = 0u;
    uint64_t publicationToken = 0u;

    friend constexpr bool operator==(
        Vu1WorkIdentity, Vu1WorkIdentity) noexcept = default;
};

enum class Vu1CommandType : uint8_t
{
    BindMemory,
    SetDiagnostics,
    SetBackend,
    MicroMemoryWrite,
    DataMemoryWrite,
    DecodedUnpack,
    VifStateUpdate,
    RegisterWrite,
    Mscal,
    Mscalf,
    Mscnt,
    Invocation,
    AdvanceSlice,
    Reset,
    Snapshot,
    Restore,
    Barrier,
    Shutdown,
};

inline constexpr size_t kVu1CommandTypeCount =
    static_cast<size_t>(Vu1CommandType::Shutdown) + 1u;

[[nodiscard]] constexpr size_t vu1CommandTypeIndex(
    Vu1CommandType type) noexcept
{
    return static_cast<size_t>(type);
}

[[nodiscard]] const char *vu1CommandTypeName(
    Vu1CommandType type) noexcept;

struct Vu1BindMemoryCommand
{
    std::vector<uint8_t> microMemory;
    std::vector<uint8_t> dataMemory;
    uint64_t codeGeneration = 0u;
    bool deferredDiagnostics = false;
};

struct Vu1SetDiagnosticsCommand
{
    bool traceEnabled = false;
    bool workloadProfileEnabled = false;
};

struct Vu1SetBackendCommand
{
    VuBackendKind backend = VuBackendKind::Auto;
};

struct Vu1MicroMemoryWriteCommand
{
    uint32_t offset = 0u;
    std::vector<uint8_t> bytes;
    bool wrap = false;
};

struct Vu1DataMemoryWriteCommand
{
    uint32_t offset = 0u;
    std::vector<uint8_t> bytes;
    bool wrap = false;
};

struct Vu1VifState
{
    uint32_t cycle = 0x0101u;
    uint32_t mode = 0u;
    uint32_t mask = 0u;
    std::array<uint32_t, 4u> row{};
    std::array<uint32_t, 4u> column{};
    uint32_t tops = 0u;

    friend constexpr bool operator==(
        const Vu1VifState &,
        const Vu1VifState &) noexcept = default;
};

// The EE-side VIF parser owns byte-stream framing. This value owns the
// complete decoded UNPACK input and enough alignment metadata for the VU1
// owner to reproduce mutation without retaining a pointer into DMA storage.
// Threaded lookahead may fold the latest parser-owned VIF state into this
// command, preserving order without a separate owner rendezvous.
struct Vu1DecodedUnpackCommand
{
    uint16_t immediate = 0u;
    uint8_t vectorLength = 0u;
    uint8_t componentCount = 0u;
    uint16_t writeVectorCount = 0u;
    uint16_t sourceVectorCount = 0u;
    uint8_t sourceWordAlignment = 4u;
    bool maskEnabled = false;
    bool zeroExtend = false;
    std::vector<uint8_t> bytes;
    std::shared_ptr<const Vu1VifState> vifStateBefore;
};

// One parser-local owner mutation. The EE still owns VIF byte framing while
// the immutable batch lets the VU1 owner apply consecutive decoded UNPACKs in
// their original order with one transport ticket.
struct Vu1DecodedUnpackBatch
{
    std::vector<Vu1DecodedUnpackCommand> commands;
};

struct Vu1DecodedUnpackBatchCommand
{
    std::shared_ptr<const Vu1DecodedUnpackBatch> batch;
};

struct Vu1VifStateUpdateCommand
{
    Vu1VifState state{};
};

enum class Vu1RegisterKind : uint8_t
{
    VectorFloat,
    VectorInteger,
    Accumulator,
    Q,
    P,
    I,
    ProgramCounter,
    Mac,
    Clip,
    Status,
    Top,
    Itop,
};

struct Vu1RegisterWriteCommand
{
    Vu1RegisterKind kind = Vu1RegisterKind::VectorInteger;
    uint8_t index = 0u;
    uint8_t laneMask = 0x0fu;
    std::array<uint32_t, 4u> words{};
};

struct Vu1MscalCommand
{
    uint32_t startPc = 0u;
    uint32_t top = 0u;
    uint32_t itop = 0u;
    std::shared_ptr<const Vu1DecodedUnpackBatch> unpacksBefore;
    std::shared_ptr<const Vu1VifState> vifStateBefore;
};

struct Vu1MscalfCommand
{
    uint32_t startPc = 0u;
    uint32_t top = 0u;
    uint32_t itop = 0u;
    std::shared_ptr<const Vu1DecodedUnpackBatch> unpacksBefore;
    std::shared_ptr<const Vu1VifState> vifStateBefore;
};

struct Vu1MscntCommand
{
    uint32_t top = 0u;
    uint32_t itop = 0u;
    std::shared_ptr<const Vu1DecodedUnpackBatch> unpacksBefore;
    std::shared_ptr<const Vu1VifState> vifStateBefore;
};

enum class Vu1InvocationKind : uint8_t
{
    Mscal,
    Mscalf,
    Mscnt,
};

[[nodiscard]] constexpr const char *vu1InvocationKindName(
    Vu1InvocationKind kind) noexcept
{
    switch (kind)
    {
    case Vu1InvocationKind::Mscal:
        return "mscal";
    case Vu1InvocationKind::Mscalf:
        return "mscalf";
    case Vu1InvocationKind::Mscnt:
        return "mscnt";
    }
    return "unknown";
}

// Coarse scheduling keeps one complete VU activation private to the owner.
// The cycle and PATH1 limits make the relaxed work item explicitly bounded;
// exact slice commands remain available as the architectural oracle.
struct Vu1InvocationCommand
{
    Vu1InvocationKind kind = Vu1InvocationKind::Mscal;
    uint32_t startPc = 0u;
    uint32_t top = 0u;
    uint32_t itop = 0u;
    uint32_t maximumCycles = 65536u;
    uint64_t maximumPath1Bytes = 64u * 1024u * 1024u;
    bool captureState = false;
    std::shared_ptr<const Vu1DecodedUnpackBatch> unpacksBefore;
    std::shared_ptr<const Vu1VifState> vifStateBefore;
};

struct Vu1AdvanceSliceCommand
{
    uint32_t maximumCycles = 0u;
    bool captureState = false;
};

struct Vu1ResetCommand
{
    bool clearMicroMemory = false;
    bool clearDataMemory = false;
};

struct Vu1SnapshotCommand
{
    bool includeBackendDiagnostics = false;
};

struct Vu1RecompilerSnapshot
{
    VuRecompilerDiagnostics diagnostics{};
    VuBlockProfilingSnapshot blockProfile{};
    std::string lastJitDiagnostic;
    bool blockLinkingEnabled = false;
    bool blockBudgetGuardsEnabled = false;
    bool blockLocalVfRegistersEnabled = false;
    bool blockLocalVfRegistersAutomatic = false;
    bool inlineXgkickEnabled = false;
};

struct Vu1BackendDiagnosticsSnapshot
{
    std::optional<VuProgramCacheDiagnostics> programCache;
    std::optional<Vu1RecompilerSnapshot> recompiler;
};

struct Vu1Snapshot
{
    VuExecutionState state{};
    std::vector<uint8_t> microMemory;
    std::vector<uint8_t> dataMemory;
    Vu1VifState vif{};
    uint64_t codeGeneration = 0u;
    VuExitReason lastExitReason = VuExitReason::Inactive;
    VuBackendKind requestedBackend = VuBackendKind::Auto;
    VuBackendKind resolvedBackend = VuBackendKind::Interpreter;
    std::string backendName;
    bool nativeInstrumentationEnabled = false;
    VuProgressSnapshot progress{};
    VuVerifyDiagnostics verify{};
    std::optional<Vu1BackendDiagnosticsSnapshot>
        backendDiagnostics;
};

struct Vu1RestoreCommand
{
    // Snapshot payloads are cold and large. Box them so every hot command
    // record remains a compact, bounded queue envelope.
    std::shared_ptr<const Vu1Snapshot> snapshot;
};

struct Vu1BarrierCommand
{
};

struct Vu1ShutdownCommand
{
};

using Vu1CommandPayload = std::variant<
    Vu1BindMemoryCommand,
    Vu1SetDiagnosticsCommand,
    Vu1SetBackendCommand,
    Vu1MicroMemoryWriteCommand,
    Vu1DataMemoryWriteCommand,
    Vu1DecodedUnpackCommand,
    Vu1DecodedUnpackBatchCommand,
    Vu1VifStateUpdateCommand,
    Vu1RegisterWriteCommand,
    Vu1MscalCommand,
    Vu1MscalfCommand,
    Vu1MscntCommand,
    Vu1InvocationCommand,
    Vu1AdvanceSliceCommand,
    Vu1ResetCommand,
    Vu1SnapshotCommand,
    Vu1RestoreCommand,
    Vu1BarrierCommand,
    Vu1ShutdownCommand>;

struct Vu1Command
{
    Vu1WorkIdentity identity{};
    Vu1CommandType type = Vu1CommandType::Barrier;
    uint64_t payloadSize = 0u;
    Vu1CommandPayload payload = Vu1BarrierCommand{};
};

struct Vu1CommandDigest
{
    uint64_t sequence = 0u;
    uint64_t generation = 0u;
    uint64_t guestTick = 0u;
    Vu1CommandType type = Vu1CommandType::Barrier;
    uint64_t payloadSize = 0u;
    uint64_t payloadHash = 0u;

    friend constexpr bool operator==(
        Vu1CommandDigest,
        Vu1CommandDigest) noexcept = default;
};

struct Vu1Path1Packet
{
    std::vector<uint8_t> bytes;
    std::optional<uint32_t> cycleOffset;
};

struct Vu1TraceInvocationDiagnostic
{
    uint32_t startPc = 0u;
    uint32_t top = 0u;
    uint32_t itop = 0u;
    bool resume = false;
    VuExecutionState state{};
};

struct Vu1TraceInstructionDiagnostic
{
    uint32_t pc = 0u;
    uint32_t lower = 0u;
    uint32_t upper = 0u;
    VuExecutionState state{};
};

struct Vu1TraceXgkickDiagnostic
{
    uint32_t sourceQword = 0u;
};

struct Vu1TraceInvocationEndDiagnostic
{
    uint32_t finalPc = 0u;
    bool ended = false;
    bool hitCycleLimit = false;
    std::vector<int32_t> viRegisters;
};

struct Vu1WorkloadBeginDiagnostic
{
    uint32_t startPc = 0u;
    std::vector<uint8_t> code;
    uint64_t generation = 0u;
};

struct Vu1WorkloadInstructionDiagnostic
{
    uint32_t pc = 0u;
    uint32_t lower = 0u;
    uint32_t upper = 0u;
};

struct Vu1WorkloadTransitionDiagnostic
{
    uint32_t pc = 0u;
    uint32_t nextPc = 0u;
};

struct Vu1WorkloadEndDiagnostic
{
    bool completed = false;
};

struct Vu1WorkloadResetDiagnostic
{
};

using Vu1DiagnosticRecord = std::variant<
    Vu1TraceInvocationDiagnostic,
    Vu1TraceInstructionDiagnostic,
    Vu1TraceXgkickDiagnostic,
    Vu1TraceInvocationEndDiagnostic,
    Vu1WorkloadBeginDiagnostic,
    Vu1WorkloadInstructionDiagnostic,
    Vu1WorkloadTransitionDiagnostic,
    Vu1WorkloadEndDiagnostic,
    Vu1WorkloadResetDiagnostic>;

struct Vu1NoResult
{
};

struct Vu1ArchitecturalStateHashes
{
    // Component hashes are diagnostics-only. Input fields retain the state
    // presented to the first instruction; the unprefixed fields retain the
    // state after the returned slice or invocation.
    uint64_t aggregate = 0u;
    uint64_t execution = 0u;
    uint64_t microMemory = 0u;
    uint64_t dataMemory = 0u;
    uint64_t vif = 0u;
    uint64_t inputAggregate = 0u;
    uint64_t inputExecution = 0u;
    uint64_t inputMicroMemory = 0u;
    uint64_t inputDataMemory = 0u;
    uint64_t inputVif = 0u;
};

struct Vu1SliceResult
{
    VuRunResult run{};
    std::vector<Vu1Path1Packet> path1Packets;
    // State capture is diagnostic-only. Keeping it out of the inline result
    // envelope avoids moving a full architectural state on every slice.
    std::unique_ptr<VuExecutionState> state;
    uint64_t architecturalStateHash = 0u;
    std::unique_ptr<Vu1ArchitecturalStateHashes>
        diagnosticStateHashes;
    bool vifCanResume = false;
    // Fault details are cold. Keep their string storage and object footprint
    // out of every scheduled slice result.
    std::unique_ptr<std::string> fault;
};

struct Vu1SnapshotResult
{
    std::shared_ptr<const Vu1Snapshot> snapshot;
    uint64_t architecturalStateHash = 0u;
};

struct Vu1BarrierResult
{
    uint64_t architecturalStateHash = 0u;
};

struct Vu1VifStateResult
{
    Vu1VifState state{};
};

struct Vu1BackendStatusResult
{
    bool accepted = false;
    VuBackendKind requested = VuBackendKind::Auto;
    VuBackendKind resolved = VuBackendKind::Interpreter;
    std::string name;
    bool active = false;
    std::string diagnostic;
};

using Vu1CommandResultPayload = std::variant<
    Vu1NoResult,
    Vu1SliceResult,
    Vu1SnapshotResult,
    Vu1BarrierResult,
    Vu1VifStateResult,
    Vu1BackendStatusResult>;

enum class Vu1CommandDisposition : uint8_t
{
    Completed,
    StaleGeneration,
    FutureGeneration,
    OutOfOrder,
    Malformed,
    Shutdown,
};

struct Vu1CommandResult
{
    Vu1WorkIdentity identity{};
    Vu1CommandDisposition disposition =
        Vu1CommandDisposition::Completed;
    Vu1CommandDigest digest{};
    uint64_t ownerGeneration = 0u;
    uint64_t codeGeneration = 0u;
    uint32_t programCounter = 0u;
    bool active = false;
    VuProgressSnapshot progress{};
    std::vector<Vu1DiagnosticRecord> diagnostics;
    Vu1CommandResultPayload payload = Vu1NoResult{};
};

void replayVu1Diagnostics(
    std::span<const Vu1DiagnosticRecord> diagnostics,
    IVuExecutionObserver &observer);

[[nodiscard]] Vu1CommandType vu1CommandType(
    const Vu1CommandPayload &payload) noexcept;
[[nodiscard]] uint64_t vu1CommandPayloadSize(
    const Vu1CommandPayload &payload) noexcept;
[[nodiscard]] uint64_t vu1CommandPayloadHash(
    const Vu1CommandPayload &payload);
[[nodiscard]] Vu1CommandDigest vu1CommandDigest(
    const Vu1Command &command);
[[nodiscard]] uint64_t vu1CommandDigestHash(
    const Vu1CommandDigest &digest) noexcept;
[[nodiscard]] uint64_t vu1ArchitecturalStateHash(
    const VuExecutionState &state,
    std::span<const uint8_t> microMemory,
    std::span<const uint8_t> dataMemory,
    const Vu1VifState &vif,
    uint64_t codeGeneration) noexcept;
[[nodiscard]] bool applyVu1DecodedUnpack(
    const Vu1DecodedUnpackCommand &command,
    Vu1VifState &vif,
    uint8_t *dataMemory,
    uint32_t dataMemorySize);

struct Vu1CommandProcessorConfiguration
{
    bool captureCommandDigests = false;
    bool captureArchitecturalStateHashes = false;
};

struct Vu1SpeculationStatistics
{
    uint64_t capturedSlices = 0u;
    uint64_t checkpointBytesCopied = 0u;
    uint64_t checkpointCaptureNanoseconds = 0u;
    uint64_t committedSlices = 0u;
    uint64_t rolledBackSlices = 0u;
    uint64_t rollbackNanoseconds = 0u;
};

enum class Vu1SpeculationResolution : uint8_t
{
    None,
    Commit,
    Rollback,
};

// This is the sole semantic entry point for VU1 owner-facing work. A
// BindMemory command transfers canonical micro/data images into processor-owned
// storage; the pointer-binding helper remains only for standalone inline
// fixtures which already own their backing storage.
class Vu1CommandProcessor final : public IVuExecutionObserver
{
public:
    explicit Vu1CommandProcessor(
        VuUnit &unit,
        Vu1CommandProcessorConfiguration configuration = {});

    void bindMemory(
        uint8_t *microMemory, uint32_t microMemorySize,
        uint8_t *dataMemory, uint32_t dataMemorySize,
        uint64_t codeGeneration,
        IVuExecutionObserver *diagnosticsObserver = nullptr);
    void bindInlineDiagnosticsObserver(
        IVuExecutionObserver *diagnosticsObserver);
    // Bounded debugger traces can request complete owner-state hashes without
    // paying their full-memory cost during ordinary execution. This flag is
    // diagnostic-only and is safe to toggle from the paused debugger thread;
    // the owner remains the only thread which reads and hashes VU state.
    void setDiagnosticArchitecturalStateHashes(bool enabled) const noexcept
    {
        m_diagnosticArchitecturalStateHashes.store(
            enabled, std::memory_order_release);
    }
    void claimOwnerThread(std::thread::id owner);
    [[nodiscard]] std::thread::id ownerThreadId() const noexcept
    {
        return m_ownerThread;
    }

    [[nodiscard]] Vu1CommandResult process(
        Vu1Command command);
    [[nodiscard]] Vu1CommandResult processDecodedUnpack(
        Vu1WorkIdentity identity,
        const Vu1DecodedUnpackCommand &command);
    [[nodiscard]] Vu1CommandResult processVifStateUpdate(
        Vu1WorkIdentity identity,
        const Vu1VifStateUpdateCommand &command);
    [[nodiscard]] Vu1CommandResult processAdvanceSlice(
        Vu1WorkIdentity identity,
        const Vu1AdvanceSliceCommand &command,
        bool speculative = false);
    [[nodiscard]] Vu1CommandResult commitExtendedSpeculativeAdvance(
        Vu1WorkIdentity identity,
        const Vu1AdvanceSliceCommand &publishedCommand,
        uint32_t extensionCycles);
    void resolveSpeculativeSlice(
        Vu1SpeculationResolution resolution);
    [[nodiscard]] Vu1SpeculationStatistics
    speculationStatistics() const noexcept;
    [[nodiscard]] uint64_t generation() const noexcept
    {
        return m_generation;
    }
    [[nodiscard]] uint64_t nextSequence() const noexcept
    {
        return m_nextSequence;
    }
    [[nodiscard]] uint64_t codeGeneration() const noexcept
    {
        return m_codeGeneration.load(
            std::memory_order_relaxed);
    }
    [[nodiscard]] const Vu1VifState &vifState() const noexcept
    {
        return m_vifState;
    }

    [[nodiscard]] bool vuTraceEnabled() const override;
    [[nodiscard]] bool vuWorkloadProfileEnabled() const override;
    [[nodiscard]] uint64_t currentVuCodeGeneration(
        VuUnitId unit) const override;
    void traceVuInvocation(
        uint32_t startPc, uint32_t top, uint32_t itop,
        bool resume, const VuExecutionState &state) override;
    void traceVuInstruction(
        uint32_t pc, uint32_t lower, uint32_t upper,
        const VuExecutionState &state) override;
    void traceVuXgkick(uint32_t sourceQword) override;
    void traceVuInvocationEnd(
        uint32_t finalPc, bool ended, bool hitCycleLimit,
        const int32_t *viRegisters,
        size_t viRegisterCount) override;
    void beginVuWorkloadProfileInvocation(
        uint32_t startPc, const uint8_t *code,
        uint32_t codeSize, uint64_t generation) override;
    void recordVuWorkloadProfileInstruction(
        uint32_t pc, uint32_t lower, uint32_t upper) override;
    void recordVuWorkloadProfileTransition(
        uint32_t pc, uint32_t nextPc) override;
    void endVuWorkloadProfileInvocation(bool completed) override;
    void resetVuWorkloadProfileEpoch() override;

private:
    friend class ThreadedVu1Executor;

    struct SpeculativeSliceCheckpoint
    {
        VuExecutionState state{};
        Vu1VifState vif{};
        VuProgressSnapshot progress{};
        VuExitReason lastExitReason = VuExitReason::Inactive;
        VuVerifyDiagnostics verify{};
        IVuExecutionObserver *workloadProfileObserver = nullptr;
        uint64_t generation = 0u;
        uint64_t nextSequence = 0u;
        uint64_t codeGeneration = 0u;
        std::vector<uint8_t> dataMemory;
        bool workloadProfileInvocationPending = false;
        bool workloadProfileInvocationActive = false;
    };

    void assertOwnerThread() const;
    [[nodiscard]] SpeculativeSliceCheckpoint
    captureProcessorCheckpoint();
    void restoreProcessorCheckpoint(
        const SpeculativeSliceCheckpoint &checkpoint);
    void captureSpeculativeSliceCheckpoint();
    void rollbackSpeculativeSlice();
    void commitSpeculativeSlice();
    [[nodiscard]] Vu1CommandResult processAdvanceSliceImpl(
        Vu1WorkIdentity identity,
        const Vu1AdvanceSliceCommand &command,
        bool speculative,
        bool epochContinuation);
    [[nodiscard]] Vu1CommandResult processEpochContinuation(
        Vu1WorkIdentity identity,
        const Vu1AdvanceSliceCommand &command);
    [[nodiscard]] bool validMemoryRange(
        uint32_t offset, size_t size,
        uint32_t memorySize, bool wrap) const noexcept;
    void writeMemory(
        uint8_t *destination, uint32_t destinationSize,
        uint32_t offset, std::span<const uint8_t> bytes,
        bool wrap);
    [[nodiscard]] Vu1Snapshot captureSnapshot(
        bool includeBackendDiagnostics) const;
    [[nodiscard]] uint64_t stateHash() const noexcept;
    [[nodiscard]] Vu1ArchitecturalStateHashes
    stateHashes() const noexcept;
    [[nodiscard]] Vu1ArchitecturalStateHashes
    stateHashes(bool diagnostic) const noexcept;
    [[nodiscard]] Vu1CommandResultPayload apply(
        Vu1CommandPayload &payload,
        Vu1CommandDisposition &disposition);
    [[nodiscard]] Vu1SliceResult advanceSlice(
        const Vu1AdvanceSliceCommand &command);
    [[nodiscard]] Vu1SliceResult runInvocation(
        const Vu1InvocationCommand &command);

    VuUnit &m_unit;
    Vu1CommandProcessorConfiguration m_configuration{};
    uint8_t *m_microMemory = nullptr;
    uint32_t m_microMemorySize = 0u;
    uint8_t *m_dataMemory = nullptr;
    uint32_t m_dataMemorySize = 0u;
    std::atomic<uint64_t> m_codeGeneration{0u};
    mutable std::atomic<bool>
        m_diagnosticArchitecturalStateHashes{false};
    IVuExecutionObserver *m_diagnosticsObserver = nullptr;
    std::vector<uint8_t> m_ownedMicroMemory;
    std::vector<uint8_t> m_ownedDataMemory;
    bool m_deferredDiagnostics = false;
    bool m_traceEnabled = false;
    bool m_workloadProfileEnabled = false;
    std::vector<Vu1DiagnosticRecord> m_pendingDiagnostics;
    Vu1VifState m_vifState{};
    uint64_t m_generation = 1u;
    uint64_t m_nextSequence = 1u;
    std::thread::id m_ownerThread{};
    bool m_shutdown = false;
    std::optional<SpeculativeSliceCheckpoint>
        m_speculativeSliceCheckpoint;
    std::atomic<uint64_t> m_speculativeSlicesCaptured{0u};
    std::atomic<uint64_t> m_speculativeCheckpointBytesCopied{0u};
    std::atomic<uint64_t> m_speculativeCheckpointCaptureNanoseconds{0u};
    std::atomic<uint64_t> m_speculativeSlicesCommitted{0u};
    std::atomic<uint64_t> m_speculativeSlicesRolledBack{0u};
    std::atomic<uint64_t> m_speculativeRollbackNanoseconds{0u};
};

enum class Vu1ExecutionMode : uint8_t
{
    Inline,
    ThreadedSynchronous,
    ThreadedAsync,
    ThreadedCoarse,
};

[[nodiscard]] constexpr const char *vu1ExecutionModeName(
    Vu1ExecutionMode mode) noexcept
{
    switch (mode)
    {
    case Vu1ExecutionMode::Inline:
        return "inline";
    case Vu1ExecutionMode::ThreadedSynchronous:
        return "threaded-sync";
    case Vu1ExecutionMode::ThreadedAsync:
        return "threaded-async";
    case Vu1ExecutionMode::ThreadedCoarse:
        return "threaded-coarse";
    }
    return "unknown";
}

class Vu1CommandExecutor
{
public:
    virtual ~Vu1CommandExecutor() = default;
    [[nodiscard]] virtual Vu1CommandResult submit(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) = 0;
    // Inline execution can avoid constructing the generic variant envelope
    // for the event scheduler's dominant command. A queued executor inherits
    // the ordered generic fallback unless it provides an equivalent fast
    // path; both routes retain the same command and result types.
    [[nodiscard]] virtual Vu1CommandResult submitAdvanceSlice(
        Vu1AdvanceSliceCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] virtual Vu1CommandResult submitDecodedUnpack(
        Vu1DecodedUnpackCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] virtual Vu1CommandResult submitVifStateUpdate(
        Vu1VifStateUpdateCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] virtual uint64_t generation() const = 0;
};

class InlineVu1Executor final : public Vu1CommandExecutor
{
public:
    explicit InlineVu1Executor(Vu1CommandProcessor &processor);

    [[nodiscard]] Vu1CommandResult submit(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) override;
    [[nodiscard]] Vu1CommandResult submitAdvanceSlice(
        Vu1AdvanceSliceCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) override;
    [[nodiscard]] Vu1CommandResult submitDecodedUnpack(
        Vu1DecodedUnpackCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) override;
    [[nodiscard]] Vu1CommandResult submitVifStateUpdate(
        Vu1VifStateUpdateCommand command,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) override;
    [[nodiscard]] uint64_t generation() const override
    {
        return m_generation;
    }

private:
    [[nodiscard]] Vu1WorkIdentity nextIdentity(
        bool startsGeneration,
        uint64_t guestTick,
        uint64_t publicationToken);
    void completeIdentity(
        bool startsGeneration,
        const Vu1CommandResult &result);

    Vu1CommandProcessor &m_processor;
    uint64_t m_generation = 1u;
    uint64_t m_nextSequence = 1u;
    uint64_t m_lastGuestTick = 0u;
};

enum class Vu1ExecutorShutdownMode : uint8_t
{
    Drain,
    Cancel,
};

struct ThreadedVu1ExecutorOptions
{
    size_t queueCapacity = 64u;
    uint64_t payloadCapacityBytes = 8u * 1024u * 1024u;
    std::function<void(const Vu1Command &, uint64_t)> beforeProcess;
    std::function<void(const Vu1CommandResult &, uint64_t)> beforePublish;
};

struct Vu1ExecutionEpochOptions
{
    uint32_t initialCycles = 0u;
    uint32_t followupCycles = 128u;
    size_t checkpointCapacity = 4u;
    bool captureState = false;
};

struct Vu1ExecutionEpochStatistics
{
    uint64_t epochsSubmitted = 0u;
    uint64_t epochsCompleted = 0u;
    uint64_t checkpointsProduced = 0u;
    uint64_t checkpointsPublished = 0u;
    uint64_t suffixCheckpointsDiscarded = 0u;
    uint64_t demandRebases = 0u;
    uint64_t positiveDemandExtensions = 0u;
    uint64_t positiveDemandPrefixCyclesReused = 0u;
    uint64_t positiveDemandExtensionCycles = 0u;
    uint64_t positiveDemandExtensionExecutedCycles = 0u;
    uint64_t checkpointWaitCount = 0u;
    uint64_t checkpointWaitNanoseconds = 0u;
    size_t maximumJournalDepth = 0u;
};

struct Vu1ExecutionEpochState;

// One bounded owner activation may compute several exact event checkpoints.
// The handle carries no per-checkpoint future; EE publishes one result at a
// time through the executor while ordered commands stop and roll back only
// the still-unpublished suffix.
class Vu1ExecutionEpoch
{
public:
    Vu1ExecutionEpoch() = default;
    Vu1ExecutionEpoch(Vu1ExecutionEpoch &&) noexcept = default;
    Vu1ExecutionEpoch &operator=(Vu1ExecutionEpoch &&) noexcept = default;

    Vu1ExecutionEpoch(const Vu1ExecutionEpoch &) = delete;
    Vu1ExecutionEpoch &operator=(const Vu1ExecutionEpoch &) = delete;

    [[nodiscard]] bool valid() const noexcept
    {
        return static_cast<bool>(m_state);
    }
    [[nodiscard]] uint64_t ticket() const noexcept
    {
        return m_ticket;
    }
    [[nodiscard]] Vu1WorkIdentity identity() const noexcept
    {
        return m_identity;
    }

private:
    friend class ThreadedVu1Executor;

    Vu1ExecutionEpoch(
        uint64_t ticket,
        Vu1WorkIdentity identity,
        std::shared_ptr<Vu1ExecutionEpochState> state)
        : m_ticket(ticket),
          m_identity(identity),
          m_state(std::move(state))
    {
    }

    uint64_t m_ticket = 0u;
    Vu1WorkIdentity m_identity{};
    std::shared_ptr<Vu1ExecutionEpochState> m_state;
};

struct ThreadedVu1CommandTypeStatistics
{
    uint64_t submitted = 0u;
    uint64_t completed = 0u;
    uint64_t resultWaitCount = 0u;
    uint64_t resultWaitNanoseconds = 0u;
};

struct ThreadedVu1ExecutorStatistics
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
    uint64_t workNotificationCount = 0u;
    uint64_t workerWakeCount = 0u;
    uint64_t workerDrainCount = 0u;
    uint64_t resultWaitCount = 0u;
    uint64_t resultWaitNanoseconds = 0u;
    uint64_t submittedDecodedUnpackOperations = 0u;
    uint64_t executedDecodedUnpackOperations = 0u;
    std::array<ThreadedVu1CommandTypeStatistics,
               kVu1CommandTypeCount>
        commandTypes{};
    Vu1SpeculationStatistics speculation{};
    Vu1ExecutionEpochStatistics executionEpoch{};
    bool started = false;
    bool running = false;
    bool accepting = false;
    bool drainRequested = false;
    bool cancelRequested = false;
    bool failed = false;
};

// A resultless admission proves that immutable command input was accepted by
// the bounded owner stream. It deliberately carries no future: callers which
// need to observe owner state must submit a later response-producing command,
// whose FIFO completion covers every preceding admission.
struct Vu1CommandAdmission
{
    uint64_t ticket = 0u;
    Vu1WorkIdentity identity{};
    Vu1CommandType type = Vu1CommandType::Barrier;
};

class Vu1CommandSubmission
{
public:
    Vu1CommandSubmission() = default;
    Vu1CommandSubmission(Vu1CommandSubmission &&) noexcept = default;
    Vu1CommandSubmission &operator=(Vu1CommandSubmission &&) noexcept = default;

    Vu1CommandSubmission(const Vu1CommandSubmission &) = delete;
    Vu1CommandSubmission &operator=(const Vu1CommandSubmission &) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool ready() const;
    [[nodiscard]] uint64_t ticket() const noexcept
    {
        return m_ticket;
    }
    [[nodiscard]] Vu1WorkIdentity identity() const noexcept
    {
        return m_identity;
    }
    [[nodiscard]] Vu1CommandType type() const noexcept
    {
        return m_type;
    }
    [[nodiscard]] Vu1CommandResult wait();

private:
    friend class ThreadedVu1Executor;

    Vu1CommandSubmission(
        uint64_t ticket,
        Vu1WorkIdentity identity,
        Vu1CommandType type,
        std::future<Vu1CommandResult> completion);

    uint64_t m_ticket = 0u;
    Vu1WorkIdentity m_identity{};
    Vu1CommandType m_type = Vu1CommandType::Barrier;
    std::future<Vu1CommandResult> m_completion;
};

class ThreadedVu1Executor final : public Vu1CommandExecutor
{
public:
    explicit ThreadedVu1Executor(
        Vu1CommandProcessor &processor,
        ThreadedVu1ExecutorOptions options = {});
    ~ThreadedVu1Executor() override;

    ThreadedVu1Executor(const ThreadedVu1Executor &) = delete;
    ThreadedVu1Executor &operator=(const ThreadedVu1Executor &) = delete;

    [[nodiscard]] Vu1CommandResult submit(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) override;
    [[nodiscard]] Vu1CommandSubmission submitAsync(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandAdmission submitResultless(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1ExecutionEpoch submitExecutionEpoch(
        Vu1ExecutionEpochOptions options,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] size_t executionEpochReadyCheckpointCount(
        const Vu1ExecutionEpoch &epoch) const;
    [[nodiscard]] bool executionEpochCheckpointReady(
        const Vu1ExecutionEpoch &epoch,
        uint32_t exactCycles) const;
    [[nodiscard]] Vu1CommandResult
    waitExecutionEpochCheckpoint(
        Vu1ExecutionEpoch &epoch,
        uint32_t exactCycles,
        uint64_t guestTick,
        uint64_t publicationToken);
    void stopExecutionEpoch(
        Vu1ExecutionEpoch &epoch) noexcept;
    [[nodiscard]] Vu1CommandSubmission submitSpeculativeAdvance(
        Vu1AdvanceSliceCommand command,
        Vu1SpeculationResolution previousResolution,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandResult submitResolvingSpeculation(
        Vu1CommandPayload payload,
        Vu1SpeculationResolution resolution,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    [[nodiscard]] Vu1CommandAdmission
    submitResultlessResolvingSpeculation(
        Vu1CommandPayload payload,
        Vu1SpeculationResolution resolution,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u);
    // A late EE event may extend a completed speculative prefix without
    // rolling it back. The extension consumes another transport ticket but
    // remains the same logical VU command identity and commits the original
    // checkpoint exactly once.
    [[nodiscard]] Vu1CommandResult commitExtendedSpeculativeAdvance(
        Vu1AdvanceSliceCommand publishedCommand,
        uint32_t extensionCycles,
        uint64_t guestTick,
        uint64_t publicationToken);
    [[nodiscard]] Vu1CommandResult wait(
        Vu1CommandSubmission submission);

    [[nodiscard]] uint64_t generation() const override;
    [[nodiscard]] uint64_t lastSubmittedSequence() const;
    [[nodiscard]] uint64_t lastSubmittedTicket() const noexcept;
    [[nodiscard]] uint64_t lastCompletedTicket() const noexcept;

    void cancelPendingBeforeGeneration(uint64_t generation);
    void shutdown(
        Vu1ExecutorShutdownMode mode =
            Vu1ExecutorShutdownMode::Drain) noexcept;
    void rethrowFailure() const;
    [[nodiscard]] ThreadedVu1ExecutorStatistics statistics() const;
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
        uint64_t decodedUnpackOperations = 0u;
        Vu1Command command{};
        Vu1SpeculationResolution speculationResolution =
            Vu1SpeculationResolution::None;
        bool beginsSpeculativeSlice = false;
        bool commitsExtendedSpeculativeAdvance = false;
        uint32_t speculativeExtensionCycles = 0u;
        std::shared_ptr<Vu1ExecutionEpochState> executionEpoch;
        std::optional<std::promise<Vu1CommandResult>> completion;
    };

    struct ProducerSpeculationCheckpoint
    {
        Vu1WorkIdentity identity{};
        uint64_t generation = 0u;
        uint64_t nextSequence = 0u;
        uint64_t lastGuestTick = 0u;
    };

    [[nodiscard]] static bool startsGeneration(
        const Vu1CommandPayload &payload) noexcept;
    [[nodiscard]] Vu1CommandSubmission submitAsyncImpl(
        Vu1CommandPayload payload,
        Vu1SpeculationResolution resolution,
        bool beginsSpeculativeSlice,
        bool commitsExtendedSpeculativeAdvance,
        uint32_t speculativeExtensionCycles,
        bool retainCompletion,
        uint64_t guestTick,
        uint64_t publicationToken);
    void requestActiveExecutionEpochStopLocked() noexcept;
    void runExecutionEpoch(WorkItem &item);
    void failExecutionEpoch(
        const std::shared_ptr<Vu1ExecutionEpochState> &state,
        const std::exception_ptr &failure) noexcept;
    void ensureStarted();
    [[nodiscard]] AdmissionResult tryEnqueue(WorkItem &item);
    [[noreturn]] void throwSubmissionUnavailable() const;
    void signalWorkAvailable() noexcept;
    void waitForSpaceChange(uint64_t spaceEpoch) noexcept;
    void signalSpaceAvailable() noexcept;
    void closeAdmissionAndQuiesceProducer() noexcept;
    void workerMain() noexcept;
    void cancelQueued(const std::exception_ptr &reason) noexcept;
    void recordFatalFailure(std::exception_ptr failure) noexcept;
    void releasePayload(uint64_t bytes) noexcept;
    void updateQueueHighWater(size_t depth) noexcept;
    void updatePayloadHighWater(uint64_t bytes) noexcept;

    Vu1CommandProcessor &m_processor;
    ThreadedVu1ExecutorOptions m_options;
    BoundedSpscQueue<WorkItem> m_queue;

    mutable std::mutex m_submitMutex;
    uint64_t m_generation = 1u;
    uint64_t m_nextSequence = 1u;
    uint64_t m_nextTicket = 0u;
    uint64_t m_lastGuestTick = 0u;
    std::optional<ProducerSpeculationCheckpoint>
        m_producerSpeculationCheckpoint;
    std::shared_ptr<Vu1ExecutionEpochState>
        m_activeExecutionEpoch;

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
    std::atomic<uint64_t> m_workEpoch{0u};
    std::atomic<bool> m_workWaitRequired{false};
    std::atomic<uint64_t> m_spaceEpoch{0u};
    std::atomic<bool> m_spaceWaitRequired{false};
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
    std::atomic<uint64_t> m_workNotificationCount{0u};
    std::atomic<uint64_t> m_workerWakeCount{0u};
    std::atomic<uint64_t> m_workerDrainCount{0u};
    std::atomic<uint64_t> m_resultWaitCount{0u};
    std::atomic<uint64_t> m_resultWaitNanoseconds{0u};
    std::atomic<uint64_t>
        m_submittedDecodedUnpackOperations{0u};
    std::atomic<uint64_t>
        m_executedDecodedUnpackOperations{0u};
    std::atomic<uint64_t> m_executionEpochsSubmitted{0u};
    std::atomic<uint64_t> m_executionEpochsCompleted{0u};
    std::atomic<uint64_t> m_epochCheckpointsProduced{0u};
    std::atomic<uint64_t> m_epochCheckpointsPublished{0u};
    std::atomic<uint64_t>
        m_epochSuffixCheckpointsDiscarded{0u};
    std::atomic<uint64_t> m_epochDemandRebases{0u};
    std::atomic<uint64_t> m_epochPositiveDemandExtensions{0u};
    std::atomic<uint64_t>
        m_epochPositiveDemandPrefixCyclesReused{0u};
    std::atomic<uint64_t>
        m_epochPositiveDemandExtensionCycles{0u};
    std::atomic<uint64_t>
        m_epochPositiveDemandExtensionExecutedCycles{0u};
    std::atomic<uint64_t> m_epochCheckpointWaitCount{0u};
    std::atomic<uint64_t>
        m_epochCheckpointWaitNanoseconds{0u};
    std::atomic<size_t> m_epochMaximumJournalDepth{0u};
    std::array<std::atomic<uint64_t>, kVu1CommandTypeCount>
        m_submittedCommandsByType{};
    std::array<std::atomic<uint64_t>, kVu1CommandTypeCount>
        m_completedCommandsByType{};
    std::array<std::atomic<uint64_t>, kVu1CommandTypeCount>
        m_resultWaitCountByType{};
    std::array<std::atomic<uint64_t>, kVu1CommandTypeCount>
        m_resultWaitNanosecondsByType{};
};

#endif
