#ifndef PS2_VU1_COMMAND_STREAM_H
#define PS2_VU1_COMMAND_STREAM_H

#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu_program_cache.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
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
    MicroMemoryWrite,
    DataMemoryWrite,
    DecodedUnpack,
    VifStateUpdate,
    RegisterWrite,
    Mscal,
    Mscalf,
    Mscnt,
    AdvanceSlice,
    Reset,
    Snapshot,
    Restore,
    Barrier,
    Shutdown,
};

[[nodiscard]] const char *vu1CommandTypeName(
    Vu1CommandType type) noexcept;

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

// The EE-side VIF parser owns byte-stream framing. This value owns the
// complete decoded UNPACK input and enough alignment metadata for the VU1
// owner to reproduce mutation without retaining a pointer into DMA storage.
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
};

struct Vu1MscalfCommand
{
    uint32_t startPc = 0u;
    uint32_t top = 0u;
    uint32_t itop = 0u;
};

struct Vu1MscntCommand
{
    uint32_t top = 0u;
    uint32_t itop = 0u;
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
};

struct Vu1Snapshot
{
    VuExecutionState state{};
    std::vector<uint8_t> microMemory;
    std::vector<uint8_t> dataMemory;
    Vu1VifState vif{};
    uint64_t codeGeneration = 0u;
};

struct Vu1RestoreCommand
{
    Vu1Snapshot snapshot{};
};

struct Vu1BarrierCommand
{
};

struct Vu1ShutdownCommand
{
};

using Vu1CommandPayload = std::variant<
    Vu1MicroMemoryWriteCommand,
    Vu1DataMemoryWriteCommand,
    Vu1DecodedUnpackCommand,
    Vu1VifStateUpdateCommand,
    Vu1RegisterWriteCommand,
    Vu1MscalCommand,
    Vu1MscalfCommand,
    Vu1MscntCommand,
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

struct Vu1NoResult
{
};

struct Vu1SliceResult
{
    VuRunResult run{};
    std::vector<Vu1Path1Packet> path1Packets;
    std::optional<VuExecutionState> state;
    uint64_t architecturalStateHash = 0u;
    bool vifCanResume = false;
    std::string fault;
};

struct Vu1SnapshotResult
{
    Vu1Snapshot snapshot{};
    uint64_t architecturalStateHash = 0u;
};

struct Vu1BarrierResult
{
    uint64_t architecturalStateHash = 0u;
};

using Vu1CommandResultPayload = std::variant<
    Vu1NoResult,
    Vu1SliceResult,
    Vu1SnapshotResult,
    Vu1BarrierResult>;

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
    Vu1CommandResultPayload payload = Vu1NoResult{};
};

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

struct Vu1CommandProcessorConfiguration
{
    bool captureCommandDigests = false;
    bool captureArchitecturalStateHashes = false;
};

// This is the sole semantic entry point for VU1 owner-facing work. Memory is
// externally allocated in inline mode so M6 remains a seam-only change; M7
// can move those allocations without changing command semantics.
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

    [[nodiscard]] Vu1CommandResult process(
        Vu1Command command);
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
    [[nodiscard]] bool validMemoryRange(
        uint32_t offset, size_t size,
        uint32_t memorySize, bool wrap) const noexcept;
    void writeMemory(
        uint8_t *destination, uint32_t destinationSize,
        uint32_t offset, std::span<const uint8_t> bytes,
        bool wrap);
    [[nodiscard]] Vu1Snapshot captureSnapshot() const;
    [[nodiscard]] uint64_t stateHash() const noexcept;
    [[nodiscard]] Vu1CommandResultPayload apply(
        const Vu1CommandPayload &payload,
        Vu1CommandDisposition &disposition);

    VuUnit &m_unit;
    Vu1CommandProcessorConfiguration m_configuration{};
    uint8_t *m_microMemory = nullptr;
    uint32_t m_microMemorySize = 0u;
    uint8_t *m_dataMemory = nullptr;
    uint32_t m_dataMemorySize = 0u;
    std::atomic<uint64_t> m_codeGeneration{0u};
    IVuExecutionObserver *m_diagnosticsObserver = nullptr;
    Vu1VifState m_vifState{};
    uint64_t m_generation = 1u;
    uint64_t m_nextSequence = 1u;
    bool m_shutdown = false;
};

class Vu1CommandExecutor
{
public:
    virtual ~Vu1CommandExecutor() = default;
    [[nodiscard]] virtual Vu1CommandResult submit(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) = 0;
    [[nodiscard]] virtual uint64_t generation() const noexcept = 0;
};

class InlineVu1Executor final : public Vu1CommandExecutor
{
public:
    explicit InlineVu1Executor(Vu1CommandProcessor &processor);

    [[nodiscard]] Vu1CommandResult submit(
        Vu1CommandPayload payload,
        uint64_t guestTick = 0u,
        uint64_t publicationToken = 0u) override;
    [[nodiscard]] uint64_t generation() const noexcept override
    {
        return m_processor.generation();
    }

private:
    Vu1CommandProcessor &m_processor;
    uint64_t m_nextSequence = 1u;
};

#endif
