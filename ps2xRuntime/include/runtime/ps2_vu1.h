#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class GS;
class PS2Memory;
class VuProgramCache;
class VuRecompilerBackend;
enum class VuUnitId : uint8_t;
struct VU1NativeEmitterSpikeAccess;

struct VuPipelineState
{
    struct Xgkick
    {
        bool active = false;
        uint32_t address = 0;
        uint32_t tagBytesRemaining = 0;
        uint32_t cycleCredit = 0;
        bool tagEop = false;
        std::vector<uint8_t> packet;
    };

    struct DelayedQ
    {
        bool active = false;
        float result = 0.0f;
        uint32_t cyclesRemaining = 0;
    };

    struct DelayedP
    {
        bool active = false;
        float result = 0.0f;
        uint32_t cyclesRemaining = 0;
    };

    struct FmacFlags
    {
        bool active = false;
        uint32_t mac = 0;
        uint32_t status = 0;
    };

    static constexpr uint8_t kFmacFlagStages = 4u;

    Xgkick xgkick;
    DelayedQ delayedQ;
    DelayedP delayedP;
    std::array<FmacFlags, kFmacFlagStages> fmacFlags{};
    uint8_t fmacFlagIndex = 0u;
    uint32_t workingMac = 0u;
};

struct VuExecutionState
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    bool ebit;
    uint32_t top;  // VIF1 TOP visible to VU1 XTOP
    uint32_t itop; // VIF1 ITOP visible to VU1 XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;

    uint8_t viBackupCycles;
    uint8_t viBackupRegister;
    int32_t viBackupValue;

    bool active;
    uint64_t issuedCycles;

    // Delayed architectural effects must travel with a cloned register state.
    // Unit-owned decode/native caches and diagnostics deliberately do not.
    VuPipelineState pipeline;
};

[[nodiscard]] bool vuExecutionStatesEqual(
    const VuExecutionState &left, const VuExecutionState &right,
    std::string *firstDifference = nullptr);

enum class VuBackendKind : uint8_t
{
    Auto,
    Interpreter,
    Recompiler,
    Verify,
};

enum class VuExitReason : uint8_t
{
    Inactive,
    CycleBudget,
    ProgramEnded,
    CodeBounds,
    BranchBoundary,
    XgkickBoundary,
    DebugObserver,
    CodeInvalidated,
    UnsupportedInstruction,
    Fault,
};

std::string_view vuBackendKindName(VuBackendKind kind);
bool parseVuBackendKind(std::string_view text, VuBackendKind &kind);
std::string_view vuExitReasonName(VuExitReason reason);

using VuInstructionObserver = std::function<void(
    uint64_t index, uint32_t pc, uint32_t lower, uint32_t upper,
    const VuExecutionState &state)>;

struct VuProgressSnapshot
{
    bool enabled = false;
    bool active = false;
    uint64_t invocations = 0;
    uint64_t cycles = 0;
    uint32_t pc = 0;
};

struct VuRunResult
{
    uint32_t requestedCycles = 0;
    uint32_t executedCycles = 0;
    VuExitReason reason = VuExitReason::Inactive;
    bool activeBefore = false;
    bool activeAfter = false;
    bool completed = false;
};

class IVuSideEffectSink
{
public:
    virtual ~IVuSideEffectSink() = default;
    virtual void submitPath1Packet(
        const uint8_t *data, uint32_t sizeBytes) = 0;
};

class VuTransactionalSideEffectSink final : public IVuSideEffectSink
{
public:
    void submitPath1Packet(
        const uint8_t *data, uint32_t sizeBytes) override;

    [[nodiscard]] const std::vector<std::vector<uint8_t>> &
    path1Packets() const
    {
        return m_path1Packets;
    }

    void clear();
    void commitTo(IVuSideEffectSink &destination) const;

private:
    std::vector<std::vector<uint8_t>> m_path1Packets;
};

struct VuExecutionContext
{
    VuExecutionState &state;
    const uint8_t *code = nullptr;
    uint32_t codeSize = 0;
    uint8_t *data = nullptr;
    uint32_t dataSize = 0;
    IVuSideEffectSink &sideEffects;
    PS2Memory *memory = nullptr;
    bool traceBudgetBoundary = false;
    bool enableInstrumentation = true;
};

class IVuExecutionBackend
{
public:
    virtual ~IVuExecutionBackend() = default;
    [[nodiscard]] virtual VuRunResult run(
        VuExecutionContext &context, uint32_t maxCycles) = 0;
    [[nodiscard]] virtual std::string_view name() const = 0;
};

class VuInterpreterBackend;
class VuRecompilerBackend;
struct VuRecompilerDiagnostics;

class VuUnit
{
public:
    VuUnit();
    explicit VuUnit(VuUnitId unit);
    ~VuUnit();

    void reset();

    // Starting/resuming and advancing are separate so a runtime scheduler can
    // make busy state visible before any VU work is performed.
    void start(uint32_t startPC = 0, uint32_t top = 0,
               uint32_t itop = 0, PS2Memory *memory = nullptr);
    void resumeState(uint32_t top = 0, uint32_t itop = 0,
                     PS2Memory *memory = nullptr);
    [[nodiscard]] VuRunResult advance(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory = nullptr,
        uint32_t maxCycles = 65536);

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VuRunResult continueExecution(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory = nullptr,
        uint32_t maxCycles = 65536);

    VuExecutionState &state() { return m_state; }
    const VuExecutionState &state() const { return m_state; }
    bool isActive() const { return m_state.active; }
    VuProgressSnapshot getProgressSnapshot() const;
    void setProgressTrackingEnabled(bool enabled);
    void setInstructionObserver(VuInstructionObserver observer);
    void setInstructionObserverEnabled(bool enabled);

    bool setBackend(
        VuBackendKind requested, std::string *diagnostic = nullptr);
    VuBackendKind requestedBackend() const { return m_requestedBackend; }
    VuBackendKind resolvedBackend() const { return m_resolvedBackend; }
    std::string_view backendName() const;
    VuUnitId unitId() const { return m_unitId; }
    VuProgramCache &programCache();
    const VuProgramCache *programCacheIfCreated() const
    {
        return m_programCache.get();
    }
    const VuRecompilerDiagnostics *
    recompilerDiagnosticsIfCreated() const;

private:
    friend class VuInterpreterBackend;
    friend class VuRecompilerBackend;
    friend struct VU1NativeEmitterSpikeAccess;

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        bool iBit = false;
        bool eBit = false;
        bool lowerBeforeUpper = false;
    };

    struct InterpreterCache
    {
        std::vector<DecodedInstructionPair> decodedCode;
        const uint8_t *vuCode = nullptr;
        const PS2Memory *memory = nullptr;
        uint32_t codeSize = 0;
        uint64_t codeGeneration = 0;
        bool valid = false;
    };

    [[nodiscard]] VuRunResult run(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory, uint32_t maxCycles,
        bool traceBudgetBoundary);
    [[nodiscard]] VuRunResult runRecompiler(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory, uint32_t maxCycles,
        bool traceBudgetBoundary);

    VuUnitId m_unitId;
    std::unique_ptr<VuProgramCache> m_programCache;
    VuExecutionState m_state{};
    std::atomic<bool> m_progressTrackingEnabled{false};
    std::atomic<uint32_t> m_progressActive{0};
    std::atomic<uint64_t> m_progressInvocations{0};
    std::atomic<uint64_t> m_progressCycles{0};
    std::atomic<uint32_t> m_progressPc{0};
    VuInstructionObserver m_instructionObserver;
    std::atomic<bool> m_instructionObserverEnabled{false};
    PS2Memory *m_workloadProfileMemory = nullptr;
    bool m_workloadProfileInvocationPending = false;
    bool m_workloadProfileInvocationActive = false;
    InterpreterCache m_interpreterCache;
    std::unique_ptr<VuInterpreterBackend> m_interpreter;
    std::unique_ptr<VuRecompilerBackend> m_recompiler;
    IVuExecutionBackend *m_backend = nullptr;
    VuBackendKind m_requestedBackend = VuBackendKind::Auto;
    VuBackendKind m_resolvedBackend = VuBackendKind::Interpreter;
};

class VuInterpreterBackend final : public IVuExecutionBackend
{
public:
    explicit VuInterpreterBackend(VuUnit &unit);

    [[nodiscard]] VuRunResult run(
        VuExecutionContext &context, uint32_t maxCycles) override;
    [[nodiscard]] std::string_view name() const override;

private:
    friend class VuIrInterpreterBackend;
    friend class VuRecompilerBackend;
    friend struct VU1NativeEmitterSpikeAccess;

    using DecodedInstructionPair = VuUnit::DecodedInstructionPair;

    VuUnit &m_unit;
    VuExecutionState *m_state = nullptr;

    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                 const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(
        uint32_t instr, uint8_t *vuData, uint32_t dataSize,
        IVuSideEffectSink &sideEffects, PS2Memory *memory,
        uint32_t upperInstr);
    void beginXgkick(uint32_t sourceQword, uint8_t *vuData, uint32_t dataSize,
                     IVuSideEffectSink &sideEffects, PS2Memory *memory);
    void advanceXgkick(uint8_t *vuData, uint32_t dataSize,
                       IVuSideEffectSink &sideEffects, PS2Memory *memory,
                       uint32_t cycles, bool flush);
    void cancelXgkick();
    void advanceQPipeline();
    void flushQPipeline();
    void scheduleQ(float result, uint32_t latency);
    void advancePPipeline();
    void flushPPipeline();
    void scheduleP(float result, uint32_t latency);
    void advanceFmacFlagPipeline();
    void flushFmacFlagPipeline();
    void commitFmacFlags(const VuPipelineState::FmacFlags &pending);
    void scheduleFmacFlags(const float result[4], uint8_t dest,
                           bool preserveUnselected);
    void backupVi(uint8_t reg);
    int32_t readViForBranch(uint8_t reg) const;

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    float broadcast(const float *vf, uint8_t bc);
};

// Development-only semantic oracle. It executes control, ordering, and
// pipeline sequencing from VuIrInstructionPair while reusing the permanent
// interpreter's proven opcode helpers.
class VuIrInterpreterBackend final : public IVuExecutionBackend
{
public:
    explicit VuIrInterpreterBackend(VuUnit &unit);

    [[nodiscard]] VuRunResult run(
        VuExecutionContext &context, uint32_t maxCycles) override;
    [[nodiscard]] std::string_view name() const override;

private:
    VuInterpreterBackend m_semantics;
};

#endif
