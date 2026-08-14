#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class GS;
class PS2Memory;
class VuProgramCache;
class VuRecompilerBackend;
enum class VuUnitId : uint8_t;
struct VuExecutionState;
struct VU1NativeEmitterSpikeAccess;
struct VuVerifyTestAccess;
struct VuXgkickTestAccess;

// Diagnostics are intentionally separated from VU execution ownership. A
// threaded owner can use a worker-local recorder, while the inline runtime can
// adapt its existing trace/profile implementation without exposing PS2Memory
// to the execution backend.
class IVuExecutionObserver
{
public:
    virtual ~IVuExecutionObserver() = default;

    [[nodiscard]] virtual bool vuTraceEnabled() const
    {
        return false;
    }
    [[nodiscard]] virtual bool vuWorkloadProfileEnabled() const
    {
        return false;
    }
    [[nodiscard]] virtual uint64_t currentVuCodeGeneration(
        VuUnitId) const
    {
        return 0u;
    }
    virtual void traceVuInvocation(
        uint32_t, uint32_t, uint32_t, bool,
        const VuExecutionState &)
    {
    }
    virtual void traceVuInstruction(
        uint32_t, uint32_t, uint32_t,
        const VuExecutionState &)
    {
    }
    virtual void traceVuXgkick(uint32_t)
    {
    }
    virtual void traceVuInvocationEnd(
        uint32_t, bool, bool, const int32_t *, size_t)
    {
    }
    virtual void beginVuWorkloadProfileInvocation(
        uint32_t, const uint8_t *, uint32_t, uint64_t)
    {
    }
    virtual void recordVuWorkloadProfileInstruction(
        uint32_t, uint32_t, uint32_t)
    {
    }
    virtual void recordVuWorkloadProfileTransition(
        uint32_t, uint32_t)
    {
    }
    virtual void endVuWorkloadProfileInvocation(bool)
    {
    }
    virtual void resetVuWorkloadProfileEpoch()
    {
    }
};

// XGKICK builds a PATH1 packet incrementally while VU memory remains live.
// The backing storage may be prepared for a complete GIF tag, but size()
// exposes only bytes earned at the current architectural cycle. The three
// native fields form an explicit generated-code cursor; copy/move operations
// always rebase nativeData onto the destination storage.
struct VuXgkickPacket
{
    std::vector<uint8_t> storage;
    uint8_t *nativeData = nullptr;
    uint32_t nativeSize = 0u;
    uint32_t nativePreparedBytes = 0u;

    VuXgkickPacket() = default;

    VuXgkickPacket(std::initializer_list<uint8_t> bytes)
        : storage(bytes),
          nativeSize(static_cast<uint32_t>(bytes.size())),
          nativePreparedBytes(
              static_cast<uint32_t>(bytes.size()))
    {
        rebase();
    }

    VuXgkickPacket(const VuXgkickPacket &other)
        : storage(other.storage),
          nativeSize(other.nativeSize),
          nativePreparedBytes(other.nativePreparedBytes)
    {
        rebase();
    }

    VuXgkickPacket(VuXgkickPacket &&other) noexcept
        : storage(std::move(other.storage)),
          nativeSize(other.nativeSize),
          nativePreparedBytes(other.nativePreparedBytes)
    {
        rebase();
        other.nativeData = nullptr;
        other.nativeSize = 0u;
        other.nativePreparedBytes = 0u;
    }

    VuXgkickPacket &operator=(
        const VuXgkickPacket &other)
    {
        if (this == &other)
            return *this;
        storage = other.storage;
        nativeSize = other.nativeSize;
        nativePreparedBytes =
            other.nativePreparedBytes;
        rebase();
        return *this;
    }

    VuXgkickPacket &operator=(
        VuXgkickPacket &&other) noexcept
    {
        if (this == &other)
            return *this;
        storage = std::move(other.storage);
        nativeSize = other.nativeSize;
        nativePreparedBytes =
            other.nativePreparedBytes;
        rebase();
        other.nativeData = nullptr;
        other.nativeSize = 0u;
        other.nativePreparedBytes = 0u;
        return *this;
    }

    VuXgkickPacket &operator=(
        std::initializer_list<uint8_t> bytes)
    {
        if (bytes.size() >
            std::numeric_limits<uint32_t>::max())
        {
            throw std::length_error(
                "XGKICK packet exceeds native cursor range");
        }
        storage.assign(bytes);
        nativeSize =
            static_cast<uint32_t>(bytes.size());
        nativePreparedBytes = nativeSize;
        rebase();
        return *this;
    }

    void prepareAppend(size_t byteCount)
    {
        if (byteCount >
            std::numeric_limits<uint32_t>::max() -
                nativeSize)
        {
            throw std::length_error(
                "XGKICK packet exceeds native cursor range");
        }
        const uint32_t required =
            nativeSize +
            static_cast<uint32_t>(byteCount);
        if (storage.size() < required)
            storage.resize(required);
        nativePreparedBytes =
            static_cast<uint32_t>(storage.size());
        rebase();
    }

    void resize(size_t byteCount)
    {
        if (byteCount >
            std::numeric_limits<uint32_t>::max())
        {
            throw std::length_error(
                "XGKICK packet exceeds native cursor range");
        }
        const uint32_t requested =
            static_cast<uint32_t>(byteCount);
        if (storage.size() < requested)
            storage.resize(requested);
        nativeSize = requested;
        nativePreparedBytes =
            static_cast<uint32_t>(storage.size());
        rebase();
    }

    void clear() noexcept
    {
        storage.clear();
        nativeData = nullptr;
        nativeSize = 0u;
        nativePreparedBytes = 0u;
    }

    [[nodiscard]] uint8_t *data() noexcept
    {
        return nativeData;
    }

    [[nodiscard]] const uint8_t *data() const noexcept
    {
        return nativeData;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return nativeSize;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return nativeSize == 0u;
    }

    [[nodiscard]] uint8_t &operator[](size_t index)
    {
        return storage[index];
    }

    [[nodiscard]] const uint8_t &operator[](
        size_t index) const
    {
        return storage[index];
    }

    [[nodiscard]] auto begin() noexcept
    {
        return storage.begin();
    }

    [[nodiscard]] auto end() noexcept
    {
        return storage.begin() + nativeSize;
    }

    [[nodiscard]] auto begin() const noexcept
    {
        return storage.begin();
    }

    [[nodiscard]] auto end() const noexcept
    {
        return storage.begin() + nativeSize;
    }

    bool operator==(
        const VuXgkickPacket &other) const
    {
        return
            nativeSize == other.nativeSize &&
            std::equal(
                begin(), end(), other.begin());
    }

private:
    void rebase() noexcept
    {
        nativeData =
            nativePreparedBytes != 0u
                ? storage.data()
                : nullptr;
    }
};

struct VuPipelineState
{
    struct Xgkick
    {
        bool active = false;
        uint32_t address = 0;
        uint32_t tagBytesRemaining = 0;
        uint32_t cycleCredit = 0;
        bool tagEop = false;
        VuXgkickPacket packet;
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
        bool fmacActive = false;
        bool clipActive = false;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t clip = 0;
    };

    static constexpr uint8_t kFmacFlagStages = 4u;

    Xgkick xgkick;
    DelayedQ delayedQ;
    DelayedP delayedP;
    std::array<FmacFlags, kFmacFlagStages> fmacFlags{};
    uint8_t fmacFlagIndex = 0u;
    uint32_t workingMac = 0u;
    uint32_t workingClip = 0u;
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

struct VuVerifyDiagnostics
{
    uint64_t runs = 0u;
    uint64_t comparedPairs = 0u;
    uint64_t publishedPairs = 0u;
    uint64_t publishedPath1Packets = 0u;
    uint64_t mismatches = 0u;
    std::string lastMismatch;
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

    [[nodiscard]] std::vector<std::vector<uint8_t>>
    takePath1Packets()
    {
        return std::move(m_path1Packets);
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
    // Explicit cache identity replaces backend discovery through PS2Memory.
    // memoryIdentity names the canonical VU-memory owner, while codeIdentity
    // names this stable code allocation within that owner.
    VuUnitId codeUnit = static_cast<VuUnitId>(0);
    uintptr_t memoryIdentity = 0u;
    uintptr_t codeIdentity = 0u;
    uint64_t codeGeneration = 0u;
    bool trackedCode = false;
    IVuExecutionObserver *observer = nullptr;
    bool traceBudgetBoundary = false;
    bool enableInstrumentation = true;
    // Opt-in CI/test path. When an observer is armed, compile a distinct
    // helper-routed native block instead of using the permanent interpreter.
    bool enableNativeInstrumentation = false;
    // False when an outer unit adapter owns aggregate progress for nested
    // backend entries and interpreter side exits.
    bool enableProgressAccounting = true;
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
               uint32_t itop = 0,
               IVuExecutionObserver *observer = nullptr);
    void resumeState(uint32_t top = 0, uint32_t itop = 0,
                     IVuExecutionObserver *observer = nullptr);
    [[nodiscard]] VuRunResult advance(
        VuExecutionContext &context,
        uint32_t maxCycles = 65536);
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
    void setNativeInstrumentationEnabled(bool enabled)
    {
        m_nativeInstrumentationEnabled = enabled;
    }
    bool nativeInstrumentationEnabled() const
    {
        return m_nativeInstrumentationEnabled;
    }

    bool setBackend(
        VuBackendKind requested, std::string *diagnostic = nullptr);
    VuBackendKind requestedBackend() const { return m_requestedBackend; }
    VuBackendKind resolvedBackend() const { return m_resolvedBackend; }
    std::string_view backendName() const;
    VuExitReason lastExitReason() const { return m_lastExitReason; }
    VuUnitId unitId() const { return m_unitId; }
    VuProgramCache &programCache();
    const VuProgramCache *programCacheIfCreated() const
    {
        return m_programCache.get();
    }
    const VuRecompilerDiagnostics *
    recompilerDiagnosticsIfCreated() const;
    const VuRecompilerBackend *recompilerIfCreated() const
    {
        return m_recompiler.get();
    }
    const VuVerifyDiagnostics &verifyDiagnostics() const
    {
        return m_verifyDiagnostics;
    }

private:
    friend class VuInterpreterBackend;
    friend class VuRecompilerBackend;
    friend struct VU1NativeEmitterSpikeAccess;
    friend struct VuVerifyTestAccess;

    class ProgressTracker
    {
    public:
        ProgressTracker(
            VuUnit &unit, VuExecutionState &state,
            bool enabled, uint32_t &executedCycles);
        ~ProgressTracker();

        void publish();

        ProgressTracker(const ProgressTracker &) = delete;
        ProgressTracker &operator=(const ProgressTracker &) = delete;

    private:
        VuUnit &m_unit;
        VuExecutionState &m_state;
        uint32_t &m_executedCycles;
        uint32_t m_committedCycles = 0u;
        bool m_enabled = false;
    };

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
        uintptr_t memoryIdentity = 0u;
        uint32_t codeSize = 0;
        uint64_t codeGeneration = 0;
        bool valid = false;
    };

    struct VerifyMismatch
    {
        const VuExecutionState &beforeState;
        const VuExecutionState &referenceState;
        const VuExecutionState &nativeState;
        const VuRunResult &referenceResult;
        const VuRunResult &nativeResult;
        const std::vector<uint8_t> &referenceData;
        const std::vector<uint8_t> &nativeData;
        const VuTransactionalSideEffectSink &referenceEffects;
        const VuTransactionalSideEffectSink &nativeEffects;
        const uint8_t *code = nullptr;
        uint32_t codeSize = 0u;
        VuUnitId codeUnit = static_cast<VuUnitId>(0);
        uintptr_t memoryIdentity = 0u;
        uintptr_t codeIdentity = 0u;
        uint64_t codeGeneration = 0u;
        bool trackedCode = false;
        IVuExecutionObserver *observer = nullptr;
        uint32_t invocationEntryPc = 0u;
        uint32_t failingPc = 0u;
        uint32_t lowerWord = 0u;
        uint32_t upperWord = 0u;
        uint32_t cycleBudget = 0u;
        uint64_t pairIndex = 0u;
        std::string_view detail;
    };

    [[nodiscard]] VuRunResult run(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory, uint32_t maxCycles,
        bool traceBudgetBoundary);
    [[nodiscard]] VuRunResult runRecompilerContext(
        VuExecutionContext &context, uint32_t maxCycles,
        bool trackProgress);
    [[nodiscard]] VuRunResult runVerify(
        VuExecutionContext &context,
        uint32_t maxCycles);
    [[nodiscard]] std::string formatVerifyMismatch(
        const VerifyMismatch &mismatch);

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
    bool m_nativeInstrumentationEnabled = false;
    IVuExecutionObserver *m_workloadProfileObserver = nullptr;
    bool m_workloadProfileInvocationPending = false;
    bool m_workloadProfileInvocationActive = false;
    InterpreterCache m_interpreterCache;
    std::unique_ptr<VuInterpreterBackend> m_interpreter;
    std::unique_ptr<VuRecompilerBackend> m_recompiler;
    IVuExecutionBackend *m_backend = nullptr;
    VuBackendKind m_requestedBackend = VuBackendKind::Auto;
    VuBackendKind m_resolvedBackend = VuBackendKind::Interpreter;
    VuExitReason m_lastExitReason = VuExitReason::Inactive;
    VuVerifyDiagnostics m_verifyDiagnostics{};
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
    friend struct VuXgkickTestAccess;

    using DecodedInstructionPair = VuUnit::DecodedInstructionPair;

    VuUnit &m_unit;
    VuExecutionState *m_state = nullptr;
    uint32_t m_codeAddressMask = 0x3fffu;

    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                 uintptr_t memoryIdentity,
                                 uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(
        uint32_t instr, uint8_t *vuData, uint32_t dataSize,
        IVuSideEffectSink &sideEffects,
        IVuExecutionObserver *observer,
        uint32_t upperInstr);
    void beginXgkick(uint32_t sourceQword, uint8_t *vuData, uint32_t dataSize,
                     IVuSideEffectSink &sideEffects,
                     IVuExecutionObserver *observer);
    void advanceXgkick(uint8_t *vuData, uint32_t dataSize,
                       IVuSideEffectSink &sideEffects,
                       IVuExecutionObserver *observer,
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
    void scheduleClipFlags(uint32_t clip);
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
