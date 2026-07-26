#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

class GS;
class PS2Memory;

struct VU1State
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
};

struct VU1ProgressSnapshot
{
    bool enabled = false;
    bool active = false;
    uint64_t invocations = 0;
    uint64_t cycles = 0;
    uint32_t pc = 0;
};

struct VU1AdvanceResult
{
    uint32_t requestedCycles = 0;
    uint32_t executedCycles = 0;
    bool activeBefore = false;
    bool activeAfter = false;
    bool completed = false;
};

class VU1Interpreter
{
public:
    using InstructionObserver = std::function<void(
        uint64_t index, uint32_t pc, uint32_t lower, uint32_t upper,
        const VU1State &state)>;

    VU1Interpreter();

    void reset();

    // Starting/resuming and advancing are separate so a runtime scheduler can
    // make busy state visible before any VU work is performed.
    void start(uint32_t startPC = 0, uint32_t top = 0,
               uint32_t itop = 0, PS2Memory *memory = nullptr);
    void resumeState(uint32_t top = 0, uint32_t itop = 0,
                     PS2Memory *memory = nullptr);
    [[nodiscard]] VU1AdvanceResult advance(
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

    VU1AdvanceResult continueExecution(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory = nullptr,
        uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }
    bool isActive() const { return m_active; }
    VU1ProgressSnapshot getProgressSnapshot() const;
    void setProgressTrackingEnabled(bool enabled);
    void setInstructionObserver(InstructionObserver observer);
    void setInstructionObserverEnabled(bool enabled);

private:
    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        bool iBit = false;
        bool eBit = false;
        bool lowerBeforeUpper = false;
    };

    VU1State m_state;

    struct XgkickState
    {
        bool active = false;
        uint32_t address = 0;
        uint32_t tagBytesRemaining = 0;
        uint32_t cycleCredit = 0;
        bool tagEop = false;
        std::vector<uint8_t> packet;
    };

    XgkickState m_xgkick;
    struct QPipelineState
    {
        bool active = false;
        float result = 0.0f;
        uint32_t cyclesRemaining = 0;
    };

    QPipelineState m_qPipeline;
    struct FmacFlagPipelineState
    {
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t cyclesRemaining = 0;
    };

    std::vector<FmacFlagPipelineState> m_fmacFlagPipeline;
    uint32_t m_workingMac = 0;
    std::vector<DecodedInstructionPair> m_decodedCodeCache;
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;
    std::atomic<bool> m_progressTrackingEnabled{false};
    std::atomic<uint32_t> m_progressActive{0};
    std::atomic<uint64_t> m_progressInvocations{0};
    std::atomic<uint64_t> m_progressCycles{0};
    std::atomic<uint32_t> m_progressPc{0};
    InstructionObserver m_instructionObserver;
    std::atomic<bool> m_instructionObserverEnabled{false};
    bool m_active = false;

    [[nodiscard]] VU1AdvanceResult run(
        uint8_t *vuCode, uint32_t codeSize,
        uint8_t *vuData, uint32_t dataSize,
        GS &gs, PS2Memory *memory, uint32_t maxCycles,
        bool traceBudgetBoundary);

    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    DecodedInstructionPair getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize,
                                                          PS2Memory *memory, uint32_t pc);
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                 const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);
    void beginXgkick(uint32_t sourceQword, uint8_t *vuData, uint32_t dataSize,
                     GS &gs, PS2Memory *memory);
    void advanceXgkick(uint8_t *vuData, uint32_t dataSize,
                       GS &gs, PS2Memory *memory, uint32_t cycles, bool flush);
    void cancelXgkick();
    void advanceQPipeline();
    void flushQPipeline();
    void scheduleQ(float result, uint32_t latency);
    void advanceFmacFlagPipeline();
    void flushFmacFlagPipeline();
    void scheduleFmacFlags(const float result[4], uint8_t dest,
                           bool preserveUnselected);
    void backupVi(uint8_t reg);
    int32_t readViForBranch(uint8_t reg) const;

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
