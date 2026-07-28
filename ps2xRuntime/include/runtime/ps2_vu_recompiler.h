#ifndef PS2_VU_RECOMPILER_H
#define PS2_VU_RECOMPILER_H

#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu1.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

enum class VuNativeBlockExit : uint32_t
{
    BlockComplete,
    CycleBudget,
    Inactive,
    ProgramEnded,
    CodeBounds,
    XgkickBoundary,
    DebugObserver,
    CodeInvalidated,
    UnsupportedInstruction,
    Fault,
};

struct VuNativeBlockResult
{
    uint32_t executedCycles = 0u;
    VuNativeBlockExit exit = VuNativeBlockExit::Fault;

    [[nodiscard]] static VuNativeBlockResult decode(
        uint64_t packed)
    {
        return {
            .executedCycles =
                static_cast<uint32_t>(packed),
            .exit = static_cast<VuNativeBlockExit>(
                packed >> 32u),
        };
    }

    [[nodiscard]] uint64_t encode() const
    {
        return
            static_cast<uint64_t>(executedCycles) |
            (static_cast<uint64_t>(exit) << 32u);
    }
};

using VuNativeBlockEntry = uint64_t (*)(
    VuExecutionContext *context, uint32_t maximumCycles);

struct VuRecompilerDiagnostics
{
    uint64_t nativeEntries = 0u;
    uint64_t nativePairs = 0u;
    uint64_t inlinePairs = 0u;
    uint64_t helperPairs = 0u;
    uint64_t blockCompletes = 0u;
    uint64_t cycleBudgetExits = 0u;
    uint64_t xgkickExits = 0u;
    uint64_t xgkickAdvanceHelperCalls = 0u;
    uint64_t unsupportedExits = 0u;
    uint64_t codeInvalidationExits = 0u;
    uint64_t faultExits = 0u;
    uint64_t interpreterInstrumentationFallbacks = 0u;
    uint64_t interpreterFallbackPairs = 0u;
};

class VuRecompilerBackend final : public IVuExecutionBackend
{
public:
    explicit VuRecompilerBackend(VuUnit &unit);

    [[nodiscard]] VuRunResult run(
        VuExecutionContext &context,
        uint32_t maximumCycles) override;
    [[nodiscard]] std::string_view name() const override;

    [[nodiscard]] static bool built();
    [[nodiscard]] static bool supported();
    [[nodiscard]] static uint64_t hostFeatures();

    [[nodiscard]] bool programKey(
        const VuExecutionContext &context,
        VuCompilationMode mode,
        VuProgramKey &key,
        std::string *diagnostic = nullptr) const;

    [[nodiscard]] VuRecompilerDiagnostics diagnostics() const
    {
        return m_diagnostics;
    }
    [[nodiscard]] const std::string &lastDiagnostic() const
    {
        return m_lastDiagnostic;
    }

private:
    friend class VuUnit;

    static constexpr uint32_t kMaximumBlockPairs = 64u;
    static constexpr size_t kMaximumDispatchEntries =
        0x4000u / 8u;

    [[nodiscard]] uint32_t executePair(
        VuExecutionContext &context, uint32_t expectedPc,
        uint32_t lowerWord, uint32_t upperWord) noexcept;
    [[nodiscard]] static uint32_t executePairThunk(
        VuRecompilerBackend *backend,
        VuExecutionContext *context,
        uint32_t expectedPc,
        uint64_t instructionWords) noexcept;
    [[nodiscard]] uint32_t advanceXgkick(
        VuExecutionContext &context) noexcept;
    [[nodiscard]] static uint32_t advanceXgkickThunk(
        VuRecompilerBackend *backend,
        VuExecutionContext *context) noexcept;
    [[nodiscard]] VuProgramHandle lookupOrCompile(
        VuExecutionContext &context,
        const VuProgramKey &key);
    [[nodiscard]] bool compile(
        VuExecutionContext &context,
        const VuProgramKey &key,
        VuCompiledProgram &program,
        std::string &diagnostic);
    [[nodiscard]] bool needsInterpreterInstrumentation(
        const VuExecutionContext &context) const;
    [[nodiscard]] VuRunResult runInterpreterFallback(
        VuExecutionContext &context,
        uint32_t maximumCycles);
    [[nodiscard]] static VuExitReason publicExit(
        VuNativeBlockExit exit);

    VuUnit &m_unit;
    VuInterpreterBackend m_semantics;
    VuRecompilerDiagnostics m_diagnostics{};
    std::array<
        VuProgramHandle,
        kMaximumDispatchEntries> m_dispatchHandles{};
    uint64_t m_helperPairsIssued = 0u;
    std::string m_lastDiagnostic;
};

[[nodiscard]] std::string_view vuNativeBlockExitName(
    VuNativeBlockExit exit);

#endif
