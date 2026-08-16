#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ps2x::timing
{
    struct Cop0TimingAdvanceResult
    {
        bool timerRaised = false;
        uint8_t performanceOverflowMask = 0u;

        void merge(Cop0TimingAdvanceResult other) noexcept
        {
            timerRaised = timerRaised || other.timerRaised;
            performanceOverflowMask |=
                other.performanceOverflowMask;
        }
    };

    struct Cop0TimingSnapshot
    {
        uint32_t count = 0u;
        uint32_t compare = 0u;
        uint32_t pccr = 0u;
        std::array<uint32_t, 2u> pcr{};
        uint64_t lastCountCycle = 0u;
        std::array<uint64_t, 2u> lastPerformanceCycle{};
        bool timerArmed = false;
        bool timerPending = false;
    };

    // Occurrences observed at the EE issue or completion boundary.  Event
    // numbers are counter-specific, so keep the semantic identities here and
    // let Cop0Timing select the field for PCR0/PCR1.
    struct Cop0PerformanceEvents
    {
        uint32_t lowOrderBranchIssued = 0u;
        uint32_t singleInstructionIssued = 0u;
        uint32_t dualInstructionIssued = 0u;
        uint32_t branchIssued = 0u;
        uint32_t branchMispredicted = 0u;
        uint32_t instructionCompleted = 0u;
        uint32_t nonBdsInstructionCompleted = 0u;
        uint32_t cop2InstructionCompleted = 0u;
        uint32_t cop1InstructionCompleted = 0u;
        uint32_t loadCompleted = 0u;
        uint32_t storeCompleted = 0u;
    };

    // Runtime-owned timing state for the R5900 Count/Compare timer and the
    // two performance counters selected through COP0 register 25.
    class Cop0Timing
    {
    public:
        void reset(
            uint64_t currentCycle = 0u,
            uint32_t count = 0u,
            uint32_t compare = 0u,
            uint32_t pccr = 0u,
            uint32_t pcr0 = 0u,
            uint32_t pcr1 = 0u) noexcept;

        Cop0TimingAdvanceResult synchronizeCount(
            uint64_t currentCycle,
            bool minimumIncrement = false) noexcept;
        [[nodiscard]] uint32_t readCount(
            uint64_t currentCycle,
            Cop0TimingAdvanceResult *advance = nullptr) noexcept;
        void writeCount(
            uint32_t value,
            uint64_t currentCycle) noexcept;
        void writeCompare(
            uint32_t value,
            uint64_t currentCycle) noexcept;

        Cop0TimingAdvanceResult synchronizePerformance(
            uint64_t currentCycle,
            uint32_t status,
            bool minimumIncrement = false) noexcept;
        Cop0TimingAdvanceResult recordPerformanceEvents(
            uint32_t status,
            const Cop0PerformanceEvents &events) noexcept;
        [[nodiscard]] uint32_t readPerformance(
            uint32_t selector,
            uint64_t currentCycle,
            uint32_t status,
            Cop0TimingAdvanceResult *advance = nullptr) noexcept;
        [[nodiscard]] bool writePerformance(
            uint32_t selector,
            uint32_t value,
            uint64_t currentCycle,
            uint32_t status,
            Cop0TimingAdvanceResult *advance = nullptr) noexcept;

        [[nodiscard]] std::optional<uint64_t>
        nextTimerCycle() const noexcept;
        [[nodiscard]] std::optional<uint64_t>
        nextPerformanceEventCycle(
            uint64_t currentCycle,
            uint32_t status) const noexcept;
        [[nodiscard]] bool performanceExceptionPending(
            uint32_t status) const noexcept;
        [[nodiscard]] Cop0TimingSnapshot snapshot() const noexcept;
        void restore(const Cop0TimingSnapshot &snapshot) noexcept;

    private:
        [[nodiscard]] static bool performanceCounterEnabled(
            uint32_t pccr,
            size_t counter,
            uint32_t status) noexcept;
        [[nodiscard]] static bool supportedPerformanceEvent(
            size_t counter,
            uint32_t event) noexcept;
        [[nodiscard]] static uint32_t performanceEventOccurrences(
            size_t counter,
            uint32_t event,
            const Cop0PerformanceEvents &events) noexcept;
        [[nodiscard]] static uint32_t performanceEvent(
            uint32_t pccr,
            size_t counter) noexcept;
        Cop0TimingAdvanceResult incrementPerformanceCounter(
            size_t counter,
            uint64_t increment) noexcept;
        [[nodiscard]] static uint64_t saturatingAdd(
            uint64_t lhs,
            uint64_t rhs) noexcept;

        uint32_t m_count = 0u;
        uint32_t m_compare = 0u;
        uint32_t m_pccr = 0u;
        std::array<uint32_t, 2u> m_pcr{};
        uint64_t m_lastCountCycle = 0u;
        std::array<uint64_t, 2u> m_lastPerformanceCycle{};
        bool m_timerArmed = false;
        bool m_timerPending = false;
    };
}
