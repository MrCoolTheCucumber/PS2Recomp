#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ps2x::timing
{
    inline constexpr uint32_t kEeCounterBaseAddress = 0x10000000u;
    inline constexpr uint32_t kEeCounterStride = 0x800u;
    inline constexpr size_t kEeCounterCount = 4u;

    enum class EeCounterRegister : uint32_t
    {
        Count = 0x00u,
        Mode = 0x10u,
        Target = 0x20u,
        Hold = 0x30u,
    };

    struct EeCounterVideoTiming
    {
        uint64_t renderCycles = 4'493'898u;
        uint64_t blankCycles = 421'302u;
        uint32_t hRenderCycles = 15'669u;
        uint32_t hBlankCycles = 3'055u;
        uint32_t hSyncErrorCycles = 149u;
    };

    enum class EeCounterSignalPhase : uint8_t
    {
        Render = 0u,
        Blank,
    };

    struct EeCounterSnapshot
    {
        uint32_t count = 0u;
        uint16_t mode = 0u;
        uint16_t target = 0xffffu;
        uint16_t hold = 0u;
        uint32_t rate = 2u;
        uint64_t startCycle = 0u;
        bool targetAfterOverflow = false;
    };

    struct EeCounterBankSnapshot
    {
        std::array<EeCounterSnapshot, kEeCounterCount> counters{};
        uint64_t currentCycle = 0u;
        uint64_t hSyncStartCycle = 0u;
        uint64_t vSyncStartCycle = 0u;
        uint32_t hSyncDeltaCycles = 0u;
        uint64_t vSyncDeltaCycles = 0u;
        EeCounterSignalPhase hSyncPhase =
            EeCounterSignalPhase::Render;
        EeCounterSignalPhase vSyncPhase =
            EeCounterSignalPhase::Render;
    };

    struct EeCounterAdvanceResult
    {
        // INTC timer causes are bits 9 through 12.
        uint32_t newlyRaisedInterruptMask = 0u;
        uint64_t hBlankStarts = 0u;
        uint64_t vBlankStarts = 0u;
    };

    class EeCounterBank
    {
    public:
        EeCounterBank() noexcept;

        void reset(uint64_t currentCycle = 0u) noexcept;
        EeCounterAdvanceResult configureVideoTiming(
            EeCounterVideoTiming timing,
            uint64_t currentCycle) noexcept;
        EeCounterAdvanceResult advanceTo(
            uint64_t currentCycle) noexcept;

        [[nodiscard]] bool readRegister(
            uint32_t address,
            uint64_t currentCycle,
            uint32_t &value,
            EeCounterAdvanceResult *advance = nullptr) noexcept;
        [[nodiscard]] bool writeRegister(
            uint32_t address,
            uint32_t value,
            uint64_t currentCycle,
            EeCounterAdvanceResult *advance = nullptr) noexcept;

        [[nodiscard]] std::optional<uint64_t>
        nextEventCycle() const noexcept;
        [[nodiscard]] EeCounterBankSnapshot snapshot() const noexcept;

        [[nodiscard]] static bool isRegisterAddress(
            uint32_t address) noexcept;

    private:
        struct Counter
        {
            uint32_t count = 0u;
            uint16_t mode = 0u;
            uint16_t target = 0xffffu;
            uint16_t hold = 0u;
            uint32_t rate = 2u;
            uint64_t startCycle = 0u;
            bool targetAfterOverflow = false;
        };

        struct Signal
        {
            EeCounterSignalPhase phase =
                EeCounterSignalPhase::Render;
            uint64_t startCycle = 0u;
            uint64_t deltaCycles = 0u;
        };

        struct DecodedRegister
        {
            size_t counter = 0u;
            EeCounterRegister reg = EeCounterRegister::Count;
        };

        [[nodiscard]] static std::optional<DecodedRegister>
        decodeRegister(uint32_t address) noexcept;
        [[nodiscard]] static uint32_t clockSource(
            const Counter &counter) noexcept;
        [[nodiscard]] static bool gateEnabled(
            const Counter &counter) noexcept;
        [[nodiscard]] static bool gateUsesVBlank(
            const Counter &counter) noexcept;
        [[nodiscard]] static uint32_t gateMode(
            const Counter &counter) noexcept;
        [[nodiscard]] static bool zeroReturn(
            const Counter &counter) noexcept;
        [[nodiscard]] static bool countEnabled(
            const Counter &counter) noexcept;
        [[nodiscard]] static bool targetInterruptEnabled(
            const Counter &counter) noexcept;
        [[nodiscard]] static bool overflowInterruptEnabled(
            const Counter &counter) noexcept;
        [[nodiscard]] bool canCount(
            const Counter &counter) const noexcept;
        [[nodiscard]] std::optional<uint64_t>
        nextScanEventCycle() const noexcept;
        [[nodiscard]] uint64_t nextHSyncEdge() const noexcept;
        [[nodiscard]] uint64_t nextVSyncEdge() const noexcept;
        [[nodiscard]] std::optional<uint64_t>
        nextInternalCounterEvent(
            const Counter &counter) const noexcept;

        void synchronizeInternalCounters(
            uint64_t cycle,
            EeCounterAdvanceResult &result) noexcept;
        void synchronizeInternalCounter(
            size_t index,
            uint64_t cycle,
            EeCounterAdvanceResult &result) noexcept;
        void addCounterTicks(
            size_t index,
            uint64_t ticks,
            EeCounterAdvanceResult &result) noexcept;
        void processCounterConditions(
            size_t index,
            EeCounterAdvanceResult &result) noexcept;
        void raiseTargetInterrupt(
            size_t index,
            EeCounterAdvanceResult &result) noexcept;
        void raiseOverflowInterrupt(
            size_t index,
            EeCounterAdvanceResult &result) noexcept;
        void processHSyncEdge(
            uint64_t cycle,
            EeCounterAdvanceResult &result) noexcept;
        void processVSyncEdge(
            uint64_t cycle,
            EeCounterAdvanceResult &result) noexcept;
        void processGateEdge(
            bool vblank,
            bool rising,
            uint64_t cycle,
            EeCounterAdvanceResult &result) noexcept;
        void resetCounterAtGate(
            Counter &counter,
            uint64_t cycle) noexcept;

        [[nodiscard]] static uint64_t saturatingAdd(
            uint64_t lhs, uint64_t rhs) noexcept;
        [[nodiscard]] static uint64_t saturatingMultiply(
            uint64_t lhs, uint64_t rhs) noexcept;
        [[nodiscard]] static uint64_t preserveEndpointStart(
            uint64_t start,
            uint64_t oldDelta,
            uint64_t newDelta) noexcept;

        std::array<Counter, kEeCounterCount> m_counters{};
        EeCounterVideoTiming m_videoTiming{};
        Signal m_hSync{};
        Signal m_vSync{};
        uint64_t m_currentCycle = 0u;
    };
}
