#include "runtime/ee_counters.h"

#include <algorithm>
#include <limits>

namespace ps2x::timing
{
    namespace
    {
        constexpr uint16_t kModeClockSourceMask = 0x0003u;
        constexpr uint16_t kModeGateEnable = 0x0004u;
        constexpr uint16_t kModeGateSource = 0x0008u;
        constexpr uint16_t kModeGateModeMask = 0x0030u;
        constexpr uint16_t kModeZeroReturn = 0x0040u;
        constexpr uint16_t kModeCountEnable = 0x0080u;
        constexpr uint16_t kModeTargetInterrupt = 0x0100u;
        constexpr uint16_t kModeOverflowInterrupt = 0x0200u;
        constexpr uint16_t kModeTargetReached = 0x0400u;
        constexpr uint16_t kModeOverflowReached = 0x0800u;
        constexpr uint16_t kModeWritableControlMask = 0x03ffu;
        constexpr uint16_t kModeStatusMask =
            kModeTargetReached | kModeOverflowReached;
        constexpr uint32_t kFirstTimerIntcCause = 9u;
    }

    EeCounterBank::EeCounterBank() noexcept
    {
        reset();
    }

    uint64_t EeCounterBank::saturatingAdd(
        uint64_t lhs, uint64_t rhs) noexcept
    {
        const uint64_t maximum =
            std::numeric_limits<uint64_t>::max();
        return rhs > maximum - lhs ? maximum : lhs + rhs;
    }

    uint64_t EeCounterBank::saturatingMultiply(
        uint64_t lhs, uint64_t rhs) noexcept
    {
        const uint64_t maximum =
            std::numeric_limits<uint64_t>::max();
        if (lhs != 0u && rhs > maximum / lhs)
        {
            return maximum;
        }
        return lhs * rhs;
    }

    uint64_t EeCounterBank::preserveEndpointStart(
        uint64_t start,
        uint64_t oldDelta,
        uint64_t newDelta) noexcept
    {
        const uint64_t endpoint =
            saturatingAdd(start, oldDelta);
        return endpoint >= newDelta
                   ? endpoint - newDelta
                   : 0u;
    }

    void EeCounterBank::reset(uint64_t currentCycle) noexcept
    {
        m_counters = {};
        for (Counter &counter : m_counters)
        {
            counter.target = 0xffffu;
            counter.rate = 2u;
            counter.startCycle = currentCycle;
        }

        m_currentCycle = currentCycle;
        m_hSync = {
            .phase = EeCounterSignalPhase::Render,
            .startCycle = currentCycle,
            .deltaCycles = m_videoTiming.hRenderCycles,
        };
        m_vSync = {
            .phase = EeCounterSignalPhase::Render,
            .startCycle = currentCycle,
            .deltaCycles = m_videoTiming.renderCycles,
        };
    }

    EeCounterAdvanceResult EeCounterBank::configureVideoTiming(
        EeCounterVideoTiming timing,
        uint64_t currentCycle) noexcept
    {
        EeCounterAdvanceResult result = advanceTo(currentCycle);

        timing.hRenderCycles =
            std::max<uint32_t>(timing.hRenderCycles, 1u);
        timing.hBlankCycles =
            std::max<uint32_t>(timing.hBlankCycles, 1u);
        timing.renderCycles =
            std::max<uint64_t>(timing.renderCycles, 1u);
        timing.blankCycles =
            std::max<uint64_t>(timing.blankCycles, 1u);

        const uint64_t oldHDelta = m_hSync.deltaCycles;
        const uint64_t oldVDelta = m_vSync.deltaCycles;
        m_videoTiming = timing;
        const uint64_t newHDelta =
            m_hSync.phase == EeCounterSignalPhase::Blank
                ? m_videoTiming.hBlankCycles
                : m_videoTiming.hRenderCycles;
        const uint64_t newVDelta =
            m_vSync.phase == EeCounterSignalPhase::Blank
                ? m_videoTiming.blankCycles
                : m_videoTiming.renderCycles;
        m_hSync.startCycle = preserveEndpointStart(
            m_hSync.startCycle, oldHDelta, newHDelta);
        m_hSync.deltaCycles = newHDelta;
        m_vSync.startCycle = preserveEndpointStart(
            m_vSync.startCycle, oldVDelta, newVDelta);
        m_vSync.deltaCycles = newVDelta;
        return result;
    }

    std::optional<EeCounterBank::DecodedRegister>
    EeCounterBank::decodeRegister(uint32_t address) noexcept
    {
        if (address < kEeCounterBaseAddress ||
            address >=
                kEeCounterBaseAddress +
                    kEeCounterStride * kEeCounterCount)
        {
            return std::nullopt;
        }

        const uint32_t offset =
            address - kEeCounterBaseAddress;
        const size_t counter =
            static_cast<size_t>(offset / kEeCounterStride);
        const uint32_t registerOffset =
            offset % kEeCounterStride;

        EeCounterRegister reg{};
        switch (registerOffset)
        {
        case static_cast<uint32_t>(EeCounterRegister::Count):
            reg = EeCounterRegister::Count;
            break;
        case static_cast<uint32_t>(EeCounterRegister::Mode):
            reg = EeCounterRegister::Mode;
            break;
        case static_cast<uint32_t>(EeCounterRegister::Target):
            reg = EeCounterRegister::Target;
            break;
        case static_cast<uint32_t>(EeCounterRegister::Hold):
            if (counter >= 2u)
            {
                return std::nullopt;
            }
            reg = EeCounterRegister::Hold;
            break;
        default:
            return std::nullopt;
        }
        return DecodedRegister{counter, reg};
    }

    bool EeCounterBank::isRegisterAddress(
        uint32_t address) noexcept
    {
        return decodeRegister(address).has_value();
    }

    uint32_t EeCounterBank::clockSource(
        const Counter &counter) noexcept
    {
        return counter.mode & kModeClockSourceMask;
    }

    bool EeCounterBank::gateEnabled(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeGateEnable) != 0u;
    }

    bool EeCounterBank::gateUsesVBlank(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeGateSource) != 0u;
    }

    uint32_t EeCounterBank::gateMode(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeGateModeMask) >> 4u;
    }

    bool EeCounterBank::zeroReturn(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeZeroReturn) != 0u;
    }

    bool EeCounterBank::countEnabled(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeCountEnable) != 0u;
    }

    bool EeCounterBank::targetInterruptEnabled(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeTargetInterrupt) != 0u;
    }

    bool EeCounterBank::overflowInterruptEnabled(
        const Counter &counter) noexcept
    {
        return (counter.mode & kModeOverflowInterrupt) != 0u;
    }

    bool EeCounterBank::canCount(
        const Counter &counter) const noexcept
    {
        if (!countEnabled(counter))
        {
            return false;
        }
        if (!gateEnabled(counter))
        {
            return true;
        }
        if (!gateUsesVBlank(counter))
        {
            if (clockSource(counter) == 3u)
            {
                return false;
            }
            return gateMode(counter) != 0u ||
                   m_hSync.phase ==
                       EeCounterSignalPhase::Render;
        }
        return gateMode(counter) != 0u ||
               m_vSync.phase ==
                   EeCounterSignalPhase::Render;
    }

    uint64_t EeCounterBank::nextHSyncEdge() const noexcept
    {
        return saturatingAdd(
            m_hSync.startCycle, m_hSync.deltaCycles);
    }

    uint64_t EeCounterBank::nextVSyncEdge() const noexcept
    {
        return saturatingAdd(
            m_vSync.startCycle, m_vSync.deltaCycles);
    }

    std::optional<uint64_t>
    EeCounterBank::nextScanEventCycle() const noexcept
    {
        bool hSyncRequired = false;
        bool vSyncRequired = false;
        for (const Counter &counter : m_counters)
        {
            if (gateEnabled(counter))
            {
                if (gateUsesVBlank(counter))
                {
                    vSyncRequired = true;
                }
                else
                {
                    hSyncRequired = true;
                }
            }
            if (clockSource(counter) == 3u &&
                countEnabled(counter) &&
                (targetInterruptEnabled(counter) ||
                 overflowInterruptEnabled(counter) ||
                 zeroReturn(counter)))
            {
                hSyncRequired = true;
            }
        }
        if (hSyncRequired && vSyncRequired)
        {
            return std::min(
                nextHSyncEdge(), nextVSyncEdge());
        }
        if (hSyncRequired)
        {
            return nextHSyncEdge();
        }
        if (vSyncRequired)
        {
            return nextVSyncEdge();
        }
        return std::nullopt;
    }

    void EeCounterBank::raiseTargetInterrupt(
        size_t index,
        EeCounterAdvanceResult &result) noexcept
    {
        Counter &counter = m_counters[index];
        if (!targetInterruptEnabled(counter) ||
            (counter.mode & kModeTargetReached) != 0u)
        {
            return;
        }
        counter.mode |= kModeTargetReached;
        result.newlyRaisedInterruptMask |=
            1u << (kFirstTimerIntcCause +
                   static_cast<uint32_t>(index));
    }

    void EeCounterBank::raiseOverflowInterrupt(
        size_t index,
        EeCounterAdvanceResult &result) noexcept
    {
        Counter &counter = m_counters[index];
        if (!overflowInterruptEnabled(counter) ||
            (counter.mode & kModeOverflowReached) != 0u)
        {
            return;
        }
        counter.mode |= kModeOverflowReached;
        result.newlyRaisedInterruptMask |=
            1u << (kFirstTimerIntcCause +
                   static_cast<uint32_t>(index));
    }

    void EeCounterBank::processCounterConditions(
        size_t index,
        EeCounterAdvanceResult &result) noexcept
    {
        Counter &counter = m_counters[index];
        if (counter.count > 0xffffu)
        {
            counter.count -= 0x10000u;
            counter.targetAfterOverflow = false;
            raiseOverflowInterrupt(index, result);
        }

        if (counter.targetAfterOverflow ||
            counter.count < counter.target)
        {
            return;
        }

        raiseTargetInterrupt(index, result);
        if (zeroReturn(counter) && counter.target != 0u)
        {
            counter.count -= counter.target;
        }
        else
        {
            // A zero compare cannot make progress by subtraction. Treat it
            // like the ordinary non-zero-return path after the first match.
            counter.targetAfterOverflow = true;
        }
    }

    void EeCounterBank::addCounterTicks(
        size_t index,
        uint64_t ticks,
        EeCounterAdvanceResult &result) noexcept
    {
        Counter &counter = m_counters[index];
        while (ticks != 0u)
        {
            uint64_t untilCondition =
                0x10000u - counter.count;
            if (!counter.targetAfterOverflow)
            {
                const uint64_t untilTarget =
                    counter.target > counter.count
                        ? static_cast<uint64_t>(
                              counter.target -
                              counter.count)
                        : 0u;
                if (untilTarget != 0u)
                {
                    untilCondition =
                        std::min(untilCondition, untilTarget);
                }
            }
            untilCondition =
                std::max<uint64_t>(untilCondition, 1u);

            const uint64_t step =
                std::min(ticks, untilCondition);
            counter.count += static_cast<uint32_t>(step);
            ticks -= step;
            processCounterConditions(index, result);

            if (ticks != 0u &&
                zeroReturn(counter) &&
                counter.target != 0u &&
                !counter.targetAfterOverflow &&
                counter.count < counter.target)
            {
                // Once the first compare has latched, repeated zero-return
                // periods have no new externally visible interrupt. Collapse
                // complete periods instead of iterating once per compare.
                const uint64_t periods =
                    ticks / counter.target;
                if (periods != 0u)
                {
                    ticks -= periods * counter.target;
                }
            }
        }
    }

    void EeCounterBank::synchronizeInternalCounter(
        size_t index,
        uint64_t cycle,
        EeCounterAdvanceResult &result) noexcept
    {
        Counter &counter = m_counters[index];
        if (clockSource(counter) == 3u)
        {
            counter.startCycle = cycle;
            return;
        }
        if (cycle < counter.startCycle)
        {
            counter.startCycle = cycle;
            return;
        }

        const uint64_t ticks =
            (cycle - counter.startCycle) / counter.rate;
        counter.startCycle = saturatingAdd(
            counter.startCycle,
            saturatingMultiply(ticks, counter.rate));
        if (ticks != 0u && canCount(counter))
        {
            addCounterTicks(index, ticks, result);
        }
    }

    void EeCounterBank::synchronizeInternalCounters(
        uint64_t cycle,
        EeCounterAdvanceResult &result) noexcept
    {
        for (size_t index = 0u;
             index < m_counters.size();
             ++index)
        {
            synchronizeInternalCounter(index, cycle, result);
        }
    }

    void EeCounterBank::resetCounterAtGate(
        Counter &counter,
        uint64_t cycle) noexcept
    {
        counter.count = 0u;
        counter.targetAfterOverflow = false;
        if (counter.rate != 0u)
        {
            counter.startCycle =
                cycle &
                ~(static_cast<uint64_t>(counter.rate) - 1u);
        }
        else
        {
            counter.startCycle = cycle;
        }
    }

    void EeCounterBank::processGateEdge(
        bool vblank,
        bool rising,
        uint64_t cycle,
        EeCounterAdvanceResult &result) noexcept
    {
        (void)result;
        for (Counter &counter : m_counters)
        {
            if (!gateEnabled(counter) ||
                gateUsesVBlank(counter) != vblank)
            {
                continue;
            }

            switch (gateMode(counter))
            {
            case 0u:
                counter.startCycle =
                    cycle &
                    ~(static_cast<uint64_t>(counter.rate) - 1u);
                break;
            case 1u:
                if (rising)
                {
                    resetCounterAtGate(counter, cycle);
                }
                break;
            case 2u:
                if (!rising)
                {
                    resetCounterAtGate(counter, cycle);
                }
                break;
            case 3u:
                resetCounterAtGate(counter, cycle);
                break;
            }
        }
    }

    void EeCounterBank::processHSyncEdge(
        uint64_t cycle,
        EeCounterAdvanceResult &result) noexcept
    {
        if (m_hSync.phase == EeCounterSignalPhase::Render)
        {
            for (size_t index = 0u;
                 index < m_counters.size();
                 ++index)
            {
                Counter &counter = m_counters[index];
                if (clockSource(counter) == 3u &&
                    canCount(counter))
                {
                    addCounterTicks(index, 1u, result);
                }
            }
            processGateEdge(
                false, true, cycle, result);
            m_hSync.phase = EeCounterSignalPhase::Blank;
            m_hSync.startCycle = cycle;
            m_hSync.deltaCycles =
                m_videoTiming.hBlankCycles;
            ++result.hBlankStarts;
        }
        else
        {
            processGateEdge(
                false, false, cycle, result);
            m_hSync.phase = EeCounterSignalPhase::Render;
            m_hSync.startCycle = cycle;
            m_hSync.deltaCycles =
                m_videoTiming.hRenderCycles;
        }
    }

    void EeCounterBank::processVSyncEdge(
        uint64_t cycle,
        EeCounterAdvanceResult &result) noexcept
    {
        if (m_vSync.phase == EeCounterSignalPhase::Render)
        {
            processGateEdge(
                true, true, cycle, result);
            m_vSync.phase = EeCounterSignalPhase::Blank;
            m_vSync.startCycle = cycle;
            m_vSync.deltaCycles =
                m_videoTiming.blankCycles;
            m_hSync.deltaCycles = saturatingAdd(
                m_hSync.deltaCycles,
                m_videoTiming.hSyncErrorCycles);
            ++result.vBlankStarts;
        }
        else
        {
            processGateEdge(
                true, false, cycle, result);
            m_vSync.phase = EeCounterSignalPhase::Render;
            m_vSync.startCycle = cycle;
            m_vSync.deltaCycles =
                m_videoTiming.renderCycles;
        }
    }

    EeCounterAdvanceResult EeCounterBank::advanceTo(
        uint64_t currentCycle) noexcept
    {
        EeCounterAdvanceResult result{};
        if (currentCycle < m_currentCycle)
        {
            return result;
        }

        while (true)
        {
            const uint64_t vEdge = nextVSyncEdge();
            const uint64_t hEdge = nextHSyncEdge();
            const uint64_t edge = std::min(vEdge, hEdge);
            if (edge > currentCycle ||
                edge == std::numeric_limits<uint64_t>::max())
            {
                break;
            }

            synchronizeInternalCounters(edge, result);
            // PCSX2 updates VSync before HSync so VSync's accumulated HSync
            // rounding correction affects an equal-cycle HSync edge.
            if (vEdge <= hEdge)
            {
                processVSyncEdge(edge, result);
            }
            else
            {
                processHSyncEdge(edge, result);
            }
        }

        synchronizeInternalCounters(currentCycle, result);
        m_currentCycle = currentCycle;
        return result;
    }

    std::optional<uint64_t>
    EeCounterBank::nextInternalCounterEvent(
        const Counter &counter) const noexcept
    {
        if (clockSource(counter) == 3u ||
            !canCount(counter))
        {
            return std::nullopt;
        }

        const bool targetWork =
            targetInterruptEnabled(counter) ||
            zeroReturn(counter);
        if (!targetWork &&
            !overflowInterruptEnabled(counter))
        {
            return std::nullopt;
        }

        uint64_t ticks =
            std::numeric_limits<uint64_t>::max();
        if (overflowInterruptEnabled(counter) ||
            (counter.targetAfterOverflow &&
             targetWork))
        {
            ticks = 0x10000u - counter.count;
        }
        if (!counter.targetAfterOverflow &&
            targetWork)
        {
            const uint64_t targetTicks =
                counter.target > counter.count
                    ? static_cast<uint64_t>(
                          counter.target -
                          counter.count)
                    : 1u;
            ticks = std::min(ticks, targetTicks);
        }
        if (ticks ==
            std::numeric_limits<uint64_t>::max())
        {
            return std::nullopt;
        }

        const uint64_t deadline = saturatingAdd(
            counter.startCycle,
            saturatingMultiply(ticks, counter.rate));
        return deadline > m_currentCycle
                   ? std::optional<uint64_t>{deadline}
                   : std::optional<uint64_t>{
                         saturatingAdd(m_currentCycle, 1u)};
    }

    std::optional<uint64_t>
    EeCounterBank::nextEventCycle() const noexcept
    {
        std::optional<uint64_t> next =
            nextScanEventCycle();
        for (const Counter &counter : m_counters)
        {
            const std::optional<uint64_t> candidate =
                nextInternalCounterEvent(counter);
            if (candidate.has_value() &&
                (!next.has_value() || *candidate < *next))
            {
                next = candidate;
            }
        }
        return next;
    }

    bool EeCounterBank::readRegister(
        uint32_t address,
        uint64_t currentCycle,
        uint32_t &value,
        EeCounterAdvanceResult *advance) noexcept
    {
        const std::optional<DecodedRegister> decoded =
            decodeRegister(address);
        if (!decoded.has_value())
        {
            return false;
        }

        const EeCounterAdvanceResult result =
            advanceTo(currentCycle);
        if (advance)
        {
            *advance = result;
        }

        const Counter &counter =
            m_counters[decoded->counter];
        switch (decoded->reg)
        {
        case EeCounterRegister::Count:
            value = counter.count & 0xffffu;
            return true;
        case EeCounterRegister::Mode:
            value = counter.mode;
            return true;
        case EeCounterRegister::Target:
            value = counter.target;
            return true;
        case EeCounterRegister::Hold:
            value = counter.hold;
            return true;
        }
        return false;
    }

    bool EeCounterBank::writeRegister(
        uint32_t address,
        uint32_t value,
        uint64_t currentCycle,
        EeCounterAdvanceResult *advance) noexcept
    {
        const std::optional<DecodedRegister> decoded =
            decodeRegister(address);
        if (!decoded.has_value())
        {
            return false;
        }

        EeCounterAdvanceResult result =
            advanceTo(currentCycle);
        Counter &counter =
            m_counters[decoded->counter];
        switch (decoded->reg)
        {
        case EeCounterRegister::Count:
            counter.count = value & 0xffffu;
            counter.targetAfterOverflow =
                counter.count >= counter.target;
            break;
        case EeCounterRegister::Mode:
        {
            const uint16_t oldStatus =
                counter.mode & kModeStatusMask;
            const uint16_t clearedStatus =
                oldStatus &
                ~static_cast<uint16_t>(
                    value & kModeStatusMask);
            counter.mode =
                clearedStatus |
                static_cast<uint16_t>(
                    value & kModeWritableControlMask);
            switch (clockSource(counter))
            {
            case 0u:
                counter.rate = 2u;
                break;
            case 1u:
                counter.rate = 32u;
                break;
            case 2u:
                counter.rate = 512u;
                break;
            case 3u:
                counter.rate =
                    m_videoTiming.hRenderCycles +
                    m_videoTiming.hBlankCycles;
                break;
            }
            counter.startCycle =
                currentCycle &
                ~(static_cast<uint64_t>(counter.rate) - 1u);
            break;
        }
        case EeCounterRegister::Target:
            counter.target =
                static_cast<uint16_t>(value);
            counter.targetAfterOverflow =
                counter.target <= counter.count;
            break;
        case EeCounterRegister::Hold:
            counter.hold =
                static_cast<uint16_t>(value);
            break;
        }

        if (advance)
        {
            *advance = result;
        }
        return true;
    }

    EeCounterBankSnapshot EeCounterBank::snapshot() const noexcept
    {
        EeCounterBankSnapshot result{};
        for (size_t index = 0u;
             index < m_counters.size();
             ++index)
        {
            const Counter &counter = m_counters[index];
            result.counters[index] = {
                .count = counter.count,
                .mode = counter.mode,
                .target = counter.target,
                .hold = counter.hold,
                .rate = counter.rate,
                .startCycle = counter.startCycle,
                .targetAfterOverflow =
                    counter.targetAfterOverflow,
            };
        }
        result.currentCycle = m_currentCycle;
        result.hSyncStartCycle = m_hSync.startCycle;
        result.vSyncStartCycle = m_vSync.startCycle;
        result.hSyncDeltaCycles =
            static_cast<uint32_t>(
                std::min<uint64_t>(
                    m_hSync.deltaCycles,
                    std::numeric_limits<uint32_t>::max()));
        result.vSyncDeltaCycles = m_vSync.deltaCycles;
        result.hSyncPhase = m_hSync.phase;
        result.vSyncPhase = m_vSync.phase;
        return result;
    }
}
