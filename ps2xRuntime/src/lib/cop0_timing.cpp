#include "runtime/cop0_timing.h"

#include <algorithm>
#include <limits>

namespace ps2x::timing
{
    namespace
    {
        constexpr uint32_t kStatusExl = 1u << 1u;
        constexpr uint32_t kStatusErl = 1u << 2u;
        constexpr uint32_t kPccrCounterEnable = 1u << 31u;
        constexpr uint32_t kPccrWritableMask = 0x800ffbfeu;
        constexpr uint32_t kPcrOverflow = 1u << 31u;
        constexpr uint64_t kCounterModulus = uint64_t{1u} << 32u;
        constexpr uint64_t kPerformanceCounterModulus =
            uint64_t{1u} << 31u;
    }

    void Cop0Timing::reset(
        uint64_t currentCycle,
        uint32_t count,
        uint32_t compare,
        uint32_t pccr,
        uint32_t pcr0,
        uint32_t pcr1) noexcept
    {
        m_count = count;
        m_compare = compare;
        m_pccr = pccr & kPccrWritableMask;
        m_pcr = {pcr0, pcr1};
        m_lastCountCycle = currentCycle;
        m_lastPerformanceCycle = {
            currentCycle, currentCycle};
        m_timerArmed = false;
        m_timerPending = false;
    }

    Cop0TimingAdvanceResult Cop0Timing::synchronizeCount(
        uint64_t currentCycle,
        bool minimumIncrement) noexcept
    {
        Cop0TimingAdvanceResult result{};
        uint64_t elapsed =
            currentCycle >= m_lastCountCycle
                ? currentCycle - m_lastCountCycle
                : 0u;
        if (minimumIncrement && elapsed == 0u)
        {
            elapsed = 1u;
        }

        if (elapsed != 0u &&
            m_timerArmed &&
            !m_timerPending)
        {
            const uint32_t modularDistance =
                m_compare - m_count;
            const uint64_t distance =
                modularDistance != 0u
                    ? modularDistance
                    : kCounterModulus;
            if (elapsed >= distance)
            {
                m_timerPending = true;
                result.timerRaised = true;
            }
        }

        m_count += static_cast<uint32_t>(elapsed);
        m_lastCountCycle = currentCycle;
        return result;
    }

    uint32_t Cop0Timing::readCount(
        uint64_t currentCycle,
        Cop0TimingAdvanceResult *advance) noexcept
    {
        const Cop0TimingAdvanceResult result =
            synchronizeCount(currentCycle, true);
        if (advance)
        {
            advance->merge(result);
        }
        return m_count;
    }

    void Cop0Timing::writeCount(
        uint32_t value,
        uint64_t currentCycle) noexcept
    {
        m_count = value;
        m_lastCountCycle = currentCycle;
    }

    void Cop0Timing::writeCompare(
        uint32_t value,
        uint64_t currentCycle) noexcept
    {
        (void)synchronizeCount(currentCycle);
        m_compare = value;
        m_timerArmed = true;
        m_timerPending = false;
    }

    Cop0TimingAdvanceResult
    Cop0Timing::synchronizePerformance(
        uint64_t currentCycle,
        uint32_t status,
        bool minimumIncrement) noexcept
    {
        Cop0TimingAdvanceResult result{};
        for (size_t counter = 0u;
             counter < m_pcr.size();
             ++counter)
        {
            uint64_t elapsed =
                currentCycle >=
                        m_lastPerformanceCycle[counter]
                    ? currentCycle -
                          m_lastPerformanceCycle[counter]
                    : 0u;

            const bool canCount =
                (status & kStatusErl) == 0u &&
                (m_pccr & kPccrCounterEnable) != 0u &&
                performanceCounterEnabled(
                    m_pccr, counter, status) &&
                performanceEvent(m_pccr, counter) == 1u;
            if (canCount)
            {
                if (minimumIncrement && elapsed == 0u)
                {
                    elapsed = 1u;
                }
                if (elapsed != 0u)
                {
                    result.merge(
                        incrementPerformanceCounter(
                            counter, elapsed));
                }
            }
            m_lastPerformanceCycle[counter] =
                currentCycle;
        }
        return result;
    }

    Cop0TimingAdvanceResult
    Cop0Timing::recordPerformanceEvents(
        uint32_t status,
        const Cop0PerformanceEvents &events) noexcept
    {
        Cop0TimingAdvanceResult result{};
        if ((status & kStatusErl) != 0u ||
            (m_pccr & kPccrCounterEnable) == 0u)
        {
            return result;
        }

        for (size_t counter = 0u;
             counter < m_pcr.size();
             ++counter)
        {
            if (!performanceCounterEnabled(
                    m_pccr, counter, status))
            {
                continue;
            }
            const uint32_t event =
                performanceEvent(m_pccr, counter);
            const uint32_t occurrences =
                performanceEventOccurrences(
                    counter, event, events);
            if (occurrences != 0u)
            {
                result.merge(
                    incrementPerformanceCounter(
                        counter, occurrences));
            }
        }
        return result;
    }

    uint32_t Cop0Timing::readPerformance(
        uint32_t selector,
        uint64_t currentCycle,
        uint32_t status,
        Cop0TimingAdvanceResult *advance) noexcept
    {
        selector &= 0x3fu;
        if ((selector & 1u) == 0u)
        {
            return m_pccr;
        }

        const Cop0TimingAdvanceResult result =
            synchronizePerformance(
                currentCycle, status, true);
        if (advance)
        {
            advance->merge(result);
        }
        const size_t counter =
            (selector & 2u) != 0u ? 1u : 0u;
        return m_pcr[counter];
    }

    bool Cop0Timing::writePerformance(
        uint32_t selector,
        uint32_t value,
        uint64_t currentCycle,
        uint32_t status,
        Cop0TimingAdvanceResult *advance) noexcept
    {
        selector &= 0x3fu;
        if ((selector & 1u) == 0u)
        {
            // MTPS accepts only selector zero. Other even selectors are
            // architecturally decoded as MFPS but have no write effect.
            if ((selector & 0x3eu) != 0u)
            {
                return false;
            }
            const Cop0TimingAdvanceResult result =
                synchronizePerformance(
                    currentCycle, status, true);
            if (advance)
            {
                advance->merge(result);
            }
            m_pccr = value & kPccrWritableMask;
            return true;
        }

        const size_t counter =
            (selector & 2u) != 0u ? 1u : 0u;
        m_pcr[counter] = value;
        m_lastPerformanceCycle[counter] =
            currentCycle;
        return true;
    }

    std::optional<uint64_t>
    Cop0Timing::nextTimerCycle() const noexcept
    {
        if (!m_timerArmed ||
            m_timerPending)
        {
            return std::nullopt;
        }
        const uint32_t modularDistance =
            m_compare - m_count;
        const uint64_t distance =
            modularDistance != 0u
                ? modularDistance
                : kCounterModulus;
        return saturatingAdd(
            m_lastCountCycle, distance);
    }

    std::optional<uint64_t>
    Cop0Timing::nextPerformanceEventCycle(
        uint64_t currentCycle,
        uint32_t status) const noexcept
    {
        if ((status & kStatusErl) != 0u ||
            (m_pccr & kPccrCounterEnable) == 0u)
        {
            return std::nullopt;
        }
        if ((m_pcr[0] & kPcrOverflow) != 0u ||
            (m_pcr[1] & kPcrOverflow) != 0u)
        {
            return currentCycle;
        }

        std::optional<uint64_t> result;
        for (size_t counter = 0u;
             counter < m_pcr.size();
             ++counter)
        {
            if (!performanceCounterEnabled(
                    m_pccr, counter, status) ||
                performanceEvent(
                    m_pccr, counter) != 1u)
            {
                continue;
            }
            const uint64_t distance =
                kPerformanceCounterModulus -
                static_cast<uint64_t>(
                    m_pcr[counter] &
                    ~kPcrOverflow);
            const uint64_t deadline =
                saturatingAdd(
                    m_lastPerformanceCycle[counter],
                    distance);
            if (!result.has_value() ||
                deadline < *result)
            {
                result = deadline;
            }
        }
        return result;
    }

    bool Cop0Timing::performanceExceptionPending(
        uint32_t status) const noexcept
    {
        return (status & kStatusErl) == 0u &&
               (m_pccr & kPccrCounterEnable) != 0u &&
               ((m_pcr[0] | m_pcr[1]) &
                kPcrOverflow) != 0u;
    }

    Cop0TimingSnapshot Cop0Timing::snapshot() const noexcept
    {
        return Cop0TimingSnapshot{
            .count = m_count,
            .compare = m_compare,
            .pccr = m_pccr,
            .pcr = m_pcr,
            .lastCountCycle = m_lastCountCycle,
            .lastPerformanceCycle =
                m_lastPerformanceCycle,
            .timerArmed = m_timerArmed,
            .timerPending = m_timerPending,
        };
    }

    bool Cop0Timing::performanceCounterEnabled(
        uint32_t pccr,
        size_t counter,
        uint32_t status) noexcept
    {
        const uint32_t base =
            counter == 0u ? 0u : 10u;
        const uint32_t mode =
            (status & kStatusExl) != 0u
                ? 1u
                : 2u + ((status >> 3u) & 3u);
        return (pccr & (1u << (base + mode))) != 0u;
    }

    bool Cop0Timing::supportedPerformanceEvent(
        size_t counter,
        uint32_t event) noexcept
    {
        if (event == 0u)
        {
            return counter == 1u;
        }
        switch (event)
        {
        case 1u:
        case 2u:
        case 3u:
        case 12u:
        case 13u:
        case 14u:
        case 15u:
            return true;
        default:
            return false;
        }
    }

    uint32_t Cop0Timing::performanceEventOccurrences(
        size_t counter,
        uint32_t event,
        const Cop0PerformanceEvents &events) noexcept
    {
        if (!supportedPerformanceEvent(counter, event) ||
            event == 1u)
        {
            return 0u;
        }
        switch (event)
        {
        case 0u:
            return events.lowOrderBranchIssued;
        case 2u:
            return counter == 0u
                       ? events.singleInstructionIssued
                       : events.dualInstructionIssued;
        case 3u:
            return counter == 0u
                       ? events.branchIssued
                       : events.branchMispredicted;
        case 12u:
            return events.instructionCompleted;
        case 13u:
            return events.nonBdsInstructionCompleted;
        case 14u:
            return counter == 0u
                       ? events.cop2InstructionCompleted
                       : events.cop1InstructionCompleted;
        case 15u:
            return counter == 0u
                       ? events.loadCompleted
                       : events.storeCompleted;
        default:
            return 0u;
        }
    }

    uint32_t Cop0Timing::performanceEvent(
        uint32_t pccr,
        size_t counter) noexcept
    {
        const uint32_t shift =
            counter == 0u ? 5u : 15u;
        return (pccr >> shift) & 0x1fu;
    }

    Cop0TimingAdvanceResult
    Cop0Timing::incrementPerformanceCounter(
        size_t counter,
        uint64_t increment) noexcept
    {
        Cop0TimingAdvanceResult result{};
        if (counter >= m_pcr.size() || increment == 0u)
        {
            return result;
        }

        const bool alreadyOverflowed =
            (m_pcr[counter] & kPcrOverflow) != 0u;
        const uint64_t value =
            static_cast<uint64_t>(
                m_pcr[counter] & ~kPcrOverflow) +
            increment;
        const bool overflowed =
            value >= kPerformanceCounterModulus;
        m_pcr[counter] =
            static_cast<uint32_t>(
                value % kPerformanceCounterModulus) |
            ((alreadyOverflowed || overflowed)
                 ? kPcrOverflow
                 : 0u);
        if (!alreadyOverflowed && overflowed)
        {
            result.performanceOverflowMask |=
                static_cast<uint8_t>(1u << counter);
        }
        return result;
    }

    uint64_t Cop0Timing::saturatingAdd(
        uint64_t lhs,
        uint64_t rhs) noexcept
    {
        if (rhs >
            std::numeric_limits<uint64_t>::max() -
                lhs)
        {
            return std::numeric_limits<uint64_t>::max();
        }
        return lhs + rhs;
    }
}
