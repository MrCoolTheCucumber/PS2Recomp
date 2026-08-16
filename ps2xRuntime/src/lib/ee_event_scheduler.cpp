#include "runtime/ee_event_scheduler.h"

#include <limits>
#include <unordered_set>

namespace ps2x::timing
{
    EeEventScheduler::EeEventScheduler() noexcept
    {
        recomputeNext();
    }

    uint64_t EeEventScheduler::nextNonzero(uint64_t value) noexcept
    {
        ++value;
        return value == 0u ? 1u : value;
    }

    EeEventScheduler::Slot *EeEventScheduler::slot(
        EeEventSource source) noexcept
    {
        const size_t index = eeEventSourceIndex(source);
        return index < m_slots.size() ? &m_slots[index] : nullptr;
    }

    const EeEventScheduler::Slot *EeEventScheduler::slot(
        EeEventSource source) const noexcept
    {
        const size_t index = eeEventSourceIndex(source);
        return index < m_slots.size() ? &m_slots[index] : nullptr;
    }

    EeEventToken EeEventScheduler::scheduleAbsolute(
        EeEventSource source, EeTick deadline) noexcept
    {
        Slot *const target = slot(source);
        if (!target)
        {
            return {source, 0u};
        }

        if (target->pending)
        {
            ++m_statistics.replaced;
        }
        target->generation = nextNonzero(target->generation);
        m_nextSequence = nextNonzero(m_nextSequence);
        target->deadline = deadline;
        target->sequence = m_nextSequence;
        target->pending = true;
        ++m_statistics.scheduled;

        if (m_servicing && deadline <= m_serviceTick)
        {
            ++m_statistics.sameTickReschedules;
        }

        recomputeNext();
        return {source, target->generation};
    }

    EeEventToken EeEventScheduler::scheduleAfter(
        EeEventSource source,
        EeTick now,
        EeTickDelta delay) noexcept
    {
        return scheduleAbsolute(
            source, saturatingAdd(now, delay));
    }

    bool EeEventScheduler::cancel(EeEventSource source) noexcept
    {
        Slot *const target = slot(source);
        if (!target || !target->pending)
        {
            return false;
        }

        target->pending = false;
        target->generation = nextNonzero(target->generation);
        ++m_statistics.cancelled;
        recomputeNext();
        return true;
    }

    bool EeEventScheduler::cancel(EeEventToken token) noexcept
    {
        Slot *const target = slot(token.source);
        if (!target ||
            !target->pending ||
            target->generation != token.generation)
        {
            return false;
        }
        return cancel(token.source);
    }

    void EeEventScheduler::reset() noexcept
    {
        for (Slot &source : m_slots)
        {
            source.pending = false;
            source.generation = nextNonzero(source.generation);
            source.sequence = 0u;
            source.deadline = {};
        }
        m_nextSequence = 0u;
        m_statistics = {};
        m_serviceTick = {};
        m_servicing = false;
        recomputeNext();
    }

    bool EeEventScheduler::pending(
        EeEventSource source) const noexcept
    {
        const Slot *const target = slot(source);
        return target && target->pending;
    }

    std::optional<EeScheduledEvent> EeEventScheduler::event(
        EeEventSource source) const noexcept
    {
        const Slot *const target = slot(source);
        if (!target || !target->pending)
        {
            return std::nullopt;
        }
        return EeScheduledEvent{
            .token = {source, target->generation},
            .deadline = target->deadline,
            .sequence = target->sequence,
            .pending = true,
        };
    }

    bool EeEventScheduler::hasPendingEvents() const noexcept
    {
        return m_hasPending;
    }

    std::optional<EeTick> EeEventScheduler::nextDeadline() const noexcept
    {
        return m_hasPending
                   ? std::optional<EeTick>{m_nextDeadline}
                   : std::nullopt;
    }

    bool EeEventScheduler::deadlineDue(EeTick now) const noexcept
    {
        return m_hasPending && now >= m_nextDeadline;
    }

    const EeEventSchedulerStatistics &
    EeEventScheduler::statistics() const noexcept
    {
        return m_statistics;
    }

    EeEventSchedulerSnapshot EeEventScheduler::snapshot() const noexcept
    {
        EeEventSchedulerSnapshot state{};
        for (size_t index = 0u; index < m_slots.size(); ++index)
        {
            const Slot &source = m_slots[index];
            state.slots[index] = {
                .source = static_cast<EeEventSource>(index),
                .deadline = source.deadline,
                .generation = source.generation,
                .sequence = source.sequence,
                .pending = source.pending,
            };
        }
        state.nextSequence = m_nextSequence;
        state.statistics = m_statistics;
        return state;
    }

    bool EeEventScheduler::restore(
        const EeEventSchedulerSnapshot &state) noexcept
    {
        std::unordered_set<uint64_t> pendingSequences;
        for (size_t index = 0u; index < state.slots.size(); ++index)
        {
            const EeEventSlotSnapshot &source = state.slots[index];
            if (source.source != static_cast<EeEventSource>(index))
                return false;
            if (source.pending &&
                (source.generation == 0u || source.sequence == 0u ||
                 source.sequence > state.nextSequence ||
                 !pendingSequences.insert(source.sequence).second))
            {
                return false;
            }
        }

        for (size_t index = 0u; index < m_slots.size(); ++index)
        {
            const EeEventSlotSnapshot &source = state.slots[index];
            m_slots[index] = {
                .deadline = source.deadline,
                .generation = source.generation,
                .sequence = source.sequence,
                .pending = source.pending,
            };
        }
        m_nextSequence = state.nextSequence;
        m_statistics = state.statistics;
        m_serviceTick = {};
        m_servicing = false;
        recomputeNext();
        return true;
    }

    void EeEventScheduler::recomputeNext() noexcept
    {
        m_hasPending = false;
        m_nextDeadline = EeTick::fromRaw(
            std::numeric_limits<uint64_t>::max());
        m_nextSource = EeEventSource::Cop0Performance;

        for (size_t index = 0u; index < m_slots.size(); ++index)
        {
            const Slot &candidate = m_slots[index];
            if (!candidate.pending)
            {
                continue;
            }

            const EeEventSource source =
                static_cast<EeEventSource>(index);
            if (!m_hasPending ||
                candidate.deadline < m_nextDeadline ||
                (candidate.deadline == m_nextDeadline &&
                 index < eeEventSourceIndex(m_nextSource)))
            {
                m_hasPending = true;
                m_nextDeadline = candidate.deadline;
                m_nextSource = source;
            }
        }
    }

    std::optional<EeEventSource> EeEventScheduler::nextDueSource(
        EeTick now) const noexcept
    {
        if (!deadlineDue(now))
        {
            return std::nullopt;
        }
        return m_nextSource;
    }

    std::optional<EeEventService> EeEventScheduler::takeNextDue(
        EeTick now) noexcept
    {
        const std::optional<EeEventSource> source =
            nextDueSource(now);
        if (!source.has_value())
        {
            return std::nullopt;
        }

        Slot *const target = slot(*source);
        if (!target || !target->pending)
        {
            recomputeNext();
            return std::nullopt;
        }

        const EeEventService service{
            .source = *source,
            .scheduledTick = target->deadline,
            .serviceTick = now,
            .latenessTicks =
                elapsedEeTicks(target->deadline, now),
            .generation = target->generation,
            .sequence = target->sequence,
        };
        target->pending = false;
        recomputeNext();

        ++m_statistics.serviced;
        if (service.latenessTicks.raw() != 0u)
        {
            ++m_statistics.late;
        }
        return service;
    }
}
