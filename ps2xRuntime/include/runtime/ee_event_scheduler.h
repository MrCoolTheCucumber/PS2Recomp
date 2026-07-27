#pragma once

#include "runtime/ee_timing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ps2x::timing
{
    enum class EeSchedulingMode : uint8_t
    {
        Legacy = 0u,
        Shadow,
        Event,
    };

    [[nodiscard]] constexpr std::string_view eeSchedulingModeName(
        EeSchedulingMode mode) noexcept
    {
        switch (mode)
        {
        case EeSchedulingMode::Legacy:
            return "legacy";
        case EeSchedulingMode::Shadow:
            return "shadow";
        case EeSchedulingMode::Event:
            return "event";
        }
        return "unknown";
    }

    // Values are also the deterministic priority for equal-deadline events.
    // PCSX2 services due device callbacks before its ordinary VU batch.
    // Add new sources only with an ordering test and a reference-backed
    // priority decision.
    enum class EeEventSource : uint8_t
    {
        // PCSX2 updates EE counters (including VSync) before dispatching
        // timed DMA callbacks in _cpuEventTest_Shared().
        VSync = 0u,
        // Compatibility-complete work is published before timed device
        // callbacks. A timed callback that itself completes work can
        // reschedule this source at the same tick for the scheduler's next
        // priority pass.
        DmacCompletion,
        DmacVif1,
        DmacFromScratchpad,
        DmacToScratchpad,
        VifVu1Finish,
        Vu0PeriodicCompatibility,
        Count,
    };

    inline constexpr size_t kEeEventSourceCount =
        static_cast<size_t>(EeEventSource::Count);

    [[nodiscard]] constexpr size_t eeEventSourceIndex(
        EeEventSource source) noexcept
    {
        return static_cast<size_t>(source);
    }

    [[nodiscard]] constexpr std::string_view eeEventSourceName(
        EeEventSource source) noexcept
    {
        switch (source)
        {
        case EeEventSource::VSync:
            return "vsync";
        case EeEventSource::DmacCompletion:
            return "dmac_completion";
        case EeEventSource::Vu0PeriodicCompatibility:
            return "vu0_periodic_compatibility";
        case EeEventSource::DmacVif1:
            return "dmac_vif1";
        case EeEventSource::DmacFromScratchpad:
            return "dmac_from_scratchpad";
        case EeEventSource::DmacToScratchpad:
            return "dmac_to_scratchpad";
        case EeEventSource::VifVu1Finish:
            return "vif_vu1_finish";
        case EeEventSource::Count:
            break;
        }
        return "unknown";
    }

    struct EeEventToken
    {
        EeEventSource source = EeEventSource::Vu0PeriodicCompatibility;
        uint64_t generation = 0u;

        friend constexpr bool operator==(
            EeEventToken, EeEventToken) noexcept = default;
    };

    struct EeScheduledEvent
    {
        EeEventToken token{};
        EeTick deadline{};
        uint64_t sequence = 0u;
        bool pending = false;
    };

    struct EeEventService
    {
        EeEventSource source = EeEventSource::Vu0PeriodicCompatibility;
        EeTick scheduledTick{};
        EeTick serviceTick{};
        EeTickDelta latenessTicks{};
        uint64_t generation = 0u;
        uint64_t sequence = 0u;
    };

    struct EeEventServiceResult
    {
        size_t serviced = 0u;
        bool limitExceeded = false;
        bool reentrantServiceRejected = false;
        std::optional<EeEventSource> offendingSource;
    };

    struct EeEventSchedulerStatistics
    {
        uint64_t scheduled = 0u;
        uint64_t replaced = 0u;
        uint64_t cancelled = 0u;
        uint64_t serviced = 0u;
        uint64_t late = 0u;
        uint64_t sameTickReschedules = 0u;
        uint64_t serviceLimitHits = 0u;
    };

    class EeEventScheduler
    {
    public:
        static constexpr size_t kMaximumServicesPerBoundary = 1024u;

        EeEventScheduler() noexcept;

        [[nodiscard]] EeEventToken scheduleAbsolute(
            EeEventSource source, EeTick deadline) noexcept;
        [[nodiscard]] EeEventToken scheduleAfter(
            EeEventSource source,
            EeTick now,
            EeTickDelta delay) noexcept;

        bool cancel(EeEventSource source) noexcept;
        bool cancel(EeEventToken token) noexcept;
        void reset() noexcept;

        [[nodiscard]] bool pending(EeEventSource source) const noexcept;
        [[nodiscard]] std::optional<EeScheduledEvent> event(
            EeEventSource source) const noexcept;
        [[nodiscard]] bool hasPendingEvents() const noexcept;
        [[nodiscard]] std::optional<EeTick> nextDeadline() const noexcept;
        [[nodiscard]] bool deadlineDue(EeTick now) const noexcept;
        [[nodiscard]] const EeEventSchedulerStatistics &statistics()
            const noexcept;

        template <typename Visitor>
        EeEventServiceResult serviceDue(
            EeTick now, Visitor &&visitor)
        {
            EeEventServiceResult result{};
            if (m_servicing)
            {
                result.reentrantServiceRejected = true;
                result.offendingSource = nextDueSource(now);
                return result;
            }

            m_servicing = true;
            m_serviceTick = now;
            try
            {
                while (result.serviced <
                       kMaximumServicesPerBoundary)
                {
                    const std::optional<EeEventService> service =
                        takeNextDue(now);
                    if (!service.has_value())
                    {
                        break;
                    }

                    visitor(*service);
                    ++result.serviced;
                }
            }
            catch (...)
            {
                m_servicing = false;
                throw;
            }
            m_servicing = false;

            if (deadlineDue(now))
            {
                result.limitExceeded = true;
                result.offendingSource = nextDueSource(now);
                ++m_statistics.serviceLimitHits;
            }
            return result;
        }

    private:
        struct Slot
        {
            EeTick deadline{};
            uint64_t generation = 0u;
            uint64_t sequence = 0u;
            bool pending = false;
        };

        [[nodiscard]] static uint64_t nextNonzero(
            uint64_t value) noexcept;
        [[nodiscard]] Slot *slot(EeEventSource source) noexcept;
        [[nodiscard]] const Slot *slot(
            EeEventSource source) const noexcept;
        void recomputeNext() noexcept;
        [[nodiscard]] std::optional<EeEventSource> nextDueSource(
            EeTick now) const noexcept;
        [[nodiscard]] std::optional<EeEventService> takeNextDue(
            EeTick now) noexcept;

        std::array<Slot, kEeEventSourceCount> m_slots{};
        EeTick m_nextDeadline{};
        EeEventSource m_nextSource =
            EeEventSource::Vu0PeriodicCompatibility;
        bool m_hasPending = false;
        uint64_t m_nextSequence = 0u;
        EeEventSchedulerStatistics m_statistics{};
        EeTick m_serviceTick{};
        bool m_servicing = false;
    };
}
