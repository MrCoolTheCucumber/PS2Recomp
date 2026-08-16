#pragma once

#include "runtime/ee_timing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ps2x::timing
{
    // Values are also the deterministic priority for equal-deadline events.
    // PCSX2 services due device callbacks before its ordinary VU batch.
    // Add new sources only with an ordering test and a reference-backed
    // priority decision.
    enum class EeEventSource : uint8_t
    {
        // The EE manual assigns the non-maskable performance-counter
        // overflow exception priority immediately below Reset/NMI.
        Cop0Performance = 0u,
        // PCSX2 updates EE counters (including VSync) before dispatching
        // timed DMA callbacks in _cpuEventTest_Shared().
        VSync,
        EeCounters,
        // Count/Compare publishes Cause.IP7 after hardware counters and
        // before timed DMA callbacks at a shared EE event boundary.
        Cop0Timer,
        // Ready completion work is published before timed device callbacks.
        // A timed callback that itself completes work can
        // reschedule this source at the same tick for the scheduler's next
        // priority pass.
        DmacCompletion,
        DmacVif1,
        DmacGif,
        // PS2Recomp's high-level IOP substitutes one retained cross-domain
        // operation for the physical EE/IOP SIF1 endpoints.
        HleSif1,
        DmacVif0,
        // PCSX2 services the physical IPU DMA callbacks after VIF0 and before
        // scratchpad DMA at a shared EE event boundary, with From-IPU before
        // To-IPU.
        DmacFromIpu,
        DmacToIpu,
        DmacFromScratchpad,
        DmacToScratchpad,
        VifVu0Finish,
        VifVu1Finish,
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
        case EeEventSource::Cop0Performance:
            return "cop0_performance";
        case EeEventSource::VSync:
            return "vsync";
        case EeEventSource::EeCounters:
            return "ee_counters";
        case EeEventSource::Cop0Timer:
            return "cop0_timer";
        case EeEventSource::DmacCompletion:
            return "dmac_completion";
        case EeEventSource::DmacVif1:
            return "dmac_vif1";
        case EeEventSource::DmacGif:
            return "dmac_gif";
        case EeEventSource::HleSif1:
            return "hle_sif1";
        case EeEventSource::DmacVif0:
            return "dmac_vif0";
        case EeEventSource::DmacFromIpu:
            return "dmac_from_ipu";
        case EeEventSource::DmacToIpu:
            return "dmac_to_ipu";
        case EeEventSource::DmacFromScratchpad:
            return "dmac_from_scratchpad";
        case EeEventSource::DmacToScratchpad:
            return "dmac_to_scratchpad";
        case EeEventSource::VifVu1Finish:
            return "vif_vu1_finish";
        case EeEventSource::VifVu0Finish:
            return "vif_vu0_finish";
        case EeEventSource::Count:
            break;
        }
        return "unknown";
    }

    struct EeEventToken
    {
        EeEventSource source = EeEventSource::Cop0Performance;
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
        EeEventSource source = EeEventSource::Cop0Performance;
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

    struct EeEventSlotSnapshot
    {
        EeEventSource source = EeEventSource::Cop0Performance;
        EeTick deadline{};
        uint64_t generation = 0u;
        uint64_t sequence = 0u;
        bool pending = false;
    };

    struct EeEventSchedulerSnapshot
    {
        std::array<EeEventSlotSnapshot, kEeEventSourceCount> slots{};
        uint64_t nextSequence = 0u;
        EeEventSchedulerStatistics statistics{};
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
        [[nodiscard]] EeEventSchedulerSnapshot snapshot() const noexcept;
        [[nodiscard]] bool restore(
            const EeEventSchedulerSnapshot &snapshot) noexcept;

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
            EeEventSource::Cop0Performance;
        bool m_hasPending = false;
        uint64_t m_nextSequence = 0u;
        EeEventSchedulerStatistics m_statistics{};
        EeTick m_serviceTick{};
        bool m_servicing = false;
    };
}
