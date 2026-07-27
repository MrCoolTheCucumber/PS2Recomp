#include "MiniTest.h"

#include "runtime/ee_event_scheduler.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

using namespace ps2x::timing;

void register_ee_event_scheduler_tests()
{
    MiniTest::Case("EeEventScheduler", [](TestCase &tc)
    {
        tc.Run("schedules replaces cancels and services fixed sources", [](TestCase &t)
        {
            EeEventScheduler scheduler;
            const EeEventToken first = scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, eeTickFromRaw(40u));
            const EeEventToken replacement = scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, eeTickFromRaw(24u));
            t.IsTrue(
                first.generation != replacement.generation,
                "replacement should advance the slot generation");
            t.IsTrue(
                !scheduler.cancel(first),
                "a replaced generation must not cancel the new event");
            t.IsTrue(
                scheduler.cancel(replacement),
                "the live generation should cancel its event");

            for (size_t index = 0u;
                 index < kEeEventSourceCount; ++index)
            {
                (void)scheduler.scheduleAbsolute(
                    static_cast<EeEventSource>(index),
                    eeTickFromRaw(100u + index));
            }

            std::array<bool, kEeEventSourceCount> observed{};
            const EeEventServiceResult result =
                scheduler.serviceDue(
                    eeTickFromRaw(200u),
                    [&](const EeEventService &service)
                    {
                        observed[eeEventSourceIndex(
                            service.source)] = true;
                        t.Equals(
                            service.serviceTick.raw(), 200u,
                            "service should retain actual time");
                        t.Equals(
                            service.latenessTicks.raw(),
                            200u - service.scheduledTick.raw(),
                            "service should expose lateness");
                    });
            t.Equals(
                result.serviced, kEeEventSourceCount,
                "all fixed event sources should be serviced");
            for (bool sourceObserved : observed)
            {
                t.IsTrue(
                    sourceObserved,
                    "each fixed source should dispatch once");
            }
            t.IsTrue(
                !scheduler.hasPendingEvents(),
                "serviced one-shot events should leave no pending slot");
        });

        tc.Run("equal deadlines use fixed source priority", [](TestCase &t)
        {
            EeEventScheduler scheduler;
            const EeTick deadline = eeTickFromRaw(80u);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::Cop0Performance, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::VSync, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::EeCounters, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::Cop0Timer, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::VifVu1Finish, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacCompletion, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacToScratchpad, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacGif, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::HleSif1, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacVif0, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacToIpu, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacFromScratchpad, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::VifVu0Finish, deadline);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::Vu0PeriodicCompatibility, deadline);

            std::vector<EeEventSource> order;
            scheduler.serviceDue(
                deadline,
                [&](const EeEventService &service)
                {
                    order.push_back(service.source);
                });
            t.Equals(
                order.size(), static_cast<size_t>(15u),
                "all equal-deadline events should dispatch");
            if (order.size() == 15u)
            {
                t.IsTrue(
                    order[0] ==
                        EeEventSource::Cop0Performance,
                    "level-2 performance overflow should have highest device-event priority");
                t.IsTrue(
                    order[1] == EeEventSource::VSync,
                    "VSync counter work should precede timed DMA callbacks");
                t.IsTrue(
                    order[2] == EeEventSource::EeCounters,
                    "timer counter work should precede timed DMA callbacks");
                t.IsTrue(
                    order[3] == EeEventSource::Cop0Timer,
                    "Count/Compare should publish before timed DMA callbacks");
                t.IsTrue(
                    order[4] == EeEventSource::DmacCompletion,
                    "ready DMAC completion should publish first");
                t.IsTrue(
                    order[5] == EeEventSource::DmacVif1,
                    "DMAC VIF1 should precede GIF");
                t.IsTrue(
                    order[6] == EeEventSource::DmacGif,
                    "DMAC GIF should precede HLE SIF1");
                t.IsTrue(
                    order[7] == EeEventSource::HleSif1,
                    "HLE SIF1 should occupy the SIF callback band");
                t.IsTrue(
                    order[8] == EeEventSource::DmacVif0,
                    "DMAC VIF0 should precede scratchpad DMA");
                t.IsTrue(
                    order[9] == EeEventSource::DmacToIpu,
                    "DMAC to-IPU should follow VIF0");
                t.IsTrue(
                    order[10] ==
                        EeEventSource::DmacFromScratchpad,
                    "SPR-from should follow physical IPU DMA");
                t.IsTrue(
                    order[11] ==
                        EeEventSource::DmacToScratchpad,
                    "SPR-to should follow SPR-from");
                t.IsTrue(
                    order[12] == EeEventSource::VifVu0Finish,
                    "VU0 finish should follow DMAC callbacks");
                t.IsTrue(
                    order[13] == EeEventSource::VifVu1Finish,
                    "VU1 finish should follow VU0 finish");
                t.IsTrue(
                    order[14] ==
                        EeEventSource::Vu0PeriodicCompatibility,
                    "the ordinary VU0 batch should follow device callbacks");
            }
        });

        tc.Run("handlers preserve and replace scheduled work", [](TestCase &t)
        {
            EeEventScheduler scheduler;
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, eeTickFromRaw(16u));
            (void)scheduler.scheduleAbsolute(
                EeEventSource::VifVu1Finish, eeTickFromRaw(64u));

            scheduler.serviceDue(
                eeTickFromRaw(16u),
                [&](const EeEventService &service)
                {
                    t.IsTrue(
                        service.source == EeEventSource::DmacVif1,
                        "only the due source should dispatch");
                    (void)scheduler.scheduleAbsolute(
                        EeEventSource::DmacVif1,
                        eeTickFromRaw(48u));
                });

            const auto dmac =
                scheduler.event(EeEventSource::DmacVif1);
            const auto finish =
                scheduler.event(EeEventSource::VifVu1Finish);
            t.IsTrue(
                dmac.has_value() &&
                    dmac->deadline.raw() == 48u,
                "handler rescheduling should survive dispatch");
            t.IsTrue(
                finish.has_value() &&
                    finish->deadline.raw() == 64u,
                "unrelated pending work should survive dispatch");
            t.IsTrue(
                scheduler.nextDeadline().has_value() &&
                    scheduler.nextDeadline()->raw() == 48u,
                "the cached minimum should reflect handler changes");
        });

        tc.Run("same-tick loops stop at the hard service bound", [](TestCase &t)
        {
            EeEventScheduler scheduler;
            const EeTick now = eeTickFromRaw(32u);
            (void)scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, now);

            const EeEventServiceResult result =
                scheduler.serviceDue(
                    now,
                    [&](const EeEventService &service)
                    {
                        (void)scheduler.scheduleAbsolute(
                            service.source, now);
                    });
            t.Equals(
                result.serviced,
                EeEventScheduler::kMaximumServicesPerBoundary,
                "a same-tick loop should stop at the fixed bound");
            t.IsTrue(
                result.limitExceeded,
                "the service result should diagnose an event storm");
            t.IsTrue(
                result.offendingSource.has_value() &&
                    *result.offendingSource ==
                        EeEventSource::DmacVif1,
                "the storm diagnostic should identify its source");
            t.IsTrue(
                scheduler.pending(EeEventSource::DmacVif1),
                "the bounded service should retain the next pending event");
            t.Equals(
                scheduler.statistics().sameTickReschedules,
                static_cast<uint64_t>(
                    EeEventScheduler::kMaximumServicesPerBoundary),
                "same-tick reschedules should be counted");
        });

        tc.Run("saturates deadlines at uint64 maximum", [](TestCase &t)
        {
            EeEventScheduler scheduler;
            const uint64_t maximum =
                std::numeric_limits<uint64_t>::max();
            (void)scheduler.scheduleAfter(
                EeEventSource::DmacVif1,
                eeTickFromRaw(maximum - 2u),
                eeTickDeltaFromRaw(10u));

            const auto pending =
                scheduler.event(EeEventSource::DmacVif1);
            t.IsTrue(
                pending.has_value() &&
                    pending->deadline.raw() == maximum,
                "relative scheduling should saturate instead of wrapping");

            size_t services = 0u;
            scheduler.serviceDue(
                eeTickFromRaw(maximum),
                [&](const EeEventService &service)
                {
                    ++services;
                    t.Equals(
                        service.latenessTicks.raw(), 0u,
                        "a maximum-tick event should service on time");
                });
            t.Equals(
                services, static_cast<size_t>(1u),
                "a maximum-tick deadline should remain serviceable");
        });

        tc.Run("reset invalidates pending generations", [](TestCase &t)
        {
            EeEventScheduler scheduler;
            const EeEventToken stale = scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, eeTickFromRaw(12u));
            scheduler.reset();

            t.IsTrue(
                !scheduler.hasPendingEvents(),
                "reset should clear every pending source");
            t.IsTrue(
                !scheduler.nextDeadline().has_value(),
                "reset should clear the cached minimum");
            const EeEventToken current = scheduler.scheduleAbsolute(
                EeEventSource::DmacVif1, eeTickFromRaw(20u));
            t.IsTrue(
                stale.generation != current.generation,
                "reset should invalidate pre-reset tokens");
            t.IsTrue(
                !scheduler.cancel(stale),
                "a pre-reset token must not cancel new work");
        });
    });
}
