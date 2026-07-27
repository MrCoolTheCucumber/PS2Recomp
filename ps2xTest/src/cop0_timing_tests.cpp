#include "MiniTest.h"

#include "ps2_runtime.h"
#include "runtime/cop0_timing.h"

#include <cstdint>

using namespace ps2x::timing;

namespace
{
    constexpr uint32_t kStatusExl = 1u << 1u;
    constexpr uint32_t kStatusErl = 1u << 2u;
    constexpr uint32_t kStatusUserMode = 2u << 3u;
    constexpr uint32_t kPccrCte = 1u << 31u;
    constexpr uint32_t kPccrEvent0Cycles = 1u << 5u;
    constexpr uint32_t kPccrEvent1Cycles = 1u << 15u;
    constexpr uint32_t kPccrK0 = 1u << 2u;
    constexpr uint32_t kPccrU1 = 1u << 14u;

    bool serviceAfterCycles(
        PS2Runtime &runtime,
        R5900Context &ctx,
        uint32_t cycles)
    {
        ctx.advanceEeCycleTicks(cycles * 8u);
        try
        {
            runtime.serviceEeEventsAtBlockBoundary(
                nullptr, &ctx);
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }
}

void register_cop0_timing_tests()
{
    MiniTest::Case("Cop0Timing", [](TestCase &tc)
    {
        tc.Run("Count reproduces the bounded PCSX2 access oracle", [](TestCase &t)
        {
            Cop0Timing timing;
            timing.reset();
            timing.writeCount(0u, 0u);

            t.Equals(
                timing.readCount(0u), 1u,
                "a same-cycle Count read should make the reference minimum increment");
            t.Equals(
                timing.readCount(127u), 128u,
                "elapsed EE cycles should accumulate from the canonical cycle");
            t.Equals(
                timing.readCount(127u), 129u,
                "a repeated same-cycle Count read should increment again");
        });

        tc.Run("Compare detects modular equality and clears only on write", [](TestCase &t)
        {
            Cop0Timing timing;
            timing.reset(
                100u, 0xfffffffeu, 1u);
            timing.writeCompare(1u, 100u);

            const auto deadline =
                timing.nextTimerCycle();
            t.IsTrue(
                deadline.has_value(),
                "an unlatched timer should expose a deadline");
            if (deadline.has_value())
            {
                t.Equals(
                    *deadline, 103u,
                    "Compare should schedule across Count wrap");
            }

            Cop0TimingAdvanceResult before =
                timing.synchronizeCount(102u);
            t.IsFalse(
                before.timerRaised,
                "the timer should remain clear before equality");
            t.IsFalse(
                timing.snapshot().timerPending,
                "pre-equality synchronization should not latch IP7");

            Cop0TimingAdvanceResult at =
                timing.synchronizeCount(103u);
            t.IsTrue(
                at.timerRaised,
                "the equality cycle should raise the timer");
            t.IsTrue(
                timing.snapshot().timerPending,
                "timer state should remain latched after equality");
            t.IsFalse(
                timing.nextTimerCycle().has_value(),
                "a latched timer should not schedule a second match");

            timing.writeCount(7u, 103u);
            t.IsTrue(
                timing.snapshot().timerPending,
                "writing Count must not acknowledge the timer");
            timing.writeCompare(9u, 103u);
            t.IsFalse(
                timing.snapshot().timerPending,
                "writing Compare should acknowledge the timer");
            t.Equals(
                *timing.nextTimerCycle(), 105u,
                "the replacement Compare value should own the new deadline");
        });

        tc.Run("equal Count and Compare schedule the next wrap", [](TestCase &t)
        {
            Cop0Timing timing;
            timing.reset(12u, 0x12345678u, 0x12345678u);
            timing.writeCompare(
                0x12345678u, 12u);
            const auto deadline =
                timing.nextTimerCycle();
            t.IsTrue(
                deadline.has_value(),
                "equal registers should retain a future equality");
            if (deadline.has_value())
            {
                t.Equals(
                    *deadline,
                    12u + (uint64_t{1u} << 32u),
                    "writing an already-equal value must not synthesize an immediate match");
            }
        });

        tc.Run("performance selectors reproduce the PCSX2 PCR oracle", [](TestCase &t)
        {
            Cop0Timing timing;
            timing.reset();
            const uint32_t pccr =
                kPccrCte |
                kPccrEvent0Cycles |
                kPccrK0;

            t.IsTrue(
                timing.writePerformance(
                    1u, 0u, 0u, 0u),
                "selector one should write PCR0");
            t.IsTrue(
                timing.writePerformance(
                    0u, pccr, 0u, 0u),
                "selector zero should write PCCR");
            t.Equals(
                timing.readPerformance(
                    1u, 127u, 0u),
                127u,
                "PCR0 should count elapsed processor cycles");
            t.Equals(
                timing.readPerformance(
                    1u, 127u, 0u),
                128u,
                "a repeated PCR read should make the minimum increment");

            t.IsTrue(
                timing.writePerformance(
                    0u, 0u, 127u, 0u),
                "disabling CTE should update the old configuration first");
            t.Equals(
                timing.readPerformance(
                    1u, 127u, 0u),
                129u,
                "a disabled counter should retain the final pre-disable increment");
        });

        tc.Run("performance selector bits choose PCCR PCR0 and PCR1", [](TestCase &t)
        {
            Cop0Timing timing;
            timing.reset();
            t.IsTrue(
                timing.writePerformance(
                    0u, 0x87654321u, 0u, 0u),
                "selector zero should write PCCR");
            t.IsFalse(
                timing.writePerformance(
                    2u, 0x11111111u, 0u, 0u),
                "nonzero even MTPS selectors should have no effect");
            t.IsTrue(
                timing.writePerformance(
                    1u, 0x22222222u, 0u, 0u),
                "selector one should write PCR0");
            t.IsTrue(
                timing.writePerformance(
                    3u, 0x33333333u, 0u, 0u),
                "selector three should write PCR1");

            t.Equals(
                timing.readPerformance(
                    2u, 0u, 0u),
                0x87654321u,
                "every even read selector should return PCCR");
            t.Equals(
                timing.readPerformance(
                    1u, 0u, kStatusErl),
                0x22222222u,
                "odd selector one should return PCR0");
            t.Equals(
                timing.readPerformance(
                    3u, 0u, kStatusErl),
                0x33333333u,
                "odd selector three should return PCR1");
        });

        tc.Run("performance mode and ERL gates are independent per counter", [](TestCase &t)
        {
            Cop0Timing timing;
            const uint32_t pccr =
                kPccrCte |
                kPccrEvent0Cycles |
                kPccrEvent1Cycles |
                kPccrK0 |
                kPccrU1;
            timing.reset(0u, 0u, 0u, pccr);

            (void)timing.synchronizePerformance(
                10u, 0u);
            Cop0TimingSnapshot kernel =
                timing.snapshot();
            t.Equals(
                kernel.pcr[0], 10u,
                "kernel mode should enable PCR0 K0");
            t.Equals(
                kernel.pcr[1], 0u,
                "kernel mode should leave user-only PCR1 stopped");

            (void)timing.synchronizePerformance(
                20u, kStatusUserMode);
            Cop0TimingSnapshot user =
                timing.snapshot();
            t.Equals(
                user.pcr[0], 10u,
                "user mode should stop kernel-only PCR0");
            t.Equals(
                user.pcr[1], 10u,
                "user mode should enable PCR1 U1");

            (void)timing.synchronizePerformance(
                30u,
                kStatusUserMode | kStatusErl);
            Cop0TimingSnapshot erl =
                timing.snapshot();
            t.Equals(
                erl.pcr[0], 10u,
                "ERL should inhibit PCR0");
            t.Equals(
                erl.pcr[1], 10u,
                "ERL should inhibit PCR1");

            (void)timing.synchronizePerformance(
                40u, kStatusExl);
            Cop0TimingSnapshot exl =
                timing.snapshot();
            t.Equals(
                exl.pcr[0], 20u,
                "without EXL0, the kernel mode bit should still enable PCR0");
        });

        tc.Run("performance overflow has an exact scheduled boundary", [](TestCase &t)
        {
            const uint32_t pccr =
                kPccrCte |
                kPccrEvent0Cycles |
                kPccrK0;
            Cop0Timing timing;
            timing.reset(
                50u, 0u, 0u, pccr,
                0x7ffffffdu, 0u);

            const auto deadline =
                timing.nextPerformanceEventCycle(
                    50u, 0u);
            t.IsTrue(
                deadline.has_value(),
                "an enabled cycle counter should expose its overflow deadline");
            if (deadline.has_value())
            {
                t.Equals(
                    *deadline, 53u,
                    "PCR0 should overflow after exactly three cycles");
            }

            Cop0TimingAdvanceResult before =
                timing.synchronizePerformance(
                    52u, 0u);
            t.Equals(
                before.performanceOverflowMask,
                static_cast<uint8_t>(0u),
                "PCR0 should not overflow early");
            Cop0TimingAdvanceResult at =
                timing.synchronizePerformance(
                    53u, 0u);
            t.Equals(
                at.performanceOverflowMask,
                static_cast<uint8_t>(1u),
                "PCR0 should report the first overflow");
            t.Equals(
                timing.snapshot().pcr[0],
                0x80000000u,
                "overflow should set OVFL and wrap the 31-bit value");
            t.IsTrue(
                timing.performanceExceptionPending(0u),
                "CTE plus OVFL should request the level-2 exception");
            t.IsFalse(
                timing.performanceExceptionPending(
                    kStatusErl),
                "ERL should inhibit repeated level-2 delivery");
            t.IsFalse(
                timing.nextPerformanceEventCycle(
                    53u, kStatusErl)
                    .has_value(),
                "ERL should cancel performance scheduler work");
        });

        tc.Run("runtime-owned Count and selectors survive EE context changes", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context first{};
            R5900Context second{};

            t.Equals(
                runtime.readCop0Count(
                    nullptr, &first),
                1u,
                "the first context should observe the minimum Count increment");
            first.advanceEeCycleTicks(5u * 8u);
            t.Equals(
                runtime.readCop0Count(
                    nullptr, &first),
                6u,
                "the first context should advance the shared Count");
            t.Equals(
                runtime.readCop0Count(
                    nullptr, &second),
                7u,
                "a fresh context should continue the runtime Count instead of restarting it");

            runtime.writeCop0Compare(
                nullptr, &first, 0x12345678u);
            t.Equals(
                runtime.readCop0Compare(&second),
                0x12345678u,
                "Compare should be one hardware register across HLE contexts");

            runtime.writeCop0Performance(
                nullptr, &first, 1u,
                0x11223344u);
            t.Equals(
                runtime.readCop0Performance(
                    nullptr, &second, 1u),
                0x11223344u,
                "PCR0 should be shared across HLE contexts while PCCR remains disabled");
        });

        tc.Run("Status writes retain only documented writable fields", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            runtime.writeCop0Status(
                nullptr, &ctx, 0xffffffffu);
            t.Equals(
                ctx.cop0_status,
                0xf0c79c1fu,
                "MTC0 Status should retain CU DEV BEV CH EDI EIE IM BEM KSU ERL EXL and IE");
            t.IsTrue(
                (ctx.cop0_status & 0x00800000u) != 0u,
                "Status.DEV must remain writable for the documented level-2 bootstrap vector");
        });

        tc.Run("event Count Compare latches IP7 while masked then vectors when enabled", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00102000u;

            runtime.writeCop0Count(
                nullptr, &ctx, 10u);
            runtime.writeCop0Compare(
                nullptr, &ctx, 13u);

            const PS2Runtime::DebugEeScheduler scheduled =
                runtime.debugEeSchedulerSnapshot();
            const auto &slot =
                scheduled.slots[
                    eeEventSourceIndex(
                        EeEventSource::Cop0Timer)];
            t.IsTrue(
                slot.pending,
                "MTC0 Compare should arm a fixed timer source");
            t.IsTrue(
                slot.device.active,
                "debug state should expose the armed Count/Compare timer");
            t.Equals(
                slot.device.qwc, 0u,
                "debug state should distinguish armed from latched timer state");
            t.Equals(
                slot.deadlineTick, 3u * 8u,
                "the timer source should use exact unsigned Count distance");

            t.IsFalse(
                serviceAfterCycles(
                    runtime, ctx, 3u),
                "a masked timer should latch without entering an exception");
            t.IsTrue(
                (ctx.cop0_cause & 0x00008000u) != 0u,
                "Count equality should publish Cause.IP7");
            const auto latched =
                runtime.debugEeSchedulerSnapshot();
            t.IsTrue(
                latched.slots[
                    eeEventSourceIndex(
                        EeEventSource::Cop0Timer)]
                    .device.active,
                "a latched timer should remain active until Compare acknowledgement");
            t.Equals(
                latched.slots[
                    eeEventSourceIndex(
                        EeEventSource::Cop0Timer)]
                    .device.qwc,
                1u,
                "debug state should expose the pending IP7 latch");
            t.Equals(
                ctx.pc, 0x00102000u,
                "masked publication should preserve the interrupted PC");

            bool interrupted = false;
            try
            {
                runtime.writeCop0Status(
                    nullptr, &ctx, 0x00018001u);
            }
            catch (const PS2GuestException &)
            {
                interrupted = true;
            }
            t.IsTrue(
                interrupted,
                "enabling IE EIE and IM7 should deliver the latched timer");
            t.Equals(
                ctx.pc, 0x80000200u,
                "the timer should use the normal level-1 interrupt vector");
            t.Equals(
                ctx.cop0_epc, 0x00102000u,
                "the timer should retain the interrupted restart PC");
            t.IsTrue(
                (ctx.cop0_status & 0x2u) != 0u,
                "timer delivery should enter EXL");
            t.IsTrue(
                (ctx.cop0_cause & 0x00008000u) != 0u,
                "IP7 should remain latched until Compare is written");
        });

        tc.Run("MTC0 Compare acknowledges IP7 and replaces its deadline", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            runtime.writeCop0Count(
                nullptr, &ctx, 20u);
            runtime.writeCop0Compare(
                nullptr, &ctx, 21u);
            t.IsFalse(
                serviceAfterCycles(
                    runtime, ctx, 1u),
                "the masked equality should not vector");
            t.IsTrue(
                (ctx.cop0_cause & 0x8000u) != 0u,
                "the first Compare should latch IP7");

            runtime.writeCop0Compare(
                nullptr, &ctx, 25u);
            t.IsFalse(
                (ctx.cop0_cause & 0x8000u) != 0u,
                "MTC0 Compare should clear IP7");
            const auto status =
                runtime.debugEeSchedulerSnapshot();
            const auto &slot =
                status.slots[
                    eeEventSourceIndex(
                        EeEventSource::Cop0Timer)];
            t.IsTrue(
                slot.pending,
                "the replacement Compare should re-arm the timer");
            t.Equals(
                slot.deadlineTick,
                5u * 8u,
                "the new deadline should remain anchored to canonical cycle one");
        });

        tc.Run("pending delivery does not commit a COP0 access twice", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.cop0_status =
                0x00008001u;
            runtime.writeCop0Count(
                nullptr, &ctx, 10u);
            runtime.writeCop0Compare(
                nullptr, &ctx, 11u);
            t.IsFalse(
                serviceAfterCycles(
                    runtime, ctx, 1u),
                "the timer should latch while EIE is clear");

            const uint64_t before =
                runtime.currentEeTick().raw();
            ctx.advanceEeCycleTicks(8u);
            bool interrupted = false;
            try
            {
                runtime.handleCop0Ei(
                    nullptr, &ctx);
            }
            catch (const PS2GuestException &)
            {
                interrupted = true;
            }
            t.IsTrue(
                interrupted,
                "EI should expose the pending timer");
            t.Equals(
                runtime.currentEeTick().raw(),
                before + 8u,
                "exception entry should reuse the COP0 access commit instead of adding another minimum cycle");
        });

        tc.Run("performance overflow enters the documented level-2 vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00103000u;
            const uint32_t pccr =
                kPccrCte |
                kPccrEvent0Cycles |
                kPccrK0;

            runtime.writeCop0Performance(
                nullptr, &ctx, 1u,
                0x7ffffffdu);
            runtime.writeCop0Performance(
                nullptr, &ctx, 0u, pccr);
            const auto scheduled =
                runtime.debugEeSchedulerSnapshot();
            const auto &slot =
                scheduled.slots[
                    eeEventSourceIndex(
                        EeEventSource::
                            Cop0Performance)];
            t.IsTrue(
                slot.pending,
                "an enabled PCR should own a fixed overflow source");
            t.Equals(
                slot.deadlineTick, 3u * 8u,
                "the PCR source should target the first OVFL cycle");

            t.IsTrue(
                serviceAfterCycles(
                    runtime, ctx, 3u),
                "performance overflow should unwind generated execution");
            t.Equals(
                ctx.cop0_pcr0, 0x80000000u,
                "PCR0 should expose sticky OVFL with a wrapped value");
            t.Equals(
                ctx.cop0_errorepc, 0x00103000u,
                "level-2 delivery should retain ErrorEPC");
            t.Equals(
                ctx.cop0_cause & 0x00070000u,
                0x00020000u,
                "Cause.EXC2 should identify performance overflow");
            t.IsTrue(
                (ctx.cop0_status & 0x4u) != 0u,
                "level-2 delivery should enter ERL");
            t.Equals(
                ctx.pc, 0x80000080u,
                "Status.DEV zero should select the normal performance vector");

            const uint64_t beforeReraise =
                runtime.currentEeTick().raw();
            ctx.advanceEeCycleTicks(8u);
            bool reraised = false;
            try
            {
                runtime.handleCop0Eret(
                    nullptr, &ctx);
            }
            catch (const PS2GuestException &)
            {
                reraised = true;
            }
            t.IsTrue(
                reraised,
                "ERET should re-raise while OVFL and CTE remain set");
            t.Equals(
                runtime.currentEeTick().raw(),
                beforeReraise + 8u,
                "level-2 re-entry should reuse ERET's committed cycle");

            runtime.writeCop0Performance(
                nullptr, &ctx, 1u, 0u);
            bool returned = true;
            try
            {
                runtime.handleCop0Eret(
                    nullptr, &ctx);
            }
            catch (const PS2GuestException &)
            {
                returned = false;
            }
            t.IsTrue(
                returned,
                "clearing OVFL should allow ERET to leave level two");
            t.IsFalse(
                (ctx.cop0_status & 0x4u) != 0u,
                "successful ERET should clear ERL");
        });

        tc.Run("performance overflow preserves delay restart state at the bootstrap vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x00104004u;
            ctx.branch_pc = 0x00104000u;
            ctx.in_delay_slot = true;
            ctx.cop0_status = 0x00800000u;
            const uint32_t pccr =
                kPccrCte |
                kPccrEvent0Cycles |
                kPccrK0;

            runtime.writeCop0Performance(
                nullptr, &ctx, 1u,
                0x7fffffffu);
            runtime.writeCop0Performance(
                nullptr, &ctx, 0u, pccr);
            t.IsTrue(
                serviceAfterCycles(
                    runtime, ctx, 1u),
                "the first overflow cycle should enter level two");
            t.Equals(
                ctx.cop0_errorepc,
                0x00104000u,
                "a delay-slot overflow should restart at the branch");
            t.IsTrue(
                (ctx.cop0_cause & 0x40000000u) != 0u,
                "a delay-slot overflow should set Cause.BD2");
            t.Equals(
                ctx.pc, 0xbfc00280u,
                "Status.DEV should select the documented bootstrap performance vector");
            t.IsFalse(
                ctx.in_delay_slot,
                "exception entry should leave delay-slot execution");
        });

        tc.Run("timing reset invalidates armed COP0 generations", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            runtime.writeCop0Count(
                nullptr, &ctx, 100u);
            runtime.writeCop0Compare(
                nullptr, &ctx, 200u);
            const auto before =
                runtime.debugEeSchedulerSnapshot();
            t.IsTrue(
                before.slots[
                    eeEventSourceIndex(
                        EeEventSource::Cop0Timer)]
                    .pending,
                "the pre-reset timer should be armed");

            runtime.resetEeTiming(&ctx);
            const auto after =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                after.slots[
                    eeEventSourceIndex(
                        EeEventSource::Cop0Timer)]
                    .pending,
                "timing reset should invalidate the old Compare generation");
            t.Equals(
                runtime.currentEeTick().raw(), 0u,
                "COP0 reset should share the canonical timeline reset");
        });
    });
}
