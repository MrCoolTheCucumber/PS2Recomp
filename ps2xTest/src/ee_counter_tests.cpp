#include "MiniTest.h"

#include "runtime/ee_counters.h"

#include <cstdint>

using namespace ps2x::timing;

namespace
{
    constexpr uint32_t counterAddress(
        size_t index,
        EeCounterRegister reg)
    {
        return kEeCounterBaseAddress +
               static_cast<uint32_t>(index) *
                   kEeCounterStride +
               static_cast<uint32_t>(reg);
    }
}

void register_ee_counter_tests()
{
    MiniTest::Case("EeCounters", [](TestCase &tc)
    {
        tc.Run("RAC1 counter1 mode uses the PCSX2 512-cycle phase", [](TestCase &t)
        {
            constexpr uint64_t kWriteCycle = 545'765'478u;
            EeCounterBank counters;
            counters.reset(kWriteCycle);

            EeCounterAdvanceResult advance{};
            t.IsTrue(
                counters.writeRegister(
                    counterAddress(
                        1u, EeCounterRegister::Mode),
                    0x82u,
                    kWriteCycle,
                    &advance),
                "counter1 mode write should decode");
            t.IsTrue(
                counters.writeRegister(
                    counterAddress(
                        1u, EeCounterRegister::Count),
                    0u,
                    kWriteCycle,
                    &advance),
                "counter1 reset write should decode");

            EeCounterBankSnapshot snapshot =
                counters.snapshot();
            t.Equals(
                snapshot.counters[1].rate,
                512u,
                "clock source two should divide the EE bus phase by 512 cycles");
            t.Equals(
                snapshot.counters[1].startCycle,
                545'765'376u,
                "mode write should align the source to PCSX2's global 512-cycle phase");

            uint32_t value = 0u;
            t.IsTrue(
                counters.readRegister(
                    counterAddress(
                        1u, EeCounterRegister::Count),
                    545'765'887u,
                    value),
                "counter1 pre-edge read should decode");
            t.Equals(
                value, 0u,
                "counter1 should remain zero one cycle before the aligned edge");
            t.IsTrue(
                counters.readRegister(
                    counterAddress(
                        1u, EeCounterRegister::Count),
                    545'765'888u,
                    value),
                "counter1 edge read should decode");
            t.Equals(
                value, 1u,
                "counter1 should increment on the aligned 512-cycle edge");
        });

        tc.Run("external source counts deterministic HBlank rising edges", [](TestCase &t)
        {
            EeCounterBank counters;
            (void)counters.configureVideoTiming(
                EeCounterVideoTiming{
                    .renderCycles = 1'000'000u,
                    .blankCycles = 100'000u,
                    .hRenderCycles = 10u,
                    .hBlankCycles = 2u,
                    .hSyncErrorCycles = 0u,
                },
                0u);
            counters.reset();

            EeCounterAdvanceResult advance{};
            t.IsTrue(
                counters.writeRegister(
                    counterAddress(
                        0u, EeCounterRegister::Mode),
                    0x83u,
                    0u,
                    &advance),
                "counter0 HBlank mode should decode");
            t.IsTrue(
                counters.writeRegister(
                    counterAddress(
                        0u, EeCounterRegister::Count),
                    0u,
                    0u,
                    &advance),
                "counter0 reset should decode");
            t.IsFalse(
                counters.nextEventCycle().has_value(),
                "RAC1's polling-only HBlank counter should not create scheduler traffic");

            uint32_t value = 0u;
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                9u,
                value);
            t.Equals(
                value, 0u,
                "external clock should not increment before HBlank");
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                10u,
                value);
            t.Equals(
                value, 1u,
                "HBlank start should increment the external clock");
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                22u,
                value);
            t.Equals(
                value, 2u,
                "the next scanline should contribute exactly one more tick");
        });

        tc.Run("RAC1 counter0 read matches the captured PCSX2 HBlank phase", [](TestCase &t)
        {
            constexpr uint64_t kHSyncStartCycle =
                5'087'551'464u;
            constexpr uint64_t kWriteCycle =
                5'087'557'068u;
            constexpr uint64_t kReadCycle =
                6'022'493'108u;

            EeCounterBank counters;
            (void)counters.configureVideoTiming(
                EeCounterVideoTiming{
                    // Keep VBlank outside this isolated HBlank oracle. The
                    // captured endpoint advances by 49,882 exact 18,743-cycle
                    // scanlines over this window.
                    .renderCycles = 2'000'000'000u,
                    .blankCycles = 100'000u,
                    .hRenderCycles = 15'685u,
                    .hBlankCycles = 3'058u,
                    .hSyncErrorCycles = 82u,
                },
                kHSyncStartCycle);
            counters.reset(kHSyncStartCycle);
            (void)counters.writeRegister(
                counterAddress(
                    0u, EeCounterRegister::Mode),
                0x83u,
                kWriteCycle);
            (void)counters.writeRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                0u,
                kWriteCycle);

            uint32_t value = 0u;
            EeCounterAdvanceResult advance{};
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                kReadCycle,
                value,
                &advance);
            t.Equals(
                advance.hBlankStarts,
                49'882ull,
                "captured interval should contain the exact PCSX2 HBlank edge count");
            t.Equals(
                value,
                0xc2dau,
                "RAC1 Timer0 read should match the captured PCSX2 count");
        });

        tc.Run("VBlank gates publish only their next relevant scan edge", [](TestCase &t)
        {
            EeCounterBank counters;
            (void)counters.configureVideoTiming(
                EeCounterVideoTiming{
                    .renderCycles = 100u,
                    .blankCycles = 20u,
                    .hRenderCycles = 10u,
                    .hBlankCycles = 2u,
                    .hSyncErrorCycles = 0u,
                },
                0u);
            counters.reset();
            (void)counters.writeRegister(
                counterAddress(
                    0u, EeCounterRegister::Mode),
                0x8cu,
                0u);

            const std::optional<uint64_t> event =
                counters.nextEventCycle();
            t.IsTrue(
                event.has_value(),
                "enabled VBlank gate should publish a scan deadline");
            if (event.has_value())
            {
                t.Equals(
                    *event, 100u,
                    "unrelated HBlank edges should not become scheduler events");
            }
        });

        tc.Run("gate-low mode pauses internal counting during HBlank", [](TestCase &t)
        {
            EeCounterBank counters;
            (void)counters.configureVideoTiming(
                EeCounterVideoTiming{
                    .renderCycles = 1'000'000u,
                    .blankCycles = 100'000u,
                    .hRenderCycles = 10u,
                    .hBlankCycles = 2u,
                    .hSyncErrorCycles = 0u,
                },
                0u);
            counters.reset();
            (void)counters.writeRegister(
                counterAddress(
                    0u, EeCounterRegister::Mode),
                0x84u,
                0u);

            uint32_t value = 0u;
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                10u,
                value);
            t.Equals(
                value, 5u,
                "BUSCLK counter should accrue five ticks during render");
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                12u,
                value);
            t.Equals(
                value, 5u,
                "gate-low mode should pause throughout HBlank");
            (void)counters.readRegister(
                counterAddress(
                    0u, EeCounterRegister::Count),
                22u,
                value);
            t.Equals(
                value, 10u,
                "counting should resume for the next render interval");
        });

        tc.Run("target and overflow events latch status and exact INTC causes", [](TestCase &t)
        {
            EeCounterBank counters;
            counters.reset();

            constexpr uint32_t kCounter2 =
                kEeCounterBaseAddress +
                2u * kEeCounterStride;
            (void)counters.writeRegister(
                kCounter2 +
                    static_cast<uint32_t>(
                        EeCounterRegister::Target),
                3u,
                0u);
            (void)counters.writeRegister(
                kCounter2 +
                    static_cast<uint32_t>(
                        EeCounterRegister::Mode),
                0x1c0u,
                0u);
            const std::optional<uint64_t> targetEvent =
                counters.nextEventCycle();
            t.IsTrue(
                targetEvent.has_value(),
                "enabled target work should publish a deadline");
            if (targetEvent.has_value())
            {
                t.Equals(
                    *targetEvent, 6u,
                    "target three on BUSCLK should mature after six EE cycles");
            }

            EeCounterAdvanceResult advance =
                counters.advanceTo(6u);
            t.IsTrue(
                (advance.newlyRaisedInterruptMask &
                 (1u << 11u)) != 0u,
                "counter2 compare should raise INTC cause 11");
            EeCounterBankSnapshot snapshot =
                counters.snapshot();
            t.Equals(
                snapshot.counters[2].count,
                0u,
                "zero return should clear count at the target");
            t.IsTrue(
                (snapshot.counters[2].mode & 0x400u) != 0u,
                "compare service should latch EQUF");

            (void)counters.writeRegister(
                kCounter2 +
                    static_cast<uint32_t>(
                        EeCounterRegister::Mode),
                0x5c0u,
                6u);
            t.IsTrue(
                (counters.snapshot().counters[2].mode &
                 0x400u) == 0u,
                "writing one should clear EQUF without changing controls");

            constexpr uint32_t kCounter3 =
                kEeCounterBaseAddress +
                3u * kEeCounterStride;
            (void)counters.writeRegister(
                kCounter3 +
                    static_cast<uint32_t>(
                        EeCounterRegister::Count),
                0xffffu,
                6u);
            (void)counters.writeRegister(
                kCounter3 +
                    static_cast<uint32_t>(
                        EeCounterRegister::Mode),
                0x280u,
                6u);
            advance = counters.advanceTo(8u);
            t.IsTrue(
                (advance.newlyRaisedInterruptMask &
                 (1u << 12u)) != 0u,
                "counter3 overflow should raise INTC cause 12");
            snapshot = counters.snapshot();
            t.Equals(
                snapshot.counters[3].count,
                0u,
                "overflow should wrap the 16-bit count");
            t.IsTrue(
                (snapshot.counters[3].mode & 0x800u) != 0u,
                "overflow service should latch OVFF");
        });
    });
}
