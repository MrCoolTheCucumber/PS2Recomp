#ifndef PS2_WATCHDOG_H
#define PS2_WATCHDOG_H

#include <cstdint>

enum class PS2WatchdogClassification : uint8_t
{
    Presenting,
    ComputeBoundGs,
    ComputeBoundVu,
    RepeatedGuestPc,
    WaitingOrDeadlock,
    ActiveNoPresentation,
    StalledNoProgress,
};

struct PS2WatchdogSample
{
    uint64_t dispatches = 0u;
    uint64_t eeInstructions = 0u;
    uint32_t pc = 0u;
    uint64_t dmaStarts = 0u;
    uint64_t gifCopies = 0u;
    uint64_t gsWrites = 0u;
    uint64_t vifWrites = 0u;
    uint64_t presentations = 0u;
    uint64_t gsDrawsStarted = 0u;
    uint64_t gsDrawsCompleted = 0u;
    uint64_t gsCandidatePixels = 0u;
    uint32_t gsActiveDraws = 0u;
    uint64_t vu0Cycles = 0u;
    uint64_t vu1Cycles = 0u;
    bool vu0Active = false;
    bool vu1Active = false;
    uint32_t guestThreadCount = 0u;
    uint32_t waitingThreadCount = 0u;
    uint32_t guestExecutionWaiters = 0u;
};

struct PS2WatchdogAssessment
{
    PS2WatchdogClassification classification =
        PS2WatchdogClassification::StalledNoProgress;
    bool madeProgress = false;
    const char *name = "stalled-no-progress";
    const char *hotOperation = "";
};

inline bool ps2WatchdogCounterAdvanced(
    const PS2WatchdogSample &previous,
    const PS2WatchdogSample &current)
{
    return
        current.dispatches > previous.dispatches ||
        current.eeInstructions > previous.eeInstructions ||
        current.dmaStarts > previous.dmaStarts ||
        current.gifCopies > previous.gifCopies ||
        current.gsWrites > previous.gsWrites ||
        current.vifWrites > previous.vifWrites ||
        current.gsDrawsStarted > previous.gsDrawsStarted ||
        current.gsDrawsCompleted > previous.gsDrawsCompleted ||
        current.gsCandidatePixels > previous.gsCandidatePixels ||
        current.vu0Cycles > previous.vu0Cycles ||
        current.vu1Cycles > previous.vu1Cycles;
}

inline PS2WatchdogAssessment ps2ClassifyWatchdogSample(
    const PS2WatchdogSample &previous,
    const PS2WatchdogSample &current)
{
    if (current.presentations > previous.presentations)
    {
        return {
            PS2WatchdogClassification::Presenting,
            true,
            "presenting",
            "",
        };
    }

    if (current.gsActiveDraws != 0u &&
        (current.gsCandidatePixels > previous.gsCandidatePixels ||
         current.gsDrawsStarted > previous.gsDrawsStarted ||
         current.gsDrawsCompleted > previous.gsDrawsCompleted))
    {
        return {
            PS2WatchdogClassification::ComputeBoundGs,
            true,
            "compute-bound-gs",
            "GSRasterizer::writePixel",
        };
    }

    if ((current.vu1Active && current.vu1Cycles > previous.vu1Cycles) ||
        (current.vu0Active && current.vu0Cycles > previous.vu0Cycles))
    {
        return {
            PS2WatchdogClassification::ComputeBoundVu,
            true,
            "compute-bound-vu",
            current.vu1Active ? "VuUnit::run"
                              : "VU0Interpreter::run",
        };
    }

    if (current.dispatches > previous.dispatches &&
        current.pc == previous.pc)
    {
        return {
            PS2WatchdogClassification::RepeatedGuestPc,
            true,
            "repeated-guest-pc",
            "PS2Runtime::dispatchLoop",
        };
    }

    const bool allKnownThreadsWaiting =
        current.guestThreadCount != 0u &&
        current.waitingThreadCount >= current.guestThreadCount;
    if (current.dispatches == previous.dispatches &&
        current.gsActiveDraws == 0u &&
        !current.vu0Active &&
        !current.vu1Active &&
        (allKnownThreadsWaiting || current.guestExecutionWaiters != 0u))
    {
        return {
            PS2WatchdogClassification::WaitingOrDeadlock,
            false,
            "waiting-or-deadlock",
            "",
        };
    }

    if (ps2WatchdogCounterAdvanced(previous, current))
    {
        return {
            PS2WatchdogClassification::ActiveNoPresentation,
            true,
            "active-no-presentation",
            "",
        };
    }

    return {
        PS2WatchdogClassification::StalledNoProgress,
        false,
        "stalled-no-progress",
        "",
    };
}

#endif
