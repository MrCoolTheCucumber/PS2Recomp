#pragma once

#include "ps2_runtime.h"

#include <cstdint>

namespace ps2x::test::observation
{
    inline constexpr uint32_t kInstructionsPerBlock = 4u;
    inline constexpr uint32_t kCycleTicksPerInstruction = 8u;
    inline constexpr uint32_t kCodeBase = 0x00190000u;
    inline constexpr uint32_t kPhysicalBase = 0x00800000u;
    inline constexpr uint32_t kDirectBase = 0x80000000u | kPhysicalBase;
    inline constexpr uint32_t kWorkingSetBytes = 4096u;
    inline constexpr uint32_t kWorkingSetWords =
        kWorkingSetBytes / sizeof(uint32_t);
    inline constexpr uint32_t kAddressSlots = 256u;

    extern "C" void EeObservationTemplateFast(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime);
    extern "C" void EeObservationTemplatePrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime);
    extern "C" void EeObservationExplicitFast(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime);
    extern "C" void EeObservationExplicitPrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime);
    extern "C" void EeObservationCompactPrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime);
    extern "C" void EeObservationOutlinedPrecise(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime);
}
