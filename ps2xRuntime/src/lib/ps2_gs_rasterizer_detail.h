#ifndef PS2_GS_RASTERIZER_DETAIL_H
#define PS2_GS_RASTERIZER_DETAIL_H

#include <array>
#include <cstdint>

namespace GSRasterizerDetail
{
    enum class PackedSpriteKernelOverride : uint8_t
    {
        Automatic,
        ForceReference,
        ForceOptimized,
    };

    void setPackedSpriteKernelOverride(
        PackedSpriteKernelOverride overrideMode);
    PackedSpriteKernelOverride packedSpriteKernelOverride();
    void resetPackedSpriteKernelDispatchCount();
    uint64_t packedSpriteKernelDispatchCount();
    void recordPackedSpriteKernelDispatch();

    struct LinearTextureTap
    {
        uint32_t raw = 0u;
        uint32_t color = 0u;
    };

    constexpr uint8_t requiredBilinearTapMask(
        uint8_t weightU, uint8_t weightV)
    {
        return static_cast<uint8_t>(
            0x1u |
            (weightU != 0u ? 0x2u : 0u) |
            (weightV != 0u ? 0x4u : 0u) |
            (weightU != 0u && weightV != 0u ? 0x8u : 0u));
    }

    template <typename ReadTap>
    inline std::array<LinearTextureTap, 4> readRequiredBilinearTaps(
        uint8_t weightU,
        uint8_t weightV,
        ReadTap &&readTap)
    {
        if (weightU != 0u && weightV != 0u)
        {
            return {
                readTap(0, 0),
                readTap(1, 0),
                readTap(0, 1),
                readTap(1, 1),
            };
        }

        const LinearTextureTap c00 = readTap(0, 0);
        if (weightU == 0u)
        {
            if (weightV == 0u)
                return {c00, c00, c00, c00};

            const LinearTextureTap c01 = readTap(0, 1);
            return {c00, c00, c01, c01};
        }

        const LinearTextureTap c10 = readTap(1, 0);
        return {c00, c10, c00, c10};
    }
}

#endif
