#ifndef PS2_VU_CLIP_H
#define PS2_VU_CLIP_H

#include <bit>
#include <cstdint>

// VU CLIP compares the sign/magnitude bit patterns rather than using the
// host's IEEE comparison rules. In particular, the reference magnitude is
// |w| and a zero or denormal w is treated as the largest denormal.
inline uint32_t Ps2VuUpdateClipFlags(uint32_t previous,
                                     uint32_t xBits,
                                     uint32_t yBits,
                                     uint32_t zBits,
                                     uint32_t wBits)
{
    const uint32_t magnitude =
        (wBits & 0x7F800000u) != 0u
            ? (wBits & 0x7FFFFFFFu)
            : 0x007FFFFFu;

    const auto outsidePositive = [magnitude](uint32_t bits)
    {
        return std::bit_cast<int32_t>(bits) >
               static_cast<int32_t>(magnitude);
    };
    const auto outsideNegative = [magnitude](uint32_t bits)
    {
        return std::bit_cast<int32_t>(bits ^ 0x80000000u) >
               static_cast<int32_t>(magnitude);
    };

    uint32_t flags = 0u;
    if (outsidePositive(xBits))
        flags |= 0x01u;
    if (outsideNegative(xBits))
        flags |= 0x02u;
    if (outsidePositive(yBits))
        flags |= 0x04u;
    if (outsideNegative(yBits))
        flags |= 0x08u;
    if (outsidePositive(zBits))
        flags |= 0x10u;
    if (outsideNegative(zBits))
        flags |= 0x20u;

    return ((previous << 6u) | flags) & 0x00FFFFFFu;
}

#endif // PS2_VU_CLIP_H
