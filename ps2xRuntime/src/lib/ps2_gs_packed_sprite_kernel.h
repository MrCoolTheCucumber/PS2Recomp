#pragma once

#include "runtime/ps2_gs_memory.h"
#include "ps2_gs_rasterizer_detail.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#define PS2X_GS_PACKED_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PS2X_GS_PACKED_NOINLINE __attribute__((noinline))
#else
#define PS2X_GS_PACKED_NOINLINE
#endif

#ifndef PS2X_GS_HAS_AVX2_KERNEL
#define PS2X_GS_HAS_AVX2_KERNEL 0
#endif

namespace GSRasterizerPacked
{
    constexpr int kPixelsPerLaneGroup = 8;
    constexpr uint32_t kRasterStripeHeight = 16u;

    struct PreparedSprite
    {
        const uint8_t *textureVram = nullptr;
        uint8_t *framebufferVram = nullptr;
        uint32_t textureBase = 0u;
        uint32_t textureWidth = 0u;
        uint32_t textureMaskU = 0u;
        uint32_t textureMaskV = 0u;
        uint32_t framebufferBase = 0u;
        uint32_t framebufferWidth = 0u;
        int drawX0 = 0;
        int drawX1 = 0;
        int drawY0 = 0;
        int drawY1 = 0;
        int alignedDrawX = 0;
        int32_t fixedBaseU = 0;
        int32_t fixedBlockStepU = 0;
        std::array<int32_t, kPixelsPerLaneGroup> fixedLaneU{};
        float fixedScanV = 0.0f;
        float fixedStepV = 0.0f;
        uint32_t scanlineWorkerIndex = 0u;
        uint32_t scanlineWorkerCount = 1u;
        bool recordVectorGroups = false;
    };

    struct LaneCoordinates
    {
        uint32_t u0;
        uint32_t u1;
        uint8_t weightU;
    };

    using DrawKernel = void (*)(const PreparedSprite &sprite);

    inline void loadScalarGroup(
        const PreparedSprite &sprite,
        const LaneCoordinates *coordinates,
        uint32_t v,
        uint32_t *result)
    {
        for (int lane = 0;
             lane < kPixelsPerLaneGroup;
             ++lane)
        {
            result[lane] = GSMem::ReadCT32(
                const_cast<uint8_t *>(sprite.textureVram),
                sprite.textureBase,
                sprite.textureWidth,
                coordinates[lane].u0,
                v);
        }
    }

    inline int arithmeticShiftRight4(int value)
    {
        return value >= 0 ? (value / 16) : -((-value + 15) / 16);
    }

    inline int floorFixed16_16(int32_t value)
    {
        int quotient = value / 65536;
        if (value < 0 && (value % 65536) != 0)
            --quotient;
        return quotient;
    }

    inline uint8_t clampU8(int value)
    {
        return static_cast<uint8_t>(
            std::clamp(value, 0, 255));
    }

    inline uint8_t lerpChannel4(uint8_t c00,
                                uint8_t c10,
                                uint8_t c01,
                                uint8_t c11,
                                uint8_t weightU,
                                uint8_t weightV)
    {
        const auto lerp = [](int from, int to, int weight)
        {
            return from +
                   arithmeticShiftRight4((to - from) * weight);
        };
        const int top = lerp(c00, c10, weightU);
        const int bottom = lerp(c01, c11, weightU);
        return clampU8(lerp(top, bottom, weightV));
    }

    inline uint32_t linearColor4(uint32_t from,
                                 uint32_t to,
                                 uint8_t weight)
    {
#if defined(__SSE4_1__)
        const __m128i fromChannels =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(from));
        const __m128i toChannels =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(to));
        __m128i result = _mm_add_epi32(
            fromChannels,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(toChannels, fromChannels),
                    _mm_set1_epi32(weight)),
                4));
        result = _mm_min_epi32(
            _mm_max_epi32(result, _mm_setzero_si128()),
            _mm_set1_epi32(255));
        return static_cast<uint32_t>(_mm_cvtsi128_si32(
            _mm_packus_epi16(
                _mm_packus_epi32(result, result),
                _mm_setzero_si128())));
#else
        uint32_t result = 0u;
        for (uint32_t shift = 0u; shift < 32u; shift += 8u)
        {
            const int fromChannel =
                static_cast<int>((from >> shift) & 0xFFu);
            const int toChannel =
                static_cast<int>((to >> shift) & 0xFFu);
            const int channel =
                fromChannel +
                arithmeticShiftRight4(
                    (toChannel - fromChannel) * weight);
            result |=
                static_cast<uint32_t>(clampU8(channel)) << shift;
        }
        return result;
#endif
    }

    inline uint32_t bilinearColor4(uint32_t c00,
                                   uint32_t c10,
                                   uint32_t c01,
                                   uint32_t c11,
                                   uint8_t weightU,
                                   uint8_t weightV)
    {
#if defined(__SSE4_1__)
        const __m128i zero = _mm_setzero_si128();
        const __m128i maximum = _mm_set1_epi32(255);
        const __m128i u = _mm_set1_epi32(weightU);
        const __m128i v = _mm_set1_epi32(weightV);
        const __m128i color00 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c00));
        const __m128i color10 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c10));
        const __m128i color01 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c01));
        const __m128i color11 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c11));
        const __m128i top = _mm_add_epi32(
            color00,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(color10, color00), u),
                4));
        const __m128i bottom = _mm_add_epi32(
            color01,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(color11, color01), u),
                4));
        __m128i result = _mm_add_epi32(
            top,
            _mm_srai_epi32(
                _mm_mullo_epi32(_mm_sub_epi32(bottom, top), v),
                4));
        result = _mm_min_epi32(
            _mm_max_epi32(result, zero), maximum);
        return static_cast<uint32_t>(_mm_cvtsi128_si32(
            _mm_packus_epi16(
                _mm_packus_epi32(result, result), zero)));
#else
        const uint8_t r = lerpChannel4(
            static_cast<uint8_t>(c00),
            static_cast<uint8_t>(c10),
            static_cast<uint8_t>(c01),
            static_cast<uint8_t>(c11),
            weightU,
            weightV);
        const uint8_t g = lerpChannel4(
            static_cast<uint8_t>(c00 >> 8u),
            static_cast<uint8_t>(c10 >> 8u),
            static_cast<uint8_t>(c01 >> 8u),
            static_cast<uint8_t>(c11 >> 8u),
            weightU,
            weightV);
        const uint8_t b = lerpChannel4(
            static_cast<uint8_t>(c00 >> 16u),
            static_cast<uint8_t>(c10 >> 16u),
            static_cast<uint8_t>(c01 >> 16u),
            static_cast<uint8_t>(c11 >> 16u),
            weightU,
            weightV);
        const uint8_t a = lerpChannel4(
            static_cast<uint8_t>(c00 >> 24u),
            static_cast<uint8_t>(c10 >> 24u),
            static_cast<uint8_t>(c01 >> 24u),
            static_cast<uint8_t>(c11 >> 24u),
            weightU,
            weightV);
        return static_cast<uint32_t>(r) |
               (static_cast<uint32_t>(g) << 8u) |
               (static_cast<uint32_t>(b) << 16u) |
               (static_cast<uint32_t>(a) << 24u);
#endif
    }

    struct ScalarVerticalInterpolator
    {
        static constexpr bool enabled = false;
    };

#if defined(__SSE4_1__)
    struct Sse41VerticalInterpolator
    {
        static constexpr bool enabled = true;

        static inline void load(
            const PreparedSprite &sprite,
            const LaneCoordinates *coordinates,
            uint32_t v,
            uint32_t *result)
        {
            loadScalarGroup(
                sprite, coordinates, v, result);
        }

        static inline void interpolate(
            const uint32_t *from,
            const uint32_t *to,
            uint8_t weight,
            uint32_t *result)
        {
            const __m128i zero = _mm_setzero_si128();
            const __m128i vectorWeight =
                _mm_set1_epi16(static_cast<int16_t>(weight));
            for (int offset = 0;
                 offset < kPixelsPerLaneGroup;
                 offset += 4)
            {
                const __m128i fromPacked = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(
                        from + offset));
                const __m128i toPacked = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(
                        to + offset));
                const __m128i fromLow =
                    _mm_unpacklo_epi8(fromPacked, zero);
                const __m128i fromHigh =
                    _mm_unpackhi_epi8(fromPacked, zero);
                const __m128i toLow =
                    _mm_unpacklo_epi8(toPacked, zero);
                const __m128i toHigh =
                    _mm_unpackhi_epi8(toPacked, zero);
                const __m128i resultLow = _mm_add_epi16(
                    fromLow,
                    _mm_srai_epi16(
                        _mm_mullo_epi16(
                            _mm_sub_epi16(toLow, fromLow),
                            vectorWeight),
                        4));
                const __m128i resultHigh = _mm_add_epi16(
                    fromHigh,
                    _mm_srai_epi16(
                        _mm_mullo_epi16(
                            _mm_sub_epi16(toHigh, fromHigh),
                            vectorWeight),
                        4));
                _mm_storeu_si128(
                    reinterpret_cast<__m128i *>(
                        result + offset),
                    _mm_packus_epi16(resultLow, resultHigh));
            }
        }
    };
#endif

#if defined(__AVX2__) || defined(_M_AVX2)
    struct Avx2VerticalInterpolator
    {
        static constexpr bool enabled = true;

        static inline void load(
            const PreparedSprite &sprite,
            const LaneCoordinates *coordinates,
            uint32_t v,
            uint32_t *result)
        {
            loadScalarGroup(
                sprite, coordinates, v, result);
        }

        static inline void interpolate(
            const uint32_t *from,
            const uint32_t *to,
            uint8_t weight,
            uint32_t *result)
        {
            const __m256i zero = _mm256_setzero_si256();
            const __m256i vectorWeight =
                _mm256_set1_epi16(static_cast<int16_t>(weight));
            const __m256i fromPacked = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(from));
            const __m256i toPacked = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(to));
            const __m256i fromLow =
                _mm256_unpacklo_epi8(fromPacked, zero);
            const __m256i fromHigh =
                _mm256_unpackhi_epi8(fromPacked, zero);
            const __m256i toLow =
                _mm256_unpacklo_epi8(toPacked, zero);
            const __m256i toHigh =
                _mm256_unpackhi_epi8(toPacked, zero);
            const __m256i resultLow = _mm256_add_epi16(
                fromLow,
                _mm256_srai_epi16(
                    _mm256_mullo_epi16(
                        _mm256_sub_epi16(toLow, fromLow),
                        vectorWeight),
                    4));
            const __m256i resultHigh = _mm256_add_epi16(
                fromHigh,
                _mm256_srai_epi16(
                    _mm256_mullo_epi16(
                        _mm256_sub_epi16(toHigh, fromHigh),
                        vectorWeight),
                    4));
            _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(result),
                _mm256_packus_epi16(resultLow, resultHigh));
        }
    };
#endif

    template <typename VerticalInterpolator>
    inline void drawCt32DecalSpriteKernel(
        const PreparedSprite &sprite)
    {
        const auto prepareLane =
            [&](int x) -> LaneCoordinates
        {
            const int block =
                (x - sprite.alignedDrawX) /
                kPixelsPerLaneGroup;
            const int32_t fixedLaneU =
                sprite.fixedLaneU[
                    x & (kPixelsPerLaneGroup - 1)];
            const int32_t fixedU =
                static_cast<int32_t>(
                    static_cast<uint32_t>(
                        sprite.fixedBaseU) +
                    static_cast<uint32_t>(fixedLaneU) +
                    static_cast<uint32_t>(
                        sprite.fixedBlockStepU) *
                        static_cast<uint32_t>(block));
            const uint8_t weightU =
                static_cast<uint8_t>(
                    (static_cast<uint32_t>(fixedU) >> 12u) &
                    0xFu);
            const uint32_t wrappedU0 =
                static_cast<uint32_t>(
                    floorFixed16_16(fixedU)) &
                sprite.textureMaskU;
            const uint32_t wrappedU1 =
                weightU != 0u
                    ? (wrappedU0 + 1u) &
                          sprite.textureMaskU
                    : wrappedU0;
            return {
                wrappedU0,
                wrappedU1,
                weightU,
            };
        };

        std::vector<LaneCoordinates> cachedCoordinates;
        bool cachedCoordinatesUseOnlyVerticalInterpolation =
            true;
        if constexpr (VerticalInterpolator::enabled)
        {
            const int pixelCount =
                sprite.drawX1 - sprite.drawX0 + 1;
            cachedCoordinates.reserve(
                static_cast<size_t>(pixelCount));
            for (int x = sprite.drawX0;
                 x <= sprite.drawX1;
                 ++x)
            {
                const LaneCoordinates coordinates =
                    prepareLane(x);
                cachedCoordinatesUseOnlyVerticalInterpolation &=
                    coordinates.weightU == 0u;
                cachedCoordinates.push_back(coordinates);
            }
        }

        float fixedScanV = sprite.fixedScanV;
        for (int y = sprite.drawY0; y <= sprite.drawY1; ++y)
        {
            const bool ownsScanline =
                sprite.scanlineWorkerCount <= 1u ||
                (static_cast<uint32_t>(y) /
                 kRasterStripeHeight) %
                        sprite.scanlineWorkerCount ==
                    sprite.scanlineWorkerIndex;
            if (!ownsScanline)
            {
                fixedScanV += sprite.fixedStepV;
                continue;
            }

            const int32_t fixedV =
                static_cast<int32_t>(fixedScanV);
            const uint8_t weightV =
                static_cast<uint8_t>(
                    (static_cast<uint32_t>(fixedV) >> 12u) &
                    0xFu);
            const uint32_t wrappedV0 =
                static_cast<uint32_t>(
                    floorFixed16_16(fixedV)) &
                sprite.textureMaskV;
            const uint32_t wrappedV1 =
                weightV != 0u
                    ? (wrappedV0 + 1u) &
                          sprite.textureMaskV
                    : wrappedV0;

            const int firstGroup =
                sprite.drawX0 &
                ~(kPixelsPerLaneGroup - 1);
            for (int groupX = firstGroup;
                 groupX <= sprite.drawX1;
                 groupX += kPixelsPerLaneGroup)
            {
                std::array<uint32_t,
                           kPixelsPerLaneGroup> pending{};
                int pendingCount = 0;
                const int firstX =
                    std::max(groupX, sprite.drawX0);
                const int lastX = std::min(
                    groupX + kPixelsPerLaneGroup - 1,
                    sprite.drawX1);

                const auto sampleLane =
                    [&](LaneCoordinates coordinates)
                        -> uint32_t
                {
                    const uint32_t c00 = GSMem::ReadCT32(
                        const_cast<uint8_t *>(
                            sprite.textureVram),
                        sprite.textureBase,
                        sprite.textureWidth,
                        coordinates.u0,
                        wrappedV0);

                    uint32_t color = c00;
                    if (coordinates.weightU == 0u)
                    {
                        if (weightV != 0u)
                        {
                            const uint32_t c01 =
                                GSMem::ReadCT32(
                                    const_cast<uint8_t *>(
                                        sprite.textureVram),
                                    sprite.textureBase,
                                    sprite.textureWidth,
                                    coordinates.u0,
                                    wrappedV1);
                            color = linearColor4(
                                c00, c01, weightV);
                        }
                    }
                    else
                    {
                        const uint32_t c10 =
                            GSMem::ReadCT32(
                                const_cast<uint8_t *>(
                                    sprite.textureVram),
                                sprite.textureBase,
                                sprite.textureWidth,
                                coordinates.u1,
                                wrappedV0);
                        if (weightV == 0u)
                        {
                            color = linearColor4(
                                c00,
                                c10,
                                coordinates.weightU);
                        }
                        else
                        {
                            const uint32_t c01 =
                                GSMem::ReadCT32(
                                    const_cast<uint8_t *>(
                                        sprite.textureVram),
                                    sprite.textureBase,
                                    sprite.textureWidth,
                                    coordinates.u0,
                                    wrappedV1);
                            const uint32_t c11 =
                                GSMem::ReadCT32(
                                    const_cast<uint8_t *>(
                                        sprite.textureVram),
                                    sprite.textureBase,
                                    sprite.textureWidth,
                                    coordinates.u1,
                                    wrappedV1);
                            color = bilinearColor4(
                                c00,
                                c10,
                                c01,
                                c11,
                                coordinates.weightU,
                                weightV);
                        }
                    }
                    return color;
                };

                const bool isFullGroup =
                    firstX == groupX &&
                    lastX ==
                        groupX + kPixelsPerLaneGroup - 1;
                if constexpr (VerticalInterpolator::enabled)
                {
                    if (isFullGroup &&
                        cachedCoordinatesUseOnlyVerticalInterpolation)
                    {
                        const LaneCoordinates *coordinates =
                            cachedCoordinates.data() +
                            (groupX - sprite.drawX0);

                        std::array<uint32_t,
                                   kPixelsPerLaneGroup>
                            top{};
                        VerticalInterpolator::load(
                            sprite,
                            coordinates,
                            wrappedV0,
                            top.data());

                        pendingCount = kPixelsPerLaneGroup;
                        if (weightV == 0u)
                        {
                            pending = top;
                        }
                        else
                        {
                            std::array<uint32_t,
                                       kPixelsPerLaneGroup>
                                bottom{};
                            VerticalInterpolator::load(
                                sprite,
                                coordinates,
                                wrappedV1,
                                bottom.data());
                            VerticalInterpolator::interpolate(
                                top.data(),
                                bottom.data(),
                                weightV,
                                pending.data());
                            if (sprite.recordVectorGroups)
                            {
                                GSRasterizerDetail::
                                    recordPackedSpriteVectorGroup();
                            }
                        }
                    }
                    else
                    {
                        for (int x = firstX; x <= lastX; ++x)
                        {
                            pending[pendingCount++] =
                                sampleLane(
                                    cachedCoordinates[
                                        x - sprite.drawX0]);
                        }
                    }
                }
                else
                {
                    for (int x = firstX; x <= lastX; ++x)
                    {
                        pending[pendingCount++] =
                            sampleLane(prepareLane(x));
                    }
                }

                for (int index = 0;
                     index < pendingCount;
                     ++index)
                {
                    GSMem::WriteCT32(
                        sprite.framebufferVram,
                        sprite.framebufferBase,
                        sprite.framebufferWidth,
                        firstX + index,
                        y,
                        pending[index]);
                }
            }

            fixedScanV += sprite.fixedStepV;
        }
    }

    PS2X_GS_PACKED_NOINLINE inline void drawScalar(
        const PreparedSprite &sprite)
    {
        drawCt32DecalSpriteKernel<ScalarVerticalInterpolator>(
            sprite);
    }

#if defined(__SSE4_1__)
    PS2X_GS_PACKED_NOINLINE inline void drawSse41(
        const PreparedSprite &sprite)
    {
        drawCt32DecalSpriteKernel<Sse41VerticalInterpolator>(
            sprite);
    }
#endif

#if PS2X_GS_HAS_AVX2_KERNEL
    void drawAvx2(const PreparedSprite &sprite);
#endif
}

#undef PS2X_GS_PACKED_NOINLINE
