#ifndef PS2_RUNTIME_MACROS_H
#define PS2_RUNTIME_MACROS_H
#include <cstdint>
#include <cmath>
#include <cstring>
#include <bit>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX intrinsics
#endif

#include "ps2_runtime.h"

static inline int32_t Ps2ExtractEpi32(__m128i v, int index)
{
    switch (index & 3)
    {
    case 0:
        return _mm_extract_epi32(v, 0);
    case 1:
        return _mm_extract_epi32(v, 1);
    case 2:
        return _mm_extract_epi32(v, 2);
    default:
        return _mm_extract_epi32(v, 3);
    }
}

static inline int64_t Ps2ExtractEpi64(__m128i v, int index)
{
    if ((index & 1) == 0)
    {
        return _mm_cvtsi128_si64(v);
    }
    else
    {
        return _mm_extract_epi64(v, 1);
    }
}

static inline uint32_t ps2_clz32(uint32_t x)
{
    return static_cast<uint32_t>(std::countl_zero(x));
}

static inline uint64_t Ps2HiLoToU64(uint64_t hi, uint64_t lo)
{
    return ((hi & 0xFFFFFFFFull) << 32) | (lo & 0xFFFFFFFFull);
}

static inline uint64_t Ps2SignExt32ToU64(uint32_t v)
{
    return (uint64_t)(int64_t)(int32_t)v;
}

static inline uint64_t Ps2MaddSigned32(uint64_t accumulator, int32_t lhs, int32_t rhs)
{
    const int64_t product = static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
    return accumulator + static_cast<uint64_t>(product);
}

static inline uint64_t Ps2MsubSigned32(uint64_t accumulator, int32_t lhs, int32_t rhs)
{
    const int64_t product = static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
    return accumulator - static_cast<uint64_t>(product);
}

static inline __m128i Ps2MakeU32Vector(uint32_t word0, uint32_t word1,
                                      uint32_t word2, uint32_t word3)
{
    const uint32_t words[4] = {word0, word1, word2, word3};
    __m128i result;
    std::memcpy(&result, words, sizeof(result));
    return result;
}

static inline __m128i Ps2MakeU16Vector(uint16_t half0, uint16_t half1,
                                      uint16_t half2, uint16_t half3,
                                      uint16_t half4, uint16_t half5,
                                      uint16_t half6, uint16_t half7)
{
    const uint16_t halves[8] = {
        half0, half1, half2, half3, half4, half5, half6, half7};
    __m128i result;
    std::memcpy(&result, halves, sizeof(result));
    return result;
}

static inline __m128i Ps2MakeU64Vector(uint64_t doubleword0, uint64_t doubleword1)
{
    const uint64_t doublewords[2] = {doubleword0, doubleword1};
    __m128i result;
    std::memcpy(&result, doublewords, sizeof(result));
    return result;
}

static inline __m128i Ps2VuFtoi(__m128 value, float scale)
{
    const __m128 scaled = _mm_mul_ps(value, _mm_set1_ps(scale));
    const __m128 largestInRange =
        _mm_castsi128_ps(_mm_set1_epi32(0x4EFFFFFF));
    const __m128 positiveOverflow = _mm_cmpgt_ps(scaled, largestInRange);
    const __m128i truncated = _mm_cvttps_epi32(scaled);

    // CVTTPS2DQ returns INT_MIN for every unrepresentable input. The VUs
    // instead saturate positive overflow to INT_MAX while retaining INT_MIN
    // for negative overflow and NaN.
    return _mm_xor_si128(truncated, _mm_castps_si128(positiveOverflow));
}

static inline __m128i Ps2GetHi128(const R5900Context *ctx)
{
    return Ps2MakeU64Vector(ctx->hi, ctx->hi1);
}

static inline __m128i Ps2GetLo128(const R5900Context *ctx)
{
    return Ps2MakeU64Vector(ctx->lo, ctx->lo1);
}

static inline void Ps2SetHi128(R5900Context *ctx, __m128i value)
{
    uint64_t doublewords[2];
    std::memcpy(doublewords, &value, sizeof(value));
    ctx->hi = doublewords[0];
    ctx->hi1 = doublewords[1];
}

static inline void Ps2SetLo128(R5900Context *ctx, __m128i value)
{
    uint64_t doublewords[2];
    std::memcpy(doublewords, &value, sizeof(value));
    ctx->lo = doublewords[0];
    ctx->lo1 = doublewords[1];
}

static inline uint32_t Ps2LowWord(uint64_t value)
{
    return static_cast<uint32_t>(value);
}

static inline uint32_t Ps2HighWord(uint64_t value)
{
    return static_cast<uint32_t>(value >> 32);
}

static inline uint16_t Ps2LowHalf(uint32_t value)
{
    return static_cast<uint16_t>(value);
}

static inline uint16_t Ps2ClampSignedWordToHalf(uint32_t bits)
{
    const int32_t value = std::bit_cast<int32_t>(bits);
    if (value > INT16_MAX)
    {
        return static_cast<uint16_t>(INT16_MAX);
    }
    if (value < INT16_MIN)
    {
        return 0x8000u;
    }
    return static_cast<uint16_t>(value);
}

static inline uint64_t Ps2ClampSignedDoublewordToWord(uint64_t bits)
{
    const int64_t value = std::bit_cast<int64_t>(bits);
    if (value > INT32_MAX)
    {
        return static_cast<uint64_t>(INT32_MAX);
    }
    if (value < INT32_MIN)
    {
        return Ps2SignExt32ToU64(0x80000000u);
    }
    return Ps2SignExt32ToU64(static_cast<uint32_t>(bits));
}

static inline __m128i Ps2PmfhlLw(const R5900Context *ctx)
{
    return Ps2MakeU32Vector(Ps2LowWord(ctx->lo), Ps2LowWord(ctx->hi),
                            Ps2LowWord(ctx->lo1), Ps2LowWord(ctx->hi1));
}

static inline __m128i Ps2PmfhlUw(const R5900Context *ctx)
{
    return Ps2MakeU32Vector(Ps2HighWord(ctx->lo), Ps2HighWord(ctx->hi),
                            Ps2HighWord(ctx->lo1), Ps2HighWord(ctx->hi1));
}

static inline __m128i Ps2PmfhlSlw(const R5900Context *ctx)
{
    const uint64_t lane0 =
        (static_cast<uint64_t>(Ps2LowWord(ctx->hi)) << 32) | Ps2LowWord(ctx->lo);
    const uint64_t lane1 =
        (static_cast<uint64_t>(Ps2LowWord(ctx->hi1)) << 32) | Ps2LowWord(ctx->lo1);
    return Ps2MakeU64Vector(Ps2ClampSignedDoublewordToWord(lane0),
                            Ps2ClampSignedDoublewordToWord(lane1));
}

static inline __m128i Ps2PmfhlLh(const R5900Context *ctx)
{
    return Ps2MakeU16Vector(
        Ps2LowHalf(Ps2LowWord(ctx->lo)), Ps2LowHalf(Ps2HighWord(ctx->lo)),
        Ps2LowHalf(Ps2LowWord(ctx->hi)), Ps2LowHalf(Ps2HighWord(ctx->hi)),
        Ps2LowHalf(Ps2LowWord(ctx->lo1)), Ps2LowHalf(Ps2HighWord(ctx->lo1)),
        Ps2LowHalf(Ps2LowWord(ctx->hi1)), Ps2LowHalf(Ps2HighWord(ctx->hi1)));
}

static inline __m128i Ps2PmfhlSh(const R5900Context *ctx)
{
    return Ps2MakeU16Vector(
        Ps2ClampSignedWordToHalf(Ps2LowWord(ctx->lo)),
        Ps2ClampSignedWordToHalf(Ps2HighWord(ctx->lo)),
        Ps2ClampSignedWordToHalf(Ps2LowWord(ctx->hi)),
        Ps2ClampSignedWordToHalf(Ps2HighWord(ctx->hi)),
        Ps2ClampSignedWordToHalf(Ps2LowWord(ctx->lo1)),
        Ps2ClampSignedWordToHalf(Ps2HighWord(ctx->lo1)),
        Ps2ClampSignedWordToHalf(Ps2LowWord(ctx->hi1)),
        Ps2ClampSignedWordToHalf(Ps2HighWord(ctx->hi1)));
}

static inline void Ps2PmthlLw(R5900Context *ctx, __m128i value)
{
    uint32_t words[4];
    std::memcpy(words, &value, sizeof(value));
    ctx->lo = (ctx->lo & 0xFFFFFFFF00000000ull) | words[0];
    ctx->hi = (ctx->hi & 0xFFFFFFFF00000000ull) | words[1];
    ctx->lo1 = (ctx->lo1 & 0xFFFFFFFF00000000ull) | words[2];
    ctx->hi1 = (ctx->hi1 & 0xFFFFFFFF00000000ull) | words[3];
}

static inline __m128i Ps2Pmulth(R5900Context *ctx, __m128i lhs, __m128i rhs)
{
    int16_t lhsHalves[8];
    int16_t rhsHalves[8];
    uint32_t products[8];
    std::memcpy(lhsHalves, &lhs, sizeof(lhs));
    std::memcpy(rhsHalves, &rhs, sizeof(rhs));

    for (size_t i = 0; i < 8; ++i)
    {
        const int32_t product =
            static_cast<int32_t>(lhsHalves[i]) * static_cast<int32_t>(rhsHalves[i]);
        products[i] = std::bit_cast<uint32_t>(product);
    }

    ctx->lo = (static_cast<uint64_t>(products[1]) << 32) | products[0];
    ctx->hi = (static_cast<uint64_t>(products[3]) << 32) | products[2];
    ctx->lo1 = (static_cast<uint64_t>(products[5]) << 32) | products[4];
    ctx->hi1 = (static_cast<uint64_t>(products[7]) << 32) | products[6];
    return Ps2MakeU32Vector(products[0], products[2], products[4], products[6]);
}

static inline __m128i Ps2Pmultuw(R5900Context *ctx, __m128i lhs, __m128i rhs)
{
    uint32_t lhsWords[4];
    uint32_t rhsWords[4];
    std::memcpy(lhsWords, &lhs, sizeof(lhs));
    std::memcpy(rhsWords, &rhs, sizeof(rhs));

    const uint64_t product0 =
        static_cast<uint64_t>(lhsWords[0]) * static_cast<uint64_t>(rhsWords[0]);
    const uint64_t product1 =
        static_cast<uint64_t>(lhsWords[2]) * static_cast<uint64_t>(rhsWords[2]);

    ctx->lo = Ps2SignExt32ToU64(Ps2LowWord(product0));
    ctx->hi = Ps2SignExt32ToU64(Ps2HighWord(product0));
    ctx->lo1 = Ps2SignExt32ToU64(Ps2LowWord(product1));
    ctx->hi1 = Ps2SignExt32ToU64(Ps2HighWord(product1));
    return Ps2MakeU64Vector(product0, product1);
}

static inline __m128i Ps2Pmultw(R5900Context *ctx, __m128i lhs, __m128i rhs)
{
    int32_t lhsWords[4];
    int32_t rhsWords[4];
    std::memcpy(lhsWords, &lhs, sizeof(lhs));
    std::memcpy(rhsWords, &rhs, sizeof(rhs));

    const int64_t signedProduct0 =
        static_cast<int64_t>(lhsWords[0]) * static_cast<int64_t>(rhsWords[0]);
    const int64_t signedProduct1 =
        static_cast<int64_t>(lhsWords[2]) * static_cast<int64_t>(rhsWords[2]);
    const uint64_t product0 = std::bit_cast<uint64_t>(signedProduct0);
    const uint64_t product1 = std::bit_cast<uint64_t>(signedProduct1);

    ctx->lo = Ps2SignExt32ToU64(Ps2LowWord(product0));
    ctx->hi = Ps2SignExt32ToU64(Ps2HighWord(product0));
    ctx->lo1 = Ps2SignExt32ToU64(Ps2LowWord(product1));
    ctx->hi1 = Ps2SignExt32ToU64(Ps2HighWord(product1));
    return Ps2MakeU64Vector(product0, product1);
}

static inline uint64_t Ps2PmaddwLane(uint32_t initialHi,
                                     uint32_t initialLo,
                                     int32_t lhs,
                                     int32_t rhs,
                                     bool applyLowerLaneCorrection)
{
    const int64_t signedProduct =
        static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
    const uint64_t product = std::bit_cast<uint64_t>(signedProduct);

    // The EE multiplier does not carry the low-word addition into HI.
    // Its HI path also divides by 0xffffffff rather than shifting by 32,
    // with one additional correction that exists only on packed lane zero.
    uint64_t highNumerator =
        product + (static_cast<uint64_t>(initialHi) << 32u);
    const uint32_t rhsBits = std::bit_cast<uint32_t>(rhs);
    if (applyLowerLaneCorrection &&
        ((rhsBits & 0x7fffffffu) == 0u ||
         (rhsBits & 0x7fffffffu) == 0x7fffffffu) &&
        lhs != rhs)
    {
        highNumerator += 0x70000000u;
    }

    constexpr int64_t highDivisor = 0xffffffffll;
    const int64_t signedNumerator =
        std::bit_cast<int64_t>(highNumerator);
    const uint32_t highResult = static_cast<uint32_t>(
        signedNumerator / highDivisor);
    const uint32_t lowResult =
        initialLo + Ps2LowWord(product);
    return (static_cast<uint64_t>(highResult) << 32u) | lowResult;
}

static inline __m128i Ps2Pmaddw(R5900Context *ctx, __m128i lhs, __m128i rhs)
{
    int32_t lhsWords[4];
    int32_t rhsWords[4];
    std::memcpy(lhsWords, &lhs, sizeof(lhs));
    std::memcpy(rhsWords, &rhs, sizeof(rhs));

    const uint64_t result0 = Ps2PmaddwLane(
        Ps2LowWord(ctx->hi),
        Ps2LowWord(ctx->lo),
        lhsWords[0],
        rhsWords[0],
        true);
    const uint64_t result1 = Ps2PmaddwLane(
        Ps2LowWord(ctx->hi1),
        Ps2LowWord(ctx->lo1),
        lhsWords[2],
        rhsWords[2],
        false);

    ctx->lo = Ps2SignExt32ToU64(Ps2LowWord(result0));
    ctx->hi = Ps2SignExt32ToU64(Ps2HighWord(result0));
    ctx->lo1 = Ps2SignExt32ToU64(Ps2LowWord(result1));
    ctx->hi1 = Ps2SignExt32ToU64(Ps2HighWord(result1));
    return Ps2MakeU64Vector(result0, result1);
}

// PLZCW: Count leading bits that match the sign bit, minus 1.
// For positive values: count leading zeros minus 1 (excludes sign bit).
// For negative values: count leading ones minus 1 (excludes sign bit).
// Special cases: 0x00000000 -> 31, 0xFFFFFFFF -> 31.
static inline uint32_t ps2_plzcw32(uint32_t x)
{
    if (x == 0 || x == 0xFFFFFFFF)
        return 31;
    if (x & 0x80000000u)
        x = ~x; // If sign bit set, invert to count leading ones as zeros
    return static_cast<uint32_t>(std::countl_zero(x)) - 1;
}

#define PS2_BLENDV_PS(a, b, mask) _mm_blendv_ps((a), (b), (mask))
#define PS2_MIN_EPI32(a, b) _mm_min_epi32((a), (b))
#define PS2_MAX_EPI32(a, b) _mm_max_epi32((a), (b))
#define PS2_SHUFFLE_EPI8(v, mask) _mm_shuffle_epi8((v), (mask))

#define PS2_EXTRACT_EPI32(v, i) Ps2ExtractEpi32((v), (i))
#define PS2_EXTRACT_EPI64(v, i) Ps2ExtractEpi64((v), (i))

#define PS2_EXTRACT_EPI32_0(v) Ps2ExtractEpi32((v), 0)
#define PS2_EXTRACT_EPI32_1(v) Ps2ExtractEpi32((v), 1)
#define PS2_EXTRACT_EPI32_2(v) Ps2ExtractEpi32((v), 2)
#define PS2_EXTRACT_EPI32_3(v) Ps2ExtractEpi32((v), 3)

#define PS2_EXTRACT_EPI64_0(v) Ps2ExtractEpi64((v), 0)
#define PS2_EXTRACT_EPI64_1(v) Ps2ExtractEpi64((v), 1)

// Basic MIPS arithmetic operations
static inline uint32_t Ps2Add32Overflow(uint32_t lhs, uint32_t rhs, bool &overflow)
{
    const uint32_t result = lhs + rhs;
    overflow = ((~(lhs ^ rhs) & (lhs ^ result)) & 0x80000000u) != 0u;
    return result;
}

static inline uint32_t Ps2Sub32Overflow(uint32_t lhs, uint32_t rhs, bool &overflow)
{
    const uint32_t result = lhs - rhs;
    overflow = (((lhs ^ rhs) & (lhs ^ result)) & 0x80000000u) != 0u;
    return result;
}

static inline uint64_t Ps2Add64Overflow(uint64_t lhs, uint64_t rhs, bool &overflow)
{
    const uint64_t result = lhs + rhs;
    overflow = ((~(lhs ^ rhs) & (lhs ^ result)) & 0x8000000000000000ull) != 0ull;
    return result;
}

static inline uint64_t Ps2Sub64Overflow(uint64_t lhs, uint64_t rhs, bool &overflow)
{
    const uint64_t result = lhs - rhs;
    overflow = (((lhs ^ rhs) & (lhs ^ result)) & 0x8000000000000000ull) != 0ull;
    return result;
}

#define ADD32(a, b) ((uint32_t)(a) + (uint32_t)(b))
#define ADD32_OV(rs, rt, result32, overflow)                                     \
    do                                                                           \
    {                                                                            \
        (result32) = Ps2Add32Overflow((uint32_t)(rs), (uint32_t)(rt), overflow); \
    } while (0)
#define SUB32(a, b) ((uint32_t)(a) - (uint32_t)(b))
#define SUB32_OV(rs, rt, result32, overflow)                                     \
    do                                                                           \
    {                                                                            \
        (result32) = Ps2Sub32Overflow((uint32_t)(rs), (uint32_t)(rt), overflow); \
    } while (0)
#define ADD64(a, b) ((uint64_t)(a) + (uint64_t)(b))
#define ADD64_OV(rs, rt, result64, overflow)                                     \
    do                                                                           \
    {                                                                            \
        (result64) = Ps2Add64Overflow((uint64_t)(rs), (uint64_t)(rt), overflow); \
    } while (0)
#define SUB64(a, b) ((uint64_t)(a) - (uint64_t)(b))
#define SUB64_OV(rs, rt, result64, overflow)                                     \
    do                                                                           \
    {                                                                            \
        (result64) = Ps2Sub64Overflow((uint64_t)(rs), (uint64_t)(rt), overflow); \
    } while (0)
#define MUL32(a, b) ((uint32_t)(a) * (uint32_t)(b))
#define DIV32(a, b) ((uint32_t)(a) / (uint32_t)(b))
#define AND32(a, b) ((uint32_t)((a) & (b)))
#define OR32(a, b) ((uint32_t)((a) | (b)))
#define XOR32(a, b) ((uint32_t)((a) ^ (b)))
#define NOR32(a, b) ((uint32_t)(~((a) | (b))))
#define SLL32(a, b) ((uint32_t)((a) << (b)))
#define SRL32(a, b) ((uint32_t)((a) >> (b)))
#define SRA32(a, b) ((uint32_t)((int32_t)(a) >> (b)))
#define SLT32(a, b) ((uint32_t)((int32_t)(a) < (int32_t)(b) ? 1 : 0))
#define SLTU32(a, b) ((uint32_t)((a) < (b) ? 1 : 0))

// PS2-specific 128-bit MMI operations
#define PS2_PEXTLW(a, b) _mm_unpacklo_epi32((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUW(a, b) _mm_unpackhi_epi32((__m128i)(b), (__m128i)(a))
#define PS2_PEXTLH(a, b) _mm_unpacklo_epi16((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUH(a, b) _mm_unpackhi_epi16((__m128i)(b), (__m128i)(a))
#define PS2_PEXTLB(a, b) _mm_unpacklo_epi8((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUB(a, b) _mm_unpackhi_epi8((__m128i)(b), (__m128i)(a))
#define PS2_PADDW(a, b) _mm_add_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PSUBW(a, b) _mm_sub_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PMAXW(a, b) PS2_MAX_EPI32((__m128i)(a), (__m128i)(b))
#define PS2_PMINW(a, b) PS2_MIN_EPI32((__m128i)(a), (__m128i)(b))
#define PS2_PADDH(a, b) _mm_add_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PSUBH(a, b) _mm_sub_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PMAXH(a, b) _mm_max_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PMINH(a, b) _mm_min_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PADDB(a, b) _mm_add_epi8((__m128i)(a), (__m128i)(b))
#define PS2_PSUBB(a, b) _mm_sub_epi8((__m128i)(a), (__m128i)(b))
#define PS2_PAND(a, b) _mm_and_si128((__m128i)(a), (__m128i)(b))
#define PS2_POR(a, b) _mm_or_si128((__m128i)(a), (__m128i)(b))
#define PS2_PXOR(a, b) _mm_xor_si128((__m128i)(a), (__m128i)(b))
#define PS2_PNOR(a, b) _mm_xor_si128(_mm_or_si128((__m128i)(a), (__m128i)(b)), _mm_set1_epi32(0xFFFFFFFF))

// PS2 VU (Vector Unit) operations
#define PS2_VADD(a, b) _mm_add_ps((__m128)(a), (__m128)(b))
#define PS2_VSUB(a, b) _mm_sub_ps((__m128)(a), (__m128)(b))
#define PS2_VMUL(a, b) _mm_mul_ps((__m128)(a), (__m128)(b))
#define PS2_VDIV(a, b) _mm_div_ps((__m128)(a), (__m128)(b))
#define PS2_VMULQ(a, q) _mm_mul_ps((__m128)(a), _mm_set1_ps(q))
#define PS2_VBLEND(a, b, mask) PS2_BLENDV_PS((__m128)(a), (__m128)(b), (__m128)(mask))

static inline float Ps2VuCanonicalizeFloat(float value)
{
    uint32_t bits = std::bit_cast<uint32_t>(value);
    switch (bits & 0x7f800000u)
    {
    case 0u:
        // VU arithmetic flushes denormals to signed zero.
        bits &= 0x80000000u;
        break;
    case 0x7f800000u:
        // The VU has no infinities or NaNs; overflowing values saturate.
        bits = (bits & 0x80000000u) | 0x7f7fffffu;
        break;
    default:
        break;
    }
    return std::bit_cast<float>(bits);
}

static inline float Ps2VuRsqrt(float fsValue, float ftValue, uint16_t &status)
{
    const uint32_t fsBits = std::bit_cast<uint32_t>(fsValue);
    const uint32_t ftBits = std::bit_cast<uint32_t>(ftValue);
    const bool negative = ((fsBits ^ ftBits) & 0x80000000u) != 0u;
    const float fs = Ps2VuCanonicalizeFloat(fsValue);
    const float ft = Ps2VuCanonicalizeFloat(ftValue);

    // VRSQRT replaces the current invalid/divide flags while preserving all
    // other current and sticky STATUS state.
    status = static_cast<uint16_t>(status & ~0x0030u);
    if (ft == 0.0f)
    {
        status = static_cast<uint16_t>(status | 0x0020u);
        if (fs != 0.0f)
        {
            return std::bit_cast<float>(
                (negative ? 0x80000000u : 0u) | 0x7f7fffffu);
        }

        status = static_cast<uint16_t>(status | 0x0010u);
        return std::bit_cast<float>(negative ? 0x80000000u : 0u);
    }

    if (ft < 0.0f)
    {
        status = static_cast<uint16_t>(status | 0x0010u);
    }
    return Ps2VuCanonicalizeFloat(fs / std::sqrt(std::fabs(ft)));
}

static inline void Ps2VuUpdateFmacFlags(
    R5900Context *ctx, __m128 result, uint8_t dest,
    bool preserveUnselected = false)
{
    uint32_t words[4];
    std::memcpy(words, &result, sizeof(words));

    uint32_t mac = preserveUnselected ? ctx->vu0_mac_flags : 0u;
    for (uint32_t lane = 0u; lane < 4u; ++lane)
    {
        const uint32_t laneMask = 0x8u >> lane;
        if ((dest & laneMask) == 0u)
        {
            continue;
        }

        const uint32_t shift = 3u - lane;
        mac &= ~(0x1111u << shift);

        const uint32_t bits = words[lane];
        if ((bits & 0x80000000u) != 0u)
        {
            mac |= 0x0010u << shift;
        }

        const uint32_t exponent = (bits >> 23u) & 0xffu;
        if ((bits & 0x7fffffffu) == 0u)
        {
            mac |= 0x0001u << shift;
        }
        else if (exponent == 0u)
        {
            mac |= 0x0101u << shift;
        }
        else if (exponent == 0xffu)
        {
            mac |= 0x1000u << shift;
        }
    }

    uint32_t status = 0u;
    if ((mac & 0x000fu) != 0u)
    {
        status |= 0x1u;
    }
    if ((mac & 0x00f0u) != 0u)
    {
        status |= 0x2u;
    }
    if ((mac & 0x0f00u) != 0u)
    {
        status |= 0x4u;
    }
    if ((mac & 0xf000u) != 0u)
    {
        status |= 0x8u;
    }

    ctx->vu0_mac_flags = mac;
    ctx->vu0_status = static_cast<uint16_t>(
        (ctx->vu0_status & 0x0ff0u) |
        status |
        (status << 6u));
}

static inline __m128 Ps2VuOuterProductTerms(__m128 fs, __m128 ft)
{
    // OPMULA/OPMSUB form the three products used by a vector cross product:
    // (fs.y * ft.z, fs.z * ft.x, fs.x * ft.y). Their W lane is not written.
    const __m128 fsYzxw =
        _mm_shuffle_ps(fs, fs, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 ftZxyw =
        _mm_shuffle_ps(ft, ft, _MM_SHUFFLE(3, 1, 0, 2));
    const __m128 products = PS2_VMUL(fsYzxw, ftZxyw);
    return _mm_and_ps(
        products,
        _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1)));
}

// Memory access helpers - Hybrid Fast/Slow Path
// Callers must validate aliases, bounds, and alignment before using the direct
// helpers. The hybrid READ/WRITE macros do that and otherwise use the runtime.

static inline bool Ps2FastRangeIsContiguous(uint32_t offset, uint32_t bytes)
{
    return offset <= (PS2_RAM_SIZE - bytes);
}

static inline uint8_t Ps2FastRead8(const uint8_t *rdram, uint32_t addr)
{
    return rdram[addr & PS2_RAM_MASK];
}

static inline uint16_t Ps2FastRead16(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint16_t)))
    {
        uint8_t wrapped[sizeof(uint16_t)];
        for (uint32_t i = 0; i < sizeof(uint16_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint16_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint16_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline uint32_t Ps2FastRead32(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint32_t)))
    {
        uint8_t wrapped[sizeof(uint32_t)];
        for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint32_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint32_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline uint64_t Ps2FastRead64(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint64_t)))
    {
        uint8_t wrapped[sizeof(uint64_t)];
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint64_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint64_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline __m128i Ps2FastRead128(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(__m128i)))
    {
        alignas(16) uint8_t wrapped[sizeof(__m128i)];
        for (uint32_t i = 0; i < sizeof(__m128i); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        __m128i value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    __m128i value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline void Ps2FastWrite8(uint8_t *rdram, uint32_t addr, uint8_t value)
{
    rdram[addr & PS2_RAM_MASK] = value;
}

static inline void Ps2FastWrite16(uint8_t *rdram, uint32_t addr, uint16_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint16_t)))
    {
        uint8_t wrapped[sizeof(uint16_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint16_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite32(uint8_t *rdram, uint32_t addr, uint32_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint32_t)))
    {
        uint8_t wrapped[sizeof(uint32_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite64(uint8_t *rdram, uint32_t addr, uint64_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint64_t)))
    {
        uint8_t wrapped[sizeof(uint64_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWriteMasked32(
    uint8_t *rdram,
    uint32_t addr,
    uint32_t value,
    uint8_t byteEnable)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    for (uint32_t index = 0u; index < 4u; ++index)
    {
        if ((byteEnable & (1u << index)) != 0u)
        {
            rdram[offset + index] = static_cast<uint8_t>(
                value >> (index * 8u));
        }
    }
}

static inline void Ps2FastWriteMasked64(
    uint8_t *rdram,
    uint32_t addr,
    uint64_t value,
    uint8_t byteEnable)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    for (uint32_t index = 0u; index < 8u; ++index)
    {
        if ((byteEnable & (1u << index)) != 0u)
        {
            rdram[offset + index] = static_cast<uint8_t>(
                value >> (index * 8u));
        }
    }
}

static inline void Ps2FastWrite128(uint8_t *rdram, uint32_t addr, __m128i value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(__m128i)))
    {
        alignas(16) uint8_t wrapped[sizeof(__m128i)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(__m128i); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

#define FAST_READ8(addr) Ps2FastRead8(rdram, (uint32_t)(addr))
#define FAST_READ16(addr) Ps2FastRead16(rdram, (uint32_t)(addr))
#define FAST_READ32(addr) Ps2FastRead32(rdram, (uint32_t)(addr))
#define FAST_READ64(addr) Ps2FastRead64(rdram, (uint32_t)(addr))
#define FAST_READ128(addr) Ps2FastRead128(rdram, (uint32_t)(addr))

#define FAST_WRITE8(addr, val) Ps2FastWrite8(rdram, (uint32_t)(addr), (uint8_t)(val))
#define FAST_WRITE16(addr, val) Ps2FastWrite16(rdram, (uint32_t)(addr), (uint16_t)(val))
#define FAST_WRITE32(addr, val) Ps2FastWrite32(rdram, (uint32_t)(addr), (uint32_t)(val))
#define FAST_WRITE64(addr, val) Ps2FastWrite64(rdram, (uint32_t)(addr), (uint64_t)(val))
#define FAST_WRITE128(addr, val) Ps2FastWrite128(rdram, (uint32_t)(addr), (val))

#define DEBUG_FAST_READ(width, type, virtualAddr, physicalOffset)                   \
    ([&]() -> type                                                                  \
     {                                                                              \
         const uint32_t _debug_addr = (uint32_t)(virtualAddr);                      \
         const uint32_t _debug_physical_offset =                                   \
             (uint32_t)(physicalOffset);                                            \
         runtime->debugObserveMemoryAccess(                                         \
             _debug_addr, (width) / 8u, PS2Runtime::DebugMemoryAccess::Read, ctx);  \
         return FAST_READ##width(_debug_physical_offset);                            \
     }())

#define DEBUG_FAST_WRITE(width, type, virtualAddr, physicalOffset, val)             \
    do                                                                              \
    {                                                                               \
        const uint32_t _debug_addr = (uint32_t)(virtualAddr);                       \
        const uint32_t _debug_physical_offset =                                    \
            (uint32_t)(physicalOffset);                                             \
        runtime->debugObserveMemoryAccess(                                          \
            _debug_addr, (width) / 8u, PS2Runtime::DebugMemoryAccess::Write, ctx);  \
        runtime->memory().observeRdramWriteFixed<(width) / 8u>(                     \
            _debug_physical_offset, ctx ? ctx->pc : 0u);                           \
        FAST_WRITE##width(_debug_physical_offset, (type)(val));                     \
    } while (0)

#define DEBUG_FAST_READ8(virtualAddr, physicalOffset)                               \
    DEBUG_FAST_READ(8, uint8_t, virtualAddr, physicalOffset)
#define DEBUG_FAST_READ16(virtualAddr, physicalOffset)                              \
    DEBUG_FAST_READ(16, uint16_t, virtualAddr, physicalOffset)
#define DEBUG_FAST_READ32(virtualAddr, physicalOffset)                              \
    DEBUG_FAST_READ(32, uint32_t, virtualAddr, physicalOffset)
#define DEBUG_FAST_READ64(virtualAddr, physicalOffset)                              \
    DEBUG_FAST_READ(64, uint64_t, virtualAddr, physicalOffset)
#define DEBUG_FAST_READ128(virtualAddr, physicalOffset)                             \
    DEBUG_FAST_READ(128, __m128i, virtualAddr, physicalOffset)

#define DEBUG_FAST_WRITE8(virtualAddr, physicalOffset, val)                         \
    DEBUG_FAST_WRITE(8, uint8_t, virtualAddr, physicalOffset, val)
#define DEBUG_FAST_WRITE16(virtualAddr, physicalOffset, val)                        \
    DEBUG_FAST_WRITE(16, uint16_t, virtualAddr, physicalOffset, val)
#define DEBUG_FAST_WRITE32(virtualAddr, physicalOffset, val)                        \
    DEBUG_FAST_WRITE(32, uint32_t, virtualAddr, physicalOffset, val)
#define DEBUG_FAST_WRITE64(virtualAddr, physicalOffset, val)                        \
    DEBUG_FAST_WRITE(64, uint64_t, virtualAddr, physicalOffset, val)
#define DEBUG_FAST_WRITE128(virtualAddr, physicalOffset, val)                       \
    do                                                                              \
    {                                                                               \
        const uint32_t _debug_addr = (uint32_t)(virtualAddr);                       \
        const uint32_t _debug_physical_offset =                                    \
            (uint32_t)(physicalOffset);                                             \
        const __m128i _debug_value = (val);                                         \
        runtime->debugObserveMemoryAccess(                                          \
            _debug_addr, 16u, PS2Runtime::DebugMemoryAccess::Write, ctx);           \
        runtime->memory().observeRdramWriteFixed<16u>(                              \
            _debug_physical_offset, ctx ? ctx->pc : 0u);                           \
        FAST_WRITE128(_debug_physical_offset, _debug_value);                        \
    } while (0)

bool Ps2ResolveFastGuestRdramAccess(
    PS2Runtime *runtime,
    const R5900Context *ctx,
    uint32_t address,
    uint32_t bytes,
    bool writeAccess,
    uint32_t &physicalOffset);

PS2X_EE_OBSERVATION_POLICY_INLINE void
Ps2MaterializeEeMemoryAccess(R5900Context *ctx) noexcept
{
    if (ctx != nullptr)
    {
        ctx->finishEeInstruction();
    }
}

template <uint32_t Bytes, bool WriteAccess>
PS2X_EE_OBSERVATION_POLICY_INLINE bool
Ps2ResolveFastGuestRdramAccessFixed(
    PS2Runtime *runtime,
    const R5900Context *ctx,
    uint32_t address,
    uint32_t &physicalOffset)
{
    if (runtime == nullptr)
    {
        return false;
    }
    const EeAddressTranslationContext translation =
        ctx != nullptr
            ? EeAddressTranslationContext::fromCop0Status(
                  ctx->cop0_status,
                  static_cast<uint8_t>(ctx->cop0_entryhi))
            : EeAddressTranslationContext::unchecked();
    return Ps2ResolveFastGuestRdramOffsetFixed<
        Bytes,
        WriteAccess>(
        runtime->memory(),
        translation,
        address,
        physicalOffset);
}

#define READ8(addr) ([&]() -> uint8_t {                              \
    const uint32_t _addr = (uint32_t)(addr);                         \
    uint32_t _physical_offset = 0u;                                  \
    return Ps2ResolveFastGuestRdramAccessFixed<1u, false>(            \
               runtime, ctx, _addr, _physical_offset)                \
        ? DEBUG_FAST_READ8(_addr, _physical_offset)                  \
        : (Ps2MaterializeEeMemoryAccess(ctx),                        \
           runtime->Load8(rdram, ctx, _addr)); }())

#define READ16(addr) ([&]() -> uint16_t {                            \
    const uint32_t _addr = (uint32_t)(addr);                         \
    uint32_t _physical_offset = 0u;                                  \
    return Ps2ResolveFastGuestRdramAccessFixed<2u, false>(            \
               runtime, ctx, _addr, _physical_offset)                \
        ? DEBUG_FAST_READ16(_addr, _physical_offset)                 \
        : (Ps2MaterializeEeMemoryAccess(ctx),                        \
           runtime->Load16(rdram, ctx, _addr)); }())

#define READ32(addr) ([&]() -> uint32_t {                            \
    const uint32_t _addr = (uint32_t)(addr);                         \
    uint32_t _physical_offset = 0u;                                  \
    return Ps2ResolveFastGuestRdramAccessFixed<4u, false>(            \
               runtime, ctx, _addr, _physical_offset)                \
        ? DEBUG_FAST_READ32(_addr, _physical_offset)                 \
        : (Ps2MaterializeEeMemoryAccess(ctx),                        \
           runtime->Load32(rdram, ctx, _addr)); }())

#define READ64(addr) ([&]() -> uint64_t {                            \
    const uint32_t _addr = (uint32_t)(addr);                         \
    uint32_t _physical_offset = 0u;                                  \
    return Ps2ResolveFastGuestRdramAccessFixed<8u, false>(            \
               runtime, ctx, _addr, _physical_offset)                \
        ? DEBUG_FAST_READ64(_addr, _physical_offset)                 \
        : (Ps2MaterializeEeMemoryAccess(ctx),                        \
           runtime->Load64(rdram, ctx, _addr)); }())

#define READ128(addr) ([&]() -> __m128i {                            \
    const uint32_t _addr = (uint32_t)(addr);                         \
    uint32_t _physical_offset = 0u;                                  \
    return Ps2ResolveFastGuestRdramAccessFixed<16u, false>(           \
               runtime, ctx, _addr, _physical_offset)                \
        ? DEBUG_FAST_READ128(_addr, _physical_offset)                \
        : (Ps2MaterializeEeMemoryAccess(ctx),                        \
           runtime->Load128(rdram, ctx, _addr)); }())

#define WRITE8(addr, val)                                                            \
    do                                                                               \
    {                                                                                \
        const uint32_t _addr = (uint32_t)(addr);                                     \
        const uint8_t _value = (uint8_t)(val);                                       \
        uint32_t _physical_offset = 0u;                                              \
        if (!Ps2ResolveFastGuestRdramAccessFixed<1u, true>(                          \
                runtime, ctx, _addr, _physical_offset))                              \
        {                                                                            \
            Ps2MaterializeEeMemoryAccess(ctx);                                       \
            runtime->Store8(rdram, ctx, _addr, _value);                              \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            runtime->debugObserveMemoryAccess(                                       \
                _addr, 1u, PS2Runtime::DebugMemoryAccess::Write, ctx);               \
            ps2TraceGuestWrite(rdram, _addr, 1u, _value, 0u, "WRITE8", ctx);        \
            runtime->memory().observeRdramWriteFixed<1u>(                            \
                _physical_offset, ctx ? ctx->pc : 0u);                              \
            FAST_WRITE8(_physical_offset, _value);                                   \
        }                                                                            \
    } while (0)

#define WRITE16(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        const uint32_t _addr = (uint32_t)(addr);                                       \
        const uint16_t _value = (uint16_t)(val);                                       \
        uint32_t _physical_offset = 0u;                                                \
        if (!Ps2ResolveFastGuestRdramAccessFixed<2u, true>(                            \
                runtime, ctx, _addr, _physical_offset))                                \
        {                                                                              \
            Ps2MaterializeEeMemoryAccess(ctx);                                         \
            runtime->Store16(rdram, ctx, _addr, _value);                               \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            runtime->debugObserveMemoryAccess(                                         \
                _addr, 2u, PS2Runtime::DebugMemoryAccess::Write, ctx);                 \
            ps2TraceGuestWrite(rdram, _addr, 2u, _value, 0u, "WRITE16", ctx);         \
            runtime->memory().observeRdramWriteFixed<2u>(                              \
                _physical_offset, ctx ? ctx->pc : 0u);                                \
            FAST_WRITE16(_physical_offset, _value);                                    \
        }                                                                              \
    } while (0)

#define WRITE32(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        const uint32_t _addr = (uint32_t)(addr);                                       \
        const uint32_t _value = (uint32_t)(val);                                       \
        uint32_t _physical_offset = 0u;                                                \
        if (!Ps2ResolveFastGuestRdramAccessFixed<4u, true>(                            \
                runtime, ctx, _addr, _physical_offset))                                \
        {                                                                              \
            Ps2MaterializeEeMemoryAccess(ctx);                                         \
            runtime->Store32(rdram, ctx, _addr, _value);                               \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            runtime->debugObserveMemoryAccess(                                         \
                _addr, 4u, PS2Runtime::DebugMemoryAccess::Write, ctx);                 \
            ps2TraceGuestWrite(rdram, _addr, 4u, _value, 0u, "WRITE32", ctx);         \
            runtime->memory().observeRdramWriteFixed<4u>(                              \
                _physical_offset, ctx ? ctx->pc : 0u);                                \
            FAST_WRITE32(_physical_offset, _value);                                    \
        }                                                                              \
    } while (0)

#define WRITE64(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        const uint32_t _addr = (uint32_t)(addr);                                       \
        const uint64_t _value = (uint64_t)(val);                                       \
        uint32_t _physical_offset = 0u;                                                \
        if (!Ps2ResolveFastGuestRdramAccessFixed<8u, true>(                            \
                runtime, ctx, _addr, _physical_offset))                                \
        {                                                                              \
            Ps2MaterializeEeMemoryAccess(ctx);                                         \
            runtime->Store64(rdram, ctx, _addr, _value);                               \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            runtime->debugObserveMemoryAccess(                                         \
                _addr, 8u, PS2Runtime::DebugMemoryAccess::Write, ctx);                 \
            ps2TraceGuestWrite(rdram, _addr, 8u, _value, 0u, "WRITE64", ctx);         \
            runtime->memory().observeRdramWriteFixed<8u>(                              \
                _physical_offset, ctx ? ctx->pc : 0u);                                \
            FAST_WRITE64(_physical_offset, _value);                                    \
        }                                                                              \
    } while (0)

#define WRITE_MASKED32(addr, val, byteEnable)                                      \
    do                                                                              \
    {                                                                               \
        const uint32_t _addr = (uint32_t)(addr);                                    \
        const uint32_t _value = (uint32_t)(val);                                    \
        const uint8_t _byte_enable = (uint8_t)(byteEnable);                         \
        const Ps2ByteEnableSpan _span =                                             \
            Ps2DecodeByteEnableSpan(_byte_enable, 4u);                              \
        uint32_t _physical_offset = 0u;                                              \
        if (!_span.valid || !Ps2ResolveFastGuestRdramAccessFixed<4u, true>(          \
                runtime, ctx, _addr, _physical_offset))                             \
        {                                                                            \
            Ps2MaterializeEeMemoryAccess(ctx);                                       \
            runtime->StoreMasked32(                                                 \
                rdram, ctx, _addr, _value, _byte_enable);                           \
        }                                                                            \
        else if (_span.size != 0u)                                                  \
        {                                                                           \
            const uint32_t _selected_addr = _addr + _span.offset;                   \
            const uint32_t _selected_value =                                        \
                _value >> (_span.offset * 8u);                                      \
            runtime->debugObserveMemoryAccess(                                      \
                _selected_addr, _span.size,                                         \
                PS2Runtime::DebugMemoryAccess::Write, ctx);                         \
            ps2TraceGuestWrite(                                                     \
                rdram, _selected_addr, _span.size, _selected_value, 0u,             \
                "WRITE_MASKED32", ctx);                                             \
            runtime->memory().observeRdramWrite(                                    \
                _physical_offset + _span.offset,                                    \
                _span.size, ctx ? ctx->pc : 0u);                                   \
            Ps2FastWriteMasked32(                                                   \
                rdram, _physical_offset, _value, _byte_enable);                     \
        }                                                                           \
    } while (0)

#define WRITE_MASKED64(addr, val, byteEnable)                                      \
    do                                                                              \
    {                                                                               \
        const uint32_t _addr = (uint32_t)(addr);                                    \
        const uint64_t _value = (uint64_t)(val);                                    \
        const uint8_t _byte_enable = (uint8_t)(byteEnable);                         \
        const Ps2ByteEnableSpan _span =                                             \
            Ps2DecodeByteEnableSpan(_byte_enable, 8u);                              \
        uint32_t _physical_offset = 0u;                                              \
        if (!_span.valid || !Ps2ResolveFastGuestRdramAccessFixed<8u, true>(          \
                runtime, ctx, _addr, _physical_offset))                             \
        {                                                                            \
            Ps2MaterializeEeMemoryAccess(ctx);                                       \
            runtime->StoreMasked64(                                                 \
                rdram, ctx, _addr, _value, _byte_enable);                           \
        }                                                                            \
        else if (_span.size != 0u)                                                  \
        {                                                                           \
            const uint32_t _selected_addr = _addr + _span.offset;                   \
            const uint64_t _selected_value =                                        \
                _value >> (_span.offset * 8u);                                      \
            runtime->debugObserveMemoryAccess(                                      \
                _selected_addr, _span.size,                                         \
                PS2Runtime::DebugMemoryAccess::Write, ctx);                         \
            ps2TraceGuestWrite(                                                     \
                rdram, _selected_addr, _span.size, _selected_value, 0u,             \
                "WRITE_MASKED64", ctx);                                             \
            runtime->memory().observeRdramWrite(                                    \
                _physical_offset + _span.offset,                                    \
                _span.size, ctx ? ctx->pc : 0u);                                   \
            Ps2FastWriteMasked64(                                                   \
                rdram, _physical_offset, _value, _byte_enable);                     \
        }                                                                           \
    } while (0)

#define WRITE128(addr, val)                                                          \
    do                                                                               \
    {                                                                                \
        const uint32_t _addr = (uint32_t)(addr);                                     \
        const __m128i _value = (val);                                                \
        uint32_t _physical_offset = 0u;                                              \
        if (!Ps2ResolveFastGuestRdramAccessFixed<16u, true>(                         \
                runtime, ctx, _addr, _physical_offset))                             \
        {                                                                            \
            Ps2MaterializeEeMemoryAccess(ctx);                                       \
            runtime->Store128(rdram, ctx, _addr, _value);                            \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            runtime->debugObserveMemoryAccess(                                      \
                _addr, 16u, PS2Runtime::DebugMemoryAccess::Write, ctx);             \
            const uint64_t _lo = static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(_value)); \
            const uint64_t _hi = static_cast<uint64_t>(PS2_EXTRACT_EPI64_1(_value)); \
            ps2TraceGuestWrite(rdram, _addr, 16u, _lo, _hi, "WRITE128", ctx);        \
            runtime->memory().observeRdramWriteFixed<16u>(                           \
                _physical_offset, ctx ? ctx->pc : 0u);                              \
            FAST_WRITE128(_physical_offset, _value);                                 \
        }                                                                            \
    } while (0)

// Keep the complete fast memory operation in one shared body per access
// width. This avoids repeating translation, fallback, observation, and code
// invalidation branches in every generated function. Precise bodies remain
// inline wrappers around the authoritative runtime access path.
template <EeArchitecturalObservationMode Mode>
struct Ps2EeMemoryAccessPolicy;

template <>
struct Ps2EeMemoryAccessPolicy<EeArchitecturalObservationMode::Fast>
{
    static PS2X_EE_MEMORY_POLICY_NOINLINE uint8_t read8(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return READ8(address);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE uint16_t read16(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return READ16(address);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE uint32_t read32(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return READ32(address);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE uint64_t read64(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return READ64(address);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE __m128i read128(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return READ128(address);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void write8(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint8_t value)
    {
        WRITE8(address, value);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void write16(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint16_t value)
    {
        WRITE16(address, value);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void write32(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint32_t value)
    {
        WRITE32(address, value);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void write64(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint64_t value)
    {
        WRITE64(address, value);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void write128(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, __m128i value)
    {
        WRITE128(address, value);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void writeMasked32(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint32_t value, uint8_t byteEnable)
    {
        WRITE_MASKED32(address, value, byteEnable);
    }

    static PS2X_EE_MEMORY_POLICY_NOINLINE void writeMasked64(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint64_t value, uint8_t byteEnable)
    {
        WRITE_MASKED64(address, value, byteEnable);
    }
};

template <>
struct Ps2EeMemoryAccessPolicy<EeArchitecturalObservationMode::Precise>
{
    static PS2X_EE_OBSERVATION_POLICY_INLINE uint8_t read8(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return runtime->Load8(rdram, ctx, address);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE uint16_t read16(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return runtime->Load16(rdram, ctx, address);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE uint32_t read32(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return runtime->Load32(rdram, ctx, address);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE uint64_t read64(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return runtime->Load64(rdram, ctx, address);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE __m128i read128(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address)
    {
        return runtime->Load128(rdram, ctx, address);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void write8(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint8_t value)
    {
        runtime->Store8(rdram, ctx, address, value);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void write16(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint16_t value)
    {
        runtime->Store16(rdram, ctx, address, value);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void write32(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint32_t value)
    {
        runtime->Store32(rdram, ctx, address, value);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void write64(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint64_t value)
    {
        runtime->Store64(rdram, ctx, address, value);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void write128(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, __m128i value)
    {
        runtime->Store128(rdram, ctx, address, value);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void writeMasked32(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint32_t value, uint8_t byteEnable)
    {
        runtime->StoreMasked32(rdram, ctx, address, value, byteEnable);
    }

    static PS2X_EE_OBSERVATION_POLICY_INLINE void writeMasked64(
        uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
        uint32_t address, uint64_t value, uint8_t byteEnable)
    {
        runtime->StoreMasked64(rdram, ctx, address, value, byteEnable);
    }
};

#define PS2X_EE_READ8(mode, addr)                                      \
    Ps2EeMemoryAccessPolicy<mode>::read8(rdram, ctx, runtime, (addr))
#define PS2X_EE_READ16(mode, addr)                                     \
    Ps2EeMemoryAccessPolicy<mode>::read16(rdram, ctx, runtime, (addr))
#define PS2X_EE_READ32(mode, addr)                                     \
    Ps2EeMemoryAccessPolicy<mode>::read32(rdram, ctx, runtime, (addr))
#define PS2X_EE_READ64(mode, addr)                                     \
    Ps2EeMemoryAccessPolicy<mode>::read64(rdram, ctx, runtime, (addr))
#define PS2X_EE_READ128(mode, addr)                                    \
    Ps2EeMemoryAccessPolicy<mode>::read128(rdram, ctx, runtime, (addr))

#define PS2X_EE_WRITE8(mode, addr, val)                                \
    Ps2EeMemoryAccessPolicy<mode>::write8(rdram, ctx, runtime, (addr), (val))
#define PS2X_EE_WRITE16(mode, addr, val)                               \
    Ps2EeMemoryAccessPolicy<mode>::write16(rdram, ctx, runtime, (addr), (val))
#define PS2X_EE_WRITE32(mode, addr, val)                               \
    Ps2EeMemoryAccessPolicy<mode>::write32(rdram, ctx, runtime, (addr), (val))
#define PS2X_EE_WRITE64(mode, addr, val)                               \
    Ps2EeMemoryAccessPolicy<mode>::write64(rdram, ctx, runtime, (addr), (val))
#define PS2X_EE_WRITE128(mode, addr, val)                              \
    Ps2EeMemoryAccessPolicy<mode>::write128(rdram, ctx, runtime, (addr), (val))

#define PS2X_EE_WRITE_MASKED32(mode, addr, val, byteEnable)            \
    Ps2EeMemoryAccessPolicy<mode>::writeMasked32(                      \
        rdram, ctx, runtime, (addr), (val), (byteEnable))
#define PS2X_EE_WRITE_MASKED64(mode, addr, val, byteEnable)            \
    Ps2EeMemoryAccessPolicy<mode>::writeMasked64(                      \
        rdram, ctx, runtime, (addr), (val), (byteEnable))

// Packed Compare Greater Than (PCGT)
#define PS2_PCGTW(a, b) _mm_cmpgt_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PCGTH(a, b) _mm_cmpgt_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PCGTB(a, b) _mm_cmpgt_epi8((__m128i)(a), (__m128i)(b))

// Packed Add with Signed Saturation Word (PADDSW)
inline __m128i ps2_paddsw(__m128i a, __m128i b)
{

    __m128i sum = _mm_add_epi32(a, b);
    // Check for over/underflow. Clamp to either INT32_MIN/INT32_MAX.
    __m128i overflow = _mm_and_si128(_mm_xor_si128(a, sum),
                                     _mm_xor_si128(b, sum));
    // Extract input sign.
    overflow = _mm_srai_epi32(overflow, 31);
    __m128i input_sign = _mm_srai_epi32(a, 31);
    // Select saturation value based on overflow sign.
    #if defined(__SSE4_1__)
    __m128i sat = _mm_blendv_epi8(
        _mm_set1_epi32(INT32_MAX),
        _mm_set1_epi32(INT32_MIN),
        input_sign);
    return _mm_blendv_epi8(sum, sat, overflow);
    #else
    __m128i sat = _mm_or_si128(_mm_and_si128(input_sign, _mm_set1_epi32(INT32_MIN)),
                               _mm_andnot_si128(input_sign, _mm_set1_epi32(INT32_MAX)));
    return _mm_or_si128(_mm_and_si128(overflow, sat),
                        _mm_andnot_si128(overflow, sum));
    #endif
}
#define PS2_PADDSW(a, b) ps2_paddsw((__m128i)(a), (__m128i)(b))

// Packed Subtract with Signed Saturation Word (PSUBSW)
inline __m128i ps2_psubsw(__m128i a, __m128i b)
{
    __m128i diff = _mm_sub_epi32(a, b);
    // Check for over/underflow. Clamp to either INT32_MIN/INT32_MAX.
    __m128i overflow = _mm_and_si128(_mm_xor_si128(a, b),
                                     _mm_xor_si128(a, diff));
    // Extract input sign.
    overflow = _mm_srai_epi32(overflow, 31);
    __m128i input_sign = _mm_srai_epi32(a, 31);
    // Select saturation value based on overflow sign.
    #if defined(__SSE4_1__)
    __m128i sat = _mm_blendv_epi8(
        _mm_set1_epi32(INT32_MAX),
        _mm_set1_epi32(INT32_MIN),
        input_sign);
    return _mm_blendv_epi8(diff, sat, overflow);
    #else
    __m128i sat = _mm_or_si128(_mm_and_si128(input_sign, _mm_set1_epi32(INT32_MIN)),
                               _mm_andnot_si128(input_sign, _mm_set1_epi32(INT32_MAX)));
    return _mm_or_si128(_mm_and_si128(overflow, sat),
                        _mm_andnot_si128(overflow, diff));
    #endif

}
#define PS2_PSUBSW(a, b) ps2_psubsw((__m128i)(a), (__m128i)(b))

// Packed Compare Equal (PCEQ)
#define PS2_PCEQW(a, b) _mm_cmpeq_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PCEQH(a, b) _mm_cmpeq_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PCEQB(a, b) _mm_cmpeq_epi8((__m128i)(a), (__m128i)(b))

// Packed Absolute (PABS)
// x86 PABS retains the minimum signed value. A matching all-ones mask
// decrements only that bit pattern to the EE's saturated maximum.
inline __m128i ps2_pabsw(__m128i value)
{
    const __m128i absolute = _mm_abs_epi32(value);
    const __m128i minimum = _mm_cmpeq_epi32(
        absolute, _mm_set1_epi32(INT32_MIN));
    return _mm_add_epi32(absolute, minimum);
}

inline __m128i ps2_pabsh(__m128i value)
{
    const __m128i absolute = _mm_abs_epi16(value);
    const __m128i minimum = _mm_cmpeq_epi16(
        absolute, _mm_set1_epi16(INT16_MIN));
    return _mm_add_epi16(absolute, minimum);
}

#define PS2_PABSW(a) ps2_pabsw((__m128i)(a))
#define PS2_PABSH(a) ps2_pabsh((__m128i)(a))
#define PS2_PABSB(a) _mm_abs_epi8((__m128i)(a))

// Packed Pack (PPAC) - Packs larger elements into smaller ones
inline __m128i ps2_paddu32(__m128i a, __m128i b)
{
    __m128i sum = _mm_add_epi32(a, b);
    __m128i overflow = _mm_cmpgt_epi32(_mm_xor_si128(a, _mm_set1_epi32(INT32_MIN)),
                                       _mm_xor_si128(sum, _mm_set1_epi32(INT32_MIN)));
    return _mm_or_si128(sum, overflow); // overflow lanes become all-1s
}
inline __m128i ps2_psubu32(__m128i a, __m128i b)
{
    __m128i diff = _mm_sub_epi32(a, b);
    // Underflow if a < b (unsigned). Clamp to 0.
    __m128i underflow = _mm_cmpgt_epi32(_mm_xor_si128(b, _mm_set1_epi32(INT32_MIN)),
                                        _mm_xor_si128(a, _mm_set1_epi32(INT32_MIN)));
    return _mm_andnot_si128(underflow, diff); // underflow lanes become 0
}

inline __m128i ps2_ppacw(__m128i rs, __m128i rt)
{
    // rs = [rs3 rs2 rs1 rs0], rt = [rt3 rt2 rt1 rt0]
    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(rt), _mm_castsi128_ps(rs), _MM_SHUFFLE(2, 0, 2, 0)));
}
#define PS2_PPACW(a, b) ps2_ppacw((__m128i)(a), (__m128i)(b))

inline __m128i ps2_ppach(__m128i rs, __m128i rt)
{
    const __m128i mask = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13,  // from rt: halfwords 0,2,4,6
        0, 1, 4, 5, 8, 9, 12, 13); // from rs: halfwords 0,2,4,6
    __m128i lo = _mm_shuffle_epi8(rt, mask);
    __m128i hi = _mm_shuffle_epi8(rs, mask);
    return _mm_unpacklo_epi64(lo, hi);
}
#define PS2_PPACH(a, b) ps2_ppach((__m128i)(a), (__m128i)(b))

inline __m128i ps2_ppacb(__m128i rs, __m128i rt)
{
    const __m128i mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,  // from rt: bytes 0,2,4,6,8,10,12,14
        0, 2, 4, 6, 8, 10, 12, 14); // from rs
    __m128i lo = _mm_shuffle_epi8(rt, mask);
    __m128i hi = _mm_shuffle_epi8(rs, mask);
    return _mm_unpacklo_epi64(lo, hi);
}
#define PS2_PPACB(a, b) ps2_ppacb((__m128i)(a), (__m128i)(b))

// Packed Interleave (PINT)
inline __m128i ps2_pinth(__m128i rs, __m128i rt)
{
    const __m128i rsHigh = _mm_unpackhi_epi64(rs, rs);
    return _mm_unpacklo_epi16(rt, rsHigh);
}

inline __m128i ps2_pinteh(__m128i rs, __m128i rt)
{
    const __m128i evenHalfwords = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13,
        0, 1, 4, 5, 8, 9, 12, 13);
    const __m128i rtEven = _mm_shuffle_epi8(rt, evenHalfwords);
    const __m128i rsEven = _mm_shuffle_epi8(rs, evenHalfwords);
    return _mm_unpacklo_epi16(rtEven, rsEven);
}

#define PS2_PINTH(a, b) ps2_pinth((__m128i)(a), (__m128i)(b))
#define PS2_PINTEH(a, b) ps2_pinteh((__m128i)(a), (__m128i)(b))

// Packed Multiply-Add (PMADD)
#define PS2_PMADDW(a, b) _mm_add_epi32(_mm_mullo_epi32(_mm_shuffle_epi32((__m128i)(a), _MM_SHUFFLE(1, 0, 3, 2)), _mm_shuffle_epi32((__m128i)(b), _MM_SHUFFLE(1, 0, 3, 2))), _mm_mullo_epi32(_mm_shuffle_epi32((__m128i)(a), _MM_SHUFFLE(3, 2, 1, 0)), _mm_shuffle_epi32((__m128i)(b), _MM_SHUFFLE(3, 2, 1, 0))))

// Packed Variable Shifts
enum class Ps2VariableWordShiftOperation
{
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
};

inline uint32_t Ps2ArithmeticShiftRight32(uint32_t value, uint32_t amount)
{
    if (amount == 0u)
    {
        return value;
    }

    const uint32_t sign = 0u - (value >> 31u);
    return (value >> amount) | (sign << (32u - amount));
}

inline __m128i Ps2VariableWordShift(
    __m128i values,
    __m128i counts,
    Ps2VariableWordShiftOperation operation)
{
    uint32_t valueWords[4];
    uint32_t countWords[4];
    uint32_t resultWords[4];
    std::memcpy(valueWords, &values, sizeof(values));
    std::memcpy(countWords, &counts, sizeof(counts));

    for (uint32_t sourceWord = 0u; sourceWord < 4u; sourceWord += 2u)
    {
        const uint32_t amount = countWords[sourceWord] & 0x1fu;
        uint32_t result = 0u;
        switch (operation)
        {
        case Ps2VariableWordShiftOperation::LogicalLeft:
            result = valueWords[sourceWord] << amount;
            break;
        case Ps2VariableWordShiftOperation::LogicalRight:
            result = valueWords[sourceWord] >> amount;
            break;
        case Ps2VariableWordShiftOperation::ArithmeticRight:
            result = Ps2ArithmeticShiftRight32(
                valueWords[sourceWord], amount);
            break;
        }

        resultWords[sourceWord] = result;
        resultWords[sourceWord + 1u] = 0u - (result >> 31u);
    }

    __m128i output;
    std::memcpy(&output, resultWords, sizeof(output));
    return output;
}

#define PS2_PSLLVW(values, counts) Ps2VariableWordShift((__m128i)(values), (__m128i)(counts), Ps2VariableWordShiftOperation::LogicalLeft)
#define PS2_PSRLVW(values, counts) Ps2VariableWordShift((__m128i)(values), (__m128i)(counts), Ps2VariableWordShiftOperation::LogicalRight)
#define PS2_PSRAVW(values, counts) Ps2VariableWordShift((__m128i)(values), (__m128i)(counts), Ps2VariableWordShiftOperation::ArithmeticRight)

// FPU (COP1) operations
#define FPU_SET_ACC(ctx, res) (ctx->f_acc = res)
#define FPU_ADD_S(a, b) ((float)(a) + (float)(b))
#define FPU_SUB_S(a, b) ((float)(a) - (float)(b))
#define FPU_MUL_S(a, b) ((float)(a) * (float)(b))
#define FPU_DIV_S(a, b) ((float)(a) / (float)(b))
#define FPU_SQRT_S(a) sqrtf((float)(a))
#define FPU_ABS_S(a) fabsf((float)(a))
#define FPU_MOV_S(a) ((float)(a))
#define FPU_NEG_S(a) (-(float)(a))
#define FPU_ROUND_L_S(a) ((int64_t)roundf((float)(a)))
#define FPU_TRUNC_L_S(a) ((int64_t)(float)(a))
#define FPU_CEIL_L_S(a) ((int64_t)ceilf((float)(a)))
#define FPU_FLOOR_L_S(a) ((int64_t)floorf((float)(a)))
#define FPU_ROUND_W_S(a) ((int32_t)nearbyintf((float)(a)))
#define FPU_TRUNC_W_S(a) ((int32_t)(float)(a))
#define FPU_CEIL_W_S(a) ((int32_t)ceilf((float)(a)))
#define FPU_FLOOR_W_S(a) ((int32_t)floorf((float)(a)))
#define FPU_CVT_S_W(a) ((float)(int32_t)(a))
#define FPU_CVT_S_L(a) ((float)(int64_t)(a))
#define FPU_CVT_W_S(a) ((int32_t)nearbyintf((float)(a)))
#define FPU_CVT_L_S(a) ((int64_t)(float)(a))
#define FPU_C_F_S(a, b) (0)
#define FPU_C_UN_S(a, b) (isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_EQ_S(a, b) ((float)(a) == (float)(b))
#define FPU_C_UEQ_S(a, b) ((float)(a) == (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_OLT_S(a, b) ((float)(a) < (float)(b))
#define FPU_C_ULT_S(a, b) ((float)(a) < (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_OLE_S(a, b) ((float)(a) <= (float)(b))
#define FPU_C_ULE_S(a, b) ((float)(a) <= (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_SF_S(a, b) (0)
#define FPU_C_NGLE_S(a, b) (isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_SEQ_S(a, b) ((float)(a) == (float)(b))
#define FPU_C_NGL_S(a, b) ((float)(a) == (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_LT_S(a, b) ((float)(a) < (float)(b))
#define FPU_C_NGE_S(a, b) ((float)(a) < (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_LE_S(a, b) ((float)(a) <= (float)(b))
#define FPU_C_NGT_S(a, b) ((float)(a) <= (float)(b) || isnan((float)(a)) || isnan((float)(b)))

// QFSRV: Quadword Funnel Shift Right Variable
// Concatenates rs || rt (256 bits) and right-shifts by SA bits, taking lower 128 bits.
inline __m128i ps2_qfsrv(__m128i rs, __m128i rt, uint32_t sa)
{
    if (sa == 0)
        return rt;
    if (sa >= 128)
    {
        if (sa >= 256)
            return _mm_setzero_si128();
        uint32_t shift = sa - 128;
        if (shift == 0)
            return rs;
        // Shift rs right by (sa-128) bits
        uint32_t byteShift = shift / 8;
        uint32_t bitShift = shift % 8;
        // Byte shift rs right
        alignas(16) uint8_t buf[16] = {};
        alignas(16) uint8_t src[16];
        _mm_store_si128((__m128i *)src, rs);
        for (uint32_t i = 0; i + byteShift < 16; i++)
            buf[i] = src[i + byteShift];
        __m128i result = _mm_load_si128((__m128i *)buf);
        if (bitShift > 0)
            result = _mm_or_si128(_mm_srli_epi64(result, bitShift),
                                  _mm_slli_epi64(_mm_bsrli_si128(result, 8), 64 - bitShift));
        return result;
    }
    // sa is 1..127: result = (rs || rt) >> sa, lower 128 bits
    uint32_t byteShift = sa / 8;
    uint32_t bitShift = sa % 8;
    alignas(16) uint8_t combined[32];
    _mm_store_si128((__m128i *)(combined), rt);      // low 128 bits
    _mm_store_si128((__m128i *)(combined + 16), rs); // high 128 bits
    // Shift right by byteShift bytes
    alignas(16) uint8_t shifted[16];
    for (uint32_t i = 0; i < 16; i++)
        shifted[i] = (i + byteShift < 32) ? combined[i + byteShift] : 0;
    __m128i result = _mm_load_si128((__m128i *)shifted);
    if (bitShift > 0)
    {
        uint8_t extra = (byteShift + 16 < 32) ? combined[byteShift + 16] : 0;
        __m128i hi_byte = _mm_insert_epi8(_mm_setzero_si128(), extra, 15);
        alignas(16) uint8_t src32[32];
        for (uint32_t i = 0; i < 32; i++)
            src32[i] = combined[i];
        uint64_t lo0, lo1, hi0, hi1;
        std::memcpy(&lo0, src32, 8);
        std::memcpy(&lo1, src32 + 8, 8);
        std::memcpy(&hi0, src32 + 16, 8);
        std::memcpy(&hi1, src32 + 24, 8);
        // 256-bit right shift by sa bits
        uint64_t r0, r1;
        if (sa < 64)
        {
            r0 = (lo0 >> sa) | (lo1 << (64 - sa));
            r1 = (lo1 >> sa) | (hi0 << (64 - sa));
        }
        else if (sa < 128)
        {
            uint32_t s = sa - 64;
            if (s == 0)
            {
                r0 = lo1;
                r1 = hi0;
            }
            else
            {
                r0 = (lo1 >> s) | (hi0 << (64 - s));
                r1 = (hi0 >> s) | (hi1 << (64 - s));
            }
        }
        else
        {
            r0 = 0;
            r1 = 0; // handled above
        }
        result = _mm_set_epi64x((long long)r1, (long long)r0);
    }
    return result;
}
#define PS2_QFSRV(rs, rt, sa) ps2_qfsrv((__m128i)(rs), (__m128i)(rt), (uint32_t)(sa))
#define PS2_PCPYLD(rs, rt) _mm_unpacklo_epi64(rt, rs)
#define PS2_PEXEH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(3, 0, 1, 2)), _MM_SHUFFLE(3, 0, 1, 2))
#define PS2_PREVH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3))
#define PS2_PEXEW(rt) _mm_shuffle_epi32((rt), _MM_SHUFFLE(3, 0, 1, 2))
#define PS2_PROT3W(rt) _mm_shuffle_epi32((rt), _MM_SHUFFLE(3, 0, 2, 1))
#define PS2_PEXCH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0))
#define PS2_PCPYH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(0, 0, 0, 0)), _MM_SHUFFLE(0, 0, 0, 0))
#define PS2_PEXCW(rt) _mm_shuffle_epi32((rt), _MM_SHUFFLE(3, 1, 2, 0))

// Additional VU0 operations
#define PS2_VSQRT(x) sqrtf(x)
#define PS2_VRSQRT(x) (1.0f / sqrtf(x))

#define GPR_U32(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0U : static_cast<uint32_t>(PS2_EXTRACT_EPI32_0(ctx_ptr->r[reg_idx])))
#define GPR_S32(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0 : PS2_EXTRACT_EPI32_0(ctx_ptr->r[reg_idx]))
#define GPR_U64(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0ULL : static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(ctx_ptr->r[reg_idx])))
#define GPR_S64(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0LL : PS2_EXTRACT_EPI64_0(ctx_ptr->r[reg_idx]))
#define GPR_VEC(ctx_ptr, reg_idx) ((reg_idx == 0) ? _mm_setzero_si128() : ctx_ptr->r[reg_idx])

static inline void Ps2SetGprLow64(R5900Context *ctx, int reg, __m128i new_low)
{
    if (reg != 0)
    {
        ctx->r[reg] = _mm_castpd_si128(_mm_move_sd(_mm_castsi128_pd(ctx->r[reg]), _mm_castsi128_pd(new_low)));
    }
}

#define SET_GPR_U32(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        auto _evaluatedVal = (val);                                       \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal =                                             \
                _mm_cvtsi64_si128((int64_t)(int32_t)_evaluatedVal);        \
                                                                          \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_S32(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        auto _evaluatedVal = (val);                                       \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal =                                             \
                _mm_cvtsi64_si128((int64_t)(int32_t)_evaluatedVal);        \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_U64(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        auto _evaluatedVal = (val);                                       \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)_evaluatedVal);  \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_S64(ctx_ptr, reg_idx, val) SET_GPR_U64(ctx_ptr, reg_idx, val)

#define SET_GPR_VEC(ctx_ptr, reg_idx, val)     \
    do                                         \
    {                                          \
        __m128i _evaluatedVal = (val);         \
        if ((reg_idx) != 0)                    \
            ctx_ptr->r[reg_idx] = _evaluatedVal; \
    } while (0)

#endif // PS2_RUNTIME_MACROS_H
