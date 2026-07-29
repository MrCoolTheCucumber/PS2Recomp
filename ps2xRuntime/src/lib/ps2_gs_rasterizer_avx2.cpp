#include "ps2_gs_packed_sprite_kernel.h"

#if !PS2X_GS_HAS_AVX2_KERNEL
#error "The AVX2 packed-sprite source requires the AVX2 kernel build"
#endif

#if !defined(__AVX2__) && !defined(_M_AVX2)
#error "The AVX2 packed-sprite source must be compiled with AVX2 enabled"
#endif

#if defined(_MSC_VER)
#define PS2X_GS_AVX2_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PS2X_GS_AVX2_NOINLINE __attribute__((noinline))
#else
#define PS2X_GS_AVX2_NOINLINE
#endif

namespace GSRasterizerPacked
{
    PS2X_GS_AVX2_NOINLINE void drawAvx2(
        const PreparedSprite &sprite)
    {
        drawCt32DecalSpriteKernel<Avx2VerticalInterpolator>(
            sprite);
    }
}

#undef PS2X_GS_AVX2_NOINLINE
