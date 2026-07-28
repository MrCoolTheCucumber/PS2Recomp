#ifndef PS2_VU_FLOAT_MODE_H
#define PS2_VU_FLOAT_MODE_H

#include <cfenv>
#include <cstdint>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

class ScopedVuFloatMode
{
public:
    ScopedVuFloatMode()
        : m_previous(std::fegetround())
    {
#if defined(__SSE__)
        m_previousMxcsr = _mm_getcsr();
#endif
        if (m_previous != FE_TOWARDZERO)
            m_changed = std::fesetround(FE_TOWARDZERO) == 0;

#if defined(__SSE__)
        // The VU treats denormal inputs as zero and clamps exponent
        // underflow to signed zero. DAZ and FTZ provide those semantics
        // for the host scalar floating-point operations.
        constexpr uint32_t kDenormalsAreZero = 1u << 6u;
        constexpr uint32_t kFlushToZero = 1u << 15u;
        _mm_setcsr(
            _mm_getcsr() |
            kDenormalsAreZero |
            kFlushToZero);
#endif
    }

    ~ScopedVuFloatMode()
    {
        if (m_changed && m_previous != -1)
            std::fesetround(m_previous);
#if defined(__SSE__)
        _mm_setcsr(m_previousMxcsr);
#endif
    }

    ScopedVuFloatMode(const ScopedVuFloatMode &) = delete;
    ScopedVuFloatMode &operator=(
        const ScopedVuFloatMode &) = delete;

private:
    int m_previous = -1;
    bool m_changed = false;
#if defined(__SSE__)
    uint32_t m_previousMxcsr = 0u;
#endif
};

#endif
