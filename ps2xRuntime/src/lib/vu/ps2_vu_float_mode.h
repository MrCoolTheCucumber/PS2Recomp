#ifndef PS2_VU_FLOAT_MODE_H
#define PS2_VU_FLOAT_MODE_H

#include <cfenv>
#include <cstdint>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#if defined(__SSE__) && \
    (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define PS2X_VU_DIRECT_X86_FLOAT_MODE 1
#else
#define PS2X_VU_DIRECT_X86_FLOAT_MODE 0
#endif

class ScopedVuFloatMode
{
public:
    ScopedVuFloatMode()
#if PS2X_VU_DIRECT_X86_FLOAT_MODE
        : m_previousX87Control(readX87Control()),
          m_previousMxcsr(_mm_getcsr())
    {
        // GNU-family x86 fenv calls update these same two control fields but
        // require out-of-line library calls. Save and restore both fields
        // exactly while changing only their rounding/denormal policy inside
        // the VU scope.
        constexpr uint16_t kX87RoundingMask = 3u << 10u;
        constexpr uint16_t kX87RoundTowardZero = 3u << 10u;
        const uint16_t vuX87Control =
            static_cast<uint16_t>(
                (m_previousX87Control &
                 ~kX87RoundingMask) |
                kX87RoundTowardZero);
        if (vuX87Control != m_previousX87Control)
        {
            writeX87Control(vuX87Control);
            m_x87Changed = true;
        }

        constexpr uint32_t kMxcsrRoundingMask = 3u << 13u;
        constexpr uint32_t kMxcsrRoundTowardZero = 3u << 13u;
        constexpr uint32_t kDenormalsAreZero = 1u << 6u;
        constexpr uint32_t kFlushToZero = 1u << 15u;
        const uint32_t vuMxcsr =
            (m_previousMxcsr &
             ~kMxcsrRoundingMask) |
            kMxcsrRoundTowardZero |
            kDenormalsAreZero |
            kFlushToZero;
        if (vuMxcsr != m_previousMxcsr)
            _mm_setcsr(vuMxcsr);
    }
#else
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
#endif

    ~ScopedVuFloatMode()
    {
#if PS2X_VU_DIRECT_X86_FLOAT_MODE
        if (m_x87Changed)
            writeX87Control(m_previousX87Control);
        _mm_setcsr(m_previousMxcsr);
#else
        if (m_changed && m_previous != -1)
            std::fesetround(m_previous);
#if defined(__SSE__)
        _mm_setcsr(m_previousMxcsr);
#endif
#endif
    }

    ScopedVuFloatMode(const ScopedVuFloatMode &) = delete;
    ScopedVuFloatMode &operator=(
        const ScopedVuFloatMode &) = delete;

private:
#if PS2X_VU_DIRECT_X86_FLOAT_MODE
    static uint16_t readX87Control()
    {
        uint16_t control = 0u;
        __asm__ __volatile__(
            "fnstcw %0"
            : "=m"(control)
            :
            : "memory");
        return control;
    }

    static void writeX87Control(uint16_t control)
    {
        __asm__ __volatile__(
            "fldcw %0"
            :
            : "m"(control)
            : "memory");
    }

    uint16_t m_previousX87Control = 0u;
    bool m_x87Changed = false;
    uint32_t m_previousMxcsr = 0u;
#else
    int m_previous = -1;
    bool m_changed = false;
#if defined(__SSE__)
    uint32_t m_previousMxcsr = 0u;
#endif
#endif
};

#undef PS2X_VU_DIRECT_X86_FLOAT_MODE

#endif
