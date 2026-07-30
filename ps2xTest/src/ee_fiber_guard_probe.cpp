#include "runtime/boost_ee_fiber.h"

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define PS2X_GUARD_NOINLINE __declspec(noinline)
#else
#define PS2X_GUARD_NOINLINE __attribute__((noinline))
#endif

namespace
{
    volatile uint64_t g_guardSink = 0u;

    PS2X_GUARD_NOINLINE void exhaustFiberStack(
        size_t depth)
    {
        volatile unsigned char storage[16384];
        for (size_t offset = 0u;
             offset < sizeof(storage);
             offset += 256u)
        {
            storage[offset] =
                static_cast<unsigned char>(
                    depth + offset);
        }

        if (depth != 0u)
        {
            exhaustFiberStack(depth - 1u);
        }
        g_guardSink += storage[depth & 0xffu];
    }
}

int main()
{
    if (!BoostEeFiber::available())
    {
        return 2;
    }

    BoostEeFiber fiber([]()
    {
        exhaustFiberStack(1000000u);
    });
    fiber.resume();
    return 3;
}
