#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

constexpr uint16_t mergeActiveLowPadButtons(uint16_t first, uint16_t second) noexcept
{
    return static_cast<uint16_t>(first & second);
}

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);
};

#endif
