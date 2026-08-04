#ifndef PS2_ADDRESS_H
#define PS2_ADDRESS_H

#include <cstdint>

#include "runtime/ps2_memory.h"

static inline constexpr uint32_t PS2_KSEG0_BASE = 0x80000000u;
static inline constexpr uint32_t PS2_KSEG0_KSEG1_SIZE = 0x40000000u;
static inline constexpr uint32_t PS2_KSEG2_BASE = 0xC0000000u;

static inline constexpr bool Ps2AddressInRange(uint32_t value, uint32_t base, uint32_t size)
{
    return (value - base) < size;
}

static inline constexpr bool Ps2IsUncachedRamMirrorAddress(uint32_t addr)
{
    return Ps2AddressInRange(addr, PS2_EE_UNCACHED_RAM_MIRROR_BASE, PS2_EE_UNCACHED_RAM_MIRROR_SIZE);
}

static inline constexpr bool Ps2IsAcceleratedRamMirrorAddress(uint32_t addr)
{
    return Ps2AddressInRange(addr,
                             PS2_EE_ACCELERATED_RAM_MIRROR_BASE,
                             PS2_EE_ACCELERATED_RAM_MIRROR_SIZE);
}

static inline constexpr bool Ps2IsDirectRdramAddress(uint32_t addr)
{
    uint32_t offset = 0u;
    return ps2ResolveDirectRdramOffset(addr, offset);
}

static inline constexpr bool Ps2CanUseFastRdramAccess(uint32_t addr, uint32_t bytes)
{
    if (bytes == 0u || bytes > PS2_RAM_SIZE || (addr % bytes) != 0u)
    {
        return false;
    }

    uint32_t offset = 0u;
    return ps2ResolveDirectRdramOffset(addr, offset) &&
           offset <= (PS2_RAM_SIZE - bytes);
}

// Resolve a complete guest access to the physical RDRAM offset consumed by
// the inline host helpers. Direct aliases are classified locally; ordinary
// mapped addresses succeed only after the authoritative translator has
// populated a matching cache entry.
static inline bool Ps2ResolveFastGuestRdramOffset(
    PS2Memory &memory,
    EeAddressTranslationContext translation,
    uint32_t address,
    uint32_t bytes,
    bool writeAccess,
    uint32_t &physicalOffset)
{
    if (!translation.permits(address))
    {
        return false;
    }

    const bool directAddress =
        !translation.mapsKusegThroughTlb(address) ||
        Ps2IsUncachedRamMirrorAddress(address) ||
        Ps2IsAcceleratedRamMirrorAddress(address);
    if (directAddress &&
        Ps2CanUseFastRdramAccess(address, bytes))
    {
        return ps2ResolveDirectRdramOffset(
            address, physicalOffset);
    }

    // KSEG2/KSEG3 and non-identity kuseg mappings are not direct RDRAM
    // aliases. Only an already-authoritative mapped cache entry may make
    // them eligible for the inline host access.
    const bool supportedAccessSize =
        bytes != 0u &&
        bytes <= sizeof(__m128i) &&
        (bytes & (bytes - 1u)) == 0u;
    if (!supportedAccessSize ||
        (address & (bytes - 1u)) != 0u ||
        (!translation.mapsKusegThroughTlb(address) &&
         address < PS2_KSEG2_BASE))
    {
        return false;
    }
    return memory.tryResolveCachedTlbRdramOffset(
        address,
        translation,
        writeAccess,
        bytes,
        physicalOffset);
}

static inline constexpr bool Ps2IsKseg01Address(uint32_t addr)
{
    return Ps2AddressInRange(addr, PS2_KSEG0_BASE, PS2_KSEG0_KSEG1_SIZE);
}

static inline constexpr bool Ps2IsKseg23Address(uint32_t addr)
{
    return addr >= PS2_KSEG2_BASE;
}

static inline constexpr uint32_t Ps2DirectMappedPhysicalAddress(uint32_t addr)
{
    return addr & 0x1FFFFFFFu;
}

static inline constexpr uint32_t Ps2PhysicalAddress(uint32_t addr)
{
    return (addr >= PS2_KSEG0_BASE) ? Ps2DirectMappedPhysicalAddress(addr) : addr;
}

static inline constexpr bool Ps2IsPhysicalSpecialAddress(uint32_t physAddr)
{
    return Ps2AddressInRange(physAddr, PS2_BIOS_BASE, PS2_BIOS_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_SCRATCHPAD_BASE, PS2_SCRATCHPAD_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_IO_BASE, PS2_IO_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE) ||
           (physAddr >= PS2_VU0_DATA_BASE && physAddr < (PS2_VU1_CODE_BASE + PS2_VU1_CODE_SIZE));
}

static inline constexpr bool Ps2IsSpecialAddress(uint32_t addr)
{
    if (Ps2IsKseg23Address(addr))
        return true;

    return Ps2IsPhysicalSpecialAddress(Ps2PhysicalAddress(addr));
}

#endif // PS2_ADDRESS_H
