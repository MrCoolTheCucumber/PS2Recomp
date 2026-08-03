#include "runtime/ee_cache.h"

#include "runtime/ps2_memory.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace
{
    constexpr uint32_t kTagPhysicalMask = 0xFFFFF000u;
    constexpr uint32_t kTagLock = 0x00000008u;
    constexpr uint32_t kTagLrf = 0x00000010u;
    constexpr uint32_t kTagValid = 0x00000020u;
    constexpr uint32_t kTagDirty = 0x00000040u;
    constexpr uint32_t kDataTagMask =
        kTagPhysicalMask | kTagLock | kTagLrf |
        kTagValid | kTagDirty;
    constexpr uint32_t kInstructionTagMask =
        kTagPhysicalMask | kTagLrf | kTagValid;
    constexpr uint32_t kDataInvalidateMask =
        kTagLock | kTagValid | kTagDirty;

    constexpr uint32_t kIxltg = 0x00u;
    constexpr uint32_t kIxldt = 0x01u;
    constexpr uint32_t kBxlbt = 0x02u;
    constexpr uint32_t kIxstg = 0x04u;
    constexpr uint32_t kIxsdt = 0x05u;
    constexpr uint32_t kBxsbt = 0x06u;
    constexpr uint32_t kIxin = 0x07u;
    constexpr uint32_t kBhinbt = 0x0Au;
    constexpr uint32_t kIhin = 0x0Bu;
    constexpr uint32_t kBfh = 0x0Cu;
    constexpr uint32_t kIfl = 0x0Eu;
    constexpr uint32_t kDxltg = 0x10u;
    constexpr uint32_t kDxldt = 0x11u;
    constexpr uint32_t kDxstg = 0x12u;
    constexpr uint32_t kDxsdt = 0x13u;
    constexpr uint32_t kDxwbin = 0x14u;
    constexpr uint32_t kDxin = 0x16u;
    constexpr uint32_t kDhwbin = 0x18u;
    constexpr uint32_t kDhin = 0x1Au;
    constexpr uint32_t kDhwoin = 0x1Cu;
}

void EeCache::reset() noexcept
{
    m_data = {};
    m_instruction = {};
    m_btac = {};
}

uint32_t EeCache::loadWord(
    const std::array<uint8_t, kLineSize> &data,
    uint32_t offset) noexcept
{
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
           (static_cast<uint32_t>(data[offset + 2u]) << 16u) |
           (static_cast<uint32_t>(data[offset + 3u]) << 24u);
}

void EeCache::storeWord(
    std::array<uint8_t, kLineSize> &data,
    uint32_t offset,
    uint32_t value) noexcept
{
    data[offset] = static_cast<uint8_t>(value);
    data[offset + 1u] = static_cast<uint8_t>(value >> 8u);
    data[offset + 2u] = static_cast<uint8_t>(value >> 16u);
    data[offset + 3u] = static_cast<uint8_t>(value >> 24u);
}

size_t EeCache::refillWay(
    const std::array<InstructionLine, kWayCount> &set) noexcept
{
    if ((set[0].tag & kTagValid) == 0u)
    {
        return 0u;
    }
    if ((set[1].tag & kTagValid) == 0u)
    {
        return 1u;
    }

    // XOR of the two LRF bits selects the documented refill way.
    return ((set[0].tag ^ set[1].tag) & kTagLrf) != 0u
               ? 1u
               : 0u;
}

bool EeCache::tagMatchesPhysical(
    uint32_t tag,
    uint32_t physicalAddress) noexcept
{
    return (tag & kTagValid) != 0u &&
           (tag & kTagPhysicalMask) ==
               (physicalAddress & kTagPhysicalMask);
}

void EeCache::invalidateAllBtac() noexcept
{
    for (BtacEntry &entry : m_btac)
    {
        entry.tagLo &= ~1u;
    }
}

void EeCache::invalidateBtacFetchLine(
    uint32_t virtualAddress) noexcept
{
    const uint32_t fetchLine = virtualAddress & 0xFFFFFFC0u;
    for (BtacEntry &entry : m_btac)
    {
        if ((entry.tagLo & 1u) != 0u &&
            (entry.tagLo & 0xFFFFFFC0u) == fetchLine)
        {
            entry.tagLo &= ~1u;
        }
    }
}

void EeCache::writeBackDataLine(
    PS2Memory &memory,
    uint32_t index,
    DataLine &line,
    uint32_t writerPc)
{
    if ((line.tag & (kTagValid | kTagDirty)) !=
        (kTagValid | kTagDirty))
    {
        return;
    }

    const uint32_t physicalBase =
        (line.tag & kTagPhysicalMask) |
        ((index & (kDataSetCount - 1u)) << 6u);
    for (uint32_t offset = 0u;
         offset < kLineSize;
         offset += sizeof(uint32_t))
    {
        memory.write32(
            physicalBase + offset,
            loadWord(line.data, offset),
            writerPc);
    }
    line.tag &= ~kTagDirty;
}

bool EeCache::execute(
    PS2Memory &memory,
    uint32_t operation,
    uint32_t virtualAddress,
    uint32_t &tagLo,
    uint32_t &tagHi,
    uint32_t writerPc)
{
    const uint32_t cacheOperation = operation & 0x1Fu;
    const uint32_t way = virtualAddress & 1u;
    const uint32_t wordOffset = virtualAddress & 0x3Cu;

    switch (cacheOperation)
    {
    case kIxltg:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        tagLo = m_instruction[index][way].tag &
                kInstructionTagMask;
        return true;
    }
    case kIxldt:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        const InstructionLine &line =
            m_instruction[index][way];
        tagLo = loadWord(line.data, wordOffset);
        tagHi = line.metadata[wordOffset >> 2u] & 0x3Fu;
        return true;
    }
    case kBxlbt:
    {
        const BtacEntry &entry =
            m_btac[virtualAddress &
                   (kBtacEntryCount - 1u)];
        tagLo = entry.tagLo;
        tagHi = entry.tagHi;
        return true;
    }
    case kIxstg:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        m_instruction[index][way].tag =
            tagLo & kInstructionTagMask;
        return true;
    }
    case kIxsdt:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        InstructionLine &line =
            m_instruction[index][way];
        storeWord(line.data, wordOffset, tagLo);
        line.metadata[wordOffset >> 2u] =
            static_cast<uint8_t>(tagHi & 0x3Fu);
        invalidateAllBtac();
        return true;
    }
    case kBxsbt:
    {
        BtacEntry &entry =
            m_btac[virtualAddress &
                   (kBtacEntryCount - 1u)];
        entry.tagLo = tagLo & 0xFFFFFFF9u;
        entry.tagHi = tagHi & 0xFFFFFFFCu;
        return true;
    }
    case kIxin:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        m_instruction[index][way].tag &= ~kTagValid;
        return true;
    }
    case kBhinbt:
    {
        const uint32_t fetchAddress =
            virtualAddress & 0xFFFFFFF8u;
        for (BtacEntry &entry : m_btac)
        {
            if ((entry.tagLo & 1u) != 0u &&
                (entry.tagLo & 0xFFFFFFF8u) ==
                    fetchAddress)
            {
                entry.tagLo &= ~1u;
            }
        }
        return true;
    }
    case kIhin:
    {
        const uint32_t physicalAddress =
            memory.translateAddress(virtualAddress);
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        for (InstructionLine &line : m_instruction[index])
        {
            if (tagMatchesPhysical(
                    line.tag, physicalAddress))
            {
                line.tag &= ~kTagValid;
            }
        }
        invalidateBtacFetchLine(virtualAddress);
        return true;
    }
    case kBfh:
        invalidateAllBtac();
        return true;
    case kIfl:
    {
        const uint32_t physicalAddress =
            memory.translateAddress(virtualAddress);
        const uint32_t physicalBase =
            physicalAddress & ~(kLineSize - 1u);
        std::array<uint8_t, kLineSize> refill{};
        for (uint32_t offset = 0u;
             offset < kLineSize;
             offset += sizeof(uint32_t))
        {
            storeWord(
                refill, offset,
                memory.read32(physicalBase + offset));
        }

        const uint32_t index =
            (virtualAddress >> 6u) &
            (kInstructionSetCount - 1u);
        auto &set = m_instruction[index];
        InstructionLine &line = set[refillWay(set)];
        const uint32_t nextLrf =
            (line.tag ^ kTagLrf) & kTagLrf;
        line.tag =
            (physicalAddress & kTagPhysicalMask) |
            kTagValid | nextLrf;
        line.data = refill;
        line.metadata = {};
        return true;
    }
    case kDxltg:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        tagLo = m_data[index][way].tag & kDataTagMask;
        return true;
    }
    case kDxldt:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        tagLo = loadWord(
            m_data[index][way].data,
            wordOffset);
        return true;
    }
    case kDxstg:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        m_data[index][way].tag =
            tagLo & kDataTagMask;
        return true;
    }
    case kDxsdt:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        storeWord(
            m_data[index][way].data,
            wordOffset,
            tagLo);
        return true;
    }
    case kDxwbin:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        DataLine &line = m_data[index][way];
        writeBackDataLine(
            memory, index, line, writerPc);
        line.tag &= ~kDataInvalidateMask;
        return true;
    }
    case kDxin:
    {
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        m_data[index][way].tag &=
            ~kDataInvalidateMask;
        return true;
    }
    case kDhwbin:
    case kDhin:
    case kDhwoin:
    {
        const uint32_t physicalAddress =
            memory.translateAddress(virtualAddress);
        const uint32_t index =
            (virtualAddress >> 6u) &
            (kDataSetCount - 1u);
        for (DataLine &line : m_data[index])
        {
            if (!tagMatchesPhysical(
                    line.tag, physicalAddress))
            {
                continue;
            }

            if (cacheOperation == kDhwbin ||
                cacheOperation == kDhwoin)
            {
                writeBackDataLine(
                    memory, index, line, writerPc);
            }
            if (cacheOperation == kDhwoin)
            {
                line.tag &= ~kTagDirty;
            }
            else
            {
                line.tag &= ~kDataInvalidateMask;
            }
            break;
        }
        return true;
    }
    default:
        return false;
    }
}
