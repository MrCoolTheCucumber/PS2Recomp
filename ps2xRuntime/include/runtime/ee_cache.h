#ifndef PS2X_RUNTIME_EE_CACHE_H
#define PS2X_RUNTIME_EE_CACHE_H

#include <array>
#include <cstddef>
#include <cstdint>

class PS2Memory;

// Architectural state touched explicitly by the R5900 CACHE instruction.
// Ordinary host memory accesses remain coherent and do not pay the cost of a
// cycle-accurate cache simulation. Diagnostic tag/data operations and explicit
// writebacks still need stable, mutually coherent state.
class EeCache
{
public:
    static constexpr uint32_t kLineSize = 64u;
    static constexpr uint32_t kDataSetCount = 64u;
    static constexpr uint32_t kInstructionSetCount = 128u;
    static constexpr uint32_t kWayCount = 2u;
    static constexpr uint32_t kBtacEntryCount = 64u;

    void reset() noexcept;

    // Returns false only for a reserved CACHE op encoding. Address translation
    // faults from hit/fill operations are reported by PS2Memory to the caller.
    bool execute(PS2Memory &memory,
                 uint32_t operation,
                 uint32_t virtualAddress,
                 uint32_t &tagLo,
                 uint32_t &tagHi,
                 uint32_t writerPc = 0u);

private:
    struct DataLine
    {
        uint32_t tag = 0u;
        std::array<uint8_t, kLineSize> data{};
    };

    struct InstructionLine
    {
        uint32_t tag = 0u;
        std::array<uint8_t, kLineSize> data{};
        std::array<uint8_t, kLineSize / sizeof(uint32_t)> metadata{};
    };

    struct BtacEntry
    {
        uint32_t tagLo = 0u;
        uint32_t tagHi = 0u;
    };

    static uint32_t loadWord(
        const std::array<uint8_t, kLineSize> &data,
        uint32_t offset) noexcept;
    static void storeWord(
        std::array<uint8_t, kLineSize> &data,
        uint32_t offset,
        uint32_t value) noexcept;
    static size_t refillWay(
        const std::array<InstructionLine, kWayCount> &set) noexcept;
    static bool tagMatchesPhysical(
        uint32_t tag,
        uint32_t physicalAddress) noexcept;

    void invalidateAllBtac() noexcept;
    void invalidateBtacFetchLine(uint32_t virtualAddress) noexcept;
    void writeBackDataLine(
        PS2Memory &memory,
        uint32_t index,
        DataLine &line,
        uint32_t writerPc);

    std::array<
        std::array<DataLine, kWayCount>,
        kDataSetCount>
        m_data{};
    std::array<
        std::array<InstructionLine, kWayCount>,
        kInstructionSetCount>
        m_instruction{};
    std::array<BtacEntry, kBtacEntryCount> m_btac{};
};

#endif
