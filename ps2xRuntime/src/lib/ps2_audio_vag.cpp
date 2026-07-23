#include "runtime/ps2_audio.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace
{
    inline int16_t clamp16(int32_t v)
    {
        if (v < -32768)
            return -32768;
        if (v > 32767)
            return 32767;
        return static_cast<int16_t>(v);
    }

    inline int8_t signExtend4(uint8_t nibble)
    {
        uint8_t s = nibble & 0x0F;
        return static_cast<int8_t>((s & 8) ? static_cast<int8_t>(s | 0xF0) : static_cast<int8_t>(s));
    }
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate)
    {
        if (!data || sizeBytes < 48)
            return false;

        const uint32_t magic = (static_cast<uint32_t>(data[0]) << 24) |
                               (static_cast<uint32_t>(data[1]) << 16) |
                               (static_cast<uint32_t>(data[2]) << 8) |
                               static_cast<uint32_t>(data[3]);
        if (magic != 0x56414770u)
        {
            const uint32_t magicLE = (static_cast<uint32_t>(data[3]) << 24) |
                                     (static_cast<uint32_t>(data[2]) << 16) |
                                     (static_cast<uint32_t>(data[1]) << 8) |
                                     static_cast<uint32_t>(data[0]);
            if (magicLE != 0x56414770u)
                return false;
        }

        uint32_t dataSize = (static_cast<uint32_t>(data[0x0c]) << 24) |
                            (static_cast<uint32_t>(data[0x0d]) << 16) |
                            (static_cast<uint32_t>(data[0x0e]) << 8) |
                            static_cast<uint32_t>(data[0x0f]);
        outSampleRate = (static_cast<uint32_t>(data[0x10]) << 24) |
                        (static_cast<uint32_t>(data[0x11]) << 16) |
                        (static_cast<uint32_t>(data[0x12]) << 8) |
                        static_cast<uint32_t>(data[0x13]);
        if (outSampleRate == 0)
            outSampleRate = 44100;

        const uint32_t numBlocks = (dataSize + 15) / 16;
        outPcm.clear();
        outPcm.reserve(numBlocks * 28);

        const uint8_t *block = data + 48;
        const uint32_t availableBlocks = (sizeBytes - 48u) / 16u;
        const uint32_t blocksToDecode = std::min(numBlocks, availableBlocks);
        if (blocksToDecode == 0u)
            return true;

        ps2_spu_adpcm::DecoderState state{};
        return ps2_spu_adpcm::decodeBlocks(block,
                                           blocksToDecode * 16u,
                                           state,
                                           outPcm);
    }
}

bool ps2_spu_adpcm::decodeBlocks(const uint8_t *data,
                                 uint32_t sizeBytes,
                                 DecoderState &state,
                                 std::vector<int16_t> &outPcm)
{
    if (!data || sizeBytes == 0u || (sizeBytes % 16u) != 0u)
        return false;

    static constexpr int32_t kPredictors[5][2] = {
        {0, 0},
        {60, 0},
        {115, -52},
        {98, -55},
        {122, -60},
    };

    const uint32_t blockCount = sizeBytes / 16u;
    outPcm.reserve(outPcm.size() + static_cast<size_t>(blockCount) * 28u);
    for (uint32_t blockIndex = 0u; blockIndex < blockCount; ++blockIndex)
    {
        const uint8_t *const block = data + blockIndex * 16u;
        const uint32_t shift = block[0] & 0x0Fu;
        const uint32_t encodedFilter = (block[0] >> 4u) & 0x0Fu;
        const uint32_t filter = encodedFilter < 5u ? encodedFilter : 0u;

        for (uint32_t sampleIndex = 0u; sampleIndex < 28u; ++sampleIndex)
        {
            const uint8_t packed = block[2u + sampleIndex / 2u];
            const uint8_t nibble = (sampleIndex & 1u) != 0u
                                       ? packed >> 4u
                                       : packed & 0x0Fu;
            const int32_t rawSample = signExtend4(nibble);
            const int32_t shiftedSample = (rawSample * 4096) >> shift;
            const int32_t predicted =
                (kPredictors[filter][0] * state.previous1 +
                 kPredictors[filter][1] * state.previous2 + 32) >>
                6;
            const int16_t sample = clamp16(shiftedSample + predicted);

            state.previous2 = state.previous1;
            state.previous1 = sample;
            outPcm.push_back(sample);
        }
    }
    return true;
}
